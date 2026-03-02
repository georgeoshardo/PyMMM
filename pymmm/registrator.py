"""Registrator – drift-correction via pystackreg, parallelised with dask."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import xarray as xr
from pystackreg import StackReg
from skimage.transform import rotate, warp
from tqdm.auto import tqdm

from pymmm._utils import normalize_channel_arg, normalize_fov_arg
from pymmm.checkpoint import CompanionStore
from pymmm.experiment import ND2Experiment


# ======================================================================
# Pure-function helpers (module-level for dask serialisation)
# ======================================================================


def _register_and_get_matrix(ref: np.ndarray, mov: np.ndarray) -> np.ndarray:
    """Register two 2-D images and return the 3×3 transformation matrix.

    A fresh ``StackReg`` instance is created per call for thread safety
    across dask workers.
    """
    ref_sq = np.squeeze(ref)
    mov_sq = np.squeeze(mov)
    sr = StackReg(StackReg.RIGID_BODY)
    sr.register(ref_sq, mov_sq)
    return sr.get_matrix()


def _register_translation(ref: np.ndarray, mov: np.ndarray) -> np.ndarray:
    """Register with TRANSLATION mode (for ``previous`` method)."""
    ref_sq = np.squeeze(ref)
    mov_sq = np.squeeze(mov)
    sr = StackReg(StackReg.TRANSLATION)
    sr.register(ref_sq, mov_sq)
    return sr.get_matrix()


def _warp_frame(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Warp a single 2-D image using a 3×3 matrix."""
    return warp(image, matrix, preserve_range=True, order=1)


def _add_identity_frame(
    matrices: xr.DataArray, original_ref: xr.DataArray
) -> xr.DataArray:
    """Prepend identity matrices along T (handles P dimension if present)."""
    non_matrix_dims = [d for d in matrices.dims if d not in {"T", "row", "col"}]
    eye_data = np.eye(3)[None, :, :]

    if non_matrix_dims:
        extra_shape = tuple(matrices.sizes[d] for d in non_matrix_dims)
        expand_shape = (1,) + extra_shape + (3, 3)
        eye_data = np.broadcast_to(eye_data, expand_shape)

    first_t_coord = original_ref.coords["T"].isel(T=[0])
    coords: Dict[str, Any] = {d: matrices.coords[d] for d in non_matrix_dims}
    coords["T"] = first_t_coord

    identity_da = xr.DataArray(
        eye_data,
        dims=["T"] + non_matrix_dims + ["row", "col"],
        coords=coords,
    )
    full_stack = xr.concat([identity_da, matrices], dim="T")
    full_stack["T"] = original_ref.coords["T"]
    return full_stack


def _cumulative_matrices(matrices_da: xr.DataArray) -> xr.DataArray:
    """Compute cumulative matrix product along T, preserving P dimension."""
    mats_np = matrices_da.values  # already computed
    cumulative = np.zeros_like(mats_np)
    cumulative[0] = mats_np[0]
    for t in range(1, len(mats_np)):
        cumulative[t] = cumulative[t - 1] @ mats_np[t]
    return xr.DataArray(
        cumulative, dims=matrices_da.dims, coords=matrices_da.coords
    )


# ======================================================================
# Registrator class
# ======================================================================


class Registrator:
    """Compute drift-correction transformation matrices using pystackreg.

    Parameters
    ----------
    experiment : ND2Experiment
        Source experiment.
    store : CompanionStore
        Companion zarr store for checkpointing.
    registration_channel : str | int
        Channel to use for registration (e.g. ``"PC"``).
    mode : str
        Registration mode: ``"mean"``, ``"previous"``, ``"first"``,
        ``"last"``, or an ``int`` frame index.
    rotation : float
        Rotation angle in degrees applied before registration.
    roi : dict | None
        Region of interest, e.g. ``{"y": (300, 900), "x": (0, -1)}``.
    mean_n_frames : int
        Number of frames to average for the reference image.
    mean_from : str
        ``"end"`` or ``"start"`` — which end of the timeseries to average.
    """

    def __init__(
        self,
        experiment: ND2Experiment,
        store: CompanionStore,
        registration_channel: Union[str, int] = 0,
        mode: Union[str, int] = "mean",
        rotation: float = 0.0,
        roi: Optional[Dict[str, tuple]] = None,
        mean_n_frames: int = 10,
        mean_from: Literal["end", "start"] = "end",
    ) -> None:
        self.experiment = experiment
        self.store = store
        self.channel = normalize_channel_arg(
            registration_channel, experiment.channel_names
        )
        self.mode = mode
        self.rotation = rotation
        self.roi = roi or {}
        self.mean_n_frames = mean_n_frames
        self.mean_from = mean_from

        # Computed state
        self._mean_images: Optional[xr.DataArray] = None  # (P, Y, X)
        self._tmats: Optional[xr.DataArray] = None  # (T, P, 3, 3)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_computed(self) -> bool:
        return self._tmats is not None

    @property
    def tmats(self) -> xr.DataArray:
        if self._tmats is None:
            raise RuntimeError("Tmats not computed yet. Call compute_tmats() first.")
        return self._tmats

    @property
    def mean_images(self) -> xr.DataArray:
        if self._mean_images is None:
            raise RuntimeError(
                "Mean images not computed yet. Call compute_mean_images() first."
            )
        return self._mean_images

    # ------------------------------------------------------------------
    # Data selection helpers
    # ------------------------------------------------------------------

    def _get_registration_data(self) -> xr.DataArray:
        """Get the channel data for registration, with rotation applied."""
        d = self.experiment.data
        if self.experiment.has_channels:
            d = d.sel(C=self.channel)
        # Drop Z if present — take first slice
        if "Z" in d.dims:
            d = d.isel(Z=0)
        return d

    def _apply_roi(self, data: xr.DataArray) -> xr.DataArray:
        """Slice data to the ROI (for registration only; tmat is still
        valid for the full image)."""
        if not self.roi:
            return data
        slices = {}
        for dim_key, (start, end) in self.roi.items():
            dim_name = dim_key.upper()
            if dim_name in data.dims:
                if end == -1:
                    end = None
                slices[dim_name] = slice(start, end)
        if slices:
            data = data.isel(**slices)
        return data

    def _apply_rotation_np(self, img: np.ndarray) -> np.ndarray:
        """Apply rotation to a numpy image if rotation is set."""
        if self.rotation != 0.0:
            return rotate(img, self.rotation, preserve_range=True)
        return img

    # ------------------------------------------------------------------
    # Mean images
    # ------------------------------------------------------------------

    def compute_mean_images(self, plot: bool = False) -> None:
        """Compute temporal mean images for each FOV.

        Uses the last (or first) ``mean_n_frames`` frames.
        """
        data = self._get_registration_data()

        n = min(self.mean_n_frames, self.experiment.n_timepoints)
        if self.mean_from == "end":
            data_subset = data.isel(T=slice(-n, None))
        else:
            data_subset = data.isel(T=slice(0, n))

        # Compute mean along T → (P, Y, X) or (Y, X)
        mean_imgs = data_subset.mean(dim="T")

        # Apply rotation if needed
        if self.rotation != 0.0:
            mean_imgs = xr.apply_ufunc(
                self._apply_rotation_np,
                mean_imgs,
                input_core_dims=[["Y", "X"]],
                output_core_dims=[["Y", "X"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[np.float64],
            )

        # Compute eagerly — mean images are small
        self._mean_images = mean_imgs.compute()

        if plot:
            from pymmm.plotting import plot_mean_image

            fov = self.experiment.fov_names[0]
            if "P" in self._mean_images.dims:
                img = self._mean_images.sel(P=fov).values
            else:
                img = self._mean_images.values
            plot_mean_image(img, title=f"Mean image – {fov}")

    # ------------------------------------------------------------------
    # Transformation matrices
    # ------------------------------------------------------------------

    def compute_tmats(self, plot: bool = False) -> None:
        """Compute transformation matrices for all FOVs.

        The chosen ``mode`` determines the registration strategy:
        - ``"mean"`` — register each frame to temporal mean (RIGID_BODY)
        - ``"previous"`` — register to predecessor, cumulative (TRANSLATION)
        - ``"first"`` / ``"last"`` / ``int`` — register to a fixed frame (RIGID_BODY)
        """
        data = self._get_registration_data()

        # Apply rotation lazily if needed
        if self.rotation != 0.0:
            data = xr.apply_ufunc(
                self._apply_rotation_np,
                data,
                input_core_dims=[["Y", "X"]],
                output_core_dims=[["Y", "X"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[data.dtype],
            )

        # Apply ROI for registration (tmat valid for full image)
        data_roi = self._apply_roi(data)

        if self.mode == "previous":
            self._compute_tmats_previous(data_roi, data)
        else:
            self._compute_tmats_fixed_ref(data_roi, data)

        if plot:
            from pymmm.plotting import plot_drift_diagnostics

            fov = self.experiment.fov_names[0]
            if "P" in self._tmats.dims:
                mats = self._tmats.sel(P=fov)
            else:
                mats = self._tmats
            plot_drift_diagnostics(mats, fov_label=fov)

    def _compute_tmats_fixed_ref(
        self, data_roi: xr.DataArray, data_full: xr.DataArray
    ) -> None:
        """Register every frame to a fixed reference image."""
        if self.mode == "mean":
            if self._mean_images is None:
                raise RuntimeError(
                    "Call compute_mean_images() before compute_tmats(mode='mean')."
                )
            ref = self._apply_roi(self._mean_images)
        elif self.mode == "first":
            ref = self._apply_roi(data_full.isel(T=0)).compute()
        elif self.mode == "last":
            ref = self._apply_roi(data_full.isel(T=-1)).compute()
        elif isinstance(self.mode, int):
            ref = self._apply_roi(data_full.isel(T=self.mode)).compute()
        else:
            raise ValueError(f"Unknown mode: {self.mode!r}")

        # Broadcast ref to have a T dimension matching data
        ref_broadcast = ref.broadcast_like(data_roi)

        tmats = xr.apply_ufunc(
            _register_and_get_matrix,
            ref_broadcast,
            data_roi,
            input_core_dims=[["Y", "X"], ["Y", "X"]],
            output_core_dims=[["row", "col"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.float64],
            dask_gufunc_kwargs={"output_sizes": {"row": 3, "col": 3}},
            keep_attrs=False,
        )

        print("Computing transformation matrices...")
        self._tmats = tmats.compute()

    def _compute_tmats_previous(
        self, data_roi: xr.DataArray, data_full: xr.DataArray
    ) -> None:
        """Register each frame to its predecessor (cumulative product)."""
        import dask.array as da

        ref_stack = data_roi.isel(T=slice(0, -1))
        mov_stack = data_roi.isel(T=slice(1, None))
        mov_stack["T"] = ref_stack["T"]

        # Ensure dask backing
        ref_stack = ref_stack.copy(data=da.asarray(ref_stack.data))
        mov_stack = mov_stack.copy(data=da.asarray(mov_stack.data))

        relative = xr.apply_ufunc(
            _register_translation,
            ref_stack,
            mov_stack,
            input_core_dims=[["Y", "X"], ["Y", "X"]],
            output_core_dims=[["row", "col"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.float64],
            dask_gufunc_kwargs={"output_sizes": {"row": 3, "col": 3}},
            keep_attrs=False,
        )

        # Prepend identity frame
        relative_full = _add_identity_frame(relative, data_full)

        print("Computing relative transformation matrices...")
        relative_computed = relative_full.compute()

        # Cumulative product per FOV
        if "P" in relative_computed.dims:
            parts = []
            for fov in relative_computed.coords["P"].values:
                part = _cumulative_matrices(relative_computed.sel(P=fov))
                parts.append(part)
            self._tmats = xr.concat(parts, dim="P")
        else:
            self._tmats = _cumulative_matrices(relative_computed)

    # ------------------------------------------------------------------
    # Lazy stabilised data access
    # ------------------------------------------------------------------

    def get_stabilized_data(
        self, channel: Optional[Union[str, int]] = None
    ) -> xr.DataArray:
        """Return lazy dask-backed DataArray of all warped frames.

        Parameters
        ----------
        channel : str | int | None
            Channel to stabilise. ``None`` uses the registration channel.
        """
        if self._tmats is None:
            raise RuntimeError("Tmats not computed yet.")

        data = self.experiment.data
        if channel is not None:
            ch = normalize_channel_arg(channel, self.experiment.channel_names)
            if self.experiment.has_channels:
                data = data.sel(C=ch)
        elif self.experiment.has_channels:
            data = data.sel(C=self.channel)

        # Drop Z if present
        if "Z" in data.dims:
            data = data.isel(Z=0)

        # Apply rotation lazily if needed
        if self.rotation != 0.0:
            data = xr.apply_ufunc(
                self._apply_rotation_np,
                data,
                input_core_dims=[["Y", "X"]],
                output_core_dims=[["Y", "X"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[data.dtype],
            )

        stabilized = xr.apply_ufunc(
            _warp_frame,
            data,
            self._tmats,
            input_core_dims=[["Y", "X"], ["row", "col"]],
            output_core_dims=[["Y", "X"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[data.dtype],
            dask_gufunc_kwargs={
                "output_sizes": {
                    "Y": data.sizes["Y"],
                    "X": data.sizes["X"],
                }
            },
            keep_attrs=True,
        )
        return stabilized

    def warp_frame(
        self,
        fov: Union[int, str] = 0,
        time: int = 0,
        channel: Optional[Union[int, str]] = None,
    ) -> np.ndarray:
        """Load a single frame from the ND2, rotate, apply tmat, return numpy."""
        if self._tmats is None:
            raise RuntimeError("Tmats not computed yet.")

        ch = channel if channel is not None else self.channel
        frame = self.experiment.get_frame(fov=fov, time=time, channel=ch)
        img = frame.values

        if self.rotation != 0.0:
            img = rotate(img, self.rotation, preserve_range=True)

        fov_name = normalize_fov_arg(fov, self.experiment.fov_names)
        if "P" in self._tmats.dims:
            tmat = self._tmats.sel(P=fov_name).isel(T=time).values
        else:
            tmat = self._tmats.isel(T=time).values

        return warp(img, tmat, preserve_range=True, order=1)

    def get_registered_mean_of_timestack(
        self,
        fov: Union[int, str] = 0,
        channel: Optional[Union[int, str]] = None,
    ) -> np.ndarray:
        """Return the mean of registered frames for one FOV + channel."""
        stabilized = self.get_stabilized_data(channel=channel)
        fov_name = normalize_fov_arg(fov, self.experiment.fov_names)
        if "P" in stabilized.dims:
            stack = stabilized.sel(P=fov_name)
        else:
            stack = stabilized
        return stack.mean(dim="T").compute().values

    # ------------------------------------------------------------------
    # Drift diagnostics shortcut
    # ------------------------------------------------------------------

    def plot_drift(self, fov: Union[int, str] = 0) -> None:
        """Plot drift diagnostics for a single FOV."""
        from pymmm.plotting import plot_drift_diagnostics

        fov_name = normalize_fov_arg(fov, self.experiment.fov_names)
        if "P" in self._tmats.dims:
            mats = self._tmats.sel(P=fov_name)
        else:
            mats = self._tmats
        plot_drift_diagnostics(mats, fov_label=fov_name)

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write tmats, mean_images, and params to the companion zarr store."""
        if self._tmats is None:
            raise RuntimeError("Nothing to save — run compute_tmats() first.")

        tmats_np = self._tmats.values
        mean_np = self._mean_images.values if self._mean_images is not None else np.array([])

        params = {
            "channel": self.channel,
            "mode": self.mode,
            "rotation": self.rotation,
            "roi": self.roi,
            "mean_n_frames": self.mean_n_frames,
            "mean_from": self.mean_from,
            "fov_names": self.experiment.fov_names,
            "tmats_dims": list(self._tmats.dims),
            "tmats_shape": list(self._tmats.shape),
        }

        # Store coordinate values for reconstruction
        if "P" in self._tmats.coords:
            params["tmats_P_coords"] = [str(v) for v in self._tmats.coords["P"].values]
        if "T" in self._tmats.coords:
            params["tmats_T_coords"] = [float(v) for v in self._tmats.coords["T"].values]

        self.store.save_registration(tmats_np, mean_np, params)
        print(f"Registration saved to {self.store.path}")

    @classmethod
    def load(cls, experiment: ND2Experiment, store: CompanionStore) -> "Registrator":
        """Reconstruct a Registrator from a checkpoint."""
        if not store.has_registration():
            raise FileNotFoundError("No registration checkpoint found in store.")

        data = store.load_registration()
        params = data["params"]

        reg = cls(
            experiment=experiment,
            store=store,
            registration_channel=params["channel"],
            mode=params["mode"],
            rotation=params.get("rotation", 0.0),
            roi=params.get("roi"),
            mean_n_frames=params.get("mean_n_frames", 10),
            mean_from=params.get("mean_from", "end"),
        )

        # Reconstruct tmats as xr.DataArray
        tmats_np = data["tmats"]
        dims = params.get("tmats_dims", ["T", "P", "row", "col"])
        coords: Dict[str, Any] = {"row": [0, 1, 2], "col": [0, 1, 2]}
        if "tmats_P_coords" in params:
            coords["P"] = params["tmats_P_coords"]
        if "tmats_T_coords" in params:
            coords["T"] = params["tmats_T_coords"]

        reg._tmats = xr.DataArray(tmats_np, dims=dims, coords=coords)

        # Reconstruct mean images if present
        mean_np = data["mean_images"]
        if mean_np.size > 0:
            mean_dims = ["P", "Y", "X"] if mean_np.ndim == 3 else ["Y", "X"]
            mean_coords: Dict[str, Any] = {}
            if "P" in mean_dims and "tmats_P_coords" in params:
                mean_coords["P"] = params["tmats_P_coords"]
            reg._mean_images = xr.DataArray(mean_np, dims=mean_dims, coords=mean_coords)

        return reg

    def __repr__(self) -> str:
        status = "computed" if self.is_computed else "not computed"
        return (
            f"Registrator(channel={self.channel!r}, mode={self.mode!r}, "
            f"rotation={self.rotation}, status={status})"
        )
