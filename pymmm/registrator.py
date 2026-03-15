"""Registrator – drift-correction via pystackreg, parallelised with ProcessPoolExecutor."""

from __future__ import annotations

import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import xarray as xr
from pystackreg import StackReg
from skimage.transform import rotate, warp
from tqdm.auto import tqdm

from pymmm._utils import normalize_channel_arg, normalize_fov_arg
from pymmm.checkpoint import CompanionStore
from pymmm.experiment import ND2Experiment

# Spawn context — fresh Python interpreters that know nothing about
# the parent's dask Client, event loops, or thread state.
_mp_ctx = multiprocessing.get_context("spawn")


# ======================================================================
# Pure-function helpers (module-level for pickling across spawn workers)
# ======================================================================


def _try_get_dask_client():
    """Return the active dask distributed Client, or None."""
    try:
        from dask.distributed import get_client
        return get_client()
    except (ImportError, ValueError):
        return None


def _register_and_get_matrix(
    ref: np.ndarray,
    mov: np.ndarray,
    transformation: int = StackReg.TRANSLATION,
) -> np.ndarray:
    """Register two 2-D images and return the 3×3 transformation matrix.

    A fresh ``StackReg`` instance is created per call for thread safety.
    """
    ref_sq = np.squeeze(ref)
    mov_sq = np.squeeze(mov)
    sr = StackReg(transformation)
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


def _slice_roi(img: np.ndarray, roi: dict) -> np.ndarray:
    """Apply ROI dict to a 2D numpy array. Returns a view (no copy).

    roi keys: ``"y"`` and/or ``"x"``, values: ``(start, end)`` where
    ``end=None`` means to-the-end.
    """
    if not roi:
        return img
    y_slice = slice(*(roi.get("y", (None, None))))
    x_slice = slice(*(roi.get("x", (None, None))))
    return img[y_slice, x_slice]


def _resolve_roi(roi: dict) -> dict:
    """Normalize ROI dict: convert ``-1`` stops to ``None``. Returns new dict."""
    if not roi:
        return {}
    out = {}
    for k, (start, end) in roi.items():
        out[k] = (start, None if end == -1 else end)
    return out


def _load_single_frame(
    nd2_path: str,
    fov_idx: int,
    time_idx: int,
    channel_idx: Optional[int],
    rotation: float,
) -> np.ndarray:
    """Open ND2, read one (Y,X) frame via random access, close file.

    Uses ``nd2.ND2File.read_frame()`` for true random access — reads ONLY
    the requested frame from disk, never loads the full dataset. Each call
    opens and closes the file; the I/O overhead (~5-50ms) is negligible
    relative to the pystackreg CPU cost (~200ms per registration).

    Parameters
    ----------
    nd2_path : str
        Path to the ND2 file.
    fov_idx : int
        Integer position index into the P dimension (0-based).
        Ignored if file has no P dimension.
    time_idx : int
        Integer index into the T dimension.
    channel_idx : int | None
        Integer index into the C dimension. None for single-channel files.
    rotation : float
        Rotation angle in degrees (0.0 = no rotation).

    Returns
    -------
        np.ndarray
        2D float64 array (Y, X).
    """
    import nd2

    with nd2.ND2File(nd2_path) as f:
        # Determine coordinate axes: everything except {X, Y, C, S}
        coord_axes = [k for k in f.sizes if k not in {"X", "Y", "C", "S"}]
        coord_shape = tuple(f.sizes[k] for k in coord_axes)

        # Build coordinate tuple in the file's axis order
        coord_map = {"T": time_idx, "P": fov_idx, "Z": 0}
        coord_tuple = tuple(coord_map.get(ax, 0) for ax in coord_axes)

        # Convert to flat frame index and read
        frame_idx = int(np.ravel_multi_index(coord_tuple, coord_shape))
        frame = f.read_frame(frame_idx).copy()  # copy: read_frame returns a view into file buffer

    # Channel selection (read_frame returns all channels)
    if channel_idx is not None and frame.ndim == 3:
        frame = frame[channel_idx]

    img = frame.astype(np.float64)

    if rotation != 0.0:
        img = rotate(img, rotation, preserve_range=True)
    return img


def _register_one_frame(
    nd2_path: str,
    fov_idx: int,
    time_idx: int,
    channel_idx: Optional[int],
    rotation: float,
    transformation: int,
    roi: dict,
    ref_np: np.ndarray,
) -> np.ndarray:
    """Worker for fixed-ref mode: load one frame, register vs ref.

    Opens the ND2 file, reads the single frame it needs via random access,
    registers against ref_np, returns 3x3 matrix.
    """
    img = _load_single_frame(nd2_path, fov_idx, time_idx, channel_idx, rotation)
    img_roi = _slice_roi(img, roi)
    return _register_and_get_matrix(ref_np, img_roi, transformation=transformation)


_worker_ref_cache: dict[str, np.ndarray] = {}


def _register_one_frame_fileref(
    nd2_path: str,
    fov_idx: int,
    time_idx: int,
    channel_idx: Optional[int],
    rotation: float,
    transformation: int,
    roi: dict,
    ref_path: str,
) -> np.ndarray:
    """Worker for fixed-ref mode: load ref from file, load frame, register.

    Like ``_register_one_frame`` but receives a path to a ``.npy`` file
    instead of the full reference array.  A per-worker cache ensures the
    file is read at most once per FOV (cache is evicted when the path
    changes).  With ``spawn`` context each worker is a separate process,
    so the module-level cache has no thread-safety concerns.
    """
    if ref_path not in _worker_ref_cache:
        _worker_ref_cache.clear()
        _worker_ref_cache[ref_path] = np.load(ref_path)
    ref_np = _worker_ref_cache[ref_path]
    img = _load_single_frame(nd2_path, fov_idx, time_idx, channel_idx, rotation)
    img_roi = _slice_roi(img, roi)
    return _register_and_get_matrix(ref_np, img_roi, transformation=transformation)


def _register_pair(
    nd2_path: str,
    fov_idx: int,
    ref_time: int,
    mov_time: int,
    channel_idx: Optional[int],
    rotation: float,
    roi: dict,
) -> np.ndarray:
    """Worker for previous mode: load two frames, register.

    Opens the ND2 file twice (once per frame), reads each via random access.
    Registers mov against ref using TRANSLATION mode. Returns 3x3 matrix.
    """
    ref = _load_single_frame(nd2_path, fov_idx, ref_time, channel_idx, rotation)
    mov = _load_single_frame(nd2_path, fov_idx, mov_time, channel_idx, rotation)
    return _register_translation(_slice_roi(ref, roi), _slice_roi(mov, roi))


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
        self._registered_mean_cache: Dict[tuple, np.ndarray] = {}  # (fov, channel) → 2D array

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
        """Get the channel data for registration (channel + Z selection only)."""
        d = self.experiment.data
        if self.experiment.has_channels:
            d = d.sel(C=self.channel)
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
            from pymmm._utils import get_diagnostics_dir
            from pymmm.plotting import plot_mean_image

            diag_dir = get_diagnostics_dir(self.experiment.path)
            fov = self.experiment.fov_names[0]
            if "P" in self._mean_images.dims:
                img = self._mean_images.sel(P=fov).values
            else:
                img = self._mean_images.values
            plot_mean_image(
                img, title=f"Mean image – {fov}",
                save_path=str(diag_dir / f"mean_image_{fov}.png"),
            )

    # ------------------------------------------------------------------
    # Transformation matrices
    # ------------------------------------------------------------------

    def compute_tmats(self, plot: bool = False, n_jobs: int = -1) -> None:
        """Compute transformation matrices for all FOVs.

        Parameters
        ----------
        plot : bool
            Show drift diagnostics for the first FOV.
        n_jobs : int
            Number of worker processes for within-FOV parallelism.
            ``-1`` = all cores (default), ``1`` = sequential (no subprocess
            overhead, useful for debugging).  FOVs are always processed
            sequentially; the parallelism axis is across timepoints
            *within* each FOV.

        The chosen ``mode`` determines the registration strategy:
        - ``"mean"`` — register each frame to temporal mean (TRANSLATION)
        - ``"previous"`` — register to predecessor, cumulative (TRANSLATION)
        - ``"first"`` / ``"last"`` / ``int`` — register to a fixed frame (TRANSLATION)
        """
        if self.mode == "previous":
            self._compute_tmats_previous(n_jobs=n_jobs, plot=plot)
        else:
            self._compute_tmats_fixed_ref(n_jobs=n_jobs, plot=plot)

    def _plot_fov_drift(self, fov: Any, fov_da: xr.DataArray, plot: bool) -> None:
        """Persist drift diagnostics for one completed FOV immediately."""
        if not plot:
            return

        from pymmm._utils import get_diagnostics_dir
        from pymmm.plotting import plot_drift_diagnostics

        fov_label = str(fov) if fov is not None else "single"
        diag_dir = get_diagnostics_dir(self.experiment.path)
        plot_drift_diagnostics(
            fov_da,
            fov_label=fov_label,
            save_path=str(diag_dir / f"drift_{fov_label}.png"),
        )

    def _compute_tmats_fixed_ref(self, n_jobs: int = -1, plot: bool = False) -> None:
        """Register every frame to a fixed reference image.

        Sequential across FOVs, parallel across timepoints within each FOV
        via ``ProcessPoolExecutor`` with spawn context.  Completely
        independent of dask — workers are fresh Python interpreters that
        know nothing about the parent's dask Client or event loop.
        """
        print("Computing transformation matrices...")
        exp = self.experiment
        nd2_path = str(exp.path)
        roi = _resolve_roi(self.roi)

        # Registration data (for T coords and FOV names only)
        reg_data = self._get_registration_data()
        fov_names = reg_data.coords["P"].values if "P" in reg_data.dims else [None]
        has_channels = exp.has_channels

        # Map FOV names → integer P indices for read_frame()
        raw_fov_names = [str(v) for v in exp._raw_data.coords["P"].values] if "P" in exp._raw_data.dims else []
        channel_idx = exp.channel_names.index(self.channel) if has_channels else None

        # Map subsetted T indices → original file T indices
        if exp._time_slice is not None:
            n_raw_T = exp._raw_data.sizes["T"]
            original_t_indices = list(range(*exp._time_slice.indices(n_raw_T)))
        else:
            original_t_indices = list(range(exp._raw_data.sizes["T"]))
        n_times = len(original_t_indices)
        transformation = StackReg.TRANSLATION

        max_workers = None if n_jobs == -1 else n_jobs

        def _compute_ref(fov_idx):
            if self.mode == "mean":
                n_ref = min(self.mean_n_frames, n_times)
                if self.mean_from == "end":
                    ref_t_indices = [original_t_indices[i] for i in range(n_times - n_ref, n_times)]
                else:
                    ref_t_indices = [original_t_indices[i] for i in range(n_ref)]
                ref_frames = [
                    _load_single_frame(nd2_path, fov_idx, t, channel_idx, self.rotation)
                    for t in ref_t_indices
                ]
                return _slice_roi(np.mean(ref_frames, axis=0), roi)
            elif self.mode == "first":
                ref_frame = _load_single_frame(
                    nd2_path, fov_idx, original_t_indices[0], channel_idx, self.rotation
                )
                return _slice_roi(ref_frame, roi)
            elif self.mode == "last":
                ref_frame = _load_single_frame(
                    nd2_path, fov_idx, original_t_indices[-1], channel_idx, self.rotation
                )
                return _slice_roi(ref_frame, roi)
            elif isinstance(self.mode, int):
                ref_frame = _load_single_frame(
                    nd2_path, fov_idx, original_t_indices[self.mode], channel_idx, self.rotation
                )
                return _slice_roi(ref_frame, roi)
            else:
                raise ValueError(f"Unknown mode: {self.mode!r}")

        def _wrap_da(fov, tmats_list):
            fov_tmats_np = np.stack(tmats_list)  # (T, 3, 3)
            if fov is not None:
                t_coords = reg_data.sel(P=fov).coords["T"].values
            else:
                t_coords = reg_data.coords["T"].values
            fov_da = xr.DataArray(
                fov_tmats_np,
                dims=["T", "row", "col"],
                coords={"T": t_coords, "row": [0, 1, 2], "col": [0, 1, 2]},
            )
            if fov is not None:
                fov_da = fov_da.expand_dims(P=[str(fov)])
            return fov_da

        parts = []

        if n_jobs == 1:
            # Sequential path — no pickling, no temp files
            for fov in tqdm(fov_names, desc="Registering FOVs"):
                fov_str = str(fov) if fov is not None else None
                fov_idx = raw_fov_names.index(fov_str) if fov_str is not None else 0
                ref_np = _compute_ref(fov_idx)
                tmats_list = [
                    _register_one_frame(
                        nd2_path, fov_idx, original_t_indices[t],
                        channel_idx, self.rotation, transformation, roi, ref_np,
                    )
                    for t in range(n_times)
                ]
                del ref_np
                fov_da = _wrap_da(fov, tmats_list)
                parts.append(fov_da)
                self._plot_fov_drift(fov, fov_da.squeeze(drop=True), plot)
        elif (client := _try_get_dask_client()) is not None:
            # Dask path — reuse the user's existing distributed cluster
            from dask.distributed import as_completed as dask_as_completed

            for fov in tqdm(fov_names, desc="Registering FOVs"):
                fov_str = str(fov) if fov is not None else None
                fov_idx = raw_fov_names.index(fov_str) if fov_str is not None else 0
                ref_np = _compute_ref(fov_idx)
                ref_future = client.scatter(ref_np, broadcast=True)
                del ref_np

                future_to_t = {
                    client.submit(
                        _register_one_frame, nd2_path, fov_idx,
                        original_t_indices[t], channel_idx,
                        self.rotation, transformation, roi, ref_future,
                    ): t
                    for t in range(n_times)
                }

                results = [None] * n_times
                for fut in tqdm(
                    dask_as_completed(future_to_t), total=n_times,
                    desc=f"  frames ({fov})", leave=False,
                ):
                    results[future_to_t[fut]] = fut.result()

                del ref_future
                fov_da = _wrap_da(fov, results)
                parts.append(fov_da)
                self._plot_fov_drift(fov, fov_da.squeeze(drop=True), plot)
        else:
            # PPE fallback — pool + tmpdir created once, reused across FOVs
            with tempfile.TemporaryDirectory(prefix="pymmm_reg_") as tmpdir, \
                 ProcessPoolExecutor(max_workers=max_workers, mp_context=_mp_ctx) as pool:
                for fov in tqdm(fov_names, desc="Registering FOVs"):
                    fov_str = str(fov) if fov is not None else None
                    fov_idx = raw_fov_names.index(fov_str) if fov_str is not None else 0
                    ref_np = _compute_ref(fov_idx)

                    ref_path = os.path.join(tmpdir, f"ref_{fov_idx}.npy")
                    np.save(ref_path, ref_np)
                    del ref_np

                    future_to_t = {
                        pool.submit(
                            _register_one_frame_fileref, nd2_path, fov_idx,
                            original_t_indices[t], channel_idx,
                            self.rotation, transformation, roi, ref_path,
                        ): t
                        for t in range(n_times)
                    }

                    results = [None] * n_times
                    for fut in tqdm(
                        as_completed(future_to_t), total=n_times,
                        desc=f"  frames ({fov})", leave=False,
                    ):
                        results[future_to_t[fut]] = fut.result()

                    os.unlink(ref_path)
                    fov_da = _wrap_da(fov, results)
                    parts.append(fov_da)
                    self._plot_fov_drift(fov, fov_da.squeeze(drop=True), plot)

        if "P" in reg_data.dims:
            self._tmats = xr.concat(parts, dim="P")
        else:
            self._tmats = parts[0]

    def _compute_tmats_previous(self, n_jobs: int = -1, plot: bool = False) -> None:
        """Register each frame to its predecessor (cumulative product).

        Sequential across FOVs, parallel across consecutive frame pairs
        within each FOV via ``ProcessPoolExecutor`` with spawn context.
        The cumulative product is computed sequentially after all pairs
        finish.
        """
        print("Computing relative transformation matrices...")
        exp = self.experiment
        nd2_path = str(exp.path)
        roi = _resolve_roi(self.roi)

        reg_data = self._get_registration_data()
        fov_names = reg_data.coords["P"].values if "P" in reg_data.dims else [None]
        has_channels = exp.has_channels

        raw_fov_names = [str(v) for v in exp._raw_data.coords["P"].values] if "P" in exp._raw_data.dims else []
        channel_idx = exp.channel_names.index(self.channel) if has_channels else None

        if exp._time_slice is not None:
            n_raw_T = exp._raw_data.sizes["T"]
            original_t_indices = list(range(*exp._time_slice.indices(n_raw_T)))
        else:
            original_t_indices = list(range(exp._raw_data.sizes["T"]))
        n_times = len(original_t_indices)

        max_workers = None if n_jobs == -1 else n_jobs

        def _collect_pairs(fov, fov_idx, pair_mats):
            cumulative = [np.eye(3)]
            for mat in pair_mats:
                cumulative.append(cumulative[-1] @ mat)
            cumulative_np = np.stack(cumulative)  # (T, 3, 3)
            if fov is not None:
                t_coords = reg_data.sel(P=fov).coords["T"].values
            else:
                t_coords = reg_data.coords["T"].values
            fov_da = xr.DataArray(
                cumulative_np,
                dims=["T", "row", "col"],
                coords={"T": t_coords, "row": [0, 1, 2], "col": [0, 1, 2]},
            )
            if fov is not None:
                fov_da = fov_da.expand_dims(P=[str(fov)])
            return fov_da

        parts = []

        if n_jobs == 1:
            for fov in tqdm(fov_names, desc="Registering FOVs"):
                fov_str = str(fov) if fov is not None else None
                fov_idx = raw_fov_names.index(fov_str) if fov_str is not None else 0
                pair_mats = [
                    _register_pair(
                        nd2_path, fov_idx, original_t_indices[t],
                        original_t_indices[t + 1], channel_idx,
                        self.rotation, roi,
                    )
                    for t in range(n_times - 1)
                ]
                fov_da = _collect_pairs(fov, fov_idx, pair_mats)
                parts.append(fov_da)
                self._plot_fov_drift(fov, fov_da.squeeze(drop=True), plot)
        elif (client := _try_get_dask_client()) is not None:
            # Dask path — no scatter needed, workers load their own frames
            from dask.distributed import as_completed as dask_as_completed

            for fov in tqdm(fov_names, desc="Registering FOVs"):
                fov_str = str(fov) if fov is not None else None
                fov_idx = raw_fov_names.index(fov_str) if fov_str is not None else 0

                future_to_t = {
                    client.submit(
                        _register_pair, nd2_path, fov_idx,
                        original_t_indices[t], original_t_indices[t + 1],
                        channel_idx, self.rotation, roi,
                    ): t
                    for t in range(n_times - 1)
                }

                results = [None] * (n_times - 1)
                for fut in tqdm(
                    dask_as_completed(future_to_t), total=n_times - 1,
                    desc=f"  pairs ({fov})", leave=False,
                ):
                    results[future_to_t[fut]] = fut.result()

                fov_da = _collect_pairs(fov, fov_idx, results)
                parts.append(fov_da)
                self._plot_fov_drift(fov, fov_da.squeeze(drop=True), plot)
        else:
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=_mp_ctx) as pool:
                for fov in tqdm(fov_names, desc="Registering FOVs"):
                    fov_str = str(fov) if fov is not None else None
                    fov_idx = raw_fov_names.index(fov_str) if fov_str is not None else 0

                    future_to_t = {
                        pool.submit(
                            _register_pair, nd2_path, fov_idx,
                            original_t_indices[t], original_t_indices[t + 1],
                            channel_idx, self.rotation, roi,
                        ): t
                        for t in range(n_times - 1)
                    }

                    results = [None] * (n_times - 1)
                    for fut in tqdm(
                        as_completed(future_to_t), total=n_times - 1,
                        desc=f"  pairs ({fov})", leave=False,
                    ):
                        results[future_to_t[fut]] = fut.result()

                    fov_da = _collect_pairs(fov, fov_idx, results)
                    parts.append(fov_da)
                    self._plot_fov_drift(fov, fov_da.squeeze(drop=True), plot)

        if "P" in reg_data.dims:
            self._tmats = xr.concat(parts, dim="P")
        else:
            self._tmats = parts[0]

    # ------------------------------------------------------------------
    # Lazy stabilised data access
    # ------------------------------------------------------------------

    def get_stabilized_data(
        self,
        channel: Optional[Union[str, int]] = None,
        fov: Optional[Union[str, int]] = None,
    ) -> xr.DataArray:
        """Return lazy dask-backed DataArray of all warped frames.

        Parameters
        ----------
        channel : str | int | None
            Channel to stabilise. ``None`` uses the registration channel.
        fov : str | int | None
            Restrict to a single FOV. Keeps the dask graph small.
        """
        if self._tmats is None:
            raise RuntimeError("Tmats not computed yet.")

        data = self.experiment.data
        tmats = self._tmats

        # Select FOV early to keep the dask graph small
        if fov is not None:
            fov_name = normalize_fov_arg(fov, self.experiment.fov_names)
            if "P" in data.dims:
                data = data.sel(P=fov_name)
            if "P" in tmats.dims:
                tmats = tmats.sel(P=fov_name)

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
            tmats,
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
        """Return the mean of registered frames for one FOV + channel.

        Results are cached per (fov, channel) so repeated calls (e.g. from
        LaneDetector and TrenchDetector) return instantly.  Only
        ``mean_n_frames`` frames are averaged (taken from the ``mean_from``
        end of the timeseries), matching the reference used for registration.
        """
        ch = channel if channel is not None else self.channel
        if isinstance(ch, int):
            ch = normalize_channel_arg(ch, self.experiment.channel_names)
        fov_name = normalize_fov_arg(fov, self.experiment.fov_names)
        cache_key = (fov_name, ch)

        if cache_key in self._registered_mean_cache:
            return self._registered_mean_cache[cache_key]

        stabilized = self.get_stabilized_data(channel=ch)
        if "P" in stabilized.dims:
            stack = stabilized.sel(P=fov_name)
        else:
            stack = stabilized

        # Average only mean_n_frames frames instead of the full timeseries
        n = min(self.mean_n_frames, stack.sizes["T"])
        if self.mean_from == "end":
            stack = stack.isel(T=slice(-n, None))
        else:
            stack = stack.isel(T=slice(0, n))

        result = stack.mean(dim="T").compute().values
        self._registered_mean_cache[cache_key] = result
        return result

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
