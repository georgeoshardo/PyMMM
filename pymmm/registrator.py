"""Registrator – drift-correction via pystackreg, parallelised with ProcessPoolExecutor."""

from __future__ import annotations

import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import xarray as xr
from pystackreg import StackReg
from skimage.transform import warp
from tqdm.auto import tqdm

from pymmm._utils import normalize_channel_arg, normalize_fov_arg
from pymmm.checkpoint import CompanionStore
from pymmm.experiment import ND2Experiment

# Spawn context — fresh Python interpreters that know nothing about
# the parent's dask Client, event loops, or thread state.
_mp_ctx = multiprocessing.get_context("spawn")
TRANSFORM_CONVENTION = "raw_translation_then_rotation_v1"


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


def _rotation_matrix(angle: float, image_shape: tuple[int, int]) -> np.ndarray:
    """Return the centre-aware inverse map used by ``skimage.rotate``."""
    height, width = image_shape
    center_x = width / 2.0 - 0.5
    center_y = height / 2.0 - 0.5
    theta = np.deg2rad(angle)
    cosine = np.cos(theta)
    sine = np.sin(theta)
    to_origin = np.array(
        [[1.0, 0.0, -center_x], [0.0, 1.0, -center_y], [0.0, 0.0, 1.0]]
    )
    rotation = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    from_origin = np.array(
        [[1.0, 0.0, center_x], [0.0, 1.0, center_y], [0.0, 0.0, 1.0]]
    )
    return from_origin @ rotation @ to_origin


def _intersect_roi(
    size: int,
    experiment_roi: Optional[tuple],
    registration_roi: Optional[tuple],
) -> slice:
    """Intersect two slices expressed in raw-image coordinates."""
    experiment_slice = slice(*(experiment_roi or (None, None)))
    registration_slice = slice(*(registration_roi or (None, None)))
    exp_start, exp_stop, _ = experiment_slice.indices(size)
    reg_start, reg_stop, _ = registration_slice.indices(size)
    return slice(max(exp_start, reg_start), min(exp_stop, reg_stop))


def _resolve_roi(roi: dict) -> dict:
    """Normalize ROI dict: convert ``-1`` stops to ``None``. Returns new dict."""
    if not roi:
        return {}
    out = {}
    for k, (start, end) in roi.items():
        out[k] = (start, None if end == -1 else end)
    return out


def _downsample_mean(img: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a 2-D image by block averaging."""
    if factor == 1:
        return img
    height = (img.shape[0] // factor) * factor
    width = (img.shape[1] // factor) * factor
    return img[:height, :width].reshape(
        height // factor, factor, width // factor, factor
    ).mean(axis=(1, 3))


def _register_opencv(ref: np.ndarray, mov: np.ndarray) -> np.ndarray:
    """Estimate translation with OpenCV's iterative phase correlation."""
    import cv2

    shift_x, shift_y = cv2.phaseCorrelateIterative(ref, mov)
    matrix = np.eye(3)
    matrix[0, 2] = shift_x
    matrix[1, 2] = shift_y
    return matrix


def _prepare_registration_frame(
    frame: np.ndarray,
    channel_idx: Optional[int],
    y_roi: Optional[tuple],
    x_roi: Optional[tuple],
    roi: dict,
    downsample: int,
    backend: str = "stackreg",
) -> np.ndarray:
    """Select one channel/ROI and prepare a frame for registration."""
    if channel_idx is not None and frame.ndim == 3:
        frame = frame[channel_idx]
    y_slice = _intersect_roi(frame.shape[-2], y_roi, roi.get("y"))
    x_slice = _intersect_roi(frame.shape[-1], x_roi, roi.get("x"))
    frame = frame[y_slice, x_slice]
    if backend == "opencv":
        import cv2

        if downsample > 1:
            height = (frame.shape[0] // downsample) * downsample
            width = (frame.shape[1] // downsample) * downsample
            frame = cv2.resize(
                frame[:height, :width],
                (width // downsample, height // downsample),
                interpolation=cv2.INTER_AREA,
            )
        return np.asarray(frame, dtype=np.float32)
    img = np.asarray(frame, dtype=np.float64)
    return _downsample_mean(img, downsample)


def _load_registration_frames(
    nd2_path: str,
    fov_idx: int,
    time_indices: list[int],
    channel_idx: Optional[int],
    y_roi: Optional[tuple],
    x_roi: Optional[tuple],
    roi: dict,
    downsample: int,
    backend: str = "stackreg",
) -> list[np.ndarray]:
    """Read a contiguous frame batch using one ND2 file handle."""
    import nd2

    frames = []
    with nd2.ND2File(nd2_path) as f:
        coord_axes = [k for k in f.sizes if k not in {"X", "Y", "C", "S"}]
        coord_shape = tuple(f.sizes[k] for k in coord_axes)
        for time_idx in time_indices:
            coord_map = {"T": time_idx, "P": fov_idx, "Z": 0}
            coord_tuple = tuple(coord_map.get(ax, 0) for ax in coord_axes)
            frame_idx = int(np.ravel_multi_index(coord_tuple, coord_shape))
            frame = f.read_frame(frame_idx)
            frames.append(
                _prepare_registration_frame(
                    frame, channel_idx, y_roi, x_roi, roi, downsample, backend
                ).copy()
            )
    return frames


def _scale_translation(matrix: np.ndarray, downsample: int) -> np.ndarray:
    """Convert a downsampled-image translation to full-resolution pixels."""
    if downsample == 1:
        return matrix
    matrix = matrix.copy()
    matrix[0, 2] *= downsample
    matrix[1, 2] *= downsample
    return matrix


def _register_fixed_batch(
    nd2_path: str,
    fov_idx: int,
    time_indices: list[int],
    channel_idx: Optional[int],
    y_roi: Optional[tuple],
    x_roi: Optional[tuple],
    transformation: int,
    roi: dict,
    downsample: int,
    ref_np: np.ndarray,
    backend: str = "stackreg",
) -> list[np.ndarray]:
    """Load and register a batch of frames against one fixed reference."""
    frames = _load_registration_frames(
        nd2_path, fov_idx, time_indices, channel_idx,
        y_roi, x_roi, roi, downsample, backend,
    )
    return [
        _scale_translation(
            _register_opencv(ref_np, img)
            if backend == "opencv"
            else _register_and_get_matrix(
                ref_np, img, transformation=transformation
            ),
            downsample,
        )
        for img in frames
    ]


_worker_ref_cache: dict[str, np.ndarray] = {}


def _register_fixed_batch_fileref(
    nd2_path: str,
    fov_idx: int,
    time_indices: list[int],
    channel_idx: Optional[int],
    y_roi: Optional[tuple],
    x_roi: Optional[tuple],
    transformation: int,
    roi: dict,
    downsample: int,
    ref_path: str,
    backend: str = "stackreg",
) -> list[np.ndarray]:
    """Register a frame batch using a per-worker cached file reference.

    Like ``_register_fixed_batch`` but receives a path to a ``.npy`` file
    instead of the full reference array.  A per-worker cache ensures the
    file is read at most once per FOV (cache is evicted when the path
    changes).  With ``spawn`` context each worker is a separate process,
    so the module-level cache has no thread-safety concerns.
    """
    if ref_path not in _worker_ref_cache:
        _worker_ref_cache.clear()
        _worker_ref_cache[ref_path] = np.load(ref_path)
    ref_np = _worker_ref_cache[ref_path]
    return _register_fixed_batch(
        nd2_path, fov_idx, time_indices, channel_idx,
        y_roi, x_roi, transformation, roi, downsample, ref_np, backend,
    )


def _register_previous_batch(
    nd2_path: str,
    fov_idx: int,
    time_indices: list[int],
    channel_idx: Optional[int],
    y_roi: Optional[tuple],
    x_roi: Optional[tuple],
    roi: dict,
    downsample: int,
) -> list[np.ndarray]:
    """Load a consecutive frame batch and register adjacent pairs."""
    frames = _load_registration_frames(
        nd2_path, fov_idx, time_indices, channel_idx,
        y_roi, x_roi, roi, downsample,
    )
    return [
        _scale_translation(_register_translation(ref, mov), downsample)
        for ref, mov in zip(frames[:-1], frames[1:])
    ]


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
        Fixed output rotation applied together with drift correction.
    roi : dict | None
        Registration region in unrotated raw-image coordinates. It is
        intersected with any experiment ROI before registration.
    mean_n_frames : int
        Number of frames to average for the reference image.
    mean_from : str
        ``"end"`` or ``"start"`` — which end of the timeseries to average.
    downsample : int
        Block-averaging factor used only while estimating translations.
    batch_size : int
        Number of consecutive frames read per worker task.
    fov_window : int
        Number of FOVs kept in flight on the Dask multi-worker path.
    backend : {"stackreg", "opencv"}
        Translation estimator. ``"opencv"`` uses iterative phase correlation
        and is available for fixed-reference modes only.
    opencv_threads : int
        Concurrent OpenCV calls used when ``backend="opencv"`` and
        ``compute_tmats(n_jobs=1)``. Limited by ``batch_size``.
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
        downsample: int = 1,
        batch_size: int = 8,
        fov_window: int = 1,
        backend: Literal["stackreg", "opencv"] = "stackreg",
        opencv_threads: int = 8,
    ) -> None:
        if int(downsample) != downsample or downsample < 1:
            raise ValueError("downsample must be a positive integer")
        if int(batch_size) != batch_size or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if int(fov_window) != fov_window or fov_window < 1:
            raise ValueError("fov_window must be a positive integer")
        if int(opencv_threads) != opencv_threads or opencv_threads < 1:
            raise ValueError("opencv_threads must be a positive integer")
        if backend not in {"stackreg", "opencv"}:
            raise ValueError("backend must be 'stackreg' or 'opencv'")
        if backend == "opencv" and mode == "previous":
            raise ValueError(
                "backend='opencv' is not supported for mode='previous' because "
                "small pairwise errors accumulate; use backend='stackreg'."
            )
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
        self.downsample = int(downsample)
        self.batch_size = int(batch_size)
        self.fov_window = int(fov_window)
        self.backend = backend
        self.opencv_threads = int(opencv_threads)

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
            ``-1`` = all cores (default). ``1`` keeps one ND2 reader in the
            parent process; the OpenCV backend can still run frame
            registrations concurrently using ``opencv_threads``. FOVs are
            always processed sequentially.

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
        y_roi = (
            (exp._y_slice.start, exp._y_slice.stop, exp._y_slice.step)
            if exp._y_slice is not None else None
        )
        x_roi = (
            (exp._x_slice.start, exp._x_slice.stop, exp._x_slice.step)
            if exp._x_slice is not None else None
        )
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
        time_batches = [
            original_t_indices[start:start + self.batch_size]
            for start in range(0, n_times, self.batch_size)
        ]

        max_workers = None if n_jobs == -1 else n_jobs

        def _compute_ref(fov_idx):
            if self.mode == "mean":
                n_ref = min(self.mean_n_frames, n_times)
                if self.mean_from == "end":
                    ref_t_indices = [original_t_indices[i] for i in range(n_times - n_ref, n_times)]
                else:
                    ref_t_indices = [original_t_indices[i] for i in range(n_ref)]
                ref_frames = _load_registration_frames(
                    nd2_path, fov_idx, ref_t_indices, channel_idx,
                    y_roi, x_roi, roi, self.downsample, self.backend,
                )
                return np.mean(ref_frames, axis=0)
            elif self.mode == "first":
                ref_time = original_t_indices[0]
            elif self.mode == "last":
                ref_time = original_t_indices[-1]
            elif isinstance(self.mode, int):
                ref_time = original_t_indices[self.mode]
            else:
                raise ValueError(f"Unknown mode: {self.mode!r}")
            return _load_registration_frames(
                nd2_path, fov_idx, [ref_time], channel_idx,
                y_roi, x_roi, roi, self.downsample, self.backend,
            )[0]

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
            # One ND2 reader; OpenCV calls can share frames across threads.
            thread_pool = (
                ThreadPoolExecutor(max_workers=self.opencv_threads)
                if self.backend == "opencv" and self.opencv_threads > 1
                else None
            )
            try:
                for fov in tqdm(fov_names, desc="Registering FOVs"):
                    fov_str = str(fov) if fov is not None else None
                    fov_idx = raw_fov_names.index(fov_str) if fov_str is not None else 0
                    ref_np = _compute_ref(fov_idx)
                    tmats_list = []
                    for time_batch in time_batches:
                        if thread_pool is None:
                            tmats_list.extend(
                                _register_fixed_batch(
                                    nd2_path, fov_idx, time_batch, channel_idx,
                                    y_roi, x_roi, transformation, roi,
                                    self.downsample, ref_np, self.backend,
                                )
                            )
                            continue
                        frames = _load_registration_frames(
                            nd2_path, fov_idx, time_batch, channel_idx,
                            y_roi, x_roi, roi, self.downsample, self.backend,
                        )
                        matrices = thread_pool.map(
                            lambda image: _register_opencv(ref_np, image), frames
                        )
                        tmats_list.extend(
                            _scale_translation(matrix, self.downsample)
                            for matrix in matrices
                        )
                    del ref_np
                    fov_da = _wrap_da(fov, tmats_list)
                    parts.append(fov_da)
                    self._plot_fov_drift(fov, fov_da.squeeze(drop=True), plot)
            finally:
                if thread_pool is not None:
                    thread_pool.shutdown()
        elif (client := _try_get_dask_client()) is not None:
            # Dask path — reuse the user's existing distributed cluster
            from dask.distributed import as_completed as dask_as_completed

            def _submit_fov(fov):
                fov_str = str(fov) if fov is not None else None
                fov_idx = raw_fov_names.index(fov_str) if fov_str is not None else 0
                ref_np = _compute_ref(fov_idx)
                ref_future = client.scatter(ref_np, broadcast=True, hash=False)
                del ref_np

                future_to_start = {
                    client.submit(
                        _register_fixed_batch, nd2_path, fov_idx,
                        time_batch, channel_idx, y_roi, x_roi,
                        transformation, roi, self.downsample,
                        ref_future, self.backend,
                    ): start
                    for start, time_batch in zip(
                        range(0, n_times, self.batch_size), time_batches
                    )
                }
                return fov, ref_future, future_to_start

            fov_iter = iter(fov_names)
            in_flight = []
            for _ in range(min(self.fov_window, len(fov_names))):
                in_flight.append(_submit_fov(next(fov_iter)))

            fov_progress = tqdm(total=len(fov_names), desc="Registering FOVs")
            while in_flight:
                fov, ref_future, future_to_start = in_flight.pop(0)
                results = [None] * n_times
                for fut in tqdm(
                    dask_as_completed(future_to_start), total=len(time_batches),
                    desc=f"  batches ({fov})", leave=False,
                ):
                    start = future_to_start[fut]
                    batch_result = fut.result()
                    results[start:start + len(batch_result)] = batch_result

                del ref_future
                fov_da = _wrap_da(fov, results)
                parts.append(fov_da)
                self._plot_fov_drift(fov, fov_da.squeeze(drop=True), plot)
                fov_progress.update()

                try:
                    next_fov = next(fov_iter)
                except StopIteration:
                    continue
                in_flight.append(_submit_fov(next_fov))
            fov_progress.close()
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

                    future_to_start = {
                        pool.submit(
                            _register_fixed_batch_fileref, nd2_path, fov_idx,
                            time_batch, channel_idx, y_roi, x_roi,
                            transformation, roi, self.downsample,
                            ref_path, self.backend,
                        ): start
                        for start, time_batch in zip(
                            range(0, n_times, self.batch_size), time_batches
                        )
                    }

                    results = [None] * n_times
                    for fut in tqdm(
                        as_completed(future_to_start), total=len(time_batches),
                        desc=f"  batches ({fov})", leave=False,
                    ):
                        start = future_to_start[fut]
                        batch_result = fut.result()
                        results[start:start + len(batch_result)] = batch_result

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
        y_roi = (
            (exp._y_slice.start, exp._y_slice.stop, exp._y_slice.step)
            if exp._y_slice is not None else None
        )
        x_roi = (
            (exp._x_slice.start, exp._x_slice.stop, exp._x_slice.step)
            if exp._x_slice is not None else None
        )

        raw_fov_names = [str(v) for v in exp._raw_data.coords["P"].values] if "P" in exp._raw_data.dims else []
        channel_idx = exp.channel_names.index(self.channel) if has_channels else None

        if exp._time_slice is not None:
            n_raw_T = exp._raw_data.sizes["T"]
            original_t_indices = list(range(*exp._time_slice.indices(n_raw_T)))
        else:
            original_t_indices = list(range(exp._raw_data.sizes["T"]))
        n_times = len(original_t_indices)
        pair_batches = [
            (
                start,
                original_t_indices[
                    start:min(start + self.batch_size, n_times - 1) + 1
                ],
            )
            for start in range(0, n_times - 1, self.batch_size)
        ]

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
                pair_mats = []
                for _, time_batch in pair_batches:
                    pair_mats.extend(
                        _register_previous_batch(
                            nd2_path, fov_idx, time_batch, channel_idx,
                            y_roi, x_roi, roi, self.downsample,
                        )
                    )
                fov_da = _collect_pairs(fov, fov_idx, pair_mats)
                parts.append(fov_da)
                self._plot_fov_drift(fov, fov_da.squeeze(drop=True), plot)
        elif (client := _try_get_dask_client()) is not None:
            # Dask path — no scatter needed, workers load their own frames
            from dask.distributed import as_completed as dask_as_completed

            for fov in tqdm(fov_names, desc="Registering FOVs"):
                fov_str = str(fov) if fov is not None else None
                fov_idx = raw_fov_names.index(fov_str) if fov_str is not None else 0

                future_to_start = {
                    client.submit(
                        _register_previous_batch, nd2_path, fov_idx,
                        time_batch, channel_idx, y_roi, x_roi,
                        roi, self.downsample,
                    ): start
                    for start, time_batch in pair_batches
                }

                results = [None] * (n_times - 1)
                for fut in tqdm(
                    dask_as_completed(future_to_start), total=len(pair_batches),
                    desc=f"  batches ({fov})", leave=False,
                ):
                    start = future_to_start[fut]
                    batch_result = fut.result()
                    results[start:start + len(batch_result)] = batch_result

                fov_da = _collect_pairs(fov, fov_idx, results)
                parts.append(fov_da)
                self._plot_fov_drift(fov, fov_da.squeeze(drop=True), plot)
        else:
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=_mp_ctx) as pool:
                for fov in tqdm(fov_names, desc="Registering FOVs"):
                    fov_str = str(fov) if fov is not None else None
                    fov_idx = raw_fov_names.index(fov_str) if fov_str is not None else 0

                    future_to_start = {
                        pool.submit(
                            _register_previous_batch, nd2_path, fov_idx,
                            time_batch, channel_idx, y_roi, x_roi,
                            roi, self.downsample,
                        ): start
                        for start, time_batch in pair_batches
                    }

                    results = [None] * (n_times - 1)
                    for fut in tqdm(
                        as_completed(future_to_start), total=len(pair_batches),
                        desc=f"  batches ({fov})", leave=False,
                    ):
                        start = future_to_start[fut]
                        batch_result = fut.result()
                        results[start:start + len(batch_result)] = batch_result

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
        time_slice: Optional[slice] = None,
    ) -> xr.DataArray:
        """Return lazy dask-backed DataArray of all warped frames.

        Parameters
        ----------
        channel : str | int | None
            Channel to stabilise. ``None`` uses the registration channel.
        fov : str | int | None
            Restrict to a single FOV. Keeps the dask graph small.
        time_slice : slice | None
            Restrict timepoints before constructing the warp graph.
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

        if time_slice is not None and "T" in data.dims:
            data = data.isel(T=time_slice)
            tmats = tmats.isel(T=time_slice)

        if channel is not None:
            ch = normalize_channel_arg(channel, self.experiment.channel_names)
            if self.experiment.has_channels:
                data = data.sel(C=ch)
        elif self.experiment.has_channels:
            data = data.sel(C=self.channel)

        # Drop Z if present
        if "Z" in data.dims:
            data = data.isel(Z=0)

        rotation_matrix = _rotation_matrix(
            self.rotation, (data.sizes["Y"], data.sizes["X"])
        )
        tmats = tmats.copy(data=np.matmul(tmats.values, rotation_matrix))

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
        """Load one raw frame and apply translation plus rotation once."""
        if self._tmats is None:
            raise RuntimeError("Tmats not computed yet.")

        ch = channel if channel is not None else self.channel
        frame = self.experiment.get_frame(fov=fov, time=time, channel=ch)
        if "Z" in frame.dims:
            frame = frame.isel(Z=0)
        img = frame.values

        fov_name = normalize_fov_arg(fov, self.experiment.fov_names)
        if "P" in self._tmats.dims:
            tmat = self._tmats.sel(P=fov_name).isel(T=time).values
        else:
            tmat = self._tmats.isel(T=time).values

        rotation_matrix = _rotation_matrix(self.rotation, img.shape)
        return warp(
            img, tmat @ rotation_matrix, preserve_range=True, order=1
        )

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

        n = min(self.mean_n_frames, self.experiment.n_timepoints)
        if self.mean_from == "end":
            time_slice = slice(-n, None)
        else:
            time_slice = slice(0, n)
        stack = self.get_stabilized_data(
            channel=ch, fov=fov_name, time_slice=time_slice,
        )

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
        mean_np = self._mean_images.values if self._mean_images is not None else None

        params = {
            "channel": self.channel,
            "backend": self.backend,
            "mode": self.mode,
            "rotation": self.rotation,
            "roi": self.roi,
            "mean_n_frames": self.mean_n_frames,
            "mean_from": self.mean_from,
            "downsample": self.downsample,
            "batch_size": self.batch_size,
            "fov_window": self.fov_window,
            "opencv_threads": self.opencv_threads,
            "transform_convention": TRANSFORM_CONVENTION,
            "fov_names": self.experiment.fov_names,
            "source_subset_metadata": self.experiment.source_subset_metadata,
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
    def load(
        cls,
        experiment: ND2Experiment,
        store: CompanionStore,
        load_mean_images: bool = False,
    ) -> "Registrator":
        """Reconstruct a Registrator from a checkpoint."""
        if not store.has_registration():
            raise FileNotFoundError("No registration checkpoint found in store.")

        data = store.load_registration(load_mean_images=load_mean_images)
        params = data["params"]
        convention = params.get("transform_convention")
        legacy_unrotated = convention is None and params.get("rotation", 0.0) == 0.0
        if convention != TRANSFORM_CONVENTION and not legacy_unrotated:
            raise ValueError(
                "Registration checkpoint uses an incompatible transform convention. "
                "Recompute registration and downstream detection checkpoints."
            )

        reg = cls(
            experiment=experiment,
            store=store,
            registration_channel=params["channel"],
            mode=params["mode"],
            rotation=params.get("rotation", 0.0),
            roi=params.get("roi"),
            mean_n_frames=params.get("mean_n_frames", 10),
            mean_from=params.get("mean_from", "end"),
            downsample=params.get("downsample", 1),
            batch_size=params.get("batch_size", 8),
            fov_window=params.get("fov_window", 1),
            backend=params.get("backend", "stackreg"),
            opencv_threads=params.get("opencv_threads", 8),
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
        if mean_np is not None and mean_np.size > 0:
            mean_dims = ["P", "Y", "X"] if mean_np.ndim == 3 else ["Y", "X"]
            mean_coords: Dict[str, Any] = {}
            if "P" in mean_dims and "tmats_P_coords" in params:
                mean_coords["P"] = params["tmats_P_coords"]
            reg._mean_images = xr.DataArray(mean_np, dims=mean_dims, coords=mean_coords)

        return reg

    def __repr__(self) -> str:
        status = "computed" if self.is_computed else "not computed"
        return (
            f"Registrator(channel={self.channel!r}, backend={self.backend!r}, "
            f"mode={self.mode!r}, "
            f"rotation={self.rotation}, downsample={self.downsample}, "
            f"status={status})"
        )
