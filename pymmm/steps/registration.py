"""Registrator – drift correction (stackreg or OpenCV), per FOV and frame."""

from __future__ import annotations

import itertools
import json
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import xarray as xr
from pystackreg import StackReg
from skimage.transform import warp
from tqdm.auto import tqdm

from pymmm._utils import normalize_channel_arg, normalize_fov_arg
from pymmm.parallel import Cancelled, Runner, dask_client, shared
from pymmm.source.nd2 import ND2Experiment
from pymmm.store import lineage
from pymmm.store.companion import CompanionStore

TRANSFORM_CONVENTION = "raw_translation_then_rotation_v1"


# ======================================================================
# Pure-function helpers (module-level for pickling across spawn workers)
# ======================================================================


def _register_stackreg(ref: np.ndarray, mov: np.ndarray) -> np.ndarray:
    """Register two 2-D images by translation; the 3×3 transformation matrix.

    A fresh ``StackReg`` instance is created per call for thread safety.
    """
    sr = StackReg(StackReg.TRANSLATION)
    sr.register(np.squeeze(ref), np.squeeze(mov))
    return sr.get_matrix()


def _warp_frame(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Warp a single 2-D image using a 3×3 matrix."""
    return warp(image, matrix, preserve_range=True, order=1)


def _registered_mean(
    nd2_path: str,
    fov_idx: int,
    time_indices: list[int],
    channel_idx: int | None,
    y_roi: tuple | None,
    x_roi: tuple | None,
    matrices: np.ndarray,
) -> np.ndarray:
    """Mean of registered frames for one FOV, read with a single ND2 handle.

    Frames are cropped to the experiment ROI and warped exactly as
    :meth:`Registrator.get_stabilized_data` does (cast back to the source
    dtype), then averaged in float64.
    """
    import nd2

    total = None
    with nd2.ND2File(nd2_path) as f:
        coord_axes = [k for k in f.sizes if k not in {"X", "Y", "C", "S"}]
        coord_shape = tuple(f.sizes[k] for k in coord_axes)
        for time_idx, matrix in zip(time_indices, matrices):
            coord_map = {"T": time_idx, "P": fov_idx, "Z": 0}
            coord_tuple = tuple(coord_map.get(ax, 0) for ax in coord_axes)
            frame = f.read_frame(int(np.ravel_multi_index(coord_tuple, coord_shape)))
            if channel_idx is not None and frame.ndim == 3:
                frame = frame[channel_idx]
            if y_roi is not None:
                frame = frame[slice(*y_roi), :]
            if x_roi is not None:
                frame = frame[:, slice(*x_roi)]
            warped = _warp_frame(frame, matrix).astype(frame.dtype)
            total = warped.astype(np.float64) if total is None else total + warped
    return total / len(time_indices)


def _json_value(value: Any) -> Any:
    """``value`` as it reads back from the store's JSON attributes (tuples as lists)."""
    return json.loads(json.dumps(value, default=str))


def _identity_tmats(data: xr.DataArray) -> xr.DataArray:
    """No-drift transforms for every FOV and frame of ``data``, labelled as
    :attr:`Registrator.tmats`.
    """
    dims = [d for d in ("P", "T") if d in data.dims]
    shape = tuple(data.sizes[d] for d in dims)
    return xr.DataArray(
        np.broadcast_to(np.eye(3), (*shape, 3, 3)),
        dims=[*dims, "row", "col"],
        coords={
            **{d: data.coords[d].values for d in dims},
            "row": [0, 1, 2],
            "col": [0, 1, 2],
        },
    )


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
    rotation = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
    from_origin = np.array(
        [[1.0, 0.0, center_x], [0.0, 1.0, center_y], [0.0, 0.0, 1.0]]
    )
    return from_origin @ rotation @ to_origin


def _intersect_roi(
    size: int,
    experiment_roi: tuple | None,
    registration_roi: tuple | None,
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
    return (
        img[:height, :width]
        .reshape(height // factor, factor, width // factor, factor)
        .mean(axis=(1, 3))
    )


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
    channel_idx: int | None,
    y_roi: tuple | None,
    x_roi: tuple | None,
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
    channel_idx: int | None,
    y_roi: tuple | None,
    x_roi: tuple | None,
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
    channel_idx: int | None,
    y_roi: tuple | None,
    x_roi: tuple | None,
    roi: dict,
    downsample: int,
    reference: Any,
    backend: str = "stackreg",
    threads: int = 1,
) -> list[np.ndarray]:
    """Load a batch of frames and register each to one fixed reference (a
    :meth:`~pymmm.parallel.Runner.share` handle). With the OpenCV backend, ``threads``
    frames are registered at once.
    """
    ref = shared(reference)
    frames = _load_registration_frames(
        nd2_path,
        fov_idx,
        time_indices,
        channel_idx,
        y_roi,
        x_roi,
        roi,
        downsample,
        backend,
    )
    if backend == "opencv":
        if threads > 1:
            with ThreadPoolExecutor(max_workers=threads) as pool:
                matrices = list(
                    pool.map(lambda image: _register_opencv(ref, image), frames)
                )
        else:
            matrices = [_register_opencv(ref, image) for image in frames]
    else:
        matrices = [_register_stackreg(ref, image) for image in frames]
    return [_scale_translation(matrix, downsample) for matrix in matrices]


def _register_previous_batch(
    nd2_path: str,
    fov_idx: int,
    time_indices: list[int],
    channel_idx: int | None,
    y_roi: tuple | None,
    x_roi: tuple | None,
    roi: dict,
    downsample: int,
) -> list[np.ndarray]:
    """Load a consecutive frame batch and register adjacent pairs."""
    frames = _load_registration_frames(
        nd2_path,
        fov_idx,
        time_indices,
        channel_idx,
        y_roi,
        x_roi,
        roi,
        downsample,
    )
    return [
        _scale_translation(_register_stackreg(ref, mov), downsample)
        for ref, mov in itertools.pairwise(frames)
    ]


def _cumulative(pair_matrices: list[np.ndarray]) -> np.ndarray:
    """Each frame's transform to the first, from the transforms between neighbours."""
    cumulative = [np.eye(3)]
    for matrix in pair_matrices:
        cumulative.append(cumulative[-1] @ matrix)
    return np.stack(cumulative)


@dataclass
class _FovRegistration:
    """One FOV's registration work: its batches (``start`` into the FOV's results,
    and the task's arguments), how many results they give, and how those become the
    FOV's ``(T, 3, 3)`` transforms.
    """

    task: Callable[..., list]
    batches: list[tuple]
    n_results: int
    finish: Callable[[list], np.ndarray]
    shared: list[Any] = field(default_factory=list)


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
        registration_channel: str | int = 0,
        mode: str | int = "mean",
        rotation: float = 0.0,
        roi: dict[str, tuple] | None = None,
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
        self._mean_images: xr.DataArray | None = None  # (P, Y, X)
        self._tmats: xr.DataArray | None = None  # (T, P, 3, 3)
        self._registered_mean_cache: dict[
            tuple, np.ndarray
        ] = {}  # (fov, channel) → 2D array

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
        valid for the full image).
        """
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

            diag_dir = get_diagnostics_dir(self.experiment.path, "mean_images")
            fov = self.experiment.fov_names[0]
            if "P" in self._mean_images.dims:
                img = self._mean_images.sel(P=fov).values
            else:
                img = self._mean_images.values
            plot_mean_image(
                img,
                title=f"Mean image – {fov}",
                save_path=str(diag_dir / f"mean_image_{fov}.png"),
            )

    # ------------------------------------------------------------------
    # Transformation matrices
    # ------------------------------------------------------------------

    def compute_tmats(
        self,
        plot: bool = False,
        n_jobs: int = -1,
        progress: Callable[[int, int], None] | None = None,
        cancel: Any = None,
        save_diagnostics: bool = False,
    ) -> None:
        """Compute transformation matrices for all FOVs.

        Parameters
        ----------
        plot : bool
            Save each FOV's drift figure as it completes, and show it.
        save_diagnostics : bool
            Save the drift figures without showing them (from a thread other than
            the GUI's, as the napari dock registers).
        n_jobs : int
            ``1`` registers in this process with one ND2 reader (the OpenCV backend
            still registers ``opencv_threads`` frames at once). Otherwise batches of
            frames run on the active Dask cluster if there is one, else in a process
            pool of ``n_jobs`` workers (``-1``: one per core), with ``fov_window``
            FOVs in flight.
        progress : callable | None
            Called as ``progress(fovs_done, n_fovs)`` as each FOV finishes, for a
            front end without tqdm (the napari dock).
        cancel : threading.Event | None
            Checked after each batch: once set, the queued batches are dropped and
            :class:`~pymmm.parallel.Cancelled` is raised, leaving no transforms. A
            notebook stops a run with the Interrupt button instead.

        The ``mode`` decides what each frame is registered to (by translation):

        - ``"mean"``: the mean of ``mean_n_frames`` frames from ``mean_from``
        - ``"first"`` / ``"last"`` / an ``int``: that frame
        - ``"previous"``: the frame before it; the transforms are then chained
        """
        previous = self.mode == "previous"
        print(
            "Computing relative transformation matrices..."
            if previous
            else "Computing transformation matrices..."
        )
        exp = self.experiment
        reg_data = self._get_registration_data()
        fov_names = (
            list(reg_data.coords["P"].values) if "P" in reg_data.dims else [None]
        )
        serial = n_jobs == 1
        runner = Runner(
            workers=1 if serial else (None if n_jobs == -1 else n_jobs), dask=not serial
        )
        window = 1 if serial else self.fov_window
        plan = self._previous_plan if previous else self._fixed_plan

        parts = []
        waiting = list(fov_names)
        in_flight: list[tuple] = []

        def launch() -> None:
            if waiting:
                fov = waiting.pop(0)
                job = plan(
                    runner,
                    exp.raw_fov_index(str(fov)) if fov is not None else 0,
                    threads=self.opencv_threads if serial else 1,
                )
                futures = {
                    runner.submit(job.task, *args): start for start, args in job.batches
                }
                in_flight.append((fov, job, futures))

        with runner:
            for _ in range(window):
                launch()
            bar = tqdm(total=len(fov_names), desc="Registering FOVs")
            while in_flight:
                fov, job, futures = in_flight.pop(0)
                results: list[Any] = [None] * job.n_results
                for future in tqdm(
                    runner.as_completed(futures),
                    total=len(futures),
                    desc=f"  batches ({fov})",
                    leave=False,
                ):
                    if cancel is not None and cancel.is_set():
                        raise Cancelled("registration cancelled")
                    start = futures[future]
                    batch = future.result()
                    results[start : start + len(batch)] = batch
                for handle in job.shared:
                    runner.release(handle)
                fov_da = self._fov_tmats(reg_data, fov, job.finish(results))
                parts.append(fov_da)
                self._plot_fov_drift(
                    fov, fov_da.squeeze(drop=True), plot, save_diagnostics
                )
                bar.update()
                if progress is not None:
                    progress(len(parts), len(fov_names))
                launch()
            bar.close()

        self._tmats = xr.concat(parts, dim="P") if "P" in reg_data.dims else parts[0]

    def _frames(self) -> dict:
        """Where :func:`_load_registration_frames` reads this registration's frames."""
        exp = self.experiment
        y_roi, x_roi = exp.source_roi
        return {
            "nd2_path": str(exp.path),
            "channel_idx": exp.channel_names.index(self.channel)
            if exp.has_channels
            else None,
            "y_roi": y_roi,
            "x_roi": x_roi,
            "roi": _resolve_roi(self.roi),
            "downsample": self.downsample,
        }

    def _reference(self, fov_idx: int) -> np.ndarray:
        """The fixed reference image of a FOV, prepared as its frames are."""
        f = self._frames()
        times = self.experiment.source_axes["original_t_index"].tolist()
        n_times = len(times)
        if self.mode == "mean":
            n_ref = min(self.mean_n_frames, n_times)
            ref_times = (
                times[n_times - n_ref :] if self.mean_from == "end" else times[:n_ref]
            )
        elif self.mode == "first":
            ref_times = [times[0]]
        elif self.mode == "last":
            ref_times = [times[-1]]
        elif isinstance(self.mode, int):
            ref_times = [times[self.mode]]
        else:
            raise ValueError(f"Unknown mode: {self.mode!r}")
        frames = _load_registration_frames(
            f["nd2_path"],
            fov_idx,
            ref_times,
            f["channel_idx"],
            f["y_roi"],
            f["x_roi"],
            f["roi"],
            f["downsample"],
            self.backend,
        )
        return np.mean(frames, axis=0) if self.mode == "mean" else frames[0]

    def _fixed_plan(
        self, runner: Runner, fov_idx: int, threads: int
    ) -> _FovRegistration:
        """Every frame of a FOV registered to its reference, in batches."""
        f = self._frames()
        times = self.experiment.source_axes["original_t_index"].tolist()
        reference = runner.share(self._reference(fov_idx))
        batches = [
            (
                start,
                (
                    f["nd2_path"],
                    fov_idx,
                    times[start : start + self.batch_size],
                    f["channel_idx"],
                    f["y_roi"],
                    f["x_roi"],
                    f["roi"],
                    f["downsample"],
                    reference,
                    self.backend,
                    threads,
                ),
            )
            for start in range(0, len(times), self.batch_size)
        ]
        return _FovRegistration(
            _register_fixed_batch, batches, len(times), np.stack, [reference]
        )

    def _previous_plan(
        self, runner: Runner, fov_idx: int, threads: int
    ) -> _FovRegistration:
        """Each frame of a FOV registered to the frame before it, in overlapping
        batches (each shares its first frame with the batch before).
        """
        f = self._frames()
        times = self.experiment.source_axes["original_t_index"].tolist()
        n_times = len(times)
        batches = [
            (
                start,
                (
                    f["nd2_path"],
                    fov_idx,
                    times[start : min(start + self.batch_size, n_times - 1) + 1],
                    f["channel_idx"],
                    f["y_roi"],
                    f["x_roi"],
                    f["roi"],
                    f["downsample"],
                ),
            )
            for start in range(0, n_times - 1, self.batch_size)
        ]
        return _FovRegistration(
            _register_previous_batch, batches, n_times - 1, _cumulative
        )

    @staticmethod
    def _fov_tmats(
        reg_data: xr.DataArray, fov: Any, matrices: np.ndarray
    ) -> xr.DataArray:
        """A FOV's ``(T, 3, 3)`` transforms, labelled as :attr:`tmats` (with ``P``)."""
        t_coords = (
            (reg_data.sel(P=fov) if fov is not None else reg_data).coords["T"].values
        )
        fov_da = xr.DataArray(
            matrices,
            dims=["T", "row", "col"],
            coords={"T": t_coords, "row": [0, 1, 2], "col": [0, 1, 2]},
        )
        return fov_da.expand_dims(P=[str(fov)]) if fov is not None else fov_da

    def _plot_fov_drift(
        self, fov: Any, fov_da: xr.DataArray, plot: bool, save: bool = False
    ) -> None:
        """Save one completed FOV's drift figure at once, and show it if ``plot``."""
        if not (plot or save):
            return

        from pymmm._utils import get_diagnostics_dir
        from pymmm.plotting import plot_drift_diagnostics

        fov_label = str(fov) if fov is not None else "single"
        diag_dir = get_diagnostics_dir(self.experiment.path, "drift")
        plot_drift_diagnostics(
            fov_da,
            fov_label=fov_label,
            save_path=str(diag_dir / f"drift_{fov_label}.png"),
            show=plot,
        )

    # ------------------------------------------------------------------
    # Lazy stabilised data access
    # ------------------------------------------------------------------

    def get_stabilized_data(
        self,
        channel: str | int | None = None,
        fov: str | int | None = None,
        time_slice: slice | None = None,
        drift: bool = True,
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
        drift : bool
            ``False``: the rotation alone, no drift correction, so the frames can be
            levelled before registering (the transforms are not needed).
        """
        data = self.experiment.data
        if drift:
            if self._tmats is None:
                raise RuntimeError("Tmats not computed yet.")
            tmats = self._tmats
        else:
            tmats = _identity_tmats(data)

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
        fov: int | str = 0,
        time: int = 0,
        channel: int | str | None = None,
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
        return warp(img, tmat @ rotation_matrix, preserve_range=True, order=1)

    def get_registered_mean_of_timestack(
        self,
        fov: int | str = 0,
        channel: int | str | None = None,
    ) -> np.ndarray:
        """Return the mean of registered frames for one FOV + channel.

        Results are cached per (fov, channel) so repeated calls (e.g. from
        LaneDetector and TrenchDetector) return instantly.  Only
        ``mean_n_frames`` frames are averaged (taken from the ``mean_from``
        end of the timeseries), matching the reference used for registration.
        """
        fov_name = normalize_fov_arg(fov, self.experiment.fov_names)
        return self.compute_registered_means([fov_name], channel)[fov_name]

    def compute_registered_means(
        self,
        fovs: list[int | str] | None = None,
        channel: int | str | None = None,
    ) -> dict[str, np.ndarray]:
        """Registered mean-of-timestack for several FOVs, computed together.

        FOVs not yet cached are computed in parallel, one task per FOV that
        reads its frames with a single ND2 handle: on the active distributed
        cluster if there is one, otherwise in a thread pool. ``fovs=None``
        means every FOV. Returns ``{fov_name: image}``.
        """
        ch = channel if channel is not None else self.channel
        if isinstance(ch, int):
            ch = normalize_channel_arg(ch, self.experiment.channel_names)
        fov_names = [
            normalize_fov_arg(fov, self.experiment.fov_names)
            for fov in (fovs if fovs is not None else self.experiment.fov_names)
        ]
        missing = [f for f in fov_names if (f, ch) not in self._registered_mean_cache]
        if missing:
            exp = self.experiment
            n = min(self.mean_n_frames, exp.n_timepoints)
            time_slice = slice(-n, None) if self.mean_from == "end" else slice(0, n)
            time_indices = exp.source_axes["original_t_index"][time_slice].tolist()
            channel_idx = exp.channel_names.index(ch) if exp.has_channels else None
            y_roi, x_roi = exp.source_roi
            rotation_matrix = _rotation_matrix(
                self.rotation, (exp.data.sizes["Y"], exp.data.sizes["X"])
            )
            tasks = []
            for fov_name in missing:
                tmats = (
                    self._tmats.sel(P=fov_name)
                    if "P" in self._tmats.dims
                    else self._tmats
                )
                matrices = np.matmul(tmats.isel(T=time_slice).values, rotation_matrix)
                tasks.append(
                    (
                        str(exp.path),
                        exp.raw_fov_index(fov_name),
                        time_indices,
                        channel_idx,
                        y_roi,
                        x_roi,
                        matrices,
                    )
                )
            client = dask_client()
            if client is not None:
                futures = [
                    client.submit(_registered_mean, *task, pure=False) for task in tasks
                ]
                results = client.gather(futures)
            else:
                with ThreadPoolExecutor(
                    max_workers=min(len(tasks), os.cpu_count() or 1)
                ) as pool:
                    results = list(
                        pool.map(lambda task: _registered_mean(*task), tasks)
                    )
            for fov_name, result in zip(missing, results):
                self._registered_mean_cache[(fov_name, ch)] = result
        return {f: self._registered_mean_cache[(f, ch)] for f in fov_names}

    # ------------------------------------------------------------------
    # Drift diagnostics shortcut
    # ------------------------------------------------------------------

    def plot_drift(self, fov: int | str = 0) -> None:
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

    def settings(self) -> dict[str, Any]:
        """The settings that decide the transforms, as :meth:`save` records them: a
        saved registration made with the same settings on the same part of the ND2
        can be loaded instead of computed again (:meth:`matches_saved`).
        """
        return {
            "channel": self.channel,
            "backend": self.backend,
            "mode": self.mode,
            "rotation": self.rotation,
            "roi": _json_value(self.roi),
            "mean_n_frames": self.mean_n_frames,
            "mean_from": self.mean_from,
            "downsample": self.downsample,
            "transform_convention": TRANSFORM_CONVENTION,
            "fov_names": self.experiment.fov_names,
            "source_subset_metadata": self.experiment.source_subset_metadata,
        }

    def matches_saved(self) -> bool:
        """Whether the store's registration was made with :meth:`settings`. Checkpoints
        from before ``backend`` and ``transform_convention`` were recorded count as
        stackreg and the current convention (the latter only without rotation).
        """
        if not self.store.has_registration():
            return False
        saved = self.store.load_registration_params()
        if saved.get("backend") is None:
            saved["backend"] = "stackreg"
        if saved.get("transform_convention") is None and saved.get("rotation", 0) == 0:
            saved["transform_convention"] = TRANSFORM_CONVENTION
        return all(saved.get(key) == value for key, value in self.settings().items())

    def fingerprint(self) -> str:
        """The fingerprint of the transforms and rotation (:mod:`pymmm.store.lineage`)."""
        return lineage.registration_fingerprint(self.tmats, self.rotation)

    def save(self) -> None:
        """Write tmats, mean_images, and params to the companion zarr store, with the
        source it was computed from.
        """
        if self._tmats is None:
            raise RuntimeError("Nothing to save — run compute_tmats() first.")

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
        }
        self.store.save_registration(
            self._tmats,
            self._mean_images,
            params,
            source_subset_metadata=self.experiment.source_subset_metadata,
            made_from={"source": lineage.source_fingerprint(self.experiment)},
        )
        print(f"Registration saved to {self.store.path}")

    @classmethod
    def load(
        cls,
        experiment: ND2Experiment,
        store: CompanionStore,
        load_mean_images: bool = False,
    ) -> Registrator:
        """Reconstruct a Registrator from a checkpoint."""
        if not store.has_registration():
            raise FileNotFoundError("No registration checkpoint found in store.")

        data = store.load_registration(load_mean_images=load_mean_images)
        params = data["params"]
        if params.get("transform_convention") != TRANSFORM_CONVENTION:
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

        def _str_fovs(array: xr.DataArray) -> xr.DataArray:
            if "P" in array.coords:
                array = array.assign_coords(
                    P=[str(p) for p in array.coords["P"].values]
                )
            return array

        reg._tmats = _str_fovs(data["tmats"])
        if data["mean_images"] is not None:
            reg._mean_images = _str_fovs(data["mean_images"])
        return reg

    def __repr__(self) -> str:
        status = "computed" if self.is_computed else "not computed"
        return (
            f"Registrator(channel={self.channel!r}, backend={self.backend!r}, "
            f"mode={self.mode!r}, "
            f"rotation={self.rotation}, downsample={self.downsample}, "
            f"status={status})"
        )
