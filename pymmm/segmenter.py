"""Segmenter – Omnipose cell labels for every trench crop in a trench store.

Each trench is scaled by its own intensity percentiles (with a store-wide floor
so trenches without cells keep their noise dark), then segmented under a few
preset Omnipose settings, the *hypotheses*. S1 is the primary segmentation; S2–S4
mostly keep pairs that have just divided as one cell, so downstream tracking can
choose between them. Labels are written to ``/segmentation/<name>`` inside the
trench store as ``labels(Hypothesis, Trench, T, Y, X)``.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import dask.array as da
import numpy as np
import xarray as xr
import zarr
from tqdm.auto import tqdm

from pymmm.provenance import SCHEMA_VERSION, provenance

_LABEL_DIMS = ("Hypothesis", "Trench", "T", "Y", "X")

_LABELS_ATTRS = {
    "long_name": (
        "cell labels; 0 is background, 1..n are ordered from the trench's closed "
        "end (Y = 0), so 1 is the cell nearest the closed end"
    ),
}

PERCENTILES = (1.0, 99.0)
"""Per-trench intensity percentiles mapped to 0 and 1 (before the gain)."""

DEFAULT_PARAMS: Dict[str, Any] = {
    "floor": None,
    "floor_sigma": 20.0,
    "gain": 1.0,
    "rescale": 1.0,
    "mask_threshold": 0.0,
    "flow_threshold": 0.4,
    "min_area": 1.75,
    "min_contrast": 5.0,
}
"""Base settings, shared by every hypothesis unless the hypothesis overrides them.

``floor``: lowest allowed upper intensity bound, in counts; ``None`` estimates it
as background + ``floor_sigma`` background-noise SDs (median over sampled
trenches). ``gain``: multiplies the trench's intensity range (above 1 lowers
contrast). ``rescale``: Omnipose ``rescale_factor``. ``min_area``: smallest mask
kept, in µm². ``min_contrast``: masks whose mean is fewer than this many
background-noise SDs above the frame's background are dropped.
"""

HYPOTHESES: Dict[str, Dict[str, Any]] = {
    "S1": {},
    "S2": {"gain": 1.5, "rescale": 0.8, "flow_threshold": 0.0},
    "S3": {"rescale": 0.8, "mask_threshold": -1.0, "flow_threshold": 0.0},
    "S4": {"gain": 2.5, "rescale": 0.8, "mask_threshold": -1.0, "flow_threshold": 0.0},
}
"""Preset overrides of the base settings. Chosen from a sweep of 360 settings on
mCherry: S1 splits dividing cells the most; S2–S4 cover the alternative
segmentations (mostly just-divided pairs kept as one cell, fuller masks in
crowded trenches) without adding masks on noise."""

# Omnipose 2.0 logs mask building through the top-level "core" and "gpu" loggers.
_OMNIPOSE_LOGGERS = ("omnipose", "core", "gpu")

# Omnipose imports the ``dbscan`` package, whose parlay scheduler threads spin
# at full CPU even when idle (about 9 cores in an idle Jupyter kernel). One
# thread costs nothing measurable in segmentation speed.
_PARLAY_THREADS = {"PARLAY_NUM_THREADS": "1"}
# What each process-pool worker gets: one thread per library, so ``workers``
# processes use ``workers`` cores.
_WORKER_THREADS = {
    var: "1" for var in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMBA_NUM_THREADS", "PARLAY_NUM_THREADS",
    )
}


def _json_native(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


@contextmanager
def _environment(values: Mapping[str, str]) -> Iterator[None]:
    """Set environment variables for the duration; restore them afterwards.

    Thread pools read their size when a library is first loaded, so spawned
    workers must inherit the limits from the parent's environment: setting them
    in a worker's initializer is too late (the worker has already imported numpy
    and torch by unpickling it).
    """
    previous = {var: os.environ.get(var) for var in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for var, value in previous.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value


@contextmanager
def _quiet_omnipose() -> Iterator[None]:
    """Hide Omnipose's per-frame INFO logging (e.g. one line per empty frame)."""
    loggers = [logging.getLogger(name) for name in _OMNIPOSE_LOGGERS]
    previous = [logger.level for logger in loggers]
    for logger in loggers:
        logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        for logger, level in zip(loggers, previous):
            logger.setLevel(level)


def background_noise(pixels: np.ndarray) -> Tuple[float, float]:
    """Background level and noise SD of an image or stack.

    The background is the median pixel and the noise is the robust SD of the
    pixels below it, so cells (bright, a minority of pixels) do not inflate
    either. Zero pixels (outside the camera image after drift correction) are
    ignored.
    """
    real = np.asarray(pixels)
    real = real[real > 0].astype(np.float64)
    if real.size == 0:
        return 0.0, 0.0
    background = float(np.median(real))
    below = real[real <= background]
    return background, float(1.4826 * np.median(background - below))


def trench_bounds(stack: np.ndarray, floor: float) -> Tuple[float, float]:
    """Intensity bounds of one trench over all its frames.

    The :data:`PERCENTILES` of the non-zero pixels; the upper bound is at least
    ``floor``, so a trench that is mostly empty does not stretch its noise.
    """
    real = np.asarray(stack)
    real = real[real > 0]
    if real.size == 0:
        return 0.0, float(floor)
    lower, upper = np.percentile(real, PERCENTILES)
    return float(lower), float(max(upper, floor, lower + 1))


def normalise(stack: np.ndarray, lower: float, upper: float, gain: float = 1.0) -> np.ndarray:
    """Map ``lower`` to 0 and ``lower + gain * (upper - lower)`` to 1.

    Zero pixels are set to ``lower`` first so the zero-filled border does not
    look like a dark structure.
    """
    stack = np.asarray(stack, dtype=np.float32)
    stack = np.where(stack > 0, stack, np.float32(lower))
    return (stack - np.float32(lower)) / np.float32(gain * (upper - lower))


def drop_faint(masks: np.ndarray, stack: np.ndarray, min_contrast: float) -> np.ndarray:
    """Remove masks fainter than ``min_contrast`` background-noise SDs.

    The background and noise of each frame come from its pixels outside every
    mask (:func:`background_noise`).
    """
    masks = np.array(masks)
    for t, (frame_masks, frame) in enumerate(zip(masks, np.asarray(stack))):
        if not frame_masks.any():
            continue
        background, noise = background_noise(frame[frame_masks == 0])
        if noise <= 0:
            continue
        flat = frame_masks.ravel()
        counts = np.bincount(flat)
        sums = np.bincount(flat, weights=frame.ravel().astype(np.float64))
        ids = np.flatnonzero(counts)
        ids = ids[ids > 0]
        contrast = (sums[ids] / counts[ids] - background) / noise
        faint = ids[contrast < min_contrast]
        if faint.size:
            frame_masks[np.isin(frame_masks, faint)] = 0
    return masks


def order_from_closed_end(masks: np.ndarray) -> np.ndarray:
    """Relabel each frame of ``masks`` 1..n by centroid row, smallest Y first.

    Trench crops are flipped so every trench opens downward; Y = 0 is the closed
    end, so label 1 is the cell nearest it.

    Parameters
    ----------
    masks : np.ndarray
        Integer labels, ``(T, Y, X)``.

    Returns
    -------
    np.ndarray
        ``uint16`` labels of the same shape.
    """
    masks = np.asarray(masks)
    out = np.zeros(masks.shape, dtype=np.uint16)
    rows = np.broadcast_to(
        np.arange(masks.shape[-2], dtype=np.float64)[:, None], masks.shape[-2:],
    ).ravel()
    for t, frame in enumerate(masks):
        flat = frame.ravel()
        counts = np.bincount(flat)
        ids = np.flatnonzero(counts)
        ids = ids[ids > 0]
        if ids.size == 0:
            continue
        centroid = np.bincount(flat, weights=rows)[ids] / counts[ids]
        lut = np.zeros(counts.size, dtype=np.uint16)
        lut[ids[np.argsort(centroid, kind="stable")]] = np.arange(
            1, ids.size + 1, dtype=np.uint16,
        )
        out[t] = lut[frame]
    return out


def label_overlay(images: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Stack ``images`` with the outlines of ``labels`` as a second channel.

    Pixels under an outline are set to the image minimum so the outline stays
    visible under percentile contrast.

    Parameters
    ----------
    images, labels : np.ndarray
        ``(..., Y, X)`` arrays of the same shape.

    Returns
    -------
    np.ndarray
        ``float32`` array ``(..., 2, Y, X)``: image, then outline (0/1).
    """
    from skimage.segmentation import find_boundaries

    images = np.asarray(images, dtype=np.float32)
    labels = np.asarray(labels)
    out = np.empty(images.shape[:-2] + (2,) + images.shape[-2:], dtype=np.float32)
    frames = images.reshape((-1,) + images.shape[-2:])
    frame_labels = labels.reshape((-1,) + labels.shape[-2:])
    flat_out = out.reshape((-1, 2) + images.shape[-2:])
    for i, (image, frame) in enumerate(zip(frames, frame_labels)):
        outline = find_boundaries(frame, mode="inner")
        flat_out[i, 0] = image
        flat_out[i, 0][outline] = image.min()
        flat_out[i, 1] = outline
    return out


class Segmenter:
    """Segment one channel of a trench store with Omnipose, under several hypotheses.

    Labels are written to ``/segmentation/<name>`` inside the trench store, as
    ``labels(Hypothesis, Trench, T, Y, X)`` with the root dataset's coordinates
    and one chunk per hypothesis and trench, so they line up with ``data``::

        tree = xr.open_datatree(store, engine="zarr", consolidated=False)
        labels = tree["segmentation/mcherry"].ds["labels"].sel(Hypothesis="S1")

    The group also holds each trench's intensity bounds (``lower``, ``upper``).

    Parameters
    ----------
    store_path : str | Path
        Trench store written by :class:`~pymmm.extractor.Extractor`.
    channel : str
        Channel to segment, a value of the root ``C`` coordinate.
    model : str
        Omnipose pretrained model name.
    name : str | None
        Segmentation group name; defaults to ``channel.lower()``.
    gpu : bool
        Run Omnipose on the GPU.
    """

    def __init__(
        self,
        store_path: Union[str, Path],
        channel: str = "mCherry",
        model: str = "bact_fluor_omni",
        name: Optional[str] = None,
        gpu: bool = True,
    ) -> None:
        self.store_path = Path(store_path)
        self.root = xr.open_zarr(self.store_path, consolidated=False)
        channels = [str(c) for c in self.root["C"].values]
        if channel not in channels:
            raise ValueError(f"channel {channel!r} not in store channels {channels}")
        self.channel = channel
        self.channel_index = channels.index(channel)
        self.model_name = model
        self.name = name or channel.lower()
        self.group = f"segmentation/{self.name}"
        self.gpu = gpu
        self.pixel_size = float(
            xr.open_zarr(self.store_path, group="acquisition", consolidated=False)["pixel_size"]
        )
        self._model = None
        self._bounds: Dict[int, Tuple[float, float]] = {}

        saved = self._saved_attrs()
        if (
            saved is not None
            and saved.get("channel") == channel
            and saved.get("model") == model
        ):
            self._params = dict(saved["params"])
        else:
            self._params = dict(DEFAULT_PARAMS)
        if self._params["floor"] is None:
            self._params["floor"] = self.estimate_floor(self._params["floor_sigma"])

    # ------------------------------------------------------------------
    # Store access
    # ------------------------------------------------------------------

    def _open_store(self, mode: str = "r") -> zarr.Group:
        return zarr.open_group(str(self.store_path), mode=mode, zarr_format=3)

    def _saved_attrs(self) -> Optional[Dict[str, Any]]:
        store = self._open_store()
        if self.group not in store:
            return None
        return dict(store[self.group].attrs)

    @property
    def params(self) -> Dict[str, Any]:
        """Current base settings (see :data:`DEFAULT_PARAMS`)."""
        return dict(self._params)

    @property
    def hypotheses(self) -> List[str]:
        return list(HYPOTHESES)

    def settings(self, hypothesis: str, params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Base settings with ``hypothesis``'s overrides applied."""
        return {**(self._params if params is None else params), **HYPOTHESES[hypothesis]}

    def is_complete(self, hypothesis: Optional[str] = None) -> bool:
        attrs = self._saved_attrs() or {}
        complete = attrs.get("complete", {})
        names = [hypothesis] if hypothesis else self.hypotheses
        return all(complete.get(h, False) for h in names)

    def _group_attrs(self) -> Dict[str, Any]:
        """Attrs that identify what produced the labels (checked on resume)."""
        root_provenance = self.root.attrs.get("provenance", {})
        return _json_native({
            "channel": self.channel,
            "model": self.model_name,
            "params": self._params,
            "hypotheses": {h: self.settings(h) for h in self.hypotheses},
            "extraction_created": root_provenance.get("created"),
        })

    def _resume_mismatches(self, attrs: Mapping[str, Any]) -> List[str]:
        current = self._group_attrs()
        return [key for key, value in current.items() if attrs.get(key) != value]

    def _create_group(self) -> None:
        """(Re)write ``/segmentation/<name>`` with empty labels; root is untouched."""
        sizes = self.root.sizes
        shape = (len(self.hypotheses),) + tuple(sizes[d] for d in _LABEL_DIMS[1:])
        chunks = (1, 1) + shape[2:]
        n_trenches = sizes["Trench"]
        dataset = xr.Dataset(
            data_vars={
                "labels": (
                    _LABEL_DIMS,
                    da.zeros(shape, chunks=chunks, dtype=np.uint16),
                    _LABELS_ATTRS,
                ),
                "lower": (("Trench",), np.zeros(n_trenches, np.float32), {
                    "long_name": "intensity mapped to 0 (0 until the trench is segmented)",
                }),
                "upper": (("Trench",), np.zeros(n_trenches, np.float32), {
                    "long_name": "intensity mapped to 1 at gain 1 (0 until the trench is segmented)",
                }),
            },
            coords={
                "Hypothesis": np.asarray(self.hypotheses, dtype=object),
                **{dim: self.root[dim].variable for dim in _LABEL_DIMS[1:]},
            },
            attrs={
                "schema_version": SCHEMA_VERSION,
                "layout": "segmentation",
                "description": (
                    f"Omnipose labels of the {self.channel} channel for each trench "
                    "crop in the root dataset, under each hypothesis's settings."
                ),
                "provenance": provenance(),
                **self._group_attrs(),
                "completed": {h: [] for h in self.hypotheses},
                "complete": {h: False for h in self.hypotheses},
                "failed_frames": {h: {} for h in self.hypotheses},
            },
        )
        dataset.to_zarr(
            str(self.store_path),
            group=self.group,
            mode="w",
            compute=False,
            consolidated=False,
            zarr_format=3,
            encoding={
                "labels": {
                    "chunks": chunks,
                    "compressors": [
                        zarr.codecs.BloscCodec(
                            cname="zstd", clevel=3, shuffle="bitshuffle",
                        )
                    ],
                },
            },
        )

    def _reset(self, hypotheses: Sequence[str]) -> None:
        """Clear the labels and progress of ``hypotheses`` only."""
        group = self._open_store(mode="r+")[self.group]
        attrs = {key: dict(group.attrs.get(key, {})) for key in ("completed", "complete", "failed_frames")}
        for hypothesis in hypotheses:
            group["labels"][self.hypotheses.index(hypothesis)] = 0
            attrs["completed"][hypothesis] = []
            attrs["complete"][hypothesis] = False
            attrs["failed_frames"][hypothesis] = {}
        group.attrs.update(attrs)

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def estimate_floor(self, floor_sigma: float = 20.0, n_trenches: int = 32, seed: int = 0) -> float:
        """Store-wide floor for the upper intensity bound, in counts.

        For each of a random sample of trenches, background + ``floor_sigma``
        noise SDs (:func:`background_noise`, every 5th frame); the median over
        trenches, so a few crowded trenches (whose median pixel is a cell) do not
        move it.
        """
        data = self._open_store()["data"]
        n_total = self.root.sizes["Trench"]
        rng = np.random.default_rng(seed)
        picks = np.sort(rng.choice(n_total, size=min(n_trenches, n_total), replace=False))
        floors = []
        for i in picks:
            background, noise = background_noise(data[int(i), ::5, self.channel_index])
            floors.append(background + floor_sigma * noise)
        return float(np.median(floors))

    def bounds(self, index: int, stack: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """Intensity bounds of the trench at position ``index`` (cached)."""
        if index not in self._bounds:
            if stack is None:
                stack = self._open_store()["data"][index, :, self.channel_index]
            self._bounds[index] = trench_bounds(stack, self._params["floor"])
        return self._bounds[index]

    # ------------------------------------------------------------------
    # Omnipose
    # ------------------------------------------------------------------

    def _get_model(self):
        if self._model is None:
            for var, value in _PARLAY_THREADS.items():
                os.environ.setdefault(var, value)
            from omnipose.models import OmniModel

            with _quiet_omnipose():
                self._model = OmniModel(gpu=self.gpu, model_type=self.model_name)
        return self._model

    def segment_stack(
        self,
        stack: np.ndarray,
        bounds: Tuple[float, float],
        hypothesis: str = "S1",
        params: Optional[Mapping[str, Any]] = None,
        batch_size: int = 64,
    ) -> Tuple[np.ndarray, List[int]]:
        """Segment a ``(T, Y, X)`` stack of one trench under one hypothesis.

        Returns the labels ordered from the closed end and the frames Omnipose
        failed on (left empty). Omnipose 2.0 can raise ``IndexError`` while
        building masks for some frames; the stack is then retried frame by frame.
        """
        s = self.settings(hypothesis, params)
        x = normalise(stack, *bounds, gain=s["gain"])
        kwargs = {
            "normalize": False,
            "show_progress": False,
            "batch_size": batch_size,
            "rescale_factor": s["rescale"],
            "mask_threshold": s["mask_threshold"],
            "flow_threshold": s["flow_threshold"],
            "min_size": max(1, int(round(s["min_area"] / self.pixel_size ** 2))),
        }
        model = self._get_model()
        failed: List[int] = []
        with _quiet_omnipose():
            try:
                masks = np.asarray(model.eval(x, **kwargs).masks)
            except IndexError:
                masks = np.zeros(x.shape, dtype=np.int32)
                for t, frame in enumerate(x):
                    try:
                        masks[t] = np.asarray(model.eval(frame[None], **kwargs).masks)[0]
                    except IndexError:
                        failed.append(t)
        masks = drop_faint(masks, stack, s["min_contrast"])
        return order_from_closed_end(masks), failed

    # ------------------------------------------------------------------
    # Batch segmentation
    # ------------------------------------------------------------------

    def segment(
        self,
        hypotheses: Optional[Sequence[str]] = None,
        resume: bool = True,
        batch_size: int = 64,
        workers: int = 1,
        show_progress: bool = True,
    ) -> None:
        """Segment every trench and write ``/segmentation/<name>/labels``.

        Hypotheses are done one after another, each over all trenches, so the
        first is complete before the next starts. Each finished trench is
        recorded in the group's ``completed`` attr, so an interrupted run
        resumes.

        Most of each trench's time is Omnipose building masks on the CPU, which
        runs no faster with more threads; ``workers > 1`` segments that many
        trenches at once in separate single-threaded processes, each with its
        own copy of the model on the GPU.

        Parameters
        ----------
        hypotheses : sequence of str | None
            Which of :data:`HYPOTHESES` to run, in order; all if ``None``.
        resume : bool
            Continue an existing group made with the same channel, model,
            settings and extraction. ``False`` clears and re-runs only
            ``hypotheses``. Results of other hypotheses are never cleared: if
            the settings changed, every hypothesis that already has results must
            be among ``hypotheses``, or this raises.
        batch_size : int
            Frames per Omnipose GPU batch.
        workers : int
            Trenches segmented at once, each in its own process (1: in this
            process).
        show_progress : bool
            Show a tqdm progress bar per hypothesis.
        """
        hypotheses = list(hypotheses or self.hypotheses)
        unknown = [h for h in hypotheses if h not in HYPOTHESES]
        if unknown:
            raise ValueError(f"unknown hypotheses {unknown}; choose from {self.hypotheses}")

        saved = self._saved_attrs()
        if saved is None:
            self._create_group()
        else:
            mismatches = self._resume_mismatches(saved)
            with_results = [h for h in self.hypotheses if saved.get("completed", {}).get(h)]
            if mismatches:
                left_behind = [h for h in with_results if h not in hypotheses]
                if resume or left_behind:
                    rerun = [h for h in self.hypotheses if h in with_results or h in hypotheses]
                    raise RuntimeError(
                        f"{self.group} was segmented with different settings "
                        f"({', '.join(mismatches)} differ). "
                        + (f"{', '.join(with_results)} already have results with the saved settings. "
                           if with_results else "")
                        + f"To use the current settings, re-run them together: "
                        f"segment(hypotheses={rerun}, resume=False). "
                        "To keep the saved results, set the saved settings back instead."
                    )
                # Every hypothesis with results is being re-run: start the group afresh.
                self._create_group()
            elif not resume:
                self._reset(hypotheses)

        store = self._open_store(mode="r+")
        data = store["data"]
        group = store[self.group]
        labels = group["labels"]
        lower_arr, upper_arr = group["lower"], group["upper"]
        stored_upper = upper_arr[:]
        stored_bounds = {int(i) for i in np.flatnonzero(stored_upper > 0)}
        for index in stored_bounds:
            self._bounds[index] = (float(lower_arr[index]), float(stored_upper[index]))

        trench_ids = [int(i) for i in self.root["Trench"].values]
        completed = {h: set(v) for h, v in group.attrs.get("completed", {}).items()}
        failed_frames = dict(group.attrs.get("failed_frames", {}))

        def _read(index: int) -> np.ndarray:
            return data[index, :, self.channel_index]

        def _record(hypothesis: str, index: int, bounds: Tuple[float, float], failed: List[int]) -> None:
            self._bounds[index] = bounds
            if index not in stored_bounds:
                lower_arr[index], upper_arr[index] = bounds
                stored_bounds.add(index)
            tid = trench_ids[index]
            completed[hypothesis].add(tid)
            if failed:
                failed_frames.setdefault(hypothesis, {})[str(tid)] = failed
                group.attrs["failed_frames"] = failed_frames
            group.attrs["completed"] = {
                h: [t for t in trench_ids if t in completed.get(h, set())]
                for h in self.hypotheses
            }

        with _environment(_WORKER_THREADS if workers > 1 else {}):
            pool = None
            if workers > 1:
                pool = ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_init_worker,
                    initargs=(str(self.store_path), self.channel, self.model_name,
                              self.name, self.gpu, self._params),
                )
            reader = ThreadPoolExecutor(max_workers=1)
            try:
                for hypothesis in hypotheses:
                    h_index = self.hypotheses.index(hypothesis)
                    done = completed.setdefault(hypothesis, set())
                    todo = [i for i, tid in enumerate(trench_ids) if tid not in done]
                    progress = tqdm(
                        total=len(todo), desc=f"{hypothesis} {self.channel}", disable=not show_progress,
                    )
                    if pool is None:
                        pending = reader.submit(_read, todo[0]) if todo else None
                        for k, index in enumerate(todo):
                            stack = pending.result()
                            if k + 1 < len(todo):
                                pending = reader.submit(_read, todo[k + 1])
                            bounds = self.bounds(index, stack)
                            masks, failed = self.segment_stack(
                                stack, bounds, hypothesis, batch_size=batch_size,
                            )
                            labels[h_index, index] = masks
                            _record(hypothesis, index, bounds, failed)
                            progress.update()
                    else:
                        # Workers write their own label chunks (one per trench, so
                        # never shared); progress and bounds are recorded here.
                        queue = iter(todo)
                        running = {}

                        def _submit_next() -> None:
                            index = next(queue, None)
                            if index is not None:
                                task = (hypothesis, index, self._bounds.get(index), batch_size)
                                running[pool.submit(_segment_trench, task)] = index

                        for _ in range(2 * workers):
                            _submit_next()
                        while running:
                            finished, _ = wait(running, return_when=FIRST_COMPLETED)
                            for future in finished:
                                running.pop(future)
                                index, bounds, failed = future.result()
                                _record(hypothesis, index, bounds, failed)
                                progress.update()
                                _submit_next()
                    progress.close()
                    complete = dict(group.attrs.get("complete", {}))
                    complete[hypothesis] = done.issuperset(trench_ids)
                    group.attrs["complete"] = complete
            finally:
                reader.shutdown(wait=False, cancel_futures=True)
                if pool is not None:
                    pool.shutdown(wait=True, cancel_futures=True)

        print(f"Segmentation written: {self.store_path / self.group} ({', '.join(hypotheses)})")
        n_failed = sum(len(v) for h in hypotheses for v in failed_frames.get(h, {}).values())
        if n_failed:
            print(f"  Omnipose failed on {n_failed} frames (left empty); see attrs['failed_frames']")

    def labels(self, hypothesis: Optional[str] = None) -> xr.DataArray:
        """Lazy ``labels(Hypothesis, Trench, T, Y, X)``, or one hypothesis's."""
        labels = xr.open_zarr(
            self.store_path, group=self.group, consolidated=False,
        )["labels"]
        return labels if hypothesis is None else labels.sel(Hypothesis=hypothesis)

    # ------------------------------------------------------------------
    # Interactive tuning and viewing
    # ------------------------------------------------------------------

    def interactive_segment(self, batch_size: int = 64) -> None:
        """Display a widget for tuning the base settings on single trenches.

        Uses ipywidgets for the controls, ndv for the channel with label
        outlines (all frames of the selected trench, under the selected
        hypothesis), and matplotlib for the number of cells per frame. The
        sliders set the base settings; S2–S4 override some of them (shown under
        the hypothesis). **Apply** keeps the settings for :meth:`segment`.
        """
        import ipywidgets as widgets
        import matplotlib.pyplot as plt
        import ndv
        from IPython.display import display

        from pymmm.plotting import NDV_CLIMS

        params = self._params
        trench_ids = [int(i) for i in self.root["Trench"].values]
        data = self._open_store()["data"]
        sizes = self.root.sizes
        style = {"description_width": "initial"}

        hypothesis_select = widgets.Dropdown(
            options=self.hypotheses, value=self.hypotheses[0], description="hypothesis",
        )
        overrides_label = widgets.HTML(value="")
        trench_slider = widgets.IntSlider(
            value=0, min=0, max=len(trench_ids) - 1,
            description="trench (index)", continuous_update=False, style=style,
        )
        floor_slider = widgets.IntSlider(
            value=int(params["floor"]), min=0, max=max(int(params["floor"] * 4), 1),
            description="upper-bound floor", continuous_update=False, style=style,
        )
        gain_slider = widgets.FloatSlider(
            value=params["gain"], min=0.5, max=3.0, step=0.05,
            description="gain", continuous_update=False, style=style,
        )
        rescale_slider = widgets.FloatSlider(
            value=params["rescale"], min=0.5, max=1.5, step=0.05,
            description="image scale", continuous_update=False, style=style,
        )
        mask_slider = widgets.FloatSlider(
            value=params["mask_threshold"], min=-3.0, max=3.0, step=0.1,
            description="mask threshold", continuous_update=False, style=style,
        )
        flow_slider = widgets.FloatSlider(
            value=params["flow_threshold"], min=0.0, max=1.5, step=0.05,
            description="flow threshold", continuous_update=False, style=style,
        )
        area_slider = widgets.FloatSlider(
            value=params["min_area"], min=0.0, max=5.0, step=0.05,
            description="min area (µm²)", continuous_update=False, style=style,
        )
        contrast_slider = widgets.FloatSlider(
            value=params["min_contrast"], min=0.0, max=20.0, step=0.5,
            description="min contrast (σ)", continuous_update=False, style=style,
        )
        run_btn = widgets.Button(description="Segment trench", button_style="primary")
        apply_btn = widgets.Button(description="Apply", button_style="success")
        preview_label = widgets.HTML(value="")
        status_label = widgets.HTML(value="")
        output = widgets.Output()

        viewer_data = np.zeros(
            (sizes["T"], 2, sizes["Y"], sizes["X"]), dtype=np.float32,
        )
        image_viewer = ndv.ArrayViewer(
            viewer_data,
            channel_axis=1,
            channel_mode="composite",
            luts={
                0: {"name": self.channel, "cmap": "gray", "clims": NDV_CLIMS},
                1: {"name": "cell outlines", "cmap": "red", "clims": (0, 1)},
            },
        )

        def _widget_params() -> Dict[str, Any]:
            return {
                **params,
                "floor": float(floor_slider.value),
                "gain": float(gain_slider.value),
                "rescale": float(rescale_slider.value),
                "mask_threshold": float(mask_slider.value),
                "flow_threshold": float(flow_slider.value),
                "min_area": float(area_slider.value),
                "min_contrast": float(contrast_slider.value),
            }

        def _show_overrides(_change: Any = None) -> None:
            over = HYPOTHESES[hypothesis_select.value]
            overrides_label.value = (
                "overrides: " + ", ".join(f"{k} {v}" for k, v in over.items())
                if over else "uses the sliders as they are"
            )

        def _update(_change: Any = None) -> None:
            _show_overrides()
            index = trench_slider.value
            tid = trench_ids[index]
            hypothesis = hypothesis_select.value
            current = _widget_params()
            status_label.value = f"<b>Segmenting trench {tid} ({hypothesis})…</b>"
            stack = data[index, :, self.channel_index]
            bounds = trench_bounds(stack, current["floor"])
            start = time.perf_counter()
            masks, failed = self.segment_stack(stack, bounds, hypothesis, current, batch_size)
            elapsed = time.perf_counter() - start
            viewer_data[:] = label_overlay(stack, masks)
            image_viewer.data_wrapper.data_changed.emit()

            n_cells = masks.reshape(masks.shape[0], -1).max(axis=1)
            fov = str(self.root["fov_name"].values[index])
            preview_label.value = (
                f"<b>Trench {tid}</b> ({fov}), {hypothesis} — up to {int(n_cells.max())} cells; "
                f"bounds {bounds[0]:.0f}–{bounds[1]:.0f}"
            )
            status_label.value = f"Segmented {sizes['T']} frames in {elapsed:.1f} s"
            if failed:
                status_label.value += f"; Omnipose failed on frames {failed}"

            with output:
                output.clear_output(wait=True)
                fig, ax = plt.subplots(1, 1, figsize=(7, 2))
                ax.plot(n_cells, color="tab:red")
                ax.set_xlabel("frame")
                ax.set_ylabel("cells")
                ax.set_title(f"Trench {tid}, {hypothesis}: cells per frame")
                fig.tight_layout()
                plt.show()

        def _on_apply(_btn: Any) -> None:
            self._params = {k: v for k, v in _widget_params().items() if k in DEFAULT_PARAMS}
            self._bounds.clear()
            status_label.value = "<b>Applied</b> — base settings kept for segment()"
            saved = self._saved_attrs() or {}
            done = [h for h in self.hypotheses if saved.get("completed", {}).get(h)]
            if done and saved.get("params") != _json_native(self._params):
                status_label.value += (
                    f"<br>{', '.join(done)} already have results with the previous "
                    "settings; segment() will ask you to re-run them too."
                )

        run_btn.on_click(_update)
        apply_btn.on_click(_on_apply)
        trench_slider.observe(_update, names="value")
        hypothesis_select.observe(_update, names="value")

        controls = widgets.VBox([
            hypothesis_select, overrides_label, trench_slider,
            floor_slider, gain_slider, rescale_slider, mask_slider, flow_slider,
            area_slider, contrast_slider,
            widgets.HBox([run_btn, apply_btn]), status_label,
        ])
        image_and_counts = widgets.VBox([
            preview_label, image_viewer.widget(), output,
        ])
        display(widgets.HBox([controls, image_and_counts]))
        _update()

    def view(self):
        """Browse the stored labels as outlines over the channel: any hypothesis, trench and frame."""
        import ndv

        from pymmm.plotting import NDV_CLIMS

        return ndv.imshow(
            _label_overlay_wrapper(self),
            channel_axis=3,
            channel_mode="composite",
            luts={
                0: {"name": self.channel, "cmap": "gray", "clims": NDV_CLIMS},
                1: {"name": "cell outlines", "cmap": "red", "clims": (0, 1)},
            },
        )

    def __repr__(self) -> str:
        attrs = self._saved_attrs()
        if attrs is None:
            state = "not segmented"
        else:
            n = self.root.sizes["Trench"]
            done = attrs.get("completed", {})
            state = ", ".join(f"{h} {len(done.get(h, []))}/{n}" for h in self.hypotheses)
        return (
            f"Segmenter({self.store_path.name}, {self.group}, channel={self.channel!r}, "
            f"model={self.model_name!r}, {state})"
        )


# ----------------------------------------------------------------------
# Worker processes for Segmenter.segment(workers > 1)
# ----------------------------------------------------------------------

_WORKER: Optional[Segmenter] = None


def _init_worker(
    store_path: str, channel: str, model: str, name: str, gpu: bool, params: Dict[str, Any],
) -> None:
    """Process-pool initializer: a single-threaded Segmenter with its own model.

    The thread limits come from the parent's environment (``_WORKER_THREADS``).
    """
    global _WORKER
    _WORKER = Segmenter(store_path, channel=channel, model=model, name=name, gpu=gpu)
    _WORKER._params = dict(params)
    _WORKER._get_model()
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass


def _segment_trench(
    task: Tuple[str, int, Optional[Tuple[float, float]], int],
) -> Tuple[int, Tuple[float, float], List[int]]:
    """Segment one trench under one hypothesis in a worker; write its label chunk."""
    hypothesis, index, bounds, batch_size = task
    seg = _WORKER
    store = seg._open_store(mode="r+")
    stack = store["data"][index, :, seg.channel_index]
    if bounds is None:
        bounds = trench_bounds(stack, seg._params["floor"])
    masks, failed = seg.segment_stack(stack, bounds, hypothesis, batch_size=batch_size)
    store[seg.group]["labels"][seg.hypotheses.index(hypothesis), index] = masks
    return index, bounds, failed


def _label_overlay_wrapper(segmenter: Segmenter):
    """ndv data ``(Hypothesis, Trench, T, C, Y, X)``: the channel and label
    outlines, read from the store only for the frames ndv asks for."""
    import ndv

    store = segmenter._open_store()
    data = store["data"]
    labels = store[segmenter.group]["labels"]
    channel_index = segmenter.channel_index
    sizes = segmenter.root.sizes
    n_hypotheses = labels.shape[0]

    class _LabelOverlay(ndv.DataWrapper):
        @classmethod
        def supports(cls, obj: Any) -> bool:
            return False

        @property
        def dims(self) -> Tuple[str, ...]:
            return ("Hypothesis", "Trench", "T", "C", "Y", "X")

        @property
        def coords(self) -> Dict[str, Sequence]:
            # ndv's Jupyter sliders only take ranges: Hypothesis and Trench are by position.
            return {
                "Hypothesis": range(n_hypotheses),
                "Trench": range(sizes["Trench"]),
                "T": range(sizes["T"]),
                "C": range(2),
                "Y": range(sizes["Y"]),
                "X": range(sizes["X"]),
            }

        def channel_names(self, channel_axis: Optional[int]) -> Dict[int, str]:
            return {0: segmenter.channel, 1: "outlines"}

        @property
        def dtype(self) -> np.dtype:
            return np.dtype(np.float32)

        def isel(self, index: Mapping[int, Union[int, slice]]) -> np.ndarray:
            h, trench, t, c, y, x = (index.get(axis, slice(None)) for axis in range(6))
            images = data[trench, t, channel_index]
            frame_labels = labels[h, trench, t]
            if isinstance(h, slice):
                images = np.broadcast_to(images, frame_labels.shape)
            overlay = label_overlay(images, frame_labels)
            return overlay[..., c, y, x]

    return _LabelOverlay(segmenter)
