"""Segmenter – Omnipose cell labels for every trench of an experiment group.

Each trench is scaled by its own intensity percentiles (with an experiment-wide
floor so trenches without cells keep their noise dark), then segmented under a few
preset Omnipose settings, the *hypotheses*. S1 is the primary segmentation; S2–S4
mostly keep pairs that have just divided as one cell, so downstream tracking can
choose between them. Each trench gets one ``(t, y, x)`` label image per hypothesis,
``labels/<name>.<hypothesis>`` beside its pixels (see :mod:`pymmm.store.experiment_group`), and
the experiment group records the segmentation's settings once, in
``attrs.pymmm.segmentations[<name>]``.
"""

from __future__ import annotations

import datetime
import logging
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import zarr
from tqdm.auto import tqdm

from pymmm.parallel import SPAWN
from pymmm.store.experiment_group import (
    calibration,
    has_segmentation,
    image,
    open_experiment,
    records,
    remove_segmentation,
    segmentation,
    trench_ids,
    update_experiment,
    write_segmentation,
)
from pymmm.store.provenance import SCHEMA_VERSION, plain_json, provenance

LONG_NAME = (
    "cell labels; 0 is background, 1..n are ordered from the trench's closed "
    "end (y = 0), so 1 is the cell nearest the closed end"
)

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


class Segmenter:
    """Segment one channel of an experiment group with Omnipose, under several hypotheses.

    A segmentation is a fixed set of hypotheses (by default all of :data:`HYPOTHESES`),
    segmented and written together for each trench: one ``(t, y, x)`` label image per
    hypothesis, ``trench_NNNN/labels/<name>.<hypothesis>``, beside the trench's pixels.
    A trench is finished once all of them are listed in its ``labels/``, so an
    interrupted run resumes with the unfinished trenches. The settings and hypotheses
    are recorded once, in the experiment group's ``attrs.pymmm.segmentations[<name>]``::

        root = experiment_group.open_experiment(path)
        names, labels = experiment_group.segmentation(root, 11, "mcherry")   # S1 first

    Each label image's ``pymmm`` attributes hold the trench's intensity bounds
    (``lower``, ``upper``) and the frames Omnipose failed on (``failed_frames``).

    Parameters
    ----------
    store_path : str | Path
        Experiment group written by :class:`~pymmm.steps.extraction.Extractor`.
    channel : str
        Channel to segment, a value of the group's ``C``.
    model : str
        Omnipose pretrained model name.
    name : str | None
        Segmentation name; defaults to ``channel.lower()``.
    hypotheses : sequence of str | None
        The hypotheses of :data:`HYPOTHESES` that make up the segmentation, the base
        first. ``None``: the recorded ones if the segmentation exists with this channel
        and model, else all.
    gpu : bool
        Run Omnipose on the GPU.
    """

    def __init__(
        self,
        store_path: Union[str, Path],
        channel: str = "mCherry",
        model: str = "bact_fluor_omni",
        name: Optional[str] = None,
        hypotheses: Optional[Sequence[str]] = None,
        gpu: bool = True,
    ) -> None:
        self.store_path = Path(store_path)
        self.root = open_experiment(self.store_path)
        channels = [str(c) for c in self.root["C"][:]]
        if channel not in channels:
            raise ValueError(f"channel {channel!r} not in the experiment's channels {channels}")
        self.channel = channel
        self.channel_index = channels.index(channel)
        self.model_name = model
        self.name = name or channel.lower()
        self.gpu = gpu
        self.pixel_size = calibration(self.root)[0]
        self.trench_ids = trench_ids(self.root)
        self._model = None
        self._bounds: Dict[int, Tuple[float, float]] = {}

        saved = self._saved_record()
        same = saved is not None and saved.get("channel") == channel and saved.get("model") == model
        if hypotheses is None:
            hypotheses = saved["hypotheses"] if same else list(HYPOTHESES)
        unknown = [h for h in hypotheses if h not in HYPOTHESES]
        if unknown or not hypotheses:
            raise ValueError(f"unknown hypotheses {unknown}; choose from {list(HYPOTHESES)}")
        self._hypotheses = [str(h) for h in hypotheses]
        self._params = dict(saved["params"]) if same else dict(DEFAULT_PARAMS)
        if self._params["floor"] is None:
            self._params["floor"] = self.estimate_floor(self._params["floor_sigma"])

    # ------------------------------------------------------------------
    # Experiment group access
    # ------------------------------------------------------------------

    def _open(self, mode: str = "r") -> zarr.Group:
        """A fresh handle, so attributes written since are seen."""
        return open_experiment(self.store_path, mode=mode)

    def _saved_record(self) -> Optional[Dict[str, Any]]:
        return records(self._open(), "segmentations").get(self.name)

    def read(self, trench_id: int) -> np.ndarray:
        """The trench's ``(t, y, x)`` frames of the channel."""
        return np.asarray(image(self.root, trench_id)[:, self.channel_index])

    @property
    def params(self) -> Dict[str, Any]:
        """Current base settings (see :data:`DEFAULT_PARAMS`)."""
        return dict(self._params)

    @property
    def hypotheses(self) -> List[str]:
        """The segmentation's hypotheses, the base first."""
        return list(self._hypotheses)

    def settings(self, hypothesis: str, params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Base settings with ``hypothesis``'s overrides applied."""
        return {**(self._params if params is None else params), **HYPOTHESES[hypothesis]}

    def _record(self) -> Dict[str, Any]:
        """What identifies the labels (compared with the saved record on resume)."""
        extraction = dict(self.root.attrs)["pymmm"].get("provenance") or {}
        return plain_json({
            "channel": self.channel,
            "model": self.model_name,
            "params": self._params,
            "hypotheses": self.hypotheses,
            "settings": {h: self.settings(h) for h in self.hypotheses},
            "extraction_created": extraction.get("created"),
        })

    def finished(self) -> List[int]:
        """The trenches whose segmentation is finished, in table order."""
        root = self._open()
        if self.name not in records(root, "segmentations"):
            return []
        return [t for t in self.trench_ids if has_segmentation(root, t, self.name)]

    def set_params(self, params: Mapping[str, Any]) -> List[int]:
        """Use these base settings (see :data:`DEFAULT_PARAMS`) from now on.

        Returns the finished trenches whose labels were made with other settings:
        :meth:`segment` will ask for them to be re-segmented too.
        """
        self._params = {k: v for k, v in params.items() if k in DEFAULT_PARAMS}
        self._bounds.clear()
        saved = self._saved_record() or {}
        done = self.finished()
        return done if done and saved.get("params") != plain_json(self._params) else []

    def is_complete(self) -> bool:
        return len(self.finished()) == len(self.trench_ids)

    def labels(self, trench_id: int, hypothesis: Optional[str] = None) -> zarr.Array:
        """A finished trench's ``(t, y, x)`` labels under one hypothesis (the base if ``None``)."""
        names, arrays = segmentation(self._open(), trench_id, self.name)
        return arrays[names.index(hypothesis or names[0])]

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def estimate_floor(self, floor_sigma: float = 20.0, n_trenches: int = 32, seed: int = 0) -> float:
        """Experiment-wide floor for the upper intensity bound, in counts.

        For each of a random sample of trenches, background + ``floor_sigma``
        noise SDs (:func:`background_noise`, every 5th frame); the median over
        trenches, so a few crowded trenches (whose median pixel is a cell) do not
        move it.
        """
        rng = np.random.default_rng(seed)
        n_total = len(self.trench_ids)
        picks = np.sort(rng.choice(n_total, size=min(n_trenches, n_total), replace=False))
        floors = []
        for i in picks:
            frames = image(self.root, self.trench_ids[int(i)])[::5, self.channel_index]
            background, noise = background_noise(frames)
            floors.append(background + floor_sigma * noise)
        return float(np.median(floors))

    def bounds(self, trench_id: int, stack: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """Intensity bounds of a trench (cached)."""
        if trench_id not in self._bounds:
            if stack is None:
                stack = self.read(trench_id)
            self._bounds[trench_id] = trench_bounds(stack, self._params["floor"])
        return self._bounds[trench_id]

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

    def _segment_trench(
        self, root: zarr.Group, trench_id: int, stack: np.ndarray, batch_size: int,
    ) -> Tuple[Tuple[float, float], Dict[str, List[int]]]:
        """Segment a trench under every hypothesis and write its label images."""
        bounds = self.bounds(trench_id, stack)
        labels: Dict[str, np.ndarray] = {}
        failed: Dict[str, List[int]] = {}
        for hypothesis in self.hypotheses:
            labels[hypothesis], frames = self.segment_stack(
                stack, bounds, hypothesis, batch_size=batch_size,
            )
            if frames:
                failed[hypothesis] = frames
        write_segmentation(root, trench_id, self.name, labels, attrs={
            "long_name": LONG_NAME, "channel": self.channel, "model": self.model_name,
            "lower": bounds[0], "upper": bounds[1], "failed_frames": failed,
        })
        return bounds, failed

    # ------------------------------------------------------------------
    # Batch segmentation
    # ------------------------------------------------------------------

    def _prepare(self, root: zarr.Group, wanted: Sequence[int], resume: bool) -> None:
        """Check the saved record against the current settings; clear what is re-run."""
        current = self._record()
        saved = records(root, "segmentations").get(self.name)
        mismatches: List[str] = []
        if saved is not None:
            done = [t for t in self.trench_ids if has_segmentation(root, t, self.name)]
            mismatches = [key for key, value in current.items() if saved.get(key) != value]
            if mismatches and done:
                left_behind = sorted(set(done) - set(wanted))
                if resume or left_behind:
                    raise RuntimeError(
                        f"{self.name} was segmented with different settings "
                        f"({', '.join(mismatches)} differ), and {len(done)} trenches have "
                        "results with the saved settings. To use the current settings, "
                        "re-segment every trench: segment(resume=False). To keep the saved "
                        "results, set the saved settings back instead."
                    )
            if mismatches or not resume:
                # Unlisted before the record changes, so no trench mixes settings.
                for trench_id in self.trench_ids if mismatches else wanted:
                    remove_segmentation(root, trench_id, self.name)
        if saved is None or mismatches:
            update_experiment(root, "segmentations", self.name, {
                **current,
                "description": (
                    f"Omnipose labels of the {self.channel} channel under each hypothesis's "
                    f"settings; each trench's labels/{self.name}.<hypothesis>"
                ),
                "schema_version": SCHEMA_VERSION,
                "provenance": plain_json(provenance()),
                "made_by": "pymmm.steps.segmentation.Segmenter",
                "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            })

    def segment(
        self,
        trenches: Optional[Sequence[int]] = None,
        resume: bool = True,
        batch_size: int = 64,
        workers: int = 1,
        show_progress: bool = True,
    ) -> None:
        """Segment the trenches under every hypothesis and write their label images.

        Each trench is finished once its label images are all written, so an
        interrupted run resumes with the trenches not finished.

        Most of each trench's time is Omnipose building masks on the CPU, which
        runs no faster with more threads; ``workers > 1`` segments that many
        trenches at once in separate single-threaded processes, each with its
        own copy of the model on the GPU, each writing its own trenches.

        Parameters
        ----------
        trenches : sequence of int | None
            Trench ids to segment (a few, to check the settings before the rest);
            all if ``None``.
        resume : bool
            Keep the trenches already finished with the same channel, model,
            settings, hypotheses and extraction. ``False`` re-segments
            ``trenches``. Results are never cleared otherwise: if the settings
            changed, every trench that has results must be re-segmented
            (``segment(resume=False)``), or this raises.
        batch_size : int
            Frames per Omnipose GPU batch.
        workers : int
            Trenches segmented at once, each in its own process (1: in this
            process).
        show_progress : bool
            Show a tqdm progress bar.
        """
        wanted = list(self.trench_ids) if trenches is None else [int(t) for t in trenches]
        unknown = sorted(set(wanted) - set(self.trench_ids))
        if unknown:
            raise ValueError(f"no trenches {unknown} in {self.store_path.name}")
        root = self._open(mode="r+")
        self._prepare(root, wanted, resume)
        todo = [t for t in wanted if not has_segmentation(root, t, self.name)]
        failed_frames: Dict[int, Dict[str, List[int]]] = {}

        progress = tqdm(
            total=len(todo), desc=f"{self.name} {', '.join(self.hypotheses)}",
            disable=not show_progress,
        )
        with _environment(_WORKER_THREADS if workers > 1 else {}):
            pool = None
            if workers > 1 and todo:
                pool = ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=SPAWN,
                    initializer=_init_worker,
                    initargs=(str(self.store_path), self.channel, self.model_name,
                              self.name, self.hypotheses, self.gpu, self._params),
                )
            reader = ThreadPoolExecutor(max_workers=1)
            try:
                if pool is None:
                    pending = reader.submit(self.read, todo[0]) if todo else None
                    for k, trench_id in enumerate(todo):
                        stack = pending.result()
                        if k + 1 < len(todo):
                            pending = reader.submit(self.read, todo[k + 1])
                        _, failed = self._segment_trench(root, trench_id, stack, batch_size)
                        if failed:
                            failed_frames[trench_id] = failed
                        progress.update()
                else:
                    # Workers write their own trenches' label images (never shared).
                    queue = iter(todo)
                    running = {}

                    def _submit_next() -> None:
                        trench_id = next(queue, None)
                        if trench_id is not None:
                            task = (trench_id, self._bounds.get(trench_id), batch_size)
                            running[pool.submit(_segment_in_worker, task)] = trench_id

                    for _ in range(2 * workers):
                        _submit_next()
                    while running:
                        finished, _ = wait(running, return_when=FIRST_COMPLETED)
                        for future in finished:
                            running.pop(future)
                            trench_id, bounds, failed = future.result()
                            self._bounds[trench_id] = bounds
                            if failed:
                                failed_frames[trench_id] = failed
                            progress.update()
                            _submit_next()
            finally:
                progress.close()
                reader.shutdown(wait=False, cancel_futures=True)
                if pool is not None:
                    pool.shutdown(wait=True, cancel_futures=True)

        done = self.finished()
        print(
            f"Segmentation {self.name} ({', '.join(self.hypotheses)}) written to "
            f"{self.store_path}/trench_NNNN/labels: {len(done)}/{len(self.trench_ids)} trenches finished"
        )
        n_failed = sum(len(v) for f in failed_frames.values() for v in f.values())
        if n_failed:
            print(f"  Omnipose failed on {n_failed} frames (left empty) in trenches "
                  f"{sorted(failed_frames)}; see each label image's attrs['pymmm']['failed_frames']")

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
        from pymmm.widgets.segmentation import segmentation_widget

        segmentation_widget(self, batch_size)

    def view(self):
        """Browse the written labels as outlines over the channel: any hypothesis,
        trench and frame (unfinished trenches show no outlines)."""
        from pymmm.widgets.segmentation import label_viewer

        return label_viewer(self)

    def __repr__(self) -> str:
        return (
            f"Segmenter({self.store_path.name}, labels/{self.name}.<{'|'.join(self.hypotheses)}>, "
            f"channel={self.channel!r}, model={self.model_name!r}, "
            f"{len(self.finished())}/{len(self.trench_ids)} trenches finished)"
        )


# ----------------------------------------------------------------------
# Worker processes for Segmenter.segment(workers > 1)
# ----------------------------------------------------------------------

_WORKER: Optional[Segmenter] = None


def _init_worker(
    store_path: str, channel: str, model: str, name: str, hypotheses: List[str],
    gpu: bool, params: Dict[str, Any],
) -> None:
    """Process-pool initializer: a single-threaded Segmenter with its own model.

    The thread limits come from the parent's environment (``_WORKER_THREADS``).
    """
    global _WORKER
    _WORKER = Segmenter(store_path, channel=channel, model=model, name=name,
                        hypotheses=hypotheses, gpu=gpu)
    _WORKER._params = dict(params)
    _WORKER._get_model()
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass


def _segment_in_worker(
    task: Tuple[int, Optional[Tuple[float, float]], int],
) -> Tuple[int, Tuple[float, float], Dict[str, List[int]]]:
    """Segment one trench under every hypothesis in a worker; write its label images."""
    trench_id, bounds, batch_size = task
    seg = _WORKER
    if bounds is not None:
        seg._bounds[trench_id] = bounds
    bounds, failed = seg._segment_trench(
        seg._open(mode="r+"), trench_id, seg.read(trench_id), batch_size,
    )
    return trench_id, bounds, failed
