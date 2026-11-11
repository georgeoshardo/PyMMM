"""Segmenter – Omnipose cell labels for every trench of an experiment group.

Each trench is scaled by its own intensity percentiles (with an experiment-wide
floor so trenches without cells can keep their noise dark), then segmented under
several *hypotheses*. A hypothesis is an Omnipose model with settings: a stock model
under a preset of settings (:data:`HYPOTHESES`: S1 is the primary segmentation, S2–S4
mostly keep pairs that have just divided as one cell), or a fine-tuned checkpoint
(:func:`checkpoint_hypotheses`: several saved epochs of one training run). Downstream
tracking chooses between them. Each trench gets one ``(t, y, x)`` label image per
hypothesis, ``labels/<name>.<hypothesis>`` beside its pixels (see
:mod:`pymmm.store.experiment_group`), and the experiment group records the
segmentation's settings once, in ``attrs.pymmm.segmentations[<name>]``.

Trenches are segmented where :class:`~pymmm.parallel.Runner` runs tasks: on the
workers of the active Dask client (on any machine with the same environment: the
task carries the trench's frames and, once per worker, the checkpoints, and returns
the labels, which this process writes), in a local process pool, or in this process.
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import os
import socket
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import zarr
from tqdm.auto import tqdm

from pymmm.parallel import Runner, shared
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
    "masks": "eval",
    "mask_threshold": 0.0,
    "flow_threshold": 0.4,
    "affinity_angle": 60.0,
    "min_area": 1.75,
    "min_contrast": 5.0,
}
"""Base settings, shared by every hypothesis unless the hypothesis overrides them.

``floor``: lowest allowed upper intensity bound, in counts; ``None`` estimates it
as background + ``floor_sigma`` background-noise SDs (median over sampled
trenches). ``gain``: multiplies the trench's intensity range (above 1 lowers
contrast). ``rescale``: Omnipose ``rescale_factor``. ``masks``: how masks are built
from the network's output, ``"eval"`` (``OmniModel.eval``'s own) or ``"affinity"``
(:func:`pymmm.steps.omnipose.affinity_masks`, with ``affinity_angle``).
``min_area``: smallest mask kept, in µm². ``min_contrast``: masks whose mean is
fewer than this many background-noise SDs above the frame's background are dropped
(``None``: none are, as for phase contrast, where cells are darker than the
background).
"""

MASKS = ("eval", "affinity")

HYPOTHESES: Dict[str, Dict[str, Any]] = {
    "S1": {},
    "S2": {"gain": 1.5, "rescale": 0.8, "flow_threshold": 0.0},
    "S3": {"rescale": 0.8, "mask_threshold": -1.0, "flow_threshold": 0.0},
    "S4": {"gain": 2.5, "rescale": 0.8, "mask_threshold": -1.0, "flow_threshold": 0.0},
}
"""Preset overrides of the base settings, for a stock model. Chosen from a sweep of
360 settings on mCherry: S1 splits dividing cells the most; S2–S4 cover the
alternative segmentations (mostly just-divided pairs kept as one cell, fuller masks
in crowded trenches) without adding masks on noise."""

_SOURCE_KEYS = ("model", "checkpoint")

# Omnipose 2.0 logs mask building through the top-level "core" and "gpu" loggers.
_OMNIPOSE_LOGGERS = ("omnipose", "core", "gpu")

# What each process-pool worker gets: one thread per library, so ``workers``
# processes use ``workers`` cores.
_WORKER_THREADS = {
    var: "1" for var in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMBA_NUM_THREADS", "PARLAY_NUM_THREADS",
    )
}

# Dask warns about any task argument over 10 MB; a trench's frames are tens of MB,
# sent once to the one worker that segments them.
_LARGE_TASK = {"distributed.admin.large-graph-warning-threshold": "4GB"}


def preset_hypotheses(
    model: str = "bact_fluor_omni", names: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """The :data:`HYPOTHESES` (all, or ``names``) for the stock Omnipose ``model``."""
    names = list(HYPOTHESES) if names is None else [str(h) for h in names]
    unknown = [h for h in names if h not in HYPOTHESES]
    if unknown or not names:
        raise ValueError(f"unknown hypotheses {unknown}; choose from {list(HYPOTHESES)}")
    return {h: {"model": model, **HYPOTHESES[h]} for h in names}


def checkpoint_hypotheses(checkpoints: Mapping[str, str]) -> Dict[str, Dict[str, Any]]:
    """One hypothesis per fine-tuned checkpoint, ``{hypothesis: checkpoint file name}``
    (in the Segmenter's ``models`` folder), the base first; all under the base settings."""
    return {str(h): {"checkpoint": str(c)} for h, c in checkpoints.items()}


def overrides(spec: Mapping[str, Any]) -> Dict[str, Any]:
    """A hypothesis's overrides of the base settings (its spec without its model)."""
    return {k: v for k, v in spec.items() if k not in _SOURCE_KEYS}


def _check_spec(name: str, spec: Mapping[str, Any]) -> Dict[str, Any]:
    spec = dict(spec)
    if sum(k in spec for k in _SOURCE_KEYS) != 1:
        raise ValueError(f"hypothesis {name}: give one of 'model' or 'checkpoint', not {spec}")
    unknown = sorted(set(overrides(spec)) - set(DEFAULT_PARAMS))
    if unknown:
        raise ValueError(f"hypothesis {name}: unknown settings {unknown}")
    return spec


def _clean_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Base settings: known keys only; ``min_contrast`` ``"off"`` (TOML has no null) is ``None``."""
    unknown = sorted(set(params) - set(DEFAULT_PARAMS))
    if unknown:
        raise ValueError(f"unknown settings {unknown}; choose from {list(DEFAULT_PARAMS)}")
    out = {**DEFAULT_PARAMS, **params}
    if out["min_contrast"] in ("off", False):
        out["min_contrast"] = None
    if out["masks"] not in MASKS:
        raise ValueError(f"masks {out['masks']!r}: choose from {list(MASKS)}")
    return out


def sha256(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def model_shape(shape: Sequence[int], pixel_size: float,
                model_pixel_size: Optional[float]) -> Tuple[int, int]:
    """A frame's ``(Y, X)`` once resampled from ``pixel_size`` to ``model_pixel_size``
    µm per pixel (its own when ``model_pixel_size`` is None)."""
    y, x = (int(n) for n in shape[-2:])
    if model_pixel_size is None:
        return y, x
    factor = float(pixel_size) / float(model_pixel_size)
    return max(1, int(round(y * factor))), max(1, int(round(x * factor)))


def model_input(stack: np.ndarray, bounds: Tuple[float, float], gain: float,
                shape: Sequence[int]) -> np.ndarray:
    """The ``(T, Y, X)`` stack Omnipose sees: :func:`normalise`, then each frame resized
    to ``shape`` (area-averaged when shrinking, bilinear when enlarging)."""
    stack = np.asarray(stack)
    shape = tuple(int(n) for n in shape)
    if stack.shape[1:] == shape:
        return normalise(stack, *bounds, gain=gain)
    import cv2

    shrink = shape[0] * shape[1] < stack.shape[1] * stack.shape[2]
    interpolation = cv2.INTER_AREA if shrink else cv2.INTER_LINEAR
    out = np.empty((len(stack), *shape), dtype=np.float32)
    for t, frame in enumerate(stack):
        out[t] = cv2.resize(normalise(frame, *bounds, gain=gain), (shape[1], shape[0]),
                            interpolation=interpolation)
    return out


def scale_labels(labels: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """``(T, y, x)`` labels resized to ``shape`` (Y, X), mask by mask: each mask resized
    bilinearly and kept where it is above one half. Masks stay disjoint (two disjoint
    masks cannot both be above one half at a pixel), and keep their numbers."""
    labels = np.asarray(labels)
    shape = tuple(int(n) for n in shape)
    if labels.shape[1:] == shape:
        return labels
    import cv2

    out = np.zeros((len(labels), *shape), dtype=labels.dtype)
    for frame, scaled in zip(labels, out):
        for label in np.unique(frame[frame > 0]):
            mask = cv2.resize((frame == label).astype(np.float32), (shape[1], shape[0]),
                              interpolation=cv2.INTER_LINEAR)
            scaled[mask > 0.5] = label
    return out


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


def segment_with(
    model: Any,
    settings: Mapping[str, Any],
    stack: np.ndarray,
    bounds: Tuple[float, float],
    pixel_size: float,
    batch_size: int = 64,
    model_pixel_size: Optional[float] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Segment a ``(T, Y, X)`` stack with one model under one hypothesis's settings.

    Returns the ``uint16`` labels ordered from the closed end and the frames
    Omnipose failed on (left empty). Omnipose 2.0 can raise ``IndexError`` while
    building masks for some frames; with ``masks="eval"`` the stack is then retried
    frame by frame, with ``masks="affinity"`` each frame is built on its own anyway.

    ``model_pixel_size``: the µm per pixel of the images the model learned from. When
    it differs from ``pixel_size``, Omnipose sees each frame resampled to it
    (:func:`model_input`) and each mask is scaled back to the stack's own size
    (:func:`scale_labels`), so the labels line up with the pixels.
    """
    s = settings
    small = model_shape(np.shape(stack), pixel_size, model_pixel_size)
    x = model_input(stack, bounds, s["gain"], small)
    scale = pixel_size if small == tuple(np.shape(stack)[1:]) else model_pixel_size
    min_size = max(1, int(round(s["min_area"] / scale ** 2)))
    failed: List[int] = []
    with _quiet_omnipose():
        if s["masks"] == "affinity":
            masks = _affinity(model, x, s, min_size, batch_size, failed)
        else:
            masks = _eval(model, x, s, min_size, batch_size, failed)
    if masks.shape != np.shape(stack):  # uint16 at full size, as written (and half the memory)
        masks = scale_labels(masks.astype(np.uint16), np.shape(stack)[1:])
    if s["min_contrast"] is not None:
        masks = drop_faint(masks, stack, s["min_contrast"])
    return order_from_closed_end(masks), failed


def _eval(model, x, s, min_size, batch_size, failed) -> np.ndarray:
    kwargs = {
        "normalize": False,
        "show_progress": False,
        "batch_size": batch_size,
        "rescale_factor": s["rescale"],
        "mask_threshold": s["mask_threshold"],
        "flow_threshold": s["flow_threshold"],
        "min_size": min_size,
    }
    try:
        return np.asarray(model.eval(x, **kwargs).masks)
    except IndexError:
        masks = np.zeros(x.shape, dtype=np.int32)
        for t, frame in enumerate(x):
            try:
                masks[t] = np.asarray(model.eval(frame[None], **kwargs).masks)[0]
            except IndexError:
                failed.append(t)
        return masks


def _affinity(model, x, s, min_size, batch_size, failed) -> np.ndarray:
    """The network's flows, ``batch_size`` frames at a time; then each frame's masks."""
    from pymmm.steps.omnipose import affinity_masks

    if s["rescale"] != 1.0:
        raise ValueError("masks='affinity' builds masks at the image's own scale: set rescale to 1")
    masks = np.zeros(x.shape, dtype=np.int32)
    for start in range(0, len(x), batch_size):
        out = model.eval(x[start:start + batch_size], normalize=False, compute_masks=False,
                         show_progress=False, batch_size=batch_size)
        for k, flows in enumerate(out.flows):
            try:
                masks[start + k] = affinity_masks(
                    np.asarray(flows.dP), np.asarray(flows.dist), model.device,
                    mask_threshold=s["mask_threshold"], flow_threshold=s["flow_threshold"],
                    angle=s["affinity_angle"], min_size=min_size,
                )
            except IndexError:
                failed.append(start + k)
    return masks


class Segmenter:
    """Segment one channel of an experiment group with Omnipose, under several hypotheses.

    A segmentation is a fixed set of hypotheses, segmented and written together for
    each trench: one ``(t, y, x)`` label image per hypothesis,
    ``trench_NNNN/labels/<name>.<hypothesis>``, beside the trench's pixels. A trench
    is finished once all of them are listed in its ``labels/``, so an interrupted run
    resumes with the unfinished trenches. The settings and hypotheses (each with its
    model, and each checkpoint's sha256) are recorded once, in the experiment group's
    ``attrs.pymmm.segmentations[<name>]``::

        root = experiment_group.open_experiment(path)
        names, labels = experiment_group.segmentation(root, 11, "mcherry")   # base first

    Each label image's ``pymmm`` attributes hold the trench's intensity bounds
    (``lower``, ``upper``) and the frames Omnipose failed on (``failed_frames``).

    Parameters
    ----------
    store_path : str | Path
        Experiment group written by :class:`~pymmm.steps.extraction.Extractor`.
    channel : str
        Channel to segment, a value of the group's ``C``.
    model : str
        Stock Omnipose model of the preset hypotheses (:data:`HYPOTHESES`).
    name : str | None
        Segmentation name; defaults to ``channel.lower()``.
    hypotheses : sequence of str | mapping | None
        The hypotheses, the base first: names of :data:`HYPOTHESES` (for ``model``),
        or ``{name: spec}`` with each spec a ``model`` or ``checkpoint`` and its
        overrides of the base settings (:func:`checkpoint_hypotheses`). ``None``: the
        recorded ones if the segmentation exists with this channel, else every preset.
    models : str | Path | None
        Folder of the checkpoint files the hypotheses name.
    params : mapping | None
        Base settings of a new segmentation, over :data:`DEFAULT_PARAMS` (an existing
        one keeps its recorded settings; change them with :meth:`set_params`).
    model_pixel_size : float | None
        µm per pixel of the images the models learned from, when the experiment's
        differ (fine-tuned at 40x, segmenting 100x): Omnipose sees each frame resampled
        to it and the masks are scaled back to the trench's own size
        (:func:`segment_with`). ``None``: the experiment's own.
    gpu : bool
        Run Omnipose on the GPU.
    """

    def __init__(
        self,
        store_path: Union[str, Path],
        channel: str = "mCherry",
        model: str = "bact_fluor_omni",
        name: Optional[str] = None,
        hypotheses: Union[Sequence[str], Mapping[str, Mapping[str, Any]], None] = None,
        models: Union[str, Path, None] = None,
        params: Optional[Mapping[str, Any]] = None,
        model_pixel_size: Optional[float] = None,
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
        self.models_dir = None if models is None else Path(models)
        self.pixel_size = calibration(self.root)[0]
        self.model_pixel_size = None if model_pixel_size is None else float(model_pixel_size)
        self.trench_ids = trench_ids(self.root)
        self._models: Dict[str, Any] = {}
        self._bounds: Dict[int, Tuple[float, float]] = {}
        self._sha: Dict[str, str] = {}
        self._loader: Optional[str] = None  # "module:function" loading models in tasks (tests)

        saved = self._saved_record()
        same = saved is not None and saved.get("channel") == channel and "specs" in saved
        if hypotheses is None:
            specs = saved["specs"] if same else preset_hypotheses(model)
        elif isinstance(hypotheses, Mapping):
            specs = hypotheses
        else:
            specs = preset_hypotheses(model, hypotheses)
        if not specs:
            raise ValueError("a segmentation needs at least one hypothesis")
        self._specs = {str(h): _check_spec(str(h), s) for h, s in specs.items()}
        for h in self._specs:
            path = self.checkpoint_path(h)
            if path is not None and not path.is_file():
                raise FileNotFoundError(f"hypothesis {h}: no checkpoint {path}")
        self._params = _clean_params(saved["params"] if same else (params or {}))
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
        return list(self._specs)

    def spec(self, hypothesis: str) -> Dict[str, Any]:
        """A hypothesis's model (``model`` or ``checkpoint``) and its overrides."""
        return dict(self._specs[hypothesis])

    def checkpoint_path(self, hypothesis: str) -> Optional[Path]:
        """The hypothesis's checkpoint file, or None for a stock model."""
        checkpoint = self._specs[hypothesis].get("checkpoint")
        if checkpoint is None:
            return None
        if self.models_dir is None:
            raise ValueError(f"hypothesis {hypothesis} is a checkpoint: give models=<folder>")
        return self.models_dir / checkpoint

    def settings(self, hypothesis: str, params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Base settings with ``hypothesis``'s overrides applied."""
        return {**(self._params if params is None else params), **overrides(self._specs[hypothesis])}

    def _checkpoint_sha(self, hypothesis: str) -> Optional[str]:
        path = self.checkpoint_path(hypothesis)
        if path is None:
            return None
        if str(path) not in self._sha:
            self._sha[str(path)] = sha256(path)
        return self._sha[str(path)]

    def _record(self) -> Dict[str, Any]:
        """What identifies the labels (compared with the saved record on resume)."""
        extraction = dict(self.root.attrs)["pymmm"].get("provenance") or {}
        shas = {h: self._checkpoint_sha(h) for h in self.hypotheses}
        return plain_json({
            "channel": self.channel,
            "params": self._params,
            "hypotheses": self.hypotheses,
            "specs": self._specs,
            "checkpoint_sha256": {h: s for h, s in shas.items() if s is not None},
            "settings": {h: self.settings(h) for h in self.hypotheses},
            "model_pixel_size": self.model_pixel_size,
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
        self._params = _clean_params({k: v for k, v in params.items() if k in DEFAULT_PARAMS})
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

    def _source(self, hypothesis: str, data: Any = None) -> "_ModelSource":
        path = self.checkpoint_path(hypothesis)
        return _ModelSource(
            hypothesis=hypothesis,
            model=self._specs[hypothesis].get("model"),
            checkpoint=None if path is None else str(path),
            sha256=self._checkpoint_sha(hypothesis),
            data=data,
        )

    def _get_model(self, hypothesis: str) -> Any:
        """The hypothesis's model, loaded once in this process."""
        return _model_for(self._source(hypothesis), self.gpu, self._loader)

    def segment_stack(
        self,
        stack: np.ndarray,
        bounds: Tuple[float, float],
        hypothesis: Optional[str] = None,
        params: Optional[Mapping[str, Any]] = None,
        batch_size: int = 64,
    ) -> Tuple[np.ndarray, List[int]]:
        """Segment a ``(T, Y, X)`` stack of one trench under one hypothesis (the base if
        ``None``), in this process: the labels ordered from the closed end and the
        frames Omnipose failed on (see :func:`segment_with`)."""
        hypothesis = hypothesis or self.hypotheses[0]
        return segment_with(self._get_model(hypothesis), self.settings(hypothesis, params),
                            stack, bounds, self.pixel_size, batch_size, self.model_pixel_size)

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
                    f"model and settings; each trench's labels/{self.name}.<hypothesis>"
                ),
                "schema_version": SCHEMA_VERSION,
                "provenance": plain_json(provenance()),
                "made_by": "pymmm.steps.segmentation.Segmenter",
                "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            })

    def _recipe(self, runner: Runner, threads: Optional[int]) -> Tuple["_Recipe", Dict[str, Any]]:
        """What every task needs, and the checkpoints' bytes (sent once per Dask worker,
        which may be on another machine; pool workers read the files)."""
        weights = {}
        if runner.kind == "dask":
            for h in self.hypotheses:
                path = self.checkpoint_path(h)
                if path is not None:
                    weights[h] = runner.share(np.fromfile(path, dtype=np.uint8))
        recipe = _Recipe(
            sources=tuple(self._source(h) for h in self.hypotheses),
            settings={h: self.settings(h) for h in self.hypotheses},
            pixel_size=float(self.pixel_size),
            model_pixel_size=self.model_pixel_size,
            gpu=self.gpu,
            threads=threads,
            digest=code_digest(),
            loader=self._loader,
        )
        return recipe, weights

    def _read(self, trench_id: int) -> Tuple[int, np.ndarray, Tuple[float, float]]:
        stack = self.read(trench_id)
        return trench_id, stack, self.bounds(trench_id, stack)

    def _write(self, root: zarr.Group, trench_id: int, bounds: Tuple[float, float],
               packed: Mapping[str, Any], failed: Mapping[str, List[int]],
               made_on: Mapping[str, str]) -> None:
        write_segmentation(root, trench_id, self.name, {h: _unpack(p) for h, p in packed.items()},
                           attrs={"long_name": LONG_NAME, "channel": self.channel,
                                  "lower": bounds[0], "upper": bounds[1],
                                  "failed_frames": dict(failed), "made_on": dict(made_on)})

    def segment(
        self,
        trenches: Optional[Sequence[int]] = None,
        resume: bool = True,
        batch_size: int = 64,
        workers: int = 1,
        threads: int = 1,
        show_progress: bool = True,
    ) -> None:
        """Segment the trenches under every hypothesis and write their label images.

        Each trench is finished once its label images are all written, so an
        interrupted run resumes with the trenches not finished.

        Trenches go where :class:`~pymmm.parallel.Runner` sends tasks: to the active
        Dask client's workers when there is one (on this machine or others with the
        same environment and PyMMM), else to ``workers`` local processes, else (1) to
        this process. Each task gets a trench's frames and returns its labels; this
        process alone writes the experiment group. A worker loads each model once.
        Most of each trench's time is Omnipose building masks on the CPU, which runs
        no faster with more threads, so workers use ``threads`` each and more
        workers segment more trenches at once.

        Parameters
        ----------
        trenches : sequence of int | None
            Trench ids to segment (a few, to check the settings before the rest);
            all if ``None``.
        resume : bool
            Keep the trenches already finished with the same channel, hypotheses,
            settings, checkpoints and extraction. ``False`` re-segments ``trenches``.
            Results are never cleared otherwise: if the settings changed, every
            trench that has results must be re-segmented (``segment(resume=False)``),
            or this raises.
        batch_size : int
            Frames per Omnipose GPU batch.
        workers : int
            Local worker processes when there is no Dask client (1: this process).
        threads : int
            CPU threads of each worker (this process, alone, uses them all).
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
        runner = Runner(workers=workers)
        pool = runner.kind == "pool"
        reader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pymmm-read")
        writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pymmm-write")
        writes: List[Any] = []
        try:
            with _environment(_WORKER_THREADS if pool else {}), runner, \
                    _dask_config(_LARGE_TASK if runner.kind == "dask" else None):
                recipe, weights = self._recipe(runner, None if runner.kind == "serial" else threads)
                queue = iter(todo)

                def _read_next():
                    trench_id = next(queue, None)
                    return None if trench_id is None else reader.submit(self._read, trench_id)

                pending = _read_next()
                running: Dict[Any, Tuple[int, Tuple[float, float]]] = {}
                while pending is not None or running:
                    # at most two trenches per worker thread: one running, one waiting
                    while pending is not None and len(running) < 2 * runner.slots:
                        trench_id, stack, bounds = pending.result()
                        pending = _read_next()
                        future = runner.submit(_segment_task, recipe, weights, stack, bounds, batch_size)
                        running[future] = (trench_id, bounds)
                    # the finished ones first, so a failure does not drop labels already made
                    for future in sorted(runner.wait_first(list(running)), key=_failed):
                        trench_id, bounds = running.pop(future)
                        packed, failed, made_on = future.result()
                        if failed:
                            failed_frames[trench_id] = failed
                        writes.append(writer.submit(self._write, root, trench_id, bounds, packed,
                                                    failed, made_on))
                        progress.update()
                    for write in [w for w in writes if w.done()]:
                        writes.remove(write)
                        write.result()
                for write in writes:
                    write.result()
        finally:
            progress.close()
            reader.shutdown(wait=False, cancel_futures=True)
            writer.shutdown(wait=True)  # the labels already made are kept

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
        sliders set the base settings; a hypothesis may override some of them
        (shown under it). **Apply** keeps the settings for :meth:`segment`.
        """
        from pymmm.widgets.segmentation import segmentation_widget

        segmentation_widget(self, batch_size)

    def view(self):
        """Browse the written labels as outlines over the channel: any hypothesis,
        trench and frame (unfinished trenches show no outlines)."""
        from pymmm.widgets.segmentation import label_viewer

        return label_viewer(self)

    def __repr__(self) -> str:
        models = sorted({s.get("model") or "checkpoints" for s in self._specs.values()})
        return (
            f"Segmenter({self.store_path.name}, labels/{self.name}.<{'|'.join(self.hypotheses)}>, "
            f"channel={self.channel!r}, model={'+'.join(models)!r}, "
            + (f"at {self.model_pixel_size:g} µm/px (the images' {self.pixel_size:g}), "
               if self.model_pixel_size is not None else "")
            + f"{len(self.finished())}/{len(self.trench_ids)} trenches finished)"
        )


# ----------------------------------------------------------------------
# Tasks: one trench under every hypothesis, wherever the Runner runs it
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class _ModelSource:
    """A hypothesis's model: a stock model by name, or a checkpoint (its file on the
    machine that made the task, its sha256, and ``data``: its bytes, shared with the
    Runner, for workers on other machines)."""

    hypothesis: str
    model: Optional[str]
    checkpoint: Optional[str]
    sha256: Optional[str]
    data: Any = None


@dataclass(frozen=True)
class _Recipe:
    """What every task of a run needs besides the trench."""

    sources: Tuple[_ModelSource, ...]
    settings: Dict[str, Dict[str, Any]]
    pixel_size: float
    model_pixel_size: Optional[float]
    gpu: bool
    threads: Optional[int]
    digest: str
    loader: Optional[str] = None


_MODELS: Dict[Tuple[str, bool], Any] = {}  # in each process: the models it loaded
_DIGEST: Optional[str] = None


def code_digest() -> str:
    """This process's segmentation code and Omnipose version, as a short hash: a task
    made by another process (another machine's notebook) must find the same."""
    global _DIGEST
    if _DIGEST is None:
        digest = hashlib.sha256()
        here = Path(__file__).resolve().parent
        for name in ("segmentation.py", "omnipose.py"):
            digest.update((here / name).read_bytes())
        try:
            digest.update(metadata.version("omnipose").encode())
        except metadata.PackageNotFoundError:
            digest.update(b"no omnipose")
        _DIGEST = digest.hexdigest()[:16]
    return _DIGEST


def _load_model(source: _ModelSource, path: Optional[str], gpu: bool) -> Any:
    from pymmm.steps.omnipose import load_model

    with _quiet_omnipose():
        if path is not None:
            return load_model(checkpoint=path, gpu=gpu)
        return load_model(model=source.model, gpu=gpu)


def _model_for(source: _ModelSource, gpu: bool, loader: Optional[str] = None) -> Any:
    """The source's model, loaded once per process. A checkpoint sent as bytes is
    written once to this machine's temporary folder, under its own file name."""
    key = (source.sha256 or f"model:{source.model}", gpu)
    if key not in _MODELS:
        path = source.checkpoint
        if source.data is not None:
            path = _materialise(source)
        load = _load_model
        if loader is not None:
            import importlib

            module, _, function = loader.partition(":")
            load = getattr(importlib.import_module(module), function)
        _MODELS[key] = load(source, path, gpu)
    return _MODELS[key]


def _materialise(source: _ModelSource) -> str:
    folder = Path(tempfile.gettempdir()) / "pymmm_models" / str(source.sha256)
    path = folder / Path(str(source.checkpoint)).name
    if not path.is_file() or sha256(path) != source.sha256:
        folder.mkdir(parents=True, exist_ok=True)
        part = path.with_name(f"{path.name}.{os.getpid()}.part")
        np.asarray(shared(source.data), dtype=np.uint8).tofile(part)
        if sha256(part) != source.sha256:
            part.unlink()
            raise RuntimeError(f"checkpoint {source.checkpoint} arrived damaged on {socket.gethostname()}")
        part.replace(path)
    return str(path)


@contextmanager
def _limit_threads(threads: Optional[int]) -> Iterator[None]:
    """At most ``threads`` CPU threads in BLAS/OpenMP, torch and numba (None: no limit)."""
    if threads is None:
        yield
        return
    from pymmm.steps.omnipose import PARLAY_THREADS

    for var, value in PARLAY_THREADS.items():
        os.environ.setdefault(var, value)
    try:  # in the segment extra's environment (scikit-learn brings it)
        from threadpoolctl import threadpool_limits
    except ImportError:
        threadpool_limits = None
    try:
        import torch
    except ImportError:
        torch = None
    previous = torch.get_num_threads() if torch is not None else None
    if torch is not None:
        torch.set_num_threads(threads)
    try:
        import numba

        numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
    except ImportError:
        pass
    try:
        with threadpool_limits(limits=threads) if threadpool_limits else _nothing():
            yield
    finally:
        if torch is not None:
            torch.set_num_threads(previous)


@contextmanager
def _nothing() -> Iterator[None]:
    yield


@contextmanager
def _dask_config(values: Optional[Mapping[str, Any]]) -> Iterator[None]:
    if not values:
        yield
        return
    import dask

    with dask.config.set(values):
        yield


def _failed(future: Any) -> bool:
    return future.exception() is not None


def _pack(labels: np.ndarray) -> Tuple[bytes, Tuple[int, ...]]:
    """``uint16`` labels, compressed for the way back (mostly background: a few % of their size)."""
    from numcodecs import Blosc

    labels = np.ascontiguousarray(labels, dtype=np.uint16)
    codec = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    return codec.encode(labels), labels.shape


def _unpack(packed: Tuple[bytes, Tuple[int, ...]]) -> np.ndarray:
    from numcodecs import Blosc

    data, shape = packed
    return np.frombuffer(Blosc().decode(data), dtype=np.uint16).reshape(shape)


def _segment_task(
    recipe: _Recipe,
    weights: Mapping[str, Any],
    stack: np.ndarray,
    bounds: Tuple[float, float],
    batch_size: int,
) -> Tuple[Dict[str, Tuple[bytes, Tuple[int, ...]]], Dict[str, List[int]], Dict[str, str]]:
    """Segment one trench's frames under every hypothesis: each one's packed labels,
    the frames Omnipose failed on, and the machine and device that made them (GPUs
    differ in the last bits, so masks may differ by a boundary pixel between
    machines). Store-free, so it runs on any machine."""
    if code_digest() != recipe.digest:
        raise RuntimeError(
            f"{socket.gethostname()} runs other segmentation code or another Omnipose "
            "than the notebook: copy this PyMMM there (rsync) and restart its workers"
        )
    packed: Dict[str, Tuple[bytes, Tuple[int, ...]]] = {}
    failed: Dict[str, List[int]] = {}
    model = None
    with _limit_threads(recipe.threads):
        for source in recipe.sources:
            if source.hypothesis in weights:
                source = _ModelSource(source.hypothesis, source.model, source.checkpoint,
                                      source.sha256, weights[source.hypothesis])
            model = _model_for(source, recipe.gpu, recipe.loader)
            labels, frames = segment_with(model, recipe.settings[source.hypothesis], stack,
                                          bounds, recipe.pixel_size, batch_size,
                                          recipe.model_pixel_size)
            packed[source.hypothesis] = _pack(labels)
            if frames:
                failed[source.hypothesis] = frames
    return packed, failed, {"host": socket.gethostname(), "device": _device_name(model)}


def _device_name(model: Any) -> str:
    device = str(getattr(model, "device", "cpu"))
    if device.startswith("cuda"):
        try:
            import torch

            return torch.cuda.get_device_name(torch.device(device))
        except Exception:  # noqa: BLE001 - a name for the record only
            pass
    return device
