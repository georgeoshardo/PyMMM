"""Omnipose glue for the Segmenter: loading a model, and building masks by affinity.

A model is either a stock Omnipose model, by name (``bact_fluor_omni``), or a
fine-tuned checkpoint file (one channel, two classes, 2D, as Omnipose's training
saves them).

:func:`affinity_masks` builds one frame's masks from the network's flow and distance
maps the way ``OmniModel.eval(affinity_seg=True)`` does in 2D, with the affinity
angle exposed. The cell pixels come from a hysteresis threshold on the distance map,
at (``mask_threshold - 1``, ``mask_threshold``). Each pixel then follows the flow for
half the mean cell diameter, neighbours are linked when their movements are within
``angle`` degrees, and each connected group is a cell. Omnipose does not expose the
angle (its link function defaults to 60°), so :func:`affinity_masks` swaps in a copy of
that function with the angle bound, for the duration of the call.
"""

from __future__ import annotations

import contextlib
import functools
import os
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import numpy as np

# Omnipose imports the ``dbscan`` package, whose parlay scheduler threads spin at full
# CPU even when idle (about 9 cores in an idle Jupyter kernel). One thread costs
# nothing measurable in segmentation speed. Read when dbscan is first imported.
PARLAY_THREADS = {"PARLAY_NUM_THREADS": "1"}

# compute_masks' settings that OmniModel.eval passes in 2D (besides the thresholds)
EVAL_KW = dict(rescale_factor=1.0, flow_factor=5.0, max_size=None, interp=True, cluster=False,
               suppress=None, despur=False, dim=2, bd=None)


def load_model(
    model: Optional[str] = None,
    checkpoint: Optional[Union[str, Path]] = None,
    gpu: bool = True,
) -> Any:
    """An ``OmniModel``: the stock ``model`` by name, or a fine-tuned ``checkpoint`` file."""
    if (model is None) == (checkpoint is None):
        raise ValueError("give one of model (a stock Omnipose model) or checkpoint (a file)")
    for var, value in PARLAY_THREADS.items():
        os.environ.setdefault(var, value)
    from omnipose.models import OmniModel

    if checkpoint is not None:
        return OmniModel(gpu=gpu, pretrained_model=str(checkpoint), nchan=1, nclasses=2, dim=2)
    return OmniModel(gpu=gpu, model_type=model)


@contextlib.contextmanager
def _affinity_angle(angle: float) -> Iterator[None]:
    import omnipose.core.masks as masks

    original = masks._get_affinity_torch
    masks._get_affinity_torch = functools.partial(original, angle_cutoff=np.deg2rad(angle))
    try:
        yield
    finally:
        masks._get_affinity_torch = original


def affinity_masks(
    dP: np.ndarray,
    dist: np.ndarray,
    device: Any,
    *,
    mask_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    angle: float = 60.0,
    min_size: int = 15,
) -> np.ndarray:
    """One frame's ``int32`` masks from the network's flow ``(2, Y, X)`` and distance
    ``(Y, X)`` maps, by affinity (see the module docstring)."""
    import torch
    from omnipose.core import compute_masks
    from omnipose.core.diam import diameters
    from omnipose.transforms.imports import hysteresis_threshold

    dt = torch.as_tensor(dist, device=device)
    iscell_t = hysteresis_threshold(dt[None, None], mask_threshold - 1, mask_threshold)[0, 0]
    iscell = iscell_t.cpu().numpy()
    if not iscell.any():
        return np.zeros(dist.shape, np.int32)
    kw = dict(EVAL_KW, mask_threshold=mask_threshold, flow_threshold=flow_threshold,
              min_size=min_size, use_gpu=torch.device(device).type != "cpu", device=device,
              nclasses=3)
    niter = max(1, int(diameters(iscell, dist) / 2))
    with _affinity_angle(angle):
        out = compute_masks(dP, dist, iscell=iscell, niter=niter, affinity_seg=True, **kw)
    return np.asarray(out[0], np.int32)
