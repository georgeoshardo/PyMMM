"""OME-Zarr 0.5 (NGFF) metadata: every ``ome`` attribute PyMMM writes is built here.

A trench image is ``(t, c, y, x)`` and its label images ``(t, y, x)``, each one
resolution level ``0`` scaled to minutes and micrometres (docs/layout.md).
"""

from __future__ import annotations

import copy
from typing import Any, Iterable, Sequence

import numpy as np

OME_VERSION = "0.5"


def _multiscale(name: str, axes: list[dict], scale: list[float]) -> dict[str, Any]:
    return {
        "name": name,
        "type": "none",
        "metadata": {"description": "one resolution level: nothing was downsampled"},
        "axes": axes,
        "datasets": [{"path": "0", "coordinateTransformations": [
            {"type": "scale", "scale": scale}]}],
    }


def image(
    name: str,
    *,
    channels: Sequence[str],
    colors: Sequence[str],
    active: str,
    dtype: np.dtype,
    frame_interval_ms: float,
    pixel_size_um: float,
) -> dict[str, Any]:
    """The ``ome`` attributes of a trench image, ``(t, c, y, x)``.

    Time is in minutes and space in micrometres, each only when the ND2 gave a
    positive interval or pixel size (else the axis has no unit and a scale of 1).
    ``omero`` shows each channel in its ``colors`` entry (``RRGGBB``), only ``active``
    switched on, over the dtype's range until :func:`with_windows` sets the windows.
    """
    time_axis = {"name": "t", "type": "time"}
    space = [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]
    scale = [1.0, 1.0, 1.0, 1.0]
    if np.isfinite(frame_interval_ms) and frame_interval_ms > 0:
        time_axis["unit"] = "minute"
        scale[0] = frame_interval_ms / 60_000.0
    if np.isfinite(pixel_size_um) and pixel_size_um > 0:
        for axis in space:
            axis["unit"] = "micrometer"
        scale[2:] = [pixel_size_um, pixel_size_um]
    top = float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 1.0
    omero_channels = [
        {"label": label, "color": color, "active": label == active,
         "window": {"min": 0.0, "max": top, "start": 0.0, "end": top}}
        for label, color in zip(channels, colors, strict=True)
    ]
    return {
        "version": OME_VERSION,
        "multiscales": [_multiscale(
            name, [time_axis, {"name": "c", "type": "channel"}, *space], scale)],
        "omero": {"channels": omero_channels, "rdefs": {"model": "color", "defaultT": 0}},
    }


def display_windows(pixels: np.ndarray) -> list[tuple[float, float]]:
    """Each channel's display window for a ``(t, c, y, x)`` trench: its 0.5th and
    99.9th percentiles, from every tenth frame or so."""
    step = max(1, pixels.shape[0] // 26)
    return [
        tuple(float(v) for v in np.percentile(pixels[::step, c], [0.5, 99.9]))
        for c in range(pixels.shape[1])
    ]


def with_windows(ome: dict[str, Any], windows: Iterable[tuple[float, float]]) -> dict[str, Any]:
    """A copy of a trench image's ``ome`` attributes with each channel's window
    ``start`` and ``end`` set from ``windows``."""
    ome = copy.deepcopy(ome)
    for channel, (start, end) in zip(ome["omero"]["channels"], windows):
        channel["window"].update(start=start, end=end)
    return ome


def label_image(name: str, *, minutes_per_frame: float, um_per_pixel: float) -> dict[str, Any]:
    """The ``ome`` attributes of a label image, ``(t, y, x)``, in ``labels/`` of the
    trench image it labels."""
    axes = [{"name": "t", "type": "time", "unit": "minute"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"}]
    return {
        "version": OME_VERSION,
        "multiscales": [_multiscale(name, axes, [minutes_per_frame, um_per_pixel, um_per_pixel])],
        "image-label": {"source": {"image": "../../"}},
    }


def labels(names: Iterable[str]) -> dict[str, Any]:
    """The ``ome`` attributes of a trench's ``labels/`` group: its finished label images."""
    return {"version": OME_VERSION, "labels": list(names)}
