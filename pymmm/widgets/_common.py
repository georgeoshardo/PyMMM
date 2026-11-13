"""What the widgets share: their sliders, the registered mean images, ndv on the canvas
that works in Jupyter, an ndv viewer of an image with 0–1 overlays, and the layout.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from types import ModuleType
from typing import Any

import numpy as np

STYLE = {"description_width": "initial"}

NDV_CANVAS = "pygfx"
"""The canvas PyMMM's ndv viewers draw on. ndv draws on vispy whenever vispy is installed
(napari brings it), and ndv 0.5's Jupyter viewer then fails on every mouse and key event
(``KeyError: 'event_type'``): it reads ``event["event_type"]``, which vispy 0.17's
canvas no longer sends. pygfx's canvas (ndv's ``jup`` extra, so the notebook extra)
sends it."""


def pygfx_ndv() -> ModuleType:
    """ndv, set to draw new viewers on :data:`NDV_CANVAS` (ndv reads the setting for
    each viewer it makes).
    """
    import ndv

    os.environ["NDV_CANVAS_BACKEND"] = NDV_CANVAS
    return ndv


def imshow(data: Any, **kwargs: Any) -> Any:
    """``ndv.imshow(data, **kwargs)`` on :data:`NDV_CANVAS`: use it for every ndv viewer
    in a notebook.
    """
    return pygfx_ndv().imshow(data, **kwargs)


def slider(
    kind: str,
    description: str,
    value: float,
    low: float,
    high: float,
    step: float,
    **kwargs: Any,
) -> Any:
    """An ``"int"`` or ``"float"`` slider that reports only when released."""
    import ipywidgets as widgets

    cls = widgets.IntSlider if kind == "int" else widgets.FloatSlider
    return cls(
        value=value,
        min=low,
        max=high,
        step=step,
        description=description,
        continuous_update=False,
        style=STYLE,
        **kwargs,
    )


class MeanImages:
    """A registrator's detection images (the mean of registered frames) of each FOV,
    each computed once.
    """

    def __init__(self, registrator: Any, channel: str) -> None:
        self.registrator, self.channel = registrator, channel
        self._images: dict[tuple, np.ndarray] = {}

    def __call__(self, fov: str, mean_n_frames: int, mean_from: str) -> np.ndarray:
        key = (fov, mean_n_frames, mean_from)
        if key not in self._images:
            self._images[key] = self.registrator.get_registered_mean_of_timestack(
                fov=fov,
                channel=self.channel,
                mean_n_frames=mean_n_frames,
                mean_from=mean_from,
            )
        return self._images[key]


def mean_controls(params: dict[str, Any]) -> tuple[Any, Any]:
    """The detection image's controls: "mean of n frames" and "frames from", set to
    ``params`` (a detection's saved settings) or the defaults.
    """
    import ipywidgets as widgets

    from pymmm.steps.registration import DETECTION_MEAN

    n = slider(
        "int",
        "mean of n frames",
        params.get("mean_n_frames", DETECTION_MEAN["mean_n_frames"]),
        1,
        500,
        1,
    )
    where = widgets.Dropdown(
        options=[
            ("spread over the movie", "spread"),
            ("the first", "start"),
            ("the last", "end"),
        ],
        value=params.get("mean_from", DETECTION_MEAN["mean_from"]),
        description="frames from",
        style=STYLE,
    )
    return n, where


def overlay_luts(
    image_name: str, overlays: Sequence[tuple[str, str]]
) -> dict[int, dict]:
    """Ndv LUTs: channel 0 the image at PyMMM's percentile contrast, then one 0–1
    overlay per ``(name, colormap)``.
    """
    from pymmm.plotting import NDV_CLIMS

    luts = {0: {"name": image_name, "cmap": "gray", "clims": NDV_CLIMS}}
    for channel, (name, cmap) in enumerate(overlays, start=1):
        luts[channel] = {"name": name, "cmap": cmap, "clims": (0, 1)}
    return luts


def overlay_viewer(
    data: np.ndarray,
    channel_axis: int,
    image_name: str,
    overlays: Sequence[tuple[str, str]],
) -> Any:
    """An ndv viewer of ``data`` (updated in place; then emit
    ``viewer.data_wrapper.data_changed``) shown as a composite of its channels.
    """
    return pygfx_ndv().ArrayViewer(
        data,
        channel_axis=channel_axis,
        channel_mode="composite",
        luts=overlay_luts(image_name, overlays),
    )


def show(controls: list[Any], preview: Any, viewer: Any, output: Any) -> None:
    """Display the controls beside the preview's label, its viewer and its plots."""
    import ipywidgets as widgets
    from IPython import display

    display.display(
        widgets.HBox(
            [widgets.VBox(controls), widgets.VBox([preview, viewer.widget(), output])]
        )
    )
