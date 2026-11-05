"""What the widgets share: their sliders, the registered mean images, an ndv viewer of
an image with 0–1 overlays, and the layout."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

STYLE = {"description_width": "initial"}


def slider(kind: str, description: str, value: float, low: float, high: float,
           step: float, **kwargs: Any) -> Any:
    """An ``"int"`` or ``"float"`` slider that reports only when released."""
    import ipywidgets as widgets

    cls = widgets.IntSlider if kind == "int" else widgets.FloatSlider
    return cls(value=value, min=low, max=high, step=step, description=description,
               continuous_update=False, style=STYLE, **kwargs)


class MeanImages:
    """A registrator's registered mean image of each FOV, computed once."""

    def __init__(self, registrator: Any, channel: str) -> None:
        self.registrator, self.channel = registrator, channel
        self._images: Dict[str, np.ndarray] = {}

    def __call__(self, fov: str) -> np.ndarray:
        if fov not in self._images:
            self._images[fov] = self.registrator.get_registered_mean_of_timestack(
                fov=fov, channel=self.channel)
        return self._images[fov]


def overlay_luts(image_name: str, overlays: Sequence[Tuple[str, str]]) -> Dict[int, dict]:
    """ndv LUTs: channel 0 the image at PyMMM's percentile contrast, then one 0–1
    overlay per ``(name, colormap)``."""
    from pymmm.plotting import NDV_CLIMS

    luts = {0: {"name": image_name, "cmap": "gray", "clims": NDV_CLIMS}}
    for channel, (name, cmap) in enumerate(overlays, start=1):
        luts[channel] = {"name": name, "cmap": cmap, "clims": (0, 1)}
    return luts


def overlay_viewer(data: np.ndarray, channel_axis: int, image_name: str,
                   overlays: Sequence[Tuple[str, str]]) -> Any:
    """An ndv viewer of ``data`` (updated in place; then emit
    ``viewer.data_wrapper.data_changed``) shown as a composite of its channels."""
    import ndv

    return ndv.ArrayViewer(data, channel_axis=channel_axis, channel_mode="composite",
                           luts=overlay_luts(image_name, overlays))


def show(controls: List[Any], preview: Any, viewer: Any, output: Any) -> None:
    """Display the controls beside the preview's label, its viewer and its plots."""
    import ipywidgets as widgets
    from IPython import display

    display.display(widgets.HBox([
        widgets.VBox(controls), widgets.VBox([preview, viewer.widget(), output])]))
