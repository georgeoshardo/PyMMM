"""Open a trench store in napari.

One image layer per channel and one labels layer per segmentation hypothesis
(``/segmentation/<name>``, ``Hypothesis`` S1…), all lazy (dask) over
``(Trench, T, Y, X)``::

    python -m pymmm.view experiment.trenches.zarr
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import xarray as xr
import zarr

from pymmm.plotting import contrast_limits


def segmentation_names(store_path: Union[str, Path]) -> List[str]:
    """Names of the ``/segmentation/<name>`` groups in a trench store."""
    store = zarr.open_group(str(store_path), mode="r", zarr_format=3)
    if "segmentation" not in store:
        return []
    return sorted(name for name, _ in store["segmentation"].groups())


def _channel_colors(store_path: Path, channels: List[str]) -> List[str]:
    """Display colours from ``/acquisition``, white where none is recorded."""
    try:
        acquisition = xr.open_zarr(store_path, group="acquisition", consolidated=False)
        colors = dict(zip(
            (str(c) for c in acquisition["C"].values),
            (str(c) for c in acquisition["color"].values),
        ))
    except (KeyError, FileNotFoundError):
        colors = {}
    return [colors.get(channel) or "#FFFFFF" for channel in channels]


def _colormap(name: str, hex_color: str):
    from napari.utils.colormaps import Colormap

    rgb = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    return Colormap([[0, 0, 0, 1], rgb + [1]], name=name)


def view_trench_store(store_path: Union[str, Path], viewer=None):
    """Add a trench store's channels and segmentations to a napari viewer.

    Parameters
    ----------
    store_path : str | Path
        Trench store written by :class:`~pymmm.extractor.Extractor`.
    viewer : napari.Viewer | None
        Viewer to add layers to; a new one is created if ``None``.

    Returns
    -------
    napari.Viewer
    """
    import napari

    store_path = Path(store_path)
    root = xr.open_zarr(store_path, consolidated=False)
    channels = [str(c) for c in root["C"].values]
    data = root["data"].transpose("Trench", "T", "C", "Y", "X").data

    # Contrast from one trench mid-experiment, so napari doesn't scan the store.
    sample = np.asarray(data[0, data.shape[1] // 2])
    limits = [contrast_limits(sample[c]) for c in range(len(channels))]
    limits = [(lo, hi if hi > lo else lo + 1) for lo, hi in limits]

    if viewer is None:
        viewer = napari.Viewer(title=store_path.name)
    viewer.add_image(
        data,
        channel_axis=2,
        name=channels,
        colormap=[
            _colormap(channel, color)
            for channel, color in zip(channels, _channel_colors(store_path, channels))
        ],
        contrast_limits=limits,
        blending="additive",
    )
    for name in segmentation_names(store_path):
        labels = xr.open_zarr(
            store_path, group=f"segmentation/{name}", consolidated=False,
        )["labels"]
        # One layer per hypothesis; only the first (the primary segmentation) shown.
        for k, hypothesis in enumerate(str(h) for h in labels["Hypothesis"].values):
            viewer.add_labels(
                labels.sel(Hypothesis=hypothesis).data,
                name=f"{name} {hypothesis}",
                opacity=0.5,
                visible=k == 0,
            )

    viewer.dims.axis_labels = ("Trench", "T", "Y", "X")
    trench_ids = root["Trench"].values
    fov_names = root["fov_name"].values
    frames = root["T"].values
    text_overlay = viewer.canvas.overlays.text

    def _show_position(_event=None) -> None:
        trench, t = viewer.dims.current_step[:2]
        text_overlay.text = (
            f"trench {trench_ids[trench]} ({fov_names[trench]}), frame {frames[t]}"
        )

    viewer.dims.events.current_step.connect(_show_position)
    text_overlay.visible = True
    _show_position()
    return viewer


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pymmm.view",
        description=(
            "Open a trench store in napari: one image layer per channel, "
            "one labels layer per segmentation."
        ),
    )
    parser.add_argument("store", type=Path, help="path to a .trenches.zarr store")
    args = parser.parse_args(argv)
    if not (args.store / "zarr.json").exists():
        parser.error(f"not a zarr store: {args.store}")

    import napari

    view_trench_store(args.store)
    napari.run()


if __name__ == "__main__":
    main()
