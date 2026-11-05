"""Open an experiment group in napari, every trench on one slider.

One image layer per channel and one labels layer per label image: each hypothesis of
each recorded segmentation (``<segmentation>.<hypothesis>``) and each tracking run's
masks (``tracks.<run>``). Every layer is a lazy stack ``(trench, t, y, x)`` of the
trench images, so the trench slider browses the experiment and all layers line up::

    python -m pymmm.view <experiment>.zarr

A trench that has no such label image yet shows nothing in that layer.

One trench on its own is an OME-Zarr image with its label images, which
napari-ome-zarr opens as it is, in µm and minutes::

    napari --plugin napari-ome-zarr <experiment>.zarr/trench_0007
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import zarr

from pymmm.plotting import contrast_limits
from pymmm.store.experiment_group import (
    image,
    label_name,
    listed_labels,
    open_experiment,
    records,
    trench_ids,
    trench_path,
)


def label_images(root: zarr.Group) -> List[str]:
    """The label image names the experiment records: each segmentation's hypotheses
    (the base first), then each tracking run's ``tracks.<run>``."""
    names = [
        label_name(segmentation, str(hypothesis))
        for segmentation, record in records(root, "segmentations").items()
        for hypothesis in record["hypotheses"]
    ]
    return names + [f"tracks.{run}" for run in records(root, "tracks")]


def _read(array_path: str, index: Tuple, dtype: np.dtype) -> np.ndarray:
    return np.asarray(zarr.open_array(array_path, mode="r")[index]).astype(dtype, copy=False)


def _stack(
    path: Path, ids: Sequence[int], key: str, index: Tuple, shape: Tuple[int, ...],
    dtype: np.dtype, present: Optional[Set[int]] = None,
):
    """Lazy ``(trench, *shape)``: ``trench_NNNN/<key>[index]`` of each trench, read only
    when napari shows it; zeros for trenches not in ``present`` (``None``: all)."""
    import dask
    import dask.array as da

    parts = [
        da.from_delayed(
            dask.delayed(_read)(str(path / trench_path(t) / key), index, dtype),
            shape=shape, dtype=dtype,
        )
        if present is None or t in present else da.zeros(shape, dtype=dtype)
        for t in ids
    ]
    return da.stack(parts)


def _colormap(name: str, hex_color: str):
    from napari.utils.colormaps import Colormap

    rgb = [int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return Colormap([[0, 0, 0, 1], rgb + [1]], name=name)


def view_experiment(path: Union[str, Path], viewer=None):
    """Add an experiment group's channels and label images to a napari viewer.

    Parameters
    ----------
    path : str | Path
        Experiment group written by :class:`~pymmm.steps.extraction.Extractor`.
    viewer : napari.Viewer | None
        Viewer to add layers to; a new one is created if ``None``.

    Returns
    -------
    napari.Viewer
    """
    import dask.array as da
    import napari

    path = Path(path)
    root = open_experiment(path)
    ids = trench_ids(root)
    channels = [str(c) for c in root["C"][:]]
    first = image(root, ids[0])
    n_t, _, n_y, n_x = first.shape

    # Channel colours, visibility and contrast from the first trench (its OME metadata,
    # and its middle frame), so napari doesn't scan the experiment.
    omero = {str(c.get("label")): c for c in
             (dict(root[trench_path(ids[0])].attrs).get("ome") or {}).get("omero", {}).get("channels", [])}
    sample = np.asarray(first[n_t // 2])
    limits = [contrast_limits(sample[c]) for c in range(len(channels))]
    pixels = da.stack(
        [_stack(path, ids, "0", (slice(None), c), (n_t, n_y, n_x), first.dtype)
         for c in range(len(channels))],
        axis=2,
    )

    if viewer is None:
        viewer = napari.Viewer(title=path.name)
    viewer.add_image(
        pixels,
        channel_axis=2,
        name=channels,
        colormap=[_colormap(c, str(omero.get(c, {}).get("color") or "FFFFFF")) for c in channels],
        contrast_limits=[(lo, hi if hi > lo else lo + 1) for lo, hi in limits],
        visible=[bool(omero.get(c, {}).get("active", True)) for c in channels],
        blending="additive",
    )

    listed: Dict[int, Set[str]] = {t: set(listed_labels(root, t)) for t in ids}
    bases = {label_name(s, str(r["hypotheses"][0]))
             for s, r in records(root, "segmentations").items() if r.get("hypotheses")}
    for name in label_images(root):
        present = {t for t in ids if name in listed[t]}
        if not present:
            continue
        dtype = root[f"{trench_path(min(present))}/labels/{name}/0"].dtype
        viewer.add_labels(
            _stack(path, ids, f"labels/{name}/0", (), (n_t, n_y, n_x), dtype, present),
            name=name,
            opacity=0.5,
            visible=name in bases,
        )

    viewer.dims.axis_labels = ("trench", "t", "y", "x")
    fov_names = [str(f) for f in root["trenches/fov_name"][:]]
    frames = root["T"][:]
    text_overlay = viewer.canvas.overlays.text

    def _show_position(_event=None) -> None:
        trench, t = viewer.dims.current_step[:2]
        text_overlay.text = f"trench {ids[trench]} ({fov_names[trench]}), frame {frames[t]}"

    viewer.dims.events.current_step.connect(_show_position)
    text_overlay.visible = True
    _show_position()
    return viewer


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pymmm.view",
        description=(
            "Open an experiment group in napari, every trench on one slider: one image "
            "layer per channel, one labels layer per segmentation hypothesis and tracking run."
        ),
    )
    parser.add_argument("store", type=Path, help="experiment group (<experiment>.zarr)")
    args = parser.parse_args(argv)
    if not (args.store / "zarr.json").exists():
        parser.error(f"not a zarr group: {args.store}")

    import napari

    view_experiment(args.store)
    napari.run()


if __name__ == "__main__":
    main()
