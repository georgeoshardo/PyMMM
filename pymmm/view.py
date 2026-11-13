"""Open an experiment group in napari, every trench on one slider.

One image layer per channel and one labels layer per label image: each hypothesis of
each recorded segmentation (``<segmentation>.<hypothesis>``) and each tracking run's
masks (``tracks.<run>``). Every layer is a lazy stack ``(trench, t, y, x)`` of the
trench images, so the trench slider browses the experiment and all layers line up.
Drop ``<experiment>.zarr`` onto napari, or::

    python -m pymmm.view <experiment>.zarr

A trench that has no such label image yet shows nothing in that layer.

PyMMM is a napari plugin (``napari.yaml``) with one reader, :func:`napari_get_reader`.
It opens experiment groups only. napari-ome-zarr and napari's own reader also offer to
open any ``.zarr`` folder, so napari asks which reader to use; choose PyMMM.

One trench on its own is an OME-Zarr image with its label images, which
napari-ome-zarr opens as it is, in µm and minutes::

    napari --plugin napari-ome-zarr <experiment>.zarr/trench_0007
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import zarr

from pymmm.plotting import contrast_limits
from pymmm.store.experiment_group import (
    image,
    is_experiment,
    label_name,
    listed_labels,
    open_experiment,
    records,
    trench_ids,
    trench_path,
)

AXES = ("trench", "t", "y", "x")
# napari's layer data tuple: (data, keyword arguments of add_<type>, layer type)
LayerData = Tuple[Any, Dict[str, Any], str]


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
    return np.asarray(zarr.open_array(array_path, mode="r")[index]).astype(
        dtype, copy=False
    )


def _stack(
    path: Path,
    ids: Sequence[int],
    key: str,
    index: Tuple,
    shape: Tuple[int, ...],
    dtype: np.dtype,
    present: Optional[Set[int]] = None,
):
    """Lazy ``(trench, *shape)``: ``trench_NNNN/<key>[index]`` of each trench, read only
    when napari shows it; zeros for trenches not in ``present`` (``None``: all)."""
    import dask
    import dask.array as da

    parts = [
        da.from_delayed(
            dask.delayed(_read)(str(path / trench_path(t) / key), index, dtype),
            shape=shape,
            dtype=dtype,
        )
        if present is None or t in present
        else da.zeros(shape, dtype=dtype)
        for t in ids
    ]
    return da.stack(parts)


def _colormap(name: str, hex_color: str):
    from napari.utils.colormaps import Colormap

    rgb = [int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    return Colormap([[0, 0, 0, 1], rgb + [1]], name=name)


def experiment_layers(path: Union[str, Path]) -> List[LayerData]:
    """An experiment group's napari layers, as ``(data, kwargs, layer type)``.

    One image per channel, coloured and shown as the first trench's OME metadata says,
    then one labels layer per label image that any trench has: the bases of the
    segmentations shown, the other hypotheses and the tracks hidden. Every layer's
    ``metadata["pymmm"]`` holds the trench ids, their FOVs and the frames, for the
    caption (:func:`show_position`).
    """
    path = Path(path)
    root = open_experiment(path)
    ids = trench_ids(root)
    channels = [str(c) for c in root["C"][:]]
    first = image(root, ids[0])
    n_t, _, n_y, n_x = first.shape
    common = {
        "axis_labels": AXES,
        "metadata": {
            "pymmm": {
                "experiment": str(path),
                "trenches": ids,
                "fovs": [str(f) for f in root["trenches/fov_name"][:]],
                "frames": [int(t) for t in root["T"][:]],
            }
        },
    }

    # Channel colours, visibility and contrast from the first trench (its OME metadata,
    # and its middle frame), so napari doesn't scan the experiment.
    omero = {
        str(c.get("label")): c
        for c in (dict(root[trench_path(ids[0])].attrs).get("ome") or {})
        .get("omero", {})
        .get("channels", [])
    }
    sample = np.asarray(first[n_t // 2])
    layers: List[LayerData] = []
    for c, channel in enumerate(channels):
        lo, hi = contrast_limits(sample[c])
        shown = omero.get(channel, {})
        layers.append(
            (
                _stack(path, ids, "0", (slice(None), c), (n_t, n_y, n_x), first.dtype),
                {
                    "name": channel,
                    "colormap": _colormap(channel, str(shown.get("color") or "FFFFFF")),
                    "contrast_limits": (lo, hi if hi > lo else lo + 1),
                    "visible": bool(shown.get("active", True)),
                    "blending": "additive",
                    **common,
                },
                "image",
            )
        )

    listed: Dict[int, Set[str]] = {t: set(listed_labels(root, t)) for t in ids}
    bases = {
        label_name(s, str(r["hypotheses"][0]))
        for s, r in records(root, "segmentations").items()
        if r.get("hypotheses")
    }
    for name in label_images(root):
        present = {t for t in ids if name in listed[t]}
        if not present:
            continue
        dtype = root[f"{trench_path(min(present))}/labels/{name}/0"].dtype
        layers.append(
            (
                _stack(
                    path, ids, f"labels/{name}/0", (), (n_t, n_y, n_x), dtype, present
                ),
                {"name": name, "opacity": 0.5, "visible": name in bases, **common},
                "labels",
            )
        )
    return layers


def _text_overlay(viewer):
    canvas = getattr(
        viewer, "canvas", None
    )  # napari >= 0.8; before, viewer.text_overlay
    return canvas.overlays.text if canvas is not None else viewer.text_overlay


def show_position(viewer) -> None:
    """Caption the viewer's canvas with the trench (and its FOV) and the frame the
    sliders are on, read from the top layer that came from an experiment group, and
    name the sliders (napari < 0.8 doesn't take the layers' axis labels)."""
    overlay = _text_overlay(viewer)

    def update(_event=None) -> None:
        where = next(
            (
                layer.metadata["pymmm"]
                for layer in reversed(viewer.layers)
                if "pymmm" in layer.metadata
            ),
            None,
        )
        if where is None or viewer.dims.ndim < len(AXES):
            return
        labels = list(viewer.dims.axis_labels)
        if all(label.lstrip("-").isdigit() for label in labels[-len(AXES) :]):
            viewer.dims.axis_labels = tuple(labels[: -len(AXES)] + list(AXES))
        trench, t = viewer.dims.current_step[-len(AXES) :][:2]
        if trench < len(where["trenches"]) and t < len(where["frames"]):
            overlay.text = (
                f"trench {where['trenches'][trench]} ({where['fovs'][trench]}), "
                f"frame {where['frames'][t]}"
            )
            overlay.visible = True

    viewer.dims.events.current_step.connect(update)
    viewer.layers.events.inserted.connect(update)
    update()


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
    import napari

    path = Path(path)
    layers = experiment_layers(path)
    if viewer is None:
        viewer = napari.Viewer(title=path.name)
    for data, kwargs, kind in layers:
        getattr(viewer, f"add_{kind}")(data, **kwargs)
    show_position(viewer)
    return viewer


def napari_get_reader(path):
    """napari's reader hook (``napari.yaml``): :func:`read_experiment` when ``path`` is
    one experiment group, else ``None``, so napari offers its other readers."""
    paths = [path] if isinstance(path, (str, Path)) else list(path)
    if len(paths) == 1 and is_experiment(paths[0]):
        return read_experiment
    return None


def read_experiment(path) -> List[LayerData]:
    """An experiment group's layers (:func:`experiment_layers`), for napari to add, and
    the trench and frame caption on napari's current viewer."""
    import napari

    viewer = napari.current_viewer()
    if viewer is not None:
        show_position(viewer)
    return experiment_layers(path if isinstance(path, (str, Path)) else path[0])


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
    if not is_experiment(args.store):
        parser.error(f"not a PyMMM experiment group: {args.store}")

    import napari

    view_experiment(args.store)
    napari.run()


if __name__ == "__main__":
    main()
