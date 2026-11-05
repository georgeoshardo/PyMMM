"""The per-trench OME-Zarr layout of an experiment, for the steps after extraction.

The experiment group (written by :class:`~pymmm.extractor.Extractor`) holds one
OME-Zarr image per trench. Segmentations and tracks are written beside each
trench's pixels::

    trench_0007/
        0                       (t, c, y, x) pixels
        labels/                 OME-Zarr labels group; ome.labels lists the finished ones
            <segmentation>      image-label (t, hypothesis, y, x), one shard with an inner
                                chunk per hypothesis; attrs pymmm.hypotheses names them,
                                the base first
            tracks.<run>        image-label (t, y, x): the masks a tracker chose; a mask's
                                value is its node's seg_id in tracks/<run>.geff
            curation.<run>      the curated masks
        tracks/<run>.geff       a tracker's GEFF; regenerable
        curation/<run>.geff     the curated copy of tracks/<run>.geff, made when the trench
                                is first edited

A label image counts once it is listed in ``labels/``'s ``ome.labels``: writers list
it after its pixels are in, so a reader never takes half a segmentation for one. The
experiment group's ``pymmm`` attributes record each segmentation's settings
(``segmentations``) and each tracking run's (``tracks``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import zarr

OME_VERSION = "0.5"
CHUNK_KEYS = {"name": "default", "separator": "."}
LABEL_COMPRESSORS = (zarr.codecs.BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle"),)
NODE_PROPS = ("t", "y", "x", "seg_id", "track_id", "lineage_id")

PathLike = Union[str, Path]


def trench_path(trench_id: int) -> str:
    """The trench's image group within the experiment group: ``trench_0007``."""
    return f"trench_{int(trench_id):04d}"


def open_experiment(path: PathLike, mode: str = "r") -> zarr.Group:
    """An experiment group written by the Extractor (``mode="r+"`` to add results)."""
    root = zarr.open_group(str(path), mode=mode, zarr_format=3)
    if (dict(root.attrs).get("pymmm") or {}).get("layout") != "experiment":
        raise ValueError(f"{path} is not a PyMMM experiment group (one OME-Zarr image per trench)")
    return root


def trench_ids(root: zarr.Group) -> list[int]:
    """The experiment's trench ids, in table order."""
    return [int(t) for t in root["trenches/trench"][:]]


def channel_index(root: zarr.Group, channel: str) -> int:
    return [str(c) for c in root["C"][:]].index(channel)


def calibration(root: zarr.Group) -> tuple[float, float]:
    """(µm per pixel, minutes per frame) of the trench images."""
    attrs = dict(root.attrs)["pymmm"]
    return float(attrs["pixel_size_um"]), float(attrs["frame_interval_ms"]) / 60_000.0


def image(root: zarr.Group, trench_id: int) -> zarr.Array:
    """A trench's (t, c, y, x) pixels."""
    return root[f"{trench_path(trench_id)}/0"]


def update_experiment(root: zarr.Group, section: str, name: str, record: Mapping[str, Any]) -> None:
    """Record ``record`` as ``attrs.pymmm[section][name]`` of the experiment group."""
    attrs = dict(dict(root.attrs)["pymmm"])
    attrs[section] = {**(attrs.get(section) or {}), name: dict(record)}
    root.attrs["pymmm"] = attrs


# ----------------------------------------------------------------------------
# label images
# ----------------------------------------------------------------------------
def listed_labels(root: zarr.Group, trench_id: int) -> list[str]:
    """The trench's finished label images (``labels/``'s ``ome.labels``)."""
    key = f"{trench_path(trench_id)}/labels"
    if key not in root:
        return []
    return list((dict(root[key].attrs).get("ome") or {}).get("labels") or [])


def _list_label(root: zarr.Group, trench_id: int, name: str) -> None:
    key = f"{trench_path(trench_id)}/labels"
    labels = root[key] if key in root else root.create_group(key)
    names = listed_labels(root, trench_id)
    if name not in names:
        labels.attrs["ome"] = {"version": OME_VERSION, "labels": [*names, name]}


def _multiscale(root: zarr.Group, name: str, hypothesis_axis: bool) -> dict:
    pixel_um, frame_min = calibration(root)
    axes = [{"name": "t", "type": "time", "unit": "minute"}]
    scale = [frame_min]
    if hypothesis_axis:
        # OME-Zarr 0.5 allows one axis of a custom type between time and space
        axes.append({"name": "hypothesis", "type": "hypothesis"})
        scale.append(1.0)
    axes += [{"name": "y", "type": "space", "unit": "micrometer"},
             {"name": "x", "type": "space", "unit": "micrometer"}]
    scale += [pixel_um, pixel_um]
    return {
        "name": name,
        "type": "none",
        "metadata": {"description": "one resolution level: nothing was downsampled"},
        "axes": axes,
        "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": scale}]}],
    }


def write_label_image(
    root: zarr.Group,
    trench_id: int,
    name: str,
    labels: np.ndarray,
    *,
    hypotheses: Optional[Sequence[str]] = None,
    attrs: Optional[Mapping[str, Any]] = None,
) -> zarr.Array:
    """Write (or replace) one label image of a trench and list it.

    Parameters
    ----------
    labels : ndarray
        ``(t, y, x)``, or ``(t, hypothesis, y, x)`` with ``hypotheses`` naming the
        hypotheses in order.
    attrs : mapping
        The label image's ``pymmm`` attributes.
    """
    labels = np.asarray(labels)
    if hypotheses is not None:
        if labels.ndim != 4 or labels.shape[1] != len(hypotheses):
            raise ValueError(f"{name}: labels {labels.shape} for {len(hypotheses)} hypotheses")
        chunks = (labels.shape[0], 1, *labels.shape[2:])
        shards: Optional[tuple[int, ...]] = labels.shape
        dims = ["t", "hypothesis", "y", "x"]
    else:
        if labels.ndim != 3:
            raise ValueError(f"{name}: labels {labels.shape} are not (t, y, x)")
        chunks, shards, dims = labels.shape, None, ["t", "y", "x"]
    pymmm = dict(attrs or {})
    if hypotheses is not None:
        pymmm["hypotheses"] = [str(h) for h in hypotheses]
    group = root.require_group(f"{trench_path(trench_id)}/labels/{name}")
    group.attrs.put({
        "ome": {
            "version": OME_VERSION,
            "multiscales": [_multiscale(root, name, hypotheses is not None)],
            "image-label": {"source": {"image": "../../"}},
        },
        "pymmm": pymmm,
    })
    array = group.create_array(
        "0", shape=labels.shape, dtype=labels.dtype, chunks=chunks, shards=shards,
        compressors=LABEL_COMPRESSORS, chunk_key_encoding=CHUNK_KEYS,
        dimension_names=dims, fill_value=0, overwrite=True,
    )
    array[...] = labels
    _list_label(root, trench_id, name)
    return array


def segmentation(root: zarr.Group, trench_id: int, name: str) -> tuple[list[str], zarr.Array]:
    """A finished segmentation of a trench: its hypotheses (the base first) and its
    ``(t, hypothesis, y, x)`` labels."""
    if name not in listed_labels(root, trench_id):
        raise KeyError(f"{trench_path(trench_id)} has no finished segmentation {name!r}")
    group = root[f"{trench_path(trench_id)}/labels/{name}"]
    return list(dict(group.attrs)["pymmm"]["hypotheses"]), group["0"]


# ----------------------------------------------------------------------------
# tracks
# ----------------------------------------------------------------------------
def write_tracks(
    root: zarr.Group,
    trench_id: int,
    run: str,
    nodes: Mapping[str, np.ndarray],
    edges: np.ndarray,
    scores: np.ndarray,
    masks: np.ndarray,
    *,
    extra: Mapping[str, Any],
) -> Path:
    """Write (or replace) a tracking run's result for one trench: its masks as the
    label image ``tracks.<run>`` and its graph as ``tracks/<run>.geff``.

    Parameters
    ----------
    nodes : mapping of str to ndarray
        Arrays of equal length keyed by :data:`NODE_PROPS`; ``t`` is the frame,
        ``y``/``x`` pixels in the trench, ``seg_id`` the node's value in ``masks``.
    edges : ndarray
        ``(n_edges, 2)`` parent, child positions into the node arrays.
    scores : ndarray
        Link score per edge.
    masks : ndarray
        ``(t, y, x)`` the masks the tracker chose.
    extra : mapping
        Provenance for the GEFF's ``extra`` metadata.

    Returns
    -------
    Path
        The GEFF group.
    """
    from geff.core_io._base_write import write_arrays
    from geff_spec import Axis, GeffMetadata, RelatedObject

    label = f"tracks.{run}"
    write_label_image(root, trench_id, label, np.asarray(masks, dtype=np.uint16), attrs={
        "tracks": f"../../tracks/{run}.geff",
        "description": "the masks the tracker chose; a mask's value is its GEFF node's seg_id",
    })
    trench = root[trench_path(trench_id)]
    if "tracks" not in trench:
        trench.create_group("tracks", attributes={"pymmm": {
            "description": "tracking runs' GEFFs (regenerable); curated copies are in curation/"}})
    pixel_um, frame_min = calibration(root)
    # Positions stay in pixels and t in frames, with no axis scale: readers such as
    # funtracks apply an axis scale to the positions (to world units) and would then
    # miss the masks. The trench image carries the calibration.
    metadata = GeffMetadata(
        directed=True,
        axes=[Axis(name="t", type="time", unit="frame"), Axis(name="y", type="space"),
              Axis(name="x", type="space")],
        node_props_metadata={},
        edge_props_metadata={},
        track_node_props={"tracklet": "track_id", "lineage": "lineage_id"},
        related_objects=[
            RelatedObject(type="labels", path=f"../../labels/{label}", node_prop="seg_id"),
            RelatedObject(type="image", path="../.."),
        ],
        extra={**extra, "trench": int(trench_id), "pixel_size_um": pixel_um,
               "frame_interval_min": frame_min},
    )
    geff_path = Path(str(root.store.root)) / trench_path(trench_id) / "tracks" / f"{run}.geff"
    n = len(nodes["t"])
    write_arrays(
        str(geff_path), np.arange(n, dtype=np.int64),
        {key: {"values": np.asarray(nodes[key]), "missing": None} for key in NODE_PROPS},
        np.asarray(edges, dtype=np.int64).reshape(-1, 2),
        {"score": {"values": np.asarray(scores, dtype=np.float32), "missing": None}},
        metadata, zarr_format=3, overwrite=True,
    )
    return geff_path


def has_tracks(root: zarr.Group, trench_id: int, run: str) -> bool:
    return (f"{trench_path(trench_id)}/tracks/{run}.geff" in root
            and f"tracks.{run}" in listed_labels(root, trench_id))
