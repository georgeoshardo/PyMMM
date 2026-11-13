"""Tracks: a tracking run's graph (GEFF) and masks beside each trench's image.

A tracker writes each trench's result with :func:`write_tracks`: its masks as the label
image ``tracks.<run>`` and its graph as ``tracks/<run>.geff`` (docs/layout.md);
:func:`tracklets_and_lineages` gives the GEFF ``track_id`` and ``lineage_id`` of a
graph's nodes. Writing needs the geff package, imported when it is used.

:func:`read_trench_geff` reads a graph as arrays; it takes the folder holding a
``tracking_graph.geff`` (the earlier per-trench collections, which tracker scoring
still compares against).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from pymmm.store.experiment_group import (
    calibration,
    listed_labels,
    trench_path,
    write_label_image,
)

NODE_PROPS = ("t", "y", "x", "seg_id", "track_id", "lineage_id")
GEFF_NAME = "tracking_graph.geff"


def read_trench_geff(trench_group: str | Path) -> dict[str, np.ndarray]:
    """Read a trench's graph (``<trench_group>/tracking_graph.geff``) as arrays.

    Faster than a graph library for summaries and plots. Works on curated copies
    too: edited graphs may have gaps in their node ids, so ``edges`` are mapped to
    positions in the node arrays.

    Returns
    -------
    dict
        ``node_id`` and the :data:`~pymmm.store.geff.NODE_PROPS` per node,
        ``edges`` (``(n_edges, 2)`` parent, child positions) and ``score`` per edge
        (when present).
    """
    geff_path = Path(trench_group) / GEFF_NAME
    graph = zarr.open_group(str(geff_path), mode="r")
    node_ids = np.asarray(graph["nodes/ids"][:])
    order = np.argsort(node_ids)
    out: dict[str, np.ndarray] = {"node_id": node_ids[order]}
    props = graph["nodes/props"]
    for key in NODE_PROPS:
        if key in props:
            out[key] = np.asarray(props[f"{key}/values"][:])[order]
    edge_ids = np.asarray(graph["edges/ids"][:]).reshape(-1, 2)
    out["edges"] = np.searchsorted(out["node_id"], edge_ids)
    if "edges/props/score" in graph:
        out["score"] = np.asarray(graph["edges/props/score/values"][:])
    return out


def tracklets_and_lineages(n_nodes: int, times: np.ndarray, edges: np.ndarray):
    """GEFF tracklet and lineage ids for a graph of ``n_nodes`` nodes.

    A node continues its parent's tracklet when it is the parent's only child
    and the parent is its only parent; lineages are the connected parts. Both are
    numbered from 1, in time order: readers such as mm-track take 0 for no track.
    """
    children: dict[int, list[int]] = {}
    parents: dict[int, list[int]] = {}
    for parent, child in np.asarray(edges, dtype=np.int64).reshape(-1, 2):
        children.setdefault(int(parent), []).append(int(child))
        parents.setdefault(int(child), []).append(int(parent))
    track_id = np.full(n_nodes, -1, dtype=np.int64)
    next_track = 1
    for node in np.argsort(times, kind="stable"):
        ps = parents.get(int(node), [])
        if len(ps) == 1 and len(children.get(ps[0], [])) == 1:
            track_id[node] = track_id[ps[0]]
        else:
            track_id[node] = next_track
            next_track += 1
    lineage_id = np.full(n_nodes, -1, dtype=np.int64)
    next_lineage = 1
    for start in range(n_nodes):
        if lineage_id[start] >= 0:
            continue
        stack = [start]
        lineage_id[start] = next_lineage
        while stack:
            node = stack.pop()
            for other in children.get(node, []) + parents.get(node, []):
                if lineage_id[other] < 0:
                    lineage_id[other] = next_lineage
                    stack.append(other)
        next_lineage += 1
    return track_id, lineage_id


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
    root : zarr.Group
        The experiment group, opened for writing.
    trench_id : int
        The trench.
    run : str
        The tracking run's name.
    nodes : mapping of str to ndarray
        Arrays of equal length keyed by :data:`NODE_PROPS`; ``t`` is the frame,
        ``y``/``x`` pixels in the trench, ``seg_id`` the node's value in ``masks``.
    edges : ndarray
        ``(n_edges, 2)`` parent, child positions into the node arrays. The GEFF's
        node ids are the positions + 1, so they start at 1: readers such as
        funtracks take node 0 for the masks' background.
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
    write_label_image(
        root,
        trench_id,
        label,
        np.asarray(masks, dtype=np.uint16),
        attrs={
            "tracks": f"../../tracks/{run}.geff",
            "description": "the masks the tracker chose; a mask's value is its GEFF node's seg_id",
        },
    )
    trench = root[trench_path(trench_id)]
    if "tracks" not in trench:
        trench.create_group(
            "tracks",
            attributes={
                "pymmm": {
                    "description": "tracking runs' GEFFs (regenerable); curated copies are in curation/"
                }
            },
        )
    pixel_um, frame_min = calibration(root)
    # Positions stay in pixels and t in frames, with no axis scale: readers such as
    # funtracks apply an axis scale to the positions (to world units) and would then
    # miss the masks. The trench image carries the calibration.
    metadata = GeffMetadata(
        directed=True,
        axes=[
            Axis(name="t", type="time", unit="frame"),
            Axis(name="y", type="space"),
            Axis(name="x", type="space"),
        ],
        node_props_metadata={},
        edge_props_metadata={},
        track_node_props={"tracklet": "track_id", "lineage": "lineage_id"},
        related_objects=[
            RelatedObject(
                type="labels", path=f"../../labels/{label}", node_prop="seg_id"
            ),
            RelatedObject(type="image", path="../.."),
        ],
        extra={
            **extra,
            "trench": int(trench_id),
            "pixel_size_um": pixel_um,
            "frame_interval_min": frame_min,
        },
    )
    geff_path = (
        Path(str(root.store.root)) / trench_path(trench_id) / "tracks" / f"{run}.geff"
    )
    n = len(nodes["t"])
    write_arrays(
        str(geff_path),
        np.arange(1, n + 1, dtype=np.int64),
        {
            key: {"values": np.asarray(nodes[key]), "missing": None}
            for key in NODE_PROPS
        },
        np.asarray(edges, dtype=np.int64).reshape(-1, 2) + 1,
        {"score": {"values": np.asarray(scores, dtype=np.float32), "missing": None}},
        metadata,
        zarr_format=3,
        overwrite=True,
    )
    return geff_path


def has_tracks(root: zarr.Group, trench_id: int, run: str) -> bool:
    return (
        f"{trench_path(trench_id)}/tracks/{run}.geff" in root
        and f"tracks.{run}" in listed_labels(root, trench_id)
    )
