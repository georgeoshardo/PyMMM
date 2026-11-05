"""Tracks – GEFF graphs as arrays, and GEFF tracklet and lineage ids.

Trackers write each trench's graph beside its image in the experiment group
(:func:`pymmm.trench_images.write_tracks`); :func:`tracklets_and_lineages` gives the
GEFF ``track_id`` and ``lineage_id`` of a graph's nodes.

:func:`read_trench_geff` reads the curated trenches of the earlier per-trench
collections (``<collection>/trench_NNNN/tracking_graph.geff``, beside its
``segmentation``), which tracker scoring still compares against.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import zarr

from pymmm.trench_images import NODE_PROPS

GEFF_NAME = "tracking_graph.geff"


def read_trench_geff(trench_group: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Read a trench's graph (``<trench_group>/tracking_graph.geff``) as arrays.

    Faster than a graph library for summaries and plots. Works on curated copies
    too: edited graphs may have gaps in their node ids, so ``edges`` are mapped to
    positions in the node arrays.

    Returns
    -------
    dict
        ``node_id`` and the :data:`~pymmm.trench_images.NODE_PROPS` per node,
        ``edges`` (``(n_edges, 2)`` parent, child positions) and ``score`` per edge
        (when present).
    """
    geff_path = Path(trench_group) / GEFF_NAME
    graph = zarr.open_group(str(geff_path), mode="r")
    node_ids = np.asarray(graph["nodes/ids"][:])
    order = np.argsort(node_ids)
    out: Dict[str, np.ndarray] = {"node_id": node_ids[order]}
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
    children: Dict[int, List[int]] = {}
    parents: Dict[int, List[int]] = {}
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
