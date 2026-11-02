"""Tracks – one GEFF per trench, from a tracker, and a curated copy of them.

The GEFF spec ties a graph's axes to one coordinate system and recommends keeping
its related segmentation and images as siblings of the graph, so each trench (its
own movie) gets its own group. A tracker writes them into the tracks store::

    <experiment>.tracks.zarr/
        trackastra_S1/                one group per tracker and hypothesis
            trench_0165/
                segmentation          (t, y, x) labels; nodes link by seg_id
                tracking_graph.geff   related_objects -> ../segmentation
            ...

That output is generated and may be regenerated at any time. Curation happens in
a copy, ``<experiment>.curation.zarr/trackastra_S1/...`` (same layout), made by
:func:`prepare_curation`, which never overwrites a trench that has been edited.
mm-track (georgeoshardo/motile_tracker) opens the curation store and saves its
edits there (in place, with an ``edit_history`` inside the trench's GEFF).

Positions are pixels in the trench crop and ``t`` is the frame (axis unit
``frame``); µm per pixel and minutes per frame are in the graph's ``extra``
metadata. The channels are not copied: ``extra`` names the trench store and the
trench, so a viewer can add them.
"""

from __future__ import annotations

import contextlib
import functools
import io
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
import xarray as xr
import zarr
from tqdm.auto import tqdm

from pymmm.provenance import SCHEMA_VERSION, provenance

NODE_PROPS = ("t", "y", "x", "seg_id", "track_id", "lineage_id")
GEFF_NAME = "tracking_graph.geff"
# mm-track keeps a dataset's edits in this database inside its GEFF group
EDIT_HISTORY = Path("edit_history") / "history.sqlite"


def group_name(trench_id: int) -> str:
    return f"trench_{int(trench_id):04d}"


def trench_of(group: Union[str, Path]) -> int:
    """``.../trench_0165`` -> 165."""
    return int(Path(group).name.removeprefix("trench_"))


def trench_groups(collection: Union[str, Path]) -> List[Path]:
    """The trench groups of a tracks or curation collection, in trench order."""
    collection = Path(collection)
    if not collection.is_dir():
        return []
    return sorted(p for p in collection.iterdir() if p.is_dir() and p.name.startswith("trench_"))


def is_edited(trench_group: Union[str, Path]) -> bool:
    """Whether mm-track has recorded edits for this trench."""
    return (Path(trench_group) / GEFF_NAME / EDIT_HISTORY).is_file()


def _ensure_group(path: Path, attrs: Optional[Dict[str, Any]] = None) -> zarr.Group:
    group = zarr.open_group(str(path), mode="a", zarr_format=3)
    if attrs:
        group.attrs.update(attrs)
    return group


def write_trench_geff(
    collection: Union[str, Path],
    trench_id: int,
    nodes: Mapping[str, np.ndarray],
    edges: np.ndarray,
    scores: np.ndarray,
    segmentation: np.ndarray,
    *,
    pixel_size: float,
    interval_min: float,
    extra: Mapping[str, Any],
) -> Path:
    """Write one trench's segmentation and tracking graph into a collection.

    Parameters
    ----------
    collection : str | Path
        The collection group (created if missing), e.g.
        ``<experiment>.tracks.zarr/trackastra_S1``.
    trench_id : int
        The trench's id in the trench store.
    nodes : mapping of str to ndarray
        Arrays of equal length keyed by :data:`NODE_PROPS`; ``t`` is the frame
        position in ``segmentation``, ``y``/``x`` pixels in the crop.
    edges : ndarray
        ``(n_edges, 2)`` parent, child positions into the node arrays.
    scores : ndarray
        Link score per edge.
    segmentation : ndarray
        ``(t, y, x)`` labels; a node's label in frame ``t`` is its ``seg_id``.
    pixel_size : float
        µm per pixel.
    interval_min : float
        Minutes per frame.
    extra : mapping
        Provenance for the graph's ``extra`` metadata.

    Returns
    -------
    Path
        The path of the GEFF group.
    """
    from geff.core_io._base_write import write_arrays
    from geff_spec import Axis, GeffMetadata, RelatedObject

    collection = Path(collection)
    _ensure_group(collection)
    group = _ensure_group(collection / group_name(trench_id))
    group.create_array(
        "segmentation", data=np.asarray(segmentation), chunks=segmentation.shape,
        dimension_names=["t", "y", "x"], overwrite=True,
    )
    n = len(nodes["t"])
    node_props = {key: {"values": np.asarray(nodes[key]), "missing": None} for key in NODE_PROPS}
    # Positions stay in pixels and t in frames, with no axis scale: readers such as
    # funtracks apply an axis scale to the positions (to world units) and would then
    # miss the segmentation. The calibration goes in `extra` instead.
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
        related_objects=[RelatedObject(type="labels", path="../segmentation", node_prop="seg_id")],
        extra={**extra, "trench": int(trench_id),
               "pixel_size_um": float(pixel_size), "frame_interval_min": float(interval_min)},
    )
    geff_path = collection / group_name(trench_id) / GEFF_NAME
    write_arrays(
        str(geff_path), np.arange(n, dtype=np.int64), node_props,
        np.asarray(edges, dtype=np.int64).reshape(-1, 2),
        {"score": {"values": np.asarray(scores, dtype=np.float32), "missing": None}},
        metadata, zarr_format=3, overwrite=True,
    )
    return geff_path



def read_trench_geff(trench_group: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Read a trench's graph (written by :func:`write_trench_geff`) as arrays.

    Faster than a graph library for summaries and plots. Works on curated copies
    too: edited graphs may have gaps in their node ids, so ``edges`` are mapped to
    positions in the node arrays.

    Returns
    -------
    dict
        ``node_id`` and the :data:`NODE_PROPS` per node, ``edges``
        (``(n_edges, 2)`` parent, child positions) and ``score`` per edge (when
        present).
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
    and the parent is its only parent; lineages are the connected parts.
    """
    children: Dict[int, List[int]] = {}
    parents: Dict[int, List[int]] = {}
    for parent, child in np.asarray(edges, dtype=np.int64).reshape(-1, 2):
        children.setdefault(int(parent), []).append(int(child))
        parents.setdefault(int(child), []).append(int(parent))
    track_id = np.full(n_nodes, -1, dtype=np.int64)
    next_track = 0
    for node in np.argsort(times, kind="stable"):
        ps = parents.get(int(node), [])
        if len(ps) == 1 and len(children.get(ps[0], [])) == 1:
            track_id[node] = track_id[ps[0]]
        else:
            track_id[node] = next_track
            next_track += 1
    lineage_id = np.full(n_nodes, -1, dtype=np.int64)
    next_lineage = 0
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


@contextlib.contextmanager
def _quiet_trackastra():
    """Trackastra and the ILP solver log every window and solve, print solution
    stats, and one progress bar ignores ``progbar_class``."""
    import trackastra.tracking.ilp  # noqa: F401  (creates the solver loggers)
    import trackastra.tracking.tracking as tracking

    loggers = [logging.getLogger(n) for n in list(logging.root.manager.loggerDict)
               if n.split(".")[0] in ("trackastra", "motile", "ilpy")]
    levels = [lg.level for lg in loggers]
    tqdm_before = tracking.tqdm
    for lg in loggers:
        lg.setLevel(logging.ERROR)
    tracking.tqdm = functools.partial(tqdm_before, disable=True)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    finally:
        tracking.tqdm = tqdm_before
        for lg, level in zip(loggers, levels):
            lg.setLevel(level)


def _no_progress(iterable, **kwargs):
    return iterable


class Tracker:
    """Track one segmentation hypothesis of a trench store with Trackastra.

    Each trench is tracked on its own and written as a GEFF group into
    ``<tracks_store>/<name>`` (default name ``trackastra_<hypothesis>``), see
    :func:`write_trench_geff`. Every segmented cell becomes a node, including
    cells Trackastra links to nothing.

    Parameters
    ----------
    store_path : str | Path
        Trench store with a ``/segmentation/<segmentation>`` group.
    tracks_store : str | Path
        Store the tracks are written to.
    segmentation : str
        Segmentation group name.
    hypothesis : str
        Segmentation hypothesis to track.
    image_channel : str
        Channel for Trackastra's intensity features (it takes one).
    model : str
        Trackastra pretrained model.
    mode : str
        ``"ilp"``, ``"greedy"`` or ``"greedy_nodiv"``.
    name : str | None
        Collection name; defaults to ``trackastra_<hypothesis>``.
    """

    def __init__(
        self,
        store_path: Union[str, Path],
        tracks_store: Union[str, Path],
        segmentation: str = "mcherry",
        hypothesis: str = "S1",
        image_channel: str = "mCherry",
        model: str = "general_2d",
        mode: str = "ilp",
        name: Optional[str] = None,
        device: str = "cuda",
    ) -> None:
        self.store_path = Path(store_path)
        self.tracks_store = Path(tracks_store)
        self.segmentation = segmentation
        self.hypothesis = hypothesis
        self.image_channel = image_channel
        self.model_name = model
        self.mode = mode
        self.name = name or f"trackastra_{hypothesis}"
        self.collection = self.tracks_store / self.name
        self.device = device
        self._model = None

        root = xr.open_zarr(self.store_path, consolidated=False)
        acquisition = xr.open_zarr(self.store_path, group="acquisition", consolidated=False)
        self.trench_ids = [int(t) for t in root["Trench"].values]
        self.frames = root["T"].values.astype(np.int64)
        self.channels = [str(c) for c in root["C"].values]
        if image_channel not in self.channels:
            raise ValueError(f"channel {image_channel!r} not in store channels {self.channels}")
        self.pixel_size = float(acquisition["pixel_size"])
        self.interval_min = float(acquisition["time_interval"]) / 60000  # ms -> min
        store = zarr.open_group(str(self.store_path), mode="r")
        self._data = store["data"]
        seg_group = store[f"segmentation/{segmentation}"]
        hypotheses = [str(h) for h in seg_group["Hypothesis"][:]]
        if hypothesis not in hypotheses:
            raise ValueError(f"hypothesis {hypothesis!r} not in {hypotheses}")
        if not dict(seg_group.attrs).get("complete", {}).get(hypothesis):
            raise RuntimeError(f"{hypothesis} is not segmented in every trench yet")
        self._labels = seg_group["labels"]
        self._h_index = hypotheses.index(hypothesis)
        self._c_index = self.channels.index(image_channel)

    @property
    def settings(self) -> Dict[str, Any]:
        """What produced the tracks (checked on resume)."""
        from importlib.metadata import PackageNotFoundError, version

        try:
            trackastra_version = version("trackastra")
        except PackageNotFoundError:
            trackastra_version = None
        return {
            "tracker": "trackastra",
            "trackastra_version": trackastra_version,
            "model": self.model_name,
            "mode": self.mode,
            "image_channel": self.image_channel,
            "trench_store": str(self.store_path),
            "labels": f"segmentation/{self.segmentation}/labels",
            "hypothesis": self.hypothesis,
        }

    def _get_model(self):
        if self._model is None:
            from trackastra.model import Trackastra

            with _quiet_trackastra():
                self._model = Trackastra.from_pretrained(self.model_name, device=self.device)
        return self._model

    def track_trench(self, index: int) -> Optional[Dict[str, np.ndarray]]:
        """Track the trench at position ``index``; ``None`` if it has no cells.

        Returns node arrays (:data:`NODE_PROPS`, ``t`` as frame positions),
        ``edges`` (positions into the node arrays) and ``score`` per edge.
        """
        masks = self._labels[self._h_index, index]
        if not masks.any():
            return None
        imgs = self._data[index, :, self._c_index]
        model = self._get_model()
        with _quiet_trackastra():
            graph, _ = model.track(imgs, masks, mode=self.mode, progbar_class=_no_progress)

        # Every segmented cell is a node, keyed by (frame, label); Trackastra leaves
        # cells it links to nothing out of its graph.
        keys = {(int(graph.nodes[n]["time"]), int(graph.nodes[n]["label"])): n for n in graph.nodes}
        rows = np.arange(masks.shape[1], dtype=np.float64)[:, None]
        cols = np.arange(masks.shape[2], dtype=np.float64)[None, :]
        t_list, label_list, y_list, x_list, position = [], [], [], [], {}
        for t in range(masks.shape[0]):
            frame = masks[t]
            if not frame.any():
                continue
            flat = frame.ravel()
            counts = np.bincount(flat)
            ids = np.flatnonzero(counts)
            ids = ids[ids > 0]
            ys = np.bincount(flat, weights=np.broadcast_to(rows, frame.shape).ravel())[ids] / counts[ids]
            xs = np.bincount(flat, weights=np.broadcast_to(cols, frame.shape).ravel())[ids] / counts[ids]
            for label, y, x in zip(ids, ys, xs):
                key = keys.get((t, int(label)))
                if key is not None:
                    position[key] = len(t_list)
                t_list.append(t)
                label_list.append(int(label))
                y_list.append(y)
                x_list.append(x)
        edges = np.array([[position[u], position[v]] for u, v in graph.edges],
                         dtype=np.int64).reshape(-1, 2)
        scores = np.array([float(graph.edges[e].get("weight", np.nan)) for e in graph.edges],
                          dtype=np.float32)
        times = np.asarray(t_list, dtype=np.int32)
        track_id, lineage_id = tracklets_and_lineages(len(t_list), times, edges)
        return {
            "t": times, "seg_id": np.asarray(label_list, dtype=np.int32),
            "y": np.asarray(y_list, dtype=np.float32), "x": np.asarray(x_list, dtype=np.float32),
            "track_id": track_id, "lineage_id": lineage_id, "edges": edges, "score": scores,
        }

    def _saved_attrs(self) -> Optional[Dict[str, Any]]:
        if not (self.collection / "zarr.json").is_file():
            return None
        return dict(zarr.open_group(str(self.collection), mode="r").attrs)

    def track(self, resume: bool = True, show_progress: bool = True) -> Dict[str, Any]:
        """Track every trench and write its GEFF group.

        Each finished trench is recorded in the collection's ``completed`` attr,
        so an interrupted run resumes. A trench that fails is recorded under
        ``failed`` and skipped.

        Parameters
        ----------
        resume : bool
            Continue a collection made with the same settings. ``False`` deletes
            the collection and starts again (it is generated output; curated
            copies live in the curation store).
        show_progress : bool
            Show a tqdm progress bar.

        Returns
        -------
        dict
            ``tracked``, ``empty`` and ``failed`` trench counts.
        """
        settings = self.settings
        saved = self._saved_attrs()
        if saved is not None and resume:
            differ = [k for k, v in settings.items() if saved.get("settings", {}).get(k) != v]
            if differ:
                raise RuntimeError(
                    f"{self.collection} was tracked with different settings "
                    f"({', '.join(differ)}). Use resume=False to track again."
                )
        if saved is None or not resume:
            if self.collection.exists():
                shutil.rmtree(self.collection)
            _ensure_group(self.tracks_store, {
                "layout": "tracks", "trench_store": str(self.store_path),
                "description": "Cell tracks; one collection per tracker and hypothesis, "
                               "one GEFF group per trench.",
            })
            _ensure_group(self.collection, {
                "schema_version": SCHEMA_VERSION,
                "layout": "trench_tracks",
                "provenance": provenance(),
                "settings": settings,
                "completed": [], "empty": [], "failed": {}, "complete": False,
            })
        group = zarr.open_group(str(self.collection), mode="r+")
        completed = set(group.attrs.get("completed", []))
        empty = set(group.attrs.get("empty", []))
        failed = dict(group.attrs.get("failed", {}))
        extra = {**settings, "channels": self.channels}

        todo = [i for i, tid in enumerate(self.trench_ids)
                if tid not in completed and tid not in empty]
        for index in tqdm(todo, desc=f"Tracking {self.hypothesis}", disable=not show_progress):
            tid = self.trench_ids[index]
            try:
                result = self.track_trench(index)
            except Exception as error:  # record and carry on
                failed[str(tid)] = repr(error)
                group.attrs["failed"] = failed
                continue
            if result is None:
                empty.add(tid)
                group.attrs["empty"] = sorted(empty)
                continue
            write_trench_geff(
                self.collection, tid, {k: result[k] for k in NODE_PROPS},
                result["edges"], result["score"], self._labels[self._h_index, index],
                pixel_size=self.pixel_size, interval_min=self.interval_min,
                extra={**extra, "trench_index": index},
            )
            completed.add(tid)
            failed.pop(str(tid), None)
            group.attrs.update({
                "completed": [t for t in self.trench_ids if t in completed],
                "failed": failed,
            })
        group.attrs["complete"] = all(t in completed or t in empty for t in self.trench_ids)
        return {"tracked": len(completed), "empty": len(empty), "failed": len(failed)}


def prepare_curation(
    tracks_collection: Union[str, Path],
    curation_collection: Union[str, Path],
    replace_unedited: bool = False,
) -> Dict[str, List[int]]:
    """Copy a tracks collection's trenches into the curation store.

    A trench that is already in the curation store is never overwritten if it has
    edits (an mm-track edit history). Without edits it is replaced only with
    ``replace_unedited``, and only if the tracks were regenerated since it was
    copied.

    Returns
    -------
    dict
        Trench ids by outcome: ``copied`` (new), ``replaced`` (unedited,
        refreshed), ``kept`` (already current), ``stale_unedited`` (unedited
        but from older tracks; pass ``replace_unedited=True``),
        ``stale_edited`` (edited, and the tracks were regenerated since: kept
        as they are).
    """
    tracks_collection = Path(tracks_collection)
    curation_collection = Path(curation_collection)
    source = zarr.open_group(str(tracks_collection), mode="r")
    run = dict(source.attrs).get("provenance", {}).get("created")
    _ensure_group(curation_collection.parent, {
        "layout": "curation",
        "description": "Curated copies of tracks; edited in place by mm-track.",
    })
    target = _ensure_group(curation_collection, {
        "layout": "trench_curation", "source": str(tracks_collection),
    })
    copied_from: Dict[str, Any] = dict(target.attrs.get("copied_from_run", {}))
    outcome: Dict[str, List[int]] = {k: [] for k in
                                     ("copied", "replaced", "kept", "stale_unedited", "stale_edited")}
    for group in trench_groups(tracks_collection):
        tid = trench_of(group)
        destination = curation_collection / group.name
        current = copied_from.get(str(tid)) == run
        if not destination.exists():
            shutil.copytree(group, destination)
            outcome["copied"].append(tid)
        elif current:
            outcome["kept"].append(tid)
            continue
        elif is_edited(destination):
            outcome["stale_edited"].append(tid)
            continue
        elif replace_unedited:
            shutil.rmtree(destination)
            shutil.copytree(group, destination)
            outcome["replaced"].append(tid)
        else:
            outcome["stale_unedited"].append(tid)
            continue
        copied_from[str(tid)] = run
    target.attrs["copied_from_run"] = copied_from
    return outcome
