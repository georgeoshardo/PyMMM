from __future__ import annotations

import json

import networkx as nx
import numpy as np
import pytest
import xarray as xr
import zarr

from pymmm import tracks as tr
from pymmm.config import Experiment
from pymmm.segmenter import Segmenter
from test_segmenter import N_T, H, W, _FakeModel, _make_store


def test_tracklets_and_lineages_follow_the_geff_definitions():
    # 0 -> 1 -> {2, 3}, 3 -> 4; 5 alone
    times = np.array([0, 1, 2, 2, 3, 0])
    edges = np.array([[0, 1], [1, 2], [1, 3], [3, 4]])
    track_id, lineage_id = tr.tracklets_and_lineages(6, times, edges)

    assert track_id[0] == track_id[1]
    assert len({track_id[1], track_id[2], track_id[3]}) == 3  # daughters start anew
    assert track_id[3] == track_id[4]
    assert len(set(lineage_id[:5])) == 1 and lineage_id[5] != lineage_id[0]


def test_write_trench_geff_is_a_valid_geff_next_to_its_segmentation(tmp_path):
    geff = pytest.importorskip("geff")
    segmentation = np.zeros((2, 8, 4), dtype=np.uint16)
    segmentation[0, 1:3, 1:3] = 1
    segmentation[1, 1:3, 1:3] = 1
    segmentation[1, 5:7, 1:3] = 2
    nodes = {
        "t": np.array([0, 1, 1], dtype=np.int32), "y": np.array([1.5, 1.5, 5.5], dtype=np.float32),
        "x": np.array([1.5, 1.5, 1.5], dtype=np.float32), "seg_id": np.array([1, 1, 2], dtype=np.int32),
        "track_id": np.array([0, 1, 2]), "lineage_id": np.array([0, 0, 0]),
    }
    path = tr.write_trench_geff(
        tmp_path / "tracks.zarr" / "trackastra_S1", 7, nodes, np.array([[0, 1], [0, 2]]),
        np.array([0.9, 0.8]), segmentation, pixel_size=0.1, interval_min=1.0,
        extra={"trench_store": "x"},
    )

    geff.validate_structure(str(path))
    graph, metadata = geff.read(str(path), backend="networkx")
    assert graph.number_of_nodes() == 3 and graph.number_of_edges() == 2
    assert metadata.related_objects[0].path == "../segmentation"
    assert metadata.extra["trench"] == 7 and metadata.extra["pixel_size_um"] == 0.1
    assert json.loads((path / "zarr.json").read_text())["zarr_format"] == 3
    seg = zarr.open_array(str(path.parent / "segmentation"), mode="r")
    assert seg.metadata.dimension_names == ("t", "y", "x")
    for n in graph.nodes:
        d = graph.nodes[n]
        assert seg[int(d["t"]), int(d["y"]), int(d["x"])] == d["seg_id"]

    arrays = tr.read_trench_geff(path.parent)
    np.testing.assert_array_equal(arrays["seg_id"], nodes["seg_id"])
    np.testing.assert_array_equal(arrays["edges"], [[0, 1], [0, 2]])
    np.testing.assert_allclose(arrays["score"], [0.9, 0.8])


class _FakeTrackastra:
    """Links each cell to the cell with the same label in the next frame, except
    trench-0's label-2 cell in the last frame, which it leaves unlinked."""

    def __init__(self):
        self.calls = 0

    def track(self, imgs, masks, mode, progbar_class):
        self.calls += 1
        graph = nx.DiGraph()
        node = {}
        for t in range(masks.shape[0]):
            for label in np.unique(masks[t])[1:]:
                if t == masks.shape[0] - 1 and label == 2:
                    continue
                node[(t, int(label))] = len(node)
                graph.add_node(node[(t, int(label))], time=t, label=int(label))
        for (t, label), n in node.items():
            if (t + 1, label) in node:
                graph.add_edge(n, node[(t + 1, label)], weight=0.9)
        return graph, masks


@pytest.fixture
def tracked_store(tmp_path, monkeypatch):
    pytest.importorskip("trackastra")
    store = _make_store(tmp_path)
    xr.Dataset({"pixel_size": ((), 0.1), "time_interval": ((), 60000.0)}).to_zarr(
        store, mode="w", group="acquisition", consolidated=False, zarr_format=3,
    )
    monkeypatch.setattr(Segmenter, "_get_model", lambda self: _FakeModel())
    Segmenter(store, channel="mCherry", gpu=False).segment(show_progress=False)
    fake = _FakeTrackastra()
    monkeypatch.setattr(tr.Tracker, "_get_model", lambda self: fake)
    tracker = tr.Tracker(store, tmp_path / "exp.tracks.zarr", hypothesis="S1", device="cpu")
    tracker.fake = fake
    return tracker


def test_tracker_writes_one_geff_per_trench_with_cells(tracked_store):
    geff = pytest.importorskip("geff")
    tracker = tracked_store
    counts = tracker.track(show_progress=False)

    assert counts == {"tracked": 2, "empty": 1, "failed": 0}  # trench 12 is noise only
    groups = tr.trench_groups(tracker.collection)
    assert [g.name for g in groups] == ["trench_0010", "trench_0011"]
    attrs = dict(zarr.open_group(str(tracker.collection), mode="r").attrs)
    assert attrs["complete"] is True and attrs["empty"] == [12]
    assert attrs["settings"]["hypothesis"] == "S1"

    graph, metadata = geff.read(str(groups[0] / tr.GEFF_NAME), backend="networkx")
    # two cells in each of four frames; the one left unlinked is still a node
    assert graph.number_of_nodes() == 2 * N_T
    assert graph.number_of_edges() == 2 * (N_T - 1) - 1
    assert metadata.extra["trench"] == 10 and metadata.extra["trench_index"] == 0
    seg = zarr.open_array(str(groups[0] / "segmentation"), mode="r")
    assert seg.shape == (N_T, H, W)


def test_tracker_resumes_and_refuses_changed_settings(tracked_store):
    tracker = tracked_store
    tracker.track(show_progress=False)
    calls = tracker.fake.calls
    tracker.track(show_progress=False)
    assert tracker.fake.calls == calls

    tracker.mode = "greedy"
    with pytest.raises(RuntimeError, match="mode"):
        tracker.track(show_progress=False)
    tracker.track(resume=False, show_progress=False)
    assert tracker.fake.calls == 2 * calls


def test_prepare_curation_never_overwrites_edits(tracked_store, tmp_path):
    tracker = tracked_store
    tracker.track(show_progress=False)
    curation = tmp_path / "exp.curation.zarr" / tracker.name

    first = tr.prepare_curation(tracker.collection, curation)
    assert first["copied"] == [10, 11]
    assert tr.prepare_curation(tracker.collection, curation)["kept"] == [10, 11]

    # trench 10 gets edited; then the tracks are regenerated
    history = curation / "trench_0010" / tr.GEFF_NAME / tr.EDIT_HISTORY
    history.parent.mkdir(parents=True)
    history.write_bytes(b"edits")
    tracker.track(resume=False, show_progress=False)

    report = tr.prepare_curation(tracker.collection, curation)
    assert report["stale_edited"] == [10] and report["stale_unedited"] == [11]
    report = tr.prepare_curation(tracker.collection, curation, replace_unedited=True)
    assert report["replaced"] == [11] and report["stale_edited"] == [10]
    assert history.read_bytes() == b"edits"


def test_experiment_resolves_relative_paths(tmp_path):
    (tmp_path / "exp.toml").write_text(
        'name = "exp"\nnd2 = "raw/exp.nd2"\ntrench_store = "exp.trenches.zarr"\n'
        'tracks_store = "exp.tracks.zarr"\ncuration_store = "/abs/exp.curation.zarr"\n'
        '[tracking]\nhypotheses = ["S1", "S2"]\n'
    )
    experiment = Experiment.load(tmp_path / "exp.toml")
    assert experiment.nd2 == tmp_path / "raw" / "exp.nd2"
    assert experiment.curation_store.as_posix() == "/abs/exp.curation.zarr"
    assert experiment.hypotheses == ["S1", "S2"]
    assert experiment.tracks_name("S2") == "trackastra_S2"
    assert experiment.segmentation_name == "mcherry"
    assert experiment.curation_collection("S1").as_posix() == "/abs/exp.curation.zarr/trackastra_S1"
    with pytest.raises(ValueError, match="missing"):
        (tmp_path / "bad.toml").write_text('name = "x"\n')
        Experiment.load(tmp_path / "bad.toml")


def test_experiment_builds_the_tracker_from_its_settings(tracked_store, tmp_path):
    store = tracked_store.store_path
    (tmp_path / "exp.toml").write_text(
        f'name = "exp"\nnd2 = "exp.nd2"\ntrench_store = "{store.name}"\n'
        'tracks_store = "exp.tracks.zarr"\ncuration_store = "exp.curation.zarr"\n'
        '[segmentation]\nchannel = "mCherry"\n'
        '[tracking]\nmode = "greedy"\ndevice = "cpu"\n'
    )
    experiment = Experiment.load(tmp_path / "exp.toml")
    tracker = experiment.tracker("S1")
    assert tracker.collection == experiment.tracks_collection("S1")
    assert (tracker.mode, tracker.segmentation, tracker.device) == ("greedy", "mcherry", "cpu")
    assert experiment.segmenter().group == "segmentation/mcherry"
