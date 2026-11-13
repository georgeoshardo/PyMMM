"""``pymmm.store.geff``: a tracking run's GEFF beside its masks, read back, and its
tracklets and lineages."""

from __future__ import annotations

import json

import numpy as np
import pytest
import zarr
from fake_experiment import make_empty_experiment

from pymmm.store import experiment_group as eg
from pymmm.store.geff import (
    GEFF_NAME,
    has_tracks,
    read_trench_geff,
    tracklets_and_lineages,
    write_tracks,
)


@pytest.fixture
def experiment(tmp_path):
    return make_empty_experiment(tmp_path)


def test_tracklets_and_lineages_follow_the_geff_definitions():
    # 0 -> 1 -> {2, 3}, 3 -> 4; 5 alone
    times = np.array([0, 1, 2, 2, 3, 0])
    edges = np.array([[0, 1], [1, 2], [1, 3], [3, 4]])
    track_id, lineage_id = tracklets_and_lineages(6, times, edges)

    assert track_id[0] == track_id[1]
    assert len({track_id[1], track_id[2], track_id[3]}) == 3  # daughters start anew
    assert track_id[3] == track_id[4]
    assert len(set(lineage_id[:5])) == 1 and lineage_id[5] != lineage_id[0]
    # numbered from 1 (0 is no track to mm-track), in time order
    assert sorted(set(track_id)) == [1, 2, 3, 4] and track_id[0] == 1
    assert sorted(set(lineage_id)) == [1, 2] and lineage_id[0] == 1


def test_read_trench_geff_maps_edges_past_gaps_in_node_ids(tmp_path):
    # a curated graph: node ids 2, 9, 5 (stored out of order, with gaps); edges 2->9, 2->5
    graph = zarr.open_group(
        str(tmp_path / "trench_0007" / GEFF_NAME), mode="w", zarr_format=3
    )
    graph.create_array("nodes/ids", data=np.array([9, 2, 5]))
    for key, values in {
        "t": [1, 0, 1],
        "seg_id": [1, 1, 2],
        "y": [1.5, 1.5, 5.5],
    }.items():
        graph.create_array(f"nodes/props/{key}/values", data=np.asarray(values))
    graph.create_array("edges/ids", data=np.array([[2, 9], [2, 5]]))
    graph.create_array("edges/props/score/values", data=np.array([0.9, 0.8]))

    arrays = read_trench_geff(tmp_path / "trench_0007")
    np.testing.assert_array_equal(arrays["node_id"], [2, 5, 9])
    np.testing.assert_array_equal(arrays["t"], [0, 1, 1])
    np.testing.assert_array_equal(arrays["seg_id"], [1, 2, 1])
    np.testing.assert_array_equal(arrays["edges"], [[0, 2], [0, 1]])
    np.testing.assert_allclose(arrays["score"], [0.9, 0.8])
    assert "x" not in arrays


@pytest.mark.tracks
def test_write_tracks_writes_a_geff_beside_its_masks(experiment):
    import geff

    root = eg.open_experiment(experiment, mode="r+")
    masks = np.zeros((5, 6, 4), dtype=np.uint16)
    masks[0, :2] = 1
    masks[1, :3] = 1
    nodes = {
        "t": np.asarray([0, 1], np.int32),
        "y": np.asarray([0.5, 1.0], np.float32),
        "x": np.asarray([1.5, 1.5], np.float32),
        "seg_id": np.asarray([1, 1], np.int32),
        "track_id": np.asarray([0, 0]),
        "lineage_id": np.asarray([0, 0]),
    }
    path = write_tracks(
        root,
        4,
        "run",
        nodes,
        np.asarray([[0, 1]]),
        np.asarray([np.nan]),
        masks,
        extra={"tracker": "test"},
    )

    assert path == experiment / "trench_0004" / "tracks" / "run.geff"
    assert has_tracks(root, 4, "run") and not has_tracks(root, 3, "run")
    assert eg.listed_labels(root, 4) == ["tracks.run"]
    np.testing.assert_array_equal(root["trench_0004/labels/tracks.run/0"][...], masks)
    meta = json.loads((path / "zarr.json").read_text())["attributes"]["geff"]
    assert meta["related_objects"][0]["path"] == "../../labels/tracks.run"
    assert meta["related_objects"][0]["node_prop"] == "seg_id"
    assert meta["related_objects"][1] == {
        "type": "image",
        "path": "../..",
        "label_prop": None,
        "node_prop": None,
    }
    assert meta["extra"]["trench"] == 4 and meta["extra"]["pixel_size_um"] == 0.1
    graph, _ = geff.read(str(path))
    assert sorted(graph.nodes) == [1, 2] and list(graph.edges) == [(1, 2)]
