from __future__ import annotations

import numpy as np
import zarr

from pymmm.store import geff


def test_tracklets_and_lineages_follow_the_geff_definitions():
    # 0 -> 1 -> {2, 3}, 3 -> 4; 5 alone
    times = np.array([0, 1, 2, 2, 3, 0])
    edges = np.array([[0, 1], [1, 2], [1, 3], [3, 4]])
    track_id, lineage_id = geff.tracklets_and_lineages(6, times, edges)

    assert track_id[0] == track_id[1]
    assert len({track_id[1], track_id[2], track_id[3]}) == 3  # daughters start anew
    assert track_id[3] == track_id[4]
    assert len(set(lineage_id[:5])) == 1 and lineage_id[5] != lineage_id[0]
    # numbered from 1 (0 is no track to mm-track), in time order
    assert sorted(set(track_id)) == [1, 2, 3, 4] and track_id[0] == 1
    assert sorted(set(lineage_id)) == [1, 2] and lineage_id[0] == 1


def test_read_trench_geff_maps_edges_past_gaps_in_node_ids(tmp_path):
    # a curated graph: node ids 2, 9, 5 (stored out of order, with gaps); edges 2->9, 2->5
    graph = zarr.open_group(str(tmp_path / "trench_0007" / geff.GEFF_NAME), mode="w", zarr_format=3)
    graph.create_array("nodes/ids", data=np.array([9, 2, 5]))
    for key, values in {"t": [1, 0, 1], "seg_id": [1, 1, 2], "y": [1.5, 1.5, 5.5]}.items():
        graph.create_array(f"nodes/props/{key}/values", data=np.asarray(values))
    graph.create_array("edges/ids", data=np.array([[2, 9], [2, 5]]))
    graph.create_array("edges/props/score/values", data=np.array([0.9, 0.8]))

    arrays = geff.read_trench_geff(tmp_path / "trench_0007")
    np.testing.assert_array_equal(arrays["node_id"], [2, 5, 9])
    np.testing.assert_array_equal(arrays["t"], [0, 1, 1])
    np.testing.assert_array_equal(arrays["seg_id"], [1, 2, 1])
    np.testing.assert_array_equal(arrays["edges"], [[0, 2], [0, 1]])
    np.testing.assert_allclose(arrays["score"], [0.9, 0.8])
    assert "x" not in arrays
