"""``pymmm.source.nd2``: the ND2Experiment and the subset of the ND2 it selects."""

from __future__ import annotations

from fake_experiment import fake_nd2

from pymmm.source.nd2 import ND2Experiment


def test_nd2experiment_subset_metadata_tracks_selection(monkeypatch, tmp_path):
    nd2_path = fake_nd2(monkeypatch, tmp_path / "subset.nd2", ["XYPos:0", "XYPos:0"], (2, 2))

    exp = ND2Experiment(nd2_path)
    exp.select_times(0, 1)
    exp.select_roi(y=(0, 1), x=(1, 2))
    exp.discard_fovs([exp.fov_names[1]])

    subset = exp.source_subset_metadata

    assert subset["raw_fov_names"] == ["XYPos:0", "XYPos:0"]
    assert subset["active_fov_names"] == ["XYPos:0-01"]
    assert subset["discarded_fov_names"] == ["XYPos:0-02"]
    assert subset["time_slice"] == {"start": 0, "stop": 1, "step": None}
    assert subset["roi"]["y"] == {"start": 0, "stop": 1, "step": None}
    assert subset["roi"]["x"] == {"start": 1, "stop": 2, "step": None}
    assert subset["raw_timepoint_count"] == 2
    assert subset["active_timepoint_count"] == 1
