"""``pymmm.source.nd2``: the ND2Experiment and the subset of the ND2 it selects."""

from __future__ import annotations

import pytest
from fake_experiment import fake_nd2

from pymmm.source.nd2 import ND2Experiment


def test_nd2experiment_subset_metadata_tracks_selection(monkeypatch, tmp_path):
    nd2_path = fake_nd2(
        monkeypatch, tmp_path / "subset.nd2", ["XYPos:0", "XYPos:0"], (2, 2)
    )

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


FOUR = [f"XYPos:{i}" for i in range(4)]


def test_fovs_are_named_or_given_by_their_p_index_in_the_nd2(monkeypatch, tmp_path):
    nd2_path = fake_nd2(monkeypatch, tmp_path / "fovs.nd2", FOUR)
    by_index = ND2Experiment(nd2_path).discard_fovs([1, 2])
    by_name = ND2Experiment(nd2_path).discard_fovs(["XYPos:1", "XYPos:2"])
    assert by_index.fov_names == by_name.fov_names == ["XYPos:0", "XYPos:3"]
    assert list(by_index.data.coords["P"].values) == ["XYPos:0", "XYPos:3"]
    assert ND2Experiment(nd2_path).select_fovs([1, "XYPos:3"]).fov_names == [
        "XYPos:1",
        "XYPos:3",
    ]
    # P 3 whatever is selected, not the second FOV kept
    assert ND2Experiment(nd2_path).select_fovs([2, 3]).discard_fovs([3]).fov_names == [
        "XYPos:2"
    ]


@pytest.mark.parametrize(
    "fovs, message",
    [
        ([4], r"FOV index 4 is out of range: the ND2 has 4 FOVs \(P 0-3\)"),
        ([-1], r"FOV index -1 is out of range"),
        (
            ["XYPos:9"],
            r"no FOV named 'XYPos:9' in the ND2 \(XYPos:0, XYPos:1, XYPos:2, XYPos:3\)",
        ),
        (["10"], r"no FOV named '10'"),
    ],
)
def test_an_unknown_fov_is_an_error_not_a_silent_no_op(
    monkeypatch, tmp_path, fovs, message
):
    nd2_path = fake_nd2(monkeypatch, tmp_path / "fovs.nd2", FOUR)
    exp = ND2Experiment(nd2_path)
    with pytest.raises(ValueError, match=message):
        exp.discard_fovs(fovs)
    with pytest.raises(ValueError, match=message):
        exp.select_fovs(fovs)
    assert exp.fov_names == FOUR  # nothing half-applied
