"""``pymmm.source.metadata``: the acquisition's and each frame's metadata, read from the
ND2 (a :class:`FakeND2File`)."""

from __future__ import annotations

import json

import pytest
from pymmm.source.metadata import (
    _median_frame_interval_ms,
    extract_acquisition_metadata,
    extract_source_frame_metadata,
)
from pymmm.source.nd2 import ND2Experiment

from fake_experiment import fake_nd2


def test_extract_acquisition_metadata_is_json_serializable(monkeypatch, tmp_path):
    nd2_path = fake_nd2(monkeypatch, tmp_path / "sample.nd2")

    bundle = extract_acquisition_metadata(
        nd2_path,
        pixel_size_um=0.108,
        time_interval_ms=30000.0,
    )

    json.dumps(bundle)
    assert bundle["file_summary"]["pixel_size_um"] == 0.108
    assert bundle["metadata"]["channels"][0]["channel"]["name"] == "PC"
    assert bundle["custom_data"]["StreamDataV1_0"]["ignored"] is True
    assert bundle["unstructured_metadata"]["ImageEventsLV"]["RLxExperimentRecord"] == {}
    assert bundle["channel_exposure_ms"] == [80.0, 200.0]


def test_extract_source_frame_metadata_tracks_subset(monkeypatch, tmp_path):
    nd2_path = fake_nd2(monkeypatch, tmp_path / "subset_source.nd2")

    exp = ND2Experiment(nd2_path)
    exp.select_fovs(["XYPos:1"]).select_times(1, 2)

    exported = extract_source_frame_metadata(exp)

    assert exported["original_p_index"].tolist() == [1]
    assert exported["original_t_index"].tolist() == [1]
    assert exported["source_seq_index"].tolist() == [[3]]
    assert exported["relative_time"].tolist() == [[30124.0]]
    assert exported["stage_position"].tolist() == [[[101.0, 201.0, 302.0]]]
    assert exported["pfs_offset"].tolist() == [[17414.0]]
    assert exported["pfs_status"].tolist() == [[0.0]]
    assert "position_name" not in exported
    static = json.loads(exported["frame_metadata_json"][0])
    assert static["channels"][1]["channel"]["name"] == "mCherry"
    assert "time" not in static["channels"][0]
    assert "stagePositionUm" not in static["channels"][0]["position"]
    assert "frame_event_records" not in exported
    assert len(exported["acquisition_event_records"]) == 4


def test_time_interval_is_median_of_real_frame_gaps():
    rows = [{"Time [s]": 0.0, "Events": "Command Executed"}]
    for t, (t0, t1) in enumerate(
        [(0.0, 1.0), (60.0, 61.5), (121.0, 122.0), (180.0, 181.0)]
    ):
        rows += [
            {"Index": 2 * t, "T Index": t, "P Index": 0, "Time [s]": t0},
            {"Index": 2 * t + 1, "T Index": t, "P Index": 1, "Time [s]": t1},
        ]
    # FOV 0 gaps: 60, 61, 59 s; FOV 1 gaps: 60.5, 60.5, 59 s
    assert _median_frame_interval_ms(rows) == pytest.approx(60250.0)
    assert _median_frame_interval_ms(rows[:1]) is None
