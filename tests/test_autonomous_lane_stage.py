from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

from pymmm.lane_detector import LaneInfo

from pymmm.autonomous.config import DetectionConfig, QCConfig
from pymmm.autonomous.lane_stage import (
    check_lane_counts,
    lane_table_from_lanes,
    write_lane_diagnostics,
)

matplotlib.use("Agg")


def test_lane_table_from_lanes_is_stable() -> None:
    lanes = {
        "XYPos:0": [
            LaneInfo(y_center=420, orientation=1, fov="XYPos:0", lane_index=0),
            LaneInfo(y_center=1600, orientation=-1, fov="XYPos:0", lane_index=1),
        ],
        "XYPos:1": [
            LaneInfo(y_center=400, orientation=1, fov="XYPos:1", lane_index=0),
        ],
    }

    table = lane_table_from_lanes(lanes)

    assert list(table.columns) == ["fov", "lane_index", "y_center", "orientation"]
    assert table.to_dict("records") == [
        {"fov": "XYPos:0", "lane_index": 0, "y_center": 420, "orientation": 1},
        {"fov": "XYPos:0", "lane_index": 1, "y_center": 1600, "orientation": -1},
        {"fov": "XYPos:1", "lane_index": 0, "y_center": 400, "orientation": 1},
    ]


def test_check_lane_counts_fails_out_of_range() -> None:
    table = pd.DataFrame(
        [
            {"fov": "XYPos:0", "lane_index": 0, "y_center": 420, "orientation": 1},
            {"fov": "XYPos:0", "lane_index": 1, "y_center": 1600, "orientation": -1},
            {"fov": "XYPos:1", "lane_index": 0, "y_center": 400, "orientation": 1},
        ]
    )

    report = check_lane_counts(table, QCConfig(min_lanes_per_fov=2, max_lanes_per_fov=2))

    assert report["passed"] is False
    assert report["counts_by_fov"] == {"XYPos:0": 2, "XYPos:1": 1}
    assert report["failed_fovs"] == ["XYPos:1"]


def test_write_lane_diagnostics_writes_csv_json_and_overlay(tmp_path: Path) -> None:
    table = pd.DataFrame(
        [
            {"fov": "XYPos:0", "lane_index": 0, "y_center": 20, "orientation": 1},
            {"fov": "XYPos:0", "lane_index": 1, "y_center": 44, "orientation": -1},
        ]
    )
    mean_images = {"XYPos:0": np.ones((64, 96), dtype=np.float32)}
    qc = {"passed": True, "counts_by_fov": {"XYPos:0": 2}, "failed_fovs": []}

    paths = write_lane_diagnostics(
        tmp_path,
        table,
        mean_images,
        qc,
        DetectionConfig(min_lanes_per_fov=2, max_lanes_per_fov=2),
    )

    assert paths.table_path.exists()
    assert paths.qc_path.exists()
    assert paths.overlay_grid_path.exists()
    assert pd.read_csv(paths.table_path).shape == (2, 4)
    assert "XYPos:0" in paths.qc_path.read_text()
