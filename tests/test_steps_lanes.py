"""``pymmm.steps.lanes``: lane detection in one FOV (the widget: test_widgets_lanes.py)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from pymmm.steps.lanes import LaneDetector, LaneInfo
from pymmm.store.companion import CompanionStore

from fake_experiment import FOV_H, fov_experiment


def test_lane_orientation_can_be_forced():
    # Brighter below the lane, so "auto" infers trenches below (open up, -1). The
    # background is not 0: zero pixels are a warp's fill, left out of the profile.
    image = np.ones((200, 10))
    image[95:105] = 10.0
    image[105:150] = 2.0
    experiment = SimpleNamespace(pixel_size_um=1.0, channel_names=["PC"])
    detector = LaneDetector(experiment, None, None, detection_channel="PC")

    def orientations(orientation):
        *_, lanes = detector.detect_fov(
            image,
            0,
            50,
            5,
            20,
            "XYPos:0",
            orientation=orientation,
        )
        return [lane.orientation for lane in lanes]

    assert orientations("auto") == [-1]
    assert orientations("down") == [1]
    assert orientations("up") == [-1]


@pytest.mark.parametrize("opening, y, orientation", [("down", FOV_H, 1), ("up", 0, -1)])
def test_without_lane_detection_the_lane_is_the_image_edge_where_trenches_open(
    tmp_path, opening, y, orientation
):
    # An image cropped to its trenches (EXPERIMENT_Y_ROI): "down", they open into a lane
    # below the image, so the lane is its bottom edge; "up", its top edge.
    experiment = fov_experiment(tmp_path, ["f0", "f1"])
    detector = LaneDetector(experiment, None, None, detection_channel="PC")
    assert detector.image_edge is None
    detector.use_image_edge(opening, save_diagnostics=False)
    assert detector.image_edge == opening
    assert detector.lanes == {f: [LaneInfo(y, orientation, f, 0)] for f in ("f0", "f1")}
    assert (
        repr(detector)
        == f"LaneDetector(the image's edge, trenches open {opening}: 2 FOVs)"
    )


def test_image_edge_lanes_are_saved_and_loaded_as_such(tmp_path):
    experiment = fov_experiment(tmp_path, ["f0", "f1"])
    experiment.source_subset_metadata = {}
    store = CompanionStore(tmp_path / "exp.pymmm.zarr")
    detector = LaneDetector(experiment, None, store, detection_channel="PC")
    detector.use_image_edge("down", save_diagnostics=False)
    detector.save()
    loaded = LaneDetector.load(experiment, None, store)
    assert loaded.image_edge == "down" and loaded.lanes == detector.lanes


def test_the_image_edge_takes_down_or_up():
    detector = LaneDetector(SimpleNamespace(channel_names=["PC"]), None, None, "PC")
    with pytest.raises(ValueError, match="'down' or 'up'"):
        detector.use_image_edge("auto")
