"""``pymmm.steps.lanes``: lane detection in one FOV (the widget: test_widgets_lanes.py)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pymmm.steps.lanes import LaneDetector


def test_lane_orientation_can_be_forced():
    # Brighter below the lane, so "auto" infers trenches below (open up, -1).
    image = np.zeros((200, 10))
    image[95:105] = 10.0
    image[105:150] = 2.0
    experiment = SimpleNamespace(pixel_size_um=1.0, channel_names=["PC"])
    detector = LaneDetector(experiment, None, None, detection_channel="PC")

    def orientations(orientation):
        *_, lanes = detector.detect_fov(
            image, 0, 50, 5, 20, "XYPos:0", orientation=orientation,
        )
        return [lane.orientation for lane in lanes]

    assert orientations("auto") == [-1]
    assert orientations("down") == [1]
    assert orientations("up") == [-1]
