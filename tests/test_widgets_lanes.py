"""``pymmm.widgets.lanes``: the lane widget, driven without a browser (the ``notebook``
fixture): its preview, its controls, and that Apply runs the detection with them."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest
from fake_experiment import fov_experiment, fov_registrator, lane_image

from pymmm.plotting import NDV_CLIMS
from pymmm.steps.lanes import LaneDetector

pytestmark = [
    pytest.mark.notebook,
    pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive"),
]


@pytest.fixture
def lane_setup(tmp_path):
    images = {"f0": lane_image(120), "f1": lane_image(110, shift=5)}
    return fov_experiment(tmp_path, images), fov_registrator(images), images


def test_lane_widget_previews_each_fov(notebook, lane_setup):
    experiment, registrator, images = lane_setup
    detector = LaneDetector(experiment, registrator, None, detection_channel="PC")
    detector.interactive_detect_lanes()

    (viewer,) = notebook.viewers
    assert viewer.kwargs["channel_axis"] == 0
    assert viewer.kwargs["luts"][0] == {
        "name": "PC",
        "cmap": "gray",
        "clims": NDV_CLIMS,
    }
    assert [viewer.kwargs["luts"][c]["name"] for c in (1, 2)] == [
        "open down",
        "open up",
    ]
    assert viewer.renders == 1
    assert notebook.text("candidate") == "<b>f0</b> — 1 candidate lane"
    (lane,) = detector.detect_fov(images["f0"], 40, 300, 5000, 20, "f0")[3]
    channel, other = (1, 2) if lane.orientation == 1 else (2, 1)
    lines = np.flatnonzero(viewer.data[channel].any(axis=1))
    np.testing.assert_array_equal(
        lines, np.arange(lane.y_center - 2, lane.y_center + 3)
    )
    assert not viewer.data[other].any()
    blanked = viewer.data[1:].any(axis=0)
    assert np.all(viewer.data[0][blanked] == images["f0"].min())
    np.testing.assert_array_equal(viewer.data[0][~blanked], images["f0"][~blanked])

    notebook.control("orientation").value = "down"
    assert notebook.control("orient. window (µm)").disabled
    assert viewer.data[1].any() and not viewer.data[2].any()
    notebook.control("orientation").value = "auto"
    assert not notebook.control("orient. window (µm)").disabled
    notebook.control("FOV").value = "f1"
    assert notebook.text("candidate").startswith("<b>f1</b>")
    assert viewer.renders == 4


def test_lane_widget_applies_its_settings_to_every_fov(notebook, lane_setup):
    experiment, registrator, _ = lane_setup
    detector = LaneDetector(experiment, registrator, None, detection_channel="PC")
    detector.interactive_detect_lanes()
    notebook.control("sigma").value = 3
    notebook.control("distance").value = 50
    notebook.control("height").value = 3000
    notebook.control("orientation").value = "down"
    notebook.control("Apply to all FOVs").click()

    expected = LaneDetector(experiment, registrator, None, detection_channel="PC")
    expected.detect_lanes(
        sigma=3,
        distance=50,
        height=3000,
        orientation_window_um=20,
        orientation="down",
        save_diagnostics=False,
    )
    assert {f: [asdict(l) for l in ls] for f, ls in detector.lanes.items()} == {
        f: [asdict(l) for l in ls] for f, ls in expected.lanes.items()
    }
    assert detector.detection_params == expected.detection_params
    assert notebook.text("Applied") == "<b>Applied</b> — 2 lanes across 2 FOVs"
