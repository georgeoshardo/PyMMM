"""``pymmm.widgets.trenches``: the trench widget, driven without a browser (the
``notebook`` fixture): its outlines, its per-lane controls, and that Apply runs the
detection with them."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from fake_experiment import FOV_H, FOV_W, fov_experiment, fov_registrator

from pymmm.steps.lanes import LaneDetector, LaneInfo
from pymmm.steps.trenches import TrenchDetector

pytestmark = [
    pytest.mark.notebook,
    pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive"),
]


@pytest.fixture
def trench_setup(tmp_path):
    image = np.full((FOV_H, FOV_W), 10.0, dtype=np.float32)
    image[30:90, 20:150:40] = 40.0  # lane 0's trenches, above it (open down)
    image[120:180, 30:150:30] = 40.0  # lane 1's trenches, below it (open up)
    experiment = fov_experiment(tmp_path, ["f0"])
    lanes = SimpleNamespace(
        lanes={"f0": [LaneInfo(100, 1, "f0", 0), LaneInfo(110, -1, "f0", 1)]},
        image_edge=None,
    )
    return experiment, fov_registrator({"f0": image}), lanes, image


def _trench_widget(notebook, setup):
    experiment, registrator, lanes, _ = setup
    detector = TrenchDetector(
        experiment, registrator, lanes, None, detection_channel="PC"
    )
    detector.interactive_detect_trenches()
    for name, value in [
        ("trench width", 6),
        ("trench length", 60),
        ("bottom offset", 10),
        ("sigma", 0.0),
        ("distance", 5),
        ("prominence", 1),
    ]:
        notebook.control(name).value = value
    return detector


def test_trench_widget_previews_outlines_and_lane_controls(notebook, trench_setup):
    detector = _trench_widget(notebook, trench_setup)
    (viewer,) = notebook.viewers
    assert viewer.kwargs["luts"][1]["name"] == "candidate trenches"
    expected, _, _ = detector.detect_fov(
        trench_setup[3],
        "f0",
        sigma=0.0,
        distance=5,
        prominence=1,
        height=0,
        trench_width=6,
        trench_length=60,
        trench_bottom_offset=10,
    )
    assert (
        notebook.text("candidate") == f"<b>f0</b> — {len(expected)} candidate trenches"
    )
    outline = viewer.data[1]
    for trench in expected:
        assert outline[trench.y_top, trench.x_left : trench.x_right].all()
        assert outline[trench.y_top : trench.y_bottom, trench.x_left].all()
    assert outline.sum() > 0 and not outline[95:115].any()

    # each lane's controls follow the global ones until set on their own
    assert (
        notebook.control("L0 sigma").value == notebook.control("L1 sigma").value == 0.0
    )
    notebook.control("L1 prominence").value = 7
    notebook.control("prominence").value = 2
    assert notebook.control("L0 prominence").value == 2
    assert notebook.control("L1 prominence").value == 7

    notebook.control("method").value = "periodic"
    assert (
        notebook.control("distance").disabled
        and notebook.control("prominence").disabled
    )
    assert not notebook.control("min symmetry").disabled


def test_trench_widget_applies_lane_overrides(notebook, trench_setup):
    detector = _trench_widget(notebook, trench_setup)
    notebook.control("L1 prominence").value = 7
    notebook.control("Apply to all FOVs").click()

    experiment, registrator, lanes, _ = trench_setup
    expected = TrenchDetector(
        experiment, registrator, lanes, None, detection_channel="PC"
    )
    expected.detect_trenches(
        sigma=0.0,
        distance=5,
        prominence=1,
        height=0,
        trench_width=6,
        trench_length=60,
        trench_bottom_offset=10,
        lane_params={1: {"prominence": 7}},
        save_diagnostics=False,
    )
    assert detector.get_trench_table().equals(expected.get_trench_table())
    assert detector.detection_params == expected.detection_params
    assert detector.detection_params["lane_params"] == {1: {"prominence": 7}}
    assert notebook.text("Applied") == (
        f"<b>Applied</b> — {expected.n_trenches} trenches across 1 FOVs"
    )


def test_with_image_edge_lanes_the_widget_starts_at_the_whole_height(
    notebook, tmp_path
):
    image = np.full((FOV_H, FOV_W), 10.0, dtype=np.float32)
    image[:, 20:150:40] = 40.0
    experiment = fov_experiment(tmp_path, ["f0"])
    registrator = fov_registrator({"f0": image})
    lanes = LaneDetector(experiment, registrator, None, detection_channel="PC")
    lanes.use_image_edge("down", save_diagnostics=False)
    detector = TrenchDetector(
        experiment, registrator, lanes, None, detection_channel="PC"
    )
    detector.interactive_detect_trenches()
    assert notebook.control("trench length").value == FOV_H
    assert notebook.control("bottom offset").value == 0
    for name, value in [
        ("trench width", 6),
        ("sigma", 0.0),
        ("distance", 5),
        ("prominence", 1),
    ]:
        notebook.control(name).value = value
    notebook.control("Apply to all FOVs").click()
    assert {(t.y_top, t.y_bottom) for t in detector.trenches["f0"]} == {(0, FOV_H)}
