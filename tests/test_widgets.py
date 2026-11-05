"""The notebook widgets, driven without a browser (see the ``notebook`` fixture): their
previews, their controls, and that Apply runs the step with the controls' settings."""

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
from fake_experiment import make_experiment
from test_segmenter import _FakeModel

from pymmm.plotting import NDV_CLIMS
from pymmm.steps.lanes import LaneDetector, LaneInfo
from pymmm.steps.segmentation import HYPOTHESES, Segmenter, trench_bounds
from pymmm.steps.trenches import TrenchDetector
from pymmm.store import experiment_group as eg
from pymmm.widgets.segmentation import label_overlay

pytestmark = pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")

H, W = 200, 160


def _lane_image(lane_y: int, shift: int = 0) -> np.ndarray:
    """A bright feeding lane at ``lane_y`` (above the widget's default height, 5000),
    bright trench columns above it."""
    image = np.full((H, W), 1000.0, dtype=np.float32)
    image[lane_y - 20:lane_y + 20] = 20000.0
    for x in range(20 + shift, W - 10, 40):
        image[lane_y - 90:lane_y - 25, x:x + 8] = 4000.0
    return image


def _experiment(tmp_path, fovs):
    return SimpleNamespace(
        fov_names=list(fovs), channel_names=["PC"], pixel_size_um=1.0,
        data=SimpleNamespace(sizes={"Y": H, "X": W}), path=tmp_path / "exp.nd2",
    )


def _registrator(images):
    return SimpleNamespace(
        get_registered_mean_of_timestack=lambda fov, channel: images[fov],
        compute_registered_means=lambda fovs=None, channel=None: {
            f: images[f] for f in (fovs or list(images))},
    )


# ----------------------------------------------------------------------------
# lanes
# ----------------------------------------------------------------------------
@pytest.fixture
def lane_setup(tmp_path):
    images = {"f0": _lane_image(120), "f1": _lane_image(110, shift=5)}
    return _experiment(tmp_path, images), _registrator(images), images


def test_lane_widget_previews_each_fov(notebook, lane_setup):
    experiment, registrator, images = lane_setup
    detector = LaneDetector(experiment, registrator, None, detection_channel="PC")
    detector.interactive_detect_lanes()

    (viewer,) = notebook.viewers
    assert viewer.kwargs["channel_axis"] == 0
    assert viewer.kwargs["luts"][0] == {"name": "PC", "cmap": "gray", "clims": NDV_CLIMS}
    assert [viewer.kwargs["luts"][c]["name"] for c in (1, 2)] == ["open down", "open up"]
    assert viewer.renders == 1
    assert notebook.text("candidate") == "<b>f0</b> — 1 candidate lane"
    (lane,) = detector.detect_fov(images["f0"], 40, 300, 5000, 20, "f0")[3]
    channel, other = (1, 2) if lane.orientation == 1 else (2, 1)
    lines = np.flatnonzero(viewer.data[channel].any(axis=1))
    np.testing.assert_array_equal(lines, np.arange(lane.y_center - 2, lane.y_center + 3))
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
    expected.detect_lanes(sigma=3, distance=50, height=3000, orientation_window_um=20,
                          orientation="down", save_diagnostics=False)
    assert {f: [asdict(l) for l in ls] for f, ls in detector.lanes.items()} == {
        f: [asdict(l) for l in ls] for f, ls in expected.lanes.items()}
    assert detector.detection_params == expected.detection_params
    assert notebook.text("Applied") == "<b>Applied</b> — 2 lanes across 2 FOVs"


# ----------------------------------------------------------------------------
# trenches
# ----------------------------------------------------------------------------
@pytest.fixture
def trench_setup(tmp_path):
    image = np.full((H, W), 10.0, dtype=np.float32)
    image[30:90, 20:150:40] = 40.0      # lane 0's trenches, above it (open down)
    image[120:180, 30:150:30] = 40.0    # lane 1's trenches, below it (open up)
    experiment = _experiment(tmp_path, ["f0"])
    lanes = SimpleNamespace(lanes={"f0": [LaneInfo(100, 1, "f0", 0), LaneInfo(110, -1, "f0", 1)]})
    return experiment, _registrator({"f0": image}), lanes, image


def _trench_widget(notebook, setup):
    experiment, registrator, lanes, _ = setup
    detector = TrenchDetector(experiment, registrator, lanes, None, detection_channel="PC")
    detector.interactive_detect_trenches()
    for name, value in [("trench width", 6), ("trench length", 60), ("bottom offset", 10),
                        ("sigma", 0.0), ("distance", 5), ("prominence", 1)]:
        notebook.control(name).value = value
    return detector


def test_trench_widget_previews_outlines_and_lane_controls(notebook, trench_setup):
    detector = _trench_widget(notebook, trench_setup)
    (viewer,) = notebook.viewers
    assert viewer.kwargs["luts"][1]["name"] == "candidate trenches"
    expected, _, _ = detector.detect_fov(
        trench_setup[3], "f0", sigma=0.0, distance=5, prominence=1, height=0, trench_width=6,
        trench_length=60, trench_bottom_offset=10)
    assert notebook.text("candidate") == f"<b>f0</b> — {len(expected)} candidate trenches"
    outline = viewer.data[1]
    for trench in expected:
        assert outline[trench.y_top, trench.x_left:trench.x_right].all()
        assert outline[trench.y_top:trench.y_bottom, trench.x_left].all()
    assert outline.sum() > 0 and not outline[95:115].any()

    # each lane's controls follow the global ones until set on their own
    assert notebook.control("L0 sigma").value == notebook.control("L1 sigma").value == 0.0
    notebook.control("L1 prominence").value = 7
    notebook.control("prominence").value = 2
    assert notebook.control("L0 prominence").value == 2
    assert notebook.control("L1 prominence").value == 7

    notebook.control("method").value = "periodic"
    assert notebook.control("distance").disabled and notebook.control("prominence").disabled
    assert not notebook.control("min symmetry").disabled


def test_trench_widget_applies_lane_overrides(notebook, trench_setup):
    detector = _trench_widget(notebook, trench_setup)
    notebook.control("L1 prominence").value = 7
    notebook.control("Apply to all FOVs").click()

    experiment, registrator, lanes, _ = trench_setup
    expected = TrenchDetector(experiment, registrator, lanes, None, detection_channel="PC")
    expected.detect_trenches(sigma=0.0, distance=5, prominence=1, height=0, trench_width=6,
                             trench_length=60, trench_bottom_offset=10,
                             lane_params={1: {"prominence": 7}}, save_diagnostics=False)
    assert detector.get_trench_table().equals(expected.get_trench_table())
    assert detector.detection_params == expected.detection_params
    assert detector.detection_params["lane_params"] == {1: {"prominence": 7}}
    assert notebook.text("Applied") == (
        f"<b>Applied</b> — {expected.n_trenches} trenches across 1 FOVs")


# ----------------------------------------------------------------------------
# segmentation
# ----------------------------------------------------------------------------
@pytest.fixture
def segmenter(tmp_path, monkeypatch):
    model = _FakeModel()
    monkeypatch.setattr(Segmenter, "_get_model", lambda self: model)
    return Segmenter(make_experiment(tmp_path), channel="mCherry", gpu=False)


def test_segmentation_widget_segments_the_chosen_trench(notebook, segmenter):
    segmenter.interactive_segment()
    (viewer,) = notebook.viewers
    assert viewer.kwargs["channel_axis"] == 1
    stack = segmenter.read(10)
    bounds = trench_bounds(stack, segmenter.params["floor"])
    masks, _ = segmenter.segment_stack(stack, bounds, "S1", segmenter.params)
    np.testing.assert_array_equal(viewer.data, label_overlay(stack, masks))
    assert notebook.text("Segmented").startswith("Segmented 4 frames in")
    assert notebook.text("Trench").startswith("<b>Trench 10</b> (f0), S1 — up to 2 cells")
    assert notebook.text("sliders") == "uses the sliders as they are"

    notebook.control("hypothesis").value = "S2"
    assert notebook.text("overrides") == "overrides: " + ", ".join(
        f"{k} {v}" for k, v in HYPOTHESES["S2"].items())
    notebook.control("trench (index)").value = 2
    assert notebook.text("Trench").startswith("<b>Trench 12</b> (f1), S2")
    assert viewer.renders == 3


def test_segmentation_widget_apply_keeps_the_settings(notebook, segmenter):
    segmenter.segment(trenches=[10], show_progress=False)
    segmenter.interactive_segment()
    notebook.control("gain").value = 2.0
    notebook.control("Apply").click()
    assert segmenter.params["gain"] == 2.0
    assert set(segmenter.params) == {"floor", "floor_sigma", "gain", "rescale",
                                     "mask_threshold", "flow_threshold", "min_area",
                                     "min_contrast"}
    assert notebook.text("Applied").startswith(
        "<b>Applied</b> — base settings kept for segment()<br>1 trenches already have "
        "results with the previous settings")


def test_segmentation_view_shows_written_labels_as_outlines(notebook, segmenter):
    segmenter.segment(trenches=[10], show_progress=False)
    viewer = segmenter.view()
    assert viewer.kwargs["channel_axis"] == 3
    wrapper = viewer.data
    assert wrapper.dims == ("Hypothesis", "Trench", "T", "C", "Y", "X")
    every = {axis: slice(None) for axis in range(2, 6)}
    shown = wrapper.isel({0: 1, 1: slice(0, 2), **every})
    root = eg.open_experiment(segmenter.store_path)
    frames = eg.image(root, 10)[:, segmenter.channel_index]
    labels = eg.segmentation(root, 10, segmenter.name)[1][1][...]
    np.testing.assert_array_equal(shown[0, 0], label_overlay(frames, labels))
    assert not shown[0, 1, :, 1].any()  # trench 11 is not segmented: no outlines
