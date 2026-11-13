"""``pymmm.widgets.segmentation``: the segmentation widget and the label viewer, driven
without a browser (the ``notebook`` fixture), on a fake model."""

from __future__ import annotations

import numpy as np
import pytest
from pymmm.steps.segmentation import HYPOTHESES, Segmenter, trench_bounds
from pymmm.store import experiment_group as eg
from pymmm.widgets.segmentation import label_overlay

from fake_experiment import make_experiment

pytestmark = [
    pytest.mark.notebook,
    pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive"),
]


@pytest.fixture
def segmenter(tmp_path, fake_model):
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
    assert notebook.text("Trench").startswith(
        "<b>Trench 10</b> (f0), S1 — up to 2 cells"
    )
    assert (
        notebook.text("sliders")
        == "model bact_fluor_omni; uses the sliders as they are"
    )

    notebook.control("hypothesis").value = "S2"
    assert notebook.text(
        "overrides"
    ) == "model bact_fluor_omni; overrides: " + ", ".join(
        f"{k} {v}" for k, v in HYPOTHESES["S2"].items()
    )
    notebook.control("trench (index)").value = 2
    assert notebook.text("Trench").startswith("<b>Trench 12</b> (f1), S2")
    assert viewer.renders == 3


def test_segmentation_widget_apply_keeps_the_settings(notebook, segmenter):
    segmenter.segment(trenches=[10], show_progress=False)
    segmenter.interactive_segment()
    notebook.control("gain").value = 2.0
    notebook.control("Apply").click()
    assert segmenter.params["gain"] == 2.0
    assert set(segmenter.params) == {
        "floor",
        "floor_sigma",
        "gain",
        "rescale",
        "masks",
        "mask_threshold",
        "flow_threshold",
        "affinity_angle",
        "min_area",
        "min_contrast",
    }
    assert notebook.text("Applied").startswith(
        "<b>Applied</b> — base settings kept for segment()<br>1 trenches already have "
        "results with the previous settings"
    )


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
