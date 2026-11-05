"""Golden tests of the detection widgets on real data, driven without a browser (the
``notebook`` fixture): set to the experiment's settings, Apply gives the golden lanes
and trenches, and the previews are pinned."""

from __future__ import annotations

import pytest
from golden_io import check, digest
from real_data import LANES, TRENCHES
from test_golden_pipeline import lanes_summary, trenches_summary

from pymmm import LaneDetector, TrenchDetector

pytestmark = [pytest.mark.golden, pytest.mark.notebook,
              pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")]

LANE_CONTROLS = {"sigma": "sigma", "distance": "distance", "height": "height",
                 "orientation_window_um": "orient. window (µm)", "orientation": "orientation"}
TRENCH_CONTROLS = {"method": "method", "sigma": "sigma", "distance": "distance",
                   "prominence": "prominence", "trench_width": "trench width",
                   "trench_length": "trench length", "trench_bottom_offset": "bottom offset",
                   "expected_pitch": "expected pitch (0 = auto)",
                   "min_symmetry": "min symmetry"}


def test_detection_widgets_reproduce_the_golden_detection(notebook, pipeline):
    lanes = LaneDetector(pipeline.exp, pipeline.reg, pipeline.store, detection_channel="PC")
    lanes.interactive_detect_lanes()
    for key, description in LANE_CONTROLS.items():
        notebook.control(description).value = LANES[key]
    lane_preview = {"text": notebook.text("candidate"),
                    "image": digest(notebook.viewers[0].data)}
    notebook.control("Apply to all FOVs").click()
    check("lanes", lanes_summary(lanes))

    notebook.shown.clear()
    trenches = TrenchDetector(pipeline.exp, pipeline.reg, lanes, pipeline.store,
                              detection_channel="PC")
    trenches.interactive_detect_trenches()
    for key, description in TRENCH_CONTROLS.items():
        notebook.control(description).value = TRENCHES[key]
    trench_preview = {"text": notebook.text("candidate"),
                      "image": digest(notebook.viewers[1].data)}
    notebook.control("Apply to all FOVs").click()
    check("trenches", trenches_summary(trenches))

    check("widget_previews", {"lanes": lane_preview, "trenches": trench_preview})
