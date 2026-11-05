"""PyMMM's ndv viewers, real ones, in a real Jupyter kernel, sent the events a browser
sends (``browser_events``). The other widget tests use the ``notebook`` fixture, whose
viewers are fakes, so only these see how a viewer handles a mouse or key event.

What is still checked by hand: how a viewer looks and responds in JupyterLab (the
browser side, jupyter_rfb's ``renderview.js``)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.notebook
nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")
pytest.importorskip("ipykernel")

TESTS = Path(__file__).resolve().parent

VIEWERS = '''
import json, sys
from pathlib import Path
sys.path.insert(0, {tests!r})
assert sys.prefix == {prefix!r}, sys.prefix
import numpy as np
from ipywidgets.widgets.widget import _instances
from jupyter_rfb import RemoteFrameBuffer
from browser_events import canvases, exercise
from fake_experiment import fov_experiment, fov_registrator, lane_image
from pymmm.steps.lanes import LaneDetector
from pymmm.widgets import imshow

report = {{}}
image = np.random.default_rng(0).integers(0, 1000, (3, 2, 64, 48)).astype(np.uint16)
report["imshow"] = [exercise(c) for c in canvases(imshow(image).widget())]

images = {{"f0": lane_image(120), "f1": lane_image(110, shift=5)}}
detector = LaneDetector(fov_experiment(Path({tmp!r}), images), fov_registrator(images), None,
                        detection_channel="PC")
known = set(_instances)
detector.interactive_detect_lanes()
report["lane widget"] = [exercise(w) for k, w in list(_instances.items())
                         if k not in known and isinstance(w, RemoteFrameBuffer)]
print(json.dumps(report))
'''


def run_in_kernel(source: str) -> str:
    """Run ``source`` in a fresh kernel; its stdout (a failure fails the test)."""
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)])
    nbclient.NotebookClient(nb, timeout=300, kernel_name="python3").execute()
    return "".join(o.text for o in nb.cells[0].outputs
                   if o.output_type == "stream" and o.name == "stdout")


def test_viewers_handle_browser_events_and_draw(tmp_path):
    stdout = run_in_kernel(VIEWERS.format(tests=str(TESTS), prefix=sys.prefix,
                                          tmp=str(tmp_path)))
    report = json.loads(stdout.strip().splitlines()[-1])
    for viewer, drawn in report.items():
        assert drawn, f"{viewer}: no canvas"
        for canvas in drawn:
            # ndv on vispy fails every event with KeyError: 'event_type' (NDV_CANVAS)
            assert canvas["canvas"].endswith("JupyterRenderCanvas"), (viewer, canvas)
            assert canvas["errors"] == [], (viewer, canvas)
            assert canvas["frame"] == [300, 400, 4], (viewer, canvas)
            assert canvas["spread"] > 0, (viewer, canvas)  # the image, not a blank frame
