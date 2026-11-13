"""Golden tests of the extraction dock (pymmm.gui.extraction) on real data: driven as a
person would, with the experiment's settings, it saves the golden registration, lanes
and trenches, its preview is the extraction, and reopening it loads what it saved.

The dock is a real Qt widget on a napari ViewerModel (the viewer without its canvas);
Qt draws offscreen.
"""

from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np
import pytest
import zarr
from golden_io import check
from pymmm import CompanionStore, Registrator
from pymmm.parallel import Cancelled
from real_data import FOVS, LANES, ND2, REGISTRATION, TIMES, TRENCHES, open_nd2
from test_golden_pipeline import lanes_summary, registration_summary, trenches_summary

pytestmark = [
    pytest.mark.golden,
    pytest.mark.view,
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]


SLOT_ERRORS: list[BaseException] = []


@pytest.fixture(scope="module")
def qapp():
    """Qt, offscreen. An exception in a Qt slot is kept and raised by :func:`_wait`
    (by default PyQt aborts the process on one)."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    previous = sys.excepthook
    sys.excepthook = lambda kind, error, trace: SLOT_ERRORS.append(error)
    yield QApplication.instance() or QApplication([])
    sys.excepthook = previous


def _wait(qapp, done, timeout: float = 300) -> None:
    start = time.time()
    while not done():
        qapp.processEvents()
        time.sleep(0.01)
        if SLOT_ERRORS:
            raise SLOT_ERRORS.pop()
        assert time.time() - start < timeout, "timed out"
    qapp.processEvents()
    if SLOT_ERRORS:
        raise SLOT_ERRORS.pop()


def coordinate_grid(viewer):
    from pymmm.gui.extraction import coordinate_grid

    return coordinate_grid(viewer)


def _dock(qapp, nd2):
    from napari.components import ViewerModel
    from pymmm.gui.extraction import ExtractionDock

    viewer = ViewerModel()
    return viewer, ExtractionDock(viewer, experiment_file=nd2)


def _choose(box, value) -> None:
    index = box.findData(value)
    assert index >= 0, value
    box.setCurrentIndex(index)


def test_dock_reproduces_the_golden_detection_and_preview(
    qapp, scratch_folder, pipeline, monkeypatch
):
    nd2 = scratch_folder("dock") / ND2.name
    viewer, dock = _dock(qapp, nd2)
    assert dock.store.path.parent == nd2.parent  # beside the symlink, not the ND2

    dock.keep_edit.setText(", ".join(FOVS))
    dock.times_edit.setText(f"{TIMES[0]}:{TIMES[1]}")
    dock.open_button.click()
    assert dock.experiment.fov_names == FOVS
    assert dock.experiment.n_timepoints == TIMES[1] - TIMES[0]
    assert [layer.name for layer in viewer.layers][:2] == [
        "PC raw",
        "PC (rotated, not registered)",
    ]
    assert viewer.dims.axis_labels == ("T", "P", "y", "x")
    grid = coordinate_grid(viewer)
    if grid is not None:  # napari 0.10 and later: on, and the dock's box turns it off
        assert grid.visible and dock.grid_box.isChecked()
        dock.grid_box.setChecked(False)
        assert not grid.visible
        grid.visible = True  # View → Coordinate Grid
        assert dock.grid_box.isChecked()

    # Register, in a process pool
    _choose(dock.reg_channel, REGISTRATION["registration_channel"])
    dock.rotation.setValue(REGISTRATION["rotation"])
    dock.reg_y_edit.setText("{}:{}".format(*REGISTRATION["roi"]["y"]))
    _choose(dock.mode, REGISTRATION["mode"])
    _choose(dock.backend, REGISTRATION["backend"])
    dock.downsample.setValue(REGISTRATION["downsample"])
    dock.mean_n_frames.setValue(REGISTRATION["mean_n_frames"])
    _choose(dock.mean_from, REGISTRATION["mean_from"])
    dock.batch_size.setValue(REGISTRATION["batch_size"])
    dock.workers.setValue(2)
    dock.register_button.click()
    _wait(qapp, lambda: dock._worker is None)
    check("registration_first", registration_summary(dock.registrator))
    diagnostics = nd2.parent / f"{nd2.stem}_diagnostics"
    assert sorted(p.name for p in (diagnostics / "drift").iterdir()) == [
        f"drift_{fov}.png" for fov in FOVS
    ]
    assert dock.reg_progress.value() == len(FOVS)
    sizes = pipeline.exp.data.sizes
    assert viewer.layers["PC"].data.shape == (10, 2, sizes["Y"], sizes["X"])
    # the registered layer is the registered frames, at the frame and FOV shown
    frame = np.asarray(viewer.layers["PC"].data[3, 1])
    expected = pipeline.reg.get_stabilized_data(channel="PC", fov=FOVS[1]).isel(T=3)
    np.testing.assert_array_equal(frame, expected.values)

    # Lanes: the preview on the FOV shown, then every FOV
    viewer.dims.current_step = (0, 1, 0, 0)
    dock.tabs.setCurrentIndex(2)
    dock.lane_sigma.setValue(LANES["sigma"])
    dock.lane_distance.setValue(LANES["distance"])
    dock.lane_height.setValue(LANES["height"])
    dock.lane_window.setValue(LANES["orientation_window_um"])
    _choose(dock.lane_orientation, LANES["orientation"])
    dock.update_lanes()
    _wait(qapp, lambda: FOVS[1] in dock.lane_status.text())
    lane_y = pipeline.lanes.lanes[FOVS[1]][0].y_center
    assert [round(line[0, 0]) for line in viewer.layers["lanes"].data] == [lane_y]
    dock.lane_apply.click()
    _wait(qapp, lambda: dock._worker is None)
    check("lanes", lanes_summary(dock.lane_detector))
    assert {p.name for p in (diagnostics / "lanes").iterdir()} == {
        f"lanes_{fov}.png" for fov in FOVS
    }

    # Trenches
    dock.tabs.setCurrentIndex(3)
    _choose(dock.trench_method, TRENCHES["method"])
    dock.trench_sigma.setValue(TRENCHES["sigma"])
    dock.trench_distance.setValue(TRENCHES["distance"])
    dock.trench_prominence.setValue(TRENCHES["prominence"])
    dock.trench_pitch.setValue(TRENCHES["expected_pitch"])
    dock.trench_symmetry.setValue(TRENCHES["min_symmetry"])
    dock.trench_width.setValue(TRENCHES["trench_width"])
    dock.trench_length.setValue(TRENCHES["trench_length"])
    dock.trench_offset.setValue(TRENCHES["trench_bottom_offset"])
    dock.update_trenches()
    _wait(qapp, lambda: f"{FOVS[1]}</b>:" in dock.trench_status.text())
    boxes = viewer.layers["trenches"].data
    assert len(boxes) == len(pipeline.trenches.trenches[FOVS[1]])
    dock.trench_apply.click()
    _wait(qapp, lambda: dock._worker is None)
    check("trenches", trenches_summary(dock.trench_detector))
    tabs = [dock.tabs.tabText(i) for i in range(dock.tabs.count())]
    assert tabs == ["Experiment", "Register ✓", "Lanes ✓", "Trenches ✓", "Preview"]
    assert dock.lineage["trench_detection"] == "current"
    assert "not extracted yet" in dock.extraction_info.text()
    # a setting changed and not applied: marked, and marked back when set back
    dock.trench_width.setValue(TRENCHES["trench_width"] + 2)
    assert dock.tabs.tabText(3) == "Trenches *"
    dock.trench_width.setValue(TRENCHES["trench_width"])
    assert dock.tabs.tabText(3) == "Trenches ✓"
    dock.times_edit.setText("20:29")
    assert dock.tabs.tabText(0) == "Experiment *"
    dock.times_edit.setText(f"{TIMES[0]}:{TIMES[1]}")
    assert dock.tabs.tabText(0) == "Experiment"
    assert {p.name for p in (diagnostics / "trenches").iterdir()} == {
        f"trenches_{fov}.png" for fov in FOVS
    }

    # Preview: the trench images extract() wrote
    from napari.components import ViewerModel

    # as a person opens it, the button (a new window: here a viewer without a canvas,
    # as offscreen Qt has no OpenGL)
    monkeypatch.setattr("napari.Viewer", lambda title: ViewerModel(title=title))
    dock.tabs.setCurrentIndex(4)
    dock.preview_button.click()
    assert dock.preview_viewer.title.endswith("trench preview")
    if coordinate_grid(dock.preview_viewer) is not None:
        assert coordinate_grid(dock.preview_viewer).visible
    preview = dock.open_preview(ViewerModel())
    assert preview.dims.axis_labels == ("trench", "t", "y", "x")
    root = zarr.open_group(str(pipeline.group), mode="r")
    layer = preview.layers["mCherry"]
    assert layer.data.trench_ids == root["trenches/trench"][:].tolist()
    for k in (0, len(layer.data) - 1):
        trench = layer.data.trench_ids[k]
        np.testing.assert_array_equal(
            layer.data[k], root[f"trench_{trench:04d}/0"][:, 1]
        )

    # Units: µm scales the y and x of every layer of both viewers, shapes too
    size = pipeline.exp.pixel_size_um
    viewer.camera.center, viewer.camera.zoom = (0.0, 300.0, 1000.0), 2.0
    dock.units.setCurrentIndex(dock.units.findData("micrometer"))
    # the view stays on the same pixels, at the same size on screen
    np.testing.assert_allclose(viewer.camera.center[-2:], (300 * size, 1000 * size))
    np.testing.assert_allclose(viewer.camera.zoom, 2.0 / size)
    assert tuple(viewer.layers["PC"].scale) == (1, 1, size, size)
    assert tuple(viewer.layers["trenches"].scale) == (size, size)
    assert tuple(dock.preview_viewer.layers["PC"].scale) == (1, 1, size, size)
    assert tuple(viewer.layers["PC"].units)[-1] == "micrometer"
    dock.units.setCurrentIndex(dock.units.findData("pixel"))
    assert tuple(viewer.layers["trenches"].scale) == (1, 1)
    np.testing.assert_allclose(viewer.camera.center[-2:], (300, 1000))
    np.testing.assert_allclose(viewer.camera.zoom, 2.0)

    # Reopened, the dock loads all three and fills its tabs from them
    viewer2, dock2 = _dock(qapp, nd2)
    assert dock2.keep_edit.text() == ""
    assert dock2.experiment.fov_names == FOVS
    assert dock2.times_edit.text() == f"{TIMES[0]}:{TIMES[1]}"
    assert dock2.reg_y_edit.text() == "{}:{}".format(*REGISTRATION["roi"]["y"])
    assert dock2.registrator is not None and dock2.registrator.is_computed
    assert dock2.trench_detector.n_trenches == pipeline.trenches.n_trenches
    assert dock2.trench_width.value() == TRENCHES["trench_width"]
    assert "loaded the saved registration, lanes, trenches" in dock2.status.text()
    assert viewer2.layers["PC"].name == "PC"  # registered, not only rotated

    # A changed setting shows the frames only rotated, locks the later steps and says
    # to register again; set back, they open again
    dock2.rotation.setValue(0.5)
    dock2._on_registration_settings()
    assert viewer2.layers[1].name == "PC (rotated, not registered)"
    assert "Register again" in dock2.reg_status.text()
    assert dock2.tabs.tabText(1) == "Register *"
    assert not any(dock2.tabs.isTabEnabled(i) for i in (2, 3, 4))
    dock2.rotation.setValue(REGISTRATION["rotation"])
    dock2._on_registration_settings()
    assert all(dock2.tabs.isTabEnabled(i) for i in (2, 3, 4))

    # Registered again elsewhere without clearing what followed (a script): the dock
    # loads the registration but not the lanes and trenches made from the old one
    other = Registrator(
        dock2.experiment, dock2.store, **{**REGISTRATION, "rotation": 1.0}
    )
    other.compute_tmats(n_jobs=1)
    other.save()
    _, dock3 = _dock(qapp, nd2)
    assert dock3.registrator is not None and dock3.registrator.rotation == 1.0
    assert dock3.lane_detector is None and dock3.trench_detector is None
    assert "were not loaded" in dock3.status.text()
    assert dock3.lineage["lane_detection"] == "stale"
    assert not dock3.tabs.isTabEnabled(3)


def test_preview_array_is_the_extraction_and_cuts_a_frame_once(pipeline, monkeypatch):
    from pymmm.steps import extraction

    root = zarr.open_group(str(pipeline.group), mode="r")
    array = pipeline.extractor.preview_array("PC")
    assert array.shape == (len(array.trench_ids), 10, 655, 94)
    np.testing.assert_array_equal(
        np.asarray(array),
        np.stack([root[f"trench_{t:04d}/0"][:, 0] for t in array.trench_ids]),
    )
    calls = []
    cut = extraction._crop_frames
    monkeypatch.setattr(
        extraction, "_crop_frames", lambda *args: calls.append(1) or cut(*args)
    )
    array = pipeline.extractor.preview_array("PC")
    same_fov = [k for k, f in enumerate(array.fov_names) if f == array.fov_names[0]]
    for k in same_fov:
        array[k, 4]
    assert len(calls) == 1
    assert array[same_fov[0], 2:5, :3, 1].shape == (3, 3)
    assert array[..., 0, 0].shape == array.shape[:2]


def test_registration_reports_progress_and_can_be_cancelled(pipeline):
    reg = Registrator(pipeline.exp, pipeline.store, **REGISTRATION)
    seen = []
    reg.compute_tmats(n_jobs=1, progress=lambda done, total: seen.append((done, total)))
    assert seen == [(1, 2), (2, 2)]

    cancel = threading.Event()
    cancel.set()
    fresh = Registrator(pipeline.exp, pipeline.store, **REGISTRATION)
    with pytest.raises(Cancelled):
        fresh.compute_tmats(n_jobs=2, cancel=cancel)
    assert not fresh.is_computed


def test_saved_registration_matches_its_settings_only(pipeline):
    assert Registrator(pipeline.exp, pipeline.store, **REGISTRATION).matches_saved()
    rotated = Registrator(
        pipeline.exp, pipeline.store, **{**REGISTRATION, "rotation": 1.0}
    )
    assert not rotated.matches_saved()
    other_frames = open_nd2(pipeline.folder).select_times(TIMES[0], TIMES[1] - 1)
    assert not Registrator(other_frames, pipeline.store, **REGISTRATION).matches_saved()
    assert not Registrator(
        pipeline.exp,
        CompanionStore(pipeline.folder / "empty.pymmm.zarr"),
        **REGISTRATION,
    ).matches_saved()


def test_rotation_only_needs_no_transforms(pipeline):
    reg = Registrator(pipeline.exp, pipeline.store, **REGISTRATION)
    assert not reg.is_computed
    unrotated = reg.get_stabilized_data(channel="PC", fov=FOVS[0], drift=False)
    raw = pipeline.exp.data.sel(C="PC", P=FOVS[0])
    np.testing.assert_array_equal(unrotated.isel(T=2).values, raw.isel(T=2).values)
    tilted = Registrator(
        pipeline.exp, pipeline.store, **{**REGISTRATION, "rotation": 2.0}
    )
    frame = tilted.get_stabilized_data(channel="PC", fov=FOVS[0], drift=False).isel(T=2)
    assert not np.array_equal(frame.values, raw.isel(T=2).values)
