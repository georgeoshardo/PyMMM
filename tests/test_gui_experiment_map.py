"""The Experiment tab's map (pymmm.gui.experiment_map), clicked as a person would."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = pytest.mark.view


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _settle(qapp, n=10):
    for _ in range(n):
        qapp.processEvents()


def _box(x, y, w=250.0, h=125.0):
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])


@pytest.fixture
def plan(qapp):
    from pymmm.gui.experiment_map import ExperimentMap

    widget = ExperimentMap()
    widget.resize(420, 400)
    widget.show()
    outlines = {f"A{k:02d}": _box(253.0 * k, 0.0) for k in range(1, 9)}
    widget.set_fovs(outlines, [f"A{k:02d}" for k in range(1, 8)])  # A08 left out
    _settle(qapp)
    yield widget
    widget.close()


def test_a_click_on_a_fov_picks_it(qapp, plan):
    import pyqtgraph as pg
    from qtpy.QtCore import QPoint, Qt
    from qtpy.QtTest import QTest

    clicked = []
    plan.fov_clicked.connect(clicked.append)
    for name in ("A03", "A08"):  # A08 is not in the experiment
        centre = plan._outlines[name].mean(axis=0)
        at = plan.view.mapFromScene(plan.plot.vb.mapViewToScene(pg.Point(*centre)))
        QTest.mouseClick(
            plan.view.viewport(),
            Qt.MouseButton.LeftButton,
            pos=QPoint(int(at.x()), int(at.y())),
        )
        _settle(qapp, 3)
    assert clicked == ["A03"]


def test_lanes_trenches_and_names_show_only_zoomed_in(qapp, plan):
    plan.set_lanes([np.array([[253.0, 100.0], [503.0, 100.0]])], [1])
    plan.set_trenches([_box(260.0, 20.0, 9.0, 70.0)])
    plan.plot.autoRange()
    _settle(qapp)
    assert not plan.detailed
    assert not plan._trenches.isVisible() and not plan._texts["A01"].isVisible()
    plan.plot.setRange(xRange=(200, 560), yRange=(-20, 150), padding=0)
    _settle(qapp)
    assert plan.detailed
    assert plan._trenches.isVisible() and plan._texts["A01"].isVisible()
    assert plan._lanes[1].isVisible()


def test_flagged_fovs_are_filled_red(qapp, plan):
    from pymmm.gui.experiment_map import FILL_COLOUR, FLAG_COLOUR

    plan.set_flagged({"A02", "A08"})  # A08 is not in the experiment
    colours = {
        name: item.brush().color().getRgb()
        for name, item in zip(plan.fovs, plan._fills)
    }
    assert colours["A02"] == FLAG_COLOUR
    assert colours["A01"] == FILL_COLOUR
    assert plan._flagged == {"A02"}
