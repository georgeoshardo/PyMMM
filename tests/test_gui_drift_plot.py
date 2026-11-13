"""The Register tab's drift plot (pymmm.gui.drift_plot), clicked as a person would."""

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


def _click(qapp, view, plot, x, y):
    """A left click on ``view``'s viewport where ``plot`` shows (x, y)."""
    import pyqtgraph as pg
    from qtpy.QtCore import QPoint, Qt
    from qtpy.QtTest import QTest

    at = view.mapFromScene(plot.vb.mapViewToScene(pg.Point(x, y)))
    QTest.mouseClick(
        view.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(int(at.x()), int(at.y()))
    )
    _settle(qapp, 3)


@pytest.fixture
def plot(qapp):
    from pymmm.gui.drift_plot import DriftPlot

    rng = np.random.default_rng(0)
    widget = DriftPlot()
    widget.resize(420, 330)
    widget.show()
    for fov in ("A01", "A02", "A03"):
        widget.set_drift(fov, np.cumsum(rng.normal(0, 1, (40, 2)), axis=0))
    widget.show_fov("A02")
    _settle(qapp)
    yield widget
    widget.close()


def test_a_click_on_the_time_axis_picks_that_frame(qapp, plot):
    clicked = []
    plot.frame_clicked.connect(clicked.append)
    current = plot._shifts["A02"]
    _click(qapp, plot.time_view, plot.y_plot, 17, current[17, 1])
    _click(qapp, plot.time_view, plot.x_plot, 33.2, current[33, 0])
    assert clicked == [17, 33]


def test_a_click_on_the_path_picks_the_nearest_frame_of_the_fov_shown(qapp, plot):
    clicked = []
    plot.frame_clicked.connect(clicked.append)
    plot.set_mode("path")
    _settle(qapp)
    current = plot._shifts["A02"]
    _click(qapp, plot.path_view, plot.path_plot, *current[25])
    assert clicked == [int(np.argmin(np.hypot(*(current - current[25]).T)))]


def test_the_frame_shown_is_marked_on_the_fov_in_colour(qapp, plot):
    plot.show_frame(12)
    current = plot._shifts["A02"]
    dots = [dot.getData() for _, _, dot in plot._markers]
    assert [(x[0], y[0]) for x, y in dots] == [
        (12, current[12, 0]),
        (12, current[12, 1]),
    ]
    plot.show_fov("B01")  # not registered: nothing marked
    assert all(len(dot.getData()[0]) == 0 for _, _, dot in plot._markers)
    plot.clear()
    assert plot.fovs == []
