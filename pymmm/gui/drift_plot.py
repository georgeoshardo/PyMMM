"""The Register tab's drift plot: every registered FOV's shift, the FOV on the P slider
in colour and the others in grey, with the frame on the T slider marked.

Two views: x and y shift over time (two plots sharing the frame axis), or the xy path
(y shift against x shift, y down as in the image). A click picks a frame: the nearest
frame on the time axis, or the nearest point of the coloured path.
"""

from __future__ import annotations

import numpy as np

# qtpy first, so pyqtgraph uses the Qt binding napari already loaded
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

# isort: split
import pyqtgraph as pg

X_COLOUR = "#f28e2b"
Y_COLOUR = "#4e9fd6"
PATH_COLOUR = "#e8c547"
OTHER_COLOUR = (150, 150, 150, 70)
OTHER_PATH_COLOUR = (150, 150, 150, 35)  # paths overlap more than time traces
MARKER_COLOUR = "#ffffff"


class DriftPlot(QWidget):
    """x and y shift per frame for each FOV, as they are registered.

    Feed it with :meth:`set_drift` (one FOV at a time, as each finishes), choose what
    stands out with :meth:`show_fov` and :meth:`show_frame`; :attr:`frame_clicked`
    carries the frame a click picked.
    """

    frame_clicked = Signal(int)

    MODES = (("x and y over time", "time"), ("xy path", "path"))

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._shifts: dict[str, np.ndarray] = {}
        self._fov: str | None = None
        self._frame = 0

        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.mode = QComboBox()
        for label, value in self.MODES:
            self.mode.addItem(label, value)
        self.mode.currentIndexChanged.connect(self._on_mode)
        top = QHBoxLayout()
        top.addWidget(self.note, 1)
        top.addWidget(self.mode)

        self.time_view = pg.GraphicsLayoutWidget()
        self.time_view.setBackground(None)
        self.x_plot = self.time_view.addPlot(row=0, col=0)
        self.y_plot = self.time_view.addPlot(row=1, col=0)
        self.y_plot.setXLink(self.x_plot)
        self.x_plot.setLabel("left", "x shift (px)")
        self.y_plot.setLabel("left", "y shift (px)")
        self.y_plot.setLabel("bottom", "frame")
        self.x_plot.getAxis("left").setWidth(48)
        self.y_plot.getAxis("left").setWidth(48)

        self.path_view = pg.GraphicsLayoutWidget()
        self.path_view.setBackground(None)
        self.path_plot = self.path_view.addPlot()
        self.path_plot.setAspectLocked(True)
        self.path_plot.invertY(True)  # y down, as in the image
        self.path_plot.setLabel("bottom", "x shift (px)")
        self.path_plot.setLabel("left", "y shift (px)")

        for view in (self.time_view, self.path_view):
            view.scene().sigMouseClicked.connect(self._clicked)
        self.views = QStackedWidget()
        self.views.addWidget(self.time_view)
        self.views.addWidget(self.path_view)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(top)
        layout.addWidget(self.views)
        self.setMinimumHeight(280)
        self._redraw()

    # ------------------------------------------------------------------

    @property
    def fovs(self) -> list[str]:
        """The FOVs drawn, in the order they came."""
        return list(self._shifts)

    def clear(self) -> None:
        """No FOVs drawn."""
        self._shifts.clear()
        self._redraw()

    def set_drift(self, fov: str, shifts: np.ndarray) -> None:
        """Draw (or redraw) one FOV: ``shifts`` is (frames, 2), x then y, in px."""
        self._shifts[str(fov)] = np.asarray(shifts, dtype=float)
        self._redraw()

    def show_fov(self, fov: str | None) -> None:
        """Colour this FOV; the others stay grey."""
        if fov != self._fov:
            self._fov = fov
            self._redraw()

    def show_frame(self, frame: int) -> None:
        """Mark this frame on the coloured FOV."""
        self._frame = int(frame)
        self._draw_marker()

    def set_mode(self, mode: str) -> None:
        """``"time"`` (x and y over time) or ``"path"`` (the xy path)."""
        self.mode.setCurrentIndex(self.mode.findData(mode))

    # ------------------------------------------------------------------

    def _on_mode(self) -> None:
        self.views.setCurrentIndex(0 if self.mode.currentData() == "time" else 1)
        self._redraw()

    def _current(self) -> np.ndarray | None:
        return self._shifts.get(self._fov) if self._fov is not None else None

    def _redraw(self) -> None:
        for plot in (self.x_plot, self.y_plot, self.path_plot):
            plot.clear()
        self._markers = []
        others = [s for fov, s in self._shifts.items() if fov != self._fov]
        current = self._current()
        grey = pg.mkPen(OTHER_COLOUR, width=1)
        if self.mode.currentData() == "time":
            for shifts in others:
                frames = np.arange(len(shifts))
                self.x_plot.plot(frames, shifts[:, 0], pen=grey)
                self.y_plot.plot(frames, shifts[:, 1], pen=grey)
            if current is not None:
                frames = np.arange(len(current))
                self.x_plot.plot(frames, current[:, 0], pen=pg.mkPen(X_COLOUR, width=2))
                self.y_plot.plot(frames, current[:, 1], pen=pg.mkPen(Y_COLOUR, width=2))
            for plot in (self.x_plot, self.y_plot):
                line = pg.InfiniteLine(
                    angle=90, pen=pg.mkPen(MARKER_COLOUR, width=1, style=Qt.DashLine)
                )
                dot = pg.ScatterPlotItem(
                    size=9, brush=pg.mkBrush(MARKER_COLOUR), pen=pg.mkPen("k")
                )
                plot.addItem(line)
                plot.addItem(dot)
                self._markers.append((plot, line, dot))
        else:
            faint = pg.mkPen(OTHER_PATH_COLOUR, width=1)
            for shifts in others:
                self.path_plot.plot(shifts[:, 0], shifts[:, 1], pen=faint)
            if current is not None:
                self.path_plot.plot(
                    current[:, 0], current[:, 1], pen=pg.mkPen(PATH_COLOUR, width=2)
                )
            dot = pg.ScatterPlotItem(
                size=10, brush=pg.mkBrush(MARKER_COLOUR), pen=pg.mkPen("k")
            )
            self.path_plot.addItem(dot)
            self._markers.append((self.path_plot, None, dot))
        self._draw_marker()

    def _draw_marker(self) -> None:
        current = self._current()
        frame = self._frame
        for plot, line, dot in self._markers:
            if line is not None:
                line.setVisible(current is not None)
                line.setValue(frame)
            if current is None or not 0 <= frame < len(current):
                dot.setData([], [])
            elif plot is self.x_plot:
                dot.setData([frame], [current[frame, 0]])
            elif plot is self.y_plot:
                dot.setData([frame], [current[frame, 1]])
            else:
                dot.setData([current[frame, 0]], [current[frame, 1]])

    def _clicked(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or event.double():
            return
        current = self._current()
        if current is None or not len(current):
            return
        pos = event.scenePos()
        if self.mode.currentData() == "time":
            for plot in (self.x_plot, self.y_plot):
                box = plot.vb
                if box.sceneBoundingRect().contains(pos):
                    x = box.mapSceneToView(pos).x()
                    self.frame_clicked.emit(int(np.clip(round(x), 0, len(current) - 1)))
                    return
            return
        box = self.path_plot.vb
        if not box.sceneBoundingRect().contains(pos):
            return
        point = box.mapSceneToView(pos)
        width, height = box.viewPixelSize()  # data units per screen pixel
        distance = np.hypot(
            (current[:, 0] - point.x()) / width, (current[:, 1] - point.y()) / height
        )
        self.frame_clicked.emit(int(np.nanargmin(distance)))
