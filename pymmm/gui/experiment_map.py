"""The Experiment tab's map: every FOV where it sat on the stage, then its lanes and
trench boxes once they are found.

Coordinates are µm on the stage, turned so that a FOV's registered frame is upright
(:class:`~pymmm.source.stage.StageLayout`), with y down as in the viewer. Each kind of
mark is one plot item (all outlines, all lanes, all trench boxes), so a map of
hundreds of trenches redraws at once. Zoomed out, each FOV is a filled box, red when it
needs a look (:meth:`ExperimentMap.set_flagged`); lanes, trench boxes and names show
once a FOV is wide enough on screen to hold them. A click on a FOV picks it.
"""

from __future__ import annotations

import numpy as np

# qtpy first, so pyqtgraph uses the Qt binding napari already loaded
from qtpy.QtCore import QPointF, Qt, Signal
from qtpy.QtGui import QBrush, QColor, QPen, QPolygonF
from qtpy.QtWidgets import QGraphicsPolygonItem, QLabel, QVBoxLayout, QWidget

# isort: split
import pyqtgraph as pg

OUTLINE_COLOUR = (190, 190, 190, 200)
DROPPED_COLOUR = (150, 150, 150, 90)
FILL_COLOUR = (150, 150, 150, 40)
FLAG_COLOUR = (220, 60, 60, 110)
CURRENT_COLOUR = "#e8c547"
LANE_COLOURS = {1: "#ff4040", -1: "#40e0e0"}  # opening down (red), up (cyan)
TRENCH_COLOUR = (255, 90, 90, 220)
DETAIL_MIN_WIDTH = 46  # screen px a FOV needs before its lanes, trenches and name show


def _path(shapes: list[np.ndarray], closed: bool) -> tuple[np.ndarray, np.ndarray]:
    """The x and y of many polylines in one array, NaN between them."""
    if not shapes:
        return np.array([]), np.array([])
    parts = []
    for shape in shapes:
        shape = np.asarray(shape, float)
        if closed:
            shape = np.vstack([shape, shape[:1]])
        parts.extend([shape, np.full((1, 2), np.nan)])
    points = np.vstack(parts)
    return points[:, 0], points[:, 1]


class ExperimentMap(QWidget):
    """Every FOV at its stage position; :attr:`fov_clicked` carries a clicked FOV."""

    fov_clicked = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._outlines: dict[str, np.ndarray] = {}
        self._active: list[str] = []
        self._flagged: set[str] = set()
        self._labels: dict[str, str] = {}
        self._fov: str | None = None

        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.view = pg.GraphicsLayoutWidget()
        self.view.setBackground(None)
        self.plot = self.view.addPlot()
        self.plot.setAspectLocked(True)
        self.plot.invertY(True)  # y down, as in the viewer
        self.plot.setLabel("bottom", "x (µm)")
        self.plot.setLabel("left", "y (µm)")
        self.plot.vb.sigRangeChanged.connect(self._show_detail)
        self.view.scene().sigMouseClicked.connect(self._clicked)

        self._dropped = self.plot.plot(pen=pg.mkPen(DROPPED_COLOUR, style=Qt.DashLine))
        self._fovs = self.plot.plot(pen=pg.mkPen(OUTLINE_COLOUR, width=1))
        self._lanes = {
            k: self.plot.plot(pen=pg.mkPen(c, width=1.5))
            for k, c in LANE_COLOURS.items()
        }
        self._trenches = self.plot.plot(pen=pg.mkPen(TRENCH_COLOUR, width=1))
        self._current = self.plot.plot(pen=pg.mkPen(CURRENT_COLOUR, width=2.5))
        self._fills: list[QGraphicsPolygonItem] = []
        self._texts: dict[str, pg.TextItem] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.note)
        layout.addWidget(self.view)
        layout.addStretch(1)  # spare height below the map, not between its parts
        self._fit_height()

    # ------------------------------------------------------------------

    @property
    def fovs(self) -> list[str]:
        """The FOVs of the experiment, in the order given."""
        return list(self._active)

    def clear(self) -> None:
        """Nothing drawn."""
        self.set_fovs({}, [])

    def set_fovs(self, outlines: dict[str, np.ndarray], active: list[str]) -> None:
        """Draw every FOV's outline ((4, 2) µm corners); those not in ``active`` (left
        out of the experiment) dashed. Lanes and trench boxes are cleared.
        """
        self._outlines = {name: np.asarray(o, float) for name, o in outlines.items()}
        self._active = [name for name in active if name in self._outlines]
        self._flagged = set()
        self._labels = {name: name for name in self._active}
        dropped = [o for name, o in self._outlines.items() if name not in self._active]
        self._dropped.setData(*_path(dropped, closed=True), connect="finite")
        self._fovs.setData(
            *_path([self._outlines[n] for n in self._active], closed=True),
            connect="finite",
        )
        self.set_lanes([], [])
        self.set_trenches([])
        self._refill()
        self._make_labels()
        self._draw_current()
        self._fit_height()
        if self._outlines:
            self.plot.autoRange()

    def set_lanes(self, segments: list[np.ndarray], orientations: list[int]) -> None:
        """Draw lanes: (2, 2) µm end points each, coloured by which way the trenches
        open (+1 down, red; -1 up, cyan).
        """
        for orientation, item in self._lanes.items():
            mine = [s for s, o in zip(segments, orientations) if o == orientation]
            item.setData(*_path(mine, closed=False), connect="finite")

    def set_trenches(self, boxes: list[np.ndarray]) -> None:
        """Draw trench boxes: (4, 2) µm corners each."""
        self._trenches.setData(*_path(boxes, closed=True), connect="finite")

    def set_flagged(self, fovs: set[str]) -> None:
        """Fill these FOVs red: they need a look."""
        self._flagged = set(fovs) & set(self._active)
        self._refill()

    def set_labels(self, labels: dict[str, str]) -> None:
        """What each FOV's label says (its name by default)."""
        self._labels = {name: labels.get(name, name) for name in self._active}
        self._make_labels()

    def show_fov(self, fov: str | None) -> None:
        """Outline this FOV in colour."""
        self._fov = fov
        self._draw_current()

    # ------------------------------------------------------------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_height()

    def _fit_height(self) -> None:
        """As tall as the layout's shape needs at this width (an experiment is
        usually much wider than tall), within 160 to 420 px.
        """
        if not self._outlines:
            self.view.setFixedHeight(200)
            return
        corners = np.vstack(list(self._outlines.values()))
        width, height = np.ptp(corners[:, 0]), np.ptp(corners[:, 1])
        axes = 70  # room the axes and their labels take
        wanted = (self.view.width() - axes) * height / max(width, 1e-9) + axes
        self.view.setFixedHeight(int(np.clip(wanted, 160, 420)))

    def _draw_current(self) -> None:
        outline = self._outlines.get(self._fov) if self._fov is not None else None
        if outline is None:
            self._current.setData([], [])
        else:
            self._current.setData(*_path([outline], closed=True), connect="finite")

    def _refill(self) -> None:
        for item in self._fills:
            self.plot.removeItem(item)
        self._fills = []
        for name in self._active:
            polygon = QPolygonF([QPointF(x, y) for x, y in self._outlines[name]])
            item = QGraphicsPolygonItem(polygon)
            colour = FLAG_COLOUR if name in self._flagged else FILL_COLOUR
            item.setBrush(QBrush(QColor(*colour)))
            item.setPen(QPen(Qt.PenStyle.NoPen))
            item.setZValue(-10)
            self.plot.addItem(item)
            self._fills.append(item)

    def _make_labels(self) -> None:
        for text in self._texts.values():
            self.plot.removeItem(text)
        self._texts = {}
        for name in self._active:
            text = pg.TextItem(self._labels.get(name, name), anchor=(0.5, 0.5))
            text.setPos(*self._outlines[name].mean(axis=0))
            self.plot.addItem(text)
            self._texts[name] = text
        self._show_detail()

    @property
    def detailed(self) -> bool:
        """Whether a FOV is wide enough on screen to show its lanes, trenches, name."""
        if not self._outlines:
            return False
        width_um = np.ptp(next(iter(self._outlines.values()))[:, 0])
        per_pixel = self.plot.vb.viewPixelSize()[0]
        return bool(per_pixel > 0 and width_um / per_pixel >= DETAIL_MIN_WIDTH)

    def _show_detail(self, *_args) -> None:
        show = self.detailed
        for item in [*self._texts.values(), *self._lanes.values(), self._trenches]:
            item.setVisible(show)

    def fov_at(self, x: float, y: float) -> str | None:
        """The FOV of the experiment under a map point (µm)."""
        point = QPointF(x, y)
        for name in self._active:
            polygon = QPolygonF([QPointF(px, py) for px, py in self._outlines[name]])
            if polygon.containsPoint(point, Qt.FillRule.OddEvenFill):
                return name
        return None

    def _clicked(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or event.double():
            return
        box = self.plot.vb
        if not box.sceneBoundingRect().contains(event.scenePos()):
            return
        point = box.mapSceneToView(event.scenePos())
        fov = self.fov_at(point.x(), point.y())
        if fov is not None:
            self.fov_clicked.emit(fov)
