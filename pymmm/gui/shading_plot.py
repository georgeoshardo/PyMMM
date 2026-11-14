"""The Shading tab's evidence plot: each trench's cells against the PDMS background at
them (:meth:`pymmm.steps.shading.ShadingCorrector.evidence`), with the slope each mode
expects. Cells that brighten one for one with the background say light is added
(subtract); in proportion to it, a gain (divide). The FOV on the P slider stands out.
"""

from __future__ import annotations

import numpy as np

# qtpy first, so pyqtgraph uses the Qt binding napari already loaded
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget

# isort: split
import pyqtgraph as pg

OTHER_COLOUR = (150, 150, 150, 90)
FOV_COLOUR = "#e8c547"
SUBTRACT_COLOUR = "#4e9fd6"
DIVIDE_COLOUR = "#f28e2b"
FIT_COLOUR = "#ffffff"


class EvidencePlot(QWidget):
    """Cells against background, one dot per trench: :meth:`set_evidence`, then
    :meth:`show_fov`.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._table = None
        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.view = pg.PlotWidget(background=None)
        self.view.setMinimumHeight(180)
        self.view.setLabel("bottom", "PDMS background (counts)")
        self.view.setLabel("left", "cells (counts)")
        self.view.addLegend(offset=(8, 8), labelTextSize="8pt")
        self.others = pg.ScatterPlotItem(
            size=4, pen=None, brush=pg.mkBrush(OTHER_COLOUR)
        )
        self.fov = pg.ScatterPlotItem(size=6, pen=None, brush=pg.mkBrush(FOV_COLOUR))
        self.subtract = pg.PlotDataItem(
            pen=pg.mkPen(SUBTRACT_COLOUR, width=2, style=pg.QtCore.Qt.PenStyle.DotLine),
            name="subtract: slope 1",
        )
        self.divide = pg.PlotDataItem(
            pen=pg.mkPen(DIVIDE_COLOUR, width=2, style=pg.QtCore.Qt.PenStyle.DashLine),
            name="divide",
        )
        self.fit = pg.PlotDataItem(pen=pg.mkPen(FIT_COLOUR, width=1), name="fit")
        for item in (self.others, self.fov, self.subtract, self.divide, self.fit):
            self.view.addItem(item)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.note)
        layout.addWidget(self.view)

    def set_evidence(self, table, slopes: dict[str, float]) -> None:
        """The trenches (``fov``, ``background``, ``cells``) and the slopes
        (:meth:`~pymmm.steps.shading.ShadingCorrector.evidence_slopes`).
        """
        self._table = table
        x = table["background"].to_numpy(float)
        y = table["cells"].to_numpy(float)
        self.others.setData(x, y)
        if len(x) < 2:
            for line in (self.subtract, self.divide, self.fit):
                line.setData([], [])
            self.note.setText("")
            return
        xs = np.array([x.min(), x.max()])
        mx, my = np.median(x), np.median(y)
        self.subtract.setData(xs, my + (xs - mx))
        for line, key, label in (
            (self.divide, "divide", "divide"),
            (self.fit, "measured", "fit"),
        ):
            slope = slopes.get(key, float("nan"))
            if np.isfinite(slope):
                line.setData(xs, my + slope * (xs - mx))
                line.opts["name"] = f"{label}: slope {slope:.2f}"
            else:
                line.setData([], [])
        legend = self.view.plotItem.legend
        legend.clear()
        for line in (self.subtract, self.divide, self.fit):
            legend.addItem(line, line.opts["name"])
        measured = slopes.get("measured", float("nan"))
        self.note.setText(
            f"{len(x)} trenches · cells per count of background {measured:.2f} · "
            f"subtract 1 · divide {slopes.get('divide', float('nan')):.2f}"
        )

    def show_fov(self, fov: str | None) -> None:
        if self._table is None or fov is None:
            self.fov.setData([], [])
            return
        rows = self._table[self._table["fov"] == fov]
        self.fov.setData(
            rows["background"].to_numpy(float), rows["cells"].to_numpy(float)
        )

    def clear(self) -> None:
        self._table = None
        for item in (self.others, self.fov, self.subtract, self.divide, self.fit):
            item.setData([], [])
        self.note.setText("")
