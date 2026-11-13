"""The extraction dock: the extraction notebook's steps before ``extract()``, in napari.

Open an experiment file (TOML) or an ND2, set which part of the ND2 to use, then work
down the tabs as the notebook works down its cells:

- Experiment: the FOVs to keep or drop, the frames, and a y/x crop of the ND2.
- Register: the registration settings. Until you register, the frames are shown
  turned by the rotation set, so you can level them by eye. The Register button
  computes every FOV's drift in a background process pool and saves it to the
  companion store beside the ND2.
- Lanes: detect the lanes, or take the image's edge. The candidate lanes of the FOV
  on the P slider follow the settings until you apply them to every FOV and save.
- Trenches: the same, for the trenches.
- Preview: the trenches as ``extract()`` would write them, in a second viewer. Each
  slider step cuts one trench at one frame
  (:class:`~pymmm.steps.extraction.TrenchPreview`).

napari asks each layer only for the plane it shows, and the raw and registered layers
are dask arrays that read one frame per plane, so the ND2 is never read whole. The
dock runs the notebook's step classes and saves the same results. On opening, it loads
a saved registration made with its settings, and the lanes and trenches saved after
it, instead of computing them again.

A tab is marked ✓ once its step is saved with the settings it shows, and * when they
have changed since. The steps after registration wait while the Register tab's settings
differ from the registration in use. On opening, a saved step made from another
registration or other lanes (``CompanionStore.lineage``) is not loaded.

The steps save their diagnostic figures beside the ND2 as the notebook does (drift,
lanes, trenches). The View box below the tabs holds what can change at any step: the
coordinate grid (napari 0.10 and later; on by default, and View → Coordinate Grid
toggles it too) and the units both viewers show, pixels or µm. The settings are in
pixels either way.

Run it with ``python -m pymmm.gui <experiment.toml or .nd2>`` (the view extra), or
open it from napari's Plugins menu.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
from qtpy.QtCore import QObject, QTimer, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pymmm.plotting import contrast_limits
from pymmm.steps.registration import DETECTION_MEAN, mean_frames
from pymmm.store import lineage

RAW_AXES = ("T", "P", "y", "x")
PREVIEW_AXES = ("trench", "t", "y", "x")

# channel colours by name, for channels other than the detection channel
_COLOURS = (
    ("mcherry", "magenta"),
    ("rfp", "magenta"),
    ("yfp", "yellow"),
    ("venus", "yellow"),
    ("gfp", "green"),
    ("cfp", "cyan"),
)


def channel_colormap(name: str) -> str:
    """A napari colormap for a channel: grey for phase contrast, else by fluorophore."""
    lower = name.lower()
    for key, colour in _COLOURS:
        if key in lower:
            return colour
    return "gray"


def coordinate_grid(viewer):
    """The viewer's coordinate grid overlay (napari 0.10 and later), or None."""
    overlays = getattr(getattr(viewer, "scene", None), "overlays", None)
    return getattr(overlays, "coordinate_grid", None)


def show_coordinate_grid(viewer) -> None:
    """Turn on the viewer's coordinate grid, in the image's pixels: where trenches,
    lanes and crops are, to read off for the settings. View → Coordinate Grid turns it
    off again.
    """
    grid = coordinate_grid(viewer)
    if grid is not None:
        grid.visible = True


def set_units(layers, unit: str, pixel_size_um: float) -> None:
    """Show ``layers`` in pixels (``"pixel"``) or µm (``"micrometer"``): each layer's
    last two axes (y, x) scaled by the pixel size, or not. The layers' data, and the
    shapes drawn on them, stay in pixels.
    """
    size = pixel_size_um if unit == "micrometer" else 1.0
    for layer in layers:
        lead = layer.ndim - 2
        layer.scale = (*layer.scale[:lead], size, size)
        if hasattr(layer, "units"):  # napari 0.6 and later
            layer.units = (*layer.units[:lead], unit, unit)


def keep_view(viewer, layers, unit: str, pixel_size_um: float) -> None:
    """:func:`set_units`, with the camera kept on the same pixels at the same size on
    screen: its centre and zoom follow the change of scale.
    """
    if not layers:
        return
    before = float(layers[0].scale[-1])
    set_units(layers, unit, pixel_size_um)
    factor = float(layers[0].scale[-1]) / before
    if factor == 1:
        return
    camera = viewer.camera
    centre = list(camera.center)
    centre[-2:] = [centre[-2] * factor, centre[-1] * factor]
    camera.center = tuple(centre)
    camera.zoom = camera.zoom / factor


# ----------------------------------------------------------------------
# The experiment tab's fields
# ----------------------------------------------------------------------


def parse_range(text: str) -> tuple[int, int] | None:
    """``"290:850"`` (or ``"290-850"``, ``"290, 850"``) as ``(290, 850)``; blank is None."""
    text = text.strip()
    if not text:
        return None
    parts = re.findall(r"\d+", text)
    if len(parts) != 2 or re.sub(r"[\d:,\-\s]", "", text):
        raise ValueError(f"expected start:stop, got {text!r}")
    start, stop = (int(p) for p in parts)
    return start, stop


def format_range(value: Any) -> str:
    """A range as the field shows it: ``(290, 850)``, ``{"start": 290, "stop": 850}``
    or a slice as ``"290:850"``; None or an open range as blank.
    """
    if value is None:
        return ""
    if isinstance(value, dict):
        value = (value.get("start"), value.get("stop"))
    elif isinstance(value, slice):
        value = (value.start, value.stop)
    start, stop = value
    if start is None and stop is None:
        return ""
    return f"{start or 0}:{'' if stop is None else stop}"


def parse_fovs(text: str) -> list[str | int]:
    """FOVs by name or P index, separated by commas or spaces: ``"A01, A02, 5"``."""
    return [
        int(p) if p.isdigit() else p for p in re.split(r"[,\s]+", text.strip()) if p
    ]


# ----------------------------------------------------------------------
# Small Qt helpers
# ----------------------------------------------------------------------


def _dspin(value: float, low: float, high: float, step: float, decimals: int = 1):
    box = QDoubleSpinBox()
    box.setRange(low, high)
    box.setSingleStep(step)
    box.setDecimals(decimals)
    box.setValue(value)
    box.setKeyboardTracking(False)  # report a typed value once, not each keystroke
    return box


def _spin(value: int, low: int, high: int, step: int = 1):
    box = QSpinBox()
    box.setRange(low, high)
    box.setSingleStep(step)
    box.setValue(value)
    box.setKeyboardTracking(False)
    return box


def _combo(options: list[tuple[str, Any]], value: Any = None):
    box = QComboBox()
    for label, data in options:
        box.addItem(label, data)
    _select(box, value)
    return box


def _select(box: QComboBox, value: Any) -> None:
    index = box.findData(value)
    if index >= 0:
        box.setCurrentIndex(index)


def _mean_controls():
    """The detection image's settings: how many registered frames to average, and
    which (:func:`pymmm.steps.registration.mean_frames`).
    """
    from pymmm.steps.registration import DETECTION_MEAN

    n = _spin(DETECTION_MEAN["mean_n_frames"], 1, 100000)
    where = _combo(
        [
            ("spread over the movie", "spread"),
            ("the first", "start"),
            ("the last", "end"),
        ],
        DETECTION_MEAN["mean_from"],
    )
    return n, where


def _note(plot, text: str) -> None:
    """A plot's note line, hidden when there is nothing to say."""
    plot.note.setText(text)
    plot.note.setVisible(bool(text))


def _shifts(mats: np.ndarray) -> np.ndarray:
    """(frames, 2) x and y shifts from (frames, 3, 3) transforms."""
    return np.stack([mats[:, 0, 2], mats[:, 1, 2]], axis=1)


def _canvas(height: float = 2.4):
    """A matplotlib figure on a Qt canvas (not pyplot, so no global state)."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(4, height), layout="constrained")
    canvas = FigureCanvasQTAgg(figure)
    canvas.setMinimumHeight(int(height * 70))
    return canvas


class _Progress(QObject):
    """Carries a background run's progress to the Qt thread (a queued signal)."""

    changed = Signal(int, int)


class _Registered(QObject):
    """Carries each FOV's drift to the Qt thread as registration finishes it."""

    fov = Signal(str, object)


# ----------------------------------------------------------------------
# The dock
# ----------------------------------------------------------------------


class ExtractionDock(QWidget):
    """The extraction notebook's steps before ``extract()`` as a napari dock widget.

    Parameters
    ----------
    napari_viewer : napari.Viewer
        The viewer the ND2's layers go into.
    experiment_file : str | Path | None
        Experiment file (TOML), or an ND2, to open at once.
    """

    def __init__(self, napari_viewer, experiment_file: str | Path | None = None):
        super().__init__()
        self.viewer = napari_viewer
        self.experiment = None  # ND2Experiment
        self.store = None  # CompanionStore
        self.registrator = None  # the computed or loaded Registrator
        self.lane_detector = None
        self.trench_detector = None
        self.preview_viewer = None
        self._layers: dict[str, Any] = {}
        self._worker = None
        self._cancel: threading.Event | None = None
        self._mean_jobs: set[tuple[str, str]] = set()
        self._pitch_job = None  # the detection images being computed for the pitch
        self._pitches: dict[tuple, float | None] = {}  # experiment pitch by settings
        self._opened_file: Path | None = None
        self._opened_fields: tuple[str, ...] | None = None
        self._legacy_mean = dict(DETECTION_MEAN)
        self._live_drift: _Registered | None = None  # the registration being drawn
        self._drift_from = None  # the Registrator the drift plot was filled from
        self.experiment_group: Path | None = None  # where extract() writes
        self.lineage: dict[str, str] = {}

        self._build()
        self.viewer.dims.events.current_step.connect(self._on_step)
        self._last_fov: str | None = None
        if experiment_file is not None:
            self.path_edit.setText(str(experiment_file))
            self.open_experiment()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build(self) -> None:
        self.tabs = QTabWidget()
        self.tabs.addTab(self._experiment_tab(), "Experiment")
        self.tabs.addTab(self._register_tab(), "Register")
        self.tabs.addTab(self._lanes_tab(), "Lanes")
        self.tabs.addTab(self._trenches_tab(), "Trenches")
        self.tabs.addTab(self._preview_tab(), "Preview")
        self.tabs.currentChanged.connect(self._on_tab)
        # every setting re-marks the tabs at once (their previews wait a moment)
        for kind in (QDoubleSpinBox, QSpinBox):
            for box in self.tabs.findChildren(kind):
                box.valueChanged.connect(self._mark_tabs)
        for box in self.tabs.findChildren(QComboBox):
            box.currentIndexChanged.connect(self._mark_tabs)
        for edit in (self.reg_y_edit, self.reg_x_edit):
            edit.textChanged.connect(self._mark_tabs)
        self.status = QLabel("Open an experiment file.")
        self.status.setWordWrap(True)
        # what can be changed whichever step is open, below the steps' tabs
        self.view_box = QGroupBox("View")
        view_layout = QVBoxLayout(self.view_box)
        self.grid_box = QCheckBox("coordinate grid")
        self.units = _combo([("pixels", "pixel"), ("µm", "micrometer")], "pixel")
        self.units.currentIndexChanged.connect(self._on_units)
        units_row = QHBoxLayout()
        units_row.addWidget(QLabel("units"))
        units_row.addWidget(self.units, 1)
        view_layout.addWidget(self.grid_box)
        view_layout.addLayout(units_row)
        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)
        layout.addWidget(self.view_box)
        layout.addWidget(self.status)
        grid = coordinate_grid(self.viewer)
        if grid is None:
            self.grid_box.hide()  # napari before 0.10
        else:
            show_coordinate_grid(self.viewer)
            self.grid_box.setChecked(grid.visible)
            self.grid_box.toggled.connect(lambda on: setattr(grid, "visible", on))
            # View → Coordinate Grid toggles it too
            # (napari's events pass the new value; older ones an event holding it)
            grid.events.visible.connect(
                lambda value: self.grid_box.setChecked(
                    bool(getattr(value, "value", value))
                )
            )
        self._enable()

    def _experiment_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self.path_edit = QLineEdit()
        browse = QPushButton("…")
        browse.setMaximumWidth(30)
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.path_edit)
        row.addWidget(browse)
        form.addRow("experiment or ND2", row)
        self.keep_edit = QLineEdit()
        self.keep_edit.setPlaceholderText("all (or e.g. A01, A02 or 0, 1)")
        self.discard_edit = QLineEdit()
        self.discard_edit.setPlaceholderText("none")
        self.times_edit = QLineEdit()
        self.times_edit.setPlaceholderText("all (or start:stop, stop excluded)")
        self.y_roi_edit = QLineEdit()
        self.y_roi_edit.setPlaceholderText("whole height (or start:stop)")
        self.x_roi_edit = QLineEdit()
        self.x_roi_edit.setPlaceholderText("whole width (or start:stop)")
        form.addRow("keep FOVs", self.keep_edit)
        form.addRow("drop FOVs", self.discard_edit)
        form.addRow("frames", self.times_edit)
        form.addRow("crop y", self.y_roi_edit)
        form.addRow("crop x", self.x_roi_edit)
        for edit in (
            self.keep_edit,
            self.discard_edit,
            self.times_edit,
            self.y_roi_edit,
            self.x_roi_edit,
        ):
            edit.textChanged.connect(self._mark_tabs)
        self.open_button = QPushButton("Open")
        self.open_button.clicked.connect(self.open_experiment)
        form.addRow(self.open_button)
        self.experiment_info = QLabel("")
        self.experiment_info.setWordWrap(True)
        form.addRow(self.experiment_info)
        return tab

    def _register_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self.reg_channel = QComboBox()
        self.rotation = _dspin(0.0, -45, 45, 0.1, decimals=2)
        self.reg_y_edit = QLineEdit()
        self.reg_y_edit.setPlaceholderText("whole height (or start:stop)")
        self.reg_x_edit = QLineEdit()
        self.reg_x_edit.setPlaceholderText("whole width (or start:stop)")
        self.mode = _combo(
            [
                ("first frame", "first"),
                ("last frame", "last"),
                ("previous frame", "previous"),
            ],
            "first",
        )
        self.backend = _combo(
            [("stackreg", "stackreg"), ("opencv (fast)", "opencv")], "stackreg"
        )
        self.downsample = _spin(4, 1, 16)
        self.batch_size = _spin(4, 1, 64)
        self.workers = _spin(0, 0, 256)
        self.workers.setSpecialValueText("one per core")
        form.addRow("channel", self.reg_channel)
        form.addRow("rotation (°)", self.rotation)
        form.addRow("region y (raw px)", self.reg_y_edit)
        form.addRow("region x (raw px)", self.reg_x_edit)
        form.addRow("register to", self.mode)
        form.addRow("backend", self.backend)
        form.addRow("downsample", self.downsample)
        form.addRow("frames per task", self.batch_size)
        form.addRow("workers", self.workers)
        buttons = QHBoxLayout()
        self.register_button = QPushButton("Register")
        self.register_button.clicked.connect(self.register)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._cancel_run)
        self.cancel_button.setEnabled(False)
        buttons.addWidget(self.register_button)
        buttons.addWidget(self.cancel_button)
        form.addRow(buttons)
        self.reg_progress = QProgressBar()
        self.reg_progress.setFormat("%v / %m FOVs")
        form.addRow(self.reg_progress)
        self.reg_status = QLabel("")
        self.reg_status.setWordWrap(True)
        form.addRow(self.reg_status)
        from pymmm.gui.drift_plot import DriftPlot

        self.drift_plot = DriftPlot()
        self.drift_plot.frame_clicked.connect(self._go_to_frame)
        form.addRow(self.drift_plot)

        self._rotation_timer = self._timer(self._on_registration_settings)
        for box in (self.rotation, self.downsample):
            box.valueChanged.connect(self._rotation_timer.start)
        for box in (self.reg_channel, self.mode, self.backend):
            box.currentIndexChanged.connect(self._rotation_timer.start)
        for edit in (self.reg_y_edit, self.reg_x_edit):
            edit.editingFinished.connect(self._rotation_timer.start)
        return tab

    def _lanes_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self.det_channel = QComboBox()
        self.lane_mode = _combo(
            [
                ("detect", None),
                ("image's bottom edge (open down)", "down"),
                ("image's top edge (open up)", "up"),
            ],
            None,
        )
        self.lane_sigma = _dspin(40, 0, 200, 1)
        self.lane_distance = _dspin(300, 10, 1000, 10)
        self.lane_height = _dspin(5000, 0, 65535, 100, decimals=0)
        self.lane_orientation = _combo(
            [
                ("auto (compare brightness)", "auto"),
                ("trenches above, open down (red)", "down"),
                ("trenches below, open up (cyan)", "up"),
            ],
            "auto",
        )
        self.lane_window = _dspin(20, 1, 100, 1)
        self.lane_mean_n, self.lane_mean_from = _mean_controls()
        form.addRow("channel", self.det_channel)
        form.addRow("mean of n frames", self.lane_mean_n)
        form.addRow("frames from", self.lane_mean_from)
        form.addRow("lanes", self.lane_mode)
        form.addRow("sigma", self.lane_sigma)
        form.addRow("distance", self.lane_distance)
        form.addRow("height", self.lane_height)
        form.addRow("orientation", self.lane_orientation)
        form.addRow("orient. window (µm)", self.lane_window)
        self.lane_apply = QPushButton("Apply to all FOVs and save")
        self.lane_apply.clicked.connect(self.apply_lanes)
        form.addRow(self.lane_apply)
        self.lane_status = QLabel("")
        self.lane_status.setWordWrap(True)
        form.addRow(self.lane_status)
        self.lane_canvas = _canvas(2.4)
        form.addRow(self.lane_canvas)

        self._lane_timer = self._timer(self.update_lanes)
        for box in (
            self.lane_sigma,
            self.lane_distance,
            self.lane_height,
            self.lane_window,
            self.lane_mean_n,
        ):
            box.valueChanged.connect(self._lane_timer.start)
        for box in (self.lane_mode, self.lane_orientation, self.lane_mean_from):
            box.currentIndexChanged.connect(self._lane_timer.start)
        self.det_channel.currentIndexChanged.connect(self._lane_timer.start)
        return tab

    def _trenches_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self.trench_method = _combo(
            [
                ("peaks (tallest maxima)", "peaks"),
                ("periodic (spacing + symmetry)", "periodic"),
            ],
            "peaks",
        )
        self.trench_sigma = _dspin(4, 0, 50, 0.5)
        self.trench_distance = _dspin(100, 1, 500, 1)
        self.trench_prominence = _dspin(10, 0, 5000, 1)
        self.trench_pitch = _dspin(0, 0, 2000, 1)
        self.trench_pitch.setSpecialValueText("auto")
        self.trench_symmetry = _dspin(0.1, 0, 1, 0.05, decimals=2)
        self.trench_width = _spin(32, 1, 400)
        self.trench_length = _spin(160, 10, 10000, 5)
        self.trench_offset = _spin(50, -2000, 2000, 5)
        self.trench_mean_n, self.trench_mean_from = _mean_controls()
        form.addRow("mean of n frames", self.trench_mean_n)
        form.addRow("frames from", self.trench_mean_from)
        form.addRow("method", self.trench_method)
        form.addRow("sigma", self.trench_sigma)
        form.addRow("distance", self.trench_distance)
        form.addRow("prominence", self.trench_prominence)
        form.addRow("expected pitch", self.trench_pitch)
        form.addRow("min symmetry", self.trench_symmetry)
        form.addRow("trench width", self.trench_width)
        form.addRow("trench length", self.trench_length)
        form.addRow("bottom offset", self.trench_offset)
        self.trench_lane_note = QLabel("")
        self.trench_lane_note.setWordWrap(True)
        form.addRow(self.trench_lane_note)
        self.trench_apply = QPushButton("Apply to all FOVs and save")
        self.trench_apply.clicked.connect(self.apply_trenches)
        form.addRow(self.trench_apply)
        self.trench_status = QLabel("")
        self.trench_status.setWordWrap(True)
        form.addRow(self.trench_status)
        self.trench_canvas = _canvas(2.6)
        form.addRow(self.trench_canvas)

        self._trench_timer = self._timer(self.update_trenches)
        for box in (
            self.trench_sigma,
            self.trench_distance,
            self.trench_prominence,
            self.trench_pitch,
            self.trench_symmetry,
            self.trench_width,
            self.trench_length,
            self.trench_offset,
            self.trench_mean_n,
        ):
            box.valueChanged.connect(self._trench_timer.start)
        for box in (self.trench_method, self.trench_mean_from):
            box.currentIndexChanged.connect(self._trench_timer.start)
        return tab

    def _preview_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.preview_button = QPushButton("Open the trenches in a new viewer")
        # not connected straight to open_preview: clicked passes its checked flag
        self.preview_button.clicked.connect(lambda: self.open_preview())
        self.preview_info = QLabel(
            "The trenches as extract() would write them, one per step of the trench "
            "slider. Nothing is written."
        )
        self.preview_info.setWordWrap(True)
        self.extraction_info = QLabel("")
        self.extraction_info.setWordWrap(True)
        layout.addWidget(self.preview_button)
        layout.addWidget(self.preview_info)
        layout.addWidget(self.extraction_info)
        layout.addStretch()
        return tab

    def _timer(self, slot) -> QTimer:
        """A short single-shot timer: settings changed quickly in a row update once."""
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.setInterval(150)
        timer.timeout.connect(slot)
        return timer

    def _enable(self) -> None:
        """Each tab once the step before it is done; nothing while a run is going."""
        busy = self._worker is not None
        opened = self.experiment is not None
        # the steps after registration only with the registration the tab shows
        registered = self.registrator is not None and self._settings_match()
        lanes = registered and self.lane_detector is not None
        trenches = (
            lanes
            and self.trench_detector is not None
            and self.trench_detector.is_detected
        )
        self.tabs.setTabEnabled(1, opened)
        self.tabs.setTabEnabled(2, registered)
        self.tabs.setTabEnabled(3, lanes)
        self.tabs.setTabEnabled(4, trenches)
        self.open_button.setEnabled(not busy)
        self.register_button.setEnabled(opened and not busy)
        self.cancel_button.setEnabled(busy and self._cancel is not None)
        self.lane_apply.setEnabled(registered and not busy)
        self.trench_apply.setEnabled(lanes and not busy)
        self.preview_button.setEnabled(trenches and not busy)
        self._mark_tabs()

    # ------------------------------------------------------------------
    # What each tab shows: saved, or changed since
    # ------------------------------------------------------------------

    TAB_NAMES = ("Experiment", "Register", "Lanes", "Trenches", "Preview")

    def _mark_tabs(self) -> None:
        """Mark each step's tab: ✓ saved with the settings shown, * settings changed
        since (not applied yet), nothing when not done.
        """
        marks = {
            "Experiment": "*" if self._experiment_changed() else "",
            "Register": self._mark(self.registrator, self._settings_match),
            "Lanes": self._mark(self.lane_detector, self._lanes_match),
            "Trenches": self._mark(self.trench_detector, self._trenches_match),
            "Preview": "",
        }
        tips = {"✓": "saved, with these settings", "*": "settings changed, not applied"}
        for index, name in enumerate(self.TAB_NAMES):
            mark = marks[name]
            self.tabs.setTabText(index, f"{name} {mark}".strip())
            self.tabs.setTabToolTip(index, tips.get(mark, ""))

    @staticmethod
    def _mark(step, matches) -> str:
        if step is None:
            return ""
        return "✓" if matches() else "*"

    def _experiment_fields(self) -> tuple[str, ...]:
        return tuple(
            " ".join(edit.text().split())
            for edit in (
                self.keep_edit,
                self.discard_edit,
                self.times_edit,
                self.y_roi_edit,
                self.x_roi_edit,
            )
        )

    def _experiment_changed(self) -> bool:
        return (
            self._opened_fields is not None
            and self._experiment_fields() != self._opened_fields
        )

    @staticmethod
    def _same(box, saved) -> bool:
        """Whether a control shows a saved value (to the decimals a spin box shows)."""
        if isinstance(box, QComboBox):
            return box.currentData() == saved
        if saved is None:
            return False
        if isinstance(box, QDoubleSpinBox):
            return round(float(saved), box.decimals()) == round(
                box.value(), box.decimals()
            )
        return int(saved) == box.value()

    def _mean_matches(self, n_box, from_box, saved: dict) -> bool:
        mean = self._saved_mean(saved)
        return self._same(n_box, mean["mean_n_frames"]) and self._same(
            from_box, mean["mean_from"]
        )

    def _lanes_match(self) -> bool:
        saved = self.lane_detector.detection_params
        if self.det_channel.currentData() != self.lane_detector.channel:
            return False
        if not self._mean_matches(self.lane_mean_n, self.lane_mean_from, saved):
            return False
        edge = self.lane_mode.currentData()
        if edge is not None or saved.get("image_edge") is not None:
            return edge == saved.get("image_edge")
        return all(
            self._same(box, saved.get(key))
            for box, key in (
                (self.lane_sigma, "sigma"),
                (self.lane_distance, "distance"),
                (self.lane_height, "height"),
                (self.lane_window, "orientation_window_um"),
                (self.lane_orientation, "orientation"),
            )
        )

    def _trenches_match(self) -> bool:
        saved = self.trench_detector.detection_params
        if self.det_channel.currentData() != self.trench_detector.channel:
            return False
        if not self._mean_matches(self.trench_mean_n, self.trench_mean_from, saved):
            return False
        return all(
            self._same(box, saved.get(key))
            for box, key in (
                (self.trench_method, "method"),
                (self.trench_sigma, "sigma"),
                (self.trench_distance, "distance"),
                (self.trench_prominence, "prominence"),
                (self.trench_pitch, "expected_pitch"),
                (self.trench_symmetry, "min_symmetry"),
                (self.trench_width, "trench_width"),
                (self.trench_length, "trench_length"),
                (self.trench_offset, "trench_bottom_offset"),
            )
        )

    def _show_extraction(self) -> None:
        """Say on the Preview tab whether the experiment group was cut with the
        registration and trenches saved now.
        """
        if self.store is None:
            return
        self.lineage = self.store.lineage(self.experiment, self.experiment_group)
        state = self.lineage["extraction"]
        text = {
            lineage.CURRENT: "extracted with the registration and trenches saved now",
            lineage.STALE: (
                "extracted with another registration or other trenches: extract "
                "again, with resume=False or to a new path"
            ),
            lineage.MISSING: "not extracted yet",
            lineage.UNRECORDED: (
                "extracted before PyMMM recorded what an extraction was made from"
            ),
        }[state]
        self.extraction_info.setText(
            f"Experiment group {self.experiment_group}: {text}."
        )

    def _say(self, text: str) -> None:
        self.status.setText(text)

    # ------------------------------------------------------------------
    # Experiment
    # ------------------------------------------------------------------

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Experiment file or ND2",
            str(Path.cwd()),
            "Experiments and ND2s (*.toml *.nd2)",
        )
        if path:
            self.path_edit.setText(path)
            self.open_experiment()

    def open_experiment(self) -> None:
        """Open the ND2 the experiment file names, with the tab's subset, and load the
        saved registration, lanes and trenches that match it.
        """
        from pymmm import CompanionStore, ND2Experiment
        from pymmm.config import Experiment

        path = Path(self.path_edit.text().strip()).expanduser()
        try:
            if path.suffix.lower() == ".nd2":
                # as given, not resolved: the companion store goes beside this path
                nd2 = path.absolute()
                if not nd2.exists():
                    raise FileNotFoundError(nd2)
                group = nd2.parent / f"{nd2.stem}.zarr"  # the Extractor's default
            else:
                config = Experiment.load(path)
                nd2, group = config.nd2, config.store
                if nd2 is None:
                    raise ValueError(f"{path} names no ND2")
            if self._opened_file != path.absolute():
                # a new file: its saved settings fill the tabs
                self._opened_file = path.absolute()
                self._fill_from_saved(nd2)
            experiment = ND2Experiment(nd2)
            if keep := parse_fovs(self.keep_edit.text()):
                experiment.select_fovs(keep)
            if drop := parse_fovs(self.discard_edit.text()):
                experiment.discard_fovs(drop)
            if times := parse_range(self.times_edit.text()):
                experiment.select_times(*times)
            y_roi, x_roi = (
                parse_range(self.y_roi_edit.text()),
                parse_range(self.x_roi_edit.text()),
            )
            if y_roi or x_roi:
                experiment.select_roi(y=y_roi, x=x_roi)
            if not experiment.fov_names:
                raise ValueError("no FOVs left after keep and discard")
        except Exception as error:  # a bad field or file: say so, keep what was open
            self._say(f"Could not open: {error}")
            return

        self.experiment = experiment
        self.experiment_group = Path(group)
        self._opened_fields = self._experiment_fields()
        self.store = CompanionStore.for_experiment(experiment)
        self.registrator = self.lane_detector = self.trench_detector = None
        channels = list(experiment.channel_names)
        for box in (self.reg_channel, self.det_channel):
            current = box.currentData()
            box.blockSignals(True)
            box.clear()
            for name in channels:
                box.addItem(name, name)
            _select(box, current if current in channels else channels[0])
            box.blockSignals(False)
        if (saved := self._saved_registration()) is not None:
            _select(self.reg_channel, saved.get("channel"))
        self._load_saved()
        self.experiment_info.setText(
            f"<b>{experiment.experiment_name}</b><br>"
            f"{experiment.n_fovs} FOVs × {experiment.n_timepoints} frames × "
            f"{len(channels)} channels ({', '.join(channels)}), "
            f"{experiment.data.sizes['Y']} × {experiment.data.sizes['X']} px, "
            f"{experiment.pixel_size_um:.4f} µm/px<br>"
            f"Companion store: {self.store.path}"
        )
        self._add_image_layers()
        self._enable()
        self._show_extraction()
        steps = [
            name
            for name, done in (
                ("registration", self.registrator is not None),
                ("lanes", self.lane_detector is not None),
                ("trenches", self.trench_detector is not None),
            )
            if done
        ]
        ignored = [
            name
            for section, name in (
                ("lane_detection", "lanes"),
                ("trench_detection", "trenches"),
            )
            if self.lineage.get(section) == lineage.STALE
        ]
        self._say(
            f"Opened {experiment.experiment_name}"
            + (
                f"; loaded the saved {', '.join(steps)}."
                if steps
                else "; nothing saved yet."
            )
            + (
                f" The saved {' and '.join(ignored)} were made from another "
                "registration or other lanes, so they were not loaded: apply them again."
                if ignored
                else ""
            )
        )
        self._draw_drift()

    def _saved_registration(self) -> dict | None:
        if self.store is None or not self.store.has_registration():
            return None
        return self.store.load_registration_params()

    def _fill_from_saved(self, nd2: Path) -> None:
        """Fill the tabs with the settings saved beside ``nd2``, if any: the subset
        and registration of its registration, and its lanes' and trenches' settings.
        """
        from pymmm.store.companion import CompanionStore

        store_path = Path(nd2).parent / f"{Path(nd2).stem}.pymmm.zarr"
        if not store_path.exists():
            return
        store = CompanionStore(store_path, mode="r")
        if store.has_registration():
            saved = store.load_registration_params()
            subset = saved.get("source_subset_metadata", {})
            raw = subset.get("raw_fov_names", [])
            dropped = subset.get("discarded_fov_names", [])
            self.keep_edit.setText("")
            self.discard_edit.setText(", ".join(f for f in raw if f in dropped))
            self.times_edit.setText(format_range(subset.get("time_slice")))
            roi = subset.get("roi", {})
            self.y_roi_edit.setText(format_range(roi.get("y")))
            self.x_roi_edit.setText(format_range(roi.get("x")))
            self.rotation.setValue(float(saved.get("rotation", 0.0)))
            reg_roi = saved.get("roi") or {}
            self.reg_y_edit.setText(format_range(reg_roi.get("y")))
            self.reg_x_edit.setText(format_range(reg_roi.get("x")))
            _select(self.mode, saved.get("mode"))
            _select(self.backend, saved.get("backend", "stackreg"))
            self.downsample.setValue(int(saved.get("downsample", 1)))
            self.batch_size.setValue(int(saved.get("batch_size", 4)))
            # lanes and trenches saved before they recorded their detection image
            # used the registration's
            self._legacy_mean = {
                k: saved.get(k, v) for k, v in self._legacy_mean.items()
            }
        if store.has_lane_detection():
            params = store.load_lane_detection()["params"]
            self._fill_lanes(params)
        if store.has_trench_detection():
            self._fill_trenches(store.load_trench_detection()["params"])

    def _saved_mean(self, params: dict) -> dict[str, Any]:
        """The detection image saved lanes or trenches were found on."""
        return {k: params.get(k, v) for k, v in self._legacy_mean.items()}

    def _fill_lanes(self, params: dict) -> None:
        mean = self._saved_mean(params)
        self.lane_mean_n.setValue(int(mean["mean_n_frames"]))
        _select(self.lane_mean_from, mean["mean_from"])
        _select(self.lane_mode, params.get("image_edge"))
        self.lane_sigma.setValue(float(params.get("sigma", 40)))
        self.lane_distance.setValue(float(params.get("distance", 300)))
        self.lane_height.setValue(float(params.get("height", 5000)))
        _select(self.lane_orientation, params.get("orientation", "auto"))
        self.lane_window.setValue(float(params.get("orientation_window_um", 20)))

    def _fill_trenches(self, params: dict) -> None:
        mean = self._saved_mean(params)
        self.trench_mean_n.setValue(int(mean["mean_n_frames"]))
        _select(self.trench_mean_from, mean["mean_from"])
        _select(self.trench_method, params.get("method", "peaks"))
        self.trench_sigma.setValue(float(params.get("sigma", 4)))
        self.trench_distance.setValue(float(params.get("distance", 100)))
        self.trench_prominence.setValue(float(params.get("prominence", 10)))
        self.trench_pitch.setValue(float(params.get("expected_pitch", 0)))
        self.trench_symmetry.setValue(float(params.get("min_symmetry", 0.1)))
        self.trench_width.setValue(int(params.get("trench_width") or 32))
        if params.get("trench_length") is not None:
            self.trench_length.setValue(int(params["trench_length"]))
        if params.get("trench_bottom_offset") is not None:
            self.trench_offset.setValue(int(params["trench_bottom_offset"]))

    def _load_saved(self) -> None:
        """Load the saved registration if the Register tab's settings made it, then
        the lanes and trenches saved after it (the notebook's checkpoint cells).
        """
        from pymmm import LaneDetector, Registrator, TrenchDetector

        self.lineage = self.store.lineage(self.experiment, self.experiment_group)
        wanted = self._form_registrator()
        if (
            wanted is None
            or not wanted.matches_saved()
            or not lineage.usable(self.lineage["registration"])
        ):
            return
        self.registrator = Registrator.load(self.experiment, self.store)
        if lineage.usable(self.lineage["lane_detection"]):
            self.lane_detector = LaneDetector.load(
                self.experiment, self.registrator, self.store
            )
            _select(self.det_channel, self.lane_detector.channel)
            if lineage.usable(self.lineage["trench_detection"]):
                self.trench_detector = TrenchDetector.load(
                    self.experiment, self.registrator, self.lane_detector, self.store
                )
        self._lane_note()

    # ------------------------------------------------------------------
    # Layers
    # ------------------------------------------------------------------

    def _add_image_layers(self) -> None:
        """Replace this dock's layers with the experiment's: each channel raw (hidden)
        and registered (or only rotated, before registering), and empty overlays.
        """
        for layer in list(self._layers.values()):
            if layer in self.viewer.layers:
                self.viewer.layers.remove(layer)
        self._layers.clear()
        exp = self.experiment
        detection = self.reg_channel.currentData()
        sample_t = exp.n_timepoints // 2
        for channel in exp.channel_names:
            raw = exp.data.sel(C=channel)
            if "Z" in raw.dims:
                raw = raw.isel(Z=0)
            raw = raw.transpose("T", "P", "Y", "X")
            # contrast from the middle frame of the first, middle and last FOV
            fovs = sorted({0, exp.n_fovs // 2, exp.n_fovs - 1})
            limits = contrast_limits(np.asarray(raw.isel(T=sample_t, P=fovs)))
            common = {
                "colormap": channel_colormap(channel)
                if channel != detection
                else "gray",
                "contrast_limits": (limits[0], max(limits[1], limits[0] + 1)),
                "blending": "additive",
                "axis_labels": RAW_AXES,
            }
            self._layers[f"raw {channel}"] = self.viewer.add_image(
                raw.data, name=f"{channel} raw", visible=False, **common
            )
            self._layers[f"registered {channel}"] = self.viewer.add_image(
                raw.data, name=channel, visible=channel == detection, **common
            )
        self._layers["mean"] = self.viewer.add_image(
            np.zeros((exp.data.sizes["Y"], exp.data.sizes["X"]), np.float32),
            name="detection image (registered mean)",
            colormap="gray",
            visible=False,
        )
        shapes = {"face_color": "transparent", "edge_width": 3, "opacity": 0.9}
        self._layers["roi"] = self.viewer.add_shapes(
            name="registration region", edge_color="yellow", visible=False, **shapes
        )
        # thicker: the lanes lie on the bright feeding channel, white at this contrast
        self._layers["lanes"] = self.viewer.add_shapes(
            name="lanes", edge_color="red", visible=False, **{**shapes, "edge_width": 6}
        )
        self._layers["trenches"] = self.viewer.add_shapes(
            name="trenches", edge_color="red", visible=False, **shapes
        )
        self.viewer.dims.axis_labels = RAW_AXES
        _first_step(self.viewer)  # napari opens on the middle frame and FOV
        set_units(self._layers.values(), self.units.currentData(), exp.pixel_size_um)
        self.viewer.layers.selection.active = self._layers[f"registered {detection}"]
        self._refresh_registered()
        self._on_tab(self.tabs.currentIndex())

    def _form_registrator(self):
        """A Registrator with the Register tab's settings (nothing computed)."""
        from pymmm import Registrator

        if self.experiment is None:
            return None
        roi = {}
        try:
            if y := parse_range(self.reg_y_edit.text()):
                roi["y"] = y
            if x := parse_range(self.reg_x_edit.text()):
                roi["x"] = x
        except ValueError as error:
            self._say(f"Registration region: {error}")
            return None
        return Registrator(
            self.experiment,
            self.store,
            registration_channel=self.reg_channel.currentData() or 0,
            mode=self.mode.currentData(),
            # as the box shows it: stepping leaves residue (-1.9000000000000006)
            rotation=round(self.rotation.value(), self.rotation.decimals()),
            roi=roi,
            backend=self.backend.currentData(),
            downsample=self.downsample.value(),
            batch_size=self.batch_size.value(),
            fov_window=2,
            opencv_threads=1,
        )

    def _settings_match(self) -> bool:
        """Whether the Register tab still shows the registration in use."""
        wanted = self._form_registrator()
        return (
            wanted is not None
            and self.registrator is not None
            and wanted.settings() == self.registrator.settings()
        )

    def _refresh_registered(self) -> None:
        """The registered layers: drift-corrected when the registration in use matches
        the Register tab, else the frames rotated by the tab's rotation only.
        """
        if self.experiment is None:
            return
        drift = self._settings_match()
        source = self.registrator if drift else self._form_registrator()
        if source is None:
            return
        for channel in self.experiment.channel_names:
            layer = self._layers[f"registered {channel}"]
            data = source.get_stabilized_data(channel=channel, drift=drift)
            layer.data = data.transpose("T", "P", "Y", "X").data
            layer.name = channel if drift else f"{channel} (rotated, not registered)"
        self._draw_region()

    def _draw_region(self) -> None:
        """The registration region, in the cropped image's pixels."""
        layer = self._layers.get("roi")
        if layer is None:
            return
        exp = self.experiment
        height, width = exp.data.sizes["Y"], exp.data.sizes["X"]
        (y_roi, x_roi) = exp.source_roi
        y0 = (y_roi[0] or 0) if y_roi else 0
        x0 = (x_roi[0] or 0) if x_roi else 0
        try:
            ys = parse_range(self.reg_y_edit.text()) or (y0, y0 + height)
            xs = parse_range(self.reg_x_edit.text()) or (x0, x0 + width)
        except ValueError:
            return
        top, bottom = max(ys[0] - y0, 0), min(ys[1] - y0, height)
        left, right = max(xs[0] - x0, 0), min(xs[1] - x0, width)
        _set_shapes(
            layer, [np.array([[top, left], [bottom, right]])], "rectangle", ["yellow"]
        )

    def _on_registration_settings(self) -> None:
        self._refresh_registered()
        self._enable()
        if self.registrator is not None and not self._settings_match():
            self.reg_status.setText(
                "The settings differ from the saved registration: the layers show the "
                "frames only rotated, and the later steps wait. Register again to use "
                "them (the lanes and trenches are then redone), or set them back."
            )
        elif self.registrator is not None:
            self.reg_status.setText("Registered with these settings.")

    def _on_tab(self, index: int) -> None:
        """Show the overlays of the tab's step."""
        name = self.TAB_NAMES[index]
        shown = {
            "Register": {"roi"},
            "Lanes": {"mean", "lanes"},
            "Trenches": {"mean", "lanes", "trenches"},
        }.get(name, set())
        for key in ("roi", "mean", "lanes", "trenches"):
            if key in self._layers:
                self._layers[key].visible = key in shown
        if name == "Lanes":
            self.update_lanes()
        elif name == "Trenches":
            self.update_trenches()
        elif name == "Register":
            self._draw_drift()

    def _showing(self, tab: str) -> bool:
        return self.TAB_NAMES[self.tabs.currentIndex()] == tab

    def current_fov(self) -> str | None:
        """The FOV on the P slider."""
        if self.experiment is None:
            return None
        labels = list(self.viewer.dims.axis_labels)
        if "P" not in labels:
            return self.experiment.fov_names[0]
        step = self.viewer.dims.current_step[labels.index("P")]
        names = self.experiment.fov_names
        return names[min(max(int(step), 0), len(names) - 1)]

    def current_frame(self) -> int:
        """The frame on the T slider."""
        labels = list(self.viewer.dims.axis_labels)
        if "T" not in labels:
            return 0
        return int(self.viewer.dims.current_step[labels.index("T")])

    def _go_to_frame(self, frame: int) -> None:
        """Move the T slider to a frame (a click on the drift plot)."""
        labels = list(self.viewer.dims.axis_labels)
        if "T" in labels:
            self.viewer.dims.set_current_step(labels.index("T"), frame)

    def _on_step(self, _event=None) -> None:
        self.drift_plot.show_frame(self.current_frame())
        fov = self.current_fov()
        if fov == self._last_fov:
            return
        self._last_fov = fov
        self._draw_drift()  # in step on every tab, so it is current when shown
        name = self.TAB_NAMES[self.tabs.currentIndex()]
        if name == "Lanes":
            self.update_lanes()
        elif name == "Trenches":
            self.update_trenches()

    # ------------------------------------------------------------------
    # Background runs
    # ------------------------------------------------------------------

    def _run(self, job, done, what: str, cancel: threading.Event | None = None):
        """Run ``job()`` off the Qt thread, then ``done(result)`` on it; the worker."""
        from napari.qt.threading import create_worker

        worker = create_worker(job, _start_thread=False)
        self._worker, self._cancel = worker, cancel

        def finished() -> None:
            self._worker = self._cancel = None
            self._enable()

        def failed(error: Exception) -> None:
            self._say(f"{what} stopped: {error}")

        worker.returned.connect(done)
        worker.errored.connect(failed)
        worker.finished.connect(finished)
        self._enable()
        worker.start()
        return worker

    def _cancel_run(self) -> None:
        if self._cancel is not None:
            self._cancel.set()
            self._say("Cancelling: the batches already running finish first…")

    def _with_mean(self, fov: str, then, mean: dict[str, Any]) -> None:
        """``then(fov, image)`` with the FOV's registered mean, the detection image,
        computed in the background the first time (about a second).
        """
        registrator = self.registrator
        channel = self.det_channel.currentData()
        if registrator is None or channel is None:
            return
        frames = tuple(mean_frames(self.experiment.n_timepoints, **mean))
        key = (fov, channel, frames)
        if key in registrator._registered_mean_cache:
            then(fov, registrator._registered_mean_cache[key])
            return
        if key in self._mean_jobs:
            return
        from napari.qt.threading import create_worker

        self._mean_jobs.add(key)
        worker = create_worker(
            registrator.get_registered_mean_of_timestack,
            fov=fov,
            channel=channel,
            **mean,
        )

        def ready(image: np.ndarray) -> None:
            self._mean_jobs.discard(key)
            if registrator is self.registrator and fov == self.current_fov():
                then(fov, image)

        worker.returned.connect(ready)
        worker.errored.connect(lambda error: self._mean_jobs.discard(key))
        self._say(f"Computing {fov}'s registered mean…")
        worker.start()

    # ------------------------------------------------------------------
    # Register
    # ------------------------------------------------------------------

    def register(self) -> None:
        """Compute every FOV's drift with the Register tab's settings, in a process pool,
        and save it; the saved lanes and trenches go, as they belong to the old one.
        """
        registrator = self._form_registrator()
        if registrator is None:
            return
        store = self.store
        cancel = threading.Event()
        progress = _Progress(self)
        progress.changed.connect(self._on_progress)
        self.reg_progress.setRange(0, self.experiment.n_fovs)
        self.reg_progress.setValue(0)
        workers = self.workers.value() or -1
        live = self._live_drift = _Registered(self)
        live.fov.connect(lambda fov, shifts: self._on_fov_registered(live, fov, shifts))

        def job():
            store.clear("registration", "lane_detection", "trench_detection")
            registrator.compute_tmats(
                save_diagnostics=True,
                n_jobs=workers,
                progress=progress.changed.emit,
                cancel=cancel,
                on_fov=lambda fov, tmats: live.fov.emit(
                    str(fov), _shifts(tmats.values)
                ),
            )
            registrator.save()
            return registrator

        def done(result) -> None:
            self._live_drift = None
            self.registrator = result
            self.lane_detector = self.trench_detector = None
            self._refresh_registered()
            self._draw_drift()
            self.reg_status.setText("Registered and saved.")
            self._show_extraction()
            self._say(
                f"Registered {self.experiment.n_fovs} FOVs; saved to {store.path}. "
                f"Drift plots: {self._diagnostics('drift')}"
            )
            self._enable()

        def ended() -> None:  # cancelled or failed: nothing was saved, so no drift
            if self._live_drift is live:
                self._live_drift = None
                self._draw_drift()

        self.registrator = self.lane_detector = self.trench_detector = None
        self.drift_plot.clear()
        self._drift_from = None
        self._draw_drift()
        self.reg_status.setText("Registering…")
        self._say("Registering in a process pool…")
        self._run(job, done, "Registration", cancel=cancel).finished.connect(ended)

    def _on_fov_registered(self, live: _Registered, fov: str, shifts) -> None:
        """Draw a FOV's drift as soon as registration finishes it."""
        if live is not self._live_drift:  # from a run that has since ended
            return
        self.drift_plot.set_drift(fov, shifts)
        self._draw_drift()

    def _on_progress(self, done: int, total: int) -> None:
        self.reg_progress.setRange(0, total)
        self.reg_progress.setValue(done)

    def _draw_drift(self) -> None:
        """Every registered FOV's drift, the one on the P slider in colour; while
        registering, each FOV as it finishes.
        """
        plot = self.drift_plot
        fov = self.current_fov()
        registrator = self.registrator
        if self._live_drift is None:
            if registrator is None or not registrator.is_computed:
                plot.clear()
                self._drift_from = None
            elif self._drift_from is not registrator:
                plot.clear()
                tmats = registrator.tmats
                for name in tmats.coords["P"].values if "P" in tmats.dims else [fov]:
                    mats = tmats.sel(P=name) if "P" in tmats.dims else tmats
                    plot.set_drift(str(name), _shifts(mats.values))
                self._drift_from = registrator
        plot.show_fov(fov)
        plot.show_frame(self.current_frame())
        n_fovs = self.experiment.n_fovs if self.experiment is not None else 0
        done = len(plot.fovs)
        if done and fov not in plot.fovs:
            _note(plot, f"{fov} pending · {done}/{n_fovs}")
        elif self._live_drift is not None:
            latest = f" · latest {plot.fovs[-1]}" if done else ""
            _note(plot, f"{done}/{n_fovs}{latest}")
        else:
            _note(plot, "")

    # ------------------------------------------------------------------
    # Lanes
    # ------------------------------------------------------------------

    def _lane_mean(self) -> dict[str, Any]:
        return {
            "mean_n_frames": self.lane_mean_n.value(),
            "mean_from": self.lane_mean_from.currentData(),
        }

    def _trench_mean(self) -> dict[str, Any]:
        return {
            "mean_n_frames": self.trench_mean_n.value(),
            "mean_from": self.trench_mean_from.currentData(),
        }

    def _lane_settings(self) -> dict[str, Any]:
        return {
            "sigma": self.lane_sigma.value(),
            "distance": self.lane_distance.value(),
            "height": self.lane_height.value(),
            "orientation_window_um": self.lane_window.value(),
            "orientation": self.lane_orientation.currentData(),
        }

    def update_lanes(self) -> None:
        """The candidate lanes of the FOV on the P slider, with the tab's settings."""
        self._mark_tabs()
        fov = self.current_fov()
        if self.registrator is None or fov is None or not self._showing("Lanes"):
            return
        edge = self.lane_mode.currentData()
        for box in (
            self.lane_sigma,
            self.lane_distance,
            self.lane_height,
            self.lane_orientation,
            self.lane_window,
        ):
            box.setEnabled(edge is None)
        self._with_mean(fov, self._draw_lanes, self._lane_mean())

    def _draw_lanes(self, fov: str, image: np.ndarray) -> None:
        from pymmm import LaneDetector

        self._show_mean(image)
        edge = self.lane_mode.currentData()
        height, width = image.shape
        detector = LaneDetector(
            self.experiment,
            self.registrator,
            self.store,
            detection_channel=self.det_channel.currentData(),
        )
        figure = self.lane_canvas.figure
        figure.clear()
        if edge is None:
            settings = self._lane_settings()
            profile, smooth, peaks, lanes = detector.detect_fov(
                image,
                settings["sigma"],
                settings["distance"],
                settings["height"],
                settings["orientation_window_um"],
                fov,
                orientation=settings["orientation"],
            )
            positions = [(lane.y_center, lane.orientation) for lane in lanes]
            ax = figure.add_subplot()
            ys = np.arange(len(profile))
            ax.plot(ys, profile, color="grey", alpha=0.5, label="raw")
            ax.plot(ys, smooth, color="tab:blue", label="smoothed")
            if len(peaks):
                ax.plot(peaks, smooth[peaks], "rv", label="peaks")
            ax.axhline(
                settings["height"], color="tab:red", ls="--", lw=1, label="height"
            )
            ax.set_xlabel("y (px)")
            ax.set_ylabel("mean intensity")
            ax.set_title(f"{fov} y-profile", fontsize=9)
            ax.legend(fontsize=7)
        else:
            positions = [(height if edge == "down" else 0, 1 if edge == "down" else -1)]
        self.lane_canvas.draw_idle()
        lines = [np.array([[y, 0], [y, width - 1]]) for y, _ in positions]
        colours = [
            "red" if orientation == 1 else "cyan" for _, orientation in positions
        ]
        _set_shapes(self._layers["lanes"], lines, "line", colours)
        n = len(positions)
        self.lane_status.setText(f"<b>{fov}</b>: {n} lane{'s' if n != 1 else ''}")

    def apply_lanes(self) -> None:
        """Detect the lanes of every FOV with the tab's settings (or take the image's
        edge) and save them; the saved trenches go, as they were cut from the old lanes.
        """
        from pymmm import LaneDetector

        edge = self.lane_mode.currentData()
        settings = self._lane_settings()
        mean = self._lane_mean()
        detector = LaneDetector(
            self.experiment,
            self.registrator,
            self.store,
            detection_channel=self.det_channel.currentData(),
        )
        store = self.store

        def job():
            if edge is None:
                detector.detect_lanes(**settings, **mean, save_diagnostics=True)
            else:
                detector.use_image_edge(edge, save_diagnostics=True, **mean)
            store.clear("lane_detection", "trench_detection")
            detector.save()
            return detector

        def done(result) -> None:
            self.lane_detector = result
            self.trench_detector = None
            shape = self._trench_detector().shape_defaults()
            if edge is not None:
                self.trench_length.setValue(shape["trench_length"])
                self.trench_offset.setValue(shape["trench_bottom_offset"])
            total = sum(len(v) for v in result.lanes.values())
            self.lane_status.setText(
                f"<b>Saved</b>: {total} lanes across {len(result.lanes)} FOVs"
            )
            self._lane_note()
            self._say(
                f"Lanes saved; overlays in {self._diagnostics('lanes')}. "
                "Next: the Trenches tab."
            )
            self._enable()
            self._show_extraction()

        self._say("Detecting the lanes of every FOV…")
        self._run(job, done, "Lane detection")

    # ------------------------------------------------------------------
    # Trenches
    # ------------------------------------------------------------------

    def _trench_detector(self):
        from pymmm import TrenchDetector

        return TrenchDetector(
            self.experiment,
            self.registrator,
            self.lane_detector,
            self.store,
            detection_channel=self.det_channel.currentData(),
        )

    def _lane_overrides(self) -> dict:
        """Per-lane settings saved with the trenches (set in the notebook's widget):
        kept when the trenches are detected again here.
        """
        if self.trench_detector is None:
            return {}
        return self.trench_detector.detection_params.get("lane_params", {}) or {}

    def _lane_note(self) -> None:
        overrides = self._lane_overrides()
        self.trench_lane_note.setText(
            f"Per-lane settings saved earlier are kept: {overrides}"
            if overrides
            else ""
        )

    def _trench_settings(self) -> dict[str, Any]:
        saved = self.trench_detector.detection_params if self.trench_detector else {}
        return {
            "sigma": self.trench_sigma.value(),
            "distance": self.trench_distance.value(),
            "prominence": self.trench_prominence.value(),
            "height": saved.get("height", 0),
            "trench_width": self.trench_width.value(),
            "trench_length": self.trench_length.value(),
            "trench_bottom_offset": self.trench_offset.value(),
            "lane_params": self._lane_overrides(),
            "method": self.trench_method.currentData(),
            "expected_pitch": self.trench_pitch.value(),
            "min_symmetry": self.trench_symmetry.value(),
        }

    def update_trenches(self) -> None:
        """The candidate trenches of the FOV on the P slider, with the tab's settings."""
        self._mark_tabs()
        fov = self.current_fov()
        if self.lane_detector is None or fov is None or not self._showing("Trenches"):
            return
        periodic = self.trench_method.currentData() == "periodic"
        self.trench_distance.setEnabled(not periodic)
        self.trench_prominence.setEnabled(not periodic)
        self.trench_pitch.setEnabled(periodic)
        self.trench_symmetry.setEnabled(periodic)
        self._with_mean(fov, self._draw_trenches, self._trench_mean())

    def _draw_trenches(self, fov: str, image: np.ndarray) -> None:
        from pymmm.widgets.trenches import _plot_lane_profile

        self._show_mean(image)
        detector = self._trench_detector()
        settings = self._trench_settings()
        pitch_note = ""
        if settings["method"] == "periodic" and not settings["expected_pitch"]:
            pitch = self._experiment_pitch(detector, settings)
            if pitch is None:
                pitch_note = (
                    "<br>Spacing: this FOV's own estimate, while the experiment's is "
                    "estimated from every FOV"
                )
            else:
                settings = {**settings, "expected_pitch": pitch}
                pitch_note = (
                    f"<br>Spacing {pitch:.1f} px, estimated across all "
                    f"{self.experiment.n_fovs} FOVs (as Apply does)"
                )
        trenches, profiles, _ = detector.detect_fov(image, fov, **settings)
        boxes = [
            np.array([[t.y_top, t.x_left], [t.y_bottom, t.x_right]]) for t in trenches
        ]
        _set_shapes(self._layers["trenches"], boxes, "rectangle", ["red"] * len(boxes))
        lanes = self.lane_detector.lanes.get(fov, [])
        lines = [
            np.array([[lane.y_center, 0], [lane.y_center, image.shape[1] - 1]])
            for lane in lanes
        ]
        _set_shapes(
            self._layers["lanes"],
            lines,
            "line",
            ["red" if lane.orientation == 1 else "cyan" for lane in lanes],
        )
        figure = self.trench_canvas.figure
        figure.clear()
        if profiles:
            axes = figure.subplots(len(profiles), 1, squeeze=False)
            for ax, profile in zip(axes[:, 0], profiles):
                _plot_lane_profile(ax, profile, settings["min_symmetry"])
                ax.title.set_fontsize(9)
                ax.legend(fontsize=7, loc="upper left")
        self.trench_canvas.draw_idle()
        n = len(trenches)
        outside = [p["outside"] for p in profiles if "outside" in p]
        self.trench_status.setText(
            f"<b>{fov}</b>: {n} trench{'es' if n != 1 else ''}"
            + "".join(f"<br>No trenches kept for the {note}" for note in outside)
            + pitch_note
        )

    def _experiment_pitch(self, detector, settings: dict[str, Any]) -> float | None:
        """The trench spacing across every FOV, as Apply finds it, for the preview
        (:meth:`TrenchDetector.experiment_pitch`). It needs every FOV's detection
        image: the first time, they are computed in the background and the preview
        redrawn when they are in; None meanwhile.
        """
        registrator, channel = self.registrator, self.det_channel.currentData()
        mean = self._trench_mean()
        frames = tuple(mean_frames(self.experiment.n_timepoints, **mean))
        fovs = self.experiment.fov_names
        cache = registrator._registered_mean_cache
        keys = [(fov, channel, frames) for fov in fovs]
        if not all(key in cache for key in keys):
            job = (registrator, channel, frames)
            if self._pitch_job != job:
                from napari.qt.threading import create_worker

                self._pitch_job = job
                worker = create_worker(
                    registrator.compute_registered_means, fovs, channel, **mean
                )
                worker.returned.connect(lambda _: self._trench_timer.start())
                worker.finished.connect(lambda: setattr(self, "_pitch_job", None))
                worker.start()
                self._say(
                    f"Computing the detection images of all {len(fovs)} FOVs, to "
                    "estimate the trench spacing across the experiment…"
                )
            return None
        shape = (
            settings["sigma"],
            settings["trench_width"],
            settings["trench_length"],
            settings["trench_bottom_offset"],
            repr(settings["lane_params"]),
        )
        key = (registrator, channel, frames, shape)
        if key not in self._pitches:
            self._pitches[key] = detector.experiment_pitch(
                {fov: cache[k] for fov, k in zip(fovs, keys)},
                settings["sigma"],
                settings["trench_width"],
                settings["trench_length"],
                settings["trench_bottom_offset"],
                lane_params=settings["lane_params"],
            )
        return self._pitches[key]

    def apply_trenches(self) -> None:
        """Detect the trenches of every FOV with the tab's settings and save them."""
        detector = self._trench_detector()
        settings = self._trench_settings()
        mean = self._trench_mean()

        def job():
            detector.detect_trenches(**settings, **mean, save_diagnostics=True)
            detector.save()
            return detector

        def done(result) -> None:
            self.trench_detector = result
            self.trench_status.setText(
                f"<b>Saved</b>: {result.n_trenches} trenches across "
                f"{len(result.trenches)} FOVs"
                + "".join(
                    f"<br><b>{fov}</b>: no trenches kept for the {'; '.join(notes)}"
                    for fov, notes in result.outside.items()
                )
            )
            self._say(
                f"Trenches saved; overlays in {self._diagnostics('trenches')}. "
                "Next: the Preview tab."
            )
            self._enable()
            self._show_extraction()

        self._say("Detecting the trenches of every FOV…")
        self._run(job, done, "Trench detection")

    def _on_units(self) -> None:
        """Show both viewers in the chosen units (the settings stay in pixels)."""
        if self.experiment is None:
            return
        unit, size = self.units.currentData(), self.experiment.pixel_size_um
        keep_view(self.viewer, list(self._layers.values()), unit, size)
        preview = self.preview_viewer
        if preview is not None and len(preview.layers):
            keep_view(preview, list(preview.layers), unit, size)

    def _diagnostics(self, kind: str) -> Path:
        """Where the steps save their diagnostic figures: beside the ND2, as from
        the notebook.
        """
        from pymmm._utils import get_diagnostics_dir

        return get_diagnostics_dir(self.experiment.path, kind)

    def _show_mean(self, image: np.ndarray) -> None:
        layer = self._layers["mean"]
        layer.data = image
        lo, hi = contrast_limits(image)
        layer.contrast_limits = (lo, max(hi, lo + 1))

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def open_preview(self, viewer=None):
        """The trenches as ``extract()`` would cut them, in a new viewer (or
        ``viewer``): one layer per channel, ``(trench, t, y, x)``, cut one frame at a
        time as the sliders move.
        """
        from pymmm import Extractor
        from pymmm.view import show_position

        exp = self.experiment
        extractor = Extractor(
            exp, self.registrator, self.trench_detector, output_path=Path("unused.zarr")
        )
        detection = self.det_channel.currentData()
        if viewer is None:
            import napari

            viewer = napari.Viewer(title=f"{exp.experiment_name}: trench preview")
        frames = [int(t) for t in exp.source_axes["original_t_index"]]
        for channel in exp.channel_names:
            array = extractor.preview_array(channel)
            sample = np.asarray(array[0, array.shape[1] // 2])
            lo, hi = contrast_limits(sample)
            viewer.add_image(
                array,
                name=channel,
                colormap="gray" if channel == detection else channel_colormap(channel),
                contrast_limits=(lo, max(hi, lo + 1)),
                blending="additive",
                visible=channel == detection,
                axis_labels=PREVIEW_AXES,
                metadata={
                    "pymmm": {
                        "trenches": array.trench_ids,
                        "fovs": array.fov_names,
                        "frames": frames,
                    }
                },
            )
        viewer.layers.selection.active = viewer.layers[detection]
        set_units(viewer.layers, self.units.currentData(), exp.pixel_size_um)
        _first_step(viewer)  # the first trench, at the first frame
        show_position(viewer)
        show_coordinate_grid(viewer)
        self.preview_viewer = viewer
        n = len(self.trench_detector.get_trench_table())
        self.preview_info.setText(
            f"{n} trenches of {array.shape[2]} × {array.shape[3]} px, "
            f"{exp.n_timepoints} frames. Nothing is written; extract() in the notebook "
            "writes them."
        )
        return viewer


def _first_step(viewer) -> None:
    """Put the viewer's sliders (every axis but the two shown) at their start."""
    for axis in range(viewer.dims.ndim - 2):
        viewer.dims.set_current_step(axis, 0)


def _set_shapes(layer, shapes: list[np.ndarray], kind: str, colours: list[str]) -> None:
    """Replace a shapes layer's shapes."""
    layer.selected_data = set()
    if len(layer.data):
        layer.selected_data = set(range(len(layer.data)))
        layer.remove_selected()
    if shapes:
        layer.add(shapes, shape_type=kind, edge_color=colours, face_color="transparent")
