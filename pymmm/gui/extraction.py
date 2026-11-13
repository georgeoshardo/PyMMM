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
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
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
        self._opened_file: Path | None = None

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
        self.status = QLabel("Open an experiment file.")
        self.status.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)
        layout.addWidget(self.status)
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
                ("mean of frames", "mean"),
                ("last frame", "last"),
                ("previous frame", "previous"),
            ],
            "first",
        )
        self.backend = _combo(
            [("stackreg", "stackreg"), ("opencv (fast)", "opencv")], "stackreg"
        )
        self.downsample = _spin(4, 1, 16)
        self.mean_n_frames = _spin(10, 1, 500)
        self.mean_from = _combo([("end", "end"), ("start", "start")], "end")
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
        form.addRow("mean of n frames", self.mean_n_frames)
        form.addRow("mean from", self.mean_from)
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
        self.drift_canvas = _canvas(2.2)
        form.addRow(self.drift_canvas)

        self._rotation_timer = self._timer(self._on_registration_settings)
        for box in (self.rotation, self.downsample, self.mean_n_frames):
            box.valueChanged.connect(self._rotation_timer.start)
        for box in (self.reg_channel, self.mode, self.backend, self.mean_from):
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
        form.addRow("channel", self.det_channel)
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
        ):
            box.valueChanged.connect(self._lane_timer.start)
        for box in (self.lane_mode, self.lane_orientation):
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
        ):
            box.valueChanged.connect(self._trench_timer.start)
        self.trench_method.currentIndexChanged.connect(self._trench_timer.start)
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
        layout.addWidget(self.preview_button)
        layout.addWidget(self.preview_info)
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
        registered = self.registrator is not None
        lanes = self.lane_detector is not None
        trenches = self.trench_detector is not None and self.trench_detector.is_detected
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
            else:
                nd2 = Experiment.load(path).nd2
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
        except Exception as error:  # a bad field or file: say so, keep what was open
            self._say(f"Could not open: {error}")
            return

        self.experiment = experiment
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
        steps = [
            name
            for name, done in (
                ("registration", self.registrator is not None),
                ("lanes", self.lane_detector is not None),
                ("trenches", self.trench_detector is not None),
            )
            if done
        ]
        self._say(
            f"Opened {experiment.experiment_name}"
            + (
                f"; loaded the saved {', '.join(steps)}."
                if steps
                else "; nothing saved yet."
            )
        )
        if self.registrator is not None:
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
            self.mean_n_frames.setValue(int(saved.get("mean_n_frames", 10)))
            _select(self.mean_from, saved.get("mean_from", "end"))
            self.batch_size.setValue(int(saved.get("batch_size", 4)))
        if store.has_lane_detection():
            params = store.load_lane_detection()["params"]
            self._fill_lanes(params)
        if store.has_trench_detection():
            self._fill_trenches(store.load_trench_detection()["params"])

    def _fill_lanes(self, params: dict) -> None:
        _select(self.lane_mode, params.get("image_edge"))
        self.lane_sigma.setValue(float(params.get("sigma", 40)))
        self.lane_distance.setValue(float(params.get("distance", 300)))
        self.lane_height.setValue(float(params.get("height", 5000)))
        _select(self.lane_orientation, params.get("orientation", "auto"))
        self.lane_window.setValue(float(params.get("orientation_window_um", 20)))

    def _fill_trenches(self, params: dict) -> None:
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

        wanted = self._form_registrator()
        if wanted is None or not wanted.matches_saved():
            return
        self.registrator = Registrator.load(self.experiment, self.store)
        if self.store.has_lane_detection():
            self.lane_detector = LaneDetector.load(
                self.experiment, self.registrator, self.store
            )
            _select(self.det_channel, self.lane_detector.channel)
            if self.store.has_trench_detection():
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
            rotation=float(self.rotation.value()),
            roi=roi,
            mean_n_frames=self.mean_n_frames.value(),
            mean_from=self.mean_from.currentData(),
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
        if self.registrator is not None and not self._settings_match():
            self.reg_status.setText(
                "The settings differ from the saved registration: the layers show the "
                "frames only rotated. Register again to use them (the lanes and "
                "trenches are then redone)."
            )
        elif self.registrator is not None:
            self.reg_status.setText("Registered with these settings.")

    def _on_tab(self, index: int) -> None:
        """Show the overlays of the tab's step."""
        name = self.tabs.tabText(index)
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
        return self.tabs.tabText(self.tabs.currentIndex()) == tab

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

    def _on_step(self, _event=None) -> None:
        fov = self.current_fov()
        if fov == self._last_fov:
            return
        self._last_fov = fov
        name = self.tabs.tabText(self.tabs.currentIndex())
        if name == "Register":
            self._draw_drift()
        elif name == "Lanes":
            self.update_lanes()
        elif name == "Trenches":
            self.update_trenches()

    # ------------------------------------------------------------------
    # Background runs
    # ------------------------------------------------------------------

    def _run(self, job, done, what: str, cancel: threading.Event | None = None) -> None:
        """Run ``job()`` off the Qt thread, then ``done(result)`` on it."""
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

    def _cancel_run(self) -> None:
        if self._cancel is not None:
            self._cancel.set()
            self._say("Cancelling: the batches already running finish first…")

    def _with_mean(self, fov: str, then) -> None:
        """``then(fov, image)`` with the FOV's registered mean, the detection image,
        computed in the background the first time (about a second).
        """
        registrator = self.registrator
        channel = self.det_channel.currentData()
        if registrator is None or channel is None:
            return
        key = (fov, channel)
        if key in registrator._registered_mean_cache:
            then(fov, registrator._registered_mean_cache[key])
            return
        if key in self._mean_jobs:
            return
        from napari.qt.threading import create_worker

        self._mean_jobs.add(key)
        worker = create_worker(
            registrator.get_registered_mean_of_timestack, fov=fov, channel=channel
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

        def job():
            store.clear("registration", "lane_detection", "trench_detection")
            registrator.compute_tmats(
                plot=False,
                n_jobs=workers,
                progress=progress.changed.emit,
                cancel=cancel,
            )
            registrator.save()
            return registrator

        def done(result) -> None:
            self.registrator = result
            self.lane_detector = self.trench_detector = None
            self._refresh_registered()
            self._draw_drift()
            self.reg_status.setText("Registered and saved.")
            self._say(
                f"Registered {self.experiment.n_fovs} FOVs; saved to {store.path}."
            )
            self._enable()

        self.registrator = self.lane_detector = self.trench_detector = None
        self.reg_status.setText("Registering…")
        self._say("Registering in a process pool…")
        self._run(job, done, "Registration", cancel=cancel)

    def _on_progress(self, done: int, total: int) -> None:
        self.reg_progress.setRange(0, total)
        self.reg_progress.setValue(done)

    def _draw_drift(self) -> None:
        """The drift of the FOV on the P slider: its x and y shift at each frame."""
        figure = self.drift_canvas.figure
        figure.clear()
        fov = self.current_fov()
        if self.registrator is None or fov is None:
            self.drift_canvas.draw_idle()
            return
        tmats = self.registrator.tmats
        mats = (tmats.sel(P=fov) if "P" in tmats.dims else tmats).values
        ax = figure.add_subplot()
        ax.plot(mats[:, 0, 2], label="x")
        ax.plot(mats[:, 1, 2], label="y")
        ax.set_xlabel("frame")
        ax.set_ylabel("shift (px)")
        ax.set_title(f"{fov} drift", fontsize=9)
        ax.legend(fontsize=8)
        self.drift_canvas.draw_idle()

    # ------------------------------------------------------------------
    # Lanes
    # ------------------------------------------------------------------

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
        self._with_mean(fov, self._draw_lanes)

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
        self._say(
            "Red: trenches above the lane, opening down; cyan: below, opening up. "
            "Move the P slider to check other FOVs."
        )

    def apply_lanes(self) -> None:
        """Detect the lanes of every FOV with the tab's settings (or take the image's
        edge) and save them; the saved trenches go, as they were cut from the old lanes.
        """
        from pymmm import LaneDetector

        edge = self.lane_mode.currentData()
        settings = self._lane_settings()
        detector = LaneDetector(
            self.experiment,
            self.registrator,
            self.store,
            detection_channel=self.det_channel.currentData(),
        )
        store = self.store

        def job():
            if edge is None:
                detector.detect_lanes(**settings, save_diagnostics=False)
            else:
                detector.use_image_edge(edge, save_diagnostics=False)
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
            self._say("Lanes saved. Next: the Trenches tab.")
            self._enable()

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
        fov = self.current_fov()
        if self.lane_detector is None or fov is None or not self._showing("Trenches"):
            return
        periodic = self.trench_method.currentData() == "periodic"
        self.trench_distance.setEnabled(not periodic)
        self.trench_prominence.setEnabled(not periodic)
        self.trench_pitch.setEnabled(periodic)
        self.trench_symmetry.setEnabled(periodic)
        self._with_mean(fov, self._draw_trenches)

    def _draw_trenches(self, fov: str, image: np.ndarray) -> None:
        from pymmm.widgets.trenches import _plot_lane_profile

        self._show_mean(image)
        detector = self._trench_detector()
        settings = self._trench_settings()
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
        self.trench_status.setText(f"<b>{fov}</b>: {n} trench{'es' if n != 1 else ''}")
        self._say("Move the P slider to check other FOVs, then apply.")

    def apply_trenches(self) -> None:
        """Detect the trenches of every FOV with the tab's settings and save them."""
        detector = self._trench_detector()
        settings = self._trench_settings()

        def job():
            detector.detect_trenches(**settings, save_diagnostics=False)
            detector.save()
            return detector

        def done(result) -> None:
            self.trench_detector = result
            self.trench_status.setText(
                f"<b>Saved</b>: {result.n_trenches} trenches across "
                f"{len(result.trenches)} FOVs"
            )
            self._say("Trenches saved. Next: the Preview tab.")
            self._enable()

        self._say("Detecting the trenches of every FOV…")
        self._run(job, done, "Trench detection")

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
        show_position(viewer)
        self.preview_viewer = viewer
        n = len(self.trench_detector.get_trench_table())
        self.preview_info.setText(
            f"{n} trenches of {array.shape[2]} × {array.shape[3]} px, "
            f"{exp.n_timepoints} frames. Nothing is written; extract() in the notebook "
            "writes them."
        )
        return viewer


def _set_shapes(layer, shapes: list[np.ndarray], kind: str, colours: list[str]) -> None:
    """Replace a shapes layer's shapes."""
    layer.selected_data = set()
    if len(layer.data):
        layer.selected_data = set(range(len(layer.data)))
        layer.remove_selected()
    if shapes:
        layer.add(shapes, shape_type=kind, edge_color=colours, face_color="transparent")
