"""LaneDetector – detect feeding-lane y-positions in each FOV."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from pymmm._utils import normalize_channel_arg, normalize_fov_arg
from pymmm.checkpoint import CompanionStore
from pymmm.experiment import ND2Experiment
from pymmm.registrator import Registrator


@dataclass
class LaneInfo:
    """Describes one feeding lane detected in a FOV."""

    y_center: int
    orientation: int  # +1 = trenches open downward, -1 = trenches open upward
    fov: str
    lane_index: int


class LaneDetector:
    """Detect feeding-lane y-positions in each FOV.

    Handles multiple lanes per FOV and detects trench orientation.

    Parameters
    ----------
    experiment : ND2Experiment
        Source experiment.
    registrator : Registrator
        Completed registrator (for registered mean images).
    store : CompanionStore
        Companion zarr store for checkpointing.
    detection_channel : str | int
        Channel to use for lane detection (typically phase contrast).
    """

    def __init__(
        self,
        experiment: ND2Experiment,
        registrator: Registrator,
        store: CompanionStore,
        detection_channel: Union[str, int] = 0,
    ) -> None:
        self.experiment = experiment
        self.registrator = registrator
        self.store = store
        self.channel = normalize_channel_arg(
            detection_channel, experiment.channel_names
        )
        self._lanes: Optional[Dict[str, List[LaneInfo]]] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def lanes(self) -> Dict[str, List[LaneInfo]]:
        if self._lanes is None:
            raise RuntimeError("Lanes not detected yet. Call detect_lanes() first.")
        return self._lanes

    @property
    def is_detected(self) -> bool:
        return self._lanes is not None

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _detect_lanes_single_fov(
        self,
        mean_img: np.ndarray,
        sigma: float,
        distance: float,
        height: float,
        orientation_window_um: float,
        fov: str,
    ) -> tuple:
        """Run lane detection on a single FOV mean image.

        Returns
        -------
        y_profile : np.ndarray
        y_profile_smooth : np.ndarray
        peaks : np.ndarray
        fov_lanes : list[LaneInfo]
        """
        pixel_size = self.experiment.pixel_size_um
        window_px = max(1, int(round(orientation_window_um / pixel_size)))

        # 1D y-profile: average along X
        y_profile = mean_img.mean(axis=1)

        # Gaussian smoothing
        if sigma > 0:
            y_profile_smooth = gaussian_filter1d(y_profile, sigma)
        else:
            y_profile_smooth = y_profile.copy()

        # Find peaks
        peaks, _ = find_peaks(y_profile_smooth, distance=distance, height=height)

        # Orientation detection
        fov_lanes: List[LaneInfo] = []
        for lane_idx, peak_y in enumerate(peaks):
            above_start = max(0, peak_y - window_px)
            above_end = peak_y
            below_start = peak_y
            below_end = min(len(y_profile), peak_y + window_px)

            mean_above = y_profile[above_start:above_end].mean()
            mean_below = y_profile[below_start:below_end].mean()

            orientation = 1 if mean_above > mean_below else -1

            fov_lanes.append(
                LaneInfo(
                    y_center=int(peak_y),
                    orientation=orientation,
                    fov=fov,
                    lane_index=lane_idx,
                )
            )

        return y_profile, y_profile_smooth, peaks, fov_lanes

    def detect_lanes(
        self,
        sigma: float = 40.0,
        distance: float = 300,
        height: float = 5000,
        orientation_window_um: float = 20.0,
        plot: bool = False,
    ) -> None:
        """Detect lane y-positions in all FOVs.

        Algorithm:
        1. Get registered mean-of-timestack for detection channel.
        2. Average along X → 1-D y-profile.
        3. Gaussian smoothing.
        4. ``find_peaks()`` → lane centre positions.
        5. Orientation detection: compare mean intensity above vs below each peak.

        Parameters
        ----------
        sigma : float
            Gaussian smoothing sigma for the y-profile.
        distance : float
            Minimum distance between peaks (in pixels).
        height : float
            Minimum peak height.
        orientation_window_um : float
            Window size (in µm) above/below peak for orientation detection.
        plot : bool
            If ``True``, show lane overlay for the first FOV.
        """
        lanes: Dict[str, List[LaneInfo]] = {}

        for fov in self.experiment.fov_names:
            mean_img = self.registrator.get_registered_mean_of_timestack(
                fov=fov, channel=self.channel
            )
            _, _, _, fov_lanes = self._detect_lanes_single_fov(
                mean_img, sigma, distance, height, orientation_window_um, fov
            )
            lanes[fov] = fov_lanes

        self._lanes = lanes

        print(
            f"Detected lanes: "
            + ", ".join(f"{fov}: {len(ls)}" for fov, ls in lanes.items())
        )

        if plot:
            self._plot_first_fov()

    def _plot_first_fov(self) -> None:
        """Plot lane overlay for the first FOV."""
        from pymmm._utils import get_diagnostics_dir
        from pymmm.plotting import plot_fov_with_lanes

        diag_dir = get_diagnostics_dir(self.experiment.path)
        fov = self.experiment.fov_names[0]
        mean_img = self.registrator.get_registered_mean_of_timestack(
            fov=fov, channel=self.channel
        )
        fov_lanes = self._lanes[fov]
        y_positions = [l.y_center for l in fov_lanes]
        orientations = [l.orientation for l in fov_lanes]
        plot_fov_with_lanes(
            mean_img,
            y_positions,
            orientations=orientations,
            title=f"Lanes – {fov}",
            save_path=str(diag_dir / f"lanes_{fov}.png"),
        )

    def plot_fov(self, fov: Union[int, str] = 0) -> None:
        """Plot lane overlay for any FOV."""
        from pymmm.plotting import plot_fov_with_lanes

        fov_name = normalize_fov_arg(fov, self.experiment.fov_names)
        mean_img = self.registrator.get_registered_mean_of_timestack(
            fov=fov_name, channel=self.channel
        )
        fov_lanes = self._lanes[fov_name]
        y_positions = [l.y_center for l in fov_lanes]
        orientations = [l.orientation for l in fov_lanes]
        plot_fov_with_lanes(
            mean_img,
            y_positions,
            orientations=orientations,
            title=f"Lanes – {fov_name}",
        )

    # ------------------------------------------------------------------
    # Interactive widget
    # ------------------------------------------------------------------

    def interactive_detect_lanes(self) -> None:
        """Display an interactive widget for lane detection tuning.

        Uses ipywidgets for controls and matplotlib for plots, which
        work reliably in VSCode notebooks (Panel/Bokeh widgets suffer
        from UnknownReferenceError in VSCode's jupyter_bokeh bridge).
        """
        import ipywidgets as widgets
        import matplotlib.pyplot as plt
        from IPython.display import display

        mean_images: Dict[str, np.ndarray] = {}

        def _mean_image(fov: str) -> np.ndarray:
            if fov not in mean_images:
                mean_images[fov] = self.registrator.get_registered_mean_of_timestack(
                    fov=fov, channel=self.channel
                )
            return mean_images[fov]

        # --- Widgets ---
        fov_select = widgets.Dropdown(
            options=list(self.experiment.fov_names),
            value=self.experiment.fov_names[0],
            description="FOV",
        )
        sigma_slider = widgets.FloatSlider(
            value=40, min=0, max=200, step=1, description="sigma",
            continuous_update=False, style={"description_width": "initial"},
        )
        distance_slider = widgets.FloatSlider(
            value=300, min=10, max=1000, step=10, description="distance",
            continuous_update=False, style={"description_width": "initial"},
        )
        height_slider = widgets.FloatSlider(
            value=5000, min=0, max=50000, step=100, description="height",
            continuous_update=False, style={"description_width": "initial"},
        )
        orientation_slider = widgets.FloatSlider(
            value=20, min=1, max=100, step=1, description="orient. window (µm)",
            continuous_update=False, style={"description_width": "initial"},
        )
        apply_btn = widgets.Button(
            description="Apply to all FOVs", button_style="primary",
        )
        status_label = widgets.HTML(value="")
        output = widgets.Output()

        def _update(_change: Any = None) -> None:
            fov = fov_select.value
            mean_img = _mean_image(fov)
            y_profile, y_profile_smooth, peaks, fov_lanes = self._detect_lanes_single_fov(
                mean_img,
                sigma_slider.value,
                distance_slider.value,
                height_slider.value,
                orientation_slider.value,
                fov,
            )
            n_lanes = len(fov_lanes)

            with output:
                output.clear_output(wait=True)
                fig, (ax_img, ax_prof) = plt.subplots(1, 2, figsize=(14, 5))

                # Image + lane overlays
                ax_img.imshow(mean_img, cmap="gray", aspect="auto")
                for lane in fov_lanes:
                    ori_label = "down" if lane.orientation == 1 else "up"
                    ax_img.axhline(lane.y_center, color="red", lw=1.5)
                    ax_img.text(
                        5, lane.y_center - 5, ori_label,
                        color="red", fontsize=9, va="bottom",
                    )
                ax_img.set_title(
                    f"{n_lanes} lane{'s' if n_lanes != 1 else ''} in {fov}"
                )

                # Y-profile
                xs = np.arange(len(y_profile))
                ax_prof.plot(xs, y_profile, color="grey", alpha=0.5, label="raw")
                ax_prof.plot(xs, y_profile_smooth, color="blue", label="smoothed")
                if len(peaks) > 0:
                    ax_prof.plot(
                        peaks, y_profile_smooth[peaks], "rv",
                        markersize=8, label="peaks",
                    )
                ax_prof.set_xlabel("y pixel")
                ax_prof.set_ylabel("intensity")
                ax_prof.set_title("Y-profile")
                ax_prof.legend()

                fig.tight_layout()
                plt.show()

        def _on_apply(_btn: Any) -> None:
            status_label.value = "<b>Applying…</b>"
            self.detect_lanes(
                sigma=sigma_slider.value,
                distance=distance_slider.value,
                height=height_slider.value,
                orientation_window_um=orientation_slider.value,
            )
            total = sum(len(v) for v in self._lanes.values())
            status_label.value = (
                f"<b>Applied</b> — {total} lanes across {len(self._lanes)} FOVs"
            )

        apply_btn.on_click(_on_apply)

        for w in [fov_select, sigma_slider, distance_slider,
                  height_slider, orientation_slider]:
            w.observe(_update, names="value")

        controls = widgets.VBox([
            fov_select, sigma_slider, distance_slider,
            height_slider, orientation_slider,
            apply_btn, status_label,
        ])
        display(widgets.HBox([controls, output]))
        _update()

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write lane detection results to the companion zarr store."""
        if self._lanes is None:
            raise RuntimeError("Nothing to save — run detect_lanes() first.")

        # Serialise LaneInfo dataclasses to dicts
        lane_data: Dict[str, list] = {}
        for fov, fov_lanes in self._lanes.items():
            lane_data[fov] = [asdict(li) for li in fov_lanes]

        params = {"channel": self.channel}
        self.store.save_lane_detection(lane_data, params)
        print(f"Lane detection saved to {self.store.path}")

    @classmethod
    def load(
        cls,
        experiment: ND2Experiment,
        registrator: Registrator,
        store: CompanionStore,
    ) -> "LaneDetector":
        """Reconstruct a LaneDetector from a checkpoint."""
        if not store.has_lane_detection():
            raise FileNotFoundError("No lane detection checkpoint found.")

        data = store.load_lane_detection()
        params = data["params"]

        det = cls(
            experiment=experiment,
            registrator=registrator,
            store=store,
            detection_channel=params["channel"],
        )

        # Reconstruct LaneInfo objects
        lanes: Dict[str, List[LaneInfo]] = {}
        for fov, lane_dicts in data["lane_data"].items():
            lanes[fov] = [LaneInfo(**d) for d in lane_dicts]

        det._lanes = lanes
        return det

    def __repr__(self) -> str:
        if self._lanes is None:
            return "LaneDetector(not detected)"
        total = sum(len(v) for v in self._lanes.values())
        return f"LaneDetector({total} lanes across {len(self._lanes)} FOVs)"
