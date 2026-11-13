"""LaneDetector – detect feeding-lane y-positions in each FOV, or, for an image cropped
to its trenches, take the image's edge as the lane (:meth:`LaneDetector.use_image_edge`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from pymmm._utils import mean_without_fill, normalize_channel_arg, normalize_fov_arg
from pymmm.source.nd2 import ND2Experiment
from pymmm.steps.registration import DETECTION_MEAN, Registrator
from pymmm.store import lineage
from pymmm.store.companion import CompanionStore

# Forced lane orientations: "down" = trenches above the lane opening down into
# it (+1), "up" = trenches below the lane opening up into it (-1).
_FORCED_ORIENTATION = {"down": 1, "up": -1}


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
        detection_channel: str | int = 0,
    ) -> None:
        self.experiment = experiment
        self.registrator = registrator
        self.store = store
        self.channel = normalize_channel_arg(
            detection_channel, experiment.channel_names
        )
        self._lanes: dict[str, list[LaneInfo]] | None = None
        self._detection_params: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def lanes(self) -> dict[str, list[LaneInfo]]:
        if self._lanes is None:
            raise RuntimeError("Lanes not detected yet. Call detect_lanes() first.")
        return self._lanes

    @property
    def is_detected(self) -> bool:
        return self._lanes is not None

    @property
    def detection_params(self) -> dict[str, Any]:
        """The settings of the last detection (or the saved one)."""
        return dict(self._detection_params)

    @property
    def detection_image(self) -> dict[str, Any]:
        """The detection image the lanes were found on (``mean_n_frames``,
        ``mean_from``); the default for lanes saved without it.
        """
        params = self._detection_params
        return {k: params.get(k, v) for k, v in DETECTION_MEAN.items()}

    @property
    def image_edge(self) -> str | None:
        """``"down"`` or ``"up"`` when the lanes are the image's edge
        (:meth:`use_image_edge`), ``None`` when they were detected.
        """
        return self._detection_params.get("image_edge")

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_fov(
        self,
        mean_img: np.ndarray,
        sigma: float,
        distance: float,
        height: float,
        orientation_window_um: float,
        fov: str,
        orientation: str = "auto",
    ) -> tuple:
        """Lane detection on one FOV's registered mean image, with these settings
        (what :meth:`detect_lanes` does for every FOV; the widget's preview).

        Returns
        -------
        y_profile : np.ndarray
        y_profile_smooth : np.ndarray
        peaks : np.ndarray
        fov_lanes : list[LaneInfo]
        """
        pixel_size = self.experiment.pixel_size_um
        window_px = max(1, round(orientation_window_um / pixel_size))

        # 1D y-profile: average along X, without the warp's zero fill
        y_profile = mean_without_fill(mean_img, axis=1)

        # Gaussian smoothing
        if sigma > 0:
            y_profile_smooth = gaussian_filter1d(y_profile, sigma)
        else:
            y_profile_smooth = y_profile.copy()

        # Find peaks
        peaks, _ = find_peaks(y_profile_smooth, distance=distance, height=height)

        # Orientation detection
        fov_lanes: list[LaneInfo] = []
        for lane_idx, peak_y in enumerate(peaks):
            above_start = max(0, peak_y - window_px)
            above_end = peak_y
            below_start = peak_y
            below_end = min(len(y_profile), peak_y + window_px)

            mean_above = y_profile[above_start:above_end].mean()
            mean_below = y_profile[below_start:below_end].mean()

            if orientation == "auto":
                lane_orientation = 1 if mean_above > mean_below else -1
            else:
                lane_orientation = _FORCED_ORIENTATION[orientation]

            fov_lanes.append(
                LaneInfo(
                    y_center=int(peak_y),
                    orientation=lane_orientation,
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
        orientation: str = "auto",
        plot: bool = False,
        save_diagnostics: bool = True,
        mean_n_frames: int = DETECTION_MEAN["mean_n_frames"],
        mean_from: str = DETECTION_MEAN["mean_from"],
    ) -> None:
        """Detect lane y-positions in all FOVs.

        Algorithm:
        1. The detection image: the mean of registered frames (``mean_n_frames``,
           ``mean_from``).
        2. Average along X → 1-D y-profile.
        3. Gaussian smoothing.
        4. ``find_peaks()`` → lane centre positions.
        5. Orientation: compare mean intensity above vs below each peak, or
           use the forced ``orientation``.

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
        orientation : {"auto", "down", "up"}
            ``"auto"`` infers each lane's orientation from intensity.
            ``"down"`` forces every lane to have its trenches above it, opening
            down into the lane; ``"up"`` forces trenches below, opening up.
        plot : bool
            If ``True``, show the lane overlay for the first FOV.
        save_diagnostics : bool
            Save a lane overlay for every FOV to the ``lanes`` diagnostics
            folder (see :meth:`save_diagnostics`).
        mean_n_frames, mean_from : int, str
            The detection image: the mean of this many registered frames, spread
            over the experiment (``"spread"``) or from its ``"start"`` or ``"end"``
            (:func:`pymmm.steps.registration.mean_frames`).
        """
        if orientation not in ("auto", *_FORCED_ORIENTATION):
            raise ValueError(
                f"orientation must be 'auto', 'down' or 'up', not {orientation!r}"
            )
        lanes: dict[str, list[LaneInfo]] = {}

        mean_images = self.registrator.compute_registered_means(
            channel=self.channel, mean_n_frames=mean_n_frames, mean_from=mean_from
        )
        for fov in self.experiment.fov_names:
            mean_img = mean_images[fov]
            _, _, _, fov_lanes = self.detect_fov(
                mean_img,
                sigma,
                distance,
                height,
                orientation_window_um,
                fov,
                orientation=orientation,
            )
            lanes[fov] = fov_lanes

        self._lanes = lanes
        self._detection_params = {
            "sigma": sigma,
            "distance": distance,
            "height": height,
            "orientation_window_um": orientation_window_um,
            "orientation": orientation,
            "mean_n_frames": mean_n_frames,
            "mean_from": mean_from,
        }

        print(
            "Detected lanes: "
            + ", ".join(f"{fov}: {len(ls)}" for fov, ls in lanes.items())
        )

        if save_diagnostics:
            self.save_diagnostics()
        if plot:
            self.plot_fov(self.experiment.fov_names[0])

    def use_image_edge(
        self,
        opening: str = "down",
        save_diagnostics: bool = True,
        mean_n_frames: int = DETECTION_MEAN["mean_n_frames"],
        mean_from: str = DETECTION_MEAN["mean_from"],
    ) -> None:
        """No lane detection: one lane per FOV, on the image's edge where its trenches
        open. For an image cropped to its trenches (``ND2Experiment.select_roi``, the
        notebook's ``EXPERIMENT_Y_ROI``), so that they fill its height.

        Parameters
        ----------
        opening : {"down", "up"}
            ``"down"``: the trenches open down into a lane below the image, so the lane is
            its bottom edge (y = the image height); ``"up"``: into a lane above it, its
            top edge (y = 0). Trench detection then takes the image's whole height by
            default (:meth:`TrenchDetector.shape_defaults`).
        save_diagnostics : bool
            Save the lane overlays, as :meth:`detect_lanes` does.
        mean_n_frames, mean_from : int, str
            The detection image the overlays and trench detection show (recorded;
            the lanes themselves don't need one).
        """
        if opening not in _FORCED_ORIENTATION:
            raise ValueError(f"opening must be 'down' or 'up', not {opening!r}")
        y = self.experiment.data.sizes["Y"] if opening == "down" else 0
        self._lanes = {
            fov: [
                LaneInfo(
                    y_center=y,
                    orientation=_FORCED_ORIENTATION[opening],
                    fov=fov,
                    lane_index=0,
                )
            ]
            for fov in self.experiment.fov_names
        }
        self._detection_params = {
            "image_edge": opening,
            "mean_n_frames": mean_n_frames,
            "mean_from": mean_from,
        }
        edge = "bottom" if opening == "down" else "top"
        print(
            f"Lanes: the image's {edge} edge (y = {y}) in every FOV; the trenches open "
            f"{opening} into it"
        )
        if save_diagnostics:
            self.save_diagnostics()

    def save_diagnostics(self) -> None:
        """Save a lane overlay for every FOV to ``<experiment>_diagnostics/lanes/``."""
        from pymmm._utils import get_diagnostics_dir
        from pymmm.plotting import plot_fov_with_lanes, save_figures

        diag_dir = get_diagnostics_dir(self.experiment.path, "lanes")
        mean_images = self.registrator.compute_registered_means(
            list(self.lanes), self.channel, **self.detection_image
        )
        save_figures(
            plot_fov_with_lanes,
            [
                {
                    "image": mean_images[fov],
                    "lane_y_positions": [lane.y_center for lane in fov_lanes],
                    "orientations": [lane.orientation for lane in fov_lanes],
                    "title": f"Lanes – {fov}",
                    "save_path": str(diag_dir / f"lanes_{fov}.png"),
                }
                for fov, fov_lanes in self.lanes.items()
            ],
        )
        print(f"Lane diagnostics saved to {diag_dir}")

    def plot_fov(
        self,
        fov: int | str = 0,
        save_path: str | None = None,
        show: bool = True,
    ) -> None:
        """Plot the lane overlay for any FOV."""
        from pymmm.plotting import plot_fov_with_lanes

        fov_name = normalize_fov_arg(fov, self.experiment.fov_names)
        mean_img = self.registrator.get_registered_mean_of_timestack(
            fov=fov_name, channel=self.channel, **self.detection_image
        )
        fov_lanes = self.lanes[fov_name]
        plot_fov_with_lanes(
            mean_img,
            [lane.y_center for lane in fov_lanes],
            orientations=[lane.orientation for lane in fov_lanes],
            title=f"Lanes – {fov_name}",
            save_path=save_path,
            show=show,
        )

    # ------------------------------------------------------------------
    # Interactive widget
    # ------------------------------------------------------------------

    def interactive_detect_lanes(self) -> None:
        """Tune the settings on one FOV at a time in a notebook widget, then apply them
        to every FOV (:func:`pymmm.widgets.lanes.lane_widget`; the notebook extra).
        """
        from pymmm.widgets.lanes import lane_widget

        lane_widget(self)

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def _table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [asdict(li) for fov_lanes in self.lanes.values() for li in fov_lanes],
            columns=[f.name for f in fields(LaneInfo)],
        )

    def fingerprint(self) -> str:
        """The fingerprint of the lanes (:mod:`pymmm.store.lineage`)."""
        return lineage.table_fingerprint(
            self._table().to_dict("records"), lineage.LANE_COLUMNS
        )

    def save(self) -> None:
        """Write lane detection results to the companion zarr store, with the
        registration they were detected on.
        """
        if self._lanes is None:
            raise RuntimeError("Nothing to save — run detect_lanes() first.")

        params = {"channel": self.channel, **self._detection_params}
        self.store.save_lane_detection(
            self._table(),
            list(self._lanes),
            params,
            source_subset_metadata=self.experiment.source_subset_metadata,
            made_from={"registration": lineage.of(self.registrator)},
        )
        print(f"Lane detection saved to {self.store.path}")

    @classmethod
    def load(
        cls,
        experiment: ND2Experiment,
        registrator: Registrator,
        store: CompanionStore,
    ) -> LaneDetector:
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
        det._detection_params = {
            key: value for key, value in params.items() if key != "channel"
        }

        lanes: dict[str, list[LaneInfo]] = {fov: [] for fov in data["fov_names"]}
        for row in data["lanes"].itertuples(index=False):
            lanes[str(row.fov)].append(
                LaneInfo(
                    y_center=int(row.y_center),
                    orientation=int(row.orientation),
                    fov=str(row.fov),
                    lane_index=int(row.lane_index),
                )
            )
        det._lanes = lanes
        return det

    def __repr__(self) -> str:
        if self._lanes is None:
            return "LaneDetector(not detected)"
        if self.image_edge is not None:
            return (
                f"LaneDetector(the image's edge, trenches open {self.image_edge}: "
                f"{len(self._lanes)} FOVs)"
            )
        total = sum(len(v) for v in self._lanes.values())
        return f"LaneDetector({total} lanes across {len(self._lanes)} FOVs)"
