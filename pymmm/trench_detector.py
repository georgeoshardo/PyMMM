"""TrenchDetector – detect trench x-positions within each lane."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import convolve, find_peaks

from pymmm._utils import normalize_channel_arg, normalize_fov_arg
from pymmm.checkpoint import CompanionStore
from pymmm.experiment import ND2Experiment
from pymmm.lane_detector import LaneDetector
from pymmm.registrator import Registrator


@dataclass
class TrenchDefinition:
    """Describes one trench crop region."""

    trench_id: int
    fov: str
    lane_index: int
    x_left: int
    x_right: int
    y_top: int
    y_bottom: int
    orientation: int  # +1 or -1
    needs_flip: bool  # True if orientation == -1


class TrenchDetector:
    """Detect trench x-positions within each lane and compute crop boundaries.

    Parameters
    ----------
    experiment : ND2Experiment
        Source experiment.
    registrator : Registrator
        Completed registrator.
    lane_detector : LaneDetector
        Completed lane detector.
    store : CompanionStore
        Companion zarr store for checkpointing.
    detection_channel : str | int
        Channel for trench detection.
    """

    def __init__(
        self,
        experiment: ND2Experiment,
        registrator: Registrator,
        lane_detector: LaneDetector,
        store: CompanionStore,
        detection_channel: Union[str, int] = 0,
    ) -> None:
        self.experiment = experiment
        self.registrator = registrator
        self.lane_detector = lane_detector
        self.store = store
        self.channel = normalize_channel_arg(
            detection_channel, experiment.channel_names
        )
        self._trenches: Optional[Dict[str, List[TrenchDefinition]]] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def trenches(self) -> Dict[str, List[TrenchDefinition]]:
        if self._trenches is None:
            raise RuntimeError(
                "Trenches not detected yet. Call detect_trenches() first."
            )
        return self._trenches

    @property
    def is_detected(self) -> bool:
        return self._trenches is not None

    @property
    def n_trenches(self) -> int:
        if self._trenches is None:
            return 0
        return sum(len(v) for v in self._trenches.values())

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_trenches(
        self,
        sigma: float = 4.0,
        distance: float = 100,
        prominence: float = 10,
        height: float = 0,
        trench_width: Optional[int] = None,
        shrink_scale: float = 2.2,
        trench_length: int = 160,
        trench_bottom_offset: int = 50,
        conv_filter: Optional[np.ndarray] = None,
        plot: bool = False,
    ) -> None:
        """Detect trench x-positions in all FOVs and lanes.

        Algorithm per FOV per lane:
        1. Crop registered mean image to the lane's y-region.
        2. Average along Y → 1-D x-profile.
        3. ``find_peaks()`` → trench centre positions.
        4. Compute x-boundaries from ``trench_width`` or peak spacing.
        5. Compute y-boundaries from lane centre + orientation.
        6. Prune trenches extending beyond image edges.
        7. Assign global trench IDs.

        Parameters
        ----------
        sigma : float
            Gaussian smoothing sigma for the x-profile.
        distance : float
            Minimum distance between peaks.
        prominence : float
            Minimum peak prominence.
        height : float
            Minimum peak height.
        trench_width : int | None
            Exact trench width in pixels. If ``None``, derived from peak spacing.
        shrink_scale : float
            When ``trench_width`` is None, boundary = spacing / shrink_scale.
        trench_length : int
            Trench crop height in pixels.
        trench_bottom_offset : int
            Offset from lane centre to trench bottom in pixels.
        conv_filter : np.ndarray | None
            Optional convolution filter for noisy data.
        plot : bool
            If ``True``, show overlay for the first FOV.
        """
        lanes_dict = self.lane_detector.lanes
        img_shape = self.experiment.data.sizes

        # Image dimensions
        img_height = img_shape["Y"]
        img_width = img_shape["X"]

        trenches: Dict[str, List[TrenchDefinition]] = {}
        trench_counter = 0

        for fov in self.experiment.fov_names:
            mean_img = self.registrator.get_registered_mean_of_timestack(
                fov=fov, channel=self.channel
            )
            fov_lanes = lanes_dict.get(fov, [])
            fov_trenches: List[TrenchDefinition] = []

            for lane_info in fov_lanes:
                lane_y = lane_info.y_center
                orientation = lane_info.orientation
                lane_idx = lane_info.lane_index

                # Crop a region around the lane for x-profile analysis
                # Use a generous vertical window around the lane centre
                crop_half = max(trench_length, 100)
                y_start = max(0, lane_y - crop_half)
                y_end = min(img_height, lane_y + crop_half)
                lane_crop = mean_img[y_start:y_end, :]

                # 1-D x-profile: average along Y
                x_profile = lane_crop.mean(axis=0)

                # Gaussian smoothing
                if sigma > 0:
                    x_profile_smooth = gaussian_filter1d(x_profile, sigma)
                else:
                    x_profile_smooth = x_profile

                # Optional convolution filter
                if conv_filter is not None:
                    x_profile_smooth = (
                        convolve(x_profile_smooth, conv_filter) / conv_filter.sum()
                    )

                # Find peaks
                peaks, _ = find_peaks(
                    x_profile_smooth,
                    distance=distance,
                    height=height,
                    prominence=prominence,
                )

                if len(peaks) == 0:
                    continue

                # Compute x-boundaries
                if trench_width is not None:
                    half_w = trench_width // 2
                    x_lefts = peaks - half_w
                    x_rights = peaks + half_w
                else:
                    spacing = np.mean(np.diff(peaks)) if len(peaks) > 1 else 100
                    half_w = int(round(spacing / shrink_scale))
                    x_lefts = peaks - half_w
                    x_rights = peaks + half_w

                # Compute y-boundaries based on orientation
                if orientation == 1:
                    # Trenches open downward from lane
                    y_top = lane_y - trench_bottom_offset - trench_length
                    y_bottom = lane_y - trench_bottom_offset
                    needs_flip = False
                else:
                    # Trenches open upward from lane
                    y_top = lane_y + trench_bottom_offset
                    y_bottom = lane_y + trench_bottom_offset + trench_length
                    needs_flip = True

                # Create trench definitions, pruning edge cases
                for x_left, x_right in zip(x_lefts, x_rights):
                    x_left = int(x_left)
                    x_right = int(x_right)

                    # Prune trenches extending beyond image edges
                    if x_left < 0 or x_right > img_width:
                        continue
                    if y_top < 0 or y_bottom > img_height:
                        continue

                    fov_trenches.append(
                        TrenchDefinition(
                            trench_id=trench_counter,
                            fov=fov,
                            lane_index=lane_idx,
                            x_left=x_left,
                            x_right=x_right,
                            y_top=y_top,
                            y_bottom=y_bottom,
                            orientation=orientation,
                            needs_flip=needs_flip,
                        )
                    )
                    trench_counter += 1

            trenches[fov] = fov_trenches

        self._trenches = trenches
        print(f"Detected {self.n_trenches} trenches across {len(trenches)} FOVs")

        if plot:
            self._plot_first_fov()

    # ------------------------------------------------------------------
    # Trench management
    # ------------------------------------------------------------------

    def discard_trenches(self, trench_ids: Sequence[int]) -> None:
        """Remove trenches by their global IDs."""
        if self._trenches is None:
            raise RuntimeError("No trenches detected yet.")

        ids_set = set(trench_ids)
        for fov in self._trenches:
            self._trenches[fov] = [
                t for t in self._trenches[fov] if t.trench_id not in ids_set
            ]
        removed = len(ids_set)
        print(f"Discarded {removed} trenches. Remaining: {self.n_trenches}")

    def get_trench_table(self) -> pd.DataFrame:
        """Return a DataFrame of all trench definitions."""
        if self._trenches is None:
            raise RuntimeError("No trenches detected yet.")

        rows = []
        for fov, fov_trenches in self._trenches.items():
            for td in fov_trenches:
                rows.append(asdict(td))
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot_first_fov(self) -> None:
        """Plot trench overlay for the first FOV."""
        from pymmm._utils import get_diagnostics_dir

        diag_dir = get_diagnostics_dir(self.experiment.path)
        fov = self.experiment.fov_names[0]
        self.plot_fov(fov, save_path=str(diag_dir / f"trenches_{fov}.png"))

    def plot_fov(
        self, fov: Union[int, str] = 0, save_path: Optional[str] = None,
    ) -> None:
        """Plot trench overlay for any FOV."""
        from pymmm.plotting import plot_fov_with_trenches

        fov_name = normalize_fov_arg(fov, self.experiment.fov_names)
        mean_img = self.registrator.get_registered_mean_of_timestack(
            fov=fov_name, channel=self.channel
        )
        fov_trenches = self._trenches.get(fov_name, [])
        plot_fov_with_trenches(
            mean_img, fov_trenches, title=f"Trenches – {fov_name}",
            save_path=save_path,
        )

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write trench detection results to the companion zarr store."""
        if self._trenches is None:
            raise RuntimeError("Nothing to save — run detect_trenches() first.")

        trench_data: Dict[str, list] = {}
        for fov, fov_trenches in self._trenches.items():
            trench_data[fov] = [asdict(td) for td in fov_trenches]

        params = {"channel": self.channel}
        self.store.save_trench_detection(trench_data, params)
        print(f"Trench detection saved to {self.store.path}")

    @classmethod
    def load(
        cls,
        experiment: ND2Experiment,
        registrator: Registrator,
        lane_detector: LaneDetector,
        store: CompanionStore,
    ) -> "TrenchDetector":
        """Reconstruct a TrenchDetector from a checkpoint."""
        if not store.has_trench_detection():
            raise FileNotFoundError("No trench detection checkpoint found.")

        data = store.load_trench_detection()
        params = data["params"]

        det = cls(
            experiment=experiment,
            registrator=registrator,
            lane_detector=lane_detector,
            store=store,
            detection_channel=params["channel"],
        )

        # Reconstruct TrenchDefinition objects
        trenches: Dict[str, List[TrenchDefinition]] = {}
        for fov, trench_dicts in data["trench_data"].items():
            trenches[fov] = [TrenchDefinition(**d) for d in trench_dicts]

        det._trenches = trenches
        return det

    def __repr__(self) -> str:
        if self._trenches is None:
            return "TrenchDetector(not detected)"
        return f"TrenchDetector({self.n_trenches} trenches across {len(self._trenches)} FOVs)"
