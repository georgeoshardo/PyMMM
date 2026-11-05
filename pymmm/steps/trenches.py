"""TrenchDetector – detect trench x-positions within each lane."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import convolve, find_peaks
from scipy.stats import theilslopes

from pymmm._utils import normalize_channel_arg, normalize_fov_arg
from pymmm.source.nd2 import ND2Experiment
from pymmm.steps.lanes import LaneDetector
from pymmm.steps.registration import Registrator
from pymmm.store.companion import CompanionStore


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


# ======================================================================
# Periodic trench detection
# ======================================================================
#
# Side-trench devices give each trench a double or triple peak in the
# x-profile (two walls, plus a central channel that is bright when empty and
# dark when full of cells), so the tallest peak is often a wall. Two
# properties are robust instead: trenches are equally spaced (lithography),
# and each trench's profile is mirror-symmetric about its centre whatever is
# inside it. The pitch comes from the profile's autocorrelation, the phase
# from all trenches' symmetry at once, and each centre is then refined
# locally and regularised by a straight-line fit.


def estimate_pitch(
    profile: np.ndarray,
    min_pitch: float,
    max_pitch: float,
    feature_scale: float = 32,
) -> float:
    """Dominant spacing of a periodic profile, from its autocorrelation.

    The autocorrelation is taken of the profile's structure envelope: the
    magnitude of its detail finer than ``feature_scale`` (the trench width).
    A trench's centre is a bright bump when empty and a dark dip when full of
    cells; in the raw profile those cancel between neighbours, in the
    envelope they look alike.

    Among the autocorrelation peaks with lags in ``[min_pitch, max_pitch]``,
    takes the shortest one at least ``0.8`` times as strong as the strongest:
    multiples of the true pitch correlate almost as well (and better when,
    say, every other trench holds cells), so the strongest alone can be a
    multiple. The lag is refined to sub-pixel precision with a parabola.
    """
    detail = profile - gaussian_filter1d(profile, max(feature_scale, 1))
    envelope = gaussian_filter1d(np.abs(detail), 3)
    centred = envelope - envelope.mean()
    autocorr = np.correlate(centred, centred, "full")[len(centred) - 1:]
    autocorr = autocorr / autocorr[0] if autocorr[0] > 0 else autocorr
    lo = max(1, int(np.floor(min_pitch)))
    hi = min(len(autocorr) - 1, int(np.ceil(max_pitch)) + 1)
    if hi - lo < 1:
        raise ValueError("Pitch search range is empty for this profile length.")
    peaks, _ = find_peaks(autocorr[lo - 1:hi + 1])
    peaks = peaks + lo - 1
    peaks = peaks[(peaks >= lo) & (peaks < hi)]
    if len(peaks):
        strong = peaks[autocorr[peaks] >= 0.8 * autocorr[peaks].max()]
        lag = int(strong.min())
    else:
        lag = lo + int(np.argmax(autocorr[lo:hi]))
    if 0 < lag < len(autocorr) - 1:
        a, b, c = autocorr[lag - 1:lag + 2]
        denominator = a - 2 * b + c
        if denominator != 0:
            return float(lag + 0.5 * (a - c) / denominator)
    return float(lag)


def mirror_symmetry(profile: np.ndarray, half_window: int) -> np.ndarray:
    """Score how mirror-symmetric ``profile`` is about every position.

    The score is the covariance between the ``half_window`` samples left of x
    and the mirrored samples right of x, after removing the slow background,
    normalised to a maximum of 1. Covariance (not correlation) rewards
    symmetric *structure*, so flat background between trenches scores near
    zero. Positions within ``half_window`` of the ends are NaN.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    detail = profile - gaussian_filter1d(profile, 3 * half_window)
    windows = sliding_window_view(detail, 2 * half_window + 1)
    left = windows[:, :half_window][:, ::-1]
    right = windows[:, half_window + 1:]
    score = np.full(len(profile), np.nan)
    score[half_window:len(profile) - half_window] = (left * right).mean(axis=1)
    peak = np.nanmax(score)
    return score / peak if peak > 0 else score


def periodic_trench_centres(
    profile: np.ndarray,
    trench_width: int,
    expected_pitch: float = 0,
    min_symmetry: float = 0.1,
) -> Dict[str, Any]:
    """Trench centres on a lane's x-profile from periodicity and symmetry.

    Parameters
    ----------
    profile : np.ndarray
        Lane x-profile (mean over the trench bodies).
    trench_width : int
        Crop width; also the smallest pitch considered.
    expected_pitch : float
        Approximate trench spacing in pixels to search around (±20%), or 0
        to search every spacing from ``trench_width`` to half the profile.
    min_symmetry : float
        Drop centres whose symmetry score is below this (0-1); catches crops
        placed where a trench is missing or blocked.

    Returns
    -------
    dict
        ``centres`` (float x positions), ``symmetry`` (score per x),
        ``pitch`` (pixels).
    """
    n = len(profile)
    if expected_pitch > 0:
        min_pitch, max_pitch = 0.8 * expected_pitch, 1.2 * expected_pitch
    else:
        min_pitch, max_pitch = max(trench_width, 3), n / 2
    empty = {"centres": np.array([]), "symmetry": np.full(n, np.nan), "pitch": np.nan}
    if max_pitch <= min_pitch or n < 3 * min_pitch:
        return empty
    pitch = estimate_pitch(profile, min_pitch, max_pitch, feature_scale=trench_width)

    half_window = max(2, int(pitch / 3))
    if 2 * half_window + 1 > n:
        return empty
    symmetry = mirror_symmetry(profile, half_window)

    # Phase shared by all trenches: circular mean of x (mod pitch) weighted by
    # strongly symmetric positions.
    xs = np.arange(n)
    valid = ~np.isnan(symmetry)
    weights = np.clip(symmetry[valid], 0, None) ** 4
    angles = 2 * np.pi * xs[valid] / pitch
    phase = (np.angle(np.sum(weights * np.exp(1j * angles))) % (2 * np.pi)) * pitch / (2 * np.pi)

    # Refine each lattice site to the local symmetry maximum, then fit a line
    # through the refined centres (absorbs small scale/rotation errors). Only
    # convincingly symmetric sites enter the fit, and Theil-Sen keeps a stray
    # site (missing or blocked trench) from tilting it.
    radius = max(1, int(pitch / 6))
    sites, refined = [], []
    for k, centre in enumerate(np.arange(phase, n, pitch)):
        lo, hi = int(max(half_window, centre - radius)), int(min(n - half_window, centre + radius + 1))
        if hi - lo < 3:
            continue
        best = lo + int(np.nanargmax(symmetry[lo:hi]))
        if symmetry[best] >= max(min_symmetry, 0.2):
            sites.append(k)
            refined.append(best)
    if len(refined) >= 2:
        slope, intercept, _, _ = theilslopes(refined, sites)
    elif refined:
        slope, intercept = pitch, refined[0] - pitch * sites[0]
    else:
        return {**empty, "symmetry": symmetry, "pitch": pitch}
    k = np.arange(np.ceil(-intercept / slope), np.floor((n - 1 - intercept) / slope) + 1)
    centres = intercept + slope * k

    # Score each centre by the best symmetry within a couple of pixels (the
    # symmetry peak is sharp). Near the ends the symmetry window runs off the
    # profile, so the score is unknown there; keep those centres and let the
    # crop bounds decide.
    tolerance = max(2, int(pitch / 30))
    keep = []
    for i in np.round(centres).astype(int):
        if i - tolerance < half_window or i + tolerance >= n - half_window:
            keep.append(True)
        else:
            keep.append(np.nanmax(symmetry[i - tolerance:i + tolerance + 1]) >= min_symmetry)
    centres = centres[np.asarray(keep, dtype=bool)]
    return {"centres": centres, "symmetry": symmetry, "pitch": float(slope)}


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
        self._detection_params: Dict[str, Any] = {}

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
    def detection_params(self) -> Dict[str, Any]:
        """The settings of the last detection (or the saved one)."""
        return copy.deepcopy(self._detection_params)

    @property
    def n_trenches(self) -> int:
        if self._trenches is None:
            return 0
        return sum(len(v) for v in self._trenches.values())

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_fov(
        self,
        mean_img: np.ndarray,
        fov: str,
        sigma: float,
        distance: float,
        prominence: float,
        height: float,
        trench_width: int,
        trench_length: int,
        trench_bottom_offset: int,
        conv_filter: Optional[np.ndarray] = None,
        start_id: int = 0,
        lane_params: Optional[Dict[int, Dict[str, Any]]] = None,
        method: str = "peaks",
        expected_pitch: float = 0,
        min_symmetry: float = 0.1,
    ) -> tuple:
        """Trench detection on one FOV's registered mean image, with these settings
        (what :meth:`detect_trenches` does for every FOV; the widget's preview).

        Returns
        -------
        fov_trenches : list[TrenchDefinition]
        lane_profiles : list[dict]
            Per lane: ``lane_index``, ``profile`` (raw x-profile), ``smooth``,
            ``centres`` (trench centre x positions), and for the periodic
            method ``symmetry`` and ``pitch``.
        next_id : int
            Next available trench ID counter.
        """
        fov_lanes = self.lane_detector.lanes.get(fov, [])
        img_height, img_width = mean_img.shape

        fov_trenches: List[TrenchDefinition] = []
        lane_profiles: list = []
        trench_counter = start_id

        for lane_info in fov_lanes:
            lane_y = lane_info.y_center
            orientation = lane_info.orientation
            lane_idx = lane_info.lane_index
            lane_override = lane_params.get(lane_idx, {}) if lane_params else {}
            lane_sigma = lane_override.get("sigma", sigma)
            lane_distance = lane_override.get("distance", distance)
            lane_prominence = lane_override.get("prominence", prominence)
            lane_height = lane_override.get("height", height)
            lane_bottom_offset = lane_override.get(
                "trench_bottom_offset", trench_bottom_offset
            )

            if orientation == 1:
                y_top = lane_y - lane_bottom_offset - trench_length
                y_bottom = lane_y - lane_bottom_offset
                needs_flip = False
            else:
                y_top = lane_y + lane_bottom_offset
                y_bottom = lane_y + lane_bottom_offset + trench_length
                needs_flip = True

            # Profile over the trench bodies only: the rows that will be
            # cropped, not the feeding channel or the far side of the lane.
            profile_rows = mean_img[max(0, y_top):min(img_height, y_bottom), :]
            if profile_rows.shape[0] == 0:
                continue
            x_profile = profile_rows.mean(axis=0)

            if lane_sigma > 0:
                x_profile_smooth = gaussian_filter1d(x_profile, lane_sigma)
            else:
                x_profile_smooth = x_profile.copy()

            if conv_filter is not None:
                x_profile_smooth = (
                    convolve(x_profile_smooth, conv_filter) / conv_filter.sum()
                )

            lane_profile = {
                "lane_index": lane_idx,
                "profile": x_profile,
                "smooth": x_profile_smooth,
            }
            if method == "periodic":
                result = periodic_trench_centres(
                    x_profile_smooth, trench_width,
                    expected_pitch=lane_override.get("expected_pitch", expected_pitch),
                    min_symmetry=lane_override.get("min_symmetry", min_symmetry),
                )
                centres = result["centres"]
                lane_profile.update(symmetry=result["symmetry"], pitch=result["pitch"])
            else:
                centres, _ = find_peaks(
                    x_profile_smooth,
                    distance=lane_distance,
                    height=lane_height,
                    prominence=lane_prominence,
                )
            lane_profile["centres"] = np.asarray(centres)
            lane_profiles.append(lane_profile)

            if len(centres) == 0:
                continue

            if method == "periodic":
                x_lefts = np.floor(centres - trench_width / 2 + 0.5).astype(int)
            else:
                x_lefts = centres - trench_width // 2
            x_rights = x_lefts + trench_width

            for x_left, x_right in zip(x_lefts, x_rights):
                x_left = int(x_left)
                x_right = int(x_right)

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

        return fov_trenches, lane_profiles, trench_counter

    def detect_trenches(
        self,
        sigma: float = 4.0,
        distance: float = 100,
        prominence: float = 10,
        height: float = 0,
        trench_width: int = 32,
        trench_length: int = 160,
        trench_bottom_offset: int = 50,
        conv_filter: Optional[np.ndarray] = None,
        plot: bool = False,
        lane_params: Optional[Dict[int, Dict[str, Any]]] = None,
        save_diagnostics: bool = True,
        method: str = "peaks",
        expected_pitch: float = 0,
        min_symmetry: float = 0.1,
    ) -> None:
        """Detect trench x-positions in all FOVs and lanes.

        Algorithm per FOV per lane:
        1. Compute y-boundaries from lane centre, orientation, bottom offset
           and trench length.
        2. Average the registered mean image over those rows (the trench
           bodies) → 1-D x-profile.
        3. Trench centres: ``method="peaks"`` takes ``find_peaks()`` maxima;
           ``method="periodic"`` uses the trench spacing and mirror symmetry
           (see :func:`periodic_trench_centres`), which is robust for
           side-trench devices whose trenches give double or triple peaks.
        4. Compute x-boundaries from ``trench_width``.
        5. Prune trenches extending beyond image edges.
        6. Assign global trench IDs.

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
        trench_width : int
            Exact trench width in pixels.
        trench_length : int
            Trench crop height in pixels.
        trench_bottom_offset : int
            Offset from lane centre to trench bottom in pixels.
        conv_filter : np.ndarray | None
            Optional convolution filter for noisy data.
        plot : bool
            If ``True``, show the trench overlay for the first FOV.
        lane_params : dict[int, dict[str, Any]] | None
            Optional per-lane detection parameter overrides keyed by
            ``lane_index``. Output width and length are always global.
        save_diagnostics : bool
            Save a trench overlay for every FOV to the ``trenches``
            diagnostics folder (see :meth:`save_diagnostics`).
        method : {"peaks", "periodic"}
            How trench centres are found (see step 3). ``distance``,
            ``prominence`` and ``height`` apply to ``"peaks"`` only;
            ``expected_pitch`` and ``min_symmetry`` to ``"periodic"`` only.
        expected_pitch : float
            Periodic method: approximate trench spacing in pixels (searched
            ±20%), or 0 to estimate it from the profile.
        min_symmetry : float
            Periodic method: drop centres whose symmetry score (0-1) is lower,
            e.g. where a trench is missing.
        """
        if method not in ("peaks", "periodic"):
            raise ValueError(f"method must be 'peaks' or 'periodic', not {method!r}")
        if trench_width <= 0:
            raise ValueError("trench_width must be greater than zero.")
        if trench_length <= 0:
            raise ValueError("trench_length must be greater than zero.")

        global_shape_parameters = {"trench_width", "trench_length", "shrink_scale"}
        for lane_idx, overrides in (lane_params or {}).items():
            invalid = global_shape_parameters.intersection(overrides)
            if invalid:
                names = ", ".join(sorted(invalid))
                raise ValueError(
                    f"Lane {lane_idx} cannot override global shape settings: {names}"
                )

        trenches: Dict[str, List[TrenchDefinition]] = {}
        trench_counter = 0

        mean_images = self.registrator.compute_registered_means(channel=self.channel)
        for fov in self.experiment.fov_names:
            mean_img = mean_images[fov]
            fov_trenches, _, trench_counter = self.detect_fov(
                mean_img, fov, sigma, distance, prominence, height,
                trench_width, trench_length, trench_bottom_offset,
                conv_filter, start_id=trench_counter, lane_params=lane_params,
                method=method, expected_pitch=expected_pitch, min_symmetry=min_symmetry,
            )
            trenches[fov] = fov_trenches

        self._trenches = trenches
        self._detection_params = {
            "sigma": sigma,
            "distance": distance,
            "prominence": prominence,
            "height": height,
            "trench_width": trench_width,
            "trench_length": trench_length,
            "trench_bottom_offset": trench_bottom_offset,
            "method": method,
            "expected_pitch": expected_pitch,
            "min_symmetry": min_symmetry,
            "lane_params": {
                int(lane_idx): values.copy()
                for lane_idx, values in (lane_params or {}).items()
            },
        }
        print(f"Detected {self.n_trenches} trenches across {len(trenches)} FOVs")

        if save_diagnostics:
            self.save_diagnostics()
        if plot:
            self.plot_fov(self.experiment.fov_names[0])

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

    def save_diagnostics(self) -> None:
        """Save a trench overlay for every FOV to ``<experiment>_diagnostics/trenches/``."""
        from pymmm._utils import get_diagnostics_dir
        from pymmm.plotting import plot_fov_with_trenches, save_figures

        diag_dir = get_diagnostics_dir(self.experiment.path, "trenches")
        mean_images = self.registrator.compute_registered_means(
            list(self._trenches), self.channel,
        )
        save_figures(plot_fov_with_trenches, [
            {
                "image": mean_images[fov],
                "trench_definitions": fov_trenches,
                "title": f"Trenches – {fov}",
                "save_path": str(diag_dir / f"trenches_{fov}.png"),
            }
            for fov, fov_trenches in self._trenches.items()
        ])
        print(f"Trench diagnostics saved to {diag_dir}")

    def plot_fov(
        self,
        fov: Union[int, str] = 0,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """Plot the trench overlay for any FOV."""
        from pymmm.plotting import plot_fov_with_trenches

        fov_name = normalize_fov_arg(fov, self.experiment.fov_names)
        mean_img = self.registrator.get_registered_mean_of_timestack(
            fov=fov_name, channel=self.channel
        )
        fov_trenches = self._trenches.get(fov_name, [])
        plot_fov_with_trenches(
            mean_img, fov_trenches, title=f"Trenches – {fov_name}",
            save_path=save_path, show=show,
        )

    # ------------------------------------------------------------------
    # Interactive widget
    # ------------------------------------------------------------------

    def interactive_detect_trenches(self) -> None:
        """Tune the settings on one FOV at a time, per lane if need be, in a notebook
        widget, then apply them to every FOV (:func:`pymmm.widgets.trenches.trench_widget`;
        the notebook extra)."""
        from pymmm.widgets.trenches import trench_widget

        trench_widget(self)

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write trench detection results to the companion zarr store."""
        if self._trenches is None:
            raise RuntimeError("Nothing to save — run detect_trenches() first.")

        trenches = pd.DataFrame(
            [asdict(td) for fov_trenches in self._trenches.values() for td in fov_trenches],
            columns=[f.name for f in fields(TrenchDefinition)],
        )
        params = {"channel": self.channel, **self._detection_params}
        self.store.save_trench_detection(
            trenches, list(self._trenches), params,
            source_subset_metadata=self.experiment.source_subset_metadata,
        )
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
        det._detection_params = {
            key: value for key, value in params.items() if key != "channel"
        }
        if "lane_params" in det._detection_params:
            det._detection_params["lane_params"] = {
                int(lane_idx): values
                for lane_idx, values in det._detection_params["lane_params"].items()
            }

        trenches: Dict[str, List[TrenchDefinition]] = {fov: [] for fov in data["fov_names"]}
        for row in data["trenches"].itertuples(index=False):
            trenches[str(row.fov)].append(TrenchDefinition(
                trench_id=int(row.trench_id), fov=str(row.fov),
                lane_index=int(row.lane_index), x_left=int(row.x_left),
                x_right=int(row.x_right), y_top=int(row.y_top),
                y_bottom=int(row.y_bottom), orientation=int(row.orientation),
                needs_flip=bool(row.needs_flip),
            ))
        det._trenches = trenches
        return det

    def __repr__(self) -> str:
        if self._trenches is None:
            return "TrenchDetector(not detected)"
        return f"TrenchDetector({self.n_trenches} trenches across {len(self._trenches)} FOVs)"
