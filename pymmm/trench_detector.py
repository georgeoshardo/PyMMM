"""TrenchDetector – detect trench x-positions within each lane."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import convolve, find_peaks
from scipy.stats import theilslopes

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
    def n_trenches(self) -> int:
        if self._trenches is None:
            return 0
        return sum(len(v) for v in self._trenches.values())

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _detect_trenches_single_fov(
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
        """Run trench detection on a single FOV.

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
            fov_trenches, _, trench_counter = self._detect_trenches_single_fov(
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
        """Display an interactive widget for trench detection tuning.

        Uses ipywidgets for controls, ndv for the registered image and
        candidate-trench mask, and matplotlib for the intensity profiles.
        """
        import ipywidgets as widgets
        import matplotlib.pyplot as plt
        import ndv

        from pymmm.plotting import NDV_CLIMS
        from IPython.display import display

        mean_images: Dict[str, np.ndarray] = {}
        params = self._detection_params

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
        lane_controls = widgets.VBox()
        method_select = widgets.Dropdown(
            options=[
                ("peaks (tallest maxima)", "peaks"),
                ("periodic (spacing + symmetry)", "periodic"),
            ],
            value=params.get("method", "peaks"),
            description="method",
            style={"description_width": "initial"},
        )
        pitch_slider = widgets.FloatSlider(
            value=params.get("expected_pitch", 0), min=0,
            max=self.experiment.data.sizes["X"] / 2, step=1,
            description="expected pitch (0 = auto)",
            continuous_update=False, style={"description_width": "initial"},
        )
        symmetry_slider = widgets.FloatSlider(
            value=params.get("min_symmetry", 0.1), min=0, max=1, step=0.05,
            description="min symmetry",
            continuous_update=False, style={"description_width": "initial"},
        )
        sigma_slider = widgets.FloatSlider(
            value=params.get("sigma", 4), min=0, max=50, step=0.5,
            description="sigma",
            continuous_update=False, style={"description_width": "initial"},
        )
        distance_slider = widgets.FloatSlider(
            value=params.get("distance", 100), min=1, max=500, step=1,
            description="distance",
            continuous_update=False, style={"description_width": "initial"},
        )
        prominence_slider = widgets.FloatSlider(
            value=params.get("prominence", 10), min=0, max=500, step=1,
            description="prominence",
            continuous_update=False, style={"description_width": "initial"},
        )
        trench_width_slider = widgets.IntSlider(
            value=params.get("trench_width") or 32,
            min=1, max=200, step=1, description="trench width",
            continuous_update=False, style={"description_width": "initial"},
        )
        trench_length_slider = widgets.IntSlider(
            value=params.get("trench_length", 160), min=10,
            max=self.experiment.data.sizes["Y"], step=5,
            description="trench length",
            continuous_update=False, style={"description_width": "initial"},
        )
        trench_bottom_offset_slider = widgets.IntSlider(
            value=params.get("trench_bottom_offset", 50),
            min=-200, max=200, step=5, description="bottom offset",
            continuous_update=False, style={"description_width": "initial"},
        )
        apply_btn = widgets.Button(
            description="Apply to all FOVs", button_style="primary",
        )
        preview_label = widgets.HTML(value="")
        status_label = widgets.HTML(value="")
        output = widgets.Output()
        viewer_data = np.zeros(
            (
                2,
                self.experiment.data.sizes["Y"],
                self.experiment.data.sizes["X"],
            ),
            dtype=np.float32,
        )
        image_viewer = ndv.ArrayViewer(
            viewer_data,
            channel_axis=0,
            channel_mode="composite",
            luts={
                0: {"name": self.channel, "cmap": "gray", "clims": NDV_CLIMS},
                1: {
                    "name": "candidate trenches",
                    "cmap": "red",
                    "clims": (0, 1),
                },
            },
        )
        lane_widget_state: Dict[int, Dict[str, widgets.Widget]] = {}
        safe_lane_parameters = {
            "sigma", "distance", "prominence", "height", "trench_bottom_offset",
        }
        lane_override_values: Dict[int, Dict[str, Any]] = {
            int(lane_idx): {
                key: value
                for key, value in values.items()
                if key in safe_lane_parameters
            }
            for lane_idx, values in params.get("lane_params", {}).items()
        }
        syncing_lane_widgets = False
        global_widget_names = {
            "sigma": sigma_slider,
            "distance": distance_slider,
            "prominence": prominence_slider,
            "trench_bottom_offset": trench_bottom_offset_slider,
        }

        def _slider_row(
            lane_idx: int,
            label: str,
            value: float | int,
            min_value: float | int,
            max_value: float | int,
            step: float | int,
            is_float: bool = False,
        ) -> widgets.Widget:
            slider_cls = widgets.FloatSlider if is_float else widgets.IntSlider
            return slider_cls(
                value=value,
                min=min_value,
                max=max_value,
                step=step,
                description=f"L{lane_idx} {label}",
                continuous_update=False,
                style={"description_width": "initial"},
            )

        def _global_widget_value(param_name: str) -> Any:
            return global_widget_names[param_name].value

        def _lane_params_from_widgets() -> Dict[int, Dict[str, Any]]:
            lane_params = {
                lane_idx: values.copy()
                for lane_idx, values in lane_override_values.items()
                if values
            }
            for lane_idx, widgets_by_name in lane_widget_state.items():
                lane_overrides = lane_params.setdefault(lane_idx, {})
                for param_name, widget in widgets_by_name.items():
                    value = widget.value
                    global_value = _global_widget_value(param_name)
                    if value == global_value:
                        lane_overrides.pop(param_name, None)
                    else:
                        lane_overrides[param_name] = value
                if not lane_overrides:
                    lane_params.pop(lane_idx, None)
            return lane_params

        def _make_lane_observer(lane_idx: int, param_name: str):
            def _observer(change: Any) -> None:
                if syncing_lane_widgets:
                    return
                value = change["new"]
                global_value = _global_widget_value(param_name)
                lane_overrides = lane_override_values.setdefault(lane_idx, {})
                if value == global_value:
                    lane_overrides.pop(param_name, None)
                    if not lane_overrides:
                        lane_override_values.pop(lane_idx, None)
                else:
                    lane_overrides[param_name] = value
                _update()

            return _observer

        def _on_global_change(_change: Any = None) -> None:
            nonlocal syncing_lane_widgets
            syncing_lane_widgets = True
            try:
                for lane_idx, widgets_by_name in lane_widget_state.items():
                    lane_overrides = lane_override_values.get(lane_idx, {})
                    for param_name, widget in widgets_by_name.items():
                        if param_name in lane_overrides:
                            continue
                        target_value = _global_widget_value(param_name)
                        if widget.value != target_value:
                            widget.value = target_value
            finally:
                syncing_lane_widgets = False
            _update()

        def _rebuild_lane_controls(_change: Any = None) -> None:
            fov = fov_select.value
            fov_lanes = self.lane_detector.lanes.get(fov, [])
            new_state: Dict[int, Dict[str, widgets.Widget]] = {}
            accordion_children: List[widgets.Widget] = []

            for lane_info in fov_lanes:
                lane_idx = lane_info.lane_index
                lane_overrides = lane_override_values.get(lane_idx, {})
                lane_widgets = {
                    "sigma": _slider_row(
                        lane_idx, "sigma",
                        lane_overrides.get("sigma", sigma_slider.value),
                        0, 50, 0.5, is_float=True,
                    ),
                    "distance": _slider_row(
                        lane_idx, "distance",
                        lane_overrides.get("distance", distance_slider.value),
                        1, 500, 1, is_float=True,
                    ),
                    "prominence": _slider_row(
                        lane_idx, "prominence",
                        lane_overrides.get("prominence", prominence_slider.value),
                        0, 500, 1, is_float=True,
                    ),
                    "trench_bottom_offset": _slider_row(
                        lane_idx, "bottom offset",
                        lane_overrides.get(
                            "trench_bottom_offset",
                            trench_bottom_offset_slider.value,
                        ),
                        -200, 200, 5,
                    ),
                }
                for param_name, widget in lane_widgets.items():
                    widget.observe(
                        _make_lane_observer(lane_idx, param_name), names="value"
                    )

                new_state[lane_idx] = lane_widgets
                accordion_children.append(widgets.VBox(list(lane_widgets.values())))

            lane_widget_state.clear()
            lane_widget_state.update(new_state)

            if accordion_children:
                accordion = widgets.Accordion(children=accordion_children)
                for idx, lane_info in enumerate(fov_lanes):
                    accordion.set_title(idx, f"Lane {lane_info.lane_index} controls")
                lane_controls.children = (accordion,)
            else:
                lane_controls.children = (
                    widgets.HTML("<i>No lanes detected for this FOV.</i>"),
                )

            _update()

        def _sync_method_controls() -> None:
            periodic = method_select.value == "periodic"
            for widget in (distance_slider, prominence_slider):
                widget.disabled = periodic
            for widget in (pitch_slider, symmetry_slider):
                widget.disabled = not periodic

        def _update(_change: Any = None) -> None:
            _sync_method_controls()
            fov = fov_select.value
            mean_img = _mean_image(fov)
            lane_params = _lane_params_from_widgets()

            fov_trenches, lane_profiles, _ = self._detect_trenches_single_fov(
                mean_img, fov,
                sigma=sigma_slider.value,
                distance=distance_slider.value,
                prominence=prominence_slider.value,
                height=params.get("height", 0),
                trench_width=trench_width_slider.value,
                trench_length=trench_length_slider.value,
                trench_bottom_offset=trench_bottom_offset_slider.value,
                lane_params=lane_params,
                method=method_select.value,
                expected_pitch=pitch_slider.value,
                min_symmetry=symmetry_slider.value,
            )
            n_trenches = len(fov_trenches)
            n_lanes = len(lane_profiles)

            viewer_data[0] = mean_img
            trench_mask = viewer_data[1]
            trench_mask.fill(0)
            image_height, image_width = trench_mask.shape
            for trench in fov_trenches:
                y_start = max(0, trench.y_top)
                y_stop = min(image_height, trench.y_bottom)
                x_start = max(0, trench.x_left)
                x_stop = min(image_width, trench.x_right)
                if y_start >= y_stop or x_start >= x_stop:
                    continue
                trench_mask[y_start:min(y_start + 2, y_stop), x_start:x_stop] = 1
                trench_mask[max(y_stop - 2, y_start):y_stop, x_start:x_stop] = 1
                trench_mask[y_start:y_stop, x_start:min(x_start + 2, x_stop)] = 1
                trench_mask[y_start:y_stop, max(x_stop - 2, x_start):x_stop] = 1
            # Blank the image under the outlines so they stay visible where the
            # percentile contrast saturates (e.g. the feeding channel).
            viewer_data[0][trench_mask > 0] = mean_img.min()
            image_viewer.data_wrapper.data_changed.emit()
            preview_label.value = (
                f"<b>{fov}</b> — {n_trenches} candidate "
                f"trench{'es' if n_trenches != 1 else ''}"
            )

            with output:
                output.clear_output(wait=True)

                if n_lanes == 0:
                    fig, ax = plt.subplots(1, 1, figsize=(7, 2))
                    ax.text(0.5, 0.5, "No lane profiles", ha="center", va="center")
                    ax.axis("off")
                else:
                    fig, axes = plt.subplots(
                        n_lanes, 1,
                        figsize=(7, max(3, 2.5 * n_lanes)),
                        squeeze=False,
                    )

                    for i, lane_profile in enumerate(lane_profiles):
                        ax = axes[i, 0]
                        x_prof, x_smooth = lane_profile["profile"], lane_profile["smooth"]
                        centres = lane_profile["centres"]
                        xs = np.arange(len(x_prof))
                        ax.plot(xs, x_prof, color="grey", alpha=0.5, label="raw")
                        ax.plot(xs, x_smooth, color="blue", label="smoothed")
                        title = f"Lane {lane_profile['lane_index']} x-profile (trench bodies)"
                        if "symmetry" in lane_profile:
                            for centre in centres:
                                ax.axvline(centre, color="red", lw=1, alpha=0.8)
                            ax_sym = ax.twinx()
                            ax_sym.plot(
                                xs, lane_profile["symmetry"], color="tab:green",
                                alpha=0.6, lw=1, label="symmetry",
                            )
                            ax_sym.axhline(
                                symmetry_slider.value, color="tab:green", ls="--", lw=1,
                            )
                            ax_sym.set_ylim(-1, 1.05)
                            ax_sym.set_ylabel("symmetry", color="tab:green")
                            title += f", pitch {lane_profile['pitch']:.1f} px"
                        elif len(centres) > 0:
                            ax.plot(
                                centres, x_smooth[centres], "rv",
                                markersize=8, label="peaks",
                            )
                        ax.set_xlabel("x pixel")
                        ax.set_ylabel("intensity")
                        ax.set_title(title)
                        ax.legend(fontsize=8, loc="upper left")

                fig.tight_layout()
                plt.show()

        def _on_apply(_btn: Any) -> None:
            status_label.value = "<b>Applying\u2026</b>"
            self.detect_trenches(
                sigma=sigma_slider.value,
                distance=distance_slider.value,
                prominence=prominence_slider.value,
                height=params.get("height", 0),
                trench_width=trench_width_slider.value,
                trench_length=trench_length_slider.value,
                trench_bottom_offset=trench_bottom_offset_slider.value,
                lane_params=_lane_params_from_widgets(),
                method=method_select.value,
                expected_pitch=pitch_slider.value,
                min_symmetry=symmetry_slider.value,
            )
            total = self.n_trenches
            status_label.value = (
                f"<b>Applied</b> \u2014 {total} trenches across "
                f"{len(self._trenches)} FOVs"
            )

        apply_btn.on_click(_on_apply)

        fov_select.observe(_rebuild_lane_controls, names="value")

        for w in [sigma_slider, distance_slider, prominence_slider,
                  trench_width_slider,
                  trench_length_slider, trench_bottom_offset_slider,
                  method_select, pitch_slider, symmetry_slider]:
            w.observe(_on_global_change, names="value")

        controls = widgets.VBox([
            fov_select, method_select, sigma_slider,
            distance_slider, prominence_slider,
            pitch_slider, symmetry_slider,
            trench_width_slider,
            trench_length_slider, trench_bottom_offset_slider,
            lane_controls,
            apply_btn, status_label,
        ])
        image_and_profiles = widgets.VBox([
            preview_label, image_viewer.widget(), output,
        ])
        display(widgets.HBox([controls, image_and_profiles]))
        _rebuild_lane_controls()

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
