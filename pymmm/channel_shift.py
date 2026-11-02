"""Shift between a segmented channel and another channel of a trench store.

Two channels of the same trench are not perfectly registered: lateral chromatic
aberration makes the shift depend on where the trench is in the field of view. Given
a segmentation of one channel, the shift to any other channel is measured by sliding
the segmentation's (binarised) masks over that channel's frames and finding where they
sit on the cells: in phase contrast cells are dark (``polarity="dark"``), in
fluorescence bright (``"bright"``). Each trench is split into segments along its length
and each segment's cost surface is summed over up to ``n_frames`` frames, which is far
steadier than any single frame; the minimum is refined to sub-pixel precision.

The measurements are then described by a model that is affine in field-of-view pixel
coordinates (dy and dx each ``c0 + cx * x + cy * y``), fitted over every trench and
every field of view at once (they share the optics), with outliers rejected. Results
go to ``/channel_shift/<name>`` in the trench store::

    measured(Trench, Segment, ShiftAxis)   per-segment measurement (NaN if not measurable)
    modelled(Trench, Segment, ShiftAxis)   the model at the same places
    quality(Trench, Segment)               depth of the cost minimum (normalised intensity)
    n_frames(Trench, Segment)              frames the measurement used
    segment_y(Segment)                     each segment's centre row in the trench crop

Convention: a shift (dy, dx) moves the segmentation's labels onto the other channel,
in trench-crop pixels (y down the crop, x across it): the label at (y, x) belongs at
(y + dy, x + dx). :meth:`ChannelShift.apply` does that for a trench's labels, using the
model row by row.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import xarray as xr
import zarr
from scipy import ndimage
from tqdm.auto import tqdm

from pymmm.provenance import SCHEMA_VERSION, provenance

DEFAULT_PARAMS: Dict[str, Any] = {
    "max_dy": 5,          # px; along the trench cells form a repeating chain, so keep it small
    "max_dx": 5,          # px
    "segments": 3,        # measurements along each trench
    "n_frames": 30,       # frames summed per measurement, spread over the experiment
    "min_mask_px": 300,   # a frame counts for a segment if its masks cover this many pixels there
    "min_frames": 5,      # fewer usable frames: not measured
    "outlier_sd": 3.0,    # model fit: drop measurements this many robust SDs off, then refit
}


def _normalise(frame: np.ndarray, polarity: str) -> np.ndarray:
    """Frame scaled to its 1st-99th percentiles (zero padding ignored), cells low."""
    real = frame[frame > 0]
    lo, hi = np.percentile(real, (1, 99)) if real.size else (0.0, 1.0)
    f = np.clip((frame.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    return f if polarity == "dark" else 1 - f


def cost_surface(frame: np.ndarray, mask: np.ndarray, max_dy: int, max_dx: int) -> np.ndarray:
    """Mean of ``frame`` under ``mask`` moved by every (dy, dx) in the search window.

    ``frame`` must already be normalised with cells low; outside the frame counts as 1
    (never preferred)."""
    f = np.pad(frame, ((max_dy, max_dy), (max_dx, max_dx)), constant_values=1.0)
    ys, xs = np.nonzero(mask)
    cost = np.empty((2 * max_dy + 1, 2 * max_dx + 1), np.float64)
    for i in range(2 * max_dy + 1):
        for j in range(2 * max_dx + 1):
            cost[i, j] = f[ys + i, xs + j].mean()
    return cost


def subpixel_minimum(cost: np.ndarray) -> Tuple[float, float, bool]:
    """(dy, dx) of the minimum relative to the window centre, refined by a parabola
    through each axis' neighbours; the flag is True if the minimum is on the edge."""
    max_dy, max_dx = (cost.shape[0] - 1) // 2, (cost.shape[1] - 1) // 2
    i, j = np.unravel_index(np.argmin(cost), cost.shape)

    def refine(c: np.ndarray, k: int) -> float:
        if 0 < k < len(c) - 1:
            d = c[k - 1] - 2 * c[k] + c[k + 1]
            if d > 0:
                return k + 0.5 * (c[k - 1] - c[k + 1]) / d
        return float(k)

    edge = i in (0, cost.shape[0] - 1) or j in (0, cost.shape[1] - 1)
    return refine(cost[:, j], i) - max_dy, refine(cost[i, :], j) - max_dx, edge


def _measure_trench(args) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    store, labels_path, hyp_index, channel_index, index, polarity, p = args
    root = zarr.open_group(store, mode="r")
    images = np.asarray(root["data"][index, :, channel_index])
    labels = np.asarray(root[labels_path][hyp_index, index])
    n_t, height = labels.shape[:2]
    edges = np.linspace(0, height, p["segments"] + 1).round().astype(int)
    shift = np.full((p["segments"], 2), np.nan, np.float32)
    quality = np.full(p["segments"], np.nan, np.float32)
    used = np.zeros(p["segments"], np.int16)
    norm: Dict[int, np.ndarray] = {}
    for s, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        ok = [t for t in range(n_t) if np.count_nonzero(labels[t, a:b]) >= p["min_mask_px"]]
        if len(ok) < p["min_frames"]:
            continue
        frames = [ok[k] for k in np.unique(np.linspace(0, len(ok) - 1, min(p["n_frames"], len(ok))).round().astype(int))]
        cost = 0.0
        for t in frames:
            if t not in norm:
                norm[t] = _normalise(images[t], polarity)
            mask = np.zeros(labels.shape[1:], bool)
            mask[a:b] = labels[t, a:b] > 0
            cost = cost + cost_surface(norm[t], mask, p["max_dy"], p["max_dx"])
        cost = cost / len(frames)
        dy, dx, edge = subpixel_minimum(cost)
        used[s] = len(frames)
        if not edge:  # a minimum on the window's edge is not a measurement
            shift[s] = (dy, dx)
            quality[s] = float(np.median(cost) - cost.min())
    return index, shift, quality, used


def fit_affine(x: np.ndarray, y: np.ndarray, v: np.ndarray, outlier_sd: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """Least-squares ``v = c0 + cx * x + cy * y``, refitted once without points more than
    ``outlier_sd`` robust SDs off. Returns (coefficients, kept mask, residual SD)."""
    a = np.column_stack([np.ones_like(x), x, y])
    keep = np.isfinite(v)
    coef = np.linalg.lstsq(a[keep], v[keep], rcond=None)[0]
    resid = v - a @ coef
    mad = 1.4826 * np.nanmedian(np.abs(resid[keep] - np.nanmedian(resid[keep])))
    keep &= np.abs(resid) <= outlier_sd * max(mad, 1e-6)
    coef = np.linalg.lstsq(a[keep], v[keep], rcond=None)[0]
    return coef, keep, float(np.std((v - a @ coef)[keep]))


class ChannelShift:
    """Measure, model, store and apply the shift from a segmentation to another channel.

    Parameters
    ----------
    store_path : str | Path
        Trench store.
    segmentation : str
        ``/segmentation/<segmentation>`` group whose labels are moved.
    hypothesis : str
        Which of its hypotheses to use.
    channel : str
        The channel the labels are moved onto, a value of the root ``C`` coordinate.
    polarity : str
        ``"dark"`` if cells are darker than their surroundings in ``channel`` (phase
        contrast), ``"bright"`` if brighter (fluorescence).
    name : str | None
        Group name under ``/channel_shift``; defaults to ``<segmentation>_to_<channel>``.
    **params
        Overrides of :data:`DEFAULT_PARAMS`.
    """

    def __init__(
        self,
        store_path: Union[str, Path],
        segmentation: str = "mcherry",
        hypothesis: str = "S1",
        channel: str = "PC",
        polarity: str = "dark",
        name: Optional[str] = None,
        **params: Any,
    ) -> None:
        if polarity not in ("dark", "bright"):
            raise ValueError("polarity must be 'dark' or 'bright'")
        self.store_path = Path(store_path)
        self.root = xr.open_zarr(self.store_path, consolidated=False)
        channels = [str(c) for c in self.root["C"].values]
        if channel not in channels:
            raise ValueError(f"channel {channel!r} not in store channels {channels}")
        seg = xr.open_zarr(self.store_path, group=f"segmentation/{segmentation}", consolidated=False)
        hyps = [str(h) for h in seg["Hypothesis"].values]
        if hypothesis not in hyps:
            raise ValueError(f"hypothesis {hypothesis!r} not in {hyps}")
        self.segmentation, self.hypothesis, self.channel, self.polarity = segmentation, hypothesis, channel, polarity
        self.channel_index, self.hypothesis_index = channels.index(channel), hyps.index(hypothesis)
        self.name = name or f"{segmentation}_to_{channel.lower()}"
        self.group = f"channel_shift/{self.name}"
        unknown = set(params) - set(DEFAULT_PARAMS)
        if unknown:
            raise ValueError(f"unknown parameters {sorted(unknown)}")
        self.params = {**DEFAULT_PARAMS, **params}

    # ------------------------------------------------------------------
    # Geometry: trench-crop positions in field-of-view coordinates
    # ------------------------------------------------------------------

    def _fov_positions(self, segment_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Field-of-view (x, y) of each trench's segments, and the sign that turns a
        crop dy into a field-of-view dy (flipped trenches run upwards)."""
        r = self.root
        xc = (r["x_left"].values + r["x_right"].values) / 2.0
        flip = r["needs_flip"].values.astype(bool)
        top, bottom = r["y_top"].values, r["y_bottom"].values
        y = np.where(flip[:, None], bottom[:, None] - 1 - segment_y[None, :], top[:, None] + segment_y[None, :])
        x = np.repeat(xc[:, None], len(segment_y), axis=1)
        return x, y.astype(np.float64), np.where(flip, -1.0, 1.0)

    # ------------------------------------------------------------------
    # Measure and fit
    # ------------------------------------------------------------------

    def measure(self, workers: int = 16, show_progress: bool = True) -> xr.Dataset:
        """Measure every trench, fit the model, write ``/channel_shift/<name>``; return it."""
        p = self.params
        n_trench = self.root.sizes["Trench"]
        height = self.root.sizes["Y"]
        edges = np.linspace(0, height, p["segments"] + 1)
        segment_y = (edges[:-1] + edges[1:]) / 2 - 0.5
        jobs = [(str(self.store_path), f"segmentation/{self.segmentation}/labels", self.hypothesis_index,
                 self.channel_index, i, self.polarity, p) for i in range(n_trench)]
        measured = np.full((n_trench, p["segments"], 2), np.nan, np.float32)
        quality = np.full((n_trench, p["segments"]), np.nan, np.float32)
        n_frames = np.zeros((n_trench, p["segments"]), np.int16)
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for i, shift, q, used in tqdm(pool.map(_measure_trench, jobs), total=n_trench,
                                          desc=f"shift {self.segmentation} -> {self.channel}", disable=not show_progress):
                measured[i], quality[i], n_frames[i] = shift, q, used
        modelled, model = self._fit(measured, segment_y)
        ds = self._dataset(measured, modelled, quality, n_frames, segment_y, model)
        ds.to_zarr(str(self.store_path), group=self.group, mode="w", consolidated=False, zarr_format=3)
        return ds

    def _fit(self, measured: np.ndarray, segment_y: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        x, y, sign = self._fov_positions(segment_y)
        dy_fov = measured[..., 0] * sign[:, None]
        model: Dict[str, Any] = {"form": "c0 + cx * x + cy * y, x and y in field-of-view pixels; dy in "
                                         "field-of-view orientation (crop dy times -1 for flipped trenches)"}
        out = np.empty_like(measured)
        for k, (axis, v) in enumerate((("dy", dy_fov), ("dx", measured[..., 1]))):
            coef, keep, sd = fit_affine(x.ravel(), y.ravel(), v.ravel().astype(np.float64), self.params["outlier_sd"])
            model[axis] = {"c0": coef[0], "cx": coef[1], "cy": coef[2], "residual_sd_px": sd,
                           "n_used": int(keep.sum()), "n_rejected": int(np.isfinite(v).sum() - keep.sum())}
            pred = coef[0] + coef[1] * x + coef[2] * y
            out[..., k] = pred * (sign[:, None] if axis == "dy" else 1.0)
        return out.astype(np.float32), model

    def _dataset(self, measured, modelled, quality, n_frames, segment_y, model) -> xr.Dataset:
        dims3, dims2 = ("Trench", "Segment", "ShiftAxis"), ("Trench", "Segment")
        return xr.Dataset(
            data_vars={
                "measured": (dims3, measured, {"long_name": "measured shift, trench-crop px (NaN: not measurable)"}),
                "modelled": (dims3, modelled, {"long_name": "shift from the field-of-view model, trench-crop px"}),
                "quality": (dims2, quality, {"long_name": "depth of the cost minimum below its median, normalised intensity"}),
                "n_frames": (dims2, n_frames, {"long_name": "frames summed for the measurement"}),
                "segment_y": (("Segment",), segment_y.astype(np.float32), {"long_name": "segment centre row in the trench crop"}),
            },
            coords={"Trench": self.root["Trench"].variable, "Segment": np.arange(len(segment_y)),
                    "ShiftAxis": np.asarray(["y", "x"], dtype=object)},
            attrs={
                "schema_version": SCHEMA_VERSION,
                "layout": "channel_shift",
                "description": (f"Shift that moves /segmentation/{self.segmentation} ({self.hypothesis}) labels "
                                f"onto the {self.channel} channel: the label at crop (y, x) belongs at (y + dy, x + dx)."),
                "segmentation": self.segmentation, "hypothesis": self.hypothesis, "channel": self.channel,
                "polarity": self.polarity, "params": dict(self.params),
                "model": _json(model), "provenance": provenance(),
            },
        )

    # ------------------------------------------------------------------
    # Read and apply
    # ------------------------------------------------------------------

    def load(self) -> xr.Dataset:
        return xr.open_zarr(self.store_path, group=self.group, consolidated=False)

    def row_shifts(self, index: int) -> np.ndarray:
        """Modelled (dy, dx) for every row of trench ``index``'s crop, shape (Y, 2)."""
        model = self.load().attrs["model"]
        rows = np.arange(self.root.sizes["Y"], dtype=np.float64)
        x, y, sign = self._fov_positions(rows)
        x, y = x[index], y[index]
        dy = (model["dy"]["c0"] + model["dy"]["cx"] * x + model["dy"]["cy"] * y) * sign[index]
        dx = model["dx"]["c0"] + model["dx"]["cx"] * x + model["dx"]["cy"] * y
        return np.column_stack([dy, dx])

    def apply(self, labels: np.ndarray, index: int) -> np.ndarray:
        """Move a trench's labels, ``(Y, X)`` or ``(T, Y, X)``, onto the channel
        (nearest-neighbour, so labels stay labels)."""
        s = self.row_shifts(index)
        h, w = labels.shape[-2:]
        yy, xx = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij")
        # output pixel (y, x) takes the label from (y - dy, x - dx); dy is read at the output row
        src = np.stack([yy - s[:, 0][:, None], xx - s[:, 1][:, None]])
        one = lambda lab: ndimage.map_coordinates(lab, src, order=0, mode="constant", cval=0)
        if labels.ndim == 2:
            return one(labels)
        return np.stack([one(lab) for lab in labels]) if len(labels) else labels.copy()

    def __repr__(self) -> str:
        return (f"ChannelShift({self.store_path.name!r}, {self.segmentation}/{self.hypothesis} -> "
                f"{self.channel}, polarity={self.polarity!r})")


def _json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json(v) for k, v in value.items()}
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value
