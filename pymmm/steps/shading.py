"""ShadingCorrector – take each FOV's shading out of the detection channel, with the
PDMS between neighbouring trenches as the reference.

The PDMS between two trenches is the same material in every FOV and at every place in
it, so where it looks brighter, more light is reaching it. A smooth surface through it
(each lane on its own, on the trench detection image) is the FOV's background ``B``.
The correction brings ``B`` to one level ``L``, the median over the experiment::

    subtract   I - (B - L)                        light added, e.g. scattered by the chip
    divide     (I - D) (L - D) / (B - D) + D      a gain, e.g. uneven illumination
    off        I

``D`` is the camera's dark level. The PDMS ends at ``L`` either way; the modes differ for
darker or brighter material such as cells, and which holds depends on the experiment.
:meth:`ShadingCorrector.evidence` shows how the cells follow the background: one for
one when light is added, in proportion when it is a gain.

Extraction applies it to every frame of the detection channel before cutting the
trenches (:class:`FovShading`); the frame's own spacers set its level, so a change of
brightness over time comes out too. Nothing is corrected where a FOV has no spacers to
measure (:attr:`ShadingCorrector.skipped`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from pymmm.source.nd2 import ND2Experiment
from pymmm.steps.registration import Registrator
from pymmm.steps.trenches import TrenchDefinition, TrenchDetector
from pymmm.store import lineage
from pymmm.store.companion import CompanionStore

MODES = ("subtract", "divide", "off")
TILE = 32  # px: the background is fitted to the median of each tile of spacer
STEP = 16  # px between the points of the saved surface
MIN_GAP = 10  # px: narrower gaps between trenches are not measured
WIDE_GAP = 1.5  # gaps this many times the lane's median hold a missed trench: skipped
MIN_TILES = 6  # a lane with fewer tiles of spacer is not fitted
OUTLIER_MADS = 4.0  # tiles this far from the first fit (debris) are dropped
# how smooth the surface is (per tile, coordinates in thousands of px): enough to average
# the tiles' noise and to stay level where trenches are missing, little enough to follow
# what the PDMS shows from one trench to the next
SMOOTHING = 3e-4
SCALE = 1000.0
MIN_PIXELS = 200  # spacer pixels a frame needs to set its own level


# ======================================================================
# The background, its surface and the correction
# ======================================================================


def correct(
    image: np.ndarray, background: np.ndarray, level: float, offset: float, mode: str
) -> np.ndarray:
    """``image`` with its ``background`` brought to ``level`` (module docstring)."""
    image = np.asarray(image, dtype=np.float32)
    if mode == "subtract":
        return image - (background - np.float32(level))
    if mode == "divide":
        dark = np.float32(offset)
        scale = (np.float32(level) - dark) / np.maximum(background - dark, 1.0)
        return (image - dark) * scale + dark
    if mode == "off":
        return image
    raise ValueError(f"mode must be one of {MODES}, not {mode!r}")


def lane_spacers(
    trenches: Sequence[TrenchDefinition],
) -> list[tuple[int, int, np.ndarray]]:
    """Each lane's rows and the columns of PDMS between its trenches: the middle half
    of every gap of at least :data:`MIN_GAP` px between neighbouring trenches. A gap
    more than :data:`WIDE_GAP` times the lane's median holds a trench that was missed or
    dropped, so it is skipped. ``[(y_top, y_bottom, columns), ...]``, one per lane,
    columns sorted.
    """
    lanes: dict[int, list[TrenchDefinition]] = {}
    for trench in trenches:
        lanes.setdefault(trench.lane_index, []).append(trench)
    out = []
    for lane in sorted(lanes):
        row = sorted(lanes[lane], key=lambda t: t.x_left)
        gaps = [(a, b, b.x_left - a.x_right) for a, b in pairwise(row)]
        usable = [gap for _, _, gap in gaps if gap >= MIN_GAP]
        widest = WIDE_GAP * np.median(usable) if usable else 0
        columns = [
            np.arange(a.x_right + gap // 4, b.x_left - gap // 4)
            for a, b, gap in gaps
            if MIN_GAP <= gap <= widest
        ]
        out.append(
            (
                min(t.y_top for t in row),
                max(t.y_bottom for t in row),
                np.concatenate(columns) if columns else np.zeros(0, int),
            )
        )
    return out


def grid_axis(size: int) -> np.ndarray:
    """The surface's points along an axis of ``size`` px: every :data:`STEP` px, and
    the last pixel.
    """
    points = np.arange(0, size, STEP)
    return points if points[-1] == size - 1 else np.append(points, size - 1)


def _weights(points: np.ndarray, at: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    index = np.clip(np.searchsorted(points, at, side="right") - 1, 0, len(points) - 2)
    span = points[index + 1] - points[index]
    return index, np.clip((at - points[index]) / span, 0.0, 1.0).astype(np.float32)


def sample(
    surface: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """The surface (on :func:`grid_axis` points of an image of ``shape``) at these
    pixel rows and columns, linearly interpolated: ``(len(rows), len(columns))``.
    """
    i, wy = _weights(grid_axis(shape[0]), np.asarray(rows))
    j, wx = _weights(grid_axis(shape[1]), np.asarray(columns))
    top = surface[i][:, j] * (1 - wx) + surface[i][:, j + 1] * wx
    bottom = surface[i + 1][:, j] * (1 - wx) + surface[i + 1][:, j + 1] * wx
    return (top * (1 - wy)[:, None] + bottom * wy[:, None]).astype(np.float32)


def _tiles(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centres and medians of the tiles at least a fifth covered by ``mask``."""
    points, values = [], []
    for y in range(0, image.shape[0], TILE):
        for x in range(0, image.shape[1], TILE):
            m = mask[y : y + TILE, x : x + TILE]
            if m.sum() >= 0.2 * TILE * TILE:
                points.append((y + m.shape[0] / 2, x + m.shape[1] / 2))
                values.append(np.median(image[y : y + TILE, x : x + TILE][m]))
    return np.array(points, float).reshape(-1, 2), np.array(values, float)


def _smooth(points: np.ndarray, values: np.ndarray, smoothing: float = SMOOTHING):
    """A thin-plate smoothing surface through ``values`` at ``points`` (y, x)."""
    from scipy.interpolate import RBFInterpolator

    rbf = RBFInterpolator(
        points / SCALE,
        values,
        kernel="thin_plate_spline",
        smoothing=smoothing * len(values),
        degree=1,
    )
    return lambda at: rbf(np.asarray(at, float) / SCALE)


def fit_surface(
    image: np.ndarray, lanes: Sequence[tuple[int, int, np.ndarray]]
) -> np.ndarray | None:
    """A FOV's background on :func:`grid_axis` points: a thin-plate smoothing surface
    through the tile medians of each lane's spacers (:func:`lane_spacers`; tiles far off
    the first fit dropped), over that lane's rows;
    above and below them, the nearest lane's edge, out to half way to the next lane.
    ``None`` when no lane has :data:`MIN_TILES` tiles of spacer.
    """
    height, width = image.shape
    gy, gx = grid_axis(height), grid_axis(width)
    surface = np.full((len(gy), len(gx)), np.nan, np.float32)
    covered = []  # (first, last) grid row of each fitted lane
    for y_top, y_bottom, columns in lanes:
        mask = np.zeros(image.shape, bool)
        mask[y_top:y_bottom, columns] = True
        if not mask.any():
            continue
        # the warp's zero fill, and half-filled pixels at the image's corners
        mask &= image > 0.5 * np.median(image[mask])
        points, values = _tiles(image, mask)
        if len(values) < MIN_TILES:
            continue
        first = _smooth(points, values)
        residual = values - first(points)
        mad = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        keep = np.abs(residual) <= OUTLIER_MADS * mad + 1e-6
        fit = _smooth(points[keep], values[keep]) if keep.sum() >= MIN_TILES else first
        rows = np.flatnonzero((gy >= y_top - STEP) & (gy < y_bottom + STEP))
        yy, xx = np.meshgrid(gy[rows], gx, indexing="ij")
        surface[rows] = fit(np.column_stack([yy.ravel(), xx.ravel()])).reshape(yy.shape)
        covered.append((rows[0], rows[-1]))
    if not covered:
        return None
    covered.sort()
    for k, (first_row, last_row) in enumerate(covered):
        start = 0 if k == 0 else (covered[k - 1][1] + first_row) // 2 + 1
        stop = (
            len(gy)
            if k == len(covered) - 1
            else (last_row + covered[k + 1][0]) // 2 + 1
        )
        surface[start:first_row] = surface[first_row]
        surface[last_row + 1 : stop] = surface[last_row]
    return surface


@dataclass(frozen=True)
class FovShading:
    """What extraction needs to correct one FOV's frames (picklable, for the
    workers): its background surface, the correction, and each lane's spacers.
    """

    surface: np.ndarray  # on grid_axis points of an image of `shape`
    shape: tuple[int, int]  # the registered image's (height, width)
    level: float
    offset: float
    mode: str
    channel: int  # the corrected channel's index in the experiment
    spacers: tuple[tuple[int, int, tuple[int, ...]], ...]  # (y_top, y_bottom, columns)

    def correct_band(self, band: np.ndarray, y_top: int) -> np.ndarray:
        """Correct the registered frame's rows ``y_top:y_top + len(band)`` (a lane's
        band, as extraction warps it). The frame's own spacers in these rows, when there
        are enough of them, move the background to the frame's level.
        """
        rows = np.arange(y_top, y_top + band.shape[0])
        background = sample(self.surface, rows, np.arange(band.shape[1]), self.shape)
        for top, bottom, columns in self.spacers:
            if top == y_top and bottom == y_top + band.shape[0] and columns:
                values = band[:, list(columns)]
                keep = values > 0
                if keep.sum() >= MIN_PIXELS:
                    reference = background[:, list(columns)][keep]
                    background += np.median(values[keep]) - np.median(reference)
                break
        corrected = correct(band, background, self.level, self.offset, self.mode)
        corrected[band <= 0] = 0  # the warp's fill stays empty
        return corrected


# ======================================================================
# ShadingCorrector
# ======================================================================


class ShadingCorrector:
    """Measure each FOV's shading on the PDMS between its trenches, for extraction to
    take out of the detection channel (module docstring).

    Parameters
    ----------
    experiment : ND2Experiment
        Source experiment.
    registrator : Registrator
        Completed registrator.
    trench_detector : TrenchDetector
        Completed trench detector: its trenches say where the spacers are, and its
        detection image is what the background is fitted to.
    store : CompanionStore
        Companion zarr store for checkpointing.
    mode : {"subtract", "divide", "off"}
        How the background comes out (module docstring).
    offset : float
        The camera's dark level in counts, for ``"divide"``.
    """

    def __init__(
        self,
        experiment: ND2Experiment,
        registrator: Registrator,
        trench_detector: TrenchDetector,
        store: CompanionStore,
        mode: str = "subtract",
        offset: float = 100.0,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
        self.experiment = experiment
        self.registrator = registrator
        self.trench_detector = trench_detector
        self.store = store
        self.mode = mode
        self.offset = round(float(offset), 6)
        self.channel = trench_detector.channel
        self._surfaces: dict[str, np.ndarray | None] | None = None
        self._level: float | None = None
        self._evidence: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._surfaces is not None

    def _fitted(self) -> dict[str, np.ndarray | None]:
        if self._surfaces is None:
            raise RuntimeError("Shading not measured yet. Call fit() first.")
        return self._surfaces

    @property
    def level(self) -> float:
        """The level every FOV's background is brought to: the median over the
        experiment.
        """
        self._fitted()
        return float(self._level)

    @property
    def skipped(self) -> list[str]:
        """FOVs left uncorrected: no lane had enough spacer to fit."""
        return [fov for fov, s in self._fitted().items() if s is None]

    @property
    def shape(self) -> tuple[int, int]:
        return (
            int(self.experiment.data.sizes["Y"]),
            int(self.experiment.data.sizes["X"]),
        )

    def detection_images(
        self, fovs: Sequence[str] | None = None
    ) -> dict[str, np.ndarray]:
        """The trench detection images (registered means), which the backgrounds are
        fitted to.
        """
        return self.registrator.compute_registered_means(
            list(fovs) if fovs is not None else None,
            channel=self.channel,
            **self.trench_detector.detection_image,
        )

    # ------------------------------------------------------------------
    # Measuring
    # ------------------------------------------------------------------

    def fit_fov(self, image: np.ndarray, fov: str) -> np.ndarray | None:
        """One FOV's background surface from its detection image (:func:`fit_surface`)."""
        return fit_surface(
            image, lane_spacers(self.trench_detector.trenches.get(fov, []))
        )

    def fit(self) -> None:
        """Fit every FOV's background, the level they are brought to, and the evidence
        (:meth:`evidence`).
        """
        images = self.detection_images()
        surfaces = {
            fov: self.fit_fov(images[fov], fov) for fov in self.experiment.fov_names
        }
        # each FOV's background over its lanes' rows (not the rows copied beyond them)
        gy = grid_axis(self.shape[0])
        levels = []
        for fov, surface in surfaces.items():
            if surface is None:
                continue
            lanes = lane_spacers(self.trench_detector.trenches.get(fov, []))
            rows = np.zeros(len(gy), bool)
            for top, bottom, _ in lanes:
                rows |= (gy >= top) & (gy < bottom)
            levels.append(np.median(surface[rows] if rows.any() else surface))
        self._surfaces = surfaces
        self._level = float(np.median(levels)) if levels else 0.0
        self._evidence = self._measure_evidence(images)
        print(
            f"Shading measured in {len(levels)} of {len(surfaces)} FOVs; backgrounds "
            f"brought to {self._level:.0f} counts ({self.mode})"
        )
        if self.skipped:
            print(
                f"  not corrected, too little spacer between trenches: {self.skipped}"
            )

    def background(self, fov: str) -> np.ndarray | None:
        """A FOV's background at every pixel of its registered frame."""
        surface = self._fitted().get(fov)
        if surface is None:
            return None
        height, width = self.shape
        return sample(surface, np.arange(height), np.arange(width), self.shape)

    def correct_image(self, image: np.ndarray, fov: str) -> np.ndarray:
        """A registered image of ``fov`` corrected with its background (without the
        per-frame level extraction adds), as float32.
        """
        background = self.background(fov)
        if background is None:
            return np.asarray(image, np.float32)
        out = correct(image, background, self.level, self.offset, self.mode)
        out[np.asarray(image) <= 0] = 0
        return out

    def fov_shading(self, fov: str) -> FovShading | None:
        """What extraction corrects ``fov``'s frames with; ``None`` when nothing is
        corrected (mode ``"off"``, or the FOV is skipped).
        """
        surface = self._fitted().get(fov)
        if self.mode == "off" or surface is None:
            return None
        return FovShading(
            surface=surface,
            shape=self.shape,
            level=self.level,
            offset=self.offset,
            mode=self.mode,
            channel=self.experiment.channel_names.index(self.channel),
            spacers=tuple(
                (top, bottom, tuple(int(c) for c in columns))
                for top, bottom, columns in lane_spacers(
                    self.trench_detector.trenches.get(fov, [])
                )
            ),
        )

    # ------------------------------------------------------------------
    # Evidence: how the cells follow the background
    # ------------------------------------------------------------------

    def _measure_evidence(self, images: dict[str, np.ndarray]) -> pd.DataFrame:
        rows = []
        for fov, trenches in self.trench_detector.trenches.items():
            surface = self._surfaces.get(fov)
            if surface is None or fov not in images:
                continue
            image = images[fov]
            for t in trenches:
                length, width = t.y_bottom - t.y_top, t.x_right - t.x_left
                # the half of the trench by its opening, where the cells are packed
                if t.orientation == 1:
                    y0, y1 = (
                        t.y_bottom - int(0.55 * length),
                        t.y_bottom - int(0.05 * length),
                    )
                else:
                    y0, y1 = t.y_top + int(0.05 * length), t.y_top + int(0.55 * length)
                x0, x1 = t.x_left + int(0.35 * width), t.x_right - int(0.35 * width)
                strip = image[y0:y1, x0:x1]
                strip = strip[strip > 0]
                if strip.size < 50:
                    continue
                background = sample(
                    surface, np.arange(y0, y1), np.arange(x0, x1), self.shape
                )
                rows.append(
                    {
                        "trench_id": t.trench_id,
                        "fov": fov,
                        "background": float(np.median(background)),
                        "cells": float(np.percentile(strip, 5)),
                    }
                )
        return pd.DataFrame(rows, columns=["trench_id", "fov", "background", "cells"])

    def evidence(self) -> pd.DataFrame:
        """One row per trench: the background at its cells (``background``) and how
        dark its cells are (``cells``: the darkest 5% of the middle of the trench's half
        by its opening, on the detection image). Cells that brighten one for one with
        the background say light is added (subtract); in proportion, a gain (divide).
        """
        self._fitted()
        return self._evidence.copy()

    def evidence_slopes(self) -> dict[str, float]:
        """How much the cells brighten per count of background (``measured``), and what
        each mode expects: ``subtract`` 1, ``divide`` the cells' level over the
        background's (both above the camera offset).
        """
        table = self.evidence()
        if len(table) < 3 or np.ptp(table["background"]) == 0:
            return {"measured": float("nan"), "subtract": 1.0, "divide": float("nan")}
        slope = np.polyfit(table["background"], table["cells"], 1)[0]
        ratio = np.median(
            (table["cells"] - self.offset) / (table["background"] - self.offset)
        )
        return {"measured": float(slope), "subtract": 1.0, "divide": float(ratio)}

    def plot(self, ax=None):
        """The evidence (:meth:`evidence`): each trench's cells against its background,
        with what each mode expects.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4.5))
        table = self.evidence()
        slopes = self.evidence_slopes()
        x, y = table["background"].to_numpy(), table["cells"].to_numpy()
        ax.scatter(x, y, s=6, c="k", alpha=0.5, label="one trench")
        if len(x):
            xs = np.linspace(x.min(), x.max(), 10)
            mx, my = np.median(x), np.median(y)
            ax.plot(
                xs, my + (xs - mx), "b:", lw=2, label="subtract (light added): slope 1"
            )
            if np.isfinite(slopes["divide"]):
                ax.plot(
                    xs,
                    my + slopes["divide"] * (xs - mx),
                    "r--",
                    label=f"divide (a gain): slope {slopes['divide']:.2f}",
                )
            if np.isfinite(slopes["measured"]):
                ax.plot(
                    xs,
                    my + slopes["measured"] * (xs - mx),
                    "k-",
                    lw=0.8,
                    label=f"fit: slope {slopes['measured']:.2f}",
                )
        ax.set_xlabel("PDMS background at the trench (counts)")
        ax.set_ylabel("cells: darkest 5% (counts)")
        ax.set_title(f"How the cells follow the shading ({self.channel})")
        ax.legend(fontsize=8)
        return ax

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def _surface_array(self) -> xr.DataArray:
        height, width = self.shape
        fovs = list(self._fitted())
        stack = np.full(
            (len(fovs), len(grid_axis(height)), len(grid_axis(width))),
            np.nan,
            np.float32,
        )
        for k, fov in enumerate(fovs):
            if self._surfaces[fov] is not None:
                stack[k] = self._surfaces[fov]
        return xr.DataArray(
            stack,
            dims=("FOV", "gy", "gx"),
            coords={"FOV": fovs, "gy": grid_axis(height), "gx": grid_axis(width)},
        )

    def fingerprint(self) -> str:
        """The fingerprint of the backgrounds and how they come out
        (:mod:`pymmm.store.lineage`).
        """
        surfaces = self._surface_array()
        return lineage.shading_fingerprint(
            surfaces.values,
            list(surfaces.coords["FOV"].values),
            self.mode,
            self.offset,
            self.level,
        )

    def save(self) -> None:
        """Write the backgrounds and the evidence to the companion store, with the
        registration and trenches they were measured on.
        """
        self._fitted()
        self.store.save_shading(
            self._surface_array(),
            self._evidence,
            params={
                "channel": self.channel,
                "mode": self.mode,
                "offset": self.offset,
                "level": self.level,
                "skipped": self.skipped,
                **self.trench_detector.detection_image,
            },
            source_subset_metadata=self.experiment.source_subset_metadata,
            made_from={
                "registration": lineage.of(self.registrator),
                "trench_detection": lineage.of(self.trench_detector),
            },
        )
        print(f"Shading saved to {self.store.path}")

    @classmethod
    def load(
        cls,
        experiment: ND2Experiment,
        registrator: Registrator,
        trench_detector: TrenchDetector,
        store: CompanionStore,
    ) -> ShadingCorrector:
        """Reconstruct a ShadingCorrector from a checkpoint."""
        if not store.has_shading():
            raise FileNotFoundError("No shading checkpoint found.")
        data = store.load_shading()
        params = data["params"]
        corrector = cls(
            experiment,
            registrator,
            trench_detector,
            store,
            mode=params["mode"],
            offset=params["offset"],
        )
        surfaces = data["surface"]
        corrector._surfaces = {
            str(fov): (None if np.all(np.isnan(s)) else np.asarray(s, np.float32))
            for fov, s in zip(surfaces.coords["FOV"].values, surfaces.values)
        }
        corrector._level = float(params["level"])
        corrector._evidence = data["evidence"]
        return corrector

    def matches(self, mode: str, offset: float) -> bool:
        """Whether these settings are the ones it corrects with (the offset to the
        decimals a spin box shows).
        """
        return mode == self.mode and round(float(offset), 6) == self.offset

    def __repr__(self) -> str:
        if self._surfaces is None:
            return f"ShadingCorrector({self.mode}, not measured)"
        n = len(self._surfaces) - len(self.skipped)
        return (
            f"ShadingCorrector({self.mode}, {n} of {len(self._surfaces)} FOVs, "
            f"level {self.level:.0f})"
        )


def describe(params: dict[str, Any]) -> str:
    """The saved settings in a few words (for status lines)."""
    mode = params.get("mode")
    if mode == "off":
        return "off"
    extra = f", camera offset {params.get('offset'):g}" if mode == "divide" else ""
    return f"{mode}, to {params.get('level', 0):.0f} counts{extra}"
