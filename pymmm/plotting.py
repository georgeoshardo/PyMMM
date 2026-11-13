"""Diagnostic plotting functions for PyMMM.

All functions are standalone — no class dependencies.
"""

from __future__ import annotations

from itertools import cycle
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    import xarray as xr

CONTRAST_PERCENTILES = (5.0, 95.0)
"""Lower/upper intensity percentiles that set the contrast of every PyMMM
image: saved diagnostics and the ndv viewers."""

NDV_CLIMS = {
    "clim_type": "percentile",
    "min_percentile": CONTRAST_PERCENTILES[0],
    "max_percentile": CONTRAST_PERCENTILES[1],
}
"""ndv LUT ``clims`` policy implementing :data:`CONTRAST_PERCENTILES`."""


def contrast_limits(image: np.ndarray) -> tuple[float, float]:
    """``(vmin, vmax)`` for displaying ``image`` at :data:`CONTRAST_PERCENTILES`."""
    vmin, vmax = np.nanpercentile(np.asarray(image), CONTRAST_PERCENTILES)
    return float(vmin), float(vmax)


def _save_figure(plot_func, kwargs: dict) -> str:
    plot_func(**kwargs, show=False)
    return kwargs["save_path"]


def save_figures(plot_func, jobs: list[dict]) -> None:
    """Render and save one figure per job without showing them.

    Each job is the keyword arguments for ``plot_func`` and must include
    ``save_path``. Figures are rendered on the active dask distributed
    cluster when there is one (each worker is its own process, so matplotlib
    runs safely in parallel), otherwise one after another.
    """
    from pymmm.parallel import dask_client

    client = dask_client()
    if client is None:
        for kwargs in jobs:
            _save_figure(plot_func, kwargs)
        return
    # Scatter the jobs first: they carry full-size images, which are too large
    # to embed in the task graph.
    job_futures = client.scatter(jobs, hash=False)
    client.gather(
        [client.submit(_save_figure, plot_func, job, pure=False) for job in job_futures]
    )


def _subplots(show: bool, *args, **kwargs):
    """``plt.subplots`` for a figure to show; for one only saved, a figure outside
    pyplot (``matplotlib.figure.Figure``), which renders with Agg whatever the backend.
    A figure only saved can then be drawn from any thread, as the napari dock's
    background runs do: pyplot's Qt backend makes its figures on the Qt thread only.
    """
    if show:
        return plt.subplots(*args, **kwargs)
    from matplotlib.figure import Figure

    figure_kwargs = {
        key: kwargs.pop(key)
        for key in ("figsize", "constrained_layout", "layout", "dpi")
        if key in kwargs
    }
    fig = Figure(**figure_kwargs)
    return fig, fig.subplots(*args, **kwargs)


def _finish(fig, save_path: str | None, show: bool) -> None:
    """Save a figure if asked, then show it or release it."""
    if save_path:
        fig.savefig(save_path)
    if show:
        plt.show()


def plot_drift_diagnostics(
    tmats: xr.DataArray,
    fov_label: str = "",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """3-panel drift diagnostic: spatial trace, X drift vs time, Y drift vs time.

    Parameters
    ----------
    tmats : xr.DataArray
        Transformation matrices with shape ``(T, 3, 3)`` or ``(T, row, col)``.
    fov_label : str
        Label for the plot title.
    save_path : str | None
        If given, save the figure to this path.
    show : bool
        Show the figure in the notebook; ``False`` only saves it.
    """
    # Extract translations
    vals = tmats.values if hasattr(tmats, "values") else np.asarray(tmats)
    x_drift = vals[:, 0, 2]
    y_drift = vals[:, 1, 2]

    # Time axis
    if hasattr(tmats, "coords") and "T" in tmats.coords:
        t_vals = tmats.coords["T"].values
        time_hours = t_vals / (1000 * 3600)
        xlabel = "Time (h)"
    else:
        time_hours = np.arange(len(x_drift))
        xlabel = "Frame"

    fig, axs = _subplots(show, 1, 3, figsize=(12, 4))

    # Panel 1: Spatial trajectory
    axs[0].plot(x_drift, y_drift, lw=1, alpha=0.8, color="k")
    axs[0].scatter(x_drift[0], y_drift[0], c="green", label="Start", zorder=5)
    axs[0].scatter(x_drift[-1], y_drift[-1], c="red", label="End", zorder=5)
    axs[0].set_title(f"Spatial Trace: {fov_label}")
    axs[0].set_xlabel("X Drift (px)")
    axs[0].set_ylabel("Y Drift (px)")
    axs[0].legend()
    axs[0].grid(True, linestyle="--", alpha=0.5)

    # Panel 2: X drift vs time
    axs[1].plot(time_hours, x_drift, lw=1.5, color="tab:blue")
    axs[1].set_title("X Drift vs Time")
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel("X Drift (px)")
    axs[1].grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Y drift vs time
    axs[2].plot(time_hours, y_drift, lw=1.5, color="tab:orange")
    axs[2].set_title("Y Drift vs Time")
    axs[2].set_xlabel(xlabel)
    axs[2].set_ylabel("Y Drift (px)")
    axs[2].grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    _finish(fig, save_path, show)


def plot_mean_image(
    image: np.ndarray,
    title: str = "Mean image",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Display a single image with a greyscale colormap."""
    fig, ax = _subplots(show, figsize=(10, 10))
    vmin, vmax = contrast_limits(image)
    ax.imshow(image, cmap="Greys_r", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.grid(which="both", color="w", linestyle="-", linewidth=0.5, alpha=0.3)
    fig.tight_layout()
    _finish(fig, save_path, show)


def plot_1d_profile_with_peaks(
    profile: np.ndarray,
    peaks: np.ndarray,
    title: str = "1D profile",
    xlabel: str = "Position (px)",
    ylabel: str = "Intensity",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot a 1-D profile with peak markers."""
    fig, ax = _subplots(show, figsize=(12, 4))
    ax.plot(profile)
    ax.plot(peaks, profile[peaks], "x", c="r", markersize=10, label="Peaks")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    _finish(fig, save_path, show)


def plot_fov_with_lanes(
    image: np.ndarray,
    lane_y_positions: list[float],
    orientations: list[int] | None = None,
    title: str = "",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Show an image with horizontal lines at lane y-positions.

    Parameters
    ----------
    image : np.ndarray
        2-D image to display.
    lane_y_positions : list[float]
        Y-coordinates of lane centres.
    orientations : list[int] | None
        ``+1`` or ``-1`` per lane. If given, lanes are drawn red (trenches
        above, opening down) or cyan (trenches below, opening up), as in the
        lane widget.
    title : str
        Plot title.
    save_path : str | None
        File to save the figure to; ``None``: not saved.
    show : bool
        Show the figure; otherwise it is closed after saving.
    """
    fig, ax = _subplots(show, figsize=(14, 8))
    vmin, vmax = contrast_limits(image)
    ax.imshow(image, cmap="Greys_r", vmin=vmin, vmax=vmax)
    for i, y in enumerate(lane_y_positions):
        color = "cyan" if orientations is not None and orientations[i] == -1 else "red"
        ax.axhline(y, color=color, lw=1.5, alpha=0.8)
        label = f"Lane {i}"
        if orientations is not None:
            label += " (open down)" if orientations[i] == 1 else " (open up)"
        ax.text(
            image.shape[1] * 0.02,
            y - 10,
            label,
            color=color,
            fontsize=10,
            verticalalignment="bottom",
        )
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    _finish(fig, save_path, show)


def plot_fov_with_trenches(
    image: np.ndarray,
    trench_definitions: list,
    title: str = "",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Show an image with coloured rectangles for trench boundaries.

    Parameters
    ----------
    image : np.ndarray
        2-D image.
    trench_definitions : list
        Sequence of objects with ``x_left``, ``x_right``, ``y_top``,
        ``y_bottom``, and ``trench_id`` attributes.
    title : str
        Plot title.
    save_path : str | None
        File to save the figure to; ``None``: not saved.
    show : bool
        Show the figure; otherwise it is closed after saving.
    """
    colors = cycle(["red", "green", "blue", "yellow", "orange", "purple", "cyan"])
    fig, ax = _subplots(show, figsize=(14, 8))
    vmin, vmax = contrast_limits(image)
    ax.imshow(image, cmap="Greys_r", vmin=vmin, vmax=vmax)

    for td, color in zip(trench_definitions, colors):
        ax.axvspan(td.x_left, td.x_right, alpha=0.1, color=color)
        ax.axvline(td.x_left, color=color, lw=0.5)
        ax.axvline(td.x_right, color=color, lw=0.5)
        ax.axhline(td.y_top, color=color, lw=0.5, alpha=0.5)
        ax.axhline(td.y_bottom, color=color, lw=0.5, alpha=0.5)
        ax.text(
            (td.x_left + td.x_right) / 2,
            td.y_top - 5,
            str(td.trench_id),
            color=color,
            fontsize=9,
            ha="center",
            va="bottom",
        )

    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    _finish(fig, save_path, show)


def plot_fov_grid(
    images: dict[str, np.ndarray],
    overlays: dict[str, list] | None = None,
    n_cols: int = 2,
    save_dir: str | None = None,
) -> None:
    """Grid of FOV images with optional trench overlays.

    Parameters
    ----------
    images : dict[str, np.ndarray]
        Mapping from FOV name to 2-D image.
    overlays : dict[str, list] | None
        Mapping from FOV name to list of trench definitions.
    n_cols : int
        Number of columns in the grid.
    save_dir : str | None
        Directory to save individual per-FOV figures.
    """
    n = len(images)
    n_rows = int(np.ceil(n / n_cols))
    _fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 8 * n_rows))
    axes_flat = np.atleast_1d(axes).flatten()

    for i, (fov, img) in enumerate(images.items()):
        ax = axes_flat[i]
        vmin, vmax = contrast_limits(img)
        ax.imshow(img, cmap="Greys_r", vmin=vmin, vmax=vmax)
        ax.set_title(fov)
        ax.set_axis_off()

        if overlays and fov in overlays:
            colors = cycle(
                ["red", "green", "blue", "yellow", "orange", "purple", "cyan"]
            )
            for td, color in zip(overlays[fov], colors):
                ax.axvspan(td.x_left, td.x_right, alpha=0.1, color=color)
                ax.axvline(td.x_left, color=color, lw=0.5)
                ax.axvline(td.x_right, color=color, lw=0.5)

    # Hide unused axes
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    if save_dir:
        from pathlib import Path

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(str(Path(save_dir) / "fov_grid.png"))
    plt.show()


def plot_trench_preview(
    trench_stack: np.ndarray,
    trench_id: int = 0,
    n_frames: int = 8,
    save_path: str | None = None,
) -> None:
    """Timepoint montage of one trench.

    Parameters
    ----------
    trench_stack : np.ndarray
        Array of shape ``(T, Y, X)`` for a single trench.
    trench_id : int
        ID for the title.
    n_frames : int
        Number of evenly-spaced frames to show.
    save_path : str | None
        File to save the figure to; ``None``: not saved.
    """
    total_t = trench_stack.shape[0]
    indices = np.linspace(0, total_t - 1, n_frames, dtype=int)

    fig, axes = plt.subplots(1, n_frames, figsize=(2 * n_frames, 6))
    vmin, vmax = contrast_limits(trench_stack[indices])
    for ax, idx in zip(axes, indices):
        ax.imshow(trench_stack[idx], cmap="Greys_r", vmin=vmin, vmax=vmax)
        ax.set_title(f"t={idx}")
        ax.set_axis_off()

    fig.suptitle(f"Trench {trench_id}", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_track_kymograph(
    store_path,
    trench_id: int,
    run: str,
    channel: str = "PC",
    minutes: float | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """One trench's kymograph beside the same kymograph with a tracking run's tracks.

    Each column is one frame: the image along the trench's centre line (the mean of the
    7 columns around it). Over it, each cell's mask on that line, coloured by its
    tracklet, so a band that keeps its colour is one cell from birth to division.
    Marked from the graph: divisions (white tick), tracks that end above the bottom
    before the last frame (red cross: a lysis, or a lost cell) and tracks that start
    with no parent after the first frame (yellow triangle: there should be none).

    Parameters
    ----------
    store_path : str | Path
        Experiment group.
    trench_id : int
        The trench.
    run : str
        Tracking run: ``tracks/<run>.geff`` and ``labels/tracks.<run>`` of the trench.
    channel : str
        Channel of the kymograph.
    minutes : float | None
        Show only the first ``minutes``; ``None``: every frame.
    save_path : str | None
        File to save the figure to; ``None``: not saved.
    show : bool
        Show the figure; otherwise it is closed after saving.
    """
    import zarr
    from matplotlib.colors import hsv_to_rgb
    from scipy import ndimage

    from pymmm.store.experiment_group import (
        calibration,
        channel_index,
        image,
        open_experiment,
        trench_path,
    )

    root = open_experiment(store_path)
    px, dt = calibration(root)
    base = f"{root.store.root}/{trench_path(trench_id)}"
    pixels = np.asarray(image(root, trench_id)[:, channel_index(root, channel)]).astype(
        float
    )
    masks = np.asarray(zarr.open_array(f"{base}/labels/tracks.{run}/0", mode="r")[:])
    graph = zarr.open_group(f"{base}/tracks/{run}.geff", mode="r")
    t, seg, track, y = (
        np.asarray(graph[f"nodes/props/{k}/values"][:])
        for k in ("t", "seg_id", "track_id", "y")
    )
    position = {int(i): k for k, i in enumerate(np.asarray(graph["nodes/ids"][:]))}
    edges = [
        (position[int(a)], position[int(b)])
        for a, b in np.asarray(graph["edges/ids"][:]).reshape(-1, 2)
    ]
    T, H, W = masks.shape

    centre = int(np.median(np.nonzero(masks)[2])) if masks.any() else W // 2
    kymo = pixels[:, :, max(centre - 3, 0) : centre + 4].mean(2).T  # (y, t)
    lo, hi = np.percentile(kymo, [1, 99.5])
    kymo = np.clip((kymo - lo) / max(hi - lo, 1e-9), 0, 1)

    # a hue per tracklet, stepped by the golden ratio so that neighbouring numbers differ
    hues = (np.arange(64) * 0.61803398875) % 1
    colours = hsv_to_rgb(np.c_[hues, np.full(64, 0.78), np.full(64, 0.95)])
    line = masks[:, :, max(centre - 3, 0) : centre + 4].max(
        2
    )  # (t, y): the mask on the centre line
    tracklet = {(int(a), int(b)): int(c) for a, b, c in zip(t, seg, track)}
    bands = np.zeros((H, T, 4))
    rows = {}
    for frame in range(T):
        for value in np.unique(line[frame][line[frame] > 0]):
            k = tracklet.get((frame, int(value)))
            if k is not None:
                bands[line[frame] == value, frame, :3] = colours[k % 64]
                bands[line[frame] == value, frame, 3] = 0.55
        for value, where in enumerate(ndimage.find_objects(masks[frame]), start=1):
            if where is not None:
                rows[(frame, value)] = (where[0].start, where[0].stop - 1)
    children, parents = np.zeros(len(t), int), np.zeros(len(t), int)
    for a, b in edges:
        children[a] += 1
        parents[b] += 1
    lowest = np.zeros(T, int)
    for (frame, _), (_, bottom) in rows.items():
        lowest[frame] = max(lowest[frame], bottom)
    end = (
        int(np.median(lowest[lowest > 0])) if (lowest > 0).any() else H
    )  # where the masks stop
    cell = float(np.median([b - a + 1 for a, b in rows.values()])) if rows else 0.0
    nodes = range(len(t))
    divisions = [k for k in nodes if children[k] == 2]
    ends = [
        k
        for k in nodes
        if children[k] == 0
        and t[k] < T - 1
        and rows[(int(t[k]), int(seg[k]))][1] < end - 1.5 * cell
    ]
    starts = [k for k in nodes if parents[k] == 0 and t[k] > t.min()]

    extent = (0, T * dt, H * px, 0)
    fig, axes = _subplots(
        show, 1, 2, figsize=(13, 8), constrained_layout=True, sharey=True
    )
    for ax in axes:
        ax.imshow(
            kymo, cmap="gray", aspect="auto", extent=extent, interpolation="nearest"
        )
        ax.set_xlabel("time (min)")
        if minutes is not None:
            ax.set_xlim(0, minutes)
    axes[0].set_ylabel("distance from the closed end (µm)")
    axes[0].set_title(channel, loc="left")
    axes[1].imshow(bands, aspect="auto", extent=extent, interpolation="nearest")
    for which, marker, colour, size in (
        (divisions, "_", "white", 26),
        (ends, "X", "#e5202a", 70),
        (starts, "^", "#f2c500", 60),
    ):
        if which:
            tick = marker == "_"
            axes[1].scatter(
                (t[which] + 0.5) * dt,
                y[which] * px,
                marker=marker,
                s=size,
                c=colour,
                edgecolors=None if tick else "black",
                linewidths=1.2 if tick else 0.6,
                zorder=5,
            )
    axes[1].set_title(
        f"{run}: {len(divisions)} divisions, {len(ends)} tracks end above the bottom, "
        f"{len(starts)} start with no parent",
        loc="left",
    )
    fig.suptitle(
        f"trench {trench_id}: a band of one colour is one cell from birth to division.  White tick: division.  "
        "Red cross: ends above the bottom.  Yellow triangle: starts with no parent.",
        x=0.01,
        ha="left",
        fontsize=9,
    )
    _finish(fig, save_path, show)
