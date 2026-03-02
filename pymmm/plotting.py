"""Diagnostic plotting functions for PyMMM.

All functions are standalone — no class dependencies.
"""

from __future__ import annotations

from itertools import cycle
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


def plot_drift_diagnostics(
    tmats: "xr.DataArray",
    fov_label: str = "",
    save_path: Optional[str] = None,
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

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

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

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_mean_image(
    image: np.ndarray,
    title: str = "Mean image",
    save_path: Optional[str] = None,
) -> None:
    """Display a single image with a greyscale colormap."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap="Greys_r")
    ax.set_title(title)
    ax.grid(which="both", color="w", linestyle="-", linewidth=0.5, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_1d_profile_with_peaks(
    profile: np.ndarray,
    peaks: np.ndarray,
    title: str = "1D profile",
    xlabel: str = "Position (px)",
    ylabel: str = "Intensity",
    save_path: Optional[str] = None,
) -> None:
    """Plot a 1-D profile with peak markers."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(profile)
    ax.plot(peaks, profile[peaks], "x", c="r", markersize=10, label="Peaks")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_fov_with_lanes(
    image: np.ndarray,
    lane_y_positions: List[float],
    orientations: Optional[List[int]] = None,
    title: str = "",
    save_path: Optional[str] = None,
) -> None:
    """Show an image with horizontal lines at lane y-positions.

    Parameters
    ----------
    image : np.ndarray
        2-D image to display.
    lane_y_positions : list[float]
        Y-coordinates of lane centres.
    orientations : list[int] | None
        ``+1`` or ``-1`` per lane. If given, annotated on the plot.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(image, cmap="Greys_r")
    for i, y in enumerate(lane_y_positions):
        color = "r"
        ax.axhline(y, color=color, lw=1.5, alpha=0.8)
        label = f"Lane {i}"
        if orientations is not None:
            label += f" (ori={orientations[i]:+d})"
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
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_fov_with_trenches(
    image: np.ndarray,
    trench_definitions: list,
    title: str = "",
    save_path: Optional[str] = None,
) -> None:
    """Show an image with coloured rectangles for trench boundaries.

    Parameters
    ----------
    image : np.ndarray
        2-D image.
    trench_definitions : list
        Sequence of objects with ``x_left``, ``x_right``, ``y_top``,
        ``y_bottom``, and ``trench_id`` attributes.
    """
    colors = cycle(["red", "green", "blue", "yellow", "orange", "purple", "cyan"])
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(image, cmap="Greys_r")

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
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_fov_grid(
    images: Dict[str, np.ndarray],
    overlays: Optional[Dict[str, list]] = None,
    n_cols: int = 2,
    save_dir: Optional[str] = None,
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
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 8 * n_rows))
    axes_flat = np.atleast_1d(axes).flatten()

    for i, (fov, img) in enumerate(images.items()):
        ax = axes_flat[i]
        ax.imshow(img, cmap="Greys_r")
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
    save_path: Optional[str] = None,
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
    """
    total_t = trench_stack.shape[0]
    indices = np.linspace(0, total_t - 1, n_frames, dtype=int)

    fig, axes = plt.subplots(1, n_frames, figsize=(2 * n_frames, 6))
    for ax, idx in zip(axes, indices):
        ax.imshow(trench_stack[idx], cmap="Greys_r")
        ax.set_title(f"t={idx}")
        ax.set_axis_off()

    fig.suptitle(f"Trench {trench_id}", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
