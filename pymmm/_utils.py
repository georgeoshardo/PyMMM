"""Shared helpers for PyMMM."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


def mean_without_fill(image: np.ndarray, axis: int) -> np.ndarray:
    """The mean of ``image`` along ``axis``, leaving out its zero pixels: the fill a
    warp leaves where it has no image (the corners a rotation opens, the edge drift
    moves in). A line with no image at all takes the value of the nearest one with
    some, so the profile has no gaps.
    """
    image = np.asarray(image, dtype=float)
    data = image != 0
    counts = data.sum(axis=axis)
    profile = np.zeros(counts.shape)
    np.divide(
        np.where(data, image, 0).sum(axis=axis), counts, out=profile, where=counts > 0
    )
    valid = counts > 0
    if valid.any() and not valid.all():
        lines = np.arange(len(profile))
        profile[~valid] = np.interp(lines[~valid], lines[valid], profile[valid])
    return profile


def get_diagnostics_dir(experiment_path: Path, kind: str | None = None) -> Path:
    """Return (and create) the diagnostics directory next to the data file.

    ``kind`` selects a subfolder for one kind of diagnostic (``"drift"``,
    ``"lanes"``, ``"trenches"``).
    """
    diag_dir = experiment_path.parent / f"{experiment_path.stem}_diagnostics"
    if kind is not None:
        diag_dir = diag_dir / kind
    diag_dir.mkdir(parents=True, exist_ok=True)
    return diag_dir


def rename_duplicate_coords(
    data: xr.DataArray | xr.Dataset,
    dim: str,
) -> xr.DataArray | xr.Dataset:
    """Rename duplicate coordinate values by appending ``-01``, ``-02`` suffixes.

    Values that occur only once are left unchanged.  Values that occur
    multiple times are *all* renamed to ensure uniqueness and order
    preservation.

    Parameters
    ----------
    data : xr.DataArray | xr.Dataset
        Input xarray object.
    dim : str
        Dimension whose coordinate labels may contain duplicates.

    Returns
    -------
    xr.DataArray | xr.Dataset
        Copy with unique coordinate labels.
    """
    if dim not in data.coords:
        raise ValueError(f"Dimension '{dim}' not found in coordinates.")

    original_values: list[Any] = data.coords[dim].values.tolist()
    value_counts: Counter = Counter(original_values)
    current_counts: defaultdict = defaultdict(int)

    new_values: list[str] = []
    for val in original_values:
        current_counts[val] += 1
        if value_counts[val] > 1:
            new_values.append(f"{val}-{current_counts[val]:02d}")
        else:
            new_values.append(str(val))

    return data.assign_coords({dim: new_values})


def normalize_channel_arg(channel: int | str, channel_names: list[str]) -> str:
    """Convert an int channel index to its string name.

    Parameters
    ----------
    channel : int | str
        Channel index or name.
    channel_names : list[str]
        Available channel names.

    Returns
    -------
    str
        Resolved channel name.
    """
    if isinstance(channel, int):
        return channel_names[channel]
    if channel not in channel_names:
        raise ValueError(f"Channel '{channel}' not in {channel_names}")
    return channel


def normalize_fov_arg(fov: int | str, fov_names: list[str]) -> str:
    """Convert an int FOV index to its string name.

    Parameters
    ----------
    fov : int | str
        FOV index or name.
    fov_names : list[str]
        Available FOV names.

    Returns
    -------
    str
        Resolved FOV name.
    """
    if isinstance(fov, int):
        return fov_names[fov]
    if fov not in fov_names:
        raise ValueError(f"FOV '{fov}' not in {fov_names}")
    return fov
