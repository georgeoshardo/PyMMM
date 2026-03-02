"""ND2Experiment – lazy ND2 file loader with uniform xarray interface."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import nd2
import numpy as np
import xarray as xr

from pymmm._utils import (
    normalize_channel_arg,
    normalize_fov_arg,
    rename_duplicate_coords,
)


class ND2Experiment:
    """Load an ND2 file lazily and provide a uniform xarray interface.

    Parameters
    ----------
    path : str | Path
        Path to the ND2 file.

    Examples
    --------
    >>> exp = ND2Experiment("/path/to/data.nd2")
    >>> print(exp)
    >>> # Interactive browse:
    >>> exp.data.hvplot.image(x="X", y="Y", cmap="Greys_r", dynamic=True,
    ...                       rasterize=True, widget_location="top", aspect="equal")
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"ND2 file not found: {self.path}")

        # Load lazily with dask backing and xarray labels
        self._raw_data: xr.DataArray = nd2.imread(
            str(self.path), dask=True, xarray=True
        )

        # Handle duplicate position names (common in ND2 files)
        if "P" in self._raw_data.coords:
            self._raw_data = rename_duplicate_coords(self._raw_data, "P")

        # Compute pixel size from Y coordinate spacing
        if "Y" in self._raw_data.coords:
            y_coords = self._raw_data.coords["Y"].values
            self._pixel_size_um: float = float(abs(y_coords[1] - y_coords[0]))
        else:
            self._pixel_size_um = 1.0

        # Compute time interval from T coordinate spacing
        if "T" in self._raw_data.coords and self._raw_data.sizes.get("T", 0) > 1:
            t_coords = self._raw_data.coords["T"].values
            self._time_interval_ms: float = float(abs(t_coords[1] - t_coords[0]))
        else:
            self._time_interval_ms = 0.0

        # Track discarded FOVs / time subsetting
        self._discarded_fovs: List[str] = []
        self._time_slice: Optional[slice] = None

    # ------------------------------------------------------------------
    # Core data access
    # ------------------------------------------------------------------

    @property
    def data(self) -> xr.DataArray:
        """Lazy dask-backed xarray DataArray, with any subsetting applied."""
        d = self._raw_data
        if self._discarded_fovs and "P" in d.dims:
            keep = [f for f in d.coords["P"].values if f not in self._discarded_fovs]
            d = d.sel(P=keep)
        if self._time_slice is not None and "T" in d.dims:
            d = d.isel(T=self._time_slice)
        return d

    # ------------------------------------------------------------------
    # Metadata properties
    # ------------------------------------------------------------------

    @property
    def experiment_name(self) -> str:
        """Experiment name derived from the ND2 filename stem."""
        return self.path.stem

    @property
    def fov_names(self) -> List[str]:
        """List of FOV (position) names after subsetting."""
        if "P" not in self.data.dims:
            return ["single"]
        return [str(v) for v in self.data.coords["P"].values]

    @property
    def channel_names(self) -> List[str]:
        """List of channel names."""
        if "C" not in self.data.dims:
            return ["single"]
        return [str(v) for v in self.data.coords["C"].values]

    @property
    def n_fovs(self) -> int:
        return len(self.fov_names) if "P" in self.data.dims else 1

    @property
    def n_timepoints(self) -> int:
        return int(self.data.sizes.get("T", 1))

    @property
    def has_z(self) -> bool:
        return "Z" in self.data.dims

    @property
    def has_channels(self) -> bool:
        return "C" in self.data.dims

    @property
    def pixel_size_um(self) -> float:
        return self._pixel_size_um

    @property
    def time_interval_ms(self) -> float:
        return self._time_interval_ms

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def get_frame(
        self,
        fov: Union[int, str] = 0,
        time: int = 0,
        channel: Union[int, str] = 0,
    ) -> xr.DataArray:
        """Return a single ``(Y, X)`` frame (lazy).

        Parameters
        ----------
        fov : int | str
            FOV index or name.
        time : int
            Time index.
        channel : int | str
            Channel index or name.
        """
        d = self.data
        if "P" in d.dims:
            fov_name = normalize_fov_arg(fov, self.fov_names)
            d = d.sel(P=fov_name)
        if "T" in d.dims:
            d = d.isel(T=time)
        if "C" in d.dims:
            ch_name = normalize_channel_arg(channel, self.channel_names)
            d = d.sel(C=ch_name)
        return d

    def get_fov_stack(
        self,
        fov: Union[int, str] = 0,
        channel: Union[int, str] = 0,
    ) -> xr.DataArray:
        """Return a ``(T, Y, X)`` stack for one FOV and channel (lazy).

        Parameters
        ----------
        fov : int | str
            FOV index or name.
        channel : int | str
            Channel index or name.
        """
        d = self.data
        if "P" in d.dims:
            fov_name = normalize_fov_arg(fov, self.fov_names)
            d = d.sel(P=fov_name)
        if "C" in d.dims:
            ch_name = normalize_channel_arg(channel, self.channel_names)
            d = d.sel(C=ch_name)
        return d

    # ------------------------------------------------------------------
    # Subsetting
    # ------------------------------------------------------------------

    def discard_fovs(self, fov_names: List[str]) -> "ND2Experiment":
        """Mark FOVs to be excluded from ``data``. Returns self for chaining."""
        self._discarded_fovs.extend(fov_names)
        return self

    def select_times(self, start: int, end: int) -> "ND2Experiment":
        """Restrict time dimension to ``[start, end)``. Returns self for chaining."""
        self._time_slice = slice(start, end)
        return self

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ND2Experiment":
        return self

    def __exit__(self, *exc) -> None:
        pass

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        dims = dict(self.data.sizes)
        dim_str = " × ".join(f"{k}={v}" for k, v in dims.items())
        lines = [
            f"ND2Experiment: {self.experiment_name}",
            f"  Path: {self.path}",
            f"  Dims: {dim_str}",
            f"  FOVs: {self.n_fovs}  ({', '.join(self.fov_names[:3])}{'...' if self.n_fovs > 3 else ''})",
            f"  Channels: {', '.join(self.channel_names)}",
            f"  Timepoints: {self.n_timepoints}",
            f"  Pixel size: {self.pixel_size_um:.4f} µm",
        ]
        if self.time_interval_ms > 0:
            lines.append(f"  Time interval: {self.time_interval_ms:.1f} ms")
        if self.has_z:
            lines.append(f"  Z slices: {self.data.sizes.get('Z', '?')}")
        return "\n".join(lines)
