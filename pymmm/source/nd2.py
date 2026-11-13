"""ND2Experiment – lazy ND2 file loader with uniform xarray interface."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path

import nd2
import numpy as np
import xarray as xr

from pymmm._utils import (
    normalize_channel_arg,
    normalize_fov_arg,
    rename_duplicate_coords,
)
from pymmm.source.metadata import (
    extract_acquisition_metadata,
    slice_to_dict,
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
    >>> ndv.imshow(exp.data)  # browse it (the notebook extra)
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"ND2 file not found: {self.path}")

        # Load lazily with dask backing and xarray labels
        self._raw_data: xr.DataArray = nd2.imread(
            str(self.path), dask=True, xarray=True
        )
        self._raw_fov_names = (
            [str(v) for v in self._raw_data.coords["P"].values]
            if "P" in self._raw_data.coords
            else ["single"]
        )

        # Handle duplicate position names (common in ND2 files)
        if "P" in self._raw_data.coords:
            self._raw_data = rename_duplicate_coords(self._raw_data, "P")

        # Coordinate spacing is only the fallback; the acquisition metadata
        # below takes pixel size from the ND2 voxel size and the interval from
        # the real frame times.
        if "Y" in self._raw_data.coords:
            y_coords = self._raw_data.coords["Y"].values
            fallback_pixel_size_um = float(abs(y_coords[1] - y_coords[0]))
        else:
            fallback_pixel_size_um = 1.0
        if "T" in self._raw_data.coords and self._raw_data.sizes.get("T", 0) > 1:
            t_coords = self._raw_data.coords["T"].values
            fallback_time_interval_ms = float(abs(t_coords[1] - t_coords[0]))
        else:
            fallback_time_interval_ms = 0.0

        # Track discarded FOVs / time subsetting
        self._discarded_fovs: list[str] = []
        self._time_slice: slice | None = None
        self._y_slice: slice | None = None
        self._x_slice: slice | None = None
        self._source_acquisition_metadata = extract_acquisition_metadata(
            self.path,
            pixel_size_um=fallback_pixel_size_um,
            time_interval_ms=fallback_time_interval_ms,
        )
        file_summary = self._source_acquisition_metadata["file_summary"]
        self._pixel_size_um: float = file_summary["pixel_size_um"]
        self._time_interval_ms: float = file_summary["time_interval_ms"]

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
        if self._y_slice is not None and "Y" in d.dims:
            d = d.isel(Y=self._y_slice)
        if self._x_slice is not None and "X" in d.dims:
            d = d.isel(X=self._x_slice)
        return d

    # ------------------------------------------------------------------
    # Metadata properties
    # ------------------------------------------------------------------

    @property
    def experiment_name(self) -> str:
        """Experiment name derived from the ND2 filename stem."""
        return self.path.stem

    @property
    def fov_names(self) -> list[str]:
        """List of FOV (position) names after subsetting."""
        if "P" not in self.data.dims:
            return ["single"]
        return [str(v) for v in self.data.coords["P"].values]

    @property
    def channel_names(self) -> list[str]:
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

    @property
    def source_acquisition_metadata(self) -> dict:
        """Full acquisition/configuration metadata extracted from the ND2."""
        return copy.deepcopy(self._source_acquisition_metadata)

    @property
    def stage_layout(self):
        """Where each FOV sat on the stage and how the camera sits on it
        (:class:`~pymmm.source.stage.StageLayout`, by raw P index); ``None`` when the
        ND2 has no XY loop or camera matrix.
        """
        from pymmm.source.stage import StageLayout

        return StageLayout.from_metadata(
            self._source_acquisition_metadata, self.pixel_size_um
        )

    @property
    def source_axes(self) -> dict:
        """Map the current subset back to indices in the source ND2.

        Returns ``fov_names`` (active FOVs), ``original_p_index`` (each active
        FOV's P index in the ND2) and ``original_t_index`` (each active
        timepoint's T index in the ND2), the latter two as int32 arrays.
        """
        if "P" in self._raw_data.dims:
            raw_fov_names = [str(v) for v in self._raw_data.coords["P"].values]
            fov_names = list(self.fov_names)
            original_p_index = np.asarray(
                [raw_fov_names.index(name) for name in fov_names],
                dtype=np.int32,
            )
        else:
            fov_names = ["single"]
            original_p_index = np.asarray([0], dtype=np.int32)

        if "T" in self._raw_data.dims:
            n_raw_times = int(self._raw_data.sizes["T"])
            time_slice = self._time_slice or slice(None)
            original_t_index = np.arange(
                *time_slice.indices(n_raw_times), dtype=np.int32
            )
        else:
            original_t_index = np.asarray([0], dtype=np.int32)

        return {
            "fov_names": fov_names,
            "original_p_index": original_p_index,
            "original_t_index": original_t_index,
        }

    def raw_fov_index(self, fov: str) -> int:
        """P index of a FOV in the source ND2 (0 when the file has no P axis)."""
        if "P" not in self._raw_data.dims:
            return 0
        return [str(v) for v in self._raw_data.coords["P"].values].index(str(fov))

    @property
    def source_roi(self) -> tuple:
        """``(y_roi, x_roi)`` as picklable ``(start, stop, step)`` tuples or ``None``."""
        return tuple(
            (s.start, s.stop, s.step) if s is not None else None
            for s in (self._y_slice, self._x_slice)
        )

    @property
    def source_subset_metadata(self) -> dict:
        """Pipeline-side subsetting and provenance derived from the experiment."""
        discarded = list(dict.fromkeys(str(v) for v in self._discarded_fovs))
        roi = {
            "y": slice_to_dict(self._y_slice),
            "x": slice_to_dict(self._x_slice),
        }
        return {
            "raw_fov_names": list(self._raw_fov_names),
            "active_fov_names": list(self.fov_names),
            "discarded_fov_names": discarded,
            "time_slice": slice_to_dict(self._time_slice),
            "roi": roi,
            "raw_timepoint_count": int(self._raw_data.sizes.get("T", 1)),
            "active_timepoint_count": int(self.n_timepoints),
        }

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def get_frame(
        self,
        fov: int | str = 0,
        time: int = 0,
        channel: int | str = 0,
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
        fov: int | str = 0,
        channel: int | str = 0,
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

    def _named_fovs(self, fovs: Sequence[str | int]) -> list[str]:
        """The names of ``fovs``, each a FOV name or the FOV's integer P index in the ND2
        (``10`` is ``P`` 10, whatever is selected). A name or index the ND2 does not have
        raises ``ValueError``: it would otherwise select or discard nothing, silently.
        """
        all_fovs = self._all_fov_names()
        names = []
        for fov in fovs:
            if isinstance(fov, (int, np.integer)) and not isinstance(fov, bool):
                if not 0 <= fov < len(all_fovs):
                    raise ValueError(
                        f"FOV index {fov} is out of range: the ND2 has {len(all_fovs)} FOVs "
                        f"(P 0-{len(all_fovs) - 1})"
                    )
                names.append(all_fovs[fov])
            elif str(fov) in all_fovs:
                names.append(str(fov))
            else:
                shown = ", ".join(
                    all_fovs
                    if len(all_fovs) <= 6
                    else [*all_fovs[:3], "...", all_fovs[-1]]
                )
                raise ValueError(
                    f"no FOV named {fov!r} in the ND2 ({shown}); give a name, "
                    f"or a P index from 0 to {len(all_fovs) - 1}"
                )
        return names

    def _all_fov_names(self) -> list[str]:
        """Every FOV's name, in the ND2's ``P`` order."""
        if "P" not in self._raw_data.dims:
            return []
        return [str(v) for v in self._raw_data.coords["P"].values]

    def select_fovs(self, fovs: Sequence[str | int]) -> ND2Experiment:
        """Keep only the listed FOVs, discarding all others. Returns self for chaining.

        Each is a FOV name or its integer P index in the ND2; one the ND2 does not have
        raises ``ValueError``.

        Examples
        --------
        >>> exp.select_fovs(["XYPos:0", "XYPos:1"])  # by name
        >>> exp.select_fovs([0, 1])                   # P 0 and 1
        """
        keep = self._named_fovs(fovs)
        self._discarded_fovs.extend(f for f in self._all_fov_names() if f not in keep)
        return self

    def discard_fovs(self, fovs: Sequence[str | int]) -> ND2Experiment:
        """Leave the listed FOVs out of ``data``. Returns self for chaining.

        Each is a FOV name or its integer P index in the ND2 (``[10, 11]`` or
        ``["XYPos:10", "XYPos:11"]``); one the ND2 does not have raises ``ValueError``.
        """
        self._discarded_fovs.extend(self._named_fovs(fovs))
        return self

    def select_times(self, start: int, end: int) -> ND2Experiment:
        """Restrict time dimension to ``[start, end)``. Returns self for chaining."""
        self._time_slice = slice(start, end)
        return self

    def select_roi(self, y=None, x=None) -> ND2Experiment:
        """Restrict Y and X dimensions to a region of interest. Returns self for chaining."""
        if y is not None:
            self._y_slice = slice(y[0], y[1])
        if x is not None:
            self._x_slice = slice(x[0], x[1])
        return self

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> ND2Experiment:
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
