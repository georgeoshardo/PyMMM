"""Extractor – crop trenches from stabilised data and write to output zarr."""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional, Union

import dask
import dask.array as da
import numcodecs
import numpy as np
import xarray as xr
import zarr
from tqdm.auto import tqdm

from pymmm.experiment import ND2Experiment
from pymmm.metadata import (
    EXTRACTOR_STORE_VERSION,
    build_acquisition_dataset,
    build_events_dataset,
    build_source_frames_dataset,
    extract_source_frame_metadata,
)
from pymmm.registrator import Registrator
from pymmm.trench_detector import TrenchDetector

_mp_ctx = multiprocessing.get_context("spawn")


# ======================================================================
# Module-level helpers (picklable for client.submit / ProcessPoolExecutor)
# ======================================================================


def _try_get_dask_client():
    """Return the active dask distributed Client, or None."""
    try:
        from dask.distributed import get_client
        return get_client()
    except (ImportError, ValueError):
        return None


def _load_crop_rotate_warp(
    nd2_path, fov_idx, time_idx, channel_idx,
    y_roi, x_roi, rotation, tmat, out_dtype,
):
    """Load one frame, crop ROI, rotate, warp.

    Matches the ``get_stabilized_data()`` pipeline exactly:
    raw frame → channel select → Y/X crop → rotate → warp → cast.
    """
    import nd2
    from skimage.transform import rotate as ski_rotate, warp as ski_warp

    with nd2.ND2File(nd2_path) as f:
        coord_axes = [k for k in f.sizes if k not in {"X", "Y", "C", "S"}]
        coord_shape = tuple(f.sizes[k] for k in coord_axes)
        coord_map = {"T": time_idx, "P": fov_idx, "Z": 0}
        coord_tuple = tuple(coord_map.get(ax, 0) for ax in coord_axes)
        frame_idx = int(np.ravel_multi_index(coord_tuple, coord_shape))
        frame = f.read_frame(frame_idx).copy()

    if channel_idx is not None and frame.ndim == 3:
        frame = frame[channel_idx]

    img = frame.astype(np.float64)

    # ROI crop (before rotation, matching experiment.data pipeline)
    if y_roi is not None:
        img = img[y_roi[0]:y_roi[1]]
    if x_roi is not None:
        img = img[:, x_roi[0]:x_roi[1]]

    if rotation != 0.0:
        img = ski_rotate(img, rotation, preserve_range=True)

    warped = ski_warp(img, tmat, preserve_range=True, order=1)
    return warped.astype(out_dtype)


def _stabilize_fov(registrator, channel, fov):
    """Compute the full stabilised (T, Y, X) array for one FOV (preview)."""
    return (
        registrator.get_stabilized_data(channel=channel, fov=fov)
        .compute(scheduler="synchronous")
        .values
    )


def _crop_from_fov_data(fov_data, y_top, y_bottom, x_left, x_right, needs_flip):
    """Slice one trench from a pre-computed FOV numpy array (preview)."""
    crop = fov_data[:, y_top:y_bottom, x_left:x_right]
    if needs_flip:
        crop = crop[:, ::-1, :]
    return np.ascontiguousarray(crop)


class Extractor:
    """Crop trenches from the lazy stabilised DataArray and write to zarr.

    Parameters
    ----------
    experiment : ND2Experiment
        Source experiment.
    registrator : Registrator
        Completed registrator.
    trench_detector : TrenchDetector
        Completed trench detector.
    output_path : str | Path | None
        Path for the output zarr store. Defaults to
        ``<nd2_dir>/<experiment_name>.trenches.zarr``.
    """

    def __init__(
        self,
        experiment: ND2Experiment,
        registrator: Registrator,
        trench_detector: TrenchDetector,
        output_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.experiment = experiment
        self.registrator = registrator
        self.trench_detector = trench_detector

        if output_path is None:
            nd2_dir = experiment.path.parent
            output_path = nd2_dir / f"{experiment.experiment_name}.trenches.zarr"
        self.output_path = Path(output_path)

    def _root_store_attrs(self) -> dict[str, Any]:
        """Return the small root-attrs bundle for the final trench store."""
        exp = self.experiment
        return {
            "extractor_store_version": EXTRACTOR_STORE_VERSION,
            "layout": "xarray_trench_store_v1",
            "source_metadata_version": int(exp.source_metadata_version),
            "source_nd2": str(exp.path),
            "experiment_name": exp.experiment_name,
            "pixel_size_um": exp.pixel_size_um,
            "registration_params": {
                "channel": self.registrator.channel,
                "mode": str(self.registrator.mode),
                "rotation": self.registrator.rotation,
            },
            "source_subset_metadata": exp.source_subset_metadata,
        }

    @staticmethod
    def _metadata_encoding(
        dataset: xr.Dataset,
        *,
        chunk_overrides: dict[str, tuple[int, ...]] | None = None,
        compressor: Any | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Return xarray ``to_zarr`` encodings for the dataset variables."""
        chunk_overrides = chunk_overrides or {}
        encoding: dict[str, dict[str, Any]] = {}
        for name in dataset.data_vars:
            spec: dict[str, Any] = {}
            if name in chunk_overrides:
                spec["chunks"] = chunk_overrides[name]
            if compressor is not None and dataset[name].dtype != object:
                spec["compressor"] = compressor
            if spec:
                encoding[name] = spec
        return encoding

    def _build_root_dataset_template(
        self,
        *,
        trench_table,
        source_metadata: dict[str, Any],
        trench_h: int,
        trench_w: int,
        channel_names: list[str],
        data_shape: tuple[int, ...],
        data_chunks: tuple[int, ...],
        dtype: np.dtype,
    ) -> xr.Dataset:
        """Build the root xarray dataset template for the final trench store."""
        fov_lookup = {
            str(fov_name): idx
            for idx, fov_name in enumerate(source_metadata["fov_names"])
        }
        fov_index = np.asarray(
            [fov_lookup[str(fov)] for fov in trench_table["fov"].tolist()],
            dtype=np.int32,
        )

        return xr.Dataset(
            data_vars={
                "data": (
                    ("Trench", "T", "C", "Y", "X"),
                    da.empty(data_shape, chunks=data_chunks, dtype=dtype),
                ),
                "fov_name": (
                    ("Trench",),
                    np.asarray(trench_table["fov"].tolist(), dtype=object),
                ),
                "fov_index": (("Trench",), fov_index),
                "original_p_index": (
                    ("Trench",),
                    source_metadata["original_p_index"][fov_index],
                ),
                "lane_index": (
                    ("Trench",),
                    trench_table["lane_index"].to_numpy(dtype=np.int32),
                ),
                "x_left": (("Trench",), trench_table["x_left"].to_numpy(dtype=np.int32)),
                "x_right": (("Trench",), trench_table["x_right"].to_numpy(dtype=np.int32)),
                "y_top": (("Trench",), trench_table["y_top"].to_numpy(dtype=np.int32)),
                "y_bottom": (
                    ("Trench",),
                    trench_table["y_bottom"].to_numpy(dtype=np.int32),
                ),
                "orientation": (
                    ("Trench",),
                    trench_table["orientation"].to_numpy(dtype=np.int8),
                ),
                "needs_flip": (
                    ("Trench",),
                    trench_table["needs_flip"].to_numpy(dtype=bool),
                ),
                "source_seq_index": (
                    ("Trench", "T"),
                    source_metadata["source_seq_index"][fov_index],
                ),
                "relative_time_ms": (
                    ("Trench", "T"),
                    source_metadata["relative_time_ms"][fov_index],
                ),
                "absolute_julian_day_number": (
                    ("Trench", "T"),
                    source_metadata["absolute_julian_day_number"][fov_index],
                ),
                "pfs_offset": (
                    ("Trench", "T"),
                    source_metadata["pfs_offset"][fov_index],
                ),
                "stage_position_um": (
                    ("Trench", "T", "Axis"),
                    source_metadata["stage_position_um"][fov_index],
                ),
            },
            coords={
                "Trench": trench_table["trench_id"].to_numpy(dtype=np.int32),
                "T": source_metadata["original_t_index"],
                "C": np.asarray(channel_names, dtype=object),
                "Y": np.arange(trench_h, dtype=np.int32),
                "X": np.arange(trench_w, dtype=np.int32),
                "Axis": np.asarray(["x", "y", "z"], dtype=object),
            },
            attrs=self._root_store_attrs(),
        )

    def _write_metadata_groups(
        self,
        *,
        source_metadata: dict[str, Any],
        compressor: Any | None,
        n_times: int,
    ) -> None:
        """Write xarray-native subgroup datasets to the trench store."""
        source_frames = build_source_frames_dataset(source_metadata)
        source_frames.to_zarr(
            str(self.output_path),
            mode="a",
            group="source_frames",
            consolidated=False,
            zarr_format=2,
            encoding=self._metadata_encoding(
                source_frames,
                chunk_overrides={
                    "original_p_index": (
                        min(256, max(1, source_frames.sizes["FOV"])),
                    ),
                    "source_seq_index": (1, n_times),
                    "relative_time_ms": (1, n_times),
                    "absolute_julian_day_number": (1, n_times),
                    "pfs_offset": (1, n_times),
                    "stage_position_um": (1, n_times, 3),
                    "position_name": (1, n_times),
                    "frame_metadata_json": (1, n_times),
                },
                compressor=compressor,
            ),
        )

        events = build_events_dataset(
            source_metadata["frame_event_records"],
            source_metadata["acquisition_event_records"],
        )
        event_chunks: dict[str, tuple[int, ...]] = {}
        if "FrameEvent" in events.sizes:
            frame_chunk = min(4096, max(1, events.sizes["FrameEvent"]))
            event_chunks.update(
                {
                    name: (frame_chunk,)
                    for name in events.data_vars
                    if events[name].dims == ("FrameEvent",)
                }
            )
        if "AcquisitionEvent" in events.sizes:
            acquisition_chunk = min(4096, max(1, events.sizes["AcquisitionEvent"]))
            event_chunks.update(
                {
                    name: (acquisition_chunk,)
                    for name in events.data_vars
                    if events[name].dims == ("AcquisitionEvent",)
                }
            )
        events.to_zarr(
            str(self.output_path),
            mode="a",
            group="events",
            consolidated=False,
            zarr_format=2,
            encoding=self._metadata_encoding(
                events,
                chunk_overrides=event_chunks,
                compressor=compressor,
            ),
        )

        acquisition = build_acquisition_dataset(
            self.experiment.source_acquisition_metadata,
            channel_names=self.experiment.channel_names,
        )
        acquisition.to_zarr(
            str(self.output_path),
            mode="a",
            group="acquisition",
            consolidated=False,
            zarr_format=2,
            encoding=self._metadata_encoding(acquisition, compressor=compressor),
        )

    def extract(
        self,
        compressor: str = "zstd",
        clevel: int = 9,
        show_progress: bool = True,
    ) -> None:
        """Extract all trenches and write to the output zarr store.

        Uses per-frame ``client.submit()`` tasks (like ``compute_tmats``)
        so each distributed worker only handles ~1 MB per task.  Falls back
        to ``ProcessPoolExecutor`` when no distributed client is active.

        Parameters
        ----------
        compressor : str
            Compression codec name (``"zstd"``).
        clevel : int
            Compression level.
        show_progress : bool
            Show a tqdm progress bar.
        """
        trench_table = self.trench_detector.get_trench_table()
        if len(trench_table) == 0:
            raise RuntimeError("No trenches to extract.")

        trench_table = (
            trench_table.sort_values("trench_id")
            .reset_index(drop=True)
            .assign(
                _output_trench_index=lambda df: np.arange(len(df), dtype=np.int32),
            )
        )

        # Determine output shape from first trench
        first = trench_table.iloc[0]
        trench_h = int(first["y_bottom"] - first["y_top"])
        trench_w = int(first["x_right"] - first["x_left"])
        n_trenches = len(trench_table)
        exp = self.experiment
        n_times = exp.n_timepoints

        # Determine channels
        channel_names = exp.channel_names
        n_channels = len(channel_names)

        # Create output zarr store
        if compressor == "zstd":
            comp = numcodecs.Zstd(level=clevel)
        else:
            comp = None

        shape = (n_trenches, n_times, n_channels, trench_h, trench_w)
        chunks = (1, n_times, 1, trench_h, trench_w)

        dtype = np.dtype(exp.data.dtype)
        source_metadata = extract_source_frame_metadata(exp)
        root_dataset = self._build_root_dataset_template(
            trench_table=trench_table,
            source_metadata=source_metadata,
            trench_h=trench_h,
            trench_w=trench_w,
            channel_names=channel_names,
            data_shape=shape,
            data_chunks=chunks,
            dtype=dtype,
        )
        root_dataset.to_zarr(
            str(self.output_path),
            mode="w",
            consolidated=False,
            compute=False,
            zarr_format=2,
            encoding=self._metadata_encoding(
                root_dataset,
                chunk_overrides={
                    "data": chunks,
                    "fov_name": (min(4096, max(1, n_trenches)),),
                    "fov_index": (min(4096, max(1, n_trenches)),),
                    "original_p_index": (min(4096, max(1, n_trenches)),),
                    "lane_index": (min(4096, max(1, n_trenches)),),
                    "x_left": (min(4096, max(1, n_trenches)),),
                    "x_right": (min(4096, max(1, n_trenches)),),
                    "y_top": (min(4096, max(1, n_trenches)),),
                    "y_bottom": (min(4096, max(1, n_trenches)),),
                    "orientation": (min(4096, max(1, n_trenches)),),
                    "needs_flip": (min(4096, max(1, n_trenches)),),
                    "source_seq_index": (1, n_times),
                    "relative_time_ms": (1, n_times),
                    "absolute_julian_day_number": (1, n_times),
                    "pfs_offset": (1, n_times),
                    "stage_position_um": (1, n_times, 3),
                },
                compressor=comp,
            ),
        )
        self._write_metadata_groups(
            source_metadata=source_metadata,
            compressor=comp,
            n_times=n_times,
        )

        store = zarr.open_group(str(self.output_path), mode="r+")
        data_arr = store["data"]

        # ----------------------------------------------------------
        # Frame-level extraction parameters
        # ----------------------------------------------------------
        nd2_path = str(exp.path)
        rotation = self.registrator.rotation
        out_dtype = dtype
        H, W = exp.data.sizes["Y"], exp.data.sizes["X"]

        # Map subsetted T indices → original file T indices
        if "T" not in exp._raw_data.dims:
            original_t_indices = [0]
        elif exp._time_slice is not None:
            n_raw_T = exp._raw_data.sizes["T"]
            original_t_indices = list(range(*exp._time_slice.indices(n_raw_T)))
        else:
            original_t_indices = list(range(exp._raw_data.sizes["T"]))

        # Map FOV names → integer P indices for read_frame()
        raw_fov_names = (
            [str(v) for v in exp._raw_data.coords["P"].values]
            if "P" in exp._raw_data.dims else []
        )

        # ROI slices (as tuples for pickling)
        y_roi = (
            (exp._y_slice.start, exp._y_slice.stop)
            if exp._y_slice is not None else None
        )
        x_roi = (
            (exp._x_slice.start, exp._x_slice.stop)
            if exp._x_slice is not None else None
        )

        tmats_da = self.registrator.tmats  # (P, T, row, col) or (T, row, col)

        # ----------------------------------------------------------
        # Distribute per-frame warp tasks
        # ----------------------------------------------------------
        client = _try_get_dask_client()
        pool = None
        if client is None:
            pool = ProcessPoolExecutor(mp_context=_mp_ctx)

        try:
            grouped = trench_table.groupby("fov", sort=False)
            fov_iter = grouped
            if show_progress:
                fov_iter = tqdm(grouped, desc="Extracting FOVs", total=len(grouped))

            for fov, fov_df in fov_iter:
                fov_str = str(fov)
                fov_idx = raw_fov_names.index(fov_str) if raw_fov_names else 0

                # Tmats for this FOV: (T, 3, 3)
                if "P" in tmats_da.dims:
                    fov_tmats = tmats_da.sel(P=fov).values
                else:
                    fov_tmats = tmats_da.values

                for c_idx, ch in enumerate(channel_names):
                    ch_idx = (
                        exp.channel_names.index(ch)
                        if exp.has_channels else None
                    )
                    fov_data = self._warp_fov_frames(
                        client, pool, nd2_path, fov_idx, n_times,
                        original_t_indices, ch_idx, y_roi, x_roi,
                        rotation, fov_tmats, out_dtype, H, W,
                    )
                    for _, row in fov_df.iterrows():
                        crop = fov_data[
                            :, row["y_top"]:row["y_bottom"],
                            row["x_left"]:row["x_right"],
                        ]
                        if row["needs_flip"]:
                            crop = crop[:, ::-1, :]
                        data_arr[int(row["_output_trench_index"]), :, c_idx, :, :] = crop

                    del fov_data
        finally:
            if pool is not None:
                pool.shutdown(wait=True)

        zarr.consolidate_metadata(str(self.output_path))
        print(f"Extraction complete: {self.output_path}")
        print(f"  Shape: {shape}, Chunks: {chunks}")

    @staticmethod
    def _warp_fov_frames(
        client, pool, nd2_path, fov_idx, n_times,
        original_t_indices, ch_idx, y_roi, x_roi,
        rotation, fov_tmats, out_dtype, H, W,
    ) -> np.ndarray:
        """Warp all frames for one FOV/channel via distributed or PPE.

        Each task opens the ND2 file, reads one frame via random access,
        crops/rotates/warps it, and returns the result (~1 MB).  No worker
        ever holds a full FOV stack.
        """
        fov_data = np.empty((n_times, H, W), dtype=out_dtype)

        if client is not None:
            from dask.distributed import as_completed as dask_as_completed

            future_to_t = {
                client.submit(
                    _load_crop_rotate_warp,
                    nd2_path, fov_idx, original_t_indices[t], ch_idx,
                    y_roi, x_roi, rotation, fov_tmats[t], out_dtype,
                ): t
                for t in range(n_times)
            }
            for fut in dask_as_completed(future_to_t):
                fov_data[future_to_t[fut]] = fut.result()
        else:
            future_to_t = {
                pool.submit(
                    _load_crop_rotate_warp,
                    nd2_path, fov_idx, original_t_indices[t], ch_idx,
                    y_roi, x_roi, rotation, fov_tmats[t], out_dtype,
                ): t
                for t in range(n_times)
            }
            for fut in as_completed(future_to_t):
                fov_data[future_to_t[fut]] = fut.result()

        return fov_data

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def preview(self, channel: Optional[Union[str, int]] = None) -> xr.DataArray:
        """Return a lazy dask-backed DataArray previewing the extraction output.

        Uses ``dask.delayed`` per-FOV so the graph stays small (~N_FOVs +
        N_trenches tasks) regardless of timepoint count. Each trench is one
        chunk of shape ``(T, trench_h, trench_w)``.

        Use with hvplot for interactive browsing before committing to full
        extraction::

            extractor.preview(channel="PC").hvplot.image(
                x="X", y="Y", cmap="Greys_r", dynamic=True,
                rasterize=True, widget_location="top", aspect="equal",
            )

        Parameters
        ----------
        channel : str | int | None
            A specific channel to preview. ``None`` includes all channels
            (matching the output zarr layout).
        """
        table = self.trench_detector.get_trench_table()
        if len(table) == 0:
            raise RuntimeError("No trenches to preview.")

        first = table.iloc[0]
        trench_h = int(first["y_bottom"] - first["y_top"])
        trench_w = int(first["x_right"] - first["x_left"])
        n_times = self.experiment.n_timepoints
        dtype = self.experiment.data.dtype

        has_channels = self.experiment.has_channels
        channel_names = self.experiment.channel_names

        if channel is not None:
            channels = [channel]
        elif has_channels and len(channel_names) > 1:
            channels = channel_names
        else:
            channels = [None]

        grouped = table.groupby("fov", sort=False)

        def _build_channel_array(ch):
            """Build a (Trench, T, Y, X) dask array for one channel."""
            blocks = []
            for fov, fov_df in grouped:
                fov_delayed = dask.delayed(_stabilize_fov)(
                    self.registrator, ch, fov,
                )
                for _, row in fov_df.iterrows():
                    crop_delayed = dask.delayed(_crop_from_fov_data)(
                        fov_delayed,
                        int(row["y_top"]), int(row["y_bottom"]),
                        int(row["x_left"]), int(row["x_right"]),
                        bool(row["needs_flip"]),
                    )
                    blocks.append(
                        da.from_delayed(
                            crop_delayed,
                            shape=(n_times, trench_h, trench_w),
                            dtype=dtype,
                        )
                    )
            return da.stack(blocks, axis=0)  # (Trench, T, Y, X)

        if len(channels) > 1:
            # (Trench, T, Y, X) per channel → insert C dim → concat
            expanded = [
                _build_channel_array(ch)[:, :, np.newaxis, :, :]
                for ch in channels
            ]
            result_da = da.concatenate(expanded, axis=2)
            result = xr.DataArray(
                result_da,
                dims=["Trench", "T", "C", "Y", "X"],
                coords={
                    "Trench": table["trench_id"].to_numpy(dtype=np.int32),
                    "C": channel_names,
                    "Y": np.arange(trench_h),
                    "X": np.arange(trench_w),
                },
            )
        else:
            result_da = _build_channel_array(channels[0])
            result = xr.DataArray(
                result_da,
                dims=["Trench", "T", "Y", "X"],
                coords={
                    "Trench": table["trench_id"].to_numpy(dtype=np.int32),
                    "Y": np.arange(trench_h),
                    "X": np.arange(trench_w),
                },
            )

        return result

    def extract_single_trench(
        self,
        trench_id: int,
        channel: Optional[Union[str, int]] = None,
    ) -> np.ndarray:
        """Extract a single trench for preview (without writing to zarr).

        Parameters
        ----------
        trench_id : int
            Global trench ID.
        channel : str | int | None
            Channel to extract. ``None`` uses the registration channel.

        Returns
        -------
        np.ndarray
            Array of shape ``(T, trench_h, trench_w)``.
        """
        table = self.trench_detector.get_trench_table()
        row = table[table["trench_id"] == trench_id].iloc[0]

        stabilized = self.registrator.get_stabilized_data(
            channel=channel, fov=row["fov"],
        )
        crop = stabilized.isel(
            Y=slice(row["y_top"], row["y_bottom"]),
            X=slice(row["x_left"], row["x_right"]),
        )
        crop_np = crop.compute().values

        if row["needs_flip"]:
            crop_np = crop_np[:, ::-1, :]

        return crop_np

    def __repr__(self) -> str:
        table = self.trench_detector.get_trench_table()
        if len(table) == 0:
            return "Extractor (no trenches detected)"

        first = table.iloc[0]
        trench_h = int(first["y_bottom"] - first["y_top"])
        trench_w = int(first["x_right"] - first["x_left"])
        n_trenches = len(table)
        n_times = self.experiment.n_timepoints
        channel_names = self.experiment.channel_names
        n_channels = len(channel_names)
        dtype = self.experiment.data.dtype
        pixel_um = self.experiment.pixel_size_um

        shape = (n_trenches, n_times, n_channels, trench_h, trench_w)
        chunks = (1, n_times, 1, trench_h, trench_w)
        dim_str = (
            f"Trench={n_trenches} × T={n_times} × C={n_channels} "
            f"× Y={trench_h} × X={trench_w}"
        )

        size_bytes = float(np.prod(shape) * np.dtype(dtype).itemsize)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size_bytes < 1024:
                size_str = f"{size_bytes:.1f} {unit}"
                break
            size_bytes /= 1024
        else:
            size_str = f"{size_bytes:.1f} PB"

        fovs = table["fov"].unique()
        lines = [
            f"Extractor: {self.experiment.experiment_name}",
            f"  Output: {self.output_path}",
            f"  Shape: {dim_str}",
            f"  Chunks: {chunks}",
            f"  Dtype: {dtype}",
            f"  Size (uncompressed): {size_str}",
            f"  Trenches: {n_trenches} across {len(fovs)} FOVs",
            f"  Trench size: {trench_h} × {trench_w} px"
            f" ({trench_h * pixel_um:.1f} × {trench_w * pixel_um:.1f} µm)",
        ]
        lines.append(f"  Channels: {', '.join(channel_names)}")
        lines.append(f"  Timepoints: {n_times}")
        return "\n".join(lines)
