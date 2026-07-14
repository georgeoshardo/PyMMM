"""Extractor – crop trenches from stabilised data and write to output zarr."""

from __future__ import annotations

import hashlib
import json
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
from pymmm.registrator import (
    TRANSFORM_CONVENTION,
    Registrator,
    _rotation_matrix,
)
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


def _extract_frame_batch(
    nd2_path, fov_idx, time_indices, y_roi, x_roi,
    tmats, band_specs, trench_specs, image_width, out_dtype,
):
    """Read all channels once per frame and return trench-sized crops."""
    import nd2
    from skimage.transform import warp as ski_warp

    n_times = len(time_indices)
    n_trenches = len(trench_specs)
    trench_h = trench_specs[0][1] - trench_specs[0][0]
    trench_w = trench_specs[0][3] - trench_specs[0][2]
    result = None

    with nd2.ND2File(nd2_path) as f:
        coord_axes = [k for k in f.sizes if k not in {"X", "Y", "C", "S"}]
        coord_shape = tuple(f.sizes[k] for k in coord_axes)
        for batch_t, time_idx in enumerate(time_indices):
            coord_map = {"T": time_idx, "P": fov_idx, "Z": 0}
            coord_tuple = tuple(coord_map.get(ax, 0) for ax in coord_axes)
            frame_idx = int(np.ravel_multi_index(coord_tuple, coord_shape))
            frame = f.read_frame(frame_idx)
            if frame.ndim == 2:
                frame = frame[np.newaxis, ...]
            if y_roi is not None:
                frame = frame[..., slice(*y_roi), :]
            if x_roi is not None:
                frame = frame[..., slice(*x_roi)]

            if result is None:
                result = np.empty(
                    (n_times, n_trenches, frame.shape[0], trench_h, trench_w),
                    dtype=out_dtype,
                )

            for channel_idx, image in enumerate(frame):
                for y_top, y_bottom, band_trenches in band_specs:
                    offset = np.eye(3)
                    offset[1, 2] = y_top
                    band = ski_warp(
                        image,
                        tmats[batch_t] @ offset,
                        output_shape=(y_bottom - y_top, image_width),
                        preserve_range=True,
                        order=1,
                    )
                    for local_idx, x_left, x_right, needs_flip in band_trenches:
                        crop = band[:, x_left:x_right]
                        if needs_flip:
                            crop = crop[::-1]
                        result[batch_t, local_idx, channel_idx] = crop

    return result


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
                "backend": getattr(self.registrator, "backend", "stackreg"),
                "mode": str(self.registrator.mode),
                "rotation": self.registrator.rotation,
                "transform_convention": TRANSFORM_CONVENTION,
            },
            "source_subset_metadata": exp.source_subset_metadata,
        }

    def _extraction_fingerprint(
        self,
        trench_table,
        shape: tuple[int, ...],
        chunks: tuple[int, ...],
        dtype: np.dtype,
        compressor: str,
        clevel: int,
        byte_shuffle: bool,
    ) -> str:
        """Hash the source, geometry, transforms, and output layout."""
        source_stat = self.experiment.path.stat()
        trench_columns = [
            "trench_id", "fov", "lane_index", "x_left", "x_right",
            "y_top", "y_bottom", "orientation", "needs_flip",
        ]
        payload = {
            "source": str(self.experiment.path.resolve()),
            "source_size": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "subset": self.experiment.source_subset_metadata,
            "channels": self.experiment.channel_names,
            "trenches": trench_table[trench_columns].to_json(orient="records"),
            "tmats": hashlib.sha256(
                np.ascontiguousarray(self.registrator.tmats.values).tobytes()
            ).hexdigest(),
            "shape": shape,
            "chunks": chunks,
            "dtype": dtype.str,
            "rotation": self.registrator.rotation,
            "registration_backend": getattr(
                self.registrator, "backend", "stackreg"
            ),
            "transform_convention": TRANSFORM_CONVENTION,
            "compressor": compressor,
            "clevel": clevel,
            "byte_shuffle": byte_shuffle,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()

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
        clevel: int = 3,
        show_progress: bool = True,
        byte_shuffle: bool = True,
        batch_size: int = 4,
        time_chunk: int | None = None,
        resume: bool = True,
    ) -> None:
        """Extract all trenches and write to the output zarr store.

        Each task reads a short time batch and all channels with one ND2
        handle. Translation and rotation are composed into one direct warp
        for each lane band containing trenches.

        Parameters
        ----------
        compressor : str
            Compression codec name (``"zstd"``).
        clevel : int
            Compression level.
        show_progress : bool
            Show a tqdm progress bar.
        byte_shuffle : bool
            Apply byte shuffle before compression for integer image data.
        batch_size : int
            Consecutive timepoints handled by each worker task.
        time_chunk : int | None
            Output time chunk. ``None`` keeps one full trajectory per chunk.
        resume : bool
            Resume an interrupted compatible output store by completed FOV.
        """
        if int(batch_size) != batch_size or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        batch_size = int(batch_size)

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
        heights = trench_table["y_bottom"] - trench_table["y_top"]
        widths = trench_table["x_right"] - trench_table["x_left"]
        if not (heights.eq(trench_h).all() and widths.eq(trench_w).all()):
            raise ValueError("All trenches must have the same output shape.")
        n_trenches = len(trench_table)
        exp = self.experiment
        n_times = exp.n_timepoints
        if time_chunk is None:
            output_time_chunk = n_times
        else:
            if int(time_chunk) != time_chunk or time_chunk < 1:
                raise ValueError("time_chunk must be a positive integer or None")
            output_time_chunk = min(n_times, int(time_chunk))

        # Determine channels
        channel_names = exp.channel_names
        n_channels = len(channel_names)

        # Create output zarr store
        if compressor == "zstd":
            comp = numcodecs.Zstd(level=clevel)
        else:
            comp = None

        shape = (n_trenches, n_times, n_channels, trench_h, trench_w)
        chunks = (1, output_time_chunk, 1, trench_h, trench_w)

        dtype = np.dtype(exp.data.dtype)
        fingerprint = self._extraction_fingerprint(
            trench_table, shape, chunks, dtype, compressor, clevel, byte_shuffle,
        )

        if resume and self.output_path.exists():
            store = zarr.open_group(str(self.output_path), mode="r+")
            if store.attrs.get("extraction_fingerprint") != fingerprint:
                raise RuntimeError(
                    "Existing output is not compatible with this extraction. "
                    "Use resume=False or choose a new output path."
                )
            completed_fovs = set(store.attrs.get("completed_fovs", []))
        else:
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
            metadata_chunk = min(128, max(1, n_trenches))
            root_encoding = self._metadata_encoding(
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
                    "source_seq_index": (metadata_chunk, n_times),
                    "relative_time_ms": (metadata_chunk, n_times),
                    "absolute_julian_day_number": (metadata_chunk, n_times),
                    "pfs_offset": (metadata_chunk, n_times),
                    "stage_position_um": (metadata_chunk, n_times, 3),
                },
                compressor=comp,
            )
            if byte_shuffle and np.issubdtype(dtype, np.integer):
                root_encoding["data"]["filters"] = [
                    numcodecs.Shuffle(elementsize=dtype.itemsize)
                ]
            root_dataset.to_zarr(
                str(self.output_path),
                mode="w",
                consolidated=False,
                compute=False,
                zarr_format=2,
                encoding=root_encoding,
            )
            self._write_metadata_groups(
                source_metadata=source_metadata,
                compressor=comp,
                n_times=n_times,
            )
            store = zarr.open_group(str(self.output_path), mode="r+")
            store.attrs["extraction_fingerprint"] = fingerprint
            store.attrs["completed_fovs"] = []
            store.attrs["extraction_complete"] = False
            completed_fovs = set()

        data_arr = store["data"]

        # ----------------------------------------------------------
        # Frame-level extraction parameters
        # ----------------------------------------------------------
        nd2_path = str(exp.path)
        image_height = exp.data.sizes["Y"]
        image_width = exp.data.sizes["X"]
        rotation_matrix = _rotation_matrix(
            self.registrator.rotation, (image_height, image_width)
        )

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
            (exp._y_slice.start, exp._y_slice.stop, exp._y_slice.step)
            if exp._y_slice is not None else None
        )
        x_roi = (
            (exp._x_slice.start, exp._x_slice.stop, exp._x_slice.step)
            if exp._x_slice is not None else None
        )

        tmats_da = self.registrator.tmats  # (P, T, row, col) or (T, row, col)

        grouped = list(trench_table.groupby("fov", sort=False))
        all_fovs = [str(fov) for fov, _ in grouped]
        grouped = [
            (fov, fov_df) for fov, fov_df in grouped
            if str(fov) not in completed_fovs
        ]

        client = _try_get_dask_client()
        pool = None
        if client is None and grouped and n_times > batch_size:
            pool = ProcessPoolExecutor(mp_context=_mp_ctx)

        try:
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
                fov_tmats = np.matmul(fov_tmats, rotation_matrix)

                fov_df = fov_df.reset_index(drop=True)
                trench_specs = [
                    (
                        int(row["y_top"]), int(row["y_bottom"]),
                        int(row["x_left"]), int(row["x_right"]),
                        bool(row["needs_flip"]),
                    )
                    for _, row in fov_df.iterrows()
                ]
                band_map = {}
                for local_idx, (y_top, y_bottom, x_left, x_right, needs_flip) in enumerate(trench_specs):
                    band_map.setdefault((y_top, y_bottom), []).append(
                        (local_idx, x_left, x_right, needs_flip)
                    )
                band_specs = [
                    (y_top, y_bottom, band_trenches)
                    for (y_top, y_bottom), band_trenches in band_map.items()
                ]
                fov_output = np.empty(
                    (len(fov_df), n_times, n_channels, trench_h, trench_w),
                    dtype=dtype,
                )

                if client is None and pool is None:
                    batch = _extract_frame_batch(
                        nd2_path, fov_idx, original_t_indices,
                        y_roi, x_roi, fov_tmats,
                        band_specs, trench_specs, image_width, dtype,
                    )
                    fov_output[:] = batch.transpose(1, 0, 2, 3, 4)
                else:
                    future_to_start = {}
                    for start in range(0, n_times, batch_size):
                        stop = min(start + batch_size, n_times)
                        args = (
                            nd2_path, fov_idx, original_t_indices[start:stop],
                            y_roi, x_roi, fov_tmats[start:stop],
                            band_specs, trench_specs, image_width, dtype,
                        )
                        future = (
                            client.submit(_extract_frame_batch, *args)
                            if client is not None
                            else pool.submit(_extract_frame_batch, *args)
                        )
                        future_to_start[future] = start

                    if client is not None:
                        from dask.distributed import as_completed as dask_as_completed
                        completed = dask_as_completed(future_to_start)
                    else:
                        completed = as_completed(future_to_start)

                    for future in completed:
                        start = future_to_start[future]
                        batch = future.result()
                        stop = start + batch.shape[0]
                        fov_output[:, start:stop] = batch.transpose(1, 0, 2, 3, 4)

                for local_idx, (_, row) in enumerate(fov_df.iterrows()):
                    output_idx = int(row["_output_trench_index"])
                    data_arr[output_idx, :, :, :, :] = fov_output[local_idx]

                completed_fovs.add(fov_str)
                store.attrs["completed_fovs"] = [
                    name for name in all_fovs if name in completed_fovs
                ]
        finally:
            if pool is not None:
                pool.shutdown(wait=True)

        store.attrs["extraction_complete"] = set(all_fovs).issubset(completed_fovs)
        zarr.consolidate_metadata(str(self.output_path))
        print(f"Extraction complete: {self.output_path}")
        print(f"  Shape: {shape}, Chunks: {chunks}")

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def preview(self, channel: Optional[Union[str, int]] = None) -> xr.DataArray:
        """Return a lazy dask-backed DataArray previewing the extraction output.

        Uses the same direct lane-band warp as :meth:`extract`, with one lazy
        task per FOV and timepoint. Selecting a frame therefore reads one raw
        frame and produces all trenches and channels for that FOV without
        writing a zarr store.

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
        table = (
            self.trench_detector.get_trench_table()
            .sort_values("trench_id")
            .reset_index(drop=True)
        )
        if len(table) == 0:
            raise RuntimeError("No trenches to preview.")

        first = table.iloc[0]
        trench_h = int(first["y_bottom"] - first["y_top"])
        trench_w = int(first["x_right"] - first["x_left"])
        heights = table["y_bottom"] - table["y_top"]
        widths = table["x_right"] - table["x_left"]
        if not (heights.eq(trench_h).all() and widths.eq(trench_w).all()):
            raise ValueError("All trenches must have the same output shape.")

        exp = self.experiment
        channel_names = exp.channel_names
        n_channels = len(channel_names)
        dtype = np.dtype(exp.data.dtype)
        image_height = exp.data.sizes["Y"]
        image_width = exp.data.sizes["X"]
        rotation_matrix = _rotation_matrix(
            self.registrator.rotation, (image_height, image_width)
        )

        if "T" not in exp._raw_data.dims:
            original_t_indices = [0]
        elif exp._time_slice is not None:
            n_raw_t = exp._raw_data.sizes["T"]
            original_t_indices = list(range(*exp._time_slice.indices(n_raw_t)))
        else:
            original_t_indices = list(range(exp._raw_data.sizes["T"]))

        raw_fov_names = (
            [str(v) for v in exp._raw_data.coords["P"].values]
            if "P" in exp._raw_data.dims else []
        )
        y_roi = (
            (exp._y_slice.start, exp._y_slice.stop, exp._y_slice.step)
            if exp._y_slice is not None else None
        )
        x_roi = (
            (exp._x_slice.start, exp._x_slice.stop, exp._x_slice.step)
            if exp._x_slice is not None else None
        )

        fov_blocks = []
        for fov, fov_df in table.groupby("fov", sort=False):
            fov_idx = raw_fov_names.index(str(fov)) if raw_fov_names else 0
            if "P" in self.registrator.tmats.dims:
                fov_tmats = self.registrator.tmats.sel(P=fov).values
            else:
                fov_tmats = self.registrator.tmats.values
            fov_tmats = np.matmul(fov_tmats, rotation_matrix)

            fov_df = fov_df.reset_index(drop=True)
            trench_specs = [
                (
                    int(row["y_top"]), int(row["y_bottom"]),
                    int(row["x_left"]), int(row["x_right"]),
                    bool(row["needs_flip"]),
                )
                for _, row in fov_df.iterrows()
            ]
            band_map = {}
            for local_idx, spec in enumerate(trench_specs):
                y_top, y_bottom, x_left, x_right, needs_flip = spec
                band_map.setdefault((y_top, y_bottom), []).append(
                    (local_idx, x_left, x_right, needs_flip)
                )
            band_specs = [
                (y_top, y_bottom, band_trenches)
                for (y_top, y_bottom), band_trenches in band_map.items()
            ]

            time_blocks = []
            for time_idx, original_t in enumerate(original_t_indices):
                delayed = dask.delayed(_extract_frame_batch)(
                    str(exp.path), fov_idx, [original_t], y_roi, x_roi,
                    fov_tmats[time_idx:time_idx + 1], band_specs,
                    trench_specs, image_width, dtype,
                )
                block = da.from_delayed(
                    delayed,
                    shape=(1, len(fov_df), n_channels, trench_h, trench_w),
                    dtype=dtype,
                ).transpose(1, 0, 2, 3, 4)
                time_blocks.append(block)
            fov_blocks.append(da.concatenate(time_blocks, axis=1))

        result_da = da.concatenate(fov_blocks, axis=0)
        if channel is not None:
            channel_idx = (
                int(channel) if isinstance(channel, int)
                else channel_names.index(str(channel))
            )
            return xr.DataArray(
                result_da[:, :, channel_idx],
                dims=["Trench", "T", "Y", "X"],
                coords={
                    "Trench": table["trench_id"].to_numpy(dtype=np.int32),
                    "Y": np.arange(trench_h),
                    "X": np.arange(trench_w),
                },
            )

        return xr.DataArray(
            result_da,
            dims=["Trench", "T", "C", "Y", "X"],
            coords={
                "Trench": table["trench_id"].to_numpy(dtype=np.int32),
                "C": channel_names,
                "Y": np.arange(trench_h),
                "X": np.arange(trench_w),
            },
        )

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
