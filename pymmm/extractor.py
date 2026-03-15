"""Extractor – crop trenches from stabilised data and write to output zarr."""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Union

import dask
import dask.array as da
import numpy as np
import xarray as xr
import zarr
import zarr.codecs
from tqdm.auto import tqdm

from pymmm.experiment import ND2Experiment
from pymmm.metadata import build_store_metadata_attrs
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

    def _write_store_metadata(
        self,
        store: zarr.Group,
        *,
        trench_h: int,
        trench_w: int,
        n_trenches: int,
        n_times: int,
        channel_names: list[str],
        trench_mapping: list[dict],
    ) -> None:
        """Write output store attrs, including propagated ND2 metadata."""
        exp = self.experiment
        store.attrs["source_nd2"] = str(exp.path)
        store.attrs["experiment_name"] = exp.experiment_name
        store.attrs["pixel_size_um"] = exp.pixel_size_um
        store.attrs["channel_names"] = channel_names
        store.attrs["n_trenches"] = n_trenches
        store.attrs["n_times"] = n_times
        store.attrs["trench_height"] = int(trench_h)
        store.attrs["trench_width"] = int(trench_w)
        store.attrs["registration_params"] = {
            "channel": self.registrator.channel,
            "mode": str(self.registrator.mode),
            "rotation": self.registrator.rotation,
        }
        store.attrs["trench_mapping"] = trench_mapping
        store.attrs.update(build_store_metadata_attrs(exp))

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

        # Determine output shape from first trench
        first = trench_table.iloc[0]
        trench_h = int(first["y_bottom"] - first["y_top"])
        trench_w = int(first["x_right"] - first["x_left"])
        n_trenches = len(trench_table)
        exp = self.experiment
        n_times = exp.n_timepoints

        # Determine channels
        has_channels = exp.has_channels
        channel_names = exp.channel_names
        n_channels = len(channel_names) if has_channels else 1

        # Create output zarr store
        if compressor == "zstd":
            comp = [zarr.codecs.ZstdCodec(level=clevel)]
        else:
            comp = None

        if n_channels > 1:
            shape = (n_trenches, n_times, n_channels, trench_h, trench_w)
            chunks = (1, n_times, 1, trench_h, trench_w)
        else:
            shape = (n_trenches, n_times, trench_h, trench_w)
            chunks = (1, n_times, trench_h, trench_w)

        dtype = exp.data.dtype
        store = zarr.open_group(str(self.output_path), mode="w")
        codec_kwargs = {"compressors": comp} if comp else {}
        data_arr = store.create_array(
            "data",
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            **codec_kwargs,
        )

        trench_mapping = trench_table.to_dict(orient="records")
        self._write_store_metadata(
            store,
            trench_h=trench_h,
            trench_w=trench_w,
            n_trenches=n_trenches,
            n_times=n_times,
            channel_names=channel_names,
            trench_mapping=trench_mapping,
        )

        # ----------------------------------------------------------
        # Frame-level extraction parameters
        # ----------------------------------------------------------
        nd2_path = str(exp.path)
        rotation = self.registrator.rotation
        out_dtype = np.dtype(dtype)
        H, W = exp.data.sizes["Y"], exp.data.sizes["X"]

        # Map subsetted T indices → original file T indices
        if exp._time_slice is not None:
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
            grouped = trench_table.groupby("fov")
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

                if n_channels > 1:
                    for c_idx, ch in enumerate(channel_names):
                        ch_idx = exp.channel_names.index(ch)
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
                            data_arr[row["trench_id"], :, c_idx, :, :] = crop

                        del fov_data
                else:
                    ch_idx = (
                        exp.channel_names.index(self.registrator.channel)
                        if has_channels else None
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
                        data_arr[row["trench_id"], :, :, :] = crop

                    del fov_data
        finally:
            if pool is not None:
                pool.shutdown(wait=True)

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

        grouped = table.groupby("fov")

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
                    "Trench": np.arange(len(table)),
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
                    "Trench": np.arange(len(table)),
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
        has_channels = self.experiment.has_channels
        channel_names = self.experiment.channel_names
        n_channels = len(channel_names) if has_channels else 1
        dtype = self.experiment.data.dtype
        pixel_um = self.experiment.pixel_size_um

        if n_channels > 1:
            shape = (n_trenches, n_times, n_channels, trench_h, trench_w)
            chunks = (1, n_times, 1, trench_h, trench_w)
            dim_str = (
                f"Trench={n_trenches} × T={n_times} × C={n_channels} "
                f"× Y={trench_h} × X={trench_w}"
            )
        else:
            shape = (n_trenches, n_times, trench_h, trench_w)
            chunks = (1, n_times, trench_h, trench_w)
            dim_str = (
                f"Trench={n_trenches} × T={n_times} "
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
        if n_channels > 1:
            lines.append(f"  Channels: {', '.join(channel_names)}")
        lines.append(f"  Timepoints: {n_times}")
        return "\n".join(lines)
