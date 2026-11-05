"""Extractor – crop trenches from stabilised data, one OME-Zarr image per trench.

The output is one zarr group per experiment::

    <experiment>.zarr/          plain group; attrs["pymmm"]: what was extracted from
                                which ND2, and how far the extraction got
        T  C                    original ND2 frame index, channel names
        trenches/               the trench table: one 1-D array per column, dimension
                                "trench"; column "path" names the trench's image
        acquisition/ detection/ events/ registration/ source_frames/
                                experiment metadata (xarray datasets)
        trench_0000/            OME-Zarr 0.5 image: multiscales (t, c, y, x) + omero;
            0                   attrs["pymmm"] is the trench's row of the table
        trench_0001/  ...

Each trench is an image of its own, so OME-Zarr readers open it as it is, and what
later steps make of it (segmentations in ``labels/``, GEFF tracks) sits beside its
pixels. A trench's ``0`` array is one shard file with an inner chunk per channel
(and time chunk), so reading one channel reads only that channel. Chunk keys use
"." (``0/c.0.0.0.0``), so a chunk is one file rather than nested directories. The
experiment as one (trench, t, c, y, x) array is a lazy stack of the trench arrays.
"""

from __future__ import annotations

import json
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional, Union

import dask
import dask.array as da
import numpy as np
import xarray as xr
import zarr
from tqdm.auto import tqdm

from pymmm.experiment import ND2Experiment
from pymmm.metadata import (
    build_acquisition_dataset,
    build_events_dataset,
    build_source_frames_dataset,
    extract_source_frame_metadata,
)
from pymmm.provenance import SCHEMA_VERSION, provenance
from pymmm.registrator import (
    TRANSFORM_CONVENTION,
    Registrator,
    _rotation_matrix,
)
from pymmm.trench_detector import TrenchDetector
from pymmm.trench_images import CHUNK_KEYS as _CHUNK_KEYS
from pymmm.trench_images import OME_VERSION, trench_path

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


def _json(value):
    return json.loads(json.dumps(value, default=str))


_TABLE_ATTRS = {
    "path": {"long_name": "the trench's OME-Zarr image, relative to the experiment group"},
    "fov_name": {"long_name": "field of view the trench was cut from"},
    "fov_index": {"long_name": "index into the FOV dimension of /source_frames and /registration"},
    "lane_index": {"long_name": "lane the trench belongs to within its FOV"},
    "x_left": {"units": "pixel", "long_name": "trench left edge in the registered FOV"},
    "x_right": {"units": "pixel", "long_name": "trench right edge in the registered FOV"},
    "y_top": {"units": "pixel", "long_name": "trench top edge in the registered FOV"},
    "y_bottom": {"units": "pixel", "long_name": "trench bottom edge in the registered FOV"},
    "orientation": {"long_name": "trench opening direction (1 = opens downward, -1 = upward)"},
    "needs_flip": {"long_name": "crop was flipped vertically so trenches share one orientation"},
    "trench": {"long_name": "trench id"},
}
_ROOT_ATTRS = {
    "T": {"long_name": "frame index (original ND2 T index) of each t of the trench images; "
                       "times are in /source_frames"},
    "C": {"long_name": "channel of each c of the trench images"},
}


def _annotate(dataset: xr.Dataset, attrs: dict[str, dict[str, str]]) -> xr.Dataset:
    for name, values in attrs.items():
        dataset[name].attrs.update(values)
    return dataset


class Extractor:
    """Crop trenches from the lazy stabilised DataArray, one OME-Zarr image each.

    Parameters
    ----------
    experiment : ND2Experiment
        Source experiment.
    registrator : Registrator
        Completed registrator.
    trench_detector : TrenchDetector
        Completed trench detector.
    output_path : str | Path | None
        Path for the experiment group (see the module docstring). Defaults to
        ``<nd2_dir>/<experiment_name>.zarr``.
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
            output_path = nd2_dir / f"{experiment.experiment_name}.zarr"
        self.output_path = Path(output_path)

    def _source_attrs(self) -> dict[str, Any]:
        """What was extracted: the ND2 and the subset of it."""
        exp = self.experiment
        source_stat = exp.path.stat()
        return {
            "source_nd2": str(exp.path),
            "source_file_size_bytes": int(source_stat.st_size),
            "source_mtime_ns": int(source_stat.st_mtime_ns),
            "experiment_name": exp.experiment_name,
            "source_subset_metadata": exp.source_subset_metadata,
        }

    def _experiment_attrs(
        self, *, channel_names: list[str], pixel_size_um: float, frame_interval_ms: float,
    ) -> dict[str, Any]:
        """The experiment group's ``pymmm`` attributes."""
        return _json({
            "schema_version": SCHEMA_VERSION,
            "layout": "experiment",
            "description": (
                "A mother-machine experiment: one OME-Zarr image per trench (column "
                "'path' of the trench table trenches/), and the experiment metadata."
            ),
            "image_path": "trench_{trench:04d}",
            "provenance": provenance(),
            **self._source_attrs(),
            "channels": list(channel_names),
            "pixel_size_um": pixel_size_um,
            "frame_interval_ms": frame_interval_ms,
        })

    def _image_attrs(
        self,
        row,
        *,
        channel_names: list[str],
        colors: list[str],
        pixel_size_um: float,
        frame_interval_ms: float,
        dtype: np.dtype,
        fov_index: int,
        windows: list[tuple[float, float]] | None = None,
    ) -> dict[str, Any]:
        """A trench image's attributes: OME multiscales and omero, and its table row."""
        trench_id = int(row["trench_id"])
        time_axis = {"name": "t", "type": "time"}
        space = [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]
        scale = [1.0, 1.0, 1.0, 1.0]
        if np.isfinite(frame_interval_ms) and frame_interval_ms > 0:
            time_axis["unit"] = "minute"
            scale[0] = frame_interval_ms / 60_000.0
        if np.isfinite(pixel_size_um) and pixel_size_um > 0:
            for axis in space:
                axis["unit"] = "micrometer"
            scale[2:] = [pixel_size_um, pixel_size_um]
        top = float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 1.0
        channels = [
            {
                "label": label,
                "color": color,
                "active": label == self.registrator.channel,
                "window": {
                    "min": 0.0, "max": top,
                    "start": windows[c][0] if windows else 0.0,
                    "end": windows[c][1] if windows else top,
                },
            }
            for c, (label, color) in enumerate(zip(channel_names, colors, strict=True))
        ]
        return _json({
            "ome": {
                "version": OME_VERSION,
                "multiscales": [{
                    "name": trench_path(trench_id),
                    "type": "none",
                    "metadata": {"description": "one resolution level: nothing was downsampled"},
                    "axes": [time_axis, {"name": "c", "type": "channel"}, *space],
                    "datasets": [{
                        "path": "0",
                        "coordinateTransformations": [{"type": "scale", "scale": scale}],
                    }],
                }],
                "omero": {"channels": channels, "rdefs": {"model": "color", "defaultT": 0}},
            },
            "pymmm": {
                "trench": trench_id,
                "fov_name": str(row["fov"]),
                "fov_index": int(fov_index),
                **{
                    key: row[key].item() if hasattr(row[key], "item") else row[key]
                    for key in ("lane_index", "x_left", "x_right", "y_top", "y_bottom",
                                "orientation", "needs_flip")
                },
                "units": {key: "pixel" for key in ("x_left", "x_right", "y_top", "y_bottom")},
                "experiment": "..",
            },
        })

    @staticmethod
    def _windows(image: np.ndarray) -> list[tuple[float, float]]:
        """Display window per channel of a (T, C, Y, X) trench: its 0.5th and 99.9th
        percentiles, from every tenth frame or so."""
        step = max(1, image.shape[0] // 26)
        return [
            tuple(float(v) for v in np.percentile(image[::step, c], [0.5, 99.9]))
            for c in range(image.shape[1])
        ]

    def _is_experiment_group(self) -> bool:
        try:
            group = zarr.open_group(str(self.output_path), mode="r", zarr_format=3)
        except (FileNotFoundError, ValueError, zarr.errors.GroupNotFoundError):
            return False
        return (dict(group.attrs).get("pymmm") or {}).get("layout") == "experiment"

    def _resume_mismatches(
        self,
        store: zarr.Group,
        trench_table,
        shape: tuple[int, ...],
        chunks: tuple[int, ...],
        shards: tuple[int, ...],
        dtype: np.dtype,
        compressor: str,
        clevel: int,
        byte_shuffle: bool,
    ) -> list[str]:
        """Compare this extraction with what an existing store already records.

        Returns a description of every input that differs; empty means the
        store can be resumed.
        """
        attrs = dict(store.attrs).get("pymmm") or {}
        if attrs.get("layout") != "experiment" or attrs.get("schema_version") != SCHEMA_VERSION:
            return [
                f"layout {attrs.get('layout')!r}, schema version {attrs.get('schema_version')} "
                f"(current 'experiment', {SCHEMA_VERSION})"
            ]

        current_attrs = _json(self._source_attrs())
        mismatches = [
            key for key in (
                "source_nd2", "source_file_size_bytes", "source_mtime_ns",
                "source_subset_metadata",
            )
            if attrs.get(key) != current_attrs[key]
        ]

        if store["C"][:].tolist() != list(self.experiment.channel_names):
            mismatches.append("channels")

        table = store["trenches"]
        trench_columns = {
            "trench": "trench_id", "fov_name": "fov", "lane_index": "lane_index",
            "x_left": "x_left", "x_right": "x_right", "y_top": "y_top",
            "y_bottom": "y_bottom", "orientation": "orientation",
            "needs_flip": "needs_flip",
        }
        stored_trench_count = table["trench"].shape[0]
        if stored_trench_count != len(trench_table):
            mismatches.append(
                f"trench count ({stored_trench_count} stored, {len(trench_table)} now)"
            )
            return mismatches
        for var, column in trench_columns.items():
            if table[var][:].tolist() != trench_table[column].tolist():
                mismatches.append(f"trench {column}")

        data = store[f"{trench_path(trench_table['trench_id'].iloc[0])}/0"]
        if data.shape != tuple(shape):
            mismatches.append(f"image shape ({data.shape} stored, {tuple(shape)} now)")
        if data.chunks != tuple(chunks) or data.shards != tuple(shards):
            mismatches.append(
                f"image chunks/shards ({data.chunks}/{data.shards} stored, "
                f"{tuple(chunks)}/{tuple(shards)} now)"
            )
        if np.dtype(data.dtype) != dtype:
            mismatches.append(f"image dtype ({data.dtype} stored, {dtype} now)")
        blosc = [c for c in data.compressors if isinstance(c, zarr.codecs.BloscCodec)]
        if compressor == "zstd":
            want_shuffle = "shuffle" if byte_shuffle else "noshuffle"
            if not blosc or (
                str(getattr(blosc[0].cname, "value", blosc[0].cname)),
                blosc[0].clevel,
                str(getattr(blosc[0].shuffle, "value", blosc[0].shuffle)),
            ) != ("zstd", clevel, want_shuffle):
                mismatches.append("compression settings")
        elif blosc:
            mismatches.append("compression settings")

        stored_reg = store["registration"]
        current_reg = self._build_registration_dataset(
            self.experiment.source_axes
        )
        for key in ("rotation", "backend", "transform_convention"):
            if stored_reg.attrs.get(key) != _json(current_reg.attrs[key]):
                mismatches.append(f"registration {key}")
        if not np.array_equal(stored_reg["tmats"][:], current_reg["tmats"].values):
            mismatches.append("registration transforms")
        return mismatches

    def _build_registration_dataset(
        self, source_metadata: dict[str, Any],
    ) -> xr.Dataset:
        """Build the per-FOV ``/registration`` dataset of transforms used for extraction."""
        reg = self.registrator
        tmats_da = reg.tmats
        fov_names = [str(f) for f in source_metadata["fov_names"]]
        if "P" in tmats_da.dims:
            tmats = np.stack([
                tmats_da.sel(P=fov).transpose("T", "row", "col").values
                for fov in fov_names
            ])
        else:
            tmats = tmats_da.transpose("T", "row", "col").values[np.newaxis]
        tmats = np.asarray(tmats, dtype=np.float64)
        shift = np.stack([tmats[..., 1, 2], tmats[..., 0, 2]], axis=-1)

        return xr.Dataset(
            data_vars={
                "tmats": (
                    ("FOV", "T", "Row", "Col"), tmats,
                    {"long_name": "registration transform, before rotation (3x3 affine)"},
                ),
                "shift": (
                    ("FOV", "T", "ShiftAxis"), shift,
                    {"units": "pixel", "long_name": "registration translation"},
                ),
            },
            coords={
                "FOV": np.asarray(fov_names, dtype=object),
                "T": source_metadata["original_t_index"],
                "Row": [0, 1, 2],
                "Col": [0, 1, 2],
                "ShiftAxis": np.asarray(["y", "x"], dtype=object),
            },
            attrs={
                "layout": "registration",
                "description": (
                    "Registration transforms per FOV and timepoint, before rotation. "
                    "shift is the translation part of tmats (y = [1, 2], x = [0, 2])."
                ),
                "channel": reg.channel,
                "backend": getattr(reg, "backend", "stackreg"),
                "mode": str(reg.mode),
                "rotation": reg.rotation,
                "roi": json.dumps(getattr(reg, "roi", None) or {}, default=str),
                "mean_n_frames": getattr(reg, "mean_n_frames", None),
                "mean_from": getattr(reg, "mean_from", None),
                "downsample": getattr(reg, "downsample", None),
                "transform_convention": TRANSFORM_CONVENTION,
            },
        )

    @staticmethod
    def _metadata_encoding(
        dataset: xr.Dataset,
        *,
        chunk_overrides: dict[str, tuple[int, ...]] | None = None,
        compressors: list[Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Return xarray ``to_zarr`` encodings for the dataset variables."""
        chunk_overrides = chunk_overrides or {}
        encoding: dict[str, dict[str, Any]] = {}
        for name in dataset.data_vars:
            spec: dict[str, Any] = {}
            if name in chunk_overrides:
                spec["chunks"] = chunk_overrides[name]
            if compressors is not None:
                spec["compressors"] = compressors
            if spec:
                encoding[name] = spec
        return encoding

    @staticmethod
    def _build_root_dataset(
        *, source_metadata: dict[str, Any], channel_names: list[str], attrs: dict[str, Any],
    ) -> xr.Dataset:
        """The experiment group itself: the t and c coordinates of the trench images."""
        return xr.Dataset(
            coords={
                "T": source_metadata["original_t_index"],
                "C": np.asarray(channel_names, dtype=object),
            },
            attrs={"pymmm": attrs},
        ).pipe(_annotate, _ROOT_ATTRS)

    @staticmethod
    def _build_trench_table(trench_table, source_metadata: dict[str, Any]) -> xr.Dataset:
        """The ``/trenches`` table: one row per trench image."""
        fov_lookup = {
            str(fov_name): idx
            for idx, fov_name in enumerate(source_metadata["fov_names"])
        }
        ids = trench_table["trench_id"].to_numpy(dtype=np.int32)
        columns = {
            "path": np.asarray([trench_path(t) for t in ids], dtype=object),
            "fov_name": np.asarray(trench_table["fov"].astype(str).tolist(), dtype=object),
            "fov_index": np.asarray(
                [fov_lookup[str(fov)] for fov in trench_table["fov"].tolist()], dtype=np.int32,
            ),
            "lane_index": trench_table["lane_index"].to_numpy(dtype=np.int32),
            "x_left": trench_table["x_left"].to_numpy(dtype=np.int32),
            "x_right": trench_table["x_right"].to_numpy(dtype=np.int32),
            "y_top": trench_table["y_top"].to_numpy(dtype=np.int32),
            "y_bottom": trench_table["y_bottom"].to_numpy(dtype=np.int32),
            "orientation": trench_table["orientation"].to_numpy(dtype=np.int8),
            "needs_flip": trench_table["needs_flip"].to_numpy(dtype=bool),
        }
        return xr.Dataset(
            data_vars={name: (("trench",), values) for name, values in columns.items()},
            coords={"trench": ids},
            attrs={"layout": "trench_table",
                   "description": "One row per trench; 'path' is its OME-Zarr image."},
        ).pipe(_annotate, _TABLE_ATTRS)

    def _write_metadata_groups(
        self,
        *,
        source_metadata: dict[str, Any],
        acquisition: xr.Dataset,
        compressors: list[Any],
        n_times: int,
    ) -> None:
        """Write the experiment metadata groups (xarray datasets)."""
        source_frames = build_source_frames_dataset(source_metadata)
        source_frames.to_zarr(
            str(self.output_path),
            mode="a",
            group="source_frames",
            consolidated=False,
            zarr_format=3,
            encoding=self._metadata_encoding(
                source_frames,
                chunk_overrides={
                    "original_p_index": (
                        min(256, max(1, source_frames.sizes["FOV"])),
                    ),
                    "source_seq_index": (1, n_times),
                    "relative_time": (1, n_times),
                    "julian_date": (1, n_times),
                    "pfs_offset": (1, n_times),
                    "pfs_status": (1, n_times),
                    "stage_position": (1, n_times, 3),
                },
                compressors=compressors,
            ),
        )

        events = build_events_dataset(
            source_metadata["acquisition_event_records"],
        )
        event_chunks: dict[str, tuple[int, ...]] = {}
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
            zarr_format=3,
            encoding=self._metadata_encoding(
                events,
                chunk_overrides=event_chunks,
                compressors=compressors,
            ),
        )

        registration = self._build_registration_dataset(source_metadata)
        registration.to_zarr(
            str(self.output_path),
            mode="a",
            group="registration",
            consolidated=False,
            zarr_format=3,
            encoding=self._metadata_encoding(
                registration, compressors=compressors,
            ),
        )

        acquisition.to_zarr(
            str(self.output_path),
            mode="a",
            group="acquisition",
            consolidated=False,
            zarr_format=3,
            encoding=self._metadata_encoding(
                acquisition, compressors=compressors,
            ),
        )

        self._build_detection_dataset().to_zarr(
            str(self.output_path),
            mode="a",
            group="detection",
            consolidated=False,
            zarr_format=3,
        )

    def _build_detection_dataset(self) -> xr.Dataset:
        """Build the ``/detection`` group: lane and trench detection settings."""
        def _settings(detector) -> dict[str, Any] | None:
            if detector is None or not hasattr(detector, "_detection_params"):
                return None
            return json.loads(json.dumps(
                {"channel": detector.channel, "params": detector._detection_params},
                default=str,
            ))

        attrs = {
            "layout": "detection",
            "description": "Settings that produced the trench geometry in the root dataset.",
            "lane_detection": _settings(getattr(self.trench_detector, "lane_detector", None)),
            "trench_detection": _settings(self.trench_detector),
        }
        return xr.Dataset(attrs={k: v for k, v in attrs.items() if v is not None})

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
        """Extract all trenches, each into its own OME-Zarr image (module docstring).

        Each task reads a short time batch and all channels with one ND2
        handle. Translation and rotation are composed into one direct warp
        for each lane band containing trenches. A FOV's trenches are written
        when all its frames are in, each trench's image in one go.

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
            Frames per inner chunk of a trench's shard. ``None`` keeps the whole
            trajectory of a channel in one chunk.
        resume : bool
            Resume an interrupted compatible output store by completed FOV.
        """
        if int(batch_size) != batch_size or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        batch_size = int(batch_size)

        trench_table = self.trench_detector.get_trench_table()
        if len(trench_table) == 0:
            raise RuntimeError("No trenches to extract.")
        trench_table = trench_table.sort_values("trench_id").reset_index(drop=True)

        # Determine output shape from first trench
        first = trench_table.iloc[0]
        trench_h = int(first["y_bottom"] - first["y_top"])
        trench_w = int(first["x_right"] - first["x_left"])
        heights = trench_table["y_bottom"] - trench_table["y_top"]
        widths = trench_table["x_right"] - trench_table["x_left"]
        if not (heights.eq(trench_h).all() and widths.eq(trench_w).all()):
            raise ValueError("All trenches must have the same output shape.")
        exp = self.experiment
        n_times = exp.n_timepoints
        if time_chunk is None:
            output_time_chunk = n_times
        else:
            if int(time_chunk) != time_chunk or time_chunk < 1:
                raise ValueError("time_chunk must be a positive integer or None")
            output_time_chunk = min(n_times, int(time_chunk))

        # Determine channels
        channel_names = list(exp.channel_names)
        n_channels = len(channel_names)

        # Create output zarr store
        if compressor == "zstd":
            metadata_compressors = [
                zarr.codecs.BloscCodec(
                    cname="zstd", clevel=clevel, shuffle="noshuffle",
                )
            ]
            data_compressors = [
                zarr.codecs.BloscCodec(
                    cname="zstd",
                    clevel=clevel,
                    shuffle="shuffle" if byte_shuffle else "noshuffle",
                )
            ]
        else:
            metadata_compressors = []
            data_compressors = []

        # one trench image: (T, C, Y, X); one shard holding a chunk per channel
        # (and time chunk)
        shape = (n_times, n_channels, trench_h, trench_w)
        chunks = (output_time_chunk, 1, trench_h, trench_w)
        shards = (
            math.ceil(n_times / output_time_chunk) * output_time_chunk,
            n_channels, trench_h, trench_w,
        )

        dtype = np.dtype(exp.data.dtype)

        if resume and self.output_path.exists():
            store = zarr.open_group(
                str(self.output_path), mode="r+", zarr_format=3,
            )
            mismatches = self._resume_mismatches(
                store, trench_table, shape, chunks, shards, dtype,
                compressor, clevel, byte_shuffle,
            )
            if mismatches:
                raise RuntimeError(
                    "Existing output is not compatible with this extraction; "
                    f"these inputs differ: {', '.join(mismatches)}. "
                    "Use resume=False or choose a new output path."
                )
            completed_fovs = set(dict(store.attrs)["pymmm"].get("completed_fovs", []))
        else:
            if self.output_path.exists() and not self._is_experiment_group():
                raise RuntimeError(
                    f"{self.output_path} exists and is not an experiment group written by "
                    "the Extractor; it is never overwritten. Choose a new output path."
                )
            source_metadata = extract_source_frame_metadata(exp)
            acquisition = build_acquisition_dataset(
                exp.source_acquisition_metadata, channel_names=channel_names,
            )
            frame_interval_ms = float(acquisition["time_interval"])
            pixel_size_um = float(exp.pixel_size_um)
            if "C" in acquisition["color"].dims:
                colors = [str(c).lstrip("#").upper() for c in acquisition["color"].values]
            else:  # the ND2's channel records don't match its channels
                colors = ["FFFFFF"] * n_channels
            experiment_attrs = self._experiment_attrs(
                channel_names=channel_names, pixel_size_um=pixel_size_um,
                frame_interval_ms=frame_interval_ms,
            )
            self._build_root_dataset(
                source_metadata=source_metadata, channel_names=channel_names,
                attrs=experiment_attrs,
            ).to_zarr(str(self.output_path), mode="w", consolidated=False, zarr_format=3)
            table = self._build_trench_table(trench_table, source_metadata)
            table.to_zarr(
                str(self.output_path),
                mode="a",
                group="trenches",
                consolidated=False,
                zarr_format=3,
                encoding=self._metadata_encoding(
                    table, compressors=metadata_compressors,
                ),
            )
            self._write_metadata_groups(
                source_metadata=source_metadata,
                acquisition=acquisition,
                compressors=metadata_compressors,
                n_times=n_times,
            )
            store = zarr.open_group(
                str(self.output_path), mode="r+", zarr_format=3,
            )
            fov_index = dict(zip(table["trench"].values.tolist(),
                                 table["fov_index"].values.tolist()))
            for _, row in trench_table.iterrows():
                image = store.create_group(
                    trench_path(row["trench_id"]),
                    attributes=self._image_attrs(
                        row, channel_names=channel_names, colors=colors,
                        pixel_size_um=pixel_size_um, frame_interval_ms=frame_interval_ms,
                        dtype=dtype, fov_index=fov_index[int(row["trench_id"])],
                    ),
                )
                image.create_array(
                    "0", shape=shape, dtype=dtype, chunks=chunks, shards=shards,
                    compressors=data_compressors, chunk_key_encoding=_CHUNK_KEYS,
                    dimension_names=["t", "c", "y", "x"], fill_value=0,
                )
            store.attrs["pymmm"] = {
                **experiment_attrs, "completed_fovs": [], "extraction_complete": False,
            }
            completed_fovs = set()

        # ----------------------------------------------------------
        # Frame-level extraction parameters
        # ----------------------------------------------------------
        nd2_path = str(exp.path)
        image_height = exp.data.sizes["Y"]
        image_width = exp.data.sizes["X"]
        rotation_matrix = _rotation_matrix(
            self.registrator.rotation, (image_height, image_width)
        )

        original_t_indices = exp.source_axes["original_t_index"].tolist()
        y_roi, x_roi = exp.source_roi

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
                fov_idx = exp.raw_fov_index(fov_str)

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
                    image = store[trench_path(row["trench_id"])]
                    image["0"][...] = fov_output[local_idx]
                    ome = dict(image.attrs)["ome"]
                    windows = self._windows(fov_output[local_idx])
                    for channel, (start, end) in zip(ome["omero"]["channels"], windows):
                        channel["window"].update(start=start, end=end)
                    image.attrs["ome"] = ome

                completed_fovs.add(fov_str)
                store.attrs["pymmm"] = {
                    **dict(store.attrs)["pymmm"],
                    "completed_fovs": [name for name in all_fovs if name in completed_fovs],
                }
        finally:
            if pool is not None:
                pool.shutdown(wait=True)

        store.attrs["pymmm"] = {
            **dict(store.attrs)["pymmm"],
            "extraction_complete": set(all_fovs).issubset(completed_fovs),
        }
        print(f"Extraction complete: {self.output_path}")
        print(f"  {len(trench_table)} trench images of {shape}, chunks {chunks}, shards {shards}")

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

        original_t_indices = exp.source_axes["original_t_index"].tolist()
        y_roi, x_roi = exp.source_roi

        fov_blocks = []
        for fov, fov_df in table.groupby("fov", sort=False):
            fov_idx = exp.raw_fov_index(fov)
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
        dim_str = (
            f"{n_trenches} trench images of T={n_times} × C={n_channels} "
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
            f"  Layout: one OME-Zarr image per trench (trench_NNNN), one shard file each",
            f"  Dtype: {dtype}",
            f"  Size (uncompressed): {size_str}",
            f"  Trenches: {n_trenches} across {len(fovs)} FOVs",
            f"  Trench size: {trench_h} × {trench_w} px"
            f" ({trench_h * pixel_um:.1f} × {trench_w * pixel_um:.1f} µm)",
        ]
        lines.append(f"  Channels: {', '.join(channel_names)}")
        lines.append(f"  Timepoints: {n_times}")
        return "\n".join(lines)
