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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import dask
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from tqdm.auto import tqdm

from pymmm._utils import normalize_channel_arg
from pymmm.parallel import Runner
from pymmm.source.metadata import (
    build_acquisition_dataset,
    build_events_dataset,
    build_source_frames_dataset,
    extract_source_frame_metadata,
)
from pymmm.source.nd2 import ND2Experiment
from pymmm.steps.registration import (
    TRANSFORM_CONVENTION,
    Registrator,
    _rotation_matrix,
)
from pymmm.steps.trenches import TrenchDetector
from pymmm.store import ome
from pymmm.store.experiment_group import CHUNK_KEYS as _CHUNK_KEYS
from pymmm.store.experiment_group import trench_path
from pymmm.store.provenance import SCHEMA_VERSION, plain_json, provenance

# ======================================================================
# Module-level helpers (picklable for client.submit / ProcessPoolExecutor)
# ======================================================================


def _crop_frames(
    nd2_path, fov_idx, time_indices, y_roi, x_roi,
    tmats, band_specs, trench_specs, image_width, out_dtype, channels=None,
):
    """Read frames (all channels, or ``channels``) with one ND2 handle and cut their
    trenches: ``(t, trench, c, y, x)``.

    Each lane band is one warp of the raw frame (``tmats``: translation then rotation,
    offset to the band's top row); its trenches are column ranges of it, flipped when
    they open upward.
    """
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
            if channels is not None:
                frame = frame[list(channels)]
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


@dataclass(frozen=True)
class FovCrops:
    """Everything needed to cut one FOV's trenches out of its frames (one
    :func:`_crop_frames` call per batch of frames, anywhere)."""

    nd2_path: str
    fov: str
    fov_index: int                      # its P index in the ND2
    time_indices: Tuple[int, ...]       # ND2 T index of each frame
    roi: Tuple[Any, Any]                # the experiment's (y, x) ROI
    image_width: int
    dtype: np.dtype
    tmats: np.ndarray                   # (T, 3, 3): registration, then rotation
    trench_ids: Tuple[int, ...]         # in table order
    trench_specs: Tuple[tuple, ...]     # (y_top, y_bottom, x_left, x_right, needs_flip)
    band_specs: Tuple[tuple, ...]       # (y_top, y_bottom, ((local index, x_left, x_right, needs_flip), ...))

    def batch(self, start: int, stop: int, channels: Optional[Sequence[int]] = None) -> tuple:
        """The :func:`_crop_frames` arguments for frames ``start:stop``."""
        return (self.nd2_path, self.fov_index, list(self.time_indices[start:stop]), *self.roi,
                self.tmats[start:stop], self.band_specs, self.trench_specs, self.image_width,
                self.dtype, None if channels is None else tuple(channels))


@dataclass(frozen=True)
class _ImageLayout:
    """A trench image's array: (t, c, y, x) shape, inner chunks, shard, dtype."""

    shape: Tuple[int, int, int, int]
    chunks: Tuple[int, int, int, int]
    shards: Tuple[int, int, int, int]
    dtype: np.dtype


def _codecs(compressor: str, clevel: int, byte_shuffle: bool) -> Tuple[list, list]:
    """The metadata's and the pixels' compressors."""
    if compressor != "zstd":
        return [], []
    return (
        [zarr.codecs.BloscCodec(cname="zstd", clevel=clevel, shuffle="noshuffle")],
        [zarr.codecs.BloscCodec(cname="zstd", clevel=clevel,
                                shuffle="shuffle" if byte_shuffle else "noshuffle")],
    )


def _cut(runner: Runner, crops: FovCrops, batch_size: int) -> np.ndarray:
    """A FOV's trenches, (trench, t, c, y, x), cut in batches of frames."""
    n_times = len(crops.time_indices)
    futures = {
        runner.submit(_crop_frames, *crops.batch(start, min(start + batch_size, n_times))): start
        for start in range(0, n_times, batch_size)
    }
    pixels = None
    for future in runner.as_completed(futures):
        start = futures[future]
        batch = future.result()
        if pixels is None:
            pixels = np.empty((batch.shape[1], n_times) + batch.shape[2:], dtype=batch.dtype)
        pixels[:, start:start + batch.shape[0]] = batch.transpose(1, 0, 2, 3, 4)
    return pixels


def _update_attrs(store: zarr.Group, **values: Any) -> None:
    """Set keys of the experiment group's ``pymmm`` attributes."""
    store.attrs["pymmm"] = {**dict(store.attrs)["pymmm"], **values}


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
        return plain_json({
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
    ) -> dict[str, Any]:
        """A trench image's attributes: OME multiscales and omero, and its table row."""
        trench_id = int(row["trench_id"])
        return plain_json({
            "ome": ome.image(
                trench_path(trench_id), channels=channel_names, colors=colors,
                active=self.registrator.channel, dtype=dtype,
                frame_interval_ms=frame_interval_ms, pixel_size_um=pixel_size_um,
            ),
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

        current_attrs = plain_json(self._source_attrs())
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
            if stored_reg.attrs.get(key) != plain_json(current_reg.attrs[key]):
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
            if detector is None or not hasattr(detector, "detection_params"):
                return None
            return plain_json(
                {"channel": detector.channel, "params": detector.detection_params})

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
        table = self._trench_table("extract")
        layout = self._layout(table, time_chunk)
        if resume and self.output_path.exists():
            store = self._reopen(table, layout, compressor, clevel, byte_shuffle)
        else:
            store = self._create(table, layout, _codecs(compressor, clevel, byte_shuffle))

        crops = self._crops(table)
        all_fovs = [c.fov for c in crops]
        completed = set(dict(store.attrs)["pymmm"].get("completed_fovs", []))
        todo = [c for c in crops if c.fov not in completed]
        # A FOV's frames in one batch run here; more batches run on the Dask
        # cluster or in a process pool.
        with Runner(workers=1 if layout.shape[0] <= batch_size else None) as runner:
            for fov_crops in tqdm(todo, desc="Extracting FOVs", total=len(todo),
                                  disable=not show_progress):
                self._write_fov(store, fov_crops, _cut(runner, fov_crops, batch_size))
                completed.add(fov_crops.fov)
                _update_attrs(store, completed_fovs=[f for f in all_fovs if f in completed])

        _update_attrs(store, extraction_complete=set(all_fovs).issubset(completed))
        print(f"Extraction complete: {self.output_path}")
        print(f"  {len(table)} trench images of {layout.shape}, chunks {layout.chunks}, "
              f"shards {layout.shards}")

    # ------------------------------------------------------------------
    # The steps of an extraction
    # ------------------------------------------------------------------

    def _trench_table(self, action: str) -> pd.DataFrame:
        """The trench table in trench-id order; every trench is the same size."""
        table = self.trench_detector.get_trench_table()
        if len(table) == 0:
            raise RuntimeError(f"No trenches to {action}.")
        table = table.sort_values("trench_id").reset_index(drop=True)
        heights = table["y_bottom"] - table["y_top"]
        widths = table["x_right"] - table["x_left"]
        if heights.nunique() != 1 or widths.nunique() != 1:
            raise ValueError("All trenches must have the same output shape.")
        return table

    def _layout(self, table: pd.DataFrame, time_chunk: int | None) -> _ImageLayout:
        """Each trench image's shape, inner chunks, shard and dtype."""
        exp = self.experiment
        n_times = exp.n_timepoints
        if time_chunk is None:
            output_time_chunk = n_times
        else:
            if int(time_chunk) != time_chunk or time_chunk < 1:
                raise ValueError("time_chunk must be a positive integer or None")
            output_time_chunk = min(n_times, int(time_chunk))
        n_channels = len(exp.channel_names)
        first = table.iloc[0]
        trench_h = int(first["y_bottom"] - first["y_top"])
        trench_w = int(first["x_right"] - first["x_left"])
        # (t, c, y, x) in one shard holding a chunk per channel (and time chunk)
        return _ImageLayout(
            shape=(n_times, n_channels, trench_h, trench_w),
            chunks=(output_time_chunk, 1, trench_h, trench_w),
            shards=(math.ceil(n_times / output_time_chunk) * output_time_chunk,
                    n_channels, trench_h, trench_w),
            dtype=np.dtype(exp.data.dtype),
        )

    def _reopen(self, table: pd.DataFrame, layout: _ImageLayout, compressor: str,
                clevel: int, byte_shuffle: bool) -> zarr.Group:
        """The existing experiment group, if this extraction can resume it."""
        store = zarr.open_group(str(self.output_path), mode="r+", zarr_format=3)
        mismatches = self._resume_mismatches(
            store, table, layout.shape, layout.chunks, layout.shards, layout.dtype,
            compressor, clevel, byte_shuffle,
        )
        if mismatches:
            raise RuntimeError(
                "Existing output is not compatible with this extraction; "
                f"these inputs differ: {', '.join(mismatches)}. "
                "Use resume=False or choose a new output path."
            )
        return store

    def _create(self, table: pd.DataFrame, layout: _ImageLayout,
                codecs: Tuple[list, list]) -> zarr.Group:
        """A new experiment group: its metadata, and every trench image, empty.
        ``codecs``: the metadata's and the pixels' compressors."""
        exp = self.experiment
        channel_names = list(exp.channel_names)
        n_channels = len(channel_names)
        n_times = layout.shape[0]
        if self.output_path.exists() and not self._is_experiment_group():
            raise RuntimeError(
                f"{self.output_path} exists and is not an experiment group written by "
                "the Extractor; it is never overwritten. Choose a new output path."
            )
        metadata_codecs, pixel_codecs = codecs
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
        table_ds = self._build_trench_table(table, source_metadata)
        table_ds.to_zarr(
            str(self.output_path),
            mode="a",
            group="trenches",
            consolidated=False,
            zarr_format=3,
            encoding=self._metadata_encoding(table_ds, compressors=metadata_codecs),
        )
        self._write_metadata_groups(
            source_metadata=source_metadata,
            acquisition=acquisition,
            compressors=metadata_codecs,
            n_times=n_times,
        )
        store = zarr.open_group(str(self.output_path), mode="r+", zarr_format=3)
        fov_index = dict(zip(table_ds["trench"].values.tolist(),
                             table_ds["fov_index"].values.tolist()))
        for _, row in table.iterrows():
            image = store.create_group(
                trench_path(row["trench_id"]),
                attributes=self._image_attrs(
                    row, channel_names=channel_names, colors=colors,
                    pixel_size_um=pixel_size_um, frame_interval_ms=frame_interval_ms,
                    dtype=layout.dtype, fov_index=fov_index[int(row["trench_id"])],
                ),
            )
            image.create_array(
                "0", shape=layout.shape, dtype=layout.dtype, chunks=layout.chunks,
                shards=layout.shards, compressors=pixel_codecs, chunk_key_encoding=_CHUNK_KEYS,
                dimension_names=["t", "c", "y", "x"], fill_value=0,
            )
        store.attrs["pymmm"] = {
            **experiment_attrs, "completed_fovs": [], "extraction_complete": False,
        }
        return store

    def _crops(self, table: pd.DataFrame) -> List[FovCrops]:
        """How each FOV's trenches are cut (FOVs in table order)."""
        exp = self.experiment
        rotation = _rotation_matrix(
            self.registrator.rotation, (exp.data.sizes["Y"], exp.data.sizes["X"])
        )
        tmats = self.registrator.tmats  # (P, T, row, col) or (T, row, col)
        times = tuple(exp.source_axes["original_t_index"].tolist())
        crops = []
        for fov, fov_df in table.groupby("fov", sort=False):
            fov_tmats = tmats.sel(P=fov).values if "P" in tmats.dims else tmats.values
            trench_specs = tuple(
                (int(row["y_top"]), int(row["y_bottom"]), int(row["x_left"]),
                 int(row["x_right"]), bool(row["needs_flip"]))
                for _, row in fov_df.iterrows()
            )
            bands: dict = {}
            for local_idx, (y_top, y_bottom, x_left, x_right, flip) in enumerate(trench_specs):
                bands.setdefault((y_top, y_bottom), []).append((local_idx, x_left, x_right, flip))
            crops.append(FovCrops(
                nd2_path=str(exp.path), fov=str(fov), fov_index=exp.raw_fov_index(str(fov)),
                time_indices=times, roi=tuple(exp.source_roi), image_width=exp.data.sizes["X"],
                dtype=np.dtype(exp.data.dtype), tmats=np.matmul(fov_tmats, rotation),
                trench_ids=tuple(int(t) for t in fov_df["trench_id"]),
                trench_specs=trench_specs,
                band_specs=tuple((y_top, y_bottom, tuple(band))
                                 for (y_top, y_bottom), band in bands.items()),
            ))
        return crops

    def _write_fov(self, store: zarr.Group, crops: FovCrops, pixels: np.ndarray) -> None:
        """Write a FOV's trench images, ``(trench, t, c, y, x)``, and their display
        windows."""
        for local_idx, trench_id in enumerate(crops.trench_ids):
            image = store[trench_path(trench_id)]
            image["0"][...] = pixels[local_idx]
            image.attrs["ome"] = ome.with_windows(
                dict(image.attrs)["ome"], ome.display_windows(pixels[local_idx]))

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def preview(self, channel: Optional[Union[str, int]] = None) -> xr.DataArray:
        """Return a lazy dask-backed DataArray previewing the extraction output.

        The same crops as :meth:`extract`, with one lazy task per FOV and timepoint:
        selecting a frame reads one raw frame and cuts all its FOV's trenches,
        without writing anything. Browse it before committing to the full extraction
        (the notebook extra)::

            ndv.imshow(extractor.preview())

        Parameters
        ----------
        channel : str | int | None
            A specific channel to preview. ``None`` includes all channels
            (matching the output zarr layout).
        """
        table = self._trench_table("preview")
        layout = self._layout(table, None)
        n_times, n_channels, trench_h, trench_w = layout.shape
        channel_names = self.experiment.channel_names
        channels = None
        if channel is not None:
            channels = (int(channel) if isinstance(channel, int)
                        else channel_names.index(str(channel)),)
            n_channels = 1

        fov_blocks = []
        for fov_crops in self._crops(table):
            n_trenches = len(fov_crops.trench_ids)
            time_blocks = [
                da.from_delayed(
                    dask.delayed(_crop_frames)(*fov_crops.batch(t, t + 1, channels)),
                    shape=(1, n_trenches, n_channels, trench_h, trench_w),
                    dtype=layout.dtype,
                ).transpose(1, 0, 2, 3, 4)
                for t in range(n_times)
            ]
            fov_blocks.append(da.concatenate(time_blocks, axis=1))
        result_da = da.concatenate(fov_blocks, axis=0)
        coords = {
            "Trench": table["trench_id"].to_numpy(dtype=np.int32),
            "Y": np.arange(trench_h),
            "X": np.arange(trench_w),
        }
        if channel is not None:
            return xr.DataArray(result_da[:, :, 0], dims=["Trench", "T", "Y", "X"],
                                coords=coords)
        return xr.DataArray(result_da, dims=["Trench", "T", "C", "Y", "X"],
                            coords={**coords, "C": channel_names})

    def extract_single_trench(
        self,
        trench_id: int,
        channel: Optional[Union[str, int]] = None,
    ) -> np.ndarray:
        """Cut one trench out of every frame, as :meth:`extract` does, without writing.

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
        (fov_crops,) = self._crops(table[table["trench_id"] == trench_id])
        exp = self.experiment
        name = normalize_channel_arg(
            channel if channel is not None else self.registrator.channel, exp.channel_names)
        n_times = len(fov_crops.time_indices)
        frames = _crop_frames(*fov_crops.batch(0, n_times, [exp.channel_names.index(name)]))
        return frames[:, 0, 0]

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
