"""Metadata helpers for ND2 acquisition/configuration and trench-store export."""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import nd2
import numpy as np
import xarray as xr

SOURCE_METADATA_VERSION = 2
EXTRACTOR_STORE_VERSION = 1

_SELECTED_CUSTOM_DATA_KEYS = (
    "AcqTimeV1_0",
    "AppInfo_V1_0",
    "GrabberCameraSettingsV1_0",
    "CustomDataV2_0",
)


def _to_builtin(value: Any) -> Any:
    """Convert nd2/numpy-rich objects to plain JSON-safe builtins."""
    if dataclasses.is_dataclass(value):
        return {key: _to_builtin(val) for key, val in dataclasses.asdict(value).items()}
    if hasattr(value, "_asdict"):
        return {key: _to_builtin(val) for key, val in value._asdict().items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        return {str(key): _to_builtin(val) for key, val in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_builtin(item) for item in value]
    return value


def _json_dumps(value: Any) -> str:
    """Serialize metadata deterministically to compact JSON."""
    return json.dumps(
        _to_builtin(value),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _exported_source_axes(experiment: Any) -> dict[str, Any]:
    """Return exported source-frame indices for the current experiment subset."""
    raw_data = experiment._raw_data

    if "P" in raw_data.dims:
        raw_fov_names = [str(v) for v in raw_data.coords["P"].values]
        active_fov_names = list(experiment.fov_names)
        original_p_index = np.asarray(
            [raw_fov_names.index(name) for name in active_fov_names],
            dtype=np.int32,
        )
    else:
        active_fov_names = ["single"]
        original_p_index = np.asarray([0], dtype=np.int32)

    if "T" in raw_data.dims:
        n_raw_times = int(raw_data.sizes["T"])
        if experiment._time_slice is not None:
            original_t_index = np.asarray(
                list(range(*experiment._time_slice.indices(n_raw_times))),
                dtype=np.int32,
            )
        else:
            original_t_index = np.arange(n_raw_times, dtype=np.int32)
    else:
        original_t_index = np.asarray([0], dtype=np.int32)

    return {
        "fov_names": active_fov_names,
        "original_p_index": original_p_index,
        "original_t_index": original_t_index,
    }


def _extract_frame_scalars(frame_metadata: dict[str, Any]) -> dict[str, Any]:
    """Extract common scalar fields from a serialized ND2 frame metadata payload."""
    channels = frame_metadata.get("channels") or []
    channel0 = channels[0] if channels else {}
    position = channel0.get("position") or {}
    time = channel0.get("time") or {}
    stage = position.get("stagePositionUm") or {}

    if isinstance(stage, Mapping):
        stage_xyz = (
            float(stage.get("x", np.nan)),
            float(stage.get("y", np.nan)),
            float(stage.get("z", np.nan)),
        )
    elif isinstance(stage, Sequence) and not isinstance(stage, (str, bytes, bytearray)):
        padded = list(stage) + [np.nan, np.nan, np.nan]
        stage_xyz = tuple(float(v) for v in padded[:3])
    else:
        stage_xyz = (np.nan, np.nan, np.nan)

    relative_time_ms = time.get("relativeTimeMs")
    absolute_julian_day_number = time.get("absoluteJulianDayNumber")
    pfs_offset = position.get("pfsOffset")

    return {
        "relative_time_ms": (
            float(relative_time_ms) if relative_time_ms is not None else np.nan
        ),
        "absolute_julian_day_number": (
            float(absolute_julian_day_number)
            if absolute_julian_day_number is not None else np.nan
        ),
        "stage_position_um": stage_xyz,
        "pfs_offset": float(pfs_offset) if pfs_offset is not None else np.nan,
        "position_name": str(position.get("name") or ""),
    }


def _replace_nan_with_none(value: Any) -> Any:
    """Normalize missing scalar values to None for dtype inference."""
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    return value


def _is_simple_scalar(value: Any) -> bool:
    builtin = _to_builtin(value)
    return builtin is None or isinstance(
        builtin,
        (str, bool, int, float),
    )


def _values_to_array(values: Sequence[Any]) -> np.ndarray:
    """Convert a list of scalar-ish values into a stable ndarray."""
    cleaned = [_replace_nan_with_none(_to_builtin(value)) for value in values]
    if not cleaned:
        return np.asarray([], dtype=np.float64)

    if all(value is None or isinstance(value, (bool, np.bool_)) for value in cleaned):
        if any(value is None for value in cleaned):
            return np.asarray(cleaned, dtype=object)
        return np.asarray(cleaned, dtype=bool)

    if all(
        value is None
        or (isinstance(value, (int, np.integer)) and not isinstance(value, bool))
        for value in cleaned
    ):
        if any(value is None for value in cleaned):
            return np.asarray(
                [np.nan if value is None else float(value) for value in cleaned],
                dtype=np.float64,
            )
        return np.asarray(cleaned, dtype=np.int64)

    if all(
        value is None
        or (
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, bool)
        )
        for value in cleaned
    ):
        return np.asarray(
            [np.nan if value is None else float(value) for value in cleaned],
            dtype=np.float64,
        )

    if all(value is None or isinstance(value, str) for value in cleaned):
        return np.asarray(
            ["" if value is None else value for value in cleaned],
            dtype=object,
        )

    return np.asarray(
        [
            _json_dumps(value)
            if isinstance(value, (Mapping, list, tuple))
            else ("" if value is None else str(value))
            for value in cleaned
        ],
        dtype=object,
    )


def _normalize_name(name: str) -> str:
    """Normalize a free-form field name into snake_case."""
    normalized = str(name).replace("µ", "u").replace("μ", "u")
    normalized = re.sub(r"\[(.*?)\]", lambda match: "_" + _normalize_name(match.group(1)), normalized)
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", normalized)
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_").lower()
    if normalized and normalized[0].isdigit():
        normalized = f"field_{normalized}"
    return normalized or "value"


def extract_source_frame_metadata(experiment: Any) -> dict[str, Any]:
    """Collect per-source-frame metadata for the frames exported by the pipeline."""
    export_axes = _exported_source_axes(experiment)
    fov_names = export_axes["fov_names"]
    original_p_index = export_axes["original_p_index"]
    original_t_index = export_axes["original_t_index"]

    n_fovs = len(fov_names)
    n_times = len(original_t_index)
    source_seq_index = np.empty((n_fovs, n_times), dtype=np.int64)
    relative_time_ms = np.full((n_fovs, n_times), np.nan, dtype=np.float64)
    absolute_julian_day_number = np.full((n_fovs, n_times), np.nan, dtype=np.float64)
    stage_position_um = np.full((n_fovs, n_times, 3), np.nan, dtype=np.float64)
    pfs_offset = np.full((n_fovs, n_times), np.nan, dtype=np.float64)
    position_name = np.empty((n_fovs, n_times), dtype=object)
    frame_metadata_json = np.empty((n_fovs, n_times), dtype=object)

    with nd2.ND2File(experiment.path) as nd2_file:
        coord_axes = [axis for axis in nd2_file.sizes if axis not in {"X", "Y", "C", "S"}]
        coord_shape = tuple(int(nd2_file.sizes[axis]) for axis in coord_axes)
        axis_positions = {axis: idx for idx, axis in enumerate(coord_axes)}

        for local_p, raw_p in enumerate(original_p_index):
            for local_t, raw_t in enumerate(original_t_index):
                coords = [0] * len(coord_axes)
                if "T" in axis_positions:
                    coords[axis_positions["T"]] = int(raw_t)
                if "P" in axis_positions:
                    coords[axis_positions["P"]] = int(raw_p)
                if "Z" in axis_positions:
                    coords[axis_positions["Z"]] = 0

                seq_index = (
                    int(np.ravel_multi_index(tuple(coords), coord_shape))
                    if coord_axes else 0
                )
                frame_dict = _to_builtin(nd2_file.frame_metadata(seq_index))
                scalars = _extract_frame_scalars(frame_dict)

                source_seq_index[local_p, local_t] = seq_index
                relative_time_ms[local_p, local_t] = scalars["relative_time_ms"]
                absolute_julian_day_number[local_p, local_t] = scalars[
                    "absolute_julian_day_number"
                ]
                stage_position_um[local_p, local_t] = scalars["stage_position_um"]
                pfs_offset[local_p, local_t] = scalars["pfs_offset"]
                position_name[local_p, local_t] = scalars["position_name"]
                frame_metadata_json[local_p, local_t] = _json_dumps(frame_dict)

        frame_event_records: list[dict[str, Any]] = []
        acquisition_event_records: list[dict[str, Any]] = []
        if not bool(nd2_file.is_legacy):
            allowed_p = {int(idx) for idx in original_p_index.tolist()}
            allowed_t = {int(idx) for idx in original_t_index.tolist()}
            for row in nd2_file.events(orient="records"):
                row_dict = _to_builtin(row)
                is_frame_row = "Index" in row_dict
                if not is_frame_row:
                    acquisition_event_records.append(row_dict)
                    continue

                row_p = int(row_dict.get("P Index", 0))
                row_t = int(row_dict.get("T Index", 0))
                row_z = int(row_dict.get("Z Index", 0))
                if row_p in allowed_p and row_t in allowed_t and row_z == 0:
                    frame_event_records.append(row_dict)

    return {
        "fov_names": fov_names,
        "original_p_index": original_p_index,
        "original_t_index": original_t_index,
        "source_seq_index": source_seq_index,
        "relative_time_ms": relative_time_ms,
        "absolute_julian_day_number": absolute_julian_day_number,
        "stage_position_um": stage_position_um,
        "pfs_offset": pfs_offset,
        "position_name": position_name,
        "frame_metadata_json": frame_metadata_json,
        "frame_event_records": frame_event_records,
        "acquisition_event_records": acquisition_event_records,
    }


def build_source_frames_dataset(metadata: dict[str, Any]) -> xr.Dataset:
    """Build the normalized ``/source_frames`` xarray dataset."""
    return xr.Dataset(
        data_vars={
            "original_p_index": (("FOV",), metadata["original_p_index"]),
            "source_seq_index": (("FOV", "T"), metadata["source_seq_index"]),
            "relative_time_ms": (("FOV", "T"), metadata["relative_time_ms"]),
            "absolute_julian_day_number": (
                ("FOV", "T"),
                metadata["absolute_julian_day_number"],
            ),
            "pfs_offset": (("FOV", "T"), metadata["pfs_offset"]),
            "stage_position_um": (
                ("FOV", "T", "Axis"),
                metadata["stage_position_um"],
            ),
            "position_name": (("FOV", "T"), metadata["position_name"]),
            "frame_metadata_json": (
                ("FOV", "T"),
                metadata["frame_metadata_json"],
            ),
        },
        coords={
            "FOV": np.asarray(metadata["fov_names"], dtype=object),
            "T": metadata["original_t_index"],
            "Axis": np.asarray(["x", "y", "z"], dtype=object),
        },
        attrs={
            "extractor_store_version": EXTRACTOR_STORE_VERSION,
            "layout": "source_frames_v1",
        },
    )


def _records_to_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    dim_name: str,
    var_prefix: str,
) -> xr.Dataset:
    """Convert a list of record dicts into a columnar xarray dataset."""
    dataset = xr.Dataset(
        coords={dim_name: np.arange(len(records), dtype=np.int32)},
    )
    if not records:
        return dataset

    columns = sorted({str(key) for record in records for key in record.keys()})
    for column in columns:
        values = [_replace_nan_with_none(record.get(column)) for record in records]
        var_name = f"{var_prefix}_{_normalize_name(column)}"
        dataset[var_name] = xr.DataArray(
            _values_to_array(values),
            dims=(dim_name,),
        )
        dataset[var_name].attrs["source_column"] = str(column)

    return dataset


def build_events_dataset(
    frame_event_records: Sequence[Mapping[str, Any]],
    acquisition_event_records: Sequence[Mapping[str, Any]],
) -> xr.Dataset:
    """Build the ``/events`` xarray dataset."""
    dataset = xr.Dataset(
        attrs={
            "extractor_store_version": EXTRACTOR_STORE_VERSION,
            "layout": "events_v1",
        },
    )
    dataset = dataset.merge(
        _records_to_dataset(
            frame_event_records,
            dim_name="FrameEvent",
            var_prefix="frame",
        ),
        compat="override",
    )
    dataset = dataset.merge(
        _records_to_dataset(
            acquisition_event_records,
            dim_name="AcquisitionEvent",
            var_prefix="acquisition",
        ),
        compat="override",
    )
    return dataset


def build_acquisition_dataset(
    acquisition_metadata: Mapping[str, Any],
    *,
    channel_names: Sequence[str] | None = None,
) -> xr.Dataset:
    """Build the ``/acquisition`` xarray dataset."""
    structured = _to_builtin(acquisition_metadata.get("structured", {}))
    metadata_block = structured.get("metadata", {})
    custom = _to_builtin(acquisition_metadata.get("custom", {}))
    channel_records = metadata_block.get("channels") or []
    acquisition_channel_names = [
        str(record.get("name") or f"channel_{idx}")
        for idx, record in enumerate(channel_records)
    ]
    canonical_channel_names = (
        [str(name) for name in channel_names]
        if channel_names is not None else acquisition_channel_names
    )
    if channel_records and len(canonical_channel_names) == len(channel_records):
        coord_name = "C"
        coord_values = np.asarray(canonical_channel_names, dtype=object)
    elif channel_records:
        coord_name = "AcquisitionChannel"
        coord_values = np.asarray(acquisition_channel_names, dtype=object)
    else:
        coord_name = None
        coord_values = None

    dataset = xr.Dataset(
        attrs={
            "extractor_store_version": EXTRACTOR_STORE_VERSION,
            "source_metadata_version": int(
                acquisition_metadata.get("schema_version", SOURCE_METADATA_VERSION)
            ),
            "nd2_library_version": acquisition_metadata.get("nd2_library_version"),
            "file_summary": _to_builtin(acquisition_metadata.get("file_summary", {})),
            "text_info": _to_builtin(structured.get("text_info", {})),
            "attributes": _to_builtin(structured.get("attributes", {})),
            "experiment": _to_builtin(structured.get("experiment", [])),
            "custom_data_keys": _to_builtin(custom.get("custom_data_keys", [])),
            "selected_custom": _to_builtin(custom.get("selected", {})),
            "unstructured_metadata_keys": _to_builtin(
                custom.get("unstructured_metadata_keys", [])
            ),
            "layout": "acquisition_v1",
        },
    )
    if coord_name is not None and coord_values is not None:
        dataset = dataset.assign_coords({coord_name: coord_values})

    if channel_records:
        channel_keys = sorted(
            {
                key
                for record in channel_records
                for key, value in record.items()
                if key != "name" and _is_simple_scalar(value)
            }
        )
        for key in channel_keys:
            var_name = _normalize_name(key)
            dataset[var_name] = xr.DataArray(
                _values_to_array([record.get(key) for record in channel_records]),
                dims=(coord_name,),
            )
            dataset[var_name].attrs["source_key"] = str(key)

    dataset["raw_json"] = xr.DataArray(_json_dumps(acquisition_metadata))
    return dataset


def extract_acquisition_metadata(
    path: Path,
    *,
    pixel_size_um: float,
    time_interval_ms: float,
) -> dict[str, Any]:
    """Read ND2 metadata without materializing image data."""
    with nd2.ND2File(path) as nd2_file:
        custom_data = nd2_file.custom_data or {}
        selected_custom = {
            key: _to_builtin(custom_data[key])
            for key in _SELECTED_CUSTOM_DATA_KEYS
            if key in custom_data
        }
        unstructured_metadata = nd2_file.unstructured_metadata()
        return {
            "schema_version": SOURCE_METADATA_VERSION,
            "nd2_library_version": nd2.__version__,
            "file_summary": {
                "source_file": str(path.resolve()),
                "source_size_bytes": int(path.stat().st_size),
                "sizes": {axis: int(size) for axis, size in nd2_file.sizes.items()},
                "shape": [int(size) for size in nd2_file.shape],
                "dtype": str(nd2_file.dtype),
                "is_legacy": bool(nd2_file.is_legacy),
                "voxel_size_um": _to_builtin(nd2_file.voxel_size()),
                "pixel_size_um": float(pixel_size_um),
                "time_interval_ms": float(time_interval_ms),
            },
            "structured": {
                "metadata": _to_builtin(nd2_file.metadata),
                "experiment": _to_builtin(nd2_file.experiment),
                "attributes": _to_builtin(nd2_file.attributes),
                "text_info": _to_builtin(nd2_file.text_info),
            },
            "custom": {
                "data": _to_builtin(custom_data),
                "selected": selected_custom,
                "custom_data_keys": sorted(custom_data.keys()),
                "unstructured": _to_builtin(unstructured_metadata),
                "unstructured_metadata_keys": sorted(unstructured_metadata.keys()),
            },
        }


def slice_to_dict(value: slice | None) -> dict[str, int | None] | None:
    """Serialize a python slice into a JSON-safe mapping."""
    if value is None:
        return None
    return {
        "start": value.start,
        "stop": value.stop,
        "step": value.step,
    }


def build_store_metadata_attrs(experiment: Any) -> dict[str, Any]:
    """Return the metadata attrs that should be written to zarr stores."""
    return {
        "source_metadata_version": int(experiment.source_metadata_version),
        "source_acquisition_metadata": experiment.source_acquisition_metadata,
        "source_subset_metadata": experiment.source_subset_metadata,
    }
