"""Metadata helpers for ND2 acquisition/configuration and trench-store export."""

from __future__ import annotations

import dataclasses
import json
import re
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import nd2
import numpy as np
import xarray as xr


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

    relative_time = time.get("relativeTimeMs")
    julian_date = time.get("absoluteJulianDayNumber")
    pfs_offset = position.get("pfsOffset")

    return {
        "relative_time": (
            float(relative_time) if relative_time is not None else np.nan
        ),
        "julian_date": (float(julian_date) if julian_date is not None else np.nan),
        "stage_position": stage_xyz,
        "pfs_offset": float(pfs_offset) if pfs_offset is not None else np.nan,
    }


def _static_frame_metadata(frame_metadata: dict[str, Any]) -> dict[str, Any]:
    """Return a frame metadata payload without its per-frame fields.

    Time, stage position and PFS offset change every frame and are stored as
    typed ``/source_frames`` variables; what remains is the same for every
    frame of a FOV.
    """
    static = json.loads(json.dumps(frame_metadata))
    for channel in static.get("channels") or []:
        channel.pop("time", None)
        position = channel.get("position") or {}
        position.pop("stagePositionUm", None)
        position.pop("pfsOffset", None)
    return static


def _replace_nan_with_none(value: Any) -> Any:
    """Normalize missing scalar values to None for dtype inference."""
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    return value


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
    normalized = re.sub(
        r"\[(.*?)\]", lambda match: "_" + _normalize_name(match.group(1)), normalized
    )
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", normalized)
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_").lower()
    if normalized and normalized[0].isdigit():
        normalized = f"field_{normalized}"
    return normalized or "value"


def extract_source_frame_metadata(experiment: Any) -> dict[str, Any]:
    """Collect per-source-frame metadata for the frames exported by the pipeline."""
    export_axes = experiment.source_axes
    fov_names = export_axes["fov_names"]
    original_p_index = export_axes["original_p_index"]
    original_t_index = export_axes["original_t_index"]

    n_fovs = len(fov_names)
    n_times = len(original_t_index)
    source_seq_index = np.empty((n_fovs, n_times), dtype=np.int64)
    relative_time = np.full((n_fovs, n_times), np.nan, dtype=np.float64)
    julian_date = np.full((n_fovs, n_times), np.nan, dtype=np.float64)
    stage_position = np.full((n_fovs, n_times, 3), np.nan, dtype=np.float64)
    pfs_offset = np.full((n_fovs, n_times), np.nan, dtype=np.float64)
    pfs_status = np.full((n_fovs, n_times), np.nan, dtype=np.float64)
    frame_metadata_json = np.empty((n_fovs,), dtype=object)

    with nd2.ND2File(experiment.path) as nd2_file:
        coord_axes = [
            axis for axis in nd2_file.sizes if axis not in {"X", "Y", "C", "S"}
        ]
        coord_shape = tuple(int(nd2_file.sizes[axis]) for axis in coord_axes)
        axis_positions = {axis: idx for idx, axis in enumerate(coord_axes)}

        changing_fovs: set[int] = set()
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
                    if coord_axes
                    else 0
                )
                frame_dict = _to_builtin(nd2_file.frame_metadata(seq_index))
                scalars = _extract_frame_scalars(frame_dict)

                source_seq_index[local_p, local_t] = seq_index
                relative_time[local_p, local_t] = scalars["relative_time"]
                julian_date[local_p, local_t] = scalars["julian_date"]
                stage_position[local_p, local_t] = scalars["stage_position"]
                pfs_offset[local_p, local_t] = scalars["pfs_offset"]

                static_json = _json_dumps(_static_frame_metadata(frame_dict))
                if local_t == 0:
                    frame_metadata_json[local_p] = static_json
                elif (
                    static_json != frame_metadata_json[local_p]
                    and local_p not in changing_fovs
                ):
                    changing_fovs.add(local_p)
                    warnings.warn(
                        f"Frame metadata of {fov_names[local_p]} changes over time "
                        f"(first at T={int(raw_t)}); only its first frame's static "
                        "metadata is stored.",
                        stacklevel=2,
                    )

        # The ND2 event table has one row per frame plus non-frame acquisition
        # events. Frame rows are the reliable source for PFS: frame metadata
        # often reports pfsOffset as null.
        acquisition_event_records: list[dict[str, Any]] = []
        if not bool(nd2_file.is_legacy):
            local_p_of = {
                int(raw): idx for idx, raw in enumerate(original_p_index.tolist())
            }
            local_t_of = {
                int(raw): idx for idx, raw in enumerate(original_t_index.tolist())
            }
            for row in nd2_file.events(orient="records"):
                row_dict = _to_builtin(row)
                if "Index" not in row_dict:
                    acquisition_event_records.append(row_dict)
                    continue

                local_p = local_p_of.get(int(row_dict.get("P Index", 0)))
                local_t = local_t_of.get(int(row_dict.get("T Index", 0)))
                if (
                    local_p is None
                    or local_t is None
                    or int(row_dict.get("Z Index", 0)) != 0
                ):
                    continue
                for column, target in (
                    ("PFS Offset", pfs_offset),
                    ("PFS Status", pfs_status),
                ):
                    value = _replace_nan_with_none(row_dict.get(column))
                    if value is not None:
                        target[local_p, local_t] = float(value)

    return {
        "fov_names": fov_names,
        "original_p_index": original_p_index,
        "original_t_index": original_t_index,
        "source_seq_index": source_seq_index,
        "relative_time": relative_time,
        "julian_date": julian_date,
        "stage_position": stage_position,
        "pfs_offset": pfs_offset,
        "pfs_status": pfs_status,
        "frame_metadata_json": frame_metadata_json,
        "acquisition_event_records": acquisition_event_records,
    }


SOURCE_FRAME_ATTRS = {
    "original_p_index": {"long_name": "FOV (P) index in the source ND2"},
    "source_seq_index": {"long_name": "frame sequence index in the source ND2"},
    "relative_time": {
        "units": "millisecond",
        "long_name": "acquisition time since experiment start",
    },
    "julian_date": {
        "units": "day",
        "long_name": "absolute acquisition time as Julian date",
    },
    "pfs_offset": {"long_name": "Nikon Perfect Focus System offset (device units)"},
    "pfs_status": {"long_name": "Nikon Perfect Focus System status (1 = in focus)"},
    "stage_position": {"units": "micrometer", "long_name": "stage position"},
    "frame_metadata_json": {
        "long_name": "ND2 frame metadata without per-frame time, stage and PFS fields, as JSON",
    },
}


def build_source_frames_dataset(metadata: dict[str, Any]) -> xr.Dataset:
    """Build the normalized ``/source_frames`` xarray dataset."""
    dataset = xr.Dataset(
        data_vars={
            "original_p_index": (("FOV",), metadata["original_p_index"]),
            "source_seq_index": (("FOV", "T"), metadata["source_seq_index"]),
            "relative_time": (("FOV", "T"), metadata["relative_time"]),
            "julian_date": (
                ("FOV", "T"),
                metadata["julian_date"],
            ),
            "pfs_offset": (("FOV", "T"), metadata["pfs_offset"]),
            "pfs_status": (("FOV", "T"), metadata["pfs_status"]),
            "stage_position": (
                ("FOV", "T", "Axis"),
                metadata["stage_position"],
            ),
            "frame_metadata_json": (("FOV",), metadata["frame_metadata_json"]),
        },
        coords={
            "FOV": np.asarray(metadata["fov_names"], dtype=object),
            "T": metadata["original_t_index"],
            "Axis": np.asarray(["x", "y", "z"], dtype=object),
        },
        attrs={"layout": "source_frames"},
    )
    for name, attrs in SOURCE_FRAME_ATTRS.items():
        dataset[name].attrs.update(attrs)
    dataset["FOV"].attrs["long_name"] = "field of view name"
    dataset["T"].attrs["long_name"] = "frame index (original ND2 T index)"
    dataset["Axis"].attrs["long_name"] = "stage axis"
    return dataset


_UNIT_NAMES = {
    "s": "second",
    "ms": "millisecond",
    "µm": "micrometer",
    "μm": "micrometer",
    "um": "micrometer",
    "nm": "nanometer",
    "°c": "degree_Celsius",
}


def _split_unit(column: str) -> tuple[str, str | None]:
    """Split an ND2 column name like ``"Time [s]"`` into ``("Time", "second")``."""
    match = re.fullmatch(r"\s*(.*?)\s*\[(.+?)\]\s*", str(column))
    if not match:
        return str(column), None
    base, unit = match.groups()
    return base, _UNIT_NAMES.get(unit.lower(), unit)


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
        base, unit = _split_unit(column)
        var_name = f"{var_prefix}_{_normalize_name(base)}"
        dataset[var_name] = xr.DataArray(
            _values_to_array(values),
            dims=(dim_name,),
        )
        dataset[var_name].attrs["source_column"] = str(column)
        if unit:
            dataset[var_name].attrs["units"] = unit

    return dataset


def build_events_dataset(
    acquisition_event_records: Sequence[Mapping[str, Any]],
) -> xr.Dataset:
    """Build the ``/events`` xarray dataset of non-frame acquisition events.

    Per-frame event columns are folded into ``/source_frames`` instead.
    """
    dataset = xr.Dataset(
        attrs={"layout": "events"},
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


def _channel_exposure_ms(nd2_file: Any, n_channels: int) -> list[float | None]:
    """Per-channel exposure times (ms) from nd2's OME metadata.

    The ND2 event table only records the first channel's exposure; the real
    per-channel values live in the picture-plane sample settings, which nd2
    exposes publicly through its OME planes.
    """
    exposures: list[float | None] = [None] * n_channels
    try:
        planes = nd2_file.ome_metadata().images[0].pixels.planes
    except Exception:
        return exposures
    for plane in planes:
        c = int(plane.the_c)
        if c < n_channels and exposures[c] is None and plane.exposure_time is not None:
            exposures[c] = float(plane.exposure_time)
    return exposures


def _color_hex(color: Mapping[str, Any] | None) -> str:
    if not color:
        return ""
    return "#{:02X}{:02X}{:02X}".format(
        int(color.get("r", 0)),
        int(color.get("g", 0)),
        int(color.get("b", 0)),
    )


def build_acquisition_dataset(
    acquisition_metadata: Mapping[str, Any],
    *,
    channel_names: Sequence[str] | None = None,
) -> xr.Dataset:
    """Build the ``/acquisition`` xarray dataset.

    Typed per-channel settings and the static optics, plus one compressed
    ``raw_json`` dump of everything read from the ND2.
    """
    nd2_metadata = _to_builtin(acquisition_metadata.get("metadata") or {})
    channel_records = nd2_metadata.get("channels") or []
    acquisition_channel_names = [
        str((record.get("channel") or {}).get("name") or f"channel_{idx}")
        for idx, record in enumerate(channel_records)
    ]
    if channel_names is not None and len(channel_names) == len(channel_records):
        coord_name = "C"
        coord_values = [str(name) for name in channel_names]
    else:
        coord_name = "AcquisitionChannel"
        coord_values = acquisition_channel_names

    exposures = list(acquisition_metadata.get("channel_exposure_ms") or [])
    exposures += [None] * (len(channel_records) - len(exposures))

    def _channel_var(values, dtype, **attrs):
        return xr.DataArray(
            np.asarray(values, dtype=dtype), dims=(coord_name,), attrs=attrs
        )

    def _nan(value):
        return np.nan if value is None else float(value)

    channels = [record.get("channel") or {} for record in channel_records]
    microscopes = [record.get("microscope") or {} for record in channel_records]
    data_vars = {
        "exposure_time": _channel_var(
            [_nan(v) for v in exposures],
            np.float64,
            units="millisecond",
            long_name="camera exposure time",
        ),
        "emission_wavelength": _channel_var(
            [_nan(ch.get("emissionLambdaNm")) for ch in channels],
            np.float64,
            units="nanometer",
            long_name="emission wavelength",
        ),
        "excitation_wavelength": _channel_var(
            [_nan(ch.get("excitationLambdaNm")) for ch in channels],
            np.float64,
            units="nanometer",
            long_name="excitation wavelength",
        ),
        "color": _channel_var(
            [_color_hex(ch.get("color")) for ch in channels],
            object,
            long_name="display color (#RRGGBB)",
        ),
        "modality": _channel_var(
            [",".join(m.get("modalityFlags") or []) for m in microscopes],
            object,
            long_name="imaging modality flags",
        ),
    }

    file_summary = acquisition_metadata.get("file_summary") or {}
    data_vars["pixel_size"] = xr.DataArray(
        float(file_summary.get("pixel_size_um", np.nan)),
        attrs={"units": "micrometer", "long_name": "image pixel size"},
    )
    data_vars["time_interval"] = xr.DataArray(
        float(file_summary.get("time_interval_ms", np.nan)),
        attrs={"units": "millisecond", "long_name": "nominal time between frames"},
    )
    data_vars["raw_json"] = xr.DataArray(
        np.asarray(_json_dumps(acquisition_metadata), dtype=object),
        attrs={"long_name": "complete metadata read from the source ND2, as JSON"},
    )

    optics = microscopes[0] if microscopes else {}
    attrs = {
        "layout": "acquisition",
        "objective_name": optics.get("objectiveName"),
        "objective_magnification": optics.get("objectiveMagnification"),
        "objective_numerical_aperture": optics.get("objectiveNumericalAperture"),
        "zoom_magnification": optics.get("zoomMagnification"),
        "immersion_refractive_index": optics.get("immersionRefractiveIndex"),
    }
    return xr.Dataset(
        data_vars=data_vars,
        coords={coord_name: np.asarray(coord_values, dtype=object)},
        attrs={key: value for key, value in attrs.items() if value is not None},
    )


def _median_frame_interval_ms(event_rows: Sequence[Mapping[str, Any]]) -> float | None:
    """Median time between consecutive timepoints of the same FOV, from the
    ND2 event table's frame rows."""
    times_by_fov: dict[int, dict[int, float]] = {}
    for row in event_rows:
        if "Index" not in row or row.get("Time [s]") is None:
            continue
        times_by_fov.setdefault(int(row.get("P Index", 0)), {})[
            int(row.get("T Index", 0))
        ] = float(row["Time [s]"]) * 1000.0
    gaps = [
        np.diff([times[t] for t in sorted(times)])
        for times in times_by_fov.values()
        if len(times) > 1
    ]
    if not gaps:
        return None
    return float(np.median(np.concatenate(gaps)))


def extract_acquisition_metadata(
    path: Path,
    *,
    pixel_size_um: float,
    time_interval_ms: float,
) -> dict[str, Any]:
    """Read ND2 metadata without materializing image data.

    ``pixel_size_um`` and ``time_interval_ms`` are fallbacks: the pixel size
    comes from the ND2 voxel size and the interval from the median gap between
    real frame times, when the file provides them.
    """
    with nd2.ND2File(path) as nd2_file:
        nd2_metadata = _to_builtin(nd2_file.metadata)
        n_channels = len((nd2_metadata or {}).get("channels") or [])
        voxel = nd2_file.voxel_size()
        if voxel is not None and voxel.x and voxel.x > 0:
            pixel_size_um = float(voxel.x)
        if not bool(nd2_file.is_legacy):
            measured = _median_frame_interval_ms(nd2_file.events(orient="records"))
            if measured is not None:
                time_interval_ms = measured
        return {
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
            "metadata": nd2_metadata,
            "experiment": _to_builtin(nd2_file.experiment),
            "attributes": _to_builtin(nd2_file.attributes),
            "text_info": _to_builtin(nd2_file.text_info),
            "custom_data": _to_builtin(nd2_file.custom_data or {}),
            "unstructured_metadata": _to_builtin(nd2_file.unstructured_metadata()),
            "channel_exposure_ms": _channel_exposure_ms(nd2_file, n_channels),
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
