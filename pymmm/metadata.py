"""Metadata helpers for serializing ND2 acquisition configuration."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import nd2
import numpy as np

SOURCE_METADATA_VERSION = 1

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
        return list(value)
    if isinstance(value, Mapping):
        return {str(key): _to_builtin(val) for key, val in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_builtin(item) for item in value]
    return value


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
                "selected": selected_custom,
                "custom_data_keys": sorted(custom_data.keys()),
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
