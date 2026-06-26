from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SourceConfig:
    source_nd2: Path


@dataclass
class PipelineConfig:
    registration_channel: str | int = "PC"
    detection_channel: str | int = "PC"
    registration_mode: str | int = "mean"
    mean_n_frames: int = 10
    mean_from: str = "end"


@dataclass
class RotationConfig:
    mode: str = "fixed"
    fixed_degrees: float = 0.0


@dataclass
class DetectionConfig:
    min_lanes_per_fov: int
    max_lanes_per_fov: int
    min_trenches_per_lane: int = 1
    max_trenches_per_lane: int = 300
    lane_sigma: float = 40.0
    lane_distance: float = 300.0
    lane_height: float = 5000.0
    lane_orientation_window_um: float = 20.0
    trench_sigma: float = 4.0
    trench_distance: float = 100.0
    trench_prominence: float = 10.0
    trench_length: int = 160
    trench_bottom_offset: int = 50
    trench_width: int = 20


@dataclass
class QCConfig:
    min_lanes_per_fov: int
    max_lanes_per_fov: int
    max_drift_jump_px: float = 8.0
    drift_jump_mad_multiplier: float = 8.0
    max_trench_spacing_cv: float = 0.35
    min_nonzero_fraction: float = 0.001
    max_saturated_fraction: float = 0.30
    representative_trenches: int = 8


@dataclass
class OutputConfig:
    diagnostics_dir: Path
    output_path: Path | None = None


@dataclass
class ResourceConfig:
    registration_n_jobs: int = 1
    show_progress: bool = True


@dataclass
class AutonomousConfig:
    source: SourceConfig
    pipeline: PipelineConfig
    rotation: RotationConfig
    detection: DetectionConfig
    qc: QCConfig
    output: OutputConfig
    resources: ResourceConfig


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"[{name}] must be a table")
    return value


def _require_lane_priors(detection: DetectionConfig) -> None:
    if detection.min_lanes_per_fov <= 0 or detection.max_lanes_per_fov <= 0:
        raise ValueError("lane priors must be filled in before lane detection")
    if detection.min_lanes_per_fov > detection.max_lanes_per_fov:
        raise ValueError("min_lanes_per_fov cannot exceed max_lanes_per_fov")


def load_config(path: str | Path) -> AutonomousConfig:
    config_path = Path(path)
    data = tomllib.loads(config_path.read_text())

    source_data = _section(data, "source")
    if "source_nd2" not in source_data:
        raise ValueError("[source].source_nd2 is required")
    source = SourceConfig(source_nd2=Path(source_data["source_nd2"]))

    pipeline_data = _section(data, "pipeline")
    pipeline = PipelineConfig(
        registration_channel=pipeline_data.get("registration_channel", "PC"),
        detection_channel=pipeline_data.get("detection_channel", "PC"),
        registration_mode=pipeline_data.get("registration_mode", "mean"),
        mean_n_frames=int(pipeline_data.get("mean_n_frames", 10)),
        mean_from=str(pipeline_data.get("mean_from", "end")),
    )

    rotation_data = _section(data, "rotation")
    rotation = RotationConfig(
        mode=str(rotation_data.get("mode", "fixed")),
        fixed_degrees=float(rotation_data.get("fixed_degrees", 0.0)),
    )

    detection_data = _section(data, "detection")
    detection = DetectionConfig(
        min_lanes_per_fov=int(detection_data.get("min_lanes_per_fov", 0)),
        max_lanes_per_fov=int(detection_data.get("max_lanes_per_fov", 0)),
        min_trenches_per_lane=int(detection_data.get("min_trenches_per_lane", 1)),
        max_trenches_per_lane=int(detection_data.get("max_trenches_per_lane", 300)),
        lane_sigma=float(detection_data.get("lane_sigma", 40.0)),
        lane_distance=float(detection_data.get("lane_distance", 300.0)),
        lane_height=float(detection_data.get("lane_height", 5000.0)),
        lane_orientation_window_um=float(
            detection_data.get("lane_orientation_window_um", 20.0)
        ),
        trench_sigma=float(detection_data.get("trench_sigma", 4.0)),
        trench_distance=float(detection_data.get("trench_distance", 100.0)),
        trench_prominence=float(detection_data.get("trench_prominence", 10.0)),
        trench_length=int(detection_data.get("trench_length", 160)),
        trench_bottom_offset=int(detection_data.get("trench_bottom_offset", 50)),
        trench_width=int(detection_data.get("trench_width", 20)),
    )
    _require_lane_priors(detection)

    qc_data = _section(data, "qc")
    qc = QCConfig(
        min_lanes_per_fov=detection.min_lanes_per_fov,
        max_lanes_per_fov=detection.max_lanes_per_fov,
        max_drift_jump_px=float(qc_data.get("max_drift_jump_px", 8.0)),
        drift_jump_mad_multiplier=float(qc_data.get("drift_jump_mad_multiplier", 8.0)),
        max_trench_spacing_cv=float(qc_data.get("max_trench_spacing_cv", 0.35)),
        min_nonzero_fraction=float(qc_data.get("min_nonzero_fraction", 0.001)),
        max_saturated_fraction=float(qc_data.get("max_saturated_fraction", 0.30)),
        representative_trenches=int(qc_data.get("representative_trenches", 8)),
    )

    output_data = _section(data, "output")
    output = OutputConfig(
        diagnostics_dir=Path(output_data.get("diagnostics_dir", "diagnostics/autonomous")),
        output_path=(
            Path(output_data["output_path"]) if "output_path" in output_data else None
        ),
    )

    resource_data = _section(data, "resources")
    resources = ResourceConfig(
        registration_n_jobs=int(resource_data.get("registration_n_jobs", 1)),
        show_progress=bool(resource_data.get("show_progress", True)),
    )

    return AutonomousConfig(
        source=source,
        pipeline=pipeline,
        rotation=rotation,
        detection=detection,
        qc=qc,
        output=output,
        resources=resources,
    )
