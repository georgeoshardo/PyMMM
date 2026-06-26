from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import nd2
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class ChannelSummary:
    index: int
    name: str
    modality_flags: tuple[str, ...] = ()
    emission_lambda_nm: float | None = None
    excitation_lambda_nm: float | None = None
    objective_name: str | None = None
    objective_magnification: float | None = None
    objective_na: float | None = None
    zoom_magnification: float | None = None

    def to_builtin(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "modality_flags": list(self.modality_flags),
            "emission_lambda_nm": self.emission_lambda_nm,
            "excitation_lambda_nm": self.excitation_lambda_nm,
            "objective_name": self.objective_name,
            "objective_magnification": self.objective_magnification,
            "objective_na": self.objective_na,
            "zoom_magnification": self.zoom_magnification,
        }


@dataclass(frozen=True)
class TimeSummary:
    count: int
    nominal_period_ms: float | None = None
    period_avg_ms: float | None = None
    period_min_ms: float | None = None
    period_max_ms: float | None = None
    duration_ms: float | None = None

    def to_builtin(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "nominal_period_ms": self.nominal_period_ms,
            "period_avg_ms": self.period_avg_ms,
            "period_min_ms": self.period_min_ms,
            "period_max_ms": self.period_max_ms,
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True)
class MetadataSummary:
    path: Path
    file_size_gb: float
    sizes: dict[str, int]
    shape: tuple[int, ...]
    dtype: str
    channels: tuple[ChannelSummary, ...]
    pixel_size_um: float | None
    field_of_view_um: tuple[float | None, float | None]
    time: TimeSummary
    position_count: int
    event_count: int
    indexed_frame_event_count: int

    def to_builtin(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "file_size_gb": self.file_size_gb,
            "sizes": self.sizes,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "channels": [channel.to_builtin() for channel in self.channels],
            "pixel_size_um": self.pixel_size_um,
            "field_of_view_um": {
                "height": self.field_of_view_um[0],
                "width": self.field_of_view_um[1],
            },
            "time": self.time.to_builtin(),
            "position_count": self.position_count,
            "event_count": self.event_count,
            "indexed_frame_event_count": self.indexed_frame_event_count,
        }


@dataclass(frozen=True)
class FrameSample:
    t: int
    p: int
    c: int
    channel_name: str

    def to_builtin(self) -> dict[str, Any]:
        return {"T": self.t, "P": self.p, "C": self.c, "channel_name": self.channel_name}


@dataclass(frozen=True)
class SamplePlan:
    seed: int
    mixed: tuple[FrameSample, ...]
    pc_only: tuple[FrameSample, ...]
    same_position_all_channels: tuple[FrameSample, ...]

    def to_builtin(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "mixed": [sample.to_builtin() for sample in self.mixed],
            "pc_only": [sample.to_builtin() for sample in self.pc_only],
            "same_position_all_channels": [
                sample.to_builtin() for sample in self.same_position_all_channels
            ],
        }


@dataclass(frozen=True)
class PreflightResult:
    diagnostics_dir: Path
    metadata_summary_path: Path
    config_template_path: Path
    sampled_frame_stats_path: Path
    sampled_frames_path: Path
    png_paths: tuple[Path, ...]


def _none_if_missing(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def extract_metadata_summary(path: str | Path) -> MetadataSummary:
    nd2_path = Path(path)
    with nd2.ND2File(nd2_path) as nd2_file:
        sizes = {axis: int(size) for axis, size in nd2_file.sizes.items()}
        metadata = nd2_file.metadata
        attributes = nd2_file.attributes
        experiment = nd2_file.experiment
        events = nd2_file.events()

        channels = []
        for channel_meta in metadata.channels:
            channel = channel_meta.channel
            microscope = channel_meta.microscope
            channels.append(
                ChannelSummary(
                    index=int(channel.index),
                    name=str(channel.name),
                    modality_flags=tuple(str(flag) for flag in microscope.modalityFlags),
                    emission_lambda_nm=_none_if_missing(channel.emissionLambdaNm),
                    excitation_lambda_nm=_none_if_missing(channel.excitationLambdaNm),
                    objective_name=(
                        None
                        if microscope.objectiveName is None
                        else str(microscope.objectiveName)
                    ),
                    objective_magnification=_none_if_missing(
                        microscope.objectiveMagnification
                    ),
                    objective_na=_none_if_missing(microscope.objectiveNumericalAperture),
                    zoom_magnification=_none_if_missing(microscope.zoomMagnification),
                )
            )

        time_loop = next(
            (loop for loop in experiment if getattr(loop, "type", None) == "TimeLoop"),
            None,
        )
        time_summary = TimeSummary(count=sizes.get("T", 1))
        if time_loop is not None:
            params = time_loop.parameters
            period_diff = params.periodDiff
            time_summary = TimeSummary(
                count=sizes.get("T", 1),
                nominal_period_ms=_none_if_missing(params.periodMs),
                period_avg_ms=(
                    None if period_diff is None else _none_if_missing(period_diff.avg)
                ),
                period_min_ms=(
                    None if period_diff is None else _none_if_missing(period_diff.min)
                ),
                period_max_ms=(
                    None if period_diff is None else _none_if_missing(period_diff.max)
                ),
                duration_ms=_none_if_missing(params.durationMs),
            )

        pixel_size_um = None
        if channels:
            volume = metadata.channels[0].volume
            if volume.axesCalibration:
                pixel_size_um = float(volume.axesCalibration[0])

        fov_height_um = None
        fov_width_um = None
        if pixel_size_um is not None:
            fov_height_um = sizes.get("Y", 0) * pixel_size_um
            fov_width_um = sizes.get("X", 0) * pixel_size_um

        indexed_events = [event for event in events if "Index" in event]

        return MetadataSummary(
            path=nd2_path,
            file_size_gb=nd2_path.stat().st_size / 1024**3,
            sizes=sizes,
            shape=tuple(int(dim) for dim in nd2_file.shape),
            dtype=str(nd2_file.dtype),
            channels=tuple(channels),
            pixel_size_um=pixel_size_um,
            field_of_view_um=(fov_height_um, fov_width_um),
            time=time_summary,
            position_count=int(attributes.sequenceCount // max(sizes.get("T", 1), 1)),
            event_count=len(events),
            indexed_frame_event_count=len(indexed_events),
        )


def build_sample_plan(
    sizes: dict[str, int],
    channel_names: list[str],
    seed: int = 20260331,
    mixed_count: int = 12,
    pc_count: int = 9,
) -> SamplePlan:
    rng = random.Random(seed)
    n_t = int(sizes.get("T", 1))
    n_p = int(sizes.get("P", 1))
    n_c = int(sizes.get("C", len(channel_names) or 1))
    names = channel_names or [str(index) for index in range(n_c)]
    pc_index = names.index("PC") if "PC" in names else 0

    def sample_unique(count: int, channel: int | None = None) -> tuple[FrameSample, ...]:
        samples: list[FrameSample] = []
        seen: set[tuple[int, int, int]] = set()
        max_unique = n_t * n_p * (1 if channel is not None else n_c)
        target = min(count, max_unique)
        while len(samples) < target:
            t = rng.randrange(n_t)
            p = rng.randrange(n_p)
            c = channel if channel is not None else rng.randrange(n_c)
            key = (t, p, c)
            if key in seen:
                continue
            seen.add(key)
            samples.append(FrameSample(t=t, p=p, c=c, channel_name=names[c]))
        return tuple(samples)

    same_t = rng.randrange(n_t)
    same_p = rng.randrange(n_p)
    same_position = tuple(
        FrameSample(t=same_t, p=same_p, c=c, channel_name=names[c]) for c in range(n_c)
    )

    return SamplePlan(
        seed=seed,
        mixed=sample_unique(mixed_count),
        pc_only=sample_unique(pc_count, channel=pc_index),
        same_position_all_channels=same_position,
    )


def render_config_template(summary: MetadataSummary) -> str:
    channel_names = [channel.name for channel in summary.channels]
    registration_channel = "PC" if "PC" in channel_names else channel_names[0]
    detection_channel = registration_channel
    source = str(summary.path)
    sizes_toml = "{ " + ", ".join(
        f"{key} = {int(summary.sizes[key])}" for key in sorted(summary.sizes)
    ) + " }"
    pixel_size_um = (
        "0.0" if summary.pixel_size_um is None else repr(float(summary.pixel_size_um))
    )
    time_interval_ms = (
        "0.0"
        if summary.time.nominal_period_ms is None
        else repr(float(summary.time.nominal_period_ms))
    )

    return f"""# Draft PyMMM autonomous config generated by preflight.
# FILL IN after inspecting preflight PNGs.
# Keep this file next to the preflight diagnostics so the choices are auditable.

[source]
source_nd2 = "{source}"
pixel_size_um = {pixel_size_um}
time_interval_ms = {time_interval_ms}
channel_names = {json.dumps(channel_names)}
sizes = {sizes_toml}

[pipeline]
registration_channel = "{registration_channel}"
detection_channel = "{detection_channel}"
registration_mode = "mean"

[rotation]
# Use "fixed" only if you know the angle. Use "auto" for bounded search.
mode = "auto"
fixed_degrees = 0.0
search_min = -5.0
search_max = 5.0
coarse_step = 0.5
fine_step = 0.1
sample_fovs = 5
min_score_margin = 0.05
max_fov_disagreement_deg = 0.5

[detection]
# FILL IN after inspecting preflight PNGs.
min_lanes_per_fov = 0
max_lanes_per_fov = 0
min_trenches_per_lane = 0
max_trenches_per_lane = 0

# These are algorithm parameters, not biological priors. Leave blank values at
# the defaults for the first diagnostic pass unless the raw images make them
# obviously wrong.
lane_sigma = 40.0
lane_distance = 300.0
lane_height = 5000.0
trench_sigma = 4.0
trench_distance = 100.0
trench_prominence = 10.0
trench_length = 160
trench_bottom_offset = 50
trench_width = 20

[qc]
# Smooth long-term drift is allowed; this catches discrete registration jumps.
max_drift_jump_px = 8.0
drift_jump_mad_multiplier = 8.0
max_trench_spacing_cv = 0.35
min_nonzero_fraction = 0.001
max_saturated_fraction = 0.30
representative_trenches = 8

[output]
diagnostics_dir = "diagnostics/{summary.path.stem}_autonomous"
output_path = "{summary.path.stem}.trenches.zarr"
overwrite_verified_output = false
compressor = "zstd"
clevel = 9

[resources]
registration_n_jobs = 1
show_progress = true
"""


def write_config_template(output_dir: Path, summary: MetadataSummary) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{summary.path.stem}.autonomous-template.toml"
    path.write_text(render_config_template(summary))
    return path


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def _channel_names_from_array(array: Any) -> list[str]:
    if "C" in array.coords:
        return [str(value) for value in array.coords["C"].values]
    return [str(index) for index in range(array.sizes.get("C", 1))]


def _load_frame(array: Any, sample: FrameSample) -> np.ndarray:
    frame = array.isel(T=sample.t, P=sample.p, C=sample.c).compute().values
    return np.asarray(frame)


def _robust_limits(frame: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(frame, [0.5, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(frame))
        high = float(np.max(frame))
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def _frame_stats(sample: FrameSample, frame: np.ndarray) -> dict[str, Any]:
    return {
        "T": sample.t,
        "P": sample.p,
        "C": sample.c,
        "channel_name": sample.channel_name,
        "min": int(np.min(frame)),
        "max": int(np.max(frame)),
        "mean": float(np.mean(frame)),
        "p01": float(np.percentile(frame, 1)),
        "p50": float(np.percentile(frame, 50)),
        "p99": float(np.percentile(frame, 99)),
    }


def _write_panel(
    array: Any,
    samples: tuple[FrameSample, ...],
    path: Path,
    title: str,
    columns: int,
) -> list[dict[str, Any]]:
    rows = int(np.ceil(len(samples) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4 * columns, 4 * rows),
        constrained_layout=True,
    )
    axes_array = np.asarray(axes).reshape(-1)
    stats = []

    for ax, sample in zip(axes_array, samples):
        frame = _load_frame(array, sample)
        stats.append(_frame_stats(sample, frame))
        low, high = _robust_limits(frame)
        ax.imshow(frame, cmap="gray", vmin=low, vmax=high, interpolation="nearest")
        ax.set_title(f"T{sample.t} P{sample.p} {sample.channel_name}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes_array[len(samples) :]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return stats


def write_frame_diagnostics(
    nd2_path: str | Path,
    output_dir: Path,
    sample_plan: SamplePlan,
) -> tuple[Path, list[dict[str, Any]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    array = nd2.imread(str(nd2_path), dask=True, xarray=True)
    stats: list[dict[str, Any]] = []

    mixed_path = output_dir / "random_frames_mixed_panel.png"
    stats.extend(
        _write_panel(
            array,
            sample_plan.mixed,
            mixed_path,
            "Random mixed-channel frames, percentile scaled per panel",
            columns=4,
        )
    )

    pc_path = output_dir / "random_pc_frames_panel.png"
    stats.extend(
        _write_panel(
            array,
            sample_plan.pc_only,
            pc_path,
            "Random PC frames, percentile scaled per panel",
            columns=3,
        )
    )

    all_channels_path = output_dir / "same_time_position_all_channels.png"
    stats.extend(
        _write_panel(
            array,
            sample_plan.same_position_all_channels,
            all_channels_path,
            "Same time/position across channels, percentile scaled per channel",
            columns=len(sample_plan.same_position_all_channels),
        )
    )

    return mixed_path, stats


def run_preflight(
    nd2_path: str | Path,
    output_dir: str | Path | None = None,
    seed: int = 20260331,
) -> PreflightResult:
    path = Path(nd2_path)
    diagnostics_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path("diagnostics") / f"{path.stem}_preflight"
    )
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    summary = extract_metadata_summary(path)
    summary_path = diagnostics_dir / "metadata_summary.json"
    _write_json(summary_path, summary.to_builtin())

    channel_names = [channel.name for channel in summary.channels]
    sample_plan = build_sample_plan(summary.sizes, channel_names, seed=seed)
    sampled_frames_path = diagnostics_dir / "sampled_frames.json"
    _write_json(sampled_frames_path, sample_plan.to_builtin())

    _, stats = write_frame_diagnostics(path, diagnostics_dir, sample_plan)
    stats_path = diagnostics_dir / "sampled_frame_stats.json"
    _write_json(stats_path, stats)

    template_path = write_config_template(diagnostics_dir, summary)
    png_paths = (
        diagnostics_dir / "random_frames_mixed_panel.png",
        diagnostics_dir / "random_pc_frames_panel.png",
        diagnostics_dir / "same_time_position_all_channels.png",
    )

    return PreflightResult(
        diagnostics_dir=diagnostics_dir,
        metadata_summary_path=summary_path,
        config_template_path=template_path,
        sampled_frame_stats_path=stats_path,
        sampled_frames_path=sampled_frames_path,
        png_paths=png_paths,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pymmm-preflight")
    parser.add_argument("nd2_path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=20260331)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_preflight(args.nd2_path, output_dir=args.output_dir, seed=args.seed)
    print(f"Diagnostics: {result.diagnostics_dir}")
    print(f"Config template: {result.config_template_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
