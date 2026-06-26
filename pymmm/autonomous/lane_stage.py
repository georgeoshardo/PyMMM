from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pymmm.checkpoint import CompanionStore
from pymmm.experiment import ND2Experiment
from pymmm.lane_detector import LaneDetector, LaneInfo
from pymmm.registrator import Registrator

from pymmm.autonomous.config import (
    AutonomousConfig,
    DetectionConfig,
    QCConfig,
    load_config,
)


@dataclass(frozen=True)
class LaneDiagnosticPaths:
    table_path: Path
    qc_path: Path
    overlay_grid_path: Path


@dataclass(frozen=True)
class LaneStageResult:
    diagnostics_dir: Path
    lane_table_path: Path
    lane_qc_path: Path
    overlay_grid_path: Path
    companion_store_path: Path
    passed: bool


def lane_table_from_lanes(lanes: dict[str, list[LaneInfo]]) -> pd.DataFrame:
    rows = []
    for fov, fov_lanes in lanes.items():
        for lane in fov_lanes:
            rows.append(
                {
                    "fov": str(fov),
                    "lane_index": int(lane.lane_index),
                    "y_center": int(lane.y_center),
                    "orientation": int(lane.orientation),
                }
            )
    return pd.DataFrame(rows, columns=["fov", "lane_index", "y_center", "orientation"])


def check_lane_counts(table: pd.DataFrame, config: QCConfig) -> dict[str, Any]:
    counts = table.groupby("fov").size().astype(int).to_dict()
    failed = [
        str(fov)
        for fov, count in counts.items()
        if count < config.min_lanes_per_fov or count > config.max_lanes_per_fov
    ]
    return {
        "passed": not failed,
        "min_lanes_per_fov": config.min_lanes_per_fov,
        "max_lanes_per_fov": config.max_lanes_per_fov,
        "counts_by_fov": {str(k): int(v) for k, v in counts.items()},
        "failed_fovs": failed,
    }


def _robust_limits(image: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(image, [0.5, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(image))
        high = float(np.max(image))
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def _write_lane_overlay_grid(
    output_path: Path,
    table: pd.DataFrame,
    mean_images: dict[str, np.ndarray],
) -> None:
    n = len(mean_images)
    columns = min(5, max(1, n))
    rows = int(np.ceil(n / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4 * columns, 3.5 * rows),
        constrained_layout=True,
    )
    axes_array = np.asarray(axes).reshape(-1)

    for ax, (fov, image) in zip(axes_array, mean_images.items()):
        low, high = _robust_limits(image)
        ax.imshow(image, cmap="gray", vmin=low, vmax=high, interpolation="nearest")
        fov_lanes = table[table["fov"] == fov]
        for _, row in fov_lanes.iterrows():
            y = float(row["y_center"])
            orientation = int(row["orientation"])
            ax.axhline(y, color="red", lw=1.0)
            ax.text(
                5,
                y - 8,
                f'{int(row["lane_index"])} ({orientation:+d})',
                color="red",
                fontsize=6,
                va="bottom",
            )
        ax.set_title(f"{fov}: {len(fov_lanes)} lanes", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes_array[n:]:
        ax.axis("off")

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_lane_diagnostics(
    diagnostics_dir: str | Path,
    table: pd.DataFrame,
    mean_images: dict[str, np.ndarray],
    qc: dict[str, Any],
    detection_config: DetectionConfig,
) -> LaneDiagnosticPaths:
    output_dir = Path(diagnostics_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table_path = output_dir / "lane_table.csv"
    qc_path = output_dir / "lane_detection_qc.json"
    overlay_path = output_dir / "lane_overlays_grid.png"

    table.to_csv(table_path, index=False)
    payload = {
        "qc": qc,
        "detection": {
            "lane_sigma": detection_config.lane_sigma,
            "lane_distance": detection_config.lane_distance,
            "lane_height": detection_config.lane_height,
            "lane_orientation_window_um": detection_config.lane_orientation_window_um,
            "min_lanes_per_fov": detection_config.min_lanes_per_fov,
            "max_lanes_per_fov": detection_config.max_lanes_per_fov,
        },
    }
    qc_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    _write_lane_overlay_grid(overlay_path, table, mean_images)

    return LaneDiagnosticPaths(
        table_path=table_path,
        qc_path=qc_path,
        overlay_grid_path=overlay_path,
    )


def _load_or_compute_registration(
    experiment: ND2Experiment,
    store: CompanionStore,
    config: AutonomousConfig,
) -> Registrator:
    if store.has_registration():
        return Registrator.load(experiment, store)

    if config.rotation.mode != "fixed":
        raise ValueError("lane detection currently requires [rotation].mode = \"fixed\"")

    registrator = Registrator(
        experiment=experiment,
        store=store,
        registration_channel=config.pipeline.registration_channel,
        mode=config.pipeline.registration_mode,
        rotation=config.rotation.fixed_degrees,
        mean_n_frames=config.pipeline.mean_n_frames,
        mean_from=config.pipeline.mean_from,
    )
    registrator.compute_mean_images(plot=False)
    registrator.compute_tmats(
        plot=False,
        n_jobs=config.resources.registration_n_jobs,
    )
    registrator.save()
    return registrator


def run_lane_detection(config: AutonomousConfig) -> LaneStageResult:
    experiment = ND2Experiment(config.source.source_nd2)
    store = CompanionStore.for_experiment(experiment)
    registrator = _load_or_compute_registration(experiment, store, config)

    detector = LaneDetector(
        experiment=experiment,
        registrator=registrator,
        store=store,
        detection_channel=config.pipeline.detection_channel,
    )
    detector.detect_lanes(
        sigma=config.detection.lane_sigma,
        distance=config.detection.lane_distance,
        height=config.detection.lane_height,
        orientation_window_um=config.detection.lane_orientation_window_um,
        plot=False,
    )
    detector.save()

    table = lane_table_from_lanes(detector.lanes)
    qc = check_lane_counts(table, config.qc)
    mean_images = {
        fov: registrator.get_registered_mean_of_timestack(
            fov=fov,
            channel=config.pipeline.detection_channel,
        )
        for fov in experiment.fov_names
    }
    paths = write_lane_diagnostics(
        config.output.diagnostics_dir,
        table,
        mean_images,
        qc,
        config.detection,
    )

    return LaneStageResult(
        diagnostics_dir=config.output.diagnostics_dir,
        lane_table_path=paths.table_path,
        lane_qc_path=paths.qc_path,
        overlay_grid_path=paths.overlay_grid_path,
        companion_store_path=store.path,
        passed=bool(qc["passed"]),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pymmm-detect-lanes")
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    result = run_lane_detection(config)
    print(f"Diagnostics: {result.diagnostics_dir}")
    print(f"Lane table: {result.lane_table_path}")
    print(f"Lane QC: {result.lane_qc_path}")
    print(f"Lane overlays: {result.overlay_grid_path}")
    print(f"Companion store: {result.companion_store_path}")
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
