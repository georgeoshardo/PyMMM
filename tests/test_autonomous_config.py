from __future__ import annotations

from pathlib import Path

import pytest

from pymmm.autonomous.config import load_config


def test_load_config_reads_filled_preflight_template(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[source]
source_nd2 = "/data/example.nd2"

[pipeline]
registration_channel = "PC"
detection_channel = "PC"
registration_mode = "mean"

[rotation]
mode = "fixed"
fixed_degrees = 0.0

[detection]
min_lanes_per_fov = 2
max_lanes_per_fov = 2
min_trenches_per_lane = 45
max_trenches_per_lane = 60
lane_sigma = 40.0
lane_distance = 300.0
lane_height = 5000.0

[qc]
max_drift_jump_px = 8.0

[output]
diagnostics_dir = "diagnostics/example_autonomous"

[resources]
registration_n_jobs = 1
show_progress = false
""".strip()
    )

    config = load_config(config_path)

    assert config.source.source_nd2 == Path("/data/example.nd2")
    assert config.pipeline.registration_channel == "PC"
    assert config.rotation.fixed_degrees == 0.0
    assert config.detection.min_lanes_per_fov == 2
    assert config.detection.max_trenches_per_lane == 60
    assert config.output.diagnostics_dir == Path("diagnostics/example_autonomous")
    assert config.resources.registration_n_jobs == 1
    assert config.resources.show_progress is False


def test_load_config_rejects_unfilled_lane_priors(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[source]
source_nd2 = "/data/example.nd2"

[detection]
min_lanes_per_fov = 0
max_lanes_per_fov = 0
""".strip()
    )

    with pytest.raises(ValueError, match="lane priors"):
        load_config(config_path)
