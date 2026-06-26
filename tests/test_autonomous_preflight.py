from __future__ import annotations

import tomllib
from pathlib import Path

from pymmm.autonomous.preflight import (
    ChannelSummary,
    MetadataSummary,
    TimeSummary,
    build_sample_plan,
    render_config_template,
    write_config_template,
)


def _summary() -> MetadataSummary:
    return MetadataSummary(
        path=Path("/data/example.nd2"),
        file_size_gb=1.25,
        sizes={"T": 8, "P": 25, "C": 4, "Y": 2304, "X": 2304},
        shape=(8, 25, 4, 2304, 2304),
        dtype="uint16",
        channels=(
            ChannelSummary(index=0, name="PC", modality_flags=("brightfield", "camera")),
            ChannelSummary(index=1, name="CFP", modality_flags=("fluorescence", "camera")),
            ChannelSummary(index=2, name="mVenus", modality_flags=("fluorescence", "camera")),
            ChannelSummary(index=3, name="mCherry", modality_flags=("fluorescence", "camera")),
        ),
        pixel_size_um=0.107869821220548,
        field_of_view_um=(248.532, 248.532),
        time=TimeSummary(
            count=8,
            nominal_period_ms=300000.0,
            period_avg_ms=299992.7,
            period_min_ms=299883.4,
            period_max_ms=300061.1,
            duration_ms=28800000.0,
        ),
        position_count=25,
        event_count=408,
        indexed_frame_event_count=200,
    )


def test_build_sample_plan_is_deterministic_and_in_bounds() -> None:
    plan = build_sample_plan(
        sizes={"T": 8, "P": 25, "C": 4},
        channel_names=["PC", "CFP", "mVenus", "mCherry"],
        seed=123,
    )
    same_plan = build_sample_plan(
        sizes={"T": 8, "P": 25, "C": 4},
        channel_names=["PC", "CFP", "mVenus", "mCherry"],
        seed=123,
    )

    assert plan == same_plan
    assert len(plan.mixed) == 12
    assert len(plan.pc_only) == 9
    assert len(plan.same_position_all_channels) == 4
    assert {sample.c for sample in plan.same_position_all_channels} == {0, 1, 2, 3}
    for sample in [*plan.mixed, *plan.pc_only, *plan.same_position_all_channels]:
        assert 0 <= sample.t < 8
        assert 0 <= sample.p < 25
        assert 0 <= sample.c < 4


def test_render_config_template_contains_metadata_defaults_and_human_priors() -> None:
    text = render_config_template(_summary())

    assert 'registration_channel = "PC"' in text
    assert 'detection_channel = "PC"' in text
    assert "pixel_size_um = 0.107869821220548" in text
    assert "# FILL IN after inspecting preflight PNGs." in text
    assert "min_lanes_per_fov = 0" in text
    assert "max_lanes_per_fov = 0" in text
    assert "max_drift_step_px" not in text
    assert "max_drift_jump_px = 8.0" in text


def test_render_config_template_is_valid_toml() -> None:
    text = render_config_template(_summary())

    parsed = tomllib.loads(text)

    assert parsed["source"]["sizes"]["T"] == 8
    assert parsed["source"]["channel_names"] == ["PC", "CFP", "mVenus", "mCherry"]


def test_write_config_template_uses_nd2_stem(tmp_path: Path) -> None:
    path = write_config_template(tmp_path, _summary())

    assert path == tmp_path / "example.autonomous-template.toml"
    assert path.exists()
    assert "source_nd2 = \"/data/example.nd2\"" in path.read_text()
