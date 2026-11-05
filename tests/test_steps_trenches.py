from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.signal import find_peaks

from pymmm.steps.lanes import LaneInfo
from pymmm.steps.trenches import TrenchDetector


def _make_detector() -> TrenchDetector:
    image = np.ones((200, 160), dtype=np.float32)
    experiment = SimpleNamespace(channel_names=["PC"], fov_names=["f0"])
    registrator = SimpleNamespace(
        get_registered_mean_of_timestack=lambda fov, channel: image,
        compute_registered_means=lambda channel: {"f0": image},
    )
    lane_detector = SimpleNamespace(
        lanes={"f0": [LaneInfo(100, 1, "f0", 0)]}
    )
    return TrenchDetector(
        experiment, registrator, lane_detector, object(), "PC"
    )


@pytest.mark.parametrize("trench_width", [32, 33])
def test_explicit_trench_width_is_exact(monkeypatch, trench_width: int) -> None:
    monkeypatch.setattr(
        "pymmm.steps.trenches.find_peaks",
        lambda *args, **kwargs: (np.array([40, 80]), {}),
    )
    detector = _make_detector()

    detector.detect_trenches(
        sigma=0,
        distance=1,
        prominence=0,
        trench_width=trench_width,
        trench_length=60,
        trench_bottom_offset=10,
        save_diagnostics=False,
    )

    trenches = detector.trenches["f0"]
    assert {trench.x_right - trench.x_left for trench in trenches} == {
        trench_width
    }
    assert {trench.y_bottom - trench.y_top for trench in trenches} == {60}


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("trench_width", 28),
        ("trench_length", 50),
        ("shrink_scale", 2.0),
    ],
)
def test_per_lane_shape_overrides_are_rejected(parameter: str, value: float) -> None:
    detector = _make_detector()

    with pytest.raises(ValueError, match=parameter):
        detector.detect_trenches(lane_params={0: {parameter: value}})


def test_trench_width_is_explicit_with_default_32() -> None:
    parameters = inspect.signature(TrenchDetector.detect_trenches).parameters

    assert parameters["trench_width"].default == 32
    assert "shrink_scale" not in parameters


def _side_trench_profile(centres, filled, length=2300, seed=0):
    """Synthetic x-profile: two bright walls per trench plus a central channel
    that is bright when empty and dark when it holds cells."""
    x = np.arange(length, dtype=float)
    profile = 4000 + 0.3 * x  # uneven illumination
    bump = lambda c, w, a: a * np.exp(-0.5 * ((x - c) / w) ** 2)
    for centre, full in zip(centres, filled):
        profile += bump(centre - 28, 3, 1500) + bump(centre + 28, 3, 1500)
        profile += bump(centre, 4, -1200 if full else 2500)
    return profile + np.random.default_rng(seed).normal(0, 40, length)


def test_periodic_centres_survive_filled_side_trenches():
    from pymmm.steps.trenches import periodic_trench_centres

    true_centres = 40.3 + 155.0 * np.arange(15)
    filled = np.random.default_rng(1).random(15) < 0.6  # most trenches hold cells
    profile = _side_trench_profile(true_centres, filled)

    result = periodic_trench_centres(profile, trench_width=32)

    assert result["pitch"] == pytest.approx(155.0, abs=0.5)
    assert len(result["centres"]) == 15
    assert np.max(np.abs(result["centres"] - true_centres)) < 2

    # The tallest-peak detector lands on walls for the filled trenches.
    peaks, _ = find_peaks(profile, distance=100, prominence=10)
    offsets = np.min(np.abs(peaks[:, None] - true_centres[None, :]), axis=1)
    assert np.max(offsets) > 20


def test_periodic_detection_drops_low_symmetry_sites():
    from pymmm.steps.trenches import periodic_trench_centres

    true_centres = 40.3 + 155.0 * np.arange(15)
    profile = _side_trench_profile(np.delete(true_centres, 7), [False] * 14)

    result = periodic_trench_centres(profile, trench_width=32, min_symmetry=0.2)

    assert len(result["centres"]) == 14
    assert np.min(np.abs(result["centres"] - true_centres[7])) > 100


def test_profile_uses_trench_rows_only(monkeypatch):
    image = np.zeros((200, 160), dtype=np.float32)
    image[100:, 70:90] = 1000.0  # bright stripe only below the lane (far side)
    experiment = SimpleNamespace(channel_names=["PC"], fov_names=["f0"])
    lane_detector = SimpleNamespace(lanes={"f0": [LaneInfo(100, 1, "f0", 0)]})
    detector = TrenchDetector(experiment, None, lane_detector, object(), "PC")

    _, lane_profiles, _ = detector.detect_fov(
        image, "f0", sigma=0, distance=1, prominence=0, height=0,
        trench_width=20, trench_length=60, trench_bottom_offset=10,
    )

    # Trenches open down (orientation 1) sit above the lane, rows 30-90.
    assert np.all(lane_profiles[0]["profile"] == 0)
