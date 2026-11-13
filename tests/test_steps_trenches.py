from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
from pymmm.steps.lanes import LaneDetector, LaneInfo
from pymmm.steps.trenches import TrenchDetector
from scipy.signal import find_peaks

from fake_experiment import FOV_H, FOV_W, fov_experiment, fov_registrator


def _make_detector() -> TrenchDetector:
    image = np.ones((200, 160), dtype=np.float32)
    experiment = SimpleNamespace(channel_names=["PC"], fov_names=["f0"])
    registrator = SimpleNamespace(
        get_registered_mean_of_timestack=lambda fov, channel: image,
        compute_registered_means=lambda channel: {"f0": image},
    )
    lane_detector = SimpleNamespace(
        lanes={"f0": [LaneInfo(100, 1, "f0", 0)]}, image_edge=None
    )
    return TrenchDetector(experiment, registrator, lane_detector, object(), "PC")


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
    assert {trench.x_right - trench.x_left for trench in trenches} == {trench_width}
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

    def bump(c, w, a):
        return a * np.exp(-0.5 * ((x - c) / w) ** 2)

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
    lane_detector = SimpleNamespace(
        lanes={"f0": [LaneInfo(100, 1, "f0", 0)]}, image_edge=None
    )
    detector = TrenchDetector(experiment, None, lane_detector, object(), "PC")

    _, lane_profiles, _ = detector.detect_fov(
        image,
        "f0",
        sigma=0,
        distance=1,
        prominence=0,
        height=0,
        trench_width=20,
        trench_length=60,
        trench_bottom_offset=10,
    )

    # Trenches open down (orientation 1) sit above the lane, rows 30-90.
    assert np.all(lane_profiles[0]["profile"] == 0)


@pytest.mark.parametrize("opening, needs_flip", [("down", False), ("up", True)])
def test_image_edge_lanes_give_trenches_the_whole_image_height(
    tmp_path, opening, needs_flip
):
    image = np.full((FOV_H, FOV_W), 10.0, dtype=np.float32)
    image[:, 20:150:40] = 40.0  # four trenches, top to bottom of the cropped image
    experiment = fov_experiment(tmp_path, ["f0"])
    registrator = fov_registrator({"f0": image})
    lanes = LaneDetector(experiment, registrator, None, detection_channel="PC")
    lanes.use_image_edge(opening, save_diagnostics=False)
    detector = TrenchDetector(experiment, registrator, lanes, None, "PC")
    assert detector.shape_defaults() == {
        "trench_length": FOV_H,
        "trench_bottom_offset": 0,
    }

    detector.detect_trenches(
        sigma=0, distance=5, prominence=1, trench_width=6, save_diagnostics=False
    )
    trenches = detector.trenches["f0"]
    assert [(t.x_left + t.x_right) // 2 for t in trenches] == [20, 60, 100, 140]
    assert {(t.y_top, t.y_bottom, t.needs_flip) for t in trenches} == {
        (0, FOV_H, needs_flip)
    }
    params = detector.detection_params
    assert (params["trench_length"], params["trench_bottom_offset"]) == (FOV_H, 0)


def test_detected_lanes_keep_the_trench_shape_defaults():
    assert _make_detector().shape_defaults() == {
        "trench_length": 160,
        "trench_bottom_offset": 50,
    }


def _trench_rows_detector(lane_y: int, orientation: int) -> TrenchDetector:
    experiment = SimpleNamespace(channel_names=["PC"], fov_names=["f0"])
    lane_detector = SimpleNamespace(
        lanes={"f0": [LaneInfo(lane_y, orientation, "f0", 0)]}, image_edge=None
    )
    return TrenchDetector(experiment, None, lane_detector, object(), "PC")


def test_the_zero_fill_of_a_rotation_is_left_out_of_the_profile():
    """A rotated frame's empty corner (zeros) reads as no image, not as dark pixels,
    so the profile is the one the image would give without it."""
    image = np.full((200, 400), 1000.0)
    for x in range(20, 400, 40):
        image[:, x : x + 8] = 2000.0  # a trench every 40 px
    corner = image.copy()
    rows, cols = np.indices(corner.shape)
    corner[rows < 60 - cols // 3] = 0  # a wedge, as a rotation leaves at the top
    detector = _trench_rows_detector(lane_y=180, orientation=1)
    settings = {
        "sigma": 2,
        "distance": 1,
        "prominence": 0,
        "height": 0,
        "trench_width": 20,
        "trench_length": 160,
        "trench_bottom_offset": 10,
        "method": "periodic",
    }
    plain = detector.detect_fov(image, "f0", **settings)
    wedged = detector.detect_fov(corner, "f0", **settings)
    np.testing.assert_allclose(wedged[1][0]["profile"], plain[1][0]["profile"])
    assert wedged[1][0]["pitch"] == pytest.approx(40, abs=0.5)
    assert len(wedged[0]) == len(plain[0]) == 10


def test_a_line_with_no_image_takes_its_neighbours_value():
    from pymmm._utils import mean_without_fill

    image = np.array([[0.0, 4.0, 6.0], [0.0, 2.0, 0.0]])
    np.testing.assert_allclose(mean_without_fill(image, axis=0), [3.0, 3.0, 6.0])
