from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from pymmm.lane_detector import LaneInfo
from pymmm.trench_detector import TrenchDetector


def _make_detector() -> TrenchDetector:
    image = np.ones((200, 160), dtype=np.float32)
    experiment = SimpleNamespace(channel_names=["PC"], fov_names=["f0"])
    registrator = SimpleNamespace(
        get_registered_mean_of_timestack=lambda fov, channel: image
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
        "pymmm.trench_detector.find_peaks",
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
