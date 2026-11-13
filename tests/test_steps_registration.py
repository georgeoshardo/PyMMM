"""Registration settings (pymmm.steps.registration) that need no ND2: the frames a
detection image averages, and the modes offered."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pymmm.steps.registration import DETECTION_MEAN, Registrator, mean_frames


def test_detection_frames_spread_over_the_movie_or_from_one_end():
    assert mean_frames(361, 5, "spread") == [0, 90, 180, 270, 360]
    assert mean_frames(361, 3, "start") == [0, 1, 2]
    assert mean_frames(361, 3, "end") == [358, 359, 360]
    assert mean_frames(10, 50, "spread") == list(range(10))  # no more than there are
    assert mean_frames(10, 0, "end") == [9]  # at least one
    assert len(mean_frames(361, **DETECTION_MEAN)) == 20
    with pytest.raises(ValueError, match="mean_from"):
        mean_frames(10, 3, "middle")


def test_registering_to_a_mean_of_frames_is_no_longer_offered():
    experiment = SimpleNamespace(channel_names=["PC"])
    with pytest.raises(ValueError, match="no longer offered"):
        Registrator(experiment, None, "PC", mode="mean")
    for mode in ("first", "last", "previous", 3):
        assert Registrator(experiment, None, "PC", mode=mode).mode == mode
