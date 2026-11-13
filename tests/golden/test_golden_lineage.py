"""What each saved step was made from, on real data: the golden pipeline's steps are
current, redoing a step with the same result keeps what follows, and steps redone out
of order make what follows stale."""

from __future__ import annotations

import shutil

import pytest
from pymmm import CompanionStore, LaneDetector, Registrator, TrenchDetector
from real_data import REGISTRATION, TIMES, open_nd2

pytestmark = pytest.mark.golden


@pytest.fixture
def copy(pipeline, scratch_folder):
    """The golden pipeline's companion store and experiment group, in a scratch
    folder of their own, to redo steps in."""
    folder = scratch_folder("lineage")
    shutil.copytree(pipeline.store.path, folder / pipeline.store.path.name)
    shutil.copytree(pipeline.group, folder / "experiment.zarr")
    exp = open_nd2(folder)
    store = CompanionStore.for_experiment(exp)
    reg = Registrator.load(exp, store)
    lanes = LaneDetector.load(exp, reg, store)
    return exp, store, reg, lanes, folder / "experiment.zarr"


def test_the_golden_pipeline_is_current(pipeline):
    status = pipeline.store.lineage(pipeline.exp, pipeline.group)
    assert set(status.values()) == {"current"}
    assert pipeline.store.made_from("lane_detection") == {
        "registration": pipeline.reg.fingerprint()
    }


def test_registering_again_with_the_same_result_keeps_what_follows(copy):
    exp, store, _, _, group = copy
    reg = Registrator(exp, store, **REGISTRATION)
    reg.compute_tmats(n_jobs=1)
    reg.save()  # without clearing the lanes and trenches
    assert set(store.lineage(exp, group).values()) == {"current"}


def test_another_registration_makes_lanes_trenches_and_extraction_stale(copy):
    exp, store, _, _, group = copy
    reg = Registrator(exp, store, **{**REGISTRATION, "rotation": 1.0})
    reg.compute_tmats(n_jobs=1)
    reg.save()  # as a script might, without clearing what followed
    assert store.lineage(exp, group) == {
        "registration": "current",
        "lane_detection": "stale",
        "trench_detection": "stale",
        "extraction": "stale",
    }


def test_other_lanes_make_the_trenches_and_extraction_stale(copy):
    exp, store, reg, lanes, group = copy
    lanes.use_image_edge("down", save_diagnostics=False)
    lanes.save()
    assert store.lineage(exp, group) == {
        "registration": "current",
        "lane_detection": "current",
        "trench_detection": "stale",
        "extraction": "stale",
    }
    trenches = TrenchDetector(exp, reg, lanes, store, detection_channel="PC")
    trenches.detect_trenches(trench_width=94, save_diagnostics=False)
    trenches.save()
    status = store.lineage(exp, group)
    assert status["trench_detection"] == "current"
    assert status["extraction"] == "stale"  # cut with the old trenches


def test_other_frames_make_everything_stale(copy):
    _, store, _, _, group = copy
    fewer = open_nd2(group.parent).select_times(TIMES[0], TIMES[1] - 1)
    assert set(store.lineage(fewer, group).values()) == {"stale"}


def test_records_from_before_are_unrecorded_and_usable(copy):
    exp, store, *_ = copy
    for section in ("registration", "lane_detection", "trench_detection"):
        del store._store[section].attrs["made_from"]
    status = store.lineage(exp)
    assert status == {
        "registration": "unrecorded",
        "lane_detection": "unrecorded",
        "trench_detection": "unrecorded",
        "extraction": "missing",
    }
