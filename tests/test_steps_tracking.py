"""``pymmm.steps.tracking``: the Tracker's command lines, what it reads back from the
group, resuming, and a chaintrack that fails.

chaintrack itself is not PyMMM's: these tests run a stand-in ``chain.py`` that notes
its command line and writes the records and folders the real one would.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from fake_experiment import H, N_T, TRENCHES, W, make_experiment

from pymmm.config import Experiment
from pymmm.steps.tracking import Tracker, trench_ranges
from pymmm.store import experiment_group as eg

STAND_IN = '''
import json, sys
from pathlib import Path

import numpy as np

from pymmm.store import experiment_group as eg

args = sys.argv[1:]
option = lambda flag: args[args.index(flag) + 1]
store = Path(option("--store"))
log = store.parent / "calls.json"
log.write_text(json.dumps([*(json.loads(log.read_text()) if log.exists() else []), args]))
trenches = []
for a in args[1:args.index("--store")]:
    lo, _, hi = a.partition("-")
    trenches += range(int(lo), int(hi or lo) + 1)
root = eg.open_experiment(store, mode="r+")
if (store.parent / "fail").exists():
    print("chain.py: no")
    sys.exit(3)
if args[0] == "fit":
    eg.update_experiment(root, "fits", option("--fit"), {"values": {"growth_per_min": 0.03}, "trenches": trenches})
    print("fit recorded")
else:
    eg.update_experiment(root, "tracks", option("--out"), {"fit": {"name": option("--fit")}})
    for t in trenches:
        root.require_group(f"{eg.trench_path(t)}/tracks/{option('--out')}.geff")
        eg.write_label_image(root, t, f"tracks.{option('--out')}", np.zeros((%d, %d, %d), np.uint16))
        print(f"trench {t}: tracked")
''' % (N_T, H, W)


@pytest.fixture
def segmented(tmp_path):
    """An experiment group whose trenches 10 and 11 are segmented (12 is not), a
    stand-in chain.py and a recipe file."""
    path = make_experiment(tmp_path)
    root = eg.open_experiment(path, mode="r+")
    eg.update_experiment(root, "segmentations", "seg", {"hypotheses": ["a", "b"]})
    blank = np.zeros((N_T, H, W), np.uint16)
    for trench in TRENCHES[:2]:
        eg.write_segmentation(root, trench, "seg", {"a": blank, "b": blank})
    (tmp_path / "chain.py").write_text(STAND_IN)
    (tmp_path / "recipe.toml").write_text("beam = 40\n")
    return path


def calls(path):
    return json.loads((path.parent / "calls.json").read_text())


def test_trench_ranges_join_neighbours():
    assert trench_ranges([7, 2, 3, 0, 5, 4, 3]) == ["0", "2-5", "7"]
    assert trench_ranges([]) == []


def test_tracker_needs_a_recorded_segmentation_and_a_chaintrack(segmented):
    with pytest.raises(ValueError, match="no segmentation 'other'"):
        Tracker(segmented, "other", segmented.parent / "chain.py")
    with pytest.raises(FileNotFoundError, match="no chaintrack"):
        Tracker(segmented, "seg", segmented.parent / "missing.py")


def test_fit_then_track_runs_chaintrack_on_the_group(segmented, capsys):
    tracker = Tracker(segmented, "seg", segmented.parent / "chain.py")
    assert tracker.fit_record is None and tracker.record is None
    assert tracker.segmented() == TRENCHES[:2] and tracker.tracked() == []
    with pytest.raises(RuntimeError, match="has no fit 'seg'"):
        tracker.track()

    values = tracker.fit(TRENCHES[:2], segmented.parent / "recipe.toml")
    assert values == {"growth_per_min": 0.03}
    assert tracker.fit_record["trenches"] == TRENCHES[:2]

    assert tracker.track() == TRENCHES[:2]          # every segmented trench, not trench 12
    assert not tracker.is_complete()
    assert tracker.record == {"fit": {"name": "seg"}}
    assert "trench 10: tracked" in capsys.readouterr().out   # chaintrack's lines are shown
    fit, run = calls(segmented)
    assert fit == ["fit", "10-11", "--store", str(segmented), "--seg", "seg", "--fit", "seg",
                   "--recipe", str(segmented.parent / "recipe.toml")]
    assert run == ["run", "10-11", "--store", str(segmented), "--seg", "seg", "--fit", "seg", "--out", "chain"]


def test_track_resumes_with_the_trenches_not_tracked_yet(segmented, capsys):
    tracker = Tracker(segmented, "seg", segmented.parent / "chain.py", run="other", fit="wide", workers=2)
    tracker.fit([10], segmented.parent / "recipe.toml")
    tracker.track([10])
    tracker.track()
    assert calls(segmented)[-1] == ["run", "11", "--store", str(segmented), "--seg", "seg", "--fit", "wide",
                                    "--out", "other", "--workers", "2"]
    n = len(calls(segmented))
    assert tracker.track() == TRENCHES[:2]
    assert len(calls(segmented)) == n and "tracked already" in capsys.readouterr().out
    tracker.track(resume=False)
    assert calls(segmented)[-1][:2] == ["run", "10-11"]


def test_a_failing_chaintrack_raises(segmented, capsys):
    tracker = Tracker(segmented, "seg", segmented.parent / "chain.py")
    (segmented.parent / "fail").touch()
    with pytest.raises(RuntimeError, match="exit status 3"):
        tracker.fit([10], segmented.parent / "recipe.toml")
    assert "chain.py: no" in capsys.readouterr().out
    with pytest.raises(FileNotFoundError, match="no recipe"):
        tracker.fit([10], segmented.parent / "missing.toml")


def test_experiment_builds_the_tracker_of_its_segmentation(segmented):
    (segmented.parent / "exp.toml").write_text(
        'name = "exp"\nstore = "exp.zarr"\n[segmentation]\nchannel = "PC"\nname = "seg"\n')
    tracker = Experiment.load(segmented.parent / "exp.toml").tracker(segmented.parent / "chain.py", run="r")
    assert (tracker.segmentation, tracker.run, tracker.store_path) == ("seg", "r", segmented)
