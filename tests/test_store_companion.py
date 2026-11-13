"""``pymmm.store.companion``: the companion store, and the steps' checkpoints in it."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import xarray as xr
import zarr
from pymmm.steps.lanes import LaneDetector, LaneInfo
from pymmm.steps.registration import Registrator
from pymmm.steps.trenches import TrenchDefinition, TrenchDetector
from pymmm.store.companion import CompanionStore
from pymmm.store.provenance import SCHEMA_VERSION


def _companion_experiment(tmp_path):
    nd2_path = tmp_path / "source.nd2"
    nd2_path.write_bytes(b"nd2")
    return SimpleNamespace(
        path=nd2_path,
        experiment_name="source",
        fov_names=["XYPos:0", "XYPos:1"],
        channel_names=["PC", "mCherry"],
        source_subset_metadata={"active_fov_names": ["XYPos:0", "XYPos:1"]},
    )


def test_companion_store_root_attrs_are_written_once(tmp_path):
    experiment = _companion_experiment(tmp_path)

    with zarr.config.set({"default_zarr_format": 2}):
        store = CompanionStore.for_experiment(experiment)
    attrs = dict(zarr.open_group(str(store.path), mode="r").attrs)

    assert (store.path / "zarr.json").exists()
    assert not (store.path / ".zgroup").exists()
    assert attrs["source_file_size_bytes"] == 3
    assert attrs["schema_version"] == SCHEMA_VERSION
    assert "pymmm_commit" in attrs["provenance"]
    assert "source_acquisition_metadata" not in attrs
    assert "fov_names" not in attrs

    experiment.fov_names = ["XYPos:0"]
    CompanionStore.for_experiment(experiment)
    assert dict(zarr.open_group(str(store.path), mode="r").attrs) == attrs


def test_companion_store_round_trips_checkpoints(tmp_path):
    experiment = _companion_experiment(tmp_path)
    store = CompanionStore.for_experiment(experiment)

    tmats = np.tile(np.eye(3), (2, 3, 1, 1))
    tmats[1, 2, 0, 2] = 2.5
    registrator = Registrator.__new__(Registrator)
    registrator.experiment = experiment
    registrator.store = store
    for name, value in {
        "channel": "PC",
        "backend": "stackreg",
        "mode": "first",
        "rotation": 0.0,
        "roi": {"y": (250, 860)},
        "downsample": 2,
        "batch_size": 4,
        "fov_window": 2,
        "opencv_threads": 1,
    }.items():
        setattr(registrator, name, value)
    registrator._tmats = xr.DataArray(
        tmats,
        dims=("P", "T", "row", "col"),
        coords={
            "P": experiment.fov_names,
            "T": [0.0, 60000.0, 120000.0],
            "row": [0, 1, 2],
            "col": [0, 1, 2],
        },
    )
    registrator.save()

    lanes = LaneDetector(experiment, registrator, store, detection_channel="PC")
    lanes._lanes = {
        "XYPos:0": [LaneInfo(y_center=880, orientation=1, fov="XYPos:0", lane_index=0)],
        "XYPos:1": [],
    }
    lanes._detection_params = {"sigma": 3.0}
    lanes.save()

    trenches = TrenchDetector(
        experiment, registrator, lanes, store, detection_channel="PC"
    )
    trenches._trenches = {
        "XYPos:0": [TrenchDefinition(5, "XYPos:0", 0, 10, 42, 700, 860, 1, False)],
        "XYPos:1": [],
    }
    trenches._detection_params = {"trench_width": 32, "lane_params": {0: {"sigma": 2}}}
    trenches.save()

    zarr_attrs = dict(zarr.open_group(str(store.path / "registration"), mode="r").attrs)
    assert zarr_attrs["complete"] is True
    assert "pymmm_commit" in zarr_attrs["provenance"]
    assert zarr_attrs["params"]["roi"] == {"y": [250, 860]}  # native JSON, not a string
    assert repr(store).endswith("[registration, lane_detection, trench_detection])")

    loaded = Registrator.load(experiment, store)
    np.testing.assert_array_equal(loaded.tmats.sel(P="XYPos:1").values, tmats[1])
    assert loaded.tmats.coords["T"].values.tolist() == [0.0, 60000.0, 120000.0]
    assert loaded.downsample == 2

    loaded_lanes = LaneDetector.load(experiment, loaded, store)
    assert loaded_lanes._lanes == lanes._lanes
    assert loaded_lanes._detection_params == {"sigma": 3.0}

    loaded_trenches = TrenchDetector.load(experiment, loaded, loaded_lanes, store)
    assert loaded_trenches._trenches == trenches._trenches
    assert loaded_trenches._detection_params["lane_params"] == {0: {"sigma": 2}}


def test_clear_drops_the_sections_present_and_leaves_the_rest(tmp_path, capsys):
    store = CompanionStore(tmp_path / "exp.pymmm.zarr")
    for name in ("registration", "lane_detection"):
        store._store.create_group(name).attrs["complete"] = True
    store.clear("lane_detection", "trench_detection")
    assert store.has_registration()
    assert not store.has_lane_detection()
    assert capsys.readouterr().out == ""
