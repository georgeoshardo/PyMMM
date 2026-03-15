from __future__ import annotations

import json
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import xarray as xr
import zarr

from pymmm.checkpoint import CompanionStore
from pymmm.experiment import ND2Experiment
from pymmm.extractor import Extractor
from pymmm.metadata import SOURCE_METADATA_VERSION, extract_acquisition_metadata


@dataclass
class FakeChannelMeta:
    name: str
    index: int


@dataclass
class FakeMetadata:
    channels: list[FakeChannelMeta]


@dataclass
class FakeLoop:
    count: int
    type: str


class FakeND2File:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.sizes = {"T": 2, "P": 2, "C": 2, "Y": 3, "X": 4}
        self.shape = (2, 2, 2, 3, 4)
        self.dtype = np.dtype("uint16")
        self.is_legacy = False
        self.metadata = FakeMetadata(
            channels=[FakeChannelMeta(name="PC", index=0), FakeChannelMeta(name="mCherry", index=1)]
        )
        self.experiment = [FakeLoop(count=2, type="TimeLoop")]
        attributes_type = namedtuple("Attributes", ["bitsPerComponentInMemory", "channelCount"])
        self.attributes = attributes_type(bitsPerComponentInMemory=16, channelCount=2)
        self.text_info = {"date": "07/03/2026 16:30:05", "optics": "Plan Apo 40x"}
        self.custom_data = {
            "AppInfo_V1_0": {"SWNameString": "NIS-Elements AR"},
            "GrabberCameraSettingsV1_0": {"GrabberCameraSettings": {"CameraUserName": "Fusion"}},
            "CustomDataV2_0": {"CustomTagDescription_v1.0": {"Tag0": {"ID": "Camera_ExposureTime1"}}},
            "StreamDataV1_0": {"ignored": True},
        }

    def __enter__(self) -> "FakeND2File":
        return self

    def __exit__(self, *exc) -> None:
        return None

    def voxel_size(self):
        voxel_type = namedtuple("VoxelSize", ["x", "y", "z"])
        return voxel_type(0.108, 0.108, 1.0)

    def unstructured_metadata(self) -> dict[str, dict]:
        return {
            "ImageMetadataLV": {"SLxExperiment": {}},
            "ImageEventsLV": {"RLxExperimentRecord": {}},
        }


def test_extract_acquisition_metadata_is_json_serializable(monkeypatch, tmp_path):
    nd2_path = tmp_path / "sample.nd2"
    nd2_path.write_bytes(b"nd2")
    monkeypatch.setattr("pymmm.metadata.nd2.ND2File", FakeND2File)

    bundle = extract_acquisition_metadata(
        nd2_path,
        pixel_size_um=0.108,
        time_interval_ms=30000.0,
    )

    json.dumps(bundle)
    assert bundle["schema_version"] == SOURCE_METADATA_VERSION
    assert bundle["file_summary"]["pixel_size_um"] == 0.108
    assert bundle["structured"]["metadata"]["channels"][0]["name"] == "PC"
    assert bundle["custom"]["custom_data_keys"] == [
        "AppInfo_V1_0",
        "CustomDataV2_0",
        "GrabberCameraSettingsV1_0",
        "StreamDataV1_0",
    ]
    assert "ImageEventsLV" in bundle["custom"]["unstructured_metadata_keys"]
    assert "StreamDataV1_0" not in bundle["custom"]["selected"]


def test_nd2experiment_subset_metadata_tracks_selection(monkeypatch, tmp_path):
    nd2_path = tmp_path / "subset.nd2"
    nd2_path.write_bytes(b"nd2")

    data = xr.DataArray(
        np.zeros((2, 2, 2, 2, 2), dtype=np.uint16),
        dims=("T", "P", "C", "Y", "X"),
        coords={
            "T": [0.0, 30000.0],
            "P": ["XYPos:0", "XYPos:0"],
            "C": ["PC", "mCherry"],
            "Y": [0.0, 0.108],
            "X": [0.0, 0.108],
        },
        attrs={"metadata": {"placeholder": True}},
    )

    monkeypatch.setattr("pymmm.experiment.nd2.imread", lambda *args, **kwargs: data)
    monkeypatch.setattr(
        "pymmm.experiment.extract_acquisition_metadata",
        lambda *args, **kwargs: {"schema_version": SOURCE_METADATA_VERSION},
    )

    exp = ND2Experiment(nd2_path)
    exp.select_times(0, 1)
    exp.select_roi(y=(0, 1), x=(1, 2))
    exp.discard_fovs([exp.fov_names[1]])

    subset = exp.source_subset_metadata

    assert subset["raw_fov_names"] == ["XYPos:0", "XYPos:0"]
    assert subset["active_fov_names"] == ["XYPos:0-01"]
    assert subset["discarded_fov_names"] == ["XYPos:0-02"]
    assert subset["time_slice"] == {"start": 0, "stop": 1, "step": None}
    assert subset["roi"]["y"] == {"start": 0, "stop": 1, "step": None}
    assert subset["roi"]["x"] == {"start": 1, "stop": 2, "step": None}
    assert subset["raw_timepoint_count"] == 2
    assert subset["active_timepoint_count"] == 1


def test_companion_store_persists_source_metadata(tmp_path):
    nd2_path = tmp_path / "source.nd2"
    nd2_path.write_bytes(b"nd2")
    experiment = SimpleNamespace(
        path=nd2_path,
        experiment_name="source",
        fov_names=["XYPos:0"],
        channel_names=["PC", "mCherry"],
        source_metadata_version=SOURCE_METADATA_VERSION,
        source_acquisition_metadata={"schema_version": SOURCE_METADATA_VERSION, "structured": {"text_info": {"date": "today"}}},
        source_subset_metadata={"active_fov_names": ["XYPos:0"]},
    )

    store = CompanionStore.for_experiment(experiment)
    attrs = dict(zarr.open_group(str(store.path), mode="r").attrs)

    assert attrs["source_metadata_version"] == SOURCE_METADATA_VERSION
    assert attrs["source_acquisition_metadata"]["structured"]["text_info"]["date"] == "today"
    assert attrs["source_subset_metadata"]["active_fov_names"] == ["XYPos:0"]


def test_extractor_writes_source_metadata_to_output(tmp_path):
    nd2_path = tmp_path / "source.nd2"
    nd2_path.write_bytes(b"nd2")
    output_path = tmp_path / "output.zarr"

    experiment = SimpleNamespace(
        path=nd2_path,
        experiment_name="source",
        pixel_size_um=0.108,
        source_metadata_version=SOURCE_METADATA_VERSION,
        source_acquisition_metadata={"schema_version": SOURCE_METADATA_VERSION, "structured": {"text_info": {"optics": "40x"}}},
        source_subset_metadata={"active_timepoint_count": 10},
    )
    registrator = SimpleNamespace(channel="PC", mode="first", rotation=0)
    trench_detector = SimpleNamespace()
    extractor = Extractor(experiment, registrator, trench_detector, output_path=output_path)

    store = zarr.open_group(str(output_path), mode="w")
    extractor._write_store_metadata(
        store,
        trench_h=20,
        trench_w=8,
        n_trenches=3,
        n_times=10,
        channel_names=["PC", "mCherry"],
        trench_mapping=[{"trench_id": 0, "fov": "XYPos:0"}],
    )

    attrs = dict(zarr.open_group(str(output_path), mode="r").attrs)
    assert attrs["source_metadata_version"] == SOURCE_METADATA_VERSION
    assert attrs["source_acquisition_metadata"]["structured"]["text_info"]["optics"] == "40x"
    assert attrs["source_subset_metadata"]["active_timepoint_count"] == 10
    assert attrs["trench_mapping"][0]["fov"] == "XYPos:0"
