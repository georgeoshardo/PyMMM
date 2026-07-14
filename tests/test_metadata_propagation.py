from __future__ import annotations

import json
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from pymmm.checkpoint import CompanionStore
from pymmm.experiment import ND2Experiment
from pymmm.extractor import Extractor
from pymmm.metadata import (
    SOURCE_METADATA_VERSION,
    extract_acquisition_metadata,
    extract_source_frame_metadata,
)


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


@dataclass
class FakeStagePosition:
    x: float
    y: float
    z: float


@dataclass
class FakePosition:
    stagePositionUm: FakeStagePosition
    pfsOffset: float | None
    name: str | None


@dataclass
class FakeTimeStamp:
    absoluteJulianDayNumber: float
    relativeTimeMs: float


@dataclass
class FakeFrameChannel:
    channel: FakeChannelMeta
    loops: dict[str, int]
    microscope: dict[str, str]
    volume: dict[str, list[float]]
    position: FakePosition
    time: FakeTimeStamp


@dataclass
class FakeFrameMetadata:
    contents: dict[str, int]
    channels: list[FakeFrameChannel]


class FakeND2File:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.sizes = {"T": 2, "P": 2, "C": 2, "Y": 3, "X": 4}
        self.shape = (2, 2, 2, 3, 4)
        self.dtype = np.dtype("uint16")
        self.is_legacy = False
        self.metadata = FakeMetadata(
            channels=[
                FakeChannelMeta(name="PC", index=0),
                FakeChannelMeta(name="mCherry", index=1),
            ]
        )
        self.experiment = [FakeLoop(count=2, type="TimeLoop")]
        attributes_type = namedtuple(
            "Attributes",
            ["bitsPerComponentInMemory", "channelCount"],
        )
        self.attributes = attributes_type(bitsPerComponentInMemory=16, channelCount=2)
        self.text_info = {"date": "07/03/2026 16:30:05", "optics": "Plan Apo 40x"}
        self.custom_data = {
            "AppInfo_V1_0": {"SWNameString": "NIS-Elements AR"},
            "GrabberCameraSettingsV1_0": {
                "GrabberCameraSettings": {"CameraUserName": "Fusion"}
            },
            "CustomDataV2_0": {
                "CustomTagDescription_v1.0": {"Tag0": {"ID": "Camera_ExposureTime1"}}
            },
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

    def frame_metadata(self, seq_index: int) -> FakeFrameMetadata:
        time_idx, p_idx = np.unravel_index(seq_index, (self.sizes["T"], self.sizes["P"]))
        rel_time_ms = 30000.0 * time_idx + 123.0 + p_idx
        abs_jdn = 2460000.0 + 0.01 * time_idx + 0.001 * p_idx
        stage_x = 100.0 + p_idx
        stage_y = 200.0 + time_idx
        stage_z = 300.0 + time_idx + p_idx
        channels = [
            FakeFrameChannel(
                channel=FakeChannelMeta(name="PC", index=0),
                loops={"TimeLoop": int(time_idx), "XYPosLoop": int(p_idx)},
                microscope={"objectiveName": "Plan Apo 40x"},
                volume={"axesCalibration": [0.108, 0.108, 1.0]},
                position=FakePosition(
                    stagePositionUm=FakeStagePosition(stage_x, stage_y, stage_z),
                    pfsOffset=10.0 + p_idx,
                    name=f"XYPos:{p_idx}",
                ),
                time=FakeTimeStamp(abs_jdn, rel_time_ms),
            ),
            FakeFrameChannel(
                channel=FakeChannelMeta(name="mCherry", index=1),
                loops={"TimeLoop": int(time_idx), "XYPosLoop": int(p_idx)},
                microscope={"objectiveName": "Plan Apo 40x"},
                volume={"axesCalibration": [0.108, 0.108, 1.0]},
                position=FakePosition(
                    stagePositionUm=FakeStagePosition(stage_x, stage_y, stage_z),
                    pfsOffset=10.0 + p_idx,
                    name=f"XYPos:{p_idx}",
                ),
                time=FakeTimeStamp(abs_jdn, rel_time_ms),
            ),
        ]
        return FakeFrameMetadata(contents={"frameCount": 4}, channels=channels)

    def read_frame(self, seq_index: int) -> np.ndarray:
        time_idx, p_idx = np.unravel_index(seq_index, (self.sizes["T"], self.sizes["P"]))
        base = np.arange(12, dtype=np.uint16).reshape(3, 4) + (100 * time_idx) + (10 * p_idx)
        return np.stack([base, base + 1000], axis=0)

    def events(self, orient: str = "records", null_value=np.nan) -> list[dict]:
        assert orient == "records"
        return [
            {"Time [s]": 0.1, "Events": "Command Executed"},
            {"Index": 0, "T Index": 0, "P Index": 0, "Position Name": "XYPos:0"},
            {"Time [s]": 0.2, "Events": "Command Executed"},
            {"Index": 1, "T Index": 0, "P Index": 1, "Position Name": "XYPos:1"},
            {"Time [s]": 30.1, "Events": "Command Executed"},
            {"Index": 2, "T Index": 1, "P Index": 0, "Position Name": "XYPos:0"},
            {"Time [s]": 30.2, "Events": "Command Executed"},
            {"Index": 3, "T Index": 1, "P Index": 1, "Position Name": "XYPos:1"},
        ]


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
    assert bundle["custom"]["data"]["StreamDataV1_0"]["ignored"] is True
    assert bundle["custom"]["unstructured"]["ImageEventsLV"]["RLxExperimentRecord"] == {}
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
        source_acquisition_metadata={
            "schema_version": SOURCE_METADATA_VERSION,
            "structured": {"text_info": {"date": "today"}},
        },
        source_subset_metadata={"active_fov_names": ["XYPos:0"]},
    )

    with zarr.config.set({"default_zarr_format": 2}):
        store = CompanionStore.for_experiment(experiment)
    attrs = dict(zarr.open_group(str(store.path), mode="r").attrs)

    assert (store.path / "zarr.json").exists()
    assert not (store.path / ".zgroup").exists()
    assert attrs["source_metadata_version"] == SOURCE_METADATA_VERSION
    assert attrs["source_acquisition_metadata"]["structured"]["text_info"]["date"] == "today"
    assert attrs["source_subset_metadata"]["active_fov_names"] == ["XYPos:0"]


def test_extract_source_frame_metadata_tracks_subset(monkeypatch, tmp_path):
    nd2_path = tmp_path / "subset_source.nd2"
    nd2_path.write_bytes(b"nd2")

    data = xr.DataArray(
        np.zeros((2, 2, 2, 3, 4), dtype=np.uint16),
        dims=("T", "P", "C", "Y", "X"),
        coords={
            "T": [0.0, 30000.0],
            "P": ["XYPos:0", "XYPos:1"],
            "C": ["PC", "mCherry"],
            "Y": [0.0, 0.108, 0.216],
            "X": [0.0, 0.108, 0.216, 0.324],
        },
        attrs={"metadata": {"placeholder": True}},
    )

    monkeypatch.setattr("pymmm.experiment.nd2.imread", lambda *args, **kwargs: data)
    monkeypatch.setattr("pymmm.metadata.nd2.ND2File", FakeND2File)
    monkeypatch.setattr(
        "pymmm.experiment.extract_acquisition_metadata",
        lambda *args, **kwargs: {"schema_version": SOURCE_METADATA_VERSION},
    )

    exp = ND2Experiment(nd2_path)
    exp.select_fovs(["XYPos:1"]).select_times(1, 2)

    exported = extract_source_frame_metadata(exp)

    assert exported["original_p_index"].tolist() == [1]
    assert exported["original_t_index"].tolist() == [1]
    assert exported["source_seq_index"].tolist() == [[3]]
    assert exported["relative_time_ms"].tolist() == [[30124.0]]
    assert exported["stage_position_um"].tolist() == [[[101.0, 201.0, 302.0]]]
    assert exported["position_name"].tolist() == [["XYPos:1"]]
    assert "mCherry" in exported["frame_metadata_json"][0, 0]
    assert len(exported["frame_event_records"]) == 1
    assert len(exported["acquisition_event_records"]) == 4


def test_extractor_writes_xarray_native_store(monkeypatch, tmp_path):
    nd2_path = tmp_path / "source.nd2"
    nd2_path.write_bytes(b"nd2")
    output_path = tmp_path / "output.zarr"

    data = xr.DataArray(
        np.zeros((2, 2, 2, 3, 4), dtype=np.uint16),
        dims=("T", "P", "C", "Y", "X"),
        coords={
            "T": [0.0, 30000.0],
            "P": ["XYPos:0", "XYPos:1"],
            "C": ["PC", "mCherry"],
            "Y": [0.0, 0.108, 0.216],
            "X": [0.0, 0.108, 0.216, 0.324],
        },
        attrs={"metadata": {"placeholder": True}},
    )

    monkeypatch.setattr("pymmm.experiment.nd2.imread", lambda *args, **kwargs: data)
    monkeypatch.setattr("pymmm.metadata.nd2.ND2File", FakeND2File)
    monkeypatch.setattr(
        "pymmm.experiment.extract_acquisition_metadata",
        lambda *args, **kwargs: {
            "schema_version": SOURCE_METADATA_VERSION,
            "structured": {"text_info": {"optics": "40x"}},
        },
    )

    experiment = ND2Experiment(nd2_path)
    experiment.select_fovs(["XYPos:1"]).select_times(1, 2)
    registrator = SimpleNamespace(
        channel="PC",
        mode="first",
        rotation=0,
        tmats=xr.DataArray(
            np.eye(3, dtype=np.float64)[np.newaxis, np.newaxis, :, :],
            dims=("P", "T", "row", "col"),
            coords={
                "P": ["XYPos:1"],
                "T": [30000.0],
                "row": [0, 1, 2],
                "col": [0, 1, 2],
            },
        ),
    )
    trench_table = pd.DataFrame(
        [
            {
                "trench_id": 7,
                "fov": "XYPos:1",
                "lane_index": 0,
                "x_left": 1,
                "x_right": 3,
                "y_top": 0,
                "y_bottom": 2,
                "orientation": 1,
                "needs_flip": False,
            },
            {
                "trench_id": 9,
                "fov": "XYPos:1",
                "lane_index": 1,
                "x_left": 0,
                "x_right": 2,
                "y_top": 1,
                "y_bottom": 3,
                "orientation": -1,
                "needs_flip": True,
            },
        ]
    )
    trench_detector = SimpleNamespace(get_trench_table=lambda: trench_table.copy())
    extractor = Extractor(experiment, registrator, trench_detector, output_path=output_path)

    extractor.extract(compressor="zstd", clevel=1, show_progress=False)

    assert (output_path / ".zmetadata").exists()

    root = xr.open_zarr(output_path, consolidated=True)
    tree = xr.open_datatree(output_path, engine="zarr", consolidated=True)

    assert root["data"].dims == ("Trench", "T", "C", "Y", "X")
    assert root["data"].shape == (2, 1, 2, 2, 2)
    assert root.coords["Trench"].values.tolist() == [7, 9]
    assert root.coords["T"].values.tolist() == [1]
    assert root.coords["C"].values.tolist() == ["PC", "mCherry"]
    assert root["fov_name"].compute().values.tolist() == ["XYPos:1", "XYPos:1"]
    assert root["fov_index"].compute().values.tolist() == [0, 0]
    assert root["original_p_index"].compute().values.tolist() == [1, 1]
    assert root["relative_time_ms"].compute().values.tolist() == [[30124.0], [30124.0]]
    assert root["stage_position_um"].compute().values.tolist() == [
        [[101.0, 201.0, 302.0]],
        [[101.0, 201.0, 302.0]],
    ]
    assert "trench_mapping" not in root.attrs
    assert "source_acquisition_metadata" not in root.attrs
    assert root.attrs["source_subset_metadata"]["active_timepoint_count"] == 1

    np.testing.assert_array_equal(
        root["data"].sel(Trench=7).isel(T=0, C=0).compute().values,
        np.asarray([[111, 112], [115, 116]], dtype=np.uint16),
    )
    np.testing.assert_array_equal(
        root["data"].sel(Trench=7).isel(T=0, C=1).compute().values,
        np.asarray([[1111, 1112], [1115, 1116]], dtype=np.uint16),
    )
    np.testing.assert_array_equal(
        root["data"].sel(Trench=9).isel(T=0, C=0).compute().values,
        np.asarray([[118, 119], [114, 115]], dtype=np.uint16),
    )

    assert set(tree.children) == {"acquisition", "events", "source_frames"}

    source_frames = tree["source_frames"].ds
    assert (
        source_frames["relative_time_ms"]
        .sel(FOV="XYPos:1", T=1)
        .compute()
        .item()
        == 30124.0
    )
    assert (
        source_frames["position_name"]
        .sel(FOV="XYPos:1", T=1)
        .compute()
        .item()
        == "XYPos:1"
    )
    assert "mCherry" in (
        source_frames["frame_metadata_json"]
        .sel(FOV="XYPos:1", T=1)
        .compute()
        .item()
    )

    events = tree["events"].ds
    assert events["frame_index"].compute().values.tolist() == [3]
    assert events["frame_t_index"].compute().values.tolist() == [1]
    assert events["frame_position_name"].compute().values.tolist() == ["XYPos:1"]
    assert events["acquisition_events"].compute().values.tolist() == [
        "Command Executed",
        "Command Executed",
        "Command Executed",
        "Command Executed",
    ]

    acquisition = tree["acquisition"].ds
    assert acquisition.attrs["text_info"]["optics"] == "40x"
    assert acquisition.coords["C"].values.tolist() == ["PC", "mCherry"]
    assert acquisition["raw_json"].compute().item().startswith("{")
