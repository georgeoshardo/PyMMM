from __future__ import annotations

import json
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

from pymmm.checkpoint import CompanionStore
from pymmm.experiment import ND2Experiment
from pymmm.extractor import Extractor
from pymmm.provenance import SCHEMA_VERSION
from pymmm.lane_detector import LaneDetector, LaneInfo
from pymmm.registrator import Registrator
from pymmm.trench_detector import TrenchDefinition, TrenchDetector
from pymmm.metadata import (
    extract_acquisition_metadata,
    _median_frame_interval_ms,
    extract_source_frame_metadata,
)


@dataclass
class FakeChannelMeta:
    name: str
    index: int
    color: dict | None = None
    emissionLambdaNm: float | None = None
    excitationLambdaNm: float | None = None


@dataclass
class FakeChannelRecord:
    channel: FakeChannelMeta
    microscope: dict


@dataclass
class FakeMetadata:
    channels: list[FakeChannelRecord]


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
        microscope = {
            "objectiveName": "Plan Apo 40x",
            "objectiveNumericalAperture": 0.95,
            "modalityFlags": ["camera"],
        }
        self.metadata = FakeMetadata(
            channels=[
                FakeChannelRecord(
                    channel=FakeChannelMeta(
                        name="PC", index=0, color={"r": 255, "g": 255, "b": 255},
                        emissionLambdaNm=595.0,
                    ),
                    microscope=microscope,
                ),
                FakeChannelRecord(
                    channel=FakeChannelMeta(
                        name="mCherry", index=1, color={"r": 255, "g": 0, "b": 0},
                        emissionLambdaNm=475.0,
                    ),
                    microscope=microscope,
                ),
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

    def ome_metadata(self):
        planes = [
            SimpleNamespace(the_c=0, exposure_time=80.0),
            SimpleNamespace(the_c=1, exposure_time=200.0),
        ]
        return SimpleNamespace(
            images=[SimpleNamespace(pixels=SimpleNamespace(planes=planes))]
        )

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
                    pfsOffset=None,
                    name=None,
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
                    pfsOffset=None,
                    name=None,
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
            # Like real ND2 files: PFS lives in the frame rows, not frame metadata.
            {"Index": 0, "T Index": 0, "P Index": 0, "PFS Offset": 17390, "PFS Status": 1},
            {"Time [s]": 0.2, "Events": "Command Executed"},
            {"Index": 1, "T Index": 0, "P Index": 1, "PFS Offset": 17398, "PFS Status": 1},
            {"Time [s]": 30.1, "Events": "Command Executed"},
            {"Index": 2, "T Index": 1, "P Index": 0, "PFS Offset": 17406, "PFS Status": 1},
            {"Time [s]": 30.2, "Events": "Command Executed"},
            {"Index": 3, "T Index": 1, "P Index": 1, "PFS Offset": 17414, "PFS Status": 0},
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
    assert bundle["file_summary"]["pixel_size_um"] == 0.108
    assert bundle["metadata"]["channels"][0]["channel"]["name"] == "PC"
    assert bundle["custom_data"]["StreamDataV1_0"]["ignored"] is True
    assert bundle["unstructured_metadata"]["ImageEventsLV"]["RLxExperimentRecord"] == {}
    assert bundle["channel_exposure_ms"] == [80.0, 200.0]


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
    monkeypatch.setattr("pymmm.metadata.nd2.ND2File", FakeND2File)

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
    for name, value in dict(
        channel="PC", backend="stackreg", mode="first", rotation=0.0,
        roi={"y": (250, 860)}, mean_n_frames=10, mean_from="end", downsample=2,
        batch_size=4, fov_window=2, opencv_threads=1,
    ).items():
        setattr(registrator, name, value)
    registrator._tmats = xr.DataArray(
        tmats, dims=("P", "T", "row", "col"),
        coords={"P": experiment.fov_names, "T": [0.0, 60000.0, 120000.0],
                "row": [0, 1, 2], "col": [0, 1, 2]},
    )
    registrator._mean_images = xr.DataArray(
        np.ones((2, 4, 5)), dims=("P", "Y", "X"), coords={"P": experiment.fov_names},
    )
    registrator.save()

    lanes = LaneDetector(experiment, registrator, store, detection_channel="PC")
    lanes._lanes = {
        "XYPos:0": [LaneInfo(y_center=880, orientation=1, fov="XYPos:0", lane_index=0)],
        "XYPos:1": [],
    }
    lanes._detection_params = {"sigma": 3.0}
    lanes.save()

    trenches = TrenchDetector(experiment, registrator, lanes, store, detection_channel="PC")
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

    loaded = Registrator.load(experiment, store, load_mean_images=True)
    np.testing.assert_array_equal(loaded.tmats.sel(P="XYPos:1").values, tmats[1])
    assert loaded.tmats.coords["T"].values.tolist() == [0.0, 60000.0, 120000.0]
    assert loaded.mean_images.dims == ("P", "Y", "X")
    assert loaded.downsample == 2

    loaded_lanes = LaneDetector.load(experiment, loaded, store)
    assert loaded_lanes._lanes == lanes._lanes
    assert loaded_lanes._detection_params == {"sigma": 3.0}

    loaded_trenches = TrenchDetector.load(experiment, loaded, loaded_lanes, store)
    assert loaded_trenches._trenches == trenches._trenches
    assert loaded_trenches._detection_params["lane_params"] == {0: {"sigma": 2}}


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

    exp = ND2Experiment(nd2_path)
    exp.select_fovs(["XYPos:1"]).select_times(1, 2)

    exported = extract_source_frame_metadata(exp)

    assert exported["original_p_index"].tolist() == [1]
    assert exported["original_t_index"].tolist() == [1]
    assert exported["source_seq_index"].tolist() == [[3]]
    assert exported["relative_time"].tolist() == [[30124.0]]
    assert exported["stage_position"].tolist() == [[[101.0, 201.0, 302.0]]]
    assert exported["pfs_offset"].tolist() == [[17414.0]]
    assert exported["pfs_status"].tolist() == [[0.0]]
    assert "position_name" not in exported
    static = json.loads(exported["frame_metadata_json"][0])
    assert static["channels"][1]["channel"]["name"] == "mCherry"
    assert "time" not in static["channels"][0]
    assert "stagePositionUm" not in static["channels"][0]["position"]
    assert "frame_event_records" not in exported
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

    assert (output_path / "zarr.json").exists()
    assert not (output_path / ".zgroup").exists()
    assert not (output_path / ".zmetadata").exists()
    assert "consolidated_metadata" not in json.loads(
        (output_path / "zarr.json").read_text()
    )

    zarr_root = zarr.open_group(output_path, mode="r", zarr_format=3)
    assert zarr_root.metadata.zarr_format == 3
    assert (
        type(zarr_root["fov_name"].metadata.data_type).__name__
        == "VariableLengthUTF8"
    )
    assert (
        type(zarr_root["acquisition"]["raw_json"].metadata.data_type).__name__
        == "VariableLengthUTF8"
    )
    data_codec = zarr_root["data"].metadata.codecs[-1]
    assert isinstance(data_codec, zarr.codecs.BloscCodec)
    assert getattr(data_codec.cname, "value", data_codec.cname) == "zstd"
    assert data_codec.clevel == 1
    assert getattr(data_codec.shuffle, "value", data_codec.shuffle) == "shuffle"

    root = xr.open_zarr(output_path, consolidated=False)
    tree = xr.open_datatree(output_path, engine="zarr", consolidated=False)

    assert root["data"].dims == ("Trench", "T", "C", "Y", "X")
    assert root["data"].shape == (2, 1, 2, 2, 2)
    assert root.coords["Trench"].values.tolist() == [7, 9]
    assert root.coords["T"].values.tolist() == [1]
    assert root.coords["C"].values.tolist() == ["PC", "mCherry"]
    assert root["fov_name"].compute().values.tolist() == ["XYPos:1", "XYPos:1"]
    assert root["fov_index"].compute().values.tolist() == [0, 0]
    per_fov = {
        "original_p_index", "source_seq_index", "relative_time",
        "julian_date", "pfs_offset", "stage_position",
    }
    assert not per_fov & set(root.data_vars)
    assert root["x_left"].attrs["units"] == "pixel"
    assert "registration_params" not in root.attrs
    assert "pixel_size_um" not in root.attrs
    assert "trench_mapping" not in root.attrs
    assert "source_acquisition_metadata" not in root.attrs
    assert root.attrs["schema_version"] == SCHEMA_VERSION
    assert root.attrs["layout"] == "trench_store"
    assert root.attrs["provenance"]["libraries"]["zarr"] == zarr.__version__
    assert "created" in root.attrs["provenance"]
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

    assert set(tree.children) == {
        "acquisition", "detection", "events", "registration", "source_frames",
    }
    assert "tmats" not in json.dumps(dict(root.attrs))
    assert "extraction_fingerprint" not in root.attrs
    assert root.attrs["source_file_size_bytes"] == nd2_path.stat().st_size
    assert root.attrs["source_mtime_ns"] == nd2_path.stat().st_mtime_ns

    registration = tree["registration"].ds
    assert registration["tmats"].dims == ("FOV", "T", "Row", "Col")
    assert registration.coords["FOV"].values.tolist() == ["XYPos:1"]
    assert registration.coords["T"].values.tolist() == [1]
    np.testing.assert_array_equal(
        registration["tmats"].sel(FOV="XYPos:1", T=1).values, np.eye(3),
    )
    assert registration["shift"].sel(FOV="XYPos:1", T=1).values.tolist() == [0.0, 0.0]
    assert registration.attrs["mode"] == "first"

    source_frames = tree["source_frames"].ds
    assert (
        source_frames["relative_time"]
        .sel(FOV="XYPos:1", T=1)
        .compute()
        .item()
        == 30124.0
    )
    assert source_frames["pfs_offset"].sel(FOV="XYPos:1", T=1).item() == 17414.0
    assert source_frames["pfs_status"].sel(FOV="XYPos:1", T=1).item() == 0.0
    assert "position_name" not in source_frames
    assert source_frames["relative_time"].attrs["units"] == "millisecond"
    assert source_frames["stage_position"].attrs["units"] == "micrometer"
    assert source_frames["frame_metadata_json"].dims == ("FOV",)
    assert "mCherry" in source_frames["frame_metadata_json"].sel(FOV="XYPos:1").item()

    events = tree["events"].ds
    assert "FrameEvent" not in events.sizes
    assert events["acquisition_events"].compute().values.tolist() == [
        "Command Executed",
        "Command Executed",
        "Command Executed",
        "Command Executed",
    ]

    acquisition = tree["acquisition"].ds
    assert acquisition.coords["C"].values.tolist() == ["PC", "mCherry"]
    assert acquisition["exposure_time"].values.tolist() == [80.0, 200.0]
    assert acquisition["exposure_time"].attrs["units"] == "millisecond"
    assert acquisition["color"].values.tolist() == ["#FFFFFF", "#FF0000"]
    assert acquisition["pixel_size"].item() == pytest.approx(0.108)
    assert acquisition.attrs["objective_name"] == "Plan Apo 40x"
    assert "text_info" not in acquisition.attrs
    raw = json.loads(acquisition["raw_json"].compute().item())
    assert raw["text_info"]["optics"] == "Plan Apo 40x"
    assert type(zarr_root["acquisition"]["raw_json"].metadata.codecs[-1]).__name__ == "BloscCodec"

    # Resuming with the same transforms is a no-op; different transforms are refused.
    extractor.extract(compressor="zstd", clevel=1, show_progress=False)
    moved = trench_table.copy()
    moved.loc[0, ["x_left", "x_right"]] = [0, 2]
    trench_detector.get_trench_table = lambda: moved.copy()
    with pytest.raises(RuntimeError, match="trench x_left, trench x_right"):
        extractor.extract(compressor="zstd", clevel=1, show_progress=False)
    with pytest.raises(RuntimeError, match="compression settings"):
        trench_detector.get_trench_table = lambda: trench_table.copy()
        extractor.extract(compressor="zstd", clevel=3, show_progress=False)
    shifted = np.eye(3, dtype=np.float64)
    shifted[1, 2] = 1.0
    registrator.tmats = registrator.tmats.copy(data=shifted[np.newaxis, np.newaxis])
    with pytest.raises(RuntimeError, match="registration transforms"):
        extractor.extract(compressor="zstd", clevel=1, show_progress=False)


def test_registration_dataset_stores_per_fov_shifts():
    tmats = np.tile(np.eye(3), (2, 3, 1, 1))
    tmats[1, 2, 0, 2] = 4.5   # x shift, XYPos:1, third timepoint
    tmats[1, 2, 1, 2] = -2.0  # y shift
    registrator = SimpleNamespace(
        channel="PC",
        mode="first",
        rotation=0,
        roi={"y": (250, 860)},
        tmats=xr.DataArray(
            tmats,
            dims=("P", "T", "row", "col"),
            coords={"P": ["XYPos:0", "XYPos:1"], "T": [0.0, 1.0, 2.0]},
        ),
    )
    extractor = Extractor(
        SimpleNamespace(path=Path("source.nd2"), experiment_name="source"),
        registrator,
        SimpleNamespace(),
    )

    registration = extractor._build_registration_dataset(
        {"fov_names": ["XYPos:1"], "original_t_index": np.asarray([5, 6, 7])}
    )

    assert registration["shift"].sel(FOV="XYPos:1", T=7).values.tolist() == [-2.0, 4.5]
    assert registration["shift"].sel(FOV="XYPos:1", T=5).values.tolist() == [0.0, 0.0]
    assert json.loads(registration.attrs["roi"]) == {"y": [250, 860]}


def test_time_interval_is_median_of_real_frame_gaps():
    rows = [{"Time [s]": 0.0, "Events": "Command Executed"}]
    for t, (t0, t1) in enumerate([(0.0, 1.0), (60.0, 61.5), (121.0, 122.0), (180.0, 181.0)]):
        rows += [
            {"Index": 2 * t, "T Index": t, "P Index": 0, "Time [s]": t0},
            {"Index": 2 * t + 1, "T Index": t, "P Index": 1, "Time [s]": t1},
        ]
    # FOV 0 gaps: 60, 61, 59 s; FOV 1 gaps: 60.5, 60.5, 59 s
    assert _median_frame_interval_ms(rows) == pytest.approx(60250.0)
    assert _median_frame_interval_ms(rows[:1]) is None


def test_lane_orientation_can_be_forced():
    # Brighter below the lane, so "auto" infers trenches below (open up, -1).
    image = np.zeros((200, 10))
    image[95:105] = 10.0
    image[105:150] = 2.0
    experiment = SimpleNamespace(pixel_size_um=1.0, channel_names=["PC"])
    detector = LaneDetector(experiment, None, None, detection_channel="PC")

    def orientations(orientation):
        *_, lanes = detector._detect_lanes_single_fov(
            image, 0, 50, 5, 20, "XYPos:0", orientation=orientation,
        )
        return [lane.orientation for lane in lanes]

    assert orientations("auto") == [-1]
    assert orientations("down") == [1]
    assert orientations("up") == [-1]
