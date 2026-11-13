"""``pymmm.steps.extraction``: one OME-Zarr image per trench in the experiment group, with
the acquisition's and the registration's records, and resuming."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
from pymmm.source.nd2 import ND2Experiment
from pymmm.steps.extraction import Extractor
from pymmm.store.provenance import SCHEMA_VERSION

from fake_experiment import fake_nd2


def test_extractor_writes_one_ome_zarr_image_per_trench(monkeypatch, tmp_path):
    nd2_path = fake_nd2(monkeypatch, tmp_path / "source.nd2")
    output_path = tmp_path / "output.zarr"

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
    extractor = Extractor(
        experiment, registrator, trench_detector, output_path=output_path
    )

    extractor.extract(compressor="zstd", clevel=1, show_progress=False)

    assert (output_path / "zarr.json").exists()
    assert not (output_path / ".zgroup").exists()
    assert not (output_path / ".zmetadata").exists()
    assert "consolidated_metadata" not in json.loads(
        (output_path / "zarr.json").read_text()
    )

    zarr_root = zarr.open_group(output_path, mode="r", zarr_format=3)
    assert zarr_root.metadata.zarr_format == 3
    attrs = zarr_root.attrs["pymmm"]
    assert attrs["schema_version"] == SCHEMA_VERSION
    assert attrs["layout"] == "experiment"
    assert attrs["provenance"]["libraries"]["zarr"] == zarr.__version__
    assert "created" in attrs["provenance"]
    assert attrs["source_subset_metadata"]["active_timepoint_count"] == 1
    assert attrs["source_file_size_bytes"] == nd2_path.stat().st_size
    assert attrs["source_mtime_ns"] == nd2_path.stat().st_mtime_ns
    assert attrs["channels"] == ["PC", "mCherry"]
    assert attrs["pixel_size_um"] == pytest.approx(0.108)
    assert attrs["completed_fovs"] == ["XYPos:1"]
    assert attrs["extraction_complete"] is True
    assert "tmats" not in json.dumps(attrs)
    assert (
        type(zarr_root["trenches"]["fov_name"].metadata.data_type).__name__
        == "VariableLengthUTF8"
    )
    assert (
        type(zarr_root["acquisition"]["raw_json"].metadata.data_type).__name__
        == "VariableLengthUTF8"
    )

    tree = xr.open_datatree(output_path, engine="zarr", consolidated=False)
    assert set(tree.children) >= {
        "acquisition",
        "detection",
        "events",
        "registration",
        "source_frames",
        "trenches",
    }
    assert tree.ds.coords["T"].values.tolist() == [1]
    assert tree.ds.coords["C"].values.tolist() == ["PC", "mCherry"]
    table = tree["trenches"].ds
    assert table.coords["trench"].values.tolist() == [7, 9]
    assert table["path"].values.tolist() == ["trench_0007", "trench_0009"]
    assert table["fov_name"].values.tolist() == ["XYPos:1", "XYPos:1"]
    assert table["fov_index"].values.tolist() == [0, 0]
    assert table["x_left"].attrs["units"] == "pixel"
    assert table["needs_flip"].values.tolist() == [False, True]

    # one OME-Zarr image per trench, and nothing else at the top
    assert sorted(k for k in zarr_root.group_keys() if k.startswith("trench_")) == [
        "trench_0007",
        "trench_0009",
    ]
    image = zarr_root["trench_0007"]
    ome = image.attrs["ome"]
    assert ome["version"] == "0.5"
    (multiscale,) = ome["multiscales"]
    assert [a["name"] for a in multiscale["axes"]] == ["t", "c", "y", "x"]
    assert [a["type"] for a in multiscale["axes"]] == [
        "time",
        "channel",
        "space",
        "space",
    ]
    (dataset,) = multiscale["datasets"]
    assert dataset["path"] == "0"
    scale = dataset["coordinateTransformations"][0]["scale"]
    assert scale[1] == 1.0 and scale[2:] == pytest.approx([0.108, 0.108])
    channels = ome["omero"]["channels"]
    assert [c["label"] for c in channels] == ["PC", "mCherry"]
    assert [c["color"] for c in channels] == ["FFFFFF", "FF0000"]
    assert [c["active"] for c in channels] == [True, False]
    assert (
        channels[1]["window"]["start"] >= 1111 and channels[1]["window"]["end"] <= 1116
    )
    row = image.attrs["pymmm"]
    assert row["trench"] == 7 and row["fov_name"] == "XYPos:1" and row["x_left"] == 1
    assert zarr_root["trench_0009"].attrs["pymmm"]["needs_flip"] is True

    array = image["0"]
    assert array.shape == (1, 2, 2, 2)
    assert array.chunks == (1, 1, 2, 2)
    assert array.shards == (1, 2, 2, 2)
    assert array.metadata.dimension_names == ("t", "c", "y", "x")
    blosc = array.compressors[-1]
    assert isinstance(blosc, zarr.codecs.BloscCodec)
    assert getattr(blosc.cname, "value", blosc.cname) == "zstd"
    assert blosc.clevel == 1
    assert getattr(blosc.shuffle, "value", blosc.shuffle) == "shuffle"
    # one shard file, its key flat
    assert sorted(p.name for p in (output_path / "trench_0007" / "0").iterdir()) == [
        "c.0.0.0.0",
        "zarr.json",
    ]

    np.testing.assert_array_equal(
        array[0, 0],
        np.asarray([[111, 112], [115, 116]], dtype=np.uint16),
    )
    np.testing.assert_array_equal(
        array[0, 1],
        np.asarray([[1111, 1112], [1115, 1116]], dtype=np.uint16),
    )
    np.testing.assert_array_equal(
        zarr_root["trench_0009/0"][0, 0],
        np.asarray([[118, 119], [114, 115]], dtype=np.uint16),
    )

    registration = tree["registration"].ds
    assert registration["tmats"].dims == ("FOV", "T", "Row", "Col")
    assert registration.coords["FOV"].values.tolist() == ["XYPos:1"]
    assert registration.coords["T"].values.tolist() == [1]
    np.testing.assert_array_equal(
        registration["tmats"].sel(FOV="XYPos:1", T=1).values,
        np.eye(3),
    )
    assert registration["shift"].sel(FOV="XYPos:1", T=1).values.tolist() == [0.0, 0.0]
    assert registration.attrs["mode"] == "first"

    source_frames = tree["source_frames"].ds
    assert (
        source_frames["relative_time"].sel(FOV="XYPos:1", T=1).compute().item()
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
    assert (
        type(zarr_root["acquisition"]["raw_json"].metadata.codecs[-1]).__name__
        == "BloscCodec"
    )

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
    # starting over replaces an experiment group, but nothing else
    registrator.tmats = registrator.tmats.copy(data=np.eye(3)[np.newaxis, np.newaxis])
    extractor.extract(compressor="zstd", clevel=1, show_progress=False, resume=False)
    assert (
        zarr.open_group(output_path, mode="r").attrs["pymmm"]["extraction_complete"]
        is True
    )
    other = tmp_path / "other.trenches.zarr"
    zarr.open_group(str(other), mode="w", zarr_format=3).attrs["layout"] = (
        "trench_store"
    )
    elsewhere = Extractor(experiment, registrator, trench_detector, output_path=other)
    with pytest.raises(RuntimeError, match="never overwritten"):
        elsewhere.extract(
            compressor="zstd", clevel=1, show_progress=False, resume=False
        )
    with pytest.raises(RuntimeError, match="layout"):
        elsewhere.extract(compressor="zstd", clevel=1, show_progress=False)
    assert zarr.open_group(str(other), mode="r").attrs["layout"] == "trench_store"


def test_registration_dataset_stores_per_fov_shifts():
    tmats = np.tile(np.eye(3), (2, 3, 1, 1))
    tmats[1, 2, 0, 2] = 4.5  # x shift, XYPos:1, third timepoint
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
