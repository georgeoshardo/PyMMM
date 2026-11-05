from __future__ import annotations

import json

import numpy as np
import pytest
import zarr

from pymmm import trench_images as ti


@pytest.fixture
def experiment(tmp_path):
    """A two-trench experiment group as the Extractor lays it out (pixels left empty)."""
    path = tmp_path / "exp.zarr"
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs["pymmm"] = {"layout": "experiment", "pixel_size_um": 0.1, "frame_interval_ms": 60_000.0}
    root.create_array("C", shape=(2,), dtype=str)[:] = np.asarray(["PC", "mCherry"], dtype=object)
    table = root.create_group("trenches")
    table.create_array("trench", data=np.asarray([3, 4], dtype=np.int32))
    for t in (3, 4):
        root.create_group(ti.trench_path(t)).create_array("0", shape=(5, 2, 6, 4), dtype=np.uint16)
    return path


def test_open_experiment_refuses_other_groups(tmp_path):
    zarr.open_group(str(tmp_path / "other.zarr"), mode="w", zarr_format=3)
    with pytest.raises(ValueError, match="not a PyMMM experiment group"):
        ti.open_experiment(tmp_path / "other.zarr")


def test_segmentation_is_one_listed_label_image_with_a_hypothesis_axis(experiment):
    root = ti.open_experiment(experiment, mode="r+")
    assert ti.trench_ids(root) == [3, 4]
    assert ti.channel_index(root, "mCherry") == 1
    assert ti.listed_labels(root, 3) == []
    labels = np.arange(5 * 3 * 6 * 4, dtype=np.uint16).reshape(5, 3, 6, 4)
    ti.write_label_image(root, 3, "seg", labels, hypotheses=["a", "b", "c"], attrs={"channel": "PC"})

    assert ti.listed_labels(root, 3) == ["seg"]
    assert ti.listed_labels(root, 4) == []
    names, array = ti.segmentation(root, 3, "seg")
    assert names == ["a", "b", "c"]
    assert array.shape == (5, 3, 6, 4) and array.chunks == (5, 1, 6, 4) and array.shards == (5, 3, 6, 4)
    np.testing.assert_array_equal(array[:, 1], labels[:, 1])
    group = experiment / "trench_0003" / "labels" / "seg"
    assert sorted(p.name for p in (group / "0").iterdir()) == ["c.0.0.0.0", "zarr.json"]
    ome = json.loads((group / "zarr.json").read_text())["attributes"]["ome"]
    (multiscale,) = ome["multiscales"]
    assert [a["name"] for a in multiscale["axes"]] == ["t", "hypothesis", "y", "x"]
    assert multiscale["datasets"][0]["coordinateTransformations"][0]["scale"] == [1.0, 1.0, 0.1, 0.1]
    assert ome["image-label"] == {"source": {"image": "../../"}}
    with pytest.raises(KeyError):
        ti.segmentation(root, 4, "seg")
    with pytest.raises(ValueError):
        ti.write_label_image(root, 4, "seg", labels, hypotheses=["a", "b"])


def test_write_tracks_writes_a_geff_beside_its_masks(experiment):
    geff = pytest.importorskip("geff")

    root = ti.open_experiment(experiment, mode="r+")
    masks = np.zeros((5, 6, 4), dtype=np.uint16)
    masks[0, :2] = 1
    masks[1, :3] = 1
    nodes = {"t": np.asarray([0, 1], np.int32), "y": np.asarray([0.5, 1.0], np.float32),
             "x": np.asarray([1.5, 1.5], np.float32), "seg_id": np.asarray([1, 1], np.int32),
             "track_id": np.asarray([0, 0]), "lineage_id": np.asarray([0, 0])}
    path = ti.write_tracks(root, 4, "run", nodes, np.asarray([[0, 1]]), np.asarray([np.nan]), masks,
                           extra={"tracker": "test"})

    assert path == experiment / "trench_0004" / "tracks" / "run.geff"
    assert ti.has_tracks(root, 4, "run") and not ti.has_tracks(root, 3, "run")
    assert ti.listed_labels(root, 4) == ["tracks.run"]
    np.testing.assert_array_equal(root["trench_0004/labels/tracks.run/0"][...], masks)
    meta = json.loads((path / "zarr.json").read_text())["attributes"]["geff"]
    assert meta["related_objects"][0]["path"] == "../../labels/tracks.run"
    assert meta["related_objects"][0]["node_prop"] == "seg_id"
    assert meta["related_objects"][1] == {"type": "image", "path": "../..", "label_prop": None,
                                          "node_prop": None}
    assert meta["extra"]["trench"] == 4 and meta["extra"]["pixel_size_um"] == 0.1
    graph, _ = geff.read(str(path))
    assert graph.number_of_nodes() == 2 and list(graph.edges) == [(0, 1)]


def test_update_experiment_keeps_other_records(experiment):
    root = ti.open_experiment(experiment, mode="r+")
    ti.update_experiment(root, "segmentations", "a", {"channel": "PC"})
    ti.update_experiment(root, "segmentations", "b", {"channel": "mCherry"})
    attrs = dict(ti.open_experiment(experiment).attrs)["pymmm"]
    assert attrs["segmentations"] == {"a": {"channel": "PC"}, "b": {"channel": "mCherry"}}
    assert attrs["pixel_size_um"] == 0.1
