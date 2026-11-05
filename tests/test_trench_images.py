from __future__ import annotations

import json

import numpy as np
import pytest
import zarr

from pymmm.store import experiment_group as eg
from pymmm.store.geff import has_tracks, write_tracks


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
        root.create_group(eg.trench_path(t)).create_array("0", shape=(5, 2, 6, 4), dtype=np.uint16)
    return path


def test_open_experiment_refuses_other_groups(tmp_path):
    zarr.open_group(str(tmp_path / "other.zarr"), mode="w", zarr_format=3)
    with pytest.raises(ValueError, match="not a PyMMM experiment group"):
        eg.open_experiment(tmp_path / "other.zarr")


def test_segmentation_is_one_label_image_per_hypothesis_listed_together(experiment):
    root = eg.open_experiment(experiment, mode="r+")
    assert eg.trench_ids(root) == [3, 4]
    assert eg.channel_index(root, "mCherry") == 1
    with pytest.raises(KeyError):
        eg.hypotheses(root, "seg")
    eg.update_experiment(root, "segmentations", "seg", {"hypotheses": ["b", "a", "c"]})
    assert eg.listed_labels(root, 3) == []
    assert not eg.has_segmentation(root, 3, "seg")
    stack = np.arange(5 * 3 * 6 * 4, dtype=np.uint16).reshape(3, 5, 6, 4)
    eg.write_segmentation(root, 3, "seg", {"b": stack[0], "a": stack[1], "c": stack[2]},
                          attrs={"channel": "PC"})

    assert eg.listed_labels(root, 3) == ["seg.b", "seg.a", "seg.c"]
    assert eg.has_segmentation(root, 3, "seg") and not eg.has_segmentation(root, 4, "seg")
    names, arrays = eg.segmentation(root, 3, "seg")
    assert names == ["b", "a", "c"]
    for k, array in enumerate(arrays):
        assert array.shape == (5, 6, 4) and array.chunks == (5, 6, 4)
        np.testing.assert_array_equal(array[...], stack[k])
    group = experiment / "trench_0003" / "labels" / "seg.a"
    assert sorted(p.name for p in (group / "0").iterdir()) == ["c.0.0.0", "zarr.json"]
    attrs = json.loads((group / "zarr.json").read_text())["attributes"]
    (multiscale,) = attrs["ome"]["multiscales"]
    assert [a["name"] for a in multiscale["axes"]] == ["t", "y", "x"]
    assert multiscale["datasets"][0]["coordinateTransformations"][0]["scale"] == [1.0, 0.1, 0.1]
    assert attrs["ome"]["image-label"] == {"source": {"image": "../../"}}
    assert attrs["pymmm"] == {"channel": "PC", "segmentation": "seg", "hypothesis": "a",
                              "hypothesis_index": 1}
    with pytest.raises(KeyError):
        eg.segmentation(root, 4, "seg")
    with pytest.raises(ValueError, match="hypotheses"):
        eg.write_segmentation(root, 4, "seg", {"a": stack[0], "b": stack[1]})
    assert eg.listed_labels(root, 4) == []
    with pytest.raises(ValueError, match=r"not \(t, y, x\)"):
        eg.write_label_image(root, 4, "flat", stack)


def test_write_tracks_writes_a_geff_beside_its_masks(experiment):
    geff = pytest.importorskip("geff")

    root = eg.open_experiment(experiment, mode="r+")
    masks = np.zeros((5, 6, 4), dtype=np.uint16)
    masks[0, :2] = 1
    masks[1, :3] = 1
    nodes = {"t": np.asarray([0, 1], np.int32), "y": np.asarray([0.5, 1.0], np.float32),
             "x": np.asarray([1.5, 1.5], np.float32), "seg_id": np.asarray([1, 1], np.int32),
             "track_id": np.asarray([0, 0]), "lineage_id": np.asarray([0, 0])}
    path = write_tracks(root, 4, "run", nodes, np.asarray([[0, 1]]), np.asarray([np.nan]), masks,
                           extra={"tracker": "test"})

    assert path == experiment / "trench_0004" / "tracks" / "run.geff"
    assert has_tracks(root, 4, "run") and not has_tracks(root, 3, "run")
    assert eg.listed_labels(root, 4) == ["tracks.run"]
    np.testing.assert_array_equal(root["trench_0004/labels/tracks.run/0"][...], masks)
    meta = json.loads((path / "zarr.json").read_text())["attributes"]["geff"]
    assert meta["related_objects"][0]["path"] == "../../labels/tracks.run"
    assert meta["related_objects"][0]["node_prop"] == "seg_id"
    assert meta["related_objects"][1] == {"type": "image", "path": "../..", "label_prop": None,
                                          "node_prop": None}
    assert meta["extra"]["trench"] == 4 and meta["extra"]["pixel_size_um"] == 0.1
    graph, _ = geff.read(str(path))
    assert sorted(graph.nodes) == [1, 2] and list(graph.edges) == [(1, 2)]


def test_update_experiment_keeps_other_records(experiment):
    root = eg.open_experiment(experiment, mode="r+")
    eg.update_experiment(root, "segmentations", "a", {"channel": "PC"})
    eg.update_experiment(root, "segmentations", "b", {"channel": "mCherry"})
    attrs = dict(eg.open_experiment(experiment).attrs)["pymmm"]
    assert attrs["segmentations"] == {"a": {"channel": "PC"}, "b": {"channel": "mCherry"}}
    assert attrs["pixel_size_um"] == 0.1
    assert eg.records(eg.open_experiment(experiment), "segmentations") == attrs["segmentations"]
    assert eg.records(root, "tracks") == {}


def test_remove_segmentation_unlists_and_deletes_only_its_label_images(experiment):
    root = eg.open_experiment(experiment, mode="r+")
    eg.update_experiment(root, "segmentations", "seg", {"hypotheses": ["a", "b"]})
    stack = np.ones((5, 6, 4), dtype=np.uint16)
    eg.write_segmentation(root, 3, "seg", {"a": stack, "b": stack})
    eg.write_label_image(root, 3, "tracks.run", stack)
    eg.remove_segmentation(root, 4, "seg")  # nothing there: no error

    eg.remove_segmentation(root, 3, "seg")
    assert not eg.has_segmentation(root, 3, "seg")
    assert eg.listed_labels(root, 3) == ["tracks.run"]
    assert [name for name, _ in root["trench_0003/labels"].groups()] == ["tracks.run"]
