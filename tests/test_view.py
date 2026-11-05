from __future__ import annotations

import numpy as np
import pytest

napari = pytest.importorskip("napari")

from napari.components import ViewerModel

from pymmm import trench_images as ti
from pymmm.view import label_images, view_experiment
from fake_experiment import CELL, H, N_T, N_TRENCH, W, cell_image, make_experiment


def test_view_stacks_every_trench_with_its_label_images(tmp_path):
    path = make_experiment(tmp_path)
    viewer = view_experiment(path, ViewerModel())
    assert [layer.name for layer in viewer.layers] == ["PC", "mCherry"]

    # A segmentation finished in trench 11 only, and a tracking run's masks in trench 10.
    root = ti.open_experiment(path, mode="r+")
    ti.update_experiment(root, "segmentations", "seg", {"hypotheses": ["b", "a"]})
    cells = (cell_image() > CELL // 2).astype(np.uint16)
    ti.write_segmentation(root, 11, "seg", {"b": cells[1], "a": 2 * cells[1]})
    ti.update_experiment(root, "tracks", "run", {"segmentation": "seg"})
    ti.write_label_image(root, 10, "tracks.run", cells[0])
    ti.update_experiment(root, "segmentations", "unwritten", {"hypotheses": ["x"]})
    assert label_images(root) == ["seg.b", "seg.a", "unwritten.x", "tracks.run"]

    viewer = view_experiment(path, ViewerModel())
    assert [layer.name for layer in viewer.layers] == ["PC", "mCherry", "seg.b", "seg.a", "tracks.run"]
    assert [layer.visible for layer in viewer.layers][2:] == [True, False, False]
    assert viewer.layers["mCherry"].colormap.colors[-1].tolist() == [1, 0, 0, 1]
    labels = viewer.layers["seg.a"]
    assert type(labels).__name__ == "Labels"
    assert labels.data.shape == viewer.layers["mCherry"].data.shape == (N_TRENCH, N_T, H, W)
    assert viewer.dims.axis_labels == ("trench", "t", "y", "x")

    np.testing.assert_array_equal(np.asarray(labels.data[1]), 2 * cells[1])
    assert not np.asarray(labels.data[0]).any() and not np.asarray(labels.data[2]).any()
    np.testing.assert_array_equal(np.asarray(viewer.layers["tracks.run"].data[0]), cells[0])
    np.testing.assert_array_equal(np.asarray(viewer.layers["mCherry"].data[2]), cell_image()[2])

    viewer.dims.current_step = (1, 3, 0, 0)
    assert viewer.canvas.overlays.text.text == "trench 11 (f0), frame 8"
    base = viewer.layers["seg.b"]  # napari slices only visible layers
    base.refresh()
    np.testing.assert_array_equal(
        np.asarray(base._slice.image.raw) > 0,
        np.asarray(viewer.layers["mCherry"].data[1, 3]) > CELL // 2,
    )
