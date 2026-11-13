"""``pymmm.view``: the experiment group in napari, every trench with its label images."""

from __future__ import annotations

import numpy as np
import pytest
from fake_experiment import CELL, N_T, N_TRENCH, H, W, cell_image, make_experiment

from pymmm.store import experiment_group as eg
from pymmm.view import (
    AXES,
    label_images,
    napari_get_reader,
    read_experiment,
    view_experiment,
)

pytestmark = pytest.mark.view


def test_napari_offers_pymmm_for_an_experiment_group_and_opens_it(
    tmp_path, monkeypatch
):
    """The plugin napari found from PyMMM's entry point (pymmm/napari.yaml), through
    napari's own reader machinery: what a drop onto napari does after the choice of reader."""
    from napari.components import ViewerModel, viewer_model
    from napari.errors import MultipleReaderError
    from napari.plugins import _initialize_plugins

    _initialize_plugins()  # as napari does at startup: register the installed plugins
    # not this computer's remembered choices of reader
    monkeypatch.setattr(viewer_model, "get_preferred_reader", lambda path: None)
    path = make_experiment(tmp_path)
    assert napari_get_reader(str(path)) is read_experiment
    assert napari_get_reader([str(path)]) is read_experiment
    for other in (str(path / "trench_0010"), str(tmp_path), [str(path), str(path)]):
        assert napari_get_reader(other) is None

    # Dropped with no reader chosen: napari asks, and PyMMM is among the choices.
    with pytest.raises(MultipleReaderError) as asked:
        ViewerModel().open(path, plugin=None)
    assert "pymmm" in asked.value.available_plugins

    viewer = ViewerModel()
    layers = viewer.open(path, plugin="pymmm")
    assert [layer.name for layer in layers] == ["PC", "mCherry"]
    assert viewer.dims.axis_labels == AXES
    assert layers[1].data.shape == (N_TRENCH, N_T, H, W)
    np.testing.assert_array_equal(np.asarray(layers[1].data[2]), cell_image()[2])
    assert layers[1].metadata["pymmm"]["trenches"] == [10, 11, 12]


def test_view_stacks_every_trench_with_its_label_images(tmp_path):
    path = make_experiment(tmp_path)
    from napari.components import ViewerModel

    viewer = view_experiment(path, ViewerModel())
    assert [layer.name for layer in viewer.layers] == ["PC", "mCherry"]

    # A segmentation finished in trench 11 only, and a tracking run's masks in trench 10.
    root = eg.open_experiment(path, mode="r+")
    eg.update_experiment(root, "segmentations", "seg", {"hypotheses": ["b", "a"]})
    cells = (cell_image() > CELL // 2).astype(np.uint16)
    eg.write_segmentation(root, 11, "seg", {"b": cells[1], "a": 2 * cells[1]})
    eg.update_experiment(root, "tracks", "run", {"segmentation": "seg"})
    eg.write_label_image(root, 10, "tracks.run", cells[0])
    eg.update_experiment(root, "segmentations", "unwritten", {"hypotheses": ["x"]})
    assert label_images(root) == ["seg.b", "seg.a", "unwritten.x", "tracks.run"]

    viewer = view_experiment(path, ViewerModel())
    assert [layer.name for layer in viewer.layers] == [
        "PC",
        "mCherry",
        "seg.b",
        "seg.a",
        "tracks.run",
    ]
    assert [layer.visible for layer in viewer.layers][2:] == [True, False, False]
    assert viewer.layers["mCherry"].colormap.colors[-1].tolist() == [1, 0, 0, 1]
    labels = viewer.layers["seg.a"]
    assert type(labels).__name__ == "Labels"
    assert (
        labels.data.shape
        == viewer.layers["mCherry"].data.shape
        == (N_TRENCH, N_T, H, W)
    )
    assert viewer.dims.axis_labels == ("trench", "t", "y", "x")

    np.testing.assert_array_equal(np.asarray(labels.data[1]), 2 * cells[1])
    assert not np.asarray(labels.data[0]).any() and not np.asarray(labels.data[2]).any()
    np.testing.assert_array_equal(
        np.asarray(viewer.layers["tracks.run"].data[0]), cells[0]
    )
    np.testing.assert_array_equal(
        np.asarray(viewer.layers["mCherry"].data[2]), cell_image()[2]
    )

    viewer.dims.current_step = (1, 3, 0, 0)
    assert viewer.canvas.overlays.text.text == "trench 11 (f0), frame 8"
    base = viewer.layers["seg.b"]  # napari slices only visible layers
    base.refresh()
    np.testing.assert_array_equal(
        np.asarray(base._slice.image.raw) > 0,
        np.asarray(viewer.layers["mCherry"].data[1, 3]) > CELL // 2,
    )
