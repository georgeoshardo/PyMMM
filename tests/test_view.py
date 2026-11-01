from __future__ import annotations

import numpy as np
import pytest

napari = pytest.importorskip("napari")

from napari.components import ViewerModel

from pymmm.segmenter import Segmenter
from pymmm.view import segmentation_names, view_trench_store
from test_segmenter import N_T, N_TRENCH, H, W, _FakeModel, _make_store


def test_view_adds_channels_and_segmentation_labels(tmp_path, monkeypatch):
    store = _make_store(tmp_path)
    viewer = view_trench_store(store, ViewerModel())
    assert [layer.name for layer in viewer.layers] == ["PC", "mCherry"]

    model = _FakeModel()
    monkeypatch.setattr(Segmenter, "_get_model", lambda self: model)
    Segmenter(store, channel="mCherry", gpu=False).segment(show_progress=False)
    assert segmentation_names(store) == ["mcherry"]

    viewer = view_trench_store(store, ViewerModel())
    labels = viewer.layers["mcherry labels"]
    assert type(labels).__name__ == "Labels"
    assert labels.data.shape == (N_TRENCH, N_T, H, W)
    assert viewer.dims.axis_labels == ("Trench", "T", "Y", "X")

    viewer.dims.current_step = (1, 3, 0, 0)
    assert viewer.canvas.overlays.text.text == "trench 11 (f0), frame 8"
    labels.refresh()
    np.testing.assert_array_equal(
        np.asarray(labels._slice.image.raw) > 0,
        np.asarray(viewer.layers["mCherry"].data[1, 3]) > 500,
    )
