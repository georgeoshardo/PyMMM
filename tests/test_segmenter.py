from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from scipy import ndimage

from pymmm.segmenter import Segmenter, order_from_closed_end

N_TRENCH, N_T, H, W = 3, 4, 12, 6
BACKGROUND, CELL = 100, 1000


def _cell_image() -> np.ndarray:
    """mCherry frames: two cells per trench, trench 1 empty at T=2."""
    image = np.full((N_TRENCH, N_T, H, W), BACKGROUND, dtype=np.uint16)
    image[:, :, 1:4, 1:5] = CELL
    image[:, :, 6:10, 1:5] = CELL
    image[1, 2] = BACKGROUND
    return image


def _make_store(tmp_path):
    path = tmp_path / "exp.trenches.zarr"
    data = np.zeros((N_TRENCH, N_T, 2, H, W), dtype=np.uint16)
    data[:, :, 1] = _cell_image()
    root = xr.Dataset(
        data_vars={
            "data": (("Trench", "T", "C", "Y", "X"), data),
            "fov_name": (("Trench",), np.asarray(["f0", "f0", "f1"], dtype=object)),
        },
        coords={
            "Trench": np.arange(10, 10 + N_TRENCH, dtype=np.int32),
            "T": np.arange(5, 5 + N_T, dtype=np.int32),
            "C": np.asarray(["PC", "mCherry"], dtype=object),
            "Y": np.arange(H, dtype=np.int32),
            "X": np.arange(W, dtype=np.int32),
        },
        attrs={"layout": "trench_store", "provenance": {"created": "2026-01-01T00:00:00"}},
    )
    root.to_zarr(
        path, mode="w", consolidated=False, zarr_format=3,
        encoding={"data": {"chunks": (1, N_T, 1, H, W)}},
    )
    xr.Dataset({"shift": (("FOV",), np.ones(2))}).to_zarr(
        path, mode="a", group="registration", consolidated=False, zarr_format=3,
    )
    return path


def _bottom_first(mask: np.ndarray) -> np.ndarray:
    """Label connected components, numbered from the bottom of the frame up."""
    labels, n = ndimage.label(mask)
    labels[labels > 0] = n + 1 - labels[labels > 0]
    return labels


class _FakeModel:
    """Stands in for OmniModel: thresholds the normalised image."""

    def __init__(self, fail_stack: bool = False, fail_frames=()):
        self.fail_stack = fail_stack
        self.fail_frames = set(fail_frames)
        self.stack_calls = 0
        self.frame_calls = 0

    def eval(self, x, **kwargs):
        assert kwargs["normalize"] is False
        if x.shape[0] > 1:
            self.stack_calls += 1
            if self.fail_stack:
                raise IndexError("index 1 is out of bounds")
        else:
            t = self.frame_calls
            self.frame_calls += 1
            if t in self.fail_frames:
                raise IndexError("index 1 is out of bounds")
        return SimpleNamespace(masks=np.stack([_bottom_first(f > 0.5) for f in x]))


@pytest.fixture
def segmenter(tmp_path, monkeypatch):
    model = _FakeModel()
    monkeypatch.setattr(Segmenter, "_get_model", lambda self: model)
    seg = Segmenter(_make_store(tmp_path), channel="mCherry", gpu=False)
    seg.fake_model = model
    return seg


def test_order_from_closed_end_numbers_cells_from_y0():
    frame = np.zeros((10, 4), dtype=np.int32)
    frame[7:9] = 5
    frame[4:6] = 9
    frame[0:2] = 2
    masks = order_from_closed_end(np.stack([frame, np.zeros_like(frame)]))

    assert masks.dtype == np.uint16
    assert masks[0, 0, 0] == 1 and masks[0, 4, 0] == 2 and masks[0, 7, 0] == 3
    assert not masks[1].any()


def test_bounds_are_estimated_from_the_store(segmenter):
    assert segmenter.params["lower"] == pytest.approx(BACKGROUND)
    assert segmenter.params["upper"] == pytest.approx(CELL)


def test_segment_writes_labels_aligned_with_data(segmenter):
    segmenter.segment(show_progress=False)

    tree = xr.open_datatree(segmenter.store_path, engine="zarr", consolidated=False)
    group = tree["segmentation/mcherry"].ds
    labels = group["labels"]
    root = tree.ds

    assert labels.dims == ("Trench", "T", "Y", "X")
    assert labels.dtype == np.uint16
    assert labels.encoding["chunks"] == (1, N_T, H, W)
    for dim in labels.dims:
        np.testing.assert_array_equal(labels[dim].values, root[dim].values)

    values = labels.values
    assert values[0, 0, 2, 2] == 1 and values[0, 0, 8, 2] == 2
    assert not values[1, 2].any()
    assert values.max() == 2

    attrs = group.attrs
    assert attrs["layout"] == "segmentation"
    assert attrs["channel"] == "mCherry" and attrs["model"] == "bact_fluor_omni"
    assert attrs["completed_trenches"] == [10, 11, 12]
    assert attrs["segmentation_complete"] is True
    assert attrs["extraction_created"] == "2026-01-01T00:00:00"

    np.testing.assert_array_equal(root["data"].sel(C="mCherry").values, _cell_image())
    np.testing.assert_array_equal(tree["registration"].ds["shift"].values, np.ones(2))


def test_segment_resumes_and_rejects_changed_settings(segmenter):
    segmenter.segment(show_progress=False)
    calls = segmenter.fake_model.stack_calls

    segmenter.segment(show_progress=False)
    assert segmenter.fake_model.stack_calls == calls

    segmenter._params["min_size"] = 50
    with pytest.raises(RuntimeError, match="params"):
        segmenter.segment(show_progress=False)

    segmenter.segment(resume=False, show_progress=False)
    assert segmenter.fake_model.stack_calls == 2 * calls
    tree = xr.open_datatree(segmenter.store_path, engine="zarr", consolidated=False)
    assert tree["segmentation/mcherry"].ds.attrs["params"]["min_size"] == 50
    np.testing.assert_array_equal(tree.ds["data"].sel(C="mCherry").values, _cell_image())


def test_reopened_segmenter_keeps_saved_params(segmenter):
    segmenter._params["mask_threshold"] = -1.0
    segmenter.segment(show_progress=False)

    reopened = Segmenter(segmenter.store_path, channel="mCherry", gpu=False)
    assert reopened.params == segmenter.params
    assert reopened.is_complete


def test_omnipose_index_error_retries_frame_by_frame(segmenter):
    segmenter.fake_model.fail_stack = True
    segmenter.fake_model.fail_frames = {2}
    masks, failed = segmenter.segment_stack(_cell_image()[0])

    assert failed == [2]
    assert not masks[2].any()
    assert masks[0, 2, 2] == 1 and masks[3, 8, 2] == 2
