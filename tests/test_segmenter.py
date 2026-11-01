from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from scipy import ndimage

from pymmm.segmenter import (
    HYPOTHESES,
    Segmenter,
    background_noise,
    drop_faint,
    normalise,
    order_from_closed_end,
    trench_bounds,
)

N_TRENCH, N_T, H, W = 3, 4, 12, 6
BACKGROUND, NOISE, CELL = 100, 5, 1000
PIXEL_SIZE = 0.1


def _cell_image() -> np.ndarray:
    """mCherry frames: two cells per trench, trench 1 empty at T=2, trench 2 noise only.

    Trench 0 has a zero-filled column (outside the camera image).
    """
    rng = np.random.default_rng(0)
    image = np.rint(rng.normal(BACKGROUND, NOISE, (N_TRENCH, N_T, H, W))).astype(np.uint16)
    image[:2, :, 1:4, 1:5] = CELL
    image[:2, :, 6:10, 1:5] = CELL
    image[1, 2] = np.rint(rng.normal(BACKGROUND, NOISE, (H, W)))
    image[0, :, :, 0] = 0
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
    xr.Dataset({"pixel_size": ((), PIXEL_SIZE)}).to_zarr(
        path, mode="a", group="acquisition", consolidated=False, zarr_format=3,
    )
    return path


def _bottom_first(mask: np.ndarray) -> np.ndarray:
    """Label connected components, numbered from the bottom of the frame up."""
    labels, n = ndimage.label(mask)
    labels[labels > 0] = n + 1 - labels[labels > 0]
    return labels


class _FakeModel:
    """Stands in for OmniModel: thresholds the normalised image."""

    THRESHOLD = 0.3

    def __init__(self, fail_stack: bool = False, fail_frames=()):
        self.fail_stack = fail_stack
        self.fail_frames = set(fail_frames)
        self.stack_calls = 0
        self.frame_calls = 0
        self.kwargs = []

    def eval(self, x, **kwargs):
        assert kwargs["normalize"] is False
        self.kwargs.append(kwargs)
        if x.shape[0] > 1:
            self.stack_calls += 1
            if self.fail_stack:
                raise IndexError("index 1 is out of bounds")
        else:
            t = self.frame_calls
            self.frame_calls += 1
            if t in self.fail_frames:
                raise IndexError("index 1 is out of bounds")
        return SimpleNamespace(masks=np.stack([_bottom_first(f > self.THRESHOLD) for f in x]))


@pytest.fixture
def segmenter(tmp_path, monkeypatch):
    model = _FakeModel()
    monkeypatch.setattr(Segmenter, "_get_model", lambda self: model)
    seg = Segmenter(_make_store(tmp_path), channel="mCherry", gpu=False)
    seg.fake_model = model
    return seg


def _open_group(seg):
    return xr.open_datatree(seg.store_path, engine="zarr", consolidated=False)


def test_order_from_closed_end_numbers_cells_from_y0():
    frame = np.zeros((10, 4), dtype=np.int32)
    frame[7:9] = 5
    frame[4:6] = 9
    frame[0:2] = 2
    masks = order_from_closed_end(np.stack([frame, np.zeros_like(frame)]))

    assert masks.dtype == np.uint16
    assert masks[0, 0, 0] == 1 and masks[0, 4, 0] == 2 and masks[0, 7, 0] == 3
    assert not masks[1].any()


def test_background_noise_ignores_cells_and_zero_fill():
    # Cells must be a minority of pixels, as in a single-file trench.
    rng = np.random.default_rng(1)
    image = rng.normal(BACKGROUND, NOISE, (40, 40))
    image[5:15, 10:20] = CELL
    image[:, 0] = 0
    background, noise = background_noise(image)
    assert background == pytest.approx(BACKGROUND, abs=1)
    assert noise == pytest.approx(NOISE, rel=0.15)


def test_floor_keeps_a_noise_only_trench_dark():
    noise_only = _cell_image()[2]
    _, upper_without = trench_bounds(noise_only, floor=0)
    lower, upper = trench_bounds(noise_only, floor=BACKGROUND + 20 * NOISE)

    assert upper_without < BACKGROUND + 4 * NOISE
    assert upper == BACKGROUND + 20 * NOISE
    assert normalise(noise_only, lower, upper).max() < _FakeModel.THRESHOLD
    assert normalise(noise_only, lower, upper_without).max() > 1


def test_normalise_maps_zero_fill_to_zero_and_applies_gain():
    stack = np.array([[0, 100, 300]], dtype=np.uint16)
    np.testing.assert_allclose(normalise(stack, 100, 300), [[0, 0, 1]])
    np.testing.assert_allclose(normalise(stack, 100, 300, gain=2), [[0, 0, 0.5]])


def test_drop_faint_removes_masks_on_background():
    stack = _cell_image()[:1, :1]
    masks = np.zeros(stack.shape, np.int32)
    masks[0, 0, 1:4, 1:5] = 1          # bright cell
    masks[0, 0, 10:12, 3:5] = 2        # background patch
    kept = drop_faint(masks, stack, min_contrast=5)
    assert set(np.unique(kept)) == {0, 1}


def test_floor_is_estimated_from_the_store(segmenter):
    assert segmenter.params["floor"] == pytest.approx(BACKGROUND + 20 * NOISE, rel=0.3)


def test_segment_writes_hypotheses_aligned_with_data(segmenter):
    segmenter.segment(show_progress=False)

    tree = _open_group(segmenter)
    group = tree["segmentation/mcherry"].ds
    labels = group["labels"]
    root = tree.ds

    assert labels.dims == ("Hypothesis", "Trench", "T", "Y", "X")
    assert labels.dtype == np.uint16
    assert labels.encoding["chunks"] == (1, 1, N_T, H, W)
    assert list(labels["Hypothesis"].values) == list(HYPOTHESES)
    for dim in labels.dims[1:]:
        np.testing.assert_array_equal(labels[dim].values, root[dim].values)

    for hypothesis in HYPOTHESES:
        values = labels.sel(Hypothesis=hypothesis).values
        assert values[0, 0, 2, 2] == 1 and values[0, 0, 8, 2] == 2, hypothesis
        assert not values[1, 2].any(), hypothesis
        assert not values[2].any(), f"{hypothesis}: cells in the noise-only trench"

    assert (group["upper"].values >= segmenter.params["floor"]).all()
    assert group["lower"].values[0] > 0  # zero-filled column ignored

    attrs = group.attrs
    assert attrs["layout"] == "segmentation"
    assert attrs["completed"] == {h: [10, 11, 12] for h in HYPOTHESES}
    assert attrs["complete"] == {h: True for h in HYPOTHESES}
    assert attrs["hypotheses"]["S2"]["gain"] == 1.5
    assert attrs["extraction_created"] == "2026-01-01T00:00:00"

    np.testing.assert_array_equal(root["data"].sel(C="mCherry").values, _cell_image())
    np.testing.assert_array_equal(tree["registration"].ds["shift"].values, np.ones(2))


def test_hypothesis_settings_reach_omnipose(segmenter):
    segmenter.segment(hypotheses=["S3"], show_progress=False)
    kwargs = segmenter.fake_model.kwargs[-1]
    assert kwargs["rescale_factor"] == 0.8
    assert kwargs["mask_threshold"] == -1.0
    assert kwargs["flow_threshold"] == 0.0
    assert kwargs["min_size"] == round(1.75 / PIXEL_SIZE ** 2)


def test_s1_first_then_the_rest_resumes(segmenter):
    segmenter.segment(hypotheses=["S1"], show_progress=False)
    assert segmenter.is_complete("S1") and not segmenter.is_complete()
    calls = segmenter.fake_model.stack_calls

    segmenter.segment(hypotheses=["S2", "S3", "S4"], show_progress=False)
    assert segmenter.is_complete()
    assert segmenter.fake_model.stack_calls == calls + 3 * N_TRENCH

    segmenter.segment(show_progress=False)
    assert segmenter.fake_model.stack_calls == calls + 3 * N_TRENCH


def _s1_labels(seg):
    return _open_group(seg)["segmentation/mcherry"].ds["labels"].sel(Hypothesis="S1").values


def test_changed_settings_after_s1_never_clear_s1(segmenter):
    segmenter.segment(hypotheses=["S1"], show_progress=False)
    s1 = _s1_labels(segmenter)
    assert s1.any()
    segmenter._params["min_area"] = 0.5

    rest = ["S2", "S3", "S4"]
    with pytest.raises(RuntimeError, match=r"S1 already have results.*\['S1', 'S2', 'S3', 'S4'\]"):
        segmenter.segment(hypotheses=rest, show_progress=False)
    with pytest.raises(RuntimeError, match="S1 already have results"):
        segmenter.segment(hypotheses=rest, resume=False, show_progress=False)
    np.testing.assert_array_equal(_s1_labels(segmenter), s1)
    assert segmenter.is_complete("S1")

    segmenter.segment(hypotheses=list(HYPOTHESES), resume=False, show_progress=False)
    attrs = _open_group(segmenter)["segmentation/mcherry"].ds.attrs
    assert attrs["params"]["min_area"] == 0.5
    assert segmenter.is_complete()


def test_resume_false_with_same_settings_resets_only_those_hypotheses(segmenter):
    segmenter.segment(show_progress=False)
    s1 = _s1_labels(segmenter)
    calls = segmenter.fake_model.stack_calls

    segmenter.segment(hypotheses=["S3"], resume=False, show_progress=False)
    assert segmenter.fake_model.stack_calls == calls + N_TRENCH
    np.testing.assert_array_equal(_s1_labels(segmenter), s1)
    assert segmenter.is_complete()


def test_reopened_segmenter_keeps_saved_params(segmenter):
    segmenter._params["mask_threshold"] = -1.0
    segmenter.segment(hypotheses=["S1"], show_progress=False)

    reopened = Segmenter(segmenter.store_path, channel="mCherry", gpu=False)
    assert reopened.params == segmenter.params
    assert reopened.is_complete("S1")


def test_omnipose_index_error_retries_frame_by_frame(segmenter):
    segmenter.fake_model.fail_stack = True
    segmenter.fake_model.fail_frames = {2}
    stack = _cell_image()[0]
    masks, failed = segmenter.segment_stack(stack, trench_bounds(stack, 200))

    assert failed == [2]
    assert not masks[2].any()
    assert masks[0, 2, 2] == 1 and masks[3, 8, 2] == 2
