from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from fake_experiment import (
    BACKGROUND,
    CELL,
    N_T,
    NOISE,
    PIXEL_SIZE,
    TRENCHES,
    H,
    W,
    cell_image,
    make_experiment,
)
from scipy import ndimage

from pymmm.steps.segmentation import (
    HYPOTHESES,
    Segmenter,
    background_noise,
    drop_faint,
    normalise,
    order_from_closed_end,
    trench_bounds,
)
from pymmm.store import experiment_group as eg


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
def fake_model(monkeypatch):
    model = _FakeModel()
    monkeypatch.setattr(Segmenter, "_get_model", lambda self: model)
    return model


@pytest.fixture
def experiment(tmp_path):
    return make_experiment(tmp_path)


@pytest.fixture
def segmenter(experiment, fake_model):
    seg = Segmenter(experiment, channel="mCherry", gpu=False)
    seg.fake_model = fake_model
    return seg


def _labels(seg, trench, hypothesis=None):
    return np.asarray(seg.labels(trench, hypothesis))


def _record(seg):
    return eg.records(eg.open_experiment(seg.store_path), "segmentations")[seg.name]


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
    noise_only = cell_image()[2]
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
    stack = cell_image()[:1, :1]
    masks = np.zeros(stack.shape, np.int32)
    masks[0, 0, 1:4, 1:5] = 1          # bright cell
    masks[0, 0, 10:12, 3:5] = 2        # background patch
    kept = drop_faint(masks, stack, min_contrast=5)
    assert set(np.unique(kept)) == {0, 1}



def test_floor_is_estimated_from_the_experiment(segmenter):
    assert segmenter.params["floor"] == pytest.approx(BACKGROUND + 20 * NOISE, rel=0.3)


def test_segment_writes_one_label_image_per_hypothesis_beside_each_trench(segmenter):
    segmenter.segment(show_progress=False)

    record = _record(segmenter)
    assert record["hypotheses"] == list(HYPOTHESES)
    assert record["settings"]["S2"]["gain"] == 1.5
    assert (record["channel"], record["model"]) == ("mCherry", "bact_fluor_omni")
    assert record["params"] == segmenter.params
    assert record["extraction_created"] == "2026-01-01T00:00:00"
    assert segmenter.finished() == TRENCHES and segmenter.is_complete()

    root = eg.open_experiment(segmenter.store_path)
    for k, trench in enumerate(TRENCHES):
        assert eg.listed_labels(root, trench) == [f"mcherry.{h}" for h in HYPOTHESES]
        names, arrays = eg.segmentation(root, trench, "mcherry")
        assert names == list(HYPOTHESES)
        for hypothesis, labels in zip(names, arrays):
            assert labels.shape == (N_T, H, W) and labels.dtype == np.uint16
            values = labels[...]
            if k == 0:
                assert values[0, 2, 2] == 1 and values[0, 8, 2] == 2, hypothesis
            elif k == 1:
                assert values[0, 2, 2] == 1 and not values[2].any(), hypothesis
            else:
                assert not values.any(), f"{hypothesis}: cells in the noise-only trench"
        attrs = root[f"{eg.trench_path(trench)}/labels/mcherry.S2"].attrs["pymmm"]
        assert attrs["hypothesis"] == "S2" and attrs["channel"] == "mCherry"
        assert attrs["upper"] >= segmenter.params["floor"]
        assert attrs["failed_frames"] == {}
    assert root["trench_0010/labels/mcherry.S1"].attrs["pymmm"]["lower"] > 0  # zero fill ignored
    np.testing.assert_array_equal(eg.image(root, 10)[:, 1], cell_image()[0])


def test_hypothesis_settings_reach_omnipose(experiment, fake_model):
    seg = Segmenter(experiment, channel="mCherry", hypotheses=["S3"], gpu=False)
    seg.segment(trenches=[10], show_progress=False)
    kwargs = fake_model.kwargs[-1]
    assert kwargs["rescale_factor"] == 0.8
    assert kwargs["mask_threshold"] == -1.0
    assert kwargs["flow_threshold"] == 0.0
    assert kwargs["min_size"] == round(1.75 / PIXEL_SIZE ** 2)
    assert eg.listed_labels(eg.open_experiment(experiment), 10) == ["mcherry.S3"]


def test_a_few_trenches_first_then_the_rest_resumes(segmenter):
    segmenter.segment(trenches=[11], show_progress=False)
    assert segmenter.finished() == [11] and not segmenter.is_complete()
    calls = segmenter.fake_model.stack_calls
    assert calls == len(HYPOTHESES)

    segmenter.segment(show_progress=False)
    assert segmenter.is_complete()
    assert segmenter.fake_model.stack_calls == calls + 2 * len(HYPOTHESES)

    segmenter.segment(show_progress=False)
    assert segmenter.fake_model.stack_calls == calls + 2 * len(HYPOTHESES)


def test_changed_settings_never_clear_results(segmenter):
    segmenter.segment(trenches=[10], show_progress=False)
    s1 = _labels(segmenter, 10)
    assert s1.any()
    segmenter._params["min_area"] = 0.5

    with pytest.raises(RuntimeError, match=r"params, settings differ.*1 trenches have results.*resume=False"):
        segmenter.segment(show_progress=False)
    with pytest.raises(RuntimeError, match="1 trenches have results"):
        segmenter.segment(trenches=[11, 12], resume=False, show_progress=False)
    np.testing.assert_array_equal(_labels(segmenter, 10), s1)
    assert segmenter.finished() == [10]

    segmenter.segment(resume=False, show_progress=False)
    assert _record(segmenter)["params"]["min_area"] == 0.5
    assert segmenter.is_complete()


def test_other_hypotheses_replace_the_segmentation_only_when_asked(segmenter, experiment):
    segmenter.segment(show_progress=False)
    fewer = Segmenter(experiment, channel="mCherry", hypotheses=["S1", "S3"], gpu=False)
    with pytest.raises(RuntimeError, match="hypotheses, settings differ"):
        fewer.segment(show_progress=False)
    assert segmenter.is_complete()

    fewer.segment(resume=False, show_progress=False)
    root = eg.open_experiment(experiment)
    assert eg.hypotheses(root, "mcherry") == ["S1", "S3"]
    for trench in TRENCHES:
        assert eg.listed_labels(root, trench) == ["mcherry.S1", "mcherry.S3"]
        assert sorted(name for name, _ in root[f"{eg.trench_path(trench)}/labels"].groups()) == [
            "mcherry.S1", "mcherry.S3"]


def test_resume_false_with_same_settings_resegments_only_those_trenches(segmenter):
    segmenter.segment(show_progress=False)
    kept = _labels(segmenter, 10, "S4")
    calls = segmenter.fake_model.stack_calls

    segmenter.segment(trenches=[11], resume=False, show_progress=False)
    assert segmenter.fake_model.stack_calls == calls + len(HYPOTHESES)
    np.testing.assert_array_equal(_labels(segmenter, 10, "S4"), kept)
    assert segmenter.is_complete()


def test_reopened_segmenter_keeps_saved_params_and_hypotheses(experiment, fake_model):
    seg = Segmenter(experiment, channel="mCherry", hypotheses=["S1", "S3"], gpu=False)
    seg._params["mask_threshold"] = -1.0
    seg.segment(trenches=[10], show_progress=False)

    reopened = Segmenter(experiment, channel="mCherry", gpu=False)
    assert reopened.params == seg.params
    assert reopened.hypotheses == ["S1", "S3"]
    assert reopened.finished() == [10]
    assert "1/3 trenches finished" in repr(reopened)
    with pytest.raises(ValueError, match="unknown hypotheses"):
        Segmenter(experiment, channel="mCherry", hypotheses=["S9"], gpu=False)


def test_worker_writes_the_same_labels_as_in_process(segmenter, monkeypatch):
    from pymmm.steps import segmentation as module

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        monkeypatch.delenv(var, raising=False)  # restored after the test
    segmenter.segment(show_progress=False)
    root = eg.open_experiment(segmenter.store_path, mode="r+")
    expected = {t: [a[...] for a in eg.segmentation(root, t, "mcherry")[1]] for t in TRENCHES}
    for trench in TRENCHES:
        eg.remove_segmentation(root, trench, "mcherry")
    assert segmenter.finished() == []

    module._init_worker(str(segmenter.store_path), "mCherry", "bact_fluor_omni", "mcherry",
                        list(HYPOTHESES), False, segmenter.params)
    for trench in TRENCHES:
        returned, bounds, failed = module._segment_in_worker((trench, None, 64))
        assert returned == trench and failed == {}
        assert bounds == segmenter.bounds(trench)
    assert segmenter.finished() == TRENCHES
    for trench in TRENCHES:
        for want, got in zip(expected[trench], eg.segmentation(root, trench, "mcherry")[1]):
            np.testing.assert_array_equal(got[...], want)


def test_segment_refuses_unknown_trenches(segmenter):
    with pytest.raises(ValueError, match=r"no trenches \[99\]"):
        segmenter.segment(trenches=[10, 99], show_progress=False)




def test_omnipose_index_error_retries_frame_by_frame(segmenter):
    segmenter.fake_model.fail_stack = True
    segmenter.fake_model.fail_frames = {2}
    stack = cell_image()[0]
    masks, failed = segmenter.segment_stack(stack, trench_bounds(stack, 200))

    assert failed == [2]
    assert not masks[2].any()
    assert masks[0, 2, 2] == 1 and masks[3, 8, 2] == 2
