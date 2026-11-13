"""``pymmm.steps.segmentation``: the Segmenter on a fake model, its normalisation and
its label images (the widget and viewer: test_widgets_segmentation.py)."""

from __future__ import annotations

import hashlib
import socket

import numpy as np
import pytest
from fake_experiment import (
    BACKGROUND,
    CELL,
    N_T,
    NOISE,
    PIXEL_SIZE,
    TRENCHES,
    FakeModel,
    H,
    W,
    cell_image,
    make_experiment,
)

from pymmm.steps.segmentation import (
    HYPOTHESES,
    Segmenter,
    background_noise,
    checkpoint_hypotheses,
    drop_faint,
    model_shape,
    normalise,
    order_from_closed_end,
    scale_labels,
    trench_bounds,
)
from pymmm.store import experiment_group as eg


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
    assert normalise(noise_only, lower, upper).max() < FakeModel.THRESHOLD
    assert normalise(noise_only, lower, upper_without).max() > 1


def test_normalise_maps_zero_fill_to_zero_and_applies_gain():
    stack = np.array([[0, 100, 300]], dtype=np.uint16)
    np.testing.assert_allclose(normalise(stack, 100, 300), [[0, 0, 1]])
    np.testing.assert_allclose(normalise(stack, 100, 300, gain=2), [[0, 0, 0.5]])


def test_drop_faint_removes_masks_on_background():
    stack = cell_image()[:1, :1]
    masks = np.zeros(stack.shape, np.int32)
    masks[0, 0, 1:4, 1:5] = 1  # bright cell
    masks[0, 0, 10:12, 3:5] = 2  # background patch
    kept = drop_faint(masks, stack, min_contrast=5)
    assert set(np.unique(kept)) == {0, 1}


def test_scale_labels_resizes_each_mask_and_keeps_its_number():
    labels = np.zeros((1, 4, 4), np.uint16)
    labels[0, :, :2] = 7
    labels[0, :, 2:] = 3
    big = scale_labels(labels, (8, 8))
    assert big.shape == (1, 8, 8) and big.dtype == np.uint16
    assert (big[0, :, :4] == 7).all() and (big[0, :, 4:] == 3).all()
    small = scale_labels(big, (4, 4))
    np.testing.assert_array_equal(small, labels)
    assert scale_labels(labels, (4, 4)) is labels
    assert model_shape((5, 944, 128), 0.065, 0.1079) == (569, 77)
    assert model_shape((944, 128), 0.065, None) == (944, 128)


def test_a_model_at_another_pixel_size_sees_resampled_frames(
    experiment, fake_model, monkeypatch
):
    """Omnipose sees frames at the model's pixel size; the labels come back at the
    trench's own, over the same cells, and the record keeps the model's pixel size."""
    shapes = []
    evaluate = fake_model.eval
    monkeypatch.setattr(
        fake_model,
        "eval",
        lambda x, **kw: (shapes.append(x.shape), evaluate(x, **kw))[1],
    )
    seg = Segmenter(
        experiment,
        channel="mCherry",
        hypotheses=["S1"],
        model_pixel_size=2 * PIXEL_SIZE,
        gpu=False,
    )
    seg.segment(trenches=[10], show_progress=False)
    assert {shape[1:] for shape in shapes} == {(H // 2, W // 2)}
    assert fake_model.kwargs[-1]["min_size"] == round(1.75 / (2 * PIXEL_SIZE) ** 2)
    labels = _labels(seg, 10, "S1")
    assert labels.shape == (N_T, H, W)
    assert labels[0, 2, 2] == 1 and labels[0, 8, 2] == 2
    assert _record(seg)["model_pixel_size"] == 2 * PIXEL_SIZE
    assert "at 0.2 µm/px" in repr(seg)

    # without it, the same segmentation is another one: resuming refuses to mix them
    again = Segmenter(experiment, channel="mCherry", hypotheses=["S1"], gpu=False)
    with pytest.raises(RuntimeError, match="model_pixel_size"):
        again.segment(trenches=[11], show_progress=False)


def test_floor_is_estimated_from_the_experiment(segmenter):
    assert segmenter.params["floor"] == pytest.approx(BACKGROUND + 20 * NOISE, rel=0.3)


def test_segment_writes_one_label_image_per_hypothesis_beside_each_trench(segmenter):
    segmenter.segment(show_progress=False)

    record = _record(segmenter)
    assert record["hypotheses"] == list(HYPOTHESES)
    assert record["settings"]["S2"]["gain"] == 1.5
    assert record["channel"] == "mCherry" and record["checkpoint_sha256"] == {}
    assert record["specs"]["S2"] == {"model": "bact_fluor_omni", **HYPOTHESES["S2"]}
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
        assert attrs["made_on"] == {"host": socket.gethostname(), "device": "cpu"}
    assert (
        root["trench_0010/labels/mcherry.S1"].attrs["pymmm"]["lower"] > 0
    )  # zero fill ignored
    np.testing.assert_array_equal(eg.image(root, 10)[:, 1], cell_image()[0])


def test_hypothesis_settings_reach_omnipose(experiment, fake_model):
    seg = Segmenter(experiment, channel="mCherry", hypotheses=["S3"], gpu=False)
    seg.segment(trenches=[10], show_progress=False)
    kwargs = fake_model.kwargs[-1]
    assert kwargs["rescale_factor"] == 0.8
    assert kwargs["mask_threshold"] == -1.0
    assert kwargs["flow_threshold"] == 0.0
    assert kwargs["min_size"] == round(1.75 / PIXEL_SIZE**2)
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

    with pytest.raises(
        RuntimeError,
        match=r"params, settings differ.*1 trenches have results.*resume=False",
    ):
        segmenter.segment(show_progress=False)
    with pytest.raises(RuntimeError, match="1 trenches have results"):
        segmenter.segment(trenches=[11, 12], resume=False, show_progress=False)
    np.testing.assert_array_equal(_labels(segmenter, 10), s1)
    assert segmenter.finished() == [10]

    segmenter.segment(resume=False, show_progress=False)
    assert _record(segmenter)["params"]["min_area"] == 0.5
    assert segmenter.is_complete()


def test_other_hypotheses_replace_the_segmentation_only_when_asked(
    segmenter, experiment
):
    segmenter.segment(show_progress=False)
    fewer = Segmenter(experiment, channel="mCherry", hypotheses=["S1", "S3"], gpu=False)
    with pytest.raises(RuntimeError, match="hypotheses, specs, settings differ"):
        fewer.segment(show_progress=False)
    assert segmenter.is_complete()

    fewer.segment(resume=False, show_progress=False)
    root = eg.open_experiment(experiment)
    assert eg.hypotheses(root, "mcherry") == ["S1", "S3"]
    for trench in TRENCHES:
        assert eg.listed_labels(root, trench) == ["mcherry.S1", "mcherry.S3"]
        assert sorted(
            name for name, _ in root[f"{eg.trench_path(trench)}/labels"].groups()
        ) == ["mcherry.S1", "mcherry.S3"]


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


# ----------------------------------------------------------------------------
# Hypotheses that are checkpoints; where the trenches are segmented
# ----------------------------------------------------------------------------
PC_PARAMS = {"masks": "affinity", "floor": 0.0, "min_contrast": "off"}
CHECKPOINTS = {"e250": "omni_epoch_250", "e050": "omni_epoch_50"}


def _models(tmp_path):
    folder = tmp_path / "models"
    folder.mkdir()
    for k, name in enumerate(CHECKPOINTS.values()):
        (folder / name).write_bytes(bytes([k]) * 1000)
    return folder


def _ensemble(experiment, models, params=PC_PARAMS):
    return Segmenter(
        experiment,
        channel="mCherry",
        name="ens",
        models=models,
        params=params,
        hypotheses=checkpoint_hypotheses(CHECKPOINTS),
        gpu=False,
    )


def test_checkpoint_hypotheses_record_their_files(
    experiment, fake_model, tmp_path, monkeypatch
):
    from pymmm.steps import segmentation as module

    monkeypatch.setattr(
        module, "drop_faint", lambda *a: pytest.fail("min_contrast is off")
    )
    models = _models(tmp_path)
    seg = _ensemble(experiment, models)
    assert seg.params["min_contrast"] is None and seg.params["floor"] == 0.0
    seg.segment(show_progress=False)

    record = _record(seg)
    assert record["hypotheses"] == ["e250", "e050"]
    assert record["specs"] == {
        "e250": {"checkpoint": "omni_epoch_250"},
        "e050": {"checkpoint": "omni_epoch_50"},
    }
    assert record["checkpoint_sha256"] == {
        "e250": hashlib.sha256(bytes([0]) * 1000).hexdigest(),
        "e050": hashlib.sha256(bytes([1]) * 1000).hexdigest(),
    }
    assert record["settings"]["e050"]["masks"] == "affinity"
    # affinity masks: the network's maps first, then each frame's masks
    assert fake_model.kwargs[-1]["compute_masks"] is False
    labels = _labels(seg, 10, "e050")
    assert labels[0, 2, 2] == 1 and labels[0, 8, 2] == 2

    reopened = Segmenter(
        experiment, channel="mCherry", name="ens", models=models, gpu=False
    )
    assert reopened.hypotheses == ["e250", "e050"] and reopened.params == seg.params
    assert reopened.is_complete()

    (models / "omni_epoch_50").write_bytes(b"retrained")
    changed = Segmenter(
        experiment, channel="mCherry", name="ens", models=models, gpu=False
    )
    with pytest.raises(
        RuntimeError,
        match="checkpoint_sha256, settings differ|checkpoint_sha256 differ",
    ):
        changed.segment(show_progress=False)


def test_checkpoint_hypotheses_need_their_files(experiment, fake_model, tmp_path):
    with pytest.raises(ValueError, match="give models="):
        _ensemble(experiment, None)
    with pytest.raises(FileNotFoundError, match="omni_epoch_250"):
        _ensemble(experiment, tmp_path)
    with pytest.raises(ValueError, match="unknown settings"):
        Segmenter(
            experiment,
            channel="mCherry",
            gpu=False,
            hypotheses={"x": {"model": "bact_fluor_omni", "gian": 2}},
        )
    with pytest.raises(ValueError, match="one of 'model' or 'checkpoint'"):
        Segmenter(
            experiment, channel="mCherry", gpu=False, hypotheses={"x": {"gain": 2}}
        )


@pytest.mark.parametrize("where", ["pool", "dask"])
def test_workers_elsewhere_write_the_same_labels(
    experiment, fake_model, tmp_path, monkeypatch, where
):
    """Pool and Dask workers (other processes, their own fake model) write what this
    process writes; Dask workers get the checkpoints as bytes."""
    seg = _ensemble(
        experiment, _models(tmp_path), params={"masks": "eval", "min_contrast": "off"}
    )
    seg.segment(show_progress=False)
    root = eg.open_experiment(experiment, mode="r+")
    expected = {
        t: [a[...] for a in eg.segmentation(root, t, "ens")[1]] for t in TRENCHES
    }
    for trench in TRENCHES:
        eg.remove_segmentation(root, trench, "ens")

    seg._loader = "fake_experiment:load_fake_model"
    worker_tmp = tmp_path / "worker_tmp"
    worker_tmp.mkdir()
    monkeypatch.setenv("TMPDIR", str(worker_tmp))  # read by the workers started below
    if where == "pool":
        seg.segment(workers=2, show_progress=False)
    else:
        pytest.importorskip("dask.distributed")
        from pymmm.parallel import local_cluster

        cluster, client = local_cluster(2, dashboard_address=None)
        try:
            seg.segment(show_progress=False)
        finally:
            client.close()
            cluster.close()
        sent = sorted(p.name for p in (worker_tmp / "pymmm_models").glob("*/*"))
        assert sent == ["omni_epoch_250", "omni_epoch_50"]
    assert seg.finished() == TRENCHES
    for trench in TRENCHES:
        for want, got in zip(expected[trench], eg.segmentation(root, trench, "ens")[1]):
            np.testing.assert_array_equal(got[...], want)


def test_at_most_two_trenches_per_worker_are_in_flight(segmenter, monkeypatch):
    from pymmm.parallel import Runner

    state = {"open": 0, "most": 0}
    submit, wait_first = Runner.submit, Runner.wait_first

    def counting_submit(self, fn, *args):
        state["open"] += 1
        state["most"] = max(state["most"], state["open"])
        return submit(self, fn, *args)

    def counting_wait(self, futures):
        done = wait_first(self, futures)
        state["open"] -= len(done)
        return done

    monkeypatch.setattr(Runner, "submit", counting_submit)
    monkeypatch.setattr(Runner, "wait_first", counting_wait)
    segmenter.segment(show_progress=False)
    assert segmenter.is_complete() and state == {"open": 0, "most": 2}


def test_a_failed_trench_keeps_the_others_and_resume_finishes(segmenter, monkeypatch):
    from pymmm.steps import segmentation as module

    task = module._segment_task
    calls = []

    def failing(recipe, weights, stack, bounds, batch_size):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("CUDA out of memory")
        return task(recipe, weights, stack, bounds, batch_size)

    monkeypatch.setattr(module, "_segment_task", failing)
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        segmenter.segment(show_progress=False)
    assert segmenter.finished() == [10]
    monkeypatch.setattr(module, "_segment_task", task)
    segmenter.segment(show_progress=False)
    assert segmenter.is_complete()


def test_a_worker_with_other_code_refuses_the_task(segmenter):
    import dataclasses

    from pymmm.parallel import Runner
    from pymmm.steps import segmentation as module

    recipe, weights = segmenter._recipe(Runner(workers=1), None)
    with pytest.raises(RuntimeError, match="other segmentation code"):
        module._segment_task(
            dataclasses.replace(recipe, digest="0" * 16),
            weights,
            segmenter.read(10),
            (0.0, 1.0),
            64,
        )


def test_packed_labels_come_back_the_same():
    from pymmm.steps.segmentation import _pack, _unpack

    labels = np.zeros((5, 60, 20), np.uint16)
    labels[:, 10:20, 5:15] = 3
    data, shape = _pack(labels)
    assert len(data) < labels.nbytes / 20
    np.testing.assert_array_equal(_unpack((data, shape)), labels)
