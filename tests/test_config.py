from __future__ import annotations

import pytest
from pymmm.config import Experiment
from pymmm.steps.segmentation import Segmenter

from fake_experiment import make_experiment


def test_experiment_resolves_relative_paths(tmp_path):
    (tmp_path / "exp.toml").write_text(
        'name = "exp"\nnd2 = "raw/exp.nd2"\nstore = "/abs/exp.zarr"\n'
    )
    experiment = Experiment.load(tmp_path / "exp.toml")
    assert experiment.nd2 == tmp_path / "raw" / "exp.nd2"
    assert experiment.store.as_posix() == "/abs/exp.zarr"
    assert experiment.segmentation_name == "mcherry"
    (tmp_path / "old.toml").write_text(
        'name = "x"\nnd2 = "x.nd2"\ntrench_store = "x.trenches.zarr"\n'
    )
    with pytest.raises(ValueError, match="missing store"):
        Experiment.load(tmp_path / "old.toml")


def test_experiment_without_an_nd2(tmp_path):
    """An experiment group imported from elsewhere has no ND2: nd2 is None."""
    (tmp_path / "exp.toml").write_text('name = "exp"\nstore = "exp.zarr"\n')
    experiment = Experiment.load(tmp_path / "exp.toml")
    assert experiment.nd2 is None
    assert experiment.store == tmp_path / "exp.zarr"


def test_experiment_builds_the_segmenter_from_its_settings(tmp_path, monkeypatch):
    make_experiment(tmp_path)
    (tmp_path / "exp.toml").write_text(
        'name = "exp"\nnd2 = "exp.nd2"\nstore = "exp.zarr"\n'
        '[segmentation]\nchannel = "mCherry"\nname = "mc"\nhypotheses = ["S1", "S3"]\n'
    )
    segmenter = Experiment.load(tmp_path / "exp.toml").segmenter(gpu=False)
    assert isinstance(segmenter, Segmenter)
    assert segmenter.store_path == tmp_path / "exp.zarr"
    assert (segmenter.channel, segmenter.name, segmenter.hypotheses) == (
        "mCherry",
        "mc",
        ["S1", "S3"],
    )
    assert segmenter.model_name == "bact_fluor_omni"


def test_experiment_builds_a_checkpoint_segmenter(tmp_path, fake_model):
    make_experiment(tmp_path)
    models = tmp_path / "models"
    models.mkdir()
    (models / "omni_epoch_250").write_bytes(b"weights")
    (tmp_path / "exp.toml").write_text(
        'name = "exp"\nnd2 = "exp.nd2"\nstore = "exp.zarr"\n'
        '[segmentation]\nchannel = "mCherry"\nname = "ens"\nmodels = "models"\n'
        'checkpoints = {e250 = "omni_epoch_250"}\n'
        'params = {masks = "affinity", floor = 0.0, min_contrast = "off"}\n'
        "model_pixel_size = 0.2\n"
    )
    segmenter = Experiment.load(tmp_path / "exp.toml").segmenter(gpu=False)
    assert segmenter.hypotheses == ["e250"]
    assert segmenter.model_pixel_size == 0.2
    assert segmenter.checkpoint_path("e250") == (models / "omni_epoch_250").resolve()
    assert (
        segmenter.params["masks"] == "affinity"
        and segmenter.params["min_contrast"] is None
    )
    assert segmenter.params["floor"] == 0.0
