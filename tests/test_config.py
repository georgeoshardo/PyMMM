from __future__ import annotations

import pytest

from pymmm.config import Experiment
from pymmm.segmenter import Segmenter
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


def test_experiment_builds_the_segmenter_from_its_settings(tmp_path, monkeypatch):
    make_experiment(tmp_path)
    (tmp_path / "exp.toml").write_text(
        'name = "exp"\nnd2 = "exp.nd2"\nstore = "exp.zarr"\n'
        '[segmentation]\nchannel = "mCherry"\nname = "mc"\nhypotheses = ["S1", "S3"]\n'
    )
    segmenter = Experiment.load(tmp_path / "exp.toml").segmenter(gpu=False)
    assert isinstance(segmenter, Segmenter)
    assert segmenter.store_path == tmp_path / "exp.zarr"
    assert (segmenter.channel, segmenter.name, segmenter.hypotheses) == ("mCherry", "mc", ["S1", "S3"])
    assert segmenter.model_name == "bact_fluor_omni"
