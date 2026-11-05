from __future__ import annotations

import pytest

from pymmm.config import Experiment


def test_experiment_resolves_relative_paths(tmp_path):
    (tmp_path / "exp.toml").write_text(
        'name = "exp"\nnd2 = "raw/exp.nd2"\ntrench_store = "exp.trenches.zarr"\n'
        'tracks_store = "exp.tracks.zarr"\ncuration_store = "/abs/exp.curation.zarr"\n'
    )
    experiment = Experiment.load(tmp_path / "exp.toml")
    assert experiment.nd2 == tmp_path / "raw" / "exp.nd2"
    assert experiment.curation_store.as_posix() == "/abs/exp.curation.zarr"
    assert experiment.segmentation_name == "mcherry"
    with pytest.raises(ValueError, match="missing"):
        (tmp_path / "bad.toml").write_text('name = "x"\n')
        Experiment.load(tmp_path / "bad.toml")
