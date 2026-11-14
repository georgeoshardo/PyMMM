"""Fixtures of the golden tests: the real-data pipeline (see real_data.py), run once.

The ND2 is reached through a symlink in a scratch folder, so the companion store and
everything else land in scratch, never beside the ND2. Without the ND2 the golden
tests are skipped.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pymmm import (
    CompanionStore,
    Extractor,
    LaneDetector,
    Registrator,
    ShadingCorrector,
    TrenchDetector,
)
from real_data import (
    EXTRACTION,
    LANES,
    ND2,
    REGISTRATION,
    SHADING,
    TRENCHES,
    no_dask_client,
    open_nd2,
)


@pytest.fixture(scope="session")
def scratch_folder(tmp_path_factory):
    """Make scratch folders, each with a symlink to the ND2."""
    if not ND2.exists():
        pytest.skip(f"golden ND2 not found: {ND2}")

    def make(name: str) -> Path:
        folder = tmp_path_factory.mktemp(name)
        (folder / ND2.name).symlink_to(ND2)
        return folder

    return make


@pytest.fixture(scope="session")
def pipeline(scratch_folder):
    """Registration (one process), lanes, trenches, shading and extraction (process
    pool), each saved, as the extraction notebook runs them."""
    assert no_dask_client()
    folder = scratch_folder("pipeline")
    exp = open_nd2(folder)
    store = CompanionStore.for_experiment(exp)
    assert store.path.parent == folder

    reg = Registrator(exp, store, **REGISTRATION)
    reg.compute_tmats(n_jobs=1)
    reg.save()
    lanes = LaneDetector(exp, reg, store, detection_channel="PC")
    lanes.detect_lanes(**LANES, save_diagnostics=False)
    lanes.save()
    trenches = TrenchDetector(exp, reg, lanes, store, detection_channel="PC")
    trenches.detect_trenches(**TRENCHES, save_diagnostics=False)
    trenches.save()
    shading = ShadingCorrector(exp, reg, trenches, store, **SHADING)
    shading.fit()
    shading.save()
    extractor = Extractor(
        exp, reg, trenches, shading, output_path=folder / "experiment.zarr"
    )
    extractor.extract(**EXTRACTION, resume=False, show_progress=False)
    return SimpleNamespace(
        folder=folder,
        exp=exp,
        store=store,
        reg=reg,
        lanes=lanes,
        trenches=trenches,
        shading=shading,
        extractor=extractor,
        group=folder / "experiment.zarr",
    )


@pytest.fixture(scope="module")
def dask_cluster():
    distributed = pytest.importorskip("dask.distributed")
    with distributed.LocalCluster(
        n_workers=2, threads_per_worker=1, processes=True, dashboard_address=None
    ) as cluster:
        yield cluster


@pytest.fixture
def dask_client(dask_cluster):
    """A client for one test only: PyMMM uses any active client it finds."""
    from dask.distributed import Client

    with Client(dask_cluster) as client:
        yield client
    assert no_dask_client()
