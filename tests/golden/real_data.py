"""The golden subset of real data: two FOVs and ten frames of the ty013 ND2, and the
settings the experiment was processed with (its companion store,
``raw_data/20261005_pbs_psc_yfp_ty013.pymmm.zarr``). Set ``PYMMM_GOLDEN_ND2`` to use
another copy of the ND2.
"""

from __future__ import annotations

import os
from pathlib import Path

from pymmm import ND2Experiment

ND2 = Path(os.environ.get(
    "PYMMM_GOLDEN_ND2", "/data/phage-signatures/raw_data/20261005_pbs_psc_yfp_ty013.nd2"))

# Not the first FOV nor the first frame, so FOV and frame indices of the subset differ
# from the ND2's; frames 20-29 drift by up to 2 px.
FOVS = ["XYPos:2", "XYPos:5"]
TIMES = (20, 30)

REGISTRATION = dict(
    registration_channel="PC", mode="first", rotation=0.0, roi={"y": (250, 800)},
    mean_n_frames=10, mean_from="end", backend="stackreg", downsample=4,
    batch_size=4, fov_window=2, opencv_threads=1,
)
LANES = dict(sigma=48.0, distance=1000.0, height=5000.0, orientation_window_um=20.0,
             orientation="down")
TRENCHES = dict(sigma=4.0, distance=100.0, prominence=10.0, height=0, trench_width=94,
                trench_length=655, trench_bottom_offset=50, method="periodic",
                expected_pitch=0.0, min_symmetry=0.1)
EXTRACTION = dict(clevel=3, byte_shuffle=True, batch_size=4, time_chunk=None)


def no_dask_client() -> bool:
    try:
        from dask.distributed import get_client
        get_client()
    except (ImportError, ValueError):
        return True
    return False


def open_nd2(folder: Path) -> ND2Experiment:
    """The golden subset of the ND2, through ``folder``'s symlink to it."""
    exp = ND2Experiment(folder / ND2.name)
    return exp.select_fovs(FOVS).select_times(*TIMES)
