"""Golden test of the Segmenter on real trenches: Omnipose ``bact_fluor_omni`` on the
CPU (the same labels on every machine), every hypothesis, the segmentation's record
and label images. Needs omnipose (the ``seg`` environment).
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest
from golden_io import check, manifest, scrub

from pymmm import trench_images as ti
from pymmm.segmenter import Segmenter

pytest.importorskip("omnipose")
pytestmark = pytest.mark.golden

SEGMENTED = [2, 8]  # crowded (about 14 cells), empty


def test_segmenter(pipeline, tmp_path):
    group = tmp_path / "experiment.zarr"
    shutil.copytree(pipeline.group, group)
    segmenter = Segmenter(group, channel="mCherry", model="bact_fluor_omni", gpu=False)
    segmenter.segment(trenches=SEGMENTED, show_progress=False)

    root = ti.open_experiment(group)
    cells = {}
    for trench in SEGMENTED:
        names, arrays = ti.segmentation(root, trench, segmenter.name)
        for name, labels in zip(names, arrays):
            cells[f"{trench}/{name}"] = [int(frame.max()) for frame in labels[...]]
            assert all(np.unique(frame).size == frame.max() + 1 for frame in labels[...])
    check("segmentation", {
        "params": segmenter.params,
        "hypotheses": segmenter.hypotheses,
        "finished": segmenter.finished(),
        "bounds": {trench: segmenter.bounds(trench) for trench in SEGMENTED},
        "record": scrub(ti.records(root, "segmentations")),
        "cells_per_frame": cells,
        "labels": {trench: manifest(group / ti.trench_path(trench) / "labels")
                   for trench in SEGMENTED},
    })
