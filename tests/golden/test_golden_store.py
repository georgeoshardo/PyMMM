"""Golden tests of the experiment group's writers (``pymmm.store.experiment_group``,
``pymmm.store.geff``) on a real trench: label images, a segmentation's hypotheses, a
tracking run's masks and GEFF, and the records of both.

The labels come from thresholding the trench's mCherry frames, not from a model, so
these tests pin the writers alone. The GEFF test needs geff (the ``seg`` environment).
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest
from golden_io import check, digest, manifest
from scipy import ndimage

from pymmm.steps.segmentation import order_from_closed_end
from pymmm.store import experiment_group as eg
from pymmm.store.geff import (
    GEFF_NAME,
    has_tracks,
    read_trench_geff,
    tracklets_and_lineages,
    write_tracks,
)

pytestmark = pytest.mark.golden

TRENCH = 2
THRESHOLD = 374  # about the Segmenter's floor for mCherry


@pytest.fixture
def group(pipeline, tmp_path):
    path = tmp_path / "experiment.zarr"
    shutil.copytree(pipeline.group, path)
    return path


def threshold_labels(root) -> np.ndarray:
    """``(t, y, x)`` uint16 labels of the bright mCherry blobs, 1 nearest the closed end."""
    bright = eg.image(root, TRENCH)[:, eg.channel_index(root, "mCherry")] > THRESHOLD
    return order_from_closed_end(np.stack([
        ndimage.label(ndimage.binary_opening(frame, iterations=2))[0] for frame in bright]))


def graph(labels: np.ndarray):
    """Nodes (one per mask) and edges: each label to the same label in the next frame,
    and the last cell to the new one when the count goes up (a division)."""
    t, y, x, seg_id, index = [], [], [], [], {}
    for frame, frame_labels in enumerate(labels):
        ids = np.unique(frame_labels)[1:]
        centres = ndimage.center_of_mass(frame_labels > 0, frame_labels, ids)
        for label, (cy, cx) in zip(ids.tolist(), centres):
            index[frame, label] = len(t)
            t.append(frame)
            y.append(cy)
            x.append(cx)
            seg_id.append(label)
    edges = []
    for frame in range(len(labels) - 1):
        now = [label for (f, label) in index if f == frame]
        later = [label for (f, label) in index if f == frame + 1]
        edges += [(index[frame, label], index[frame + 1, label]) for label in now
                  if label in later]
        if now and len(later) > len(now) and len(now) + 1 in later:
            edges.append((index[frame, len(now)], index[frame + 1, len(now) + 1]))
    nodes = {"t": np.asarray(t, np.int32), "y": np.asarray(y, np.float32),
             "x": np.asarray(x, np.float32), "seg_id": np.asarray(seg_id, np.int32)}
    return nodes, np.asarray(edges, np.int64).reshape(-1, 2)


def test_label_images(group):
    root = eg.open_experiment(group, mode="r+")
    labels = threshold_labels(root)
    merged = np.where(labels == 2, 1, labels).astype(np.uint16)  # cells 1 and 2 as one
    eg.update_experiment(root, "segmentations", "golden",
                         {"channel": "mCherry", "hypotheses": ["A", "B"]})
    eg.write_segmentation(root, TRENCH, "golden", {"A": labels, "B": merged},
                          attrs={"lower": 100.0, "upper": 900.0})
    eg.write_label_image(root, TRENCH, "unlisted", labels[:, ::-1], listed=False,
                         attrs={"note": "flipped"})

    names, arrays = eg.segmentation(root, TRENCH, "golden")
    np.testing.assert_array_equal(arrays[0][...], labels)
    check("store_label_images", {
        "hypotheses": names,
        "listed": eg.listed_labels(root, TRENCH),
        "has_segmentation": {t: eg.has_segmentation(root, t, "golden") for t in (TRENCH, 3)},
        "records": eg.records(root, "segmentations"),
        "trench": manifest(group / eg.trench_path(TRENCH)),
    })

    eg.remove_segmentation(root, TRENCH, "golden")
    check("store_label_images_removed", {
        "listed": eg.listed_labels(root, TRENCH),
        "trench": manifest(group / eg.trench_path(TRENCH)),
    })


@pytest.mark.tracks
def test_tracks(group, tmp_path):
    root = eg.open_experiment(group, mode="r+")
    labels = threshold_labels(root)
    nodes, edges = graph(labels)
    track_id, lineage_id = tracklets_and_lineages(len(nodes["t"]), nodes["t"], edges)
    scores = np.linspace(0.0, 1.0, len(edges), dtype=np.float32)
    scores[0] = np.nan
    path = write_tracks(root, TRENCH, "golden",
                           {**nodes, "track_id": track_id, "lineage_id": lineage_id},
                           edges, scores, labels, extra={"tracker": "golden test"})
    eg.update_experiment(root, "tracks", "golden", {"tracker": "golden test"})

    legacy = tmp_path / "legacy"  # read_trench_geff reads the earlier per-trench layout
    legacy.mkdir()
    (legacy / GEFF_NAME).symlink_to(path)
    check("store_tracks", {
        "geff": path.relative_to(group).as_posix(),
        "has_tracks": {t: has_tracks(root, t, "golden") for t in (TRENCH, 3)},
        "n_nodes": len(nodes["t"]), "n_edges": len(edges),
        "track_id": track_id, "lineage_id": lineage_id,
        "read_trench_geff": {key: digest(value)
                             for key, value in read_trench_geff(legacy).items()},
        "records": eg.records(root, "tracks"),
        "trench": manifest(group / eg.trench_path(TRENCH)),
    })
