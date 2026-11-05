"""A small experiment group laid out as the Extractor writes it, for the tests of the
steps after extraction."""

from __future__ import annotations

import numpy as np
import zarr

from pymmm import trench_images as ti
from pymmm.store import ome

N_TRENCH, N_T, H, W = 3, 4, 12, 6
TRENCHES = [10, 11, 12]
BACKGROUND, NOISE, CELL = 100, 5, 1000
PIXEL_SIZE = 0.1


def cell_image() -> np.ndarray:
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


def make_experiment(tmp_path):
    """An experiment group laid out as the Extractor writes it: trenches 10–12, each an
    OME-Zarr image (t, c, y, x) of PC (empty) and mCherry (:func:`cell_image`)."""
    path = tmp_path / "exp.zarr"
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs["pymmm"] = {
        "layout": "experiment", "pixel_size_um": PIXEL_SIZE, "frame_interval_ms": 60_000.0,
        "provenance": {"created": "2026-01-01T00:00:00"},
    }
    root.create_array("C", shape=(2,), dtype=str)[:] = np.asarray(["PC", "mCherry"], dtype=object)
    root.create_array("T", data=np.arange(5, 5 + N_T, dtype=np.int32))
    table = root.create_group("trenches")
    table.create_array("trench", data=np.asarray(TRENCHES, dtype=np.int32))
    table.create_array("fov_name", shape=(N_TRENCH,), dtype=str)[:] = np.asarray(
        ["f0", "f0", "f1"], dtype=object)
    cells = cell_image()
    for k, trench in enumerate(TRENCHES):
        group = root.create_group(ti.trench_path(trench))
        group.attrs["ome"] = {
            "version": ome.OME_VERSION,
            "multiscales": [{
                "name": ti.trench_path(trench),
                "axes": [{"name": "t", "type": "time", "unit": "minute"},
                         {"name": "c", "type": "channel"},
                         {"name": "y", "type": "space", "unit": "micrometer"},
                         {"name": "x", "type": "space", "unit": "micrometer"}],
                "datasets": [{"path": "0", "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 1.0, PIXEL_SIZE, PIXEL_SIZE]}]}],
            }],
            "omero": {"channels": [{"label": "PC", "color": "FFFFFF"},
                                   {"label": "mCherry", "color": "FF0000"}]},
        }
        pixels = np.zeros((N_T, 2, H, W), dtype=np.uint16)
        pixels[:, 1] = cells[k]
        group.create_array("0", data=pixels, chunks=(N_T, 1, H, W),
                           dimension_names=["t", "c", "y", "x"])
    return path
