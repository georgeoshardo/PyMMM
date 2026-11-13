"""The stitched view (pymmm.gui.mosaic): lazy frames, placement, culling."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

pytestmark = pytest.mark.view

PIXEL = 0.5
SENSOR = (64, 32)  # width, height
STEP = 40.0  # µm between FOV centres on the stage: 64 px * 0.5 µm = 32 µm wide


@pytest.fixture(scope="module", autouse=True)
def offscreen():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _experiment(n_fovs=5, n_t=4):
    from pymmm.source.stage import StageLayout

    names = [f"A{k:02d}" for k in range(1, n_fovs + 1)]
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 4000, (n_t, n_fovs, 1, SENSOR[1], SENSOR[0]), np.uint16)
    data = xr.DataArray(
        pixels,
        dims=("T", "P", "C", "Y", "X"),
        coords={"P": names, "C": ["PC"]},
    )
    metadata = {
        "experiment": [
            {
                "type": "XYPosLoop",
                "parameters": {
                    "points": [
                        {"stagePositionUm": {"x": -STEP * k, "y": 0.0, "z": 0.0}}
                        for k in range(n_fovs)
                    ]
                },
            }
        ],
        "metadata": {
            "channels": [
                {
                    "volume": {
                        "cameraTransformationMatrix": [-1.0, 0.0, 0.0, -1.0],
                        "voxelCount": [*SENSOR, 1],
                    }
                }
            ]
        },
    }
    return SimpleNamespace(
        data=data,
        fov_names=names,
        stage_layout=StageLayout.from_metadata(metadata, PIXEL),
        source_subset_metadata={"raw_fov_names": names},
        source_roi=(None, None),
        pixel_size_um=PIXEL,
    )


def test_frames_are_read_once_and_reduced_by_block_mean():
    from pymmm.gui.mosaic import REDUCTION, FovFrames, FrameSource, reduce

    exp = _experiment()
    source = FrameSource(exp, "PC")
    full, small = FovFrames(source, 2), FovFrames(source, 2, reduced=True)
    expected = exp.data.sel(C="PC").values[1, 2]
    np.testing.assert_array_equal(full[1], expected)
    np.testing.assert_array_equal(full[1, 3:7, 10:20], expected[3:7, 10:20])
    assert full[0:2].shape == (2, SENSOR[1], SENSOR[0])
    assert small.shape == (4, SENSOR[1] // REDUCTION, SENSOR[0] // REDUCTION)
    np.testing.assert_array_equal(small[1], reduce(expected, REDUCTION))
    assert source.reads == 2  # frame 0 and frame 1, each read once
    block = expected[:REDUCTION, :REDUCTION].mean()
    assert small[1][0, 0] == np.uint16(block)


@pytest.fixture
def mosaic():
    from napari.components import ViewerModel
    from pymmm.gui.mosaic import MosaicView

    viewer = ViewerModel()
    view = MosaicView(viewer, _experiment(), "PC", contrast_limits=(0, 4000))
    yield viewer, view
    view.close()


def test_each_fov_is_placed_on_its_map_outline(mosaic):
    viewer, view = mosaic
    for fov, layer in view.layers.items():
        corners = np.array(
            [[0, -0.5, -0.5], [0, -0.5, SENSOR[0] - 0.5], [0, SENSOR[1] - 0.5, -0.5]]
        )
        world = np.array([layer.data_to_world(c) for c in corners])[:, 1:][:, ::-1]
        outline = view.outlines[fov]
        np.testing.assert_allclose(world, outline[[0, 1, 3]], atol=1e-9)
    assert viewer.dims.axis_labels == ("T", "y", "x")


def test_only_fovs_in_view_load_and_contrast_is_shared(mosaic):
    _, view = mosaic
    view.show_all()
    assert view.visible_fovs == list(view.layers)
    view.go_to(*view.outlines["A03"].mean(axis=0))  # zoomed out: zooms to A03
    assert set(view.visible_fovs) <= {"A02", "A03", "A04"}
    assert "A03" in view.visible_fovs and "A05" not in view.visible_fovs
    x0, y0, x1, y1 = view.view_rect()
    assert view.fov_at((x0 + x1) / 2, (y0 + y1) / 2) == "A03"
    view.layers["A01"].contrast_limits = (100, 2000)
    assert all(
        tuple(lay.contrast_limits) == (100, 2000) for lay in view.layers.values()
    )


def test_closing_removes_every_layer(mosaic):
    viewer, view = mosaic
    view.set_marks([np.array([[0.0, 5.0], [30.0, 5.0]])], [1], [])
    assert len(viewer.layers) == 6
    view.close()
    assert len(viewer.layers) == 0
