"""FOVs on the stage (pymmm.source.stage): the ND2's XY points and camera matrix."""

from __future__ import annotations

import numpy as np
import pytest
from pymmm.source.stage import StageLayout, rotation

PIXEL = 0.1
SENSOR = (200, 100)  # width, height


def _metadata(points, camera=(-1.0, 0.0, 0.0, -1.0)):
    return {
        "experiment": [
            {"type": "TimeLoop"},
            {
                "type": "XYPosLoop",
                "parameters": {
                    "points": [
                        {"stagePositionUm": {"x": x, "y": y, "z": 0.0}}
                        for x, y in points
                    ]
                },
            },
        ],
        "metadata": {
            "channels": [
                {
                    "volume": {
                        "cameraTransformationMatrix": list(camera),
                        "voxelCount": [*SENSOR, 1],
                    }
                }
            ]
        },
    }


def _stage(layout, raw_p, p, degrees, offset=(0, 0), shape=SENSOR):
    """Where registered pixel ``p`` of a FOV lies on the stage (the module's model)."""
    c_crop = np.array([shape[0] / 2 - 0.5, shape[1] / 2 - 0.5])
    c_sensor = np.array([SENSOR[0] / 2 - 0.5, SENSOR[1] / 2 - 0.5])
    raw = rotation(degrees) @ (np.asarray(p, float) - c_crop) + c_crop
    return layout.positions[raw_p] + layout.pixel_size * layout.camera @ (
        raw + np.asarray(offset, float) - c_sensor
    )


def test_fovs_that_touch_on_the_stage_touch_on_the_map():
    # the second FOV one frame width further along stage x; the camera's x runs the
    # other way, so on the map (image axes) it sits to the right of the first
    layout = StageLayout.from_metadata(_metadata([(100.0, 50.0), (80.0, 50.0)]), PIXEL)
    first, second = layout.outline(0), layout.outline(1)
    np.testing.assert_allclose(second[0], first[1])  # top left = top right
    np.testing.assert_allclose(np.ptp(first, axis=0), [20.0, 10.0])


@pytest.mark.parametrize("degrees", [0.0, 2.5, -1.9])
@pytest.mark.parametrize("offset, shape", [((0, 0), SENSOR), ((30, 10), (120, 60))])
def test_a_pixel_sits_on_the_map_where_it_sat_on_the_stage(degrees, offset, shape):
    camera = (-0.999, 0.036, -0.036, -0.999)  # turned about 2 degrees
    layout = StageLayout.from_metadata(_metadata([(10.0, 20.0)], camera), PIXEL)
    to_map = rotation(degrees).T @ np.linalg.inv(layout.camera)
    origin = layout.origin(0, degrees, offset, shape)
    for p in ([0.0, 0.0], [17.0, 33.0], [shape[0] - 1.0, shape[1] - 1.0]):
        expected = to_map @ _stage(layout, 0, p, degrees, offset, shape)
        np.testing.assert_allclose(origin + PIXEL * np.asarray(p), expected, atol=1e-9)


def test_a_crop_is_where_it_was_cut_from():
    layout = StageLayout.from_metadata(_metadata([(0.0, 0.0)]), PIXEL)
    whole = layout.origin(0)
    cropped = layout.origin(0, 0.0, (30, 10), (120, 60))
    # crop pixel (0, 0) is sensor pixel (30, 10)
    np.testing.assert_allclose(cropped, whole + PIXEL * np.array([30, 10]))


def test_no_layout_without_positions_or_a_camera():
    assert StageLayout.from_metadata({}, PIXEL) is None
    no_xy = _metadata([(0.0, 0.0)])
    no_xy["experiment"] = [{"type": "TimeLoop"}]
    assert StageLayout.from_metadata(no_xy, PIXEL) is None
    assert StageLayout.from_metadata(_metadata([(np.nan, 0.0)]), PIXEL) is None
    flat = _metadata([(0.0, 0.0)], camera=(0.0, 0.0, 0.0, 0.0))
    assert StageLayout.from_metadata(flat, PIXEL) is None
