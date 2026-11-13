"""Where each FOV sat on the stage: the ND2's XY points and its camera matrix, to lay the
FOVs out as a map of the experiment.

A pixel ``p`` (x, y) of a FOV's registered frame lies on the stage at::

    stage = position + pixel_size * C @ (R @ (p - c_crop) + c_crop + crop_offset - c_sensor)

``C`` is the camera matrix (the ND2's ``cameraTransformationMatrix``, image axes to
stage axes), ``R`` the registration's rotation about the crop's centre, ``c_sensor``
the centre of the whole sensor, which is the point the stage position names. Frame
drift (a few pixels) is left out.

:meth:`StageLayout.origin` puts this on the map's axes: the stage turned back by
``C`` and ``R``, so a FOV's registered frame is upright there (as in the viewer) and
FOVs that touch on the stage touch on the map.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


def rotation(degrees: float) -> np.ndarray:
    """The 2 × 2 rotation the registration applies (its inverse map, as for
    ``skimage.transform.rotate``: an output pixel reads the input at ``R @ p``).
    """
    theta = np.deg2rad(degrees)
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


@dataclass(frozen=True)
class StageLayout:
    """Every FOV's stage position and how the camera sits on the stage.

    Attributes
    ----------
    positions : np.ndarray
        (n_fovs, 2) stage x, y in µm, by raw P index (every FOV in the ND2).
    camera : np.ndarray
        2 × 2, image (x, y) to stage (x, y).
    pixel_size : float
        µm per pixel.
    sensor : tuple[int, int]
        Width and height of the whole frame, in pixels.
    """

    positions: np.ndarray
    camera: np.ndarray
    pixel_size: float
    sensor: tuple[int, int]

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any], pixel_size: float):
        """From an ND2's acquisition metadata
        (:attr:`~pymmm.ND2Experiment.source_acquisition_metadata`); ``None`` without an
        XY loop, a camera matrix or a position for every point.
        """
        loops = metadata.get("experiment") or []
        xy = [loop for loop in loops if loop.get("type") == "XYPosLoop"]
        try:
            volume = metadata["metadata"]["channels"][0]["volume"]
            camera = np.asarray(volume["cameraTransformationMatrix"], float).reshape(
                2, 2
            )
            sensor = (int(volume["voxelCount"][0]), int(volume["voxelCount"][1]))
            points = xy[0]["parameters"]["points"]
            positions = np.array(
                [
                    [float(p["stagePositionUm"]["x"]), float(p["stagePositionUm"]["y"])]
                    for p in points
                ]
            )
        except (KeyError, IndexError, TypeError, ValueError):
            return None
        if (
            not len(positions)
            or not np.all(np.isfinite(positions))
            or not np.all(np.isfinite(camera))
            or abs(np.linalg.det(camera)) < 1e-6
        ):
            return None
        return cls(positions, camera, float(pixel_size), sensor)

    def _to_map(self, degrees: float) -> np.ndarray:
        """Stage µm to map µm: undo the camera, then the registration's rotation."""
        return rotation(degrees).T @ np.linalg.inv(self.camera)

    def origin(
        self,
        raw_p: int,
        degrees: float = 0.0,
        crop_offset: Sequence[int] = (0, 0),
        crop_shape: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Where pixel (0, 0) of a FOV's registered frame sits on the map (µm); pixel
        ``p`` sits at ``origin + pixel_size * p``.

        Parameters
        ----------
        raw_p : int
            The FOV's P index in the ND2.
        degrees : float
            The registration's rotation.
        crop_offset : (x, y)
            Where the experiment's crop starts on the sensor, in pixels.
        crop_shape : (width, height)
            The crop's size; the whole sensor by default.
        """
        width, height = crop_shape if crop_shape is not None else self.sensor
        c_crop = np.array([width / 2 - 0.5, height / 2 - 0.5])
        c_sensor = np.array([self.sensor[0] / 2 - 0.5, self.sensor[1] / 2 - 0.5])
        turn = rotation(degrees).T
        return (
            self._to_map(degrees) @ self.positions[raw_p]
            + self.pixel_size
            * (turn @ (c_crop + np.asarray(crop_offset, float) - c_sensor))
            - self.pixel_size * c_crop
        )

    def outline(
        self,
        raw_p: int,
        degrees: float = 0.0,
        crop_offset: Sequence[int] = (0, 0),
        crop_shape: Sequence[int] | None = None,
    ) -> np.ndarray:
        """The corners (4, 2) of the area a FOV's frames cover, on the map (µm): the
        crop of the sensor, turned by the registration's rotation.
        """
        width, height = crop_shape if crop_shape is not None else self.sensor
        corners = np.array(
            [[0, 0], [width, 0], [width, height], [0, height]], float
        ) - np.array([0.5, 0.5])
        c_crop = np.array([width / 2 - 0.5, height / 2 - 0.5])
        # a raw pixel q of the crop shows at registered p = c + R^T (q - c)
        registered = (rotation(degrees).T @ (corners - c_crop).T).T + c_crop
        return (
            self.origin(raw_p, degrees, crop_offset, crop_shape)
            + self.pixel_size * registered
        )


def fov_frames(experiment: Any, degrees: float = 0.0) -> dict[str, tuple] | None:
    """Every FOV of an experiment's ND2 on the map: ``{name: (origin, outline)}``
    (:meth:`StageLayout.origin`, :meth:`StageLayout.outline`) for its crop and a
    registration rotated by ``degrees``; ``None`` when the ND2 has no stage layout.
    FOVs left out of the experiment are there too.
    """
    layout = experiment.stage_layout
    names = experiment.source_subset_metadata["raw_fov_names"]
    if layout is None or len(names) != len(layout.positions):
        return None
    y_roi, x_roi = experiment.source_roi
    offset = ((x_roi[0] or 0) if x_roi else 0, (y_roi[0] or 0) if y_roi else 0)
    shape = (int(experiment.data.sizes["X"]), int(experiment.data.sizes["Y"]))
    return {
        str(name): (
            layout.origin(raw_p, degrees, offset, shape),
            layout.outline(raw_p, degrees, offset, shape),
        )
        for raw_p, name in enumerate(names)
    }
