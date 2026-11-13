"""Fakes shared by the tests: each step's inputs, small enough to check by hand.

- :func:`make_experiment`, :func:`make_empty_experiment`: experiment groups laid out as
  the Extractor writes them, for the steps after extraction and the store.
- :class:`FakeND2File`, :func:`fake_nd2`: an ND2 file of 2 frames, 2 FOVs and 2
  channels, for the source and the extraction.
- :func:`lane_image`, :func:`fov_experiment`, :func:`fov_registrator`: registered FOVs,
  for the lane and trench detectors and their widgets.
- :class:`FakeModel`: Omnipose's model, thresholding the image, for the Segmenter.
"""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import xarray as xr
import zarr
from pymmm.store import experiment_group as eg
from pymmm.store import ome
from scipy import ndimage

# ----------------------------------------------------------------------------
# Experiment groups
# ----------------------------------------------------------------------------
N_TRENCH, N_T, H, W = 3, 4, 12, 6
TRENCHES = [10, 11, 12]
BACKGROUND, NOISE, CELL = 100, 5, 1000
PIXEL_SIZE = 0.1


def cell_image() -> np.ndarray:
    """mCherry frames: two cells per trench, trench 1 empty at T=2, trench 2 noise only.

    Trench 0 has a zero-filled column (outside the camera image).
    """
    rng = np.random.default_rng(0)
    image = np.rint(rng.normal(BACKGROUND, NOISE, (N_TRENCH, N_T, H, W))).astype(
        np.uint16
    )
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
        "layout": "experiment",
        "pixel_size_um": PIXEL_SIZE,
        "frame_interval_ms": 60_000.0,
        "provenance": {"created": "2026-01-01T00:00:00"},
    }
    root.create_array("C", shape=(2,), dtype=str)[:] = np.asarray(
        ["PC", "mCherry"], dtype=object
    )
    root.create_array("T", data=np.arange(5, 5 + N_T, dtype=np.int32))
    table = root.create_group("trenches")
    table.create_array("trench", data=np.asarray(TRENCHES, dtype=np.int32))
    table.create_array("fov_name", shape=(N_TRENCH,), dtype=str)[:] = np.asarray(
        ["f0", "f0", "f1"], dtype=object
    )
    cells = cell_image()
    for k, trench in enumerate(TRENCHES):
        group = root.create_group(eg.trench_path(trench))
        group.attrs["ome"] = {
            "version": ome.OME_VERSION,
            "multiscales": [
                {
                    "name": eg.trench_path(trench),
                    "axes": [
                        {"name": "t", "type": "time", "unit": "minute"},
                        {"name": "c", "type": "channel"},
                        {"name": "y", "type": "space", "unit": "micrometer"},
                        {"name": "x", "type": "space", "unit": "micrometer"},
                    ],
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                {
                                    "type": "scale",
                                    "scale": [1.0, 1.0, PIXEL_SIZE, PIXEL_SIZE],
                                }
                            ],
                        }
                    ],
                }
            ],
            "omero": {
                "channels": [
                    {"label": "PC", "color": "FFFFFF"},
                    {"label": "mCherry", "color": "FF0000"},
                ]
            },
        }
        pixels = np.zeros((N_T, 2, H, W), dtype=np.uint16)
        pixels[:, 1] = cells[k]
        group.create_array(
            "0",
            data=pixels,
            chunks=(N_T, 1, H, W),
            dimension_names=["t", "c", "y", "x"],
        )
    return path


def make_empty_experiment(tmp_path):
    """A two-trench experiment group as the Extractor lays it out: trenches 3 and 4, each
    5 frames of PC and mCherry, 6 × 4 pixels, left empty."""
    path = tmp_path / "exp.zarr"
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs["pymmm"] = {
        "layout": "experiment",
        "pixel_size_um": 0.1,
        "frame_interval_ms": 60_000.0,
    }
    root.create_array("C", shape=(2,), dtype=str)[:] = np.asarray(
        ["PC", "mCherry"], dtype=object
    )
    table = root.create_group("trenches")
    table.create_array("trench", data=np.asarray([3, 4], dtype=np.int32))
    for t in (3, 4):
        root.create_group(eg.trench_path(t)).create_array(
            "0", shape=(5, 2, 6, 4), dtype=np.uint16
        )
    return path


# ----------------------------------------------------------------------------
# An ND2 file
# ----------------------------------------------------------------------------
@dataclass
class FakeChannelMeta:
    name: str
    index: int
    color: dict | None = None
    emissionLambdaNm: float | None = None
    excitationLambdaNm: float | None = None


@dataclass
class FakeChannelRecord:
    channel: FakeChannelMeta
    microscope: dict


@dataclass
class FakeMetadata:
    channels: list[FakeChannelRecord]


@dataclass
class FakeLoop:
    count: int
    type: str


@dataclass
class FakeStagePosition:
    x: float
    y: float
    z: float


@dataclass
class FakePosition:
    stagePositionUm: FakeStagePosition
    pfsOffset: float | None
    name: str | None


@dataclass
class FakeTimeStamp:
    absoluteJulianDayNumber: float
    relativeTimeMs: float


@dataclass
class FakeFrameChannel:
    channel: FakeChannelMeta
    loops: dict[str, int]
    microscope: dict[str, str]
    volume: dict[str, list[float]]
    position: FakePosition
    time: FakeTimeStamp


@dataclass
class FakeFrameMetadata:
    contents: dict[str, int]
    channels: list[FakeFrameChannel]


class FakeND2File:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.sizes = {"T": 2, "P": 2, "C": 2, "Y": 3, "X": 4}
        self.shape = (2, 2, 2, 3, 4)
        self.dtype = np.dtype("uint16")
        self.is_legacy = False
        microscope = {
            "objectiveName": "Plan Apo 40x",
            "objectiveNumericalAperture": 0.95,
            "modalityFlags": ["camera"],
        }
        self.metadata = FakeMetadata(
            channels=[
                FakeChannelRecord(
                    channel=FakeChannelMeta(
                        name="PC",
                        index=0,
                        color={"r": 255, "g": 255, "b": 255},
                        emissionLambdaNm=595.0,
                    ),
                    microscope=microscope,
                ),
                FakeChannelRecord(
                    channel=FakeChannelMeta(
                        name="mCherry",
                        index=1,
                        color={"r": 255, "g": 0, "b": 0},
                        emissionLambdaNm=475.0,
                    ),
                    microscope=microscope,
                ),
            ]
        )
        self.experiment = [FakeLoop(count=2, type="TimeLoop")]
        attributes_type = namedtuple(
            "Attributes",
            ["bitsPerComponentInMemory", "channelCount"],
        )
        self.attributes = attributes_type(bitsPerComponentInMemory=16, channelCount=2)
        self.text_info = {"date": "07/03/2026 16:30:05", "optics": "Plan Apo 40x"}
        self.custom_data = {
            "AppInfo_V1_0": {"SWNameString": "NIS-Elements AR"},
            "GrabberCameraSettingsV1_0": {
                "GrabberCameraSettings": {"CameraUserName": "Fusion"}
            },
            "CustomDataV2_0": {
                "CustomTagDescription_v1.0": {"Tag0": {"ID": "Camera_ExposureTime1"}}
            },
            "StreamDataV1_0": {"ignored": True},
        }

    def __enter__(self) -> FakeND2File:
        return self

    def __exit__(self, *exc) -> None:
        return None

    def voxel_size(self):
        voxel_type = namedtuple("VoxelSize", ["x", "y", "z"])
        return voxel_type(0.108, 0.108, 1.0)

    def ome_metadata(self):
        planes = [
            SimpleNamespace(the_c=0, exposure_time=80.0),
            SimpleNamespace(the_c=1, exposure_time=200.0),
        ]
        return SimpleNamespace(
            images=[SimpleNamespace(pixels=SimpleNamespace(planes=planes))]
        )

    def unstructured_metadata(self) -> dict[str, dict]:
        return {
            "ImageMetadataLV": {"SLxExperiment": {}},
            "ImageEventsLV": {"RLxExperimentRecord": {}},
        }

    def frame_metadata(self, seq_index: int) -> FakeFrameMetadata:
        time_idx, p_idx = np.unravel_index(
            seq_index, (self.sizes["T"], self.sizes["P"])
        )
        rel_time_ms = 30000.0 * time_idx + 123.0 + p_idx
        abs_jdn = 2460000.0 + 0.01 * time_idx + 0.001 * p_idx
        stage_x = 100.0 + p_idx
        stage_y = 200.0 + time_idx
        stage_z = 300.0 + time_idx + p_idx
        channels = [
            FakeFrameChannel(
                channel=FakeChannelMeta(name="PC", index=0),
                loops={"TimeLoop": int(time_idx), "XYPosLoop": int(p_idx)},
                microscope={"objectiveName": "Plan Apo 40x"},
                volume={"axesCalibration": [0.108, 0.108, 1.0]},
                position=FakePosition(
                    stagePositionUm=FakeStagePosition(stage_x, stage_y, stage_z),
                    pfsOffset=None,
                    name=None,
                ),
                time=FakeTimeStamp(abs_jdn, rel_time_ms),
            ),
            FakeFrameChannel(
                channel=FakeChannelMeta(name="mCherry", index=1),
                loops={"TimeLoop": int(time_idx), "XYPosLoop": int(p_idx)},
                microscope={"objectiveName": "Plan Apo 40x"},
                volume={"axesCalibration": [0.108, 0.108, 1.0]},
                position=FakePosition(
                    stagePositionUm=FakeStagePosition(stage_x, stage_y, stage_z),
                    pfsOffset=None,
                    name=None,
                ),
                time=FakeTimeStamp(abs_jdn, rel_time_ms),
            ),
        ]
        return FakeFrameMetadata(contents={"frameCount": 4}, channels=channels)

    def read_frame(self, seq_index: int) -> np.ndarray:
        time_idx, p_idx = np.unravel_index(
            seq_index, (self.sizes["T"], self.sizes["P"])
        )
        base = (
            np.arange(12, dtype=np.uint16).reshape(3, 4)
            + (100 * time_idx)
            + (10 * p_idx)
        )
        return np.stack([base, base + 1000], axis=0)

    def events(self, orient: str = "records", null_value=np.nan) -> list[dict]:
        assert orient == "records"
        return [
            {"Time [s]": 0.1, "Events": "Command Executed"},
            # Like real ND2 files: PFS lives in the frame rows, not frame metadata.
            {
                "Index": 0,
                "T Index": 0,
                "P Index": 0,
                "PFS Offset": 17390,
                "PFS Status": 1,
            },
            {"Time [s]": 0.2, "Events": "Command Executed"},
            {
                "Index": 1,
                "T Index": 0,
                "P Index": 1,
                "PFS Offset": 17398,
                "PFS Status": 1,
            },
            {"Time [s]": 30.1, "Events": "Command Executed"},
            {
                "Index": 2,
                "T Index": 1,
                "P Index": 0,
                "PFS Offset": 17406,
                "PFS Status": 1,
            },
            {"Time [s]": 30.2, "Events": "Command Executed"},
            {
                "Index": 3,
                "T Index": 1,
                "P Index": 1,
                "PFS Offset": 17414,
                "PFS Status": 0,
            },
        ]


def fake_nd2(
    monkeypatch, path: Path, fov_names=("XYPos:0", "XYPos:1"), shape_yx=(3, 4)
) -> Path:
    """An ND2 file at ``path``, read as :class:`FakeND2File`: 2 frames 30 s apart, the
    FOVs named ``fov_names`` and 2 channels, PC and mCherry. ``nd2.imread`` gives zeros
    of ``shape_yx``; ``read_frame`` gives the 3 × 4 frames of :class:`FakeND2File`."""
    path.write_bytes(b"nd2")
    data = xr.DataArray(
        np.zeros((2, len(fov_names), 2, *tuple(shape_yx)), dtype=np.uint16),
        dims=("T", "P", "C", "Y", "X"),
        coords={
            "T": [0.0, 30000.0],
            "P": list(fov_names),
            "C": ["PC", "mCherry"],
            "Y": [round(0.108 * i, 3) for i in range(shape_yx[0])],
            "X": [round(0.108 * i, 3) for i in range(shape_yx[1])],
        },
        attrs={"metadata": {"placeholder": True}},
    )
    monkeypatch.setattr("pymmm.source.nd2.nd2.imread", lambda *args, **kwargs: data)
    monkeypatch.setattr("pymmm.source.metadata.nd2.ND2File", FakeND2File)
    return path


# ----------------------------------------------------------------------------
# Registered FOVs
# ----------------------------------------------------------------------------
FOV_H, FOV_W = 200, 160


def lane_image(lane_y: int, shift: int = 0) -> np.ndarray:
    """A bright feeding lane at ``lane_y`` (above the widget's default height, 5000),
    bright trench columns above it."""
    image = np.full((FOV_H, FOV_W), 1000.0, dtype=np.float32)
    image[lane_y - 20 : lane_y + 20] = 20000.0
    for x in range(20 + shift, FOV_W - 10, 40):
        image[lane_y - 90 : lane_y - 25, x : x + 8] = 4000.0
    return image


def fov_experiment(tmp_path, fovs):
    """An ND2Experiment of these FOVs' names, one channel, PC, 1 µm pixels."""
    return SimpleNamespace(
        fov_names=list(fovs),
        channel_names=["PC"],
        pixel_size_um=1.0,
        data=SimpleNamespace(sizes={"Y": FOV_H, "X": FOV_W}),
        path=tmp_path / "exp.nd2",
    )


def fov_registrator(images):
    """A Registrator whose registered means are ``images`` (by FOV)."""
    return SimpleNamespace(
        get_registered_mean_of_timestack=lambda fov, channel, **mean: images[fov],
        compute_registered_means=lambda fovs=None, channel=None, **mean: {
            f: images[f] for f in (fovs or list(images))
        },
    )


# ----------------------------------------------------------------------------
# Omnipose
# ----------------------------------------------------------------------------
def bottom_first(mask: np.ndarray) -> np.ndarray:
    """Label connected components, numbered from the bottom of the frame up."""
    labels, n = ndimage.label(mask)
    labels[labels > 0] = n + 1 - labels[labels > 0]
    return labels


class FakeModel:
    """Stands in for OmniModel: thresholds the normalised image."""

    THRESHOLD = 0.3

    def __init__(self, fail_stack: bool = False, fail_frames=()):
        self.fail_stack = fail_stack
        self.fail_frames = set(fail_frames)
        self.stack_calls = 0
        self.frame_calls = 0
        self.kwargs = []

    device = "cpu"

    def eval(self, x, **kwargs):
        assert kwargs["normalize"] is False
        self.kwargs.append(kwargs)
        if kwargs.get("compute_masks") is False:  # masks="affinity": the network's maps
            self.stack_calls += 1
            return SimpleNamespace(
                flows=[SimpleNamespace(dP=np.zeros((2, *f.shape)), dist=f) for f in x]
            )
        if x.shape[0] > 1:
            self.stack_calls += 1
            if self.fail_stack:
                raise IndexError("index 1 is out of bounds")
        else:
            t = self.frame_calls
            self.frame_calls += 1
            if t in self.fail_frames:
                raise IndexError("index 1 is out of bounds")
        return SimpleNamespace(
            masks=np.stack([bottom_first(f > self.THRESHOLD) for f in x])
        )


def fake_affinity_masks(
    dP, dist, device, *, mask_threshold, flow_threshold, angle, min_size
):
    """Stands in for ``omnipose.affinity_masks``: thresholds the distance map (here the
    normalised image) as :class:`FakeModel` does."""
    return bottom_first(dist > FakeModel.THRESHOLD)


def load_fake_model(source, path, gpu):
    """A Segmenter task's model loader in a worker process (``segmenter._loader``)."""
    return FakeModel()
