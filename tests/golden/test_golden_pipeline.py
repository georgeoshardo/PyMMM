"""Golden tests of registration, lane and trench detection, their checkpoints, and
extraction, on real data (see conftest.py).

Every compute path is pinned to the same reference: one process, a process pool and a
Dask cluster, for registration, registered means and extraction; so are preview,
resume, time chunks and an output rotation.
"""

from __future__ import annotations

import copy
import shutil
from dataclasses import asdict

import numpy as np
import pytest
import zarr
from golden_io import canonical, check, digest, manifest
from pymmm import (
    CompanionStore,
    Extractor,
    LaneDetector,
    Registrator,
    ShadingCorrector,
    TrenchDetector,
)
from pymmm.steps.segmentation import background_noise, normalise, trench_bounds
from real_data import (
    EXTRACTION,
    LANES,
    REGISTRATION,
    TRENCHES,
    no_dask_client,
    open_nd2,
)

pytestmark = pytest.mark.golden


def registration_summary(reg: Registrator) -> dict:
    tmats = reg.tmats
    return {
        "dims": list(tmats.dims),
        "P": [str(p) for p in tmats["P"].values],
        "T": tmats["T"].values,
        "translation_xy": tmats.values[..., :2, 2],
        "matrices": digest(tmats.values),
    }


def lanes_summary(lanes: LaneDetector) -> dict:
    return {
        "lanes": {
            fov: [asdict(lane) for lane in fov_lanes]
            for fov, fov_lanes in lanes.lanes.items()
        },
        "params": lanes.detection_params,
    }


def trenches_summary(trenches: TrenchDetector) -> dict:
    return {
        "trenches": trenches.get_trench_table().to_dict(orient="records"),
        "params": trenches.detection_params,
    }


def extraction_detail(node: str) -> bool:
    """Keep the metadata of every node in full but the trench images after the first
    (hashed: they differ only in geometry and display windows)."""
    return not node.startswith("trench_") or node.split("/")[0] == "trench_0000"


def run_on(path: str, request) -> int:
    """Set up a compute path; the ``n_jobs`` to pass."""
    if path == "dask":
        request.getfixturevalue("dask_client")
        return -1
    assert no_dask_client()
    return 1 if path == "serial" else 2


# ----------------------------------------------------------------------------
# registration
# ----------------------------------------------------------------------------
def test_registration(pipeline):
    check("registration_first", registration_summary(pipeline.reg))


@pytest.mark.parametrize(
    "reference, overrides, path",
    [
        ("registration_first", {}, "pool"),
        ("registration_first", {}, "dask"),
        ("registration_first_opencv", {"backend": "opencv"}, "serial"),
        (
            "registration_first_opencv",
            {"backend": "opencv", "opencv_threads": 2},
            "serial",
        ),
        ("registration_first_opencv", {"backend": "opencv"}, "pool"),
        ("registration_first_opencv", {"backend": "opencv"}, "dask"),
        ("registration_frame3", {"mode": 3}, "serial"),
        ("registration_previous", {"mode": "previous"}, "serial"),
        ("registration_previous", {"mode": "previous"}, "pool"),
        ("registration_previous", {"mode": "previous"}, "dask"),
    ],
)
def test_registration_variants(pipeline, request, reference, overrides, path):
    reg = Registrator(pipeline.exp, pipeline.store, **{**REGISTRATION, **overrides})
    reg.compute_tmats(n_jobs=run_on(path, request))
    check(reference, registration_summary(reg))


@pytest.mark.parametrize("path", ["serial", "dask"])
def test_registered_images(pipeline, request, path):
    """The registered means detection runs on, a stabilised stack and a warped frame."""
    run_on(path, request)
    reg = Registrator.load(pipeline.exp, pipeline.store)  # nothing cached
    means = reg.compute_registered_means(
        channel="PC", mean_n_frames=10, mean_from="end"
    )
    stabilised = reg.get_stabilized_data(
        channel="mCherry", fov="XYPos:5", time_slice=slice(3, 5)
    )
    check(
        "registered_images",
        {
            "registered_means_PC": {fov: digest(image) for fov, image in means.items()},
            "registered_mean_mVenus": digest(
                reg.get_registered_mean_of_timestack(
                    1, "mVenus", mean_n_frames=10, mean_from="end"
                )
            ),
            "stabilized_mCherry_XYPos5_t3_5": {
                "dims": list(stabilised.dims),
                "data": digest(stabilised.values),
            },
            "warp_frame_fov1_t7_PC": digest(
                reg.warp_frame(fov=1, time=7, channel="PC")
            ),
        },
    )


# ----------------------------------------------------------------------------
# detection
# ----------------------------------------------------------------------------
def test_lanes(pipeline):
    check("lanes", lanes_summary(pipeline.lanes))


def test_trenches(pipeline):
    check("trenches", trenches_summary(pipeline.trenches))


def test_trenches_peaks_method(pipeline):
    trenches = TrenchDetector(
        pipeline.exp,
        pipeline.reg,
        pipeline.lanes,
        pipeline.store,
        detection_channel="PC",
    )
    trenches.detect_trenches(**{**TRENCHES, "method": "peaks"}, save_diagnostics=False)
    check("trenches_peaks", trenches_summary(trenches))


def test_checkpoints(pipeline):
    """The companion store, and the steps loaded back from it."""
    check("companion_store", manifest(pipeline.store.path))

    store = CompanionStore(pipeline.store.path, mode="r")
    reg = Registrator.load(pipeline.exp, store)
    lanes = LaneDetector.load(pipeline.exp, reg, store)
    trenches = TrenchDetector.load(pipeline.exp, reg, lanes, store)
    shading = ShadingCorrector.load(pipeline.exp, reg, trenches, store)
    assert shading.fingerprint() == pipeline.shading.fingerprint()
    assert shading.evidence().equals(pipeline.shading.evidence())
    check("registration_first", registration_summary(reg))
    check("lanes", lanes_summary(lanes))
    check("trenches", trenches_summary(trenches))
    for key in (
        "channel",
        "mode",
        "rotation",
        "roi",
        "downsample",
        "batch_size",
        "fov_window",
        "backend",
        "opencv_threads",
    ):
        assert canonical(getattr(reg, key)) == canonical(getattr(pipeline.reg, key)), (
            key
        )


# ----------------------------------------------------------------------------
# extraction
# ----------------------------------------------------------------------------
def test_extraction(pipeline):
    check("extraction", manifest(pipeline.group, extraction_detail))


@pytest.mark.parametrize("path, batch_size", [("serial", 10), ("dask", 4)])
def test_extraction_paths(pipeline, request, tmp_path, path, batch_size):
    run_on(path, request)
    output = tmp_path / "experiment.zarr"
    Extractor(
        pipeline.exp,
        pipeline.reg,
        pipeline.trenches,
        pipeline.shading,
        output_path=output,
    ).extract(
        **{**EXTRACTION, "batch_size": batch_size}, resume=False, show_progress=False
    )
    check("extraction", manifest(output, extraction_detail))


def test_extraction_resumes_an_interrupted_run(pipeline, tmp_path):
    output = tmp_path / "experiment.zarr"
    shutil.copytree(pipeline.group, output)
    extractor = Extractor(
        pipeline.exp,
        pipeline.reg,
        pipeline.trenches,
        pipeline.shading,
        output_path=output,
    )

    extractor.extract(**EXTRACTION, resume=True, show_progress=False)  # complete: no-op
    check("extraction", manifest(output, extraction_detail))

    # interrupted after the first FOV: the second FOV's trenches are empty
    root = zarr.open_group(str(output), mode="r+")
    attrs = dict(root.attrs)["pymmm"]
    root.attrs["pymmm"] = {
        **attrs,
        "completed_fovs": attrs["completed_fovs"][:1],
        "extraction_complete": False,
    }
    table = pipeline.trenches.get_trench_table()
    second = table.loc[table["fov"] == attrs["completed_fovs"][1], "trench_id"]
    for trench in second:
        root[f"trench_{trench:04d}/0"][...] = 0
    extractor.extract(**EXTRACTION, resume=True, show_progress=False)
    check("extraction", manifest(output, extraction_detail))

    with pytest.raises(RuntimeError, match="not compatible"):
        extractor.extract(
            **{**EXTRACTION, "clevel": 5}, resume=True, show_progress=False
        )
    # nor with the shading taken out another way: it would mix the two in one group
    other = copy.copy(pipeline.shading)
    other.mode = "subtract"
    with pytest.raises(RuntimeError, match="shading"):
        Extractor(
            pipeline.exp, pipeline.reg, pipeline.trenches, other, output_path=output
        ).extract(**EXTRACTION, resume=True, show_progress=False)


def test_extraction_time_chunks(pipeline, tmp_path):
    output = tmp_path / "experiment.zarr"
    Extractor(
        pipeline.exp,
        pipeline.reg,
        pipeline.trenches,
        pipeline.shading,
        output_path=output,
    ).extract(**{**EXTRACTION, "time_chunk": 4}, resume=False, show_progress=False)
    check(
        "extraction_time_chunk4",
        manifest(output, lambda node: node == "trench_0000/0", list_files=False),
    )


def test_preview_matches_extraction(pipeline):
    root = zarr.open_group(str(pipeline.group), mode="r")
    preview = pipeline.extractor.preview()
    assert list(preview.dims) == ["Trench", "T", "C", "Y", "X"]
    assert preview["C"].values.tolist() == ["PC", "mCherry", "mVenus", "CFP"]
    trench_ids = preview["Trench"].values.tolist()
    assert trench_ids == root["trenches/trench"][:].tolist()
    pixels = preview.values
    for k, trench in enumerate(trench_ids):
        np.testing.assert_array_equal(pixels[k], root[f"trench_{trench:04d}/0"][...])
    mcherry = pipeline.extractor.preview(channel="mCherry")
    assert list(mcherry.dims) == ["Trench", "T", "Y", "X"]
    np.testing.assert_array_equal(mcherry.values, pixels[:, :, 1])


def test_extract_single_trench(pipeline):
    """One trench and one channel cut on their own: the same pixels as the extraction,
    shading and all (a frame's level comes from all its FOV's spacers)."""
    single = pipeline.extractor.extract_single_trench(7, channel="PC")
    root = zarr.open_group(str(pipeline.group), mode="r")
    np.testing.assert_array_equal(single, root["trench_0007/0"][:, 0])
    check(
        "extract_single_trench",
        {
            "trench_7_PC": digest(single),
            "trench_20_mCherry": digest(
                pipeline.extractor.extract_single_trench(20, channel="mCherry")
            ),
        },
    )


def test_shading_changes_only_its_channel(pipeline):
    plain = Extractor(pipeline.exp, pipeline.reg, pipeline.trenches)
    for channel in pipeline.exp.channel_names:
        shaded = np.asarray(pipeline.extractor.preview_array(channel)[...])
        unshaded = np.asarray(plain.preview_array(channel)[...])
        if channel == pipeline.shading.channel:
            assert not np.array_equal(shaded, unshaded)
        else:
            np.testing.assert_array_equal(shaded, unshaded)


def test_segmenter_normalisation(pipeline):
    """The intensity scaling of a real trench, which the PC candidates also use."""
    root = zarr.open_group(str(pipeline.group), mode="r")
    stack = root["trench_0007/0"][:, 0]
    lower, upper = trench_bounds(stack, floor=3000.0)
    check(
        "segmenter_normalisation",
        {
            "background_noise": background_noise(stack),
            "bounds": [lower, upper],
            "normalised_gain_1.5": digest(normalise(stack, lower, upper, gain=1.5)),
        },
    )


# ----------------------------------------------------------------------------
# an output rotation
# ----------------------------------------------------------------------------
def test_rotated_pipeline(scratch_folder):
    """Rotation composes into every warp: detection images and extraction."""
    assert no_dask_client()
    folder = scratch_folder("rotated")
    exp = open_nd2(folder)
    store = CompanionStore.for_experiment(exp)
    reg = Registrator(exp, store, **{**REGISTRATION, "rotation": 1.5})
    reg.compute_tmats(n_jobs=1)
    lanes = LaneDetector(exp, reg, store, detection_channel="PC")
    lanes.detect_lanes(**LANES, save_diagnostics=False)
    trenches = TrenchDetector(exp, reg, lanes, store, detection_channel="PC")
    trenches.detect_trenches(**TRENCHES, save_diagnostics=False)
    output = folder / "experiment.zarr"
    Extractor(exp, reg, trenches, output_path=output).extract(
        **{**EXTRACTION, "batch_size": 10}, resume=False, show_progress=False
    )
    check(
        "rotated",
        {
            "registration": registration_summary(reg),
            "lanes": lanes_summary(lanes),
            "trenches": trenches_summary(trenches),
            "extraction": manifest(
                output,
                lambda node: node in (".", "trench_0000", "trench_0000/0"),
                list_files=False,
            ),
        },
    )
