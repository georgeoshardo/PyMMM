"""``pymmm.steps.shading``: the background from the PDMS between trenches, the two ways
it comes out, the frame's own level, and the saved checkpoint."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from pymmm.steps import shading
from pymmm.steps.extraction import _crop_frames
from pymmm.steps.shading import FovShading, ShadingCorrector
from pymmm.steps.trenches import TrenchDefinition
from pymmm.store.companion import CompanionStore

from fake_experiment import FakeND2File, fake_nd2

H, W = 400, 640
PDMS, CELLS = 4000.0, 3000.0  # what PDMS and cells look like without shading
LANES = ((40, 180), (220, 360))  # two rows of trenches, both opening downward


def trenches(
    fov: str = "A", width: int = 30, pitch: int = 60
) -> list[TrenchDefinition]:
    out = []
    for lane, (top, bottom) in enumerate(LANES):
        for x in range(10, W - width, pitch):
            out.append(
                TrenchDefinition(
                    len(out), fov, lane, x, x + width, top, bottom, 1, False
                )
            )
    return out


def sample_image(fov_trenches, shade: str, strength: float = 1.0) -> np.ndarray:
    """PDMS everywhere, cells filling the trenches; then light added across the field
    (``"added"``) or the field lit unevenly (``"gain"``)."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    image = np.full((H, W), PDMS, np.float32)
    for t in fov_trenches:
        image[t.y_top : t.y_bottom, t.x_left : t.x_right] = CELLS
    ramp = strength * (x / W + 0.5 * y / H)  # 0 to 1.5
    if shade == "added":
        return image + 1000.0 * ramp
    return image * (1.0 + 0.25 * ramp)


def test_lane_spacers_keep_rows_apart_even_when_they_open_the_same_way():
    lanes = shading.lane_spacers(trenches())
    assert [(top, bottom) for top, bottom, _ in lanes] == list(LANES)
    columns = lanes[0][2]
    # the middle half of each 30 px gap: 40 + 7 to 70 - 7
    assert columns[0] == 47 and 62 in columns and 40 not in columns
    assert np.array_equal(columns, lanes[1][2])


def test_a_gap_left_by_a_missed_trench_is_not_taken_for_pdms():
    row = [t for t in trenches() if t.lane_index == 0]
    missing = row[3]
    (_, _, columns), *_ = shading.lane_spacers([t for t in row if t is not missing])
    assert not np.isin(np.arange(missing.x_left, missing.x_right), columns).any()
    assert len(columns) == len(shading.lane_spacers(row)[0][2]) - 2 * 16  # 47 to 62


def test_the_surface_follows_the_pdms_and_ignores_the_cells():
    fov_trenches = trenches()
    image = sample_image(fov_trenches, "added")
    surface = shading.fit_surface(image, shading.lane_spacers(fov_trenches))
    background = shading.sample(surface, np.arange(H), np.arange(W), (H, W))
    y, x = np.mgrid[0:H, 0:W]
    truth = PDMS + 1000.0 * (x / W + 0.5 * y / H)
    inside = slice(LANES[0][0], LANES[1][1])
    assert np.abs(background[inside] - truth[inside]).max() < 30  # of a 1500 range
    assert np.isfinite(surface).all()  # carried above and below the rows


@pytest.mark.parametrize("shade, right", [("added", "subtract"), ("gain", "divide")])
def test_each_mode_takes_out_its_kind_of_shading(shade, right):
    fov_trenches = trenches()
    image = sample_image(fov_trenches, shade)
    surface = shading.fit_surface(image, shading.lane_spacers(fov_trenches))
    background = shading.sample(surface, np.arange(H), np.arange(W), (H, W))
    level = 4500.0
    cells = np.zeros((H, W), bool)
    for t in fov_trenches:
        cells[t.y_top + 20 : t.y_bottom - 20, t.x_left + 5 : t.x_right - 5] = True
    spacers = np.zeros((H, W), bool)
    for top, bottom, columns in shading.lane_spacers(fov_trenches):
        spacers[top:bottom, columns] = True
    spread = {}
    for mode in ("subtract", "divide"):
        out = shading.correct(image, background, level, 0.0, mode)
        # the PDMS comes to the level either way
        assert np.abs(np.median(out[spacers]) - level) < 10
        spread[mode] = np.ptp(out[cells])
    wrong = ({"subtract", "divide"} - {right}).pop()
    assert spread[right] < 60  # flat, to the fit's error
    assert spread[wrong] > 3 * spread[right]


def test_a_fov_without_room_between_trenches_is_not_fitted():
    packed = trenches(width=58)  # 2 px gaps
    image = sample_image(packed, "added")
    assert shading.fit_surface(image, shading.lane_spacers(packed)) is None


def test_a_frame_is_corrected_to_its_own_level():
    fov_trenches = trenches()
    image = sample_image(fov_trenches, "added")
    surface = shading.fit_surface(image, shading.lane_spacers(fov_trenches))
    fov = FovShading(
        surface=surface,
        shape=(H, W),
        level=4500.0,
        offset=100.0,
        mode="subtract",
        channel=0,
        spacers=tuple(
            (top, bottom, tuple(int(c) for c in cols))
            for top, bottom, cols in shading.lane_spacers(fov_trenches)
        ),
    )
    top, bottom = LANES[1]
    band = image[top:bottom].astype(np.float64)
    brighter = band + 150.0  # the lamp, a moment later
    brighter[:, :3] = 0  # the warp's fill
    first = fov.correct_band(band, top)
    later = fov.correct_band(brighter, top)
    np.testing.assert_allclose(later[:, 3:], first[:, 3:], atol=1.0)
    assert (later[:, :3] == 0).all()
    # rows that are no lane's band keep the saved background, without a frame level
    rows = np.arange(top + 10, top + 50)
    saved = shading.sample(surface, rows, np.arange(W), (H, W))
    np.testing.assert_allclose(
        fov.correct_band(band[10:50], top + 10),
        shading.correct(band[10:50], saved, 4500.0, 100.0, "subtract"),
        atol=0.01,
    )


def test_extraction_corrects_only_its_channel_and_clips_at_zero(monkeypatch, tmp_path):
    fake_nd2(monkeypatch, tmp_path / "source.nd2")
    fov = FovShading(
        surface=np.full((2, 2), 5000.0, np.float32),  # grid of a 3 x 4 frame
        shape=(3, 4),
        level=4000.0,
        offset=100.0,
        mode="subtract",
        channel=0,
        spacers=(),
    )
    args = (
        str(tmp_path / "source.nd2"),
        1,
        [0],
        None,
        None,
        np.eye(3)[np.newaxis],
        ((0, 3, ((0, 0, 4, False),)),),
        ((0, 3, 0, 4, False),),
        4,
        np.dtype("uint16"),
    )
    plain = _crop_frames(*args)
    out = _crop_frames(*args, None, fov)
    raw = FakeND2File(tmp_path).read_frame(1)  # T 0, P 1
    assert (out[0, 0, 0] == 0).all()  # PC 10 to 21 counts, less 1000: clipped
    np.testing.assert_array_equal(out[0, 0, 1], raw[1])  # mCherry untouched
    np.testing.assert_array_equal(plain[0, 0, 0], raw[0])
    only_pc = _crop_frames(*args, (0,), fov)
    np.testing.assert_array_equal(only_pc[0, 0, 0], out[0, 0, 0])


def _corrector(tmp_path, mode="divide", offset=100.0, rotation=0.0):
    images, table = {}, {}
    for fov, shade in (("A", "added"), ("B", "gain")):
        table[fov] = trenches(fov)
        images[fov] = sample_image(table[fov], shade)
    table["C"] = trenches("C", width=58)  # no spacers
    images["C"] = sample_image(table["C"], "added")
    experiment = SimpleNamespace(
        fov_names=["A", "B", "C"],
        channel_names=["PC", "YFP"],
        data=SimpleNamespace(sizes={"Y": H, "X": W}),
        source_subset_metadata={},
    )
    registrator = SimpleNamespace(
        compute_registered_means=lambda fovs=None, channel=None, **mean: {
            f: images[f] for f in (fovs or images)
        },
        fingerprint=lambda: "r",
        rotation=rotation,
    )
    detector = SimpleNamespace(
        channel="PC",
        trenches=table,
        detection_image={"mean_n_frames": 1, "mean_from": "end"},
        fingerprint=lambda: "t",
    )
    store = CompanionStore(tmp_path / "exp.pymmm.zarr")
    corrector = ShadingCorrector(
        experiment, registrator, detector, store, mode=mode, offset=offset
    )
    return corrector, (experiment, registrator, detector, store)


def test_saved_shading_loads_with_its_fingerprint(tmp_path):
    corrector, (experiment, registrator, detector, store) = _corrector(tmp_path)
    corrector.fit()
    assert corrector.skipped == ["C"]
    assert corrector.fov_shading("C") is None
    assert corrector.fov_shading("A").channel == 0
    corrector.save()
    assert store.made_from("shading") == {"registration": "r", "trench_detection": "t"}

    loaded = ShadingCorrector.load(experiment, registrator, detector, store)
    assert loaded.fingerprint() == corrector.fingerprint()
    assert store.fingerprints()["shading"] == corrector.fingerprint()
    assert (loaded.mode, loaded.offset, loaded.level) == (
        "divide",
        100.0,
        corrector.level,
    )
    assert loaded.skipped == ["C"]
    np.testing.assert_array_equal(loaded.evidence(), corrector.evidence())
    # an offset stepped in a spin box is the same setting
    assert loaded.matches("divide", 100.00000000001)
    stepped, _ = _corrector(tmp_path / "again", offset=100.00000000001)
    stepped.fit()
    assert stepped.fingerprint() == corrector.fingerprint()


def test_the_mode_is_part_of_the_fingerprint_and_off_corrects_nothing(tmp_path):
    corrector, _ = _corrector(tmp_path, mode="subtract")
    corrector.fit()
    first = corrector.fingerprint()
    corrector.mode = "off"
    assert corrector.fingerprint() != first
    assert corrector.fov_shading("A") is None
    with pytest.raises(ValueError, match="mode"):
        _corrector(tmp_path, mode="flat")


def test_the_evidence_tells_added_light_from_a_gain(tmp_path):
    """Cells brighten one for one with the background where light is added (FOV A),
    in proportion where it is a gain (FOV B)."""
    corrector, _ = _corrector(tmp_path, offset=0.0)
    corrector.fit()
    table = corrector.evidence()
    for fov, expected in (("A", 1.0), ("B", CELLS / PDMS)):
        rows = table[table["fov"] == fov]
        slope = np.polyfit(rows["background"], rows["cells"], 1)[0]
        assert slope == pytest.approx(expected, abs=0.05)


def test_the_raw_background_is_the_registered_one_turned_back(tmp_path):
    """For the stitched view, which shows raw frames: raw pixel q is registered pixel
    M^-1 q, M the rotation extraction composes into its warp."""
    from pymmm.steps.registration import _rotation_matrix

    flat, _ = _corrector(tmp_path)
    flat.fit()
    np.testing.assert_allclose(flat.raw_background("A"), flat.background("A"), atol=0.5)
    turned, _ = _corrector(tmp_path / "turned", rotation=3.0)
    turned.fit()
    gy, gx = shading.grid_axis(H), shading.grid_axis(W)
    yy, xx = np.meshgrid(gy[5:-5], gx[5:-5], indexing="ij")  # inside the frame
    x, y, _ = np.linalg.inv(_rotation_matrix(3.0, (H, W))) @ np.stack(
        [xx.ravel(), yy.ravel(), np.ones(xx.size)]
    )
    expected = shading.sample_points(turned._surfaces["A"], y, x, (H, W))
    raw = turned.raw_background("A")
    np.testing.assert_allclose(raw[yy.ravel(), xx.ravel()], expected, atol=0.5)
    small = turned.raw_background("A", 8)  # each 8 x 8 block's centre
    assert small.shape == (H // 8, W // 8)
    np.testing.assert_allclose(small[10, 20], raw[83:85, 163:165].mean(), atol=2)
    assert turned.raw_background("C") is None  # not measured
    frame = np.full((H, W), 5000.0)
    assert np.array_equal(turned.correct_raw(frame, "C"), frame)
