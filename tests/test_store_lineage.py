"""What each step was made from (pymmm.store.lineage): the status rules, and fingerprints
that depend on the result only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from pymmm.store import lineage

FINGERPRINTS = {
    "source": "s",
    "registration": "r",
    "lane_detection": "l",
    "trench_detection": "t",
    "shading": "h",
    "extraction": "e",
}
MADE_FROM = {
    "registration": {"source": "s"},
    "lane_detection": {"registration": "r"},
    "trench_detection": {"registration": "r", "lane_detection": "l"},
    "shading": {"registration": "r", "trench_detection": "t"},
    "extraction": {"registration": "r", "trench_detection": "t", "shading": "h"},
}


def test_every_step_made_from_what_is_there_is_current():
    assert set(lineage.status(FINGERPRINTS, MADE_FROM).values()) == {lineage.CURRENT}


def test_a_changed_result_makes_what_was_made_from_it_stale():
    status = lineage.status({**FINGERPRINTS, "lane_detection": "l2"}, MADE_FROM)
    assert status == {
        "registration": "current",
        "lane_detection": "current",
        "trench_detection": "stale",
        "shading": "stale",
        "extraction": "stale",
    }


def test_shading_measured_again_makes_only_the_extraction_stale():
    status = lineage.status({**FINGERPRINTS, "shading": "h2"}, MADE_FROM)
    assert status["trench_detection"] == "current"
    assert status["shading"] == "current"
    assert status["extraction"] == "stale"


def test_an_extraction_without_shading_is_not_current():
    """Cut without measuring the shading (Extractor(..., shading=None)): it records
    none, and none is saved, so it is stale, as one whose input is gone."""
    status = lineage.status(
        {**FINGERPRINTS, "shading": None},
        {**MADE_FROM, "extraction": {**MADE_FROM["extraction"], "shading": None}},
    )
    assert status["shading"] == "missing"
    assert status["extraction"] == "stale"


def test_stale_passes_down_even_where_the_record_matches():
    """Another source (other frames, say) leaves the lanes' record matching the
    registration, but the registration is stale, so they are too."""
    status = lineage.status({**FINGERPRINTS, "source": "s2"}, MADE_FROM)
    assert set(status.values()) == {lineage.STALE}


def test_missing_and_unrecorded_results():
    status = lineage.status(
        {**FINGERPRINTS, "registration": None},
        {**MADE_FROM, "lane_detection": None},
    )
    assert status["registration"] == "missing"
    assert status["lane_detection"] == "unrecorded"  # nothing says what it came from
    assert status["trench_detection"] == "stale"  # its registration is gone
    old = lineage.status(FINGERPRINTS, dict.fromkeys(MADE_FROM))
    assert set(old.values()) == {lineage.UNRECORDED}
    assert all(lineage.usable(s) for s in old.values())
    assert not lineage.usable(lineage.STALE) and not lineage.usable(lineage.MISSING)


def test_table_fingerprints_ignore_dtypes_and_row_order():
    table = pd.DataFrame(
        {
            "fov": ["B", "A"],
            "lane_index": [0, 0],
            "y_center": [900, 880],
            "orientation": [1, -1],
            "extra": [1.5, 2.5],  # not a lane column
        }
    )
    same = table.iloc[::-1].astype(
        {"lane_index": "int8", "y_center": "int32", "orientation": "int8"}
    )
    first = lineage.table_fingerprint(table.to_dict("records"), lineage.LANE_COLUMNS)
    assert first == lineage.table_fingerprint(
        same.drop(columns="extra").to_dict("records"), lineage.LANE_COLUMNS
    )
    moved = table.assign(y_center=[901, 880])
    assert first != lineage.table_fingerprint(
        moved.to_dict("records"), lineage.LANE_COLUMNS
    )


def test_registration_fingerprint_is_of_the_transforms_fovs_and_rotation():
    values = np.tile(np.eye(3), (2, 4, 1, 1))
    values[1, 2, 0, 2] = 1.5
    tmats = xr.DataArray(
        values,
        dims=("P", "T", "row", "col"),
        coords={"P": ["A01", "B32"], "T": np.arange(4)},
    )
    first = lineage.registration_fingerprint(tmats, -1.7)
    # the same transforms in another dimension order and dtype, other T labels
    assert first == lineage.registration_fingerprint(
        tmats.transpose("T", "P", "row", "col").astype("float32").astype("float64"),
        -1.7,
    )
    assert first == lineage.registration_fingerprint(
        tmats.assign_coords(T=np.arange(4) * 60.0), -1.7
    )
    assert first != lineage.registration_fingerprint(tmats, 0.0)
    shifted = tmats.copy(data=values + np.eye(3) * 0)
    shifted[0, 0, 1, 2] = 0.25
    assert first != lineage.registration_fingerprint(shifted, -1.7)


def test_of_gives_none_without_a_computed_step():
    class NotComputed:
        def fingerprint(self):
            raise RuntimeError("Tmats not computed yet.")

    assert lineage.of(None) is None
    assert lineage.of(NotComputed()) is None
