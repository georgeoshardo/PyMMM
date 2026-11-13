"""What each step's saved result was made from, and whether it still holds.

Extraction is a chain of steps (:data:`GRAPH`). Each saved result records the
fingerprints of the results it was made from (``made_from``):

    registration        the source: the ND2's size and the part of it used
    lane_detection      registration
    trench_detection    registration, lane_detection
    extraction          registration, trench_detection

A result's own fingerprint is a hash of the result, so redoing a step that gives the
same result (registering again with the same settings, say) leaves what follows
current. The code's version is recorded as provenance and plays no part here.

A result is *current* when every input it records still has the fingerprint it had,
and its inputs are current too. Otherwise it is *stale*, and so is everything made
from it. A result saved before these records were kept is *unrecorded*: nothing says
what it was made from. One that isn't there is *missing*.

Only numpy and the standard library: :mod:`pymmm.store` stays light.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

GRAPH: dict[str, tuple[str, ...]] = {
    "registration": ("source",),
    "lane_detection": ("registration",),
    "trench_detection": ("registration", "lane_detection"),
    "extraction": ("registration", "trench_detection"),
}
"""Each step, and the steps whose results it uses, in the order they run."""

CURRENT, STALE, MISSING, UNRECORDED = "current", "stale", "missing", "unrecorded"

LANE_COLUMNS = ("fov", "lane_index", "y_center", "orientation")
TRENCH_COLUMNS = (
    "trench_id",
    "fov",
    "lane_index",
    "x_left",
    "x_right",
    "y_top",
    "y_bottom",
    "orientation",
    "needs_flip",
)


def _plain(value: Any) -> Any:
    """``value`` as plain JSON, an array as its dtype, shape and the hash of its bytes."""
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def fingerprint(value: Any) -> str:
    """A short hash of ``value`` (plain JSON, numpy arrays and scalars)."""
    text = json.dumps(_plain(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def source_fingerprint(experiment: Any) -> str:
    """An ND2Experiment's source: the ND2's size and the part of it used
    (``source_subset_metadata``). Not its path or modification time, so a moved or
    copied ND2 is the same source.
    """
    return fingerprint(
        {
            "size_bytes": Path(experiment.path).stat().st_size,
            "subset": dict(experiment.source_subset_metadata),
        }
    )


def registration_fingerprint(tmats: Any, rotation: float) -> str:
    """The transforms (``(P, T, row, col)``, or ``(T, row, col)``) as float64, the FOVs
    they belong to, and the rotation composed with them: what extraction uses.
    """
    fovs: list[str] = []
    if "P" in tmats.dims:
        tmats = tmats.transpose("P", "T", "row", "col")
        fovs = [str(p) for p in tmats.coords["P"].values]
    values = np.asarray(tmats.values, dtype="<f8")
    return fingerprint({"tmats": values, "fovs": fovs, "rotation": float(rotation)})


def table_fingerprint(rows: Iterable[Mapping[str, Any]], columns: Iterable[str]) -> str:
    """A table's ``columns``, whatever their dtypes, in a fixed row order (sorted)."""
    columns = tuple(columns)
    plain = sorted(
        (tuple(_plain(row[c]) for c in columns) for row in rows),
        key=lambda row: json.dumps(row),
    )
    return fingerprint({"columns": list(columns), "rows": plain})


def status(
    fingerprints: Mapping[str, str | None],
    made_from: Mapping[str, Mapping[str, str | None] | None],
) -> dict[str, str]:
    """Each step's status from what is saved now: ``fingerprints`` of the results there
    (``None`` for one missing, ``"source"`` always given) and what each recorded it was
    made from (``None`` for a result saved before records were kept).
    """
    result: dict[str, str] = {}
    for step, inputs in GRAPH.items():
        recorded = made_from.get(step)
        if fingerprints.get(step) is None:
            result[step] = MISSING
        elif recorded is None:
            result[step] = UNRECORDED
        elif any(result.get(i) in (MISSING, STALE) for i in inputs) or any(
            recorded.get(i) != fingerprints.get(i) for i in inputs
        ):
            result[step] = STALE
        else:
            result[step] = CURRENT
    return result


def of(step: Any) -> str | None:
    """A step's fingerprint (its ``fingerprint()``), or None when there is no step or
    nothing computed yet.
    """
    try:
        return step.fingerprint()
    except (AttributeError, RuntimeError):
        return None


def usable(state: str) -> bool:
    """Whether a result with this status can be loaded and used: current, or saved
    before records were kept (nothing says it is stale).
    """
    return state in (CURRENT, UNRECORDED)
