"""Digests of arrays and zarr stores, and the reference files they are checked against.

A golden test pins what PyMMM makes from real data today, so that a refactor can show
it changes nothing. A reference is a JSON file in ``reference/``: hashes of pixels,
small arrays in full, and the scrubbed metadata of every zarr node (never the data).

After an intended change of output, regenerate the references and review their diff
before committing it::

    PYMMM_GOLDEN_UPDATE=1 python -m pytest tests/golden
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pytest
import zarr

REFERENCE_DIR = Path(__file__).parent / "reference"
UPDATE = os.environ.get("PYMMM_GOLDEN_UPDATE") == "1"
_updated: set[str] = set()  # references written by this run

VOLATILE_KEYS = frozenset({
    "provenance",          # code and library versions, and when
    "created",
    "extraction_created",  # the extraction's "created", copied into a segmentation's record
    "made_on",             # the machine and GPU that segmented a trench
    "source_nd2",          # the path of the scratch symlink
    "source_mtime_ns",
    "geff_version",
})
"""Keys whose values differ between runs; their presence is still checked."""
VOLATILE = "<volatile>"


def scrub(value: Any) -> Any:
    """``value`` with the values of :data:`VOLATILE_KEYS` replaced, at any depth."""
    if isinstance(value, dict):
        return {k: VOLATILE if k in VOLATILE_KEYS else scrub(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub(v) for v in value]
    return value


def _native(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def canonical(value: Any) -> Any:
    """``value`` as plain JSON types (tuples become lists, numpy scalars Python ones)."""
    return json.loads(json.dumps(value, default=_native))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(array: Any) -> dict:
    """Shape, dtype, a hash of the values and, for numbers, their sum, min and max.

    The sum tells a rounding flip of a pixel (off by one) from a real change.
    """
    a = np.asarray(array)
    if a.dtype.kind in "OUST":
        text = "\x00".join(str(v) for v in a.ravel().tolist())
        return {"shape": list(a.shape), "dtype": "str",
                "sha256": sha256(f"{a.shape}|".encode() + text.encode())}
    a = np.ascontiguousarray(a)
    out = {"shape": list(a.shape), "dtype": a.dtype.str,
           "sha256": sha256(f"{a.dtype.str}{a.shape}|".encode() + a.tobytes())}
    if a.dtype.kind in "biuf" and a.size:
        values = a.astype(np.float64)
        out.update(sum=float(np.nansum(values)), min=float(np.nanmin(values)),
                   max=float(np.nanmax(values)))
    return out


def manifest(
    path: Path | str,
    detail: Callable[[str], bool] = lambda node: True,
    list_files: bool = True,
) -> dict:
    """Every file of a zarr store, and every node's metadata and data digest.

    ``detail(node)`` chooses the nodes whose (scrubbed) ``zarr.json`` is kept in full;
    the others keep its hash. Array data is always digested. ``list_files=False``
    keeps a hash of the file list instead of the list.
    """
    root = Path(path)
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    if not list_files:
        files = sha256("\n".join(files).encode())
    nodes = {}
    for meta_path in sorted(root.rglob("zarr.json")):
        key = meta_path.parent.relative_to(root).as_posix()
        meta = scrub(json.loads(meta_path.read_text()))
        entry = (meta if detail(key) else
                 {"zarr_json_sha256": sha256(json.dumps(meta, sort_keys=True).encode())})
        if meta.get("node_type") == "array":
            entry["data"] = digest(zarr.open_array(str(meta_path.parent), mode="r")[...])
        nodes[key] = entry
    return {"files": files, "nodes": nodes}


def differences(expected: Any, actual: Any, where: str = "") -> Iterator[str]:
    """Where ``actual`` differs from ``expected`` (both plain JSON values)."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            here = f"{where}/{key}"
            if key not in actual:
                yield f"{here}: missing"
            elif key not in expected:
                yield f"{here}: unexpected {_short(actual[key])}"
            else:
                yield from differences(expected[key], actual[key], here)
    elif (isinstance(expected, list) and isinstance(actual, list)
          and len(expected) == len(actual)):
        for i, (e, a) in enumerate(zip(expected, actual)):
            yield from differences(e, a, f"{where}[{i}]")
    elif json.dumps(expected) != json.dumps(actual):  # NaN equals NaN here
        yield f"{where or '/'}: expected {_short(expected)}, got {_short(actual)}"


def _short(value: Any, limit: int = 160) -> str:
    text = json.dumps(value)
    return text if len(text) <= limit else text[:limit] + "…"


def dumps(value: Any, depth: int = 0) -> str:
    """JSON with one key per line and lists of numbers or strings on one line."""
    pad, inner = " " * depth, " " * (depth + 1)
    if isinstance(value, dict) and value:
        items = [f"{inner}{json.dumps(k)}: {dumps(v, depth + 1)}"
                 for k, v in sorted(value.items())]
        return "{\n" + ",\n".join(items) + f"\n{pad}}}"
    if isinstance(value, list) and any(isinstance(v, (dict, list)) for v in value):
        return "[\n" + ",\n".join(inner + dumps(v, depth + 1) for v in value) + f"\n{pad}]"
    return json.dumps(value)


def check(name: str, actual: Any) -> None:
    """Assert ``actual`` equals ``reference/<name>.json``.

    When updating, the first check of a reference in a run writes it and later ones
    compare with it, so compute paths pinned to one reference must still agree.
    """
    actual = canonical(actual)
    path = REFERENCE_DIR / f"{name}.json"
    if UPDATE and name not in _updated:
        REFERENCE_DIR.mkdir(exist_ok=True)
        path.write_text(dumps(actual) + "\n")
        _updated.add(name)
        return
    if not path.exists():
        pytest.fail(f"no reference {path.name}: make it with PYMMM_GOLDEN_UPDATE=1")
    found = list(differences(json.loads(path.read_text()), actual))
    if found:
        shown = "\n  ".join(found[:40])
        more = f"\n  … and {len(found) - 40} more" if len(found) > 40 else ""
        pytest.fail(f"{name}: {len(found)} differences from {path.name}\n  {shown}{more}",
                    pytrace=False)
