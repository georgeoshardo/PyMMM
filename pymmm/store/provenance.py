"""What PyMMM writes into its stores' attributes: the schema version, provenance
records, and values as plain JSON."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from functools import lru_cache
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 5
"""Layout version of the companion and trench stores. A record of what wrote a
store; nothing checks it."""

_LIBRARIES = {
    "nd2": "nd2",
    "zarr": "zarr",
    "xarray": "xarray",
    "numpy": "numpy",
    "dask": "dask",
    "pystackreg": "pystackreg",
    "opencv": "opencv-python-headless",
    "scikit-image": "scikit-image",
    "omnipose": "omnipose",
    "torch": "torch",
}


def plain_json(value: Any) -> Any:
    """``value`` as zarr attributes store it: plain JSON types, tuples as lists, and
    anything else JSON cannot encode (numpy integers, paths) as its ``str``."""
    return json.loads(json.dumps(value, default=str))


def _distribution_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


@lru_cache(maxsize=1)
def _code_version() -> dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    dirty = None
    if commit:
        status = _git("status", "--porcelain", "--untracked-files=no")
        dirty = bool(status) if status is not None else None
    return {
        "pymmm_version": _distribution_version("pymmm"),
        "pymmm_commit": commit,
        "pymmm_dirty": dirty,
        "libraries": {
            key: _distribution_version(dist) for key, dist in _LIBRARIES.items()
        },
    }


def provenance() -> dict[str, Any]:
    """Describe the code that is writing a store, and when.

    ``pymmm_commit`` and ``pymmm_dirty`` come from git when PyMMM runs from a
    checkout (``pymmm_dirty`` means tracked files had uncommitted changes);
    both are ``None`` for an installed release.
    """
    return {
        **_code_version(),
        "created": datetime.now().astimezone().isoformat(),
    }
