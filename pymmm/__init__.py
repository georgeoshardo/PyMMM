"""PyMMM – Python Mother Machine Manager.

ND2 to one OME-Zarr image per trench, for mother-machine time-lapses::

    pymmm.source    the ND2 and its metadata
    pymmm.store     the stores PyMMM writes: the experiment group, its OME and GEFF
                    metadata, the companion checkpoints, provenance
    pymmm.steps     registration, lanes, trenches, extraction, segmentation

The step classes and the companion store are importable from ``pymmm`` itself. They
are imported when first used, so ``pymmm.store`` alone needs only numpy and zarr.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

_PUBLIC = {
    "ND2Experiment": "pymmm.source.nd2",
    "Registrator": "pymmm.steps.registration",
    "LaneDetector": "pymmm.steps.lanes",
    "TrenchDetector": "pymmm.steps.trenches",
    "Extractor": "pymmm.steps.extraction",
    "Segmenter": "pymmm.steps.segmentation",
    "CompanionStore": "pymmm.store.companion",
}

__all__ = list(_PUBLIC)

if TYPE_CHECKING:
    from pymmm.source.nd2 import ND2Experiment
    from pymmm.steps.extraction import Extractor
    from pymmm.steps.lanes import LaneDetector
    from pymmm.steps.registration import Registrator
    from pymmm.steps.segmentation import Segmenter
    from pymmm.steps.trenches import TrenchDetector
    from pymmm.store.companion import CompanionStore


def __getattr__(name: str) -> Any:
    if name in _PUBLIC:
        value = getattr(import_module(_PUBLIC[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'pymmm' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *_PUBLIC})
