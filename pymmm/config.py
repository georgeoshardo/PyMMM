"""Experiment configuration – one TOML file per experiment.

Every notebook of the pipeline (extract, segment, curate) reads the same file, so
a new dataset means writing one file::

    name = "20261005_pbs_psc_yfp_ty013"
    nd2 = "raw_data/20261005_pbs_psc_yfp_ty013.nd2"
    store = "20261005_pbs_psc_yfp_ty013.zarr"

    [segmentation]
    channel = "mCherry"
    model = "bact_fluor_omni"
    name = "mcherry"         # labels/<name>.<hypothesis> in each trench image
    hypotheses = ["S1", "S2", "S3", "S4"]
    batch_size = 64
    workers = 8

    [curation]
    kymograph_channel = "PC"

``store`` is the experiment group the Extractor writes: one OME-Zarr image per
trench, with each trench's segmentations and tracks beside its pixels (see
:mod:`pymmm.trench_images`). Relative paths are relative to the TOML file's
folder. ``experiment.segmenter()`` builds the Segmenter from it.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Union

if TYPE_CHECKING:
    from pymmm.segmenter import Segmenter

_PATH_KEYS = ("nd2", "store")


@dataclass(frozen=True)
class Experiment:
    """An experiment's paths and per-step settings."""

    name: str
    nd2: Path
    store: Path
    segmentation: Dict[str, Any] = field(default_factory=dict)
    curation: Dict[str, Any] = field(default_factory=dict)
    source: Path | None = None

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Experiment":
        path = Path(path).expanduser().resolve()
        with path.open("rb") as file:
            raw = tomllib.load(file)
        missing = [k for k in ("name", *_PATH_KEYS) if k not in raw]
        if missing:
            raise ValueError(f"{path}: missing {', '.join(missing)}")
        paths = {
            key: (Path(raw[key]) if Path(raw[key]).is_absolute() else path.parent / raw[key])
            for key in _PATH_KEYS
        }
        return cls(
            name=str(raw["name"]),
            segmentation=dict(raw.get("segmentation", {})),
            curation=dict(raw.get("curation", {})),
            source=path,
            **{k: v.resolve() for k, v in paths.items()},
        )

    @property
    def segmentation_name(self) -> str:
        """The segmentation's name (default: the channel, lower case)."""
        seg = self.segmentation
        return str(seg.get("name") or seg.get("channel", "mCherry").lower())

    def segmenter(self, **overrides: Any) -> "Segmenter":
        """A :class:`~pymmm.segmenter.Segmenter` for the experiment group."""
        from pymmm.segmenter import Segmenter

        seg = self.segmentation
        kwargs = {"channel": seg.get("channel", "mCherry"), "name": self.segmentation_name}
        for key in ("model", "hypotheses"):
            if key in seg:
                kwargs[key] = seg[key]
        return Segmenter(self.store, **{**kwargs, **overrides})
