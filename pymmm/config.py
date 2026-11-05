"""Experiment configuration – one TOML file per experiment.

Every notebook of the pipeline (extract, segment, track, curate) reads the same
file, so a new dataset means writing one file::

    name = "20261005_pbs_psc_yfp_ty013"
    nd2 = "raw_data/20261005_pbs_psc_yfp_ty013.nd2"
    trench_store = "20261005_pbs_psc_yfp_ty013.trenches.zarr"
    tracks_store = "20261005_pbs_psc_yfp_ty013.tracks.zarr"
    curation_store = "20261005_pbs_psc_yfp_ty013.curation.zarr"

    [segmentation]
    channel = "mCherry"
    model = "bact_fluor_omni"
    name = "mcherry"         # /segmentation/<name> in the trench store
    batch_size = 64
    workers = 8

    [curation]
    kymograph_channel = "PC"

Relative paths are relative to the TOML file's folder. ``experiment.segmenter()``
builds the Segmenter from it.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Union

if TYPE_CHECKING:
    from pymmm.segmenter import Segmenter

_PATH_KEYS = ("nd2", "trench_store", "tracks_store", "curation_store")


@dataclass(frozen=True)
class Experiment:
    """An experiment's paths and per-step settings."""

    name: str
    nd2: Path
    trench_store: Path
    tracks_store: Path
    curation_store: Path
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
        """The segmentation group name (default: the channel, lower case)."""
        seg = self.segmentation
        return str(seg.get("name") or seg.get("channel", "mCherry").lower())

    def segmenter(self, **overrides: Any) -> "Segmenter":
        """A :class:`~pymmm.segmenter.Segmenter` for the trench store."""
        from pymmm.segmenter import Segmenter

        seg = self.segmentation
        kwargs = {"channel": seg.get("channel", "mCherry"), "name": self.segmentation_name}
        if "model" in seg:
            kwargs["model"] = seg["model"]
        return Segmenter(self.trench_store, **{**kwargs, **overrides})
