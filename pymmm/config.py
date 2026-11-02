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

    [tracking]
    tracker = "trackastra"
    hypotheses = ["S1"]
    model = "general_2d"
    mode = "ilp"
    image_channel = "mCherry"

    [curation]
    kymograph_channel = "PC"

Relative paths are relative to the TOML file's folder. ``experiment.segmenter()``
and ``experiment.tracker("S1")`` build the pipeline objects from it.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Union

if TYPE_CHECKING:
    from pymmm.segmenter import Segmenter
    from pymmm.tracks import Tracker

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
    tracking: Dict[str, Any] = field(default_factory=dict)
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
            tracking=dict(raw.get("tracking", {})),
            curation=dict(raw.get("curation", {})),
            source=path,
            **{k: v.resolve() for k, v in paths.items()},
        )

    @property
    def hypotheses(self) -> List[str]:
        """The segmentation hypotheses to track (default S1)."""
        return list(self.tracking.get("hypotheses", ["S1"]))

    def tracks_name(self, hypothesis: str) -> str:
        """The name of a hypothesis's tracks, e.g. ``trackastra_S1``."""
        return f"{self.tracking.get('tracker', 'trackastra')}_{hypothesis}"

    @property
    def segmentation_name(self) -> str:
        """The segmentation group name (default: the channel, lower case)."""
        seg = self.segmentation
        return str(seg.get("name") or seg.get("channel", "mCherry").lower())

    def tracks_collection(self, hypothesis: str) -> Path:
        """Where a hypothesis's generated tracks are written."""
        return self.tracks_store / self.tracks_name(hypothesis)

    def curation_collection(self, hypothesis: str) -> Path:
        """Where a hypothesis's tracks are curated (edited in place by mm-track)."""
        return self.curation_store / self.tracks_name(hypothesis)

    def segmenter(self, **overrides: Any) -> "Segmenter":
        """A :class:`~pymmm.segmenter.Segmenter` for the trench store."""
        from pymmm.segmenter import Segmenter

        seg = self.segmentation
        kwargs = {"channel": seg.get("channel", "mCherry"), "name": self.segmentation_name}
        if "model" in seg:
            kwargs["model"] = seg["model"]
        return Segmenter(self.trench_store, **{**kwargs, **overrides})

    def tracker(self, hypothesis: str, **overrides: Any) -> "Tracker":
        """A :class:`~pymmm.tracks.Tracker` for one hypothesis."""
        from pymmm.tracks import Tracker

        tracking = self.tracking
        kwargs = {key: tracking[key] for key in ("image_channel", "model", "mode", "device")
                  if key in tracking}
        return Tracker(
            self.trench_store, self.tracks_store, segmentation=self.segmentation_name,
            hypothesis=hypothesis, name=self.tracks_name(hypothesis), **{**kwargs, **overrides},
        )
