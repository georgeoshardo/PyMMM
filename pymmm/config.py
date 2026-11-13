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

or, for hypotheses that are fine-tuned checkpoints, in place of ``model`` and
``hypotheses``::

    models = "models"                       # the checkpoints' folder
    checkpoints = {e250 = "omni_epoch_250", e050 = "omni_epoch_50"}   # the base first
    params = {masks = "affinity", floor = 0.0, min_contrast = "off"}
    model_pixel_size = 0.1079               # µm/px the checkpoints learned from, if not the images'

``store`` is the experiment group the Extractor writes: one OME-Zarr image per
trench, with each trench's segmentations and tracks beside its pixels (see
:mod:`pymmm.store.experiment_group`). ``nd2`` is the raw acquisition the Extractor
reads; an experiment group made some other way (imported from another tool's
trenches, say) has none, and leaves it out. Relative paths (``models`` too) are
relative to the TOML file's folder. ``experiment.segmenter()`` builds the Segmenter from it;
``params`` are the base settings of a new segmentation. ``experiment.tracker(chaintrack)``
builds the Tracker of that segmentation; tracking has no settings in the file (its
constants are chaintrack's recipe, frozen in the group when the experiment is fitted).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Union

if TYPE_CHECKING:
    from pymmm.steps.segmentation import Segmenter
    from pymmm.steps.tracking import Tracker

_PATH_KEYS = ("store", "nd2")
_REQUIRED = ("name", "store")


@dataclass(frozen=True)
class Experiment:
    """An experiment's paths and per-step settings (``nd2`` is None when it has no ND2)."""

    name: str
    store: Path
    nd2: Path | None = None
    segmentation: Dict[str, Any] = field(default_factory=dict)
    curation: Dict[str, Any] = field(default_factory=dict)
    source: Path | None = None

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Experiment":
        path = Path(path).expanduser().resolve()
        with path.open("rb") as file:
            raw = tomllib.load(file)
        missing = [k for k in _REQUIRED if k not in raw]
        if missing:
            raise ValueError(f"{path}: missing {', '.join(missing)}")
        paths = {
            key: (Path(raw[key]) if Path(raw[key]).is_absolute() else path.parent / raw[key])
            for key in _PATH_KEYS
            if key in raw
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
        """A :class:`~pymmm.steps.segmentation.Segmenter` for the experiment group."""
        from pymmm.steps.segmentation import Segmenter

        from pymmm.steps.segmentation import checkpoint_hypotheses

        seg = self.segmentation
        kwargs: Dict[str, Any] = {"channel": seg.get("channel", "mCherry"), "name": self.segmentation_name}
        for key in ("model", "hypotheses", "params", "model_pixel_size"):
            if key in seg:
                kwargs[key] = seg[key]
        if "checkpoints" in seg:
            if "hypotheses" in seg:
                raise ValueError(f"{self.source}: [segmentation] gives both hypotheses and checkpoints")
            kwargs["hypotheses"] = checkpoint_hypotheses(seg["checkpoints"])
        if "models" in seg:
            models = Path(seg["models"]).expanduser()
            if not models.is_absolute() and self.source is not None:
                models = self.source.parent / models
            kwargs["models"] = models.resolve()
        return Segmenter(self.store, **{**kwargs, **overrides})

    def tracker(self, chaintrack: Union[str, Path], **overrides: Any) -> "Tracker":
        """A :class:`~pymmm.steps.tracking.Tracker` of the experiment's segmentation,
        running the chaintrack at ``chaintrack`` (its ``chain.py``)."""
        from pymmm.steps.tracking import Tracker

        return Tracker(self.store, self.segmentation_name, chaintrack, **overrides)
