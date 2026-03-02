"""CompanionStore – zarr-based checkpoint I/O for intermediate results."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import zarr

if TYPE_CHECKING:
    from pymmm.experiment import ND2Experiment


class CompanionStore:
    """Manages a companion ``.pymmm.zarr`` store for persisting intermediate
    pipeline results (registration matrices, lane/trench definitions, etc.).

    Structure::

        experiment_name.pymmm.zarr/
        ├── .zattrs               # source ND2 path, creation date, FOV/channel names
        ├── registration/
        │   ├── tmats             # zarr array (n_FOVs, n_times, 3, 3) float64
        │   ├── mean_images       # zarr array (n_FOVs, Y, X) float64
        │   └── .zattrs           # registration params
        ├── lane_detection/
        │   └── .zattrs           # params + per-FOV lane data as JSON
        └── trench_detection/
            └── .zattrs           # params + trench table as JSON
    """

    def __init__(self, path: str | Path, mode: str = "a") -> None:
        self.path = Path(path)
        self._store = zarr.open_group(str(self.path), mode=mode)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def for_experiment(cls, experiment: "ND2Experiment") -> "CompanionStore":
        """Derive the companion store path from the ND2 file location.

        The store is placed next to the ND2 file as
        ``<experiment_name>.pymmm.zarr``.
        """
        nd2_path = Path(experiment.path)
        store_path = nd2_path.parent / f"{experiment.experiment_name}.pymmm.zarr"
        store = cls(store_path, mode="a")

        # Write global attrs on first creation
        attrs = store._store.attrs
        if "source_nd2" not in attrs:
            attrs["source_nd2"] = str(nd2_path)
            attrs["created"] = datetime.now().isoformat()
            attrs["fov_names"] = list(experiment.fov_names)
            attrs["channel_names"] = list(experiment.channel_names)

        return store

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def save_registration(
        self,
        tmats: np.ndarray,
        mean_images: np.ndarray,
        params: Dict[str, Any],
    ) -> None:
        """Persist registration results.

        Parameters
        ----------
        tmats : np.ndarray
            Transformation matrices, shape ``(n_fovs, n_times, 3, 3)``.
        mean_images : np.ndarray
            Mean reference images, shape ``(n_fovs, Y, X)``.
        params : dict
            Registration parameters for reproducibility.
        """
        grp = self._store.require_group("registration")

        # Overwrite arrays if they exist
        tmats_np = np.asarray(tmats, dtype="float64")
        if "tmats" in grp:
            del grp["tmats"]
        arr = grp.create_array("tmats", shape=tmats_np.shape, dtype="float64")
        arr[:] = tmats_np

        mean_np = np.asarray(mean_images, dtype="float64")
        if "mean_images" in grp:
            del grp["mean_images"]
        arr = grp.create_array("mean_images", shape=mean_np.shape, dtype="float64")
        arr[:] = mean_np

        # Store params as JSON string in attrs
        grp.attrs["params"] = json.dumps(params, default=str)

    def load_registration(self) -> Dict[str, Any]:
        """Load registration checkpoint.

        Returns
        -------
        dict
            Keys: ``"tmats"`` (np.ndarray), ``"mean_images"`` (np.ndarray),
            ``"params"`` (dict).
        """
        grp = self._store["registration"]
        return {
            "tmats": np.array(grp["tmats"]),
            "mean_images": np.array(grp["mean_images"]),
            "params": json.loads(grp.attrs["params"]),
        }

    def has_registration(self) -> bool:
        """Check whether registration data is checkpointed."""
        return "registration" in self._store and "tmats" in self._store["registration"]

    # ------------------------------------------------------------------
    # Lane detection
    # ------------------------------------------------------------------

    def save_lane_detection(
        self,
        lane_data: Dict[str, Any],
        params: Dict[str, Any],
    ) -> None:
        """Persist lane detection results.

        Parameters
        ----------
        lane_data : dict
            Per-FOV lane information (serialisable to JSON).
        params : dict
            Detection parameters.
        """
        grp = self._store.require_group("lane_detection")
        grp.attrs["lane_data"] = json.dumps(lane_data, default=str)
        grp.attrs["params"] = json.dumps(params, default=str)

    def load_lane_detection(self) -> Dict[str, Any]:
        """Load lane detection checkpoint.

        Returns
        -------
        dict
            Keys: ``"lane_data"`` (dict), ``"params"`` (dict).
        """
        grp = self._store["lane_detection"]
        return {
            "lane_data": json.loads(grp.attrs["lane_data"]),
            "params": json.loads(grp.attrs["params"]),
        }

    def has_lane_detection(self) -> bool:
        """Check whether lane detection data is checkpointed."""
        return (
            "lane_detection" in self._store
            and "lane_data" in self._store["lane_detection"].attrs
        )

    # ------------------------------------------------------------------
    # Trench detection
    # ------------------------------------------------------------------

    def save_trench_detection(
        self,
        trench_data: Dict[str, Any],
        params: Dict[str, Any],
    ) -> None:
        """Persist trench detection results.

        Parameters
        ----------
        trench_data : dict
            Per-FOV trench definitions (serialisable to JSON).
        params : dict
            Detection parameters.
        """
        grp = self._store.require_group("trench_detection")
        grp.attrs["trench_data"] = json.dumps(trench_data, default=str)
        grp.attrs["params"] = json.dumps(params, default=str)

    def load_trench_detection(self) -> Dict[str, Any]:
        """Load trench detection checkpoint.

        Returns
        -------
        dict
            Keys: ``"trench_data"`` (dict), ``"params"`` (dict).
        """
        grp = self._store["trench_detection"]
        return {
            "trench_data": json.loads(grp.attrs["trench_data"]),
            "params": json.loads(grp.attrs["params"]),
        }

    def has_trench_detection(self) -> bool:
        """Check whether trench detection data is checkpointed."""
        return (
            "trench_detection" in self._store
            and "trench_data" in self._store["trench_detection"].attrs
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, section: Optional[str] = None) -> None:
        """Delete checkpointed data and re-open the store.

        Parameters
        ----------
        section : str | None
            ``"registration"``, ``"lane_detection"``, or
            ``"trench_detection"`` to clear just that section.
            ``None`` (default) deletes the entire store and re-creates it.
        """
        import shutil

        if section is not None:
            if section in self._store:
                del self._store[section]
                print(f"Cleared '{section}' from {self.path}")
            else:
                print(f"No '{section}' section found — nothing to clear")
        else:
            shutil.rmtree(self.path)
            self._store = zarr.open_group(str(self.path), mode="a")
            print(f"Store reset: {self.path}")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        sections = []
        if self.has_registration():
            sections.append("registration")
        if self.has_lane_detection():
            sections.append("lane_detection")
        if self.has_trench_detection():
            sections.append("trench_detection")
        status = ", ".join(sections) if sections else "empty"
        return f"CompanionStore({self.path}, [{status}])"
