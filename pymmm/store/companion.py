"""CompanionStore – zarr-based checkpoint I/O for intermediate results."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from pymmm.store import lineage
from pymmm.store.provenance import SCHEMA_VERSION, plain_json, provenance

if TYPE_CHECKING:
    from pymmm.source.nd2 import ND2Experiment

_SECTIONS = ("registration", "lane_detection", "trench_detection")

_LANE_COLUMNS = {
    "fov": {"long_name": "field of view"},
    "lane_index": {"long_name": "lane index within its FOV"},
    "y_center": {
        "units": "pixel",
        "long_name": "lane centre row in the registered FOV",
    },
    "orientation": {
        "long_name": "trench opening direction (1 = opens downward, -1 = upward)"
    },
}
_TRENCH_COLUMNS = {
    "fov": {"long_name": "field of view"},
    "lane_index": {"long_name": "lane the trench belongs to within its FOV"},
    "x_left": {"units": "pixel", "long_name": "trench left edge in the registered FOV"},
    "x_right": {
        "units": "pixel",
        "long_name": "trench right edge in the registered FOV",
    },
    "y_top": {"units": "pixel", "long_name": "trench top edge in the registered FOV"},
    "y_bottom": {
        "units": "pixel",
        "long_name": "trench bottom edge in the registered FOV",
    },
    "orientation": {
        "long_name": "trench opening direction (1 = opens downward, -1 = upward)"
    },
    "needs_flip": {
        "long_name": "crop is flipped vertically so trenches share one orientation"
    },
}


class CompanionStore:
    """Manages a companion ``.pymmm.zarr`` store for persisting intermediate
    pipeline results (registration matrices, lane/trench definitions).

    Each section is an xarray dataset written with ``to_zarr``. Its attrs hold
    the parameters that produced it as native JSON, the experiment subset it
    was computed on, the provenance of the code that wrote it (which includes
    the time), and a ``complete`` flag set only after all of it is written.
    Root attrs identify the source ND2 and are written once, when the store is
    created.

    Structure::

        experiment_name.pymmm.zarr/
        ├── zarr.json             # source ND2 identity, creation time
        ├── registration/         # tmats (P, T, row, col)
        ├── lane_detection/       # Lane-indexed table: fov, lane_index, y_center, orientation
        └── trench_detection/     # Trench-indexed table: fov, lane_index, edges, orientation
    """

    def __init__(self, path: str | Path, mode: str = "a") -> None:
        self.path = Path(path)
        self._store = zarr.open_group(
            str(self.path),
            mode=mode,
            zarr_format=3,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def for_experiment(cls, experiment: ND2Experiment) -> CompanionStore:
        """Derive the companion store path from the ND2 file location.

        The store is placed next to the ND2 file as
        ``<experiment_name>.pymmm.zarr``.
        """
        nd2_path = Path(experiment.path)
        store_path = nd2_path.parent / f"{experiment.experiment_name}.pymmm.zarr"
        store = cls(store_path, mode="a")

        attrs = store._store.attrs
        if "created" not in attrs:
            source_stat = nd2_path.stat()
            attrs.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "layout": "pymmm_companion",
                    "source_nd2": str(nd2_path),
                    "source_file_size_bytes": int(source_stat.st_size),
                    "source_mtime_ns": int(source_stat.st_mtime_ns),
                    "created": datetime.now().astimezone().isoformat(),
                    "provenance": provenance(),
                }
            )
        return store

    # ------------------------------------------------------------------
    # Section I/O
    # ------------------------------------------------------------------

    def _write_section(
        self,
        name: str,
        dataset: xr.Dataset,
        params: dict[str, Any],
        source_subset_metadata: dict[str, Any] | None,
        extra_attrs: dict[str, Any] | None = None,
    ) -> None:
        if name in self._store:
            del self._store[name]
        dataset = dataset.copy()
        dataset.attrs = plain_json(
            {
                "layout": name,
                "params": params,
                "source_subset_metadata": source_subset_metadata,
                "provenance": provenance(),
                **(extra_attrs or {}),
                "complete": False,
            }
        )
        dataset.to_zarr(
            str(self.path),
            group=name,
            mode="a",
            consolidated=False,
            zarr_format=3,
        )
        self._store = zarr.open_group(str(self.path), mode="a", zarr_format=3)
        self._store[name].attrs["complete"] = True

    def _open_section(self, name: str) -> xr.Dataset:
        return xr.open_zarr(str(self.path), group=name, consolidated=False)

    def _has_section(self, name: str) -> bool:
        return name in self._store and bool(self._store[name].attrs.get("complete"))

    def section_attrs(self, name: str) -> dict[str, Any]:
        """Return a section's attrs (params, subset, provenance, complete)."""
        return dict(self._store[name].attrs)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def save_registration(
        self,
        tmats: xr.DataArray,
        params: dict[str, Any],
        source_subset_metadata: dict[str, Any] | None = None,
        made_from: dict[str, str | None] | None = None,
    ) -> None:
        """Persist registration results.

        Parameters
        ----------
        tmats : xr.DataArray
            Transformation matrices with dims ``(P, T, row, col)``.
        params : dict
            Registration parameters for reproducibility.
        source_subset_metadata : dict | None
            The experiment subset the registration was computed on.
        made_from : dict | None
            Fingerprints of what it was made from (:mod:`pymmm.store.lineage`).
        """
        dataset = xr.Dataset({"tmats": tmats.astype("float64")})
        dataset["tmats"].attrs["long_name"] = "registration transform (3x3 affine)"
        self._write_section(
            "registration",
            dataset,
            params,
            source_subset_metadata,
            extra_attrs={"made_from": made_from},
        )

    def load_registration(self) -> dict[str, Any]:
        """Load registration checkpoint.

        Returns
        -------
        dict
            Keys: ``"tmats"`` (xr.DataArray) and ``"params"`` (dict).
        """
        dataset = self._open_section("registration")
        return {"tmats": dataset["tmats"].load(), "params": dataset.attrs["params"]}

    def load_registration_params(self) -> dict[str, Any]:
        """Load registration parameters without reading image arrays."""
        return self.section_attrs("registration")["params"]

    def has_registration(self) -> bool:
        """Check whether registration data is checkpointed."""
        return self._has_section("registration")

    # ------------------------------------------------------------------
    # Lane and trench detection
    # ------------------------------------------------------------------

    def _save_table(
        self,
        name: str,
        dim: str,
        table: pd.DataFrame,
        columns: dict[str, dict[str, str]],
        fov_names: list[str],
        params: dict[str, Any],
        source_subset_metadata: dict[str, Any] | None,
        index: np.ndarray | None = None,
        made_from: dict[str, str | None] | None = None,
    ) -> None:
        data_vars = {}
        for column, attrs in columns.items():
            values = table[column].to_numpy()
            if column == "fov":
                values = np.asarray([str(v) for v in values], dtype=object)
            data_vars[column] = xr.DataArray(values, dims=(dim,), attrs=attrs)
        coords = {
            dim: index if index is not None else np.arange(len(table), dtype=np.int32)
        }
        self._write_section(
            name,
            xr.Dataset(data_vars, coords=coords),
            params,
            source_subset_metadata,
            extra_attrs={"fov_names": fov_names, "made_from": made_from},
        )

    def _load_table(self, name: str, dim: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        dataset = self._open_section(name).load()
        table = dataset.to_dataframe().reset_index()
        if dim == "Lane":
            table = table.drop(columns=dim)
        else:
            table = table.rename(columns={dim: "trench_id"})
        return table, dict(dataset.attrs)

    def save_lane_detection(
        self,
        lanes: pd.DataFrame,
        fov_names: list[str],
        params: dict[str, Any],
        source_subset_metadata: dict[str, Any] | None = None,
        made_from: dict[str, str | None] | None = None,
    ) -> None:
        """Persist lane detection results.

        Parameters
        ----------
        lanes : pd.DataFrame
            One row per lane with columns ``fov``, ``lane_index``,
            ``y_center`` and ``orientation``.
        fov_names : list[str]
            FOVs that were processed, including any with no lanes.
        params : dict
            Detection parameters.
        source_subset_metadata : dict | None
            The experiment subset the lanes were detected on.
        made_from : dict | None
            Fingerprints of what they were made from (:mod:`pymmm.store.lineage`).
        """
        self._save_table(
            "lane_detection",
            "Lane",
            lanes,
            _LANE_COLUMNS,
            fov_names,
            params,
            source_subset_metadata,
            made_from=made_from,
        )

    def load_lane_detection(self) -> dict[str, Any]:
        """Load lane detection checkpoint.

        Returns
        -------
        dict
            Keys: ``"lanes"`` (pd.DataFrame), ``"fov_names"`` (list),
            ``"params"`` (dict).
        """
        table, attrs = self._load_table("lane_detection", "Lane")
        return {
            "lanes": table,
            "fov_names": attrs["fov_names"],
            "params": attrs["params"],
        }

    def has_lane_detection(self) -> bool:
        """Check whether lane detection data is checkpointed."""
        return self._has_section("lane_detection")

    def save_trench_detection(
        self,
        trenches: pd.DataFrame,
        fov_names: list[str],
        params: dict[str, Any],
        source_subset_metadata: dict[str, Any] | None = None,
        made_from: dict[str, str | None] | None = None,
    ) -> None:
        """Persist trench detection results.

        Parameters
        ----------
        trenches : pd.DataFrame
            One row per trench with a ``trench_id`` column plus the
            ``TrenchDefinition`` fields.
        fov_names : list[str]
            FOVs that were processed, including any with no trenches.
        params : dict
            Detection parameters.
        source_subset_metadata : dict | None
            The experiment subset the trenches were detected on.
        made_from : dict | None
            Fingerprints of what they were made from (:mod:`pymmm.store.lineage`).
        """
        self._save_table(
            "trench_detection",
            "Trench",
            trenches,
            _TRENCH_COLUMNS,
            fov_names,
            params,
            source_subset_metadata,
            index=trenches["trench_id"].to_numpy(dtype=np.int32),
            made_from=made_from,
        )

    def load_trench_detection(self) -> dict[str, Any]:
        """Load trench detection checkpoint.

        Returns
        -------
        dict
            Keys: ``"trenches"`` (pd.DataFrame), ``"fov_names"`` (list),
            ``"params"`` (dict).
        """
        table, attrs = self._load_table("trench_detection", "Trench")
        return {
            "trenches": table,
            "fov_names": attrs["fov_names"],
            "params": attrs["params"],
        }

    def has_trench_detection(self) -> bool:
        """Check whether trench detection data is checkpointed."""
        return self._has_section("trench_detection")

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Lineage
    # ------------------------------------------------------------------

    def made_from(self, section: str) -> dict[str, str | None] | None:
        """What a saved section records it was made from; None if it records nothing."""
        if not self._has_section(section):
            return None
        return self.section_attrs(section).get("made_from")

    def fingerprints(self) -> dict[str, str | None]:
        """The fingerprint of each section's result as saved; None for one missing."""
        found: dict[str, str | None] = dict.fromkeys(_SECTIONS)
        if self.has_registration():
            saved = self.load_registration()
            found["registration"] = lineage.registration_fingerprint(
                saved["tmats"], saved["params"].get("rotation", 0.0)
            )
        if self.has_lane_detection():
            found["lane_detection"] = lineage.table_fingerprint(
                self.load_lane_detection()["lanes"].to_dict("records"),
                lineage.LANE_COLUMNS,
            )
        if self.has_trench_detection():
            found["trench_detection"] = lineage.table_fingerprint(
                self.load_trench_detection()["trenches"].to_dict("records"),
                lineage.TRENCH_COLUMNS,
            )
        return found

    def lineage(
        self, experiment: ND2Experiment, experiment_group: str | Path | None = None
    ) -> dict[str, str]:
        """Each step's status (:func:`pymmm.store.lineage.status`): ``"current"``,
        ``"stale"``, ``"missing"`` or ``"unrecorded"``, for ``experiment`` (the ND2 and
        the part of it used). With ``experiment_group``, the extraction's too.
        """
        fingerprints = {
            "source": lineage.source_fingerprint(experiment),
            **self.fingerprints(),
            "extraction": None,
        }
        made_from = {section: self.made_from(section) for section in _SECTIONS}
        if experiment_group is not None:
            group = Path(experiment_group)
            if (group / "zarr.json").exists():
                attrs = dict(zarr.open_group(str(group), mode="r").attrs)
                record = attrs.get("pymmm") or {}
                if record.get("layout") == "experiment":
                    made_from["extraction"] = record.get("made_from")
                    fingerprints["extraction"] = lineage.fingerprint(
                        record.get("made_from") or {}
                    )
        return lineage.status(fingerprints, made_from)

    def clear(self, *sections: str) -> None:
        """Delete these sections, quietly, the ones present (partly written ones too):
        a step about to be redone, and the steps after it.
        """
        for section in sections:
            if section in self._store:
                del self._store[section]

    def reset(self, section: str | None = None) -> None:
        """Delete checkpointed data and re-open the store.

        Parameters
        ----------
        section : str | None
            ``"registration"``, ``"lane_detection"``, or
            ``"trench_detection"`` to clear just that section.
            ``None`` (default) deletes the entire store and re-creates it.
        """
        if section is not None:
            if section in self._store:
                del self._store[section]
                print(f"Cleared '{section}' from {self.path}")
            else:
                print(f"No '{section}' section found — nothing to clear")
        else:
            shutil.rmtree(self.path)
            self._store = zarr.open_group(
                str(self.path),
                mode="a",
                zarr_format=3,
            )
            print(f"Store reset: {self.path}")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        sections = [name for name in _SECTIONS if self._has_section(name)]
        status = ", ".join(sections) if sections else "empty"
        return f"CompanionStore({self.path}, [{status}])"
