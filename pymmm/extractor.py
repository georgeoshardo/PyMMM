"""Extractor – crop trenches from stabilised data and write to output zarr."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import zarr
from numcodecs import Zstd
from tqdm.auto import tqdm

from pymmm._utils import normalize_channel_arg
from pymmm.experiment import ND2Experiment
from pymmm.registrator import Registrator
from pymmm.trench_detector import TrenchDetector


class Extractor:
    """Crop trenches from the lazy stabilised DataArray and write to zarr.

    Parameters
    ----------
    experiment : ND2Experiment
        Source experiment.
    registrator : Registrator
        Completed registrator.
    trench_detector : TrenchDetector
        Completed trench detector.
    output_path : str | Path | None
        Path for the output zarr store. Defaults to
        ``<nd2_dir>/<experiment_name>.trenches.zarr``.
    """

    def __init__(
        self,
        experiment: ND2Experiment,
        registrator: Registrator,
        trench_detector: TrenchDetector,
        output_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.experiment = experiment
        self.registrator = registrator
        self.trench_detector = trench_detector

        if output_path is None:
            nd2_dir = experiment.path.parent
            output_path = nd2_dir / f"{experiment.experiment_name}.trenches.zarr"
        self.output_path = Path(output_path)

    def extract(
        self,
        compressor: str = "zstd",
        clevel: int = 9,
        show_progress: bool = True,
    ) -> None:
        """Extract all trenches and write to the output zarr store.

        Algorithm:
        1. Build lazy stabilised DataArray via ``reg.get_stabilized_data()``.
        2. Create output zarr with shape
           ``(n_trenches, n_times, trench_h, trench_w)``.
        3. For each trench: slice, flip if needed, compute, write.

        Parameters
        ----------
        compressor : str
            Compression codec name (``"zstd"``).
        clevel : int
            Compression level.
        show_progress : bool
            Show a tqdm progress bar.
        """
        trench_table = self.trench_detector.get_trench_table()
        if len(trench_table) == 0:
            raise RuntimeError("No trenches to extract.")

        # Determine output shape from first trench
        first = trench_table.iloc[0]
        trench_h = first["y_bottom"] - first["y_top"]
        trench_w = first["x_right"] - first["x_left"]
        n_trenches = len(trench_table)
        n_times = self.experiment.n_timepoints

        # Determine channels
        has_channels = self.experiment.has_channels
        channel_names = self.experiment.channel_names
        n_channels = len(channel_names) if has_channels else 1

        # Create output zarr store
        if compressor == "zstd":
            comp = Zstd(level=clevel)
        else:
            comp = None

        if n_channels > 1:
            shape = (n_trenches, n_times, n_channels, trench_h, trench_w)
            chunks = (1, n_times, 1, trench_h, trench_w)
        else:
            shape = (n_trenches, n_times, trench_h, trench_w)
            chunks = (1, n_times, trench_h, trench_w)

        store = zarr.open_group(str(self.output_path), mode="w")
        data_arr = store.create_dataset(
            "data",
            shape=shape,
            chunks=chunks,
            dtype=self.experiment.data.dtype,
            compressor=comp,
        )

        # Write metadata
        store.attrs["source_nd2"] = str(self.experiment.path)
        store.attrs["experiment_name"] = self.experiment.experiment_name
        store.attrs["pixel_size_um"] = self.experiment.pixel_size_um
        store.attrs["channel_names"] = channel_names
        store.attrs["n_trenches"] = n_trenches
        store.attrs["n_times"] = n_times
        store.attrs["trench_height"] = int(trench_h)
        store.attrs["trench_width"] = int(trench_w)
        store.attrs["registration_params"] = {
            "channel": self.registrator.channel,
            "mode": str(self.registrator.mode),
            "rotation": self.registrator.rotation,
        }

        # Write trench mapping
        trench_mapping = trench_table.to_dict(orient="records")
        store.attrs["trench_mapping"] = trench_mapping

        # Extract per channel
        iterator = trench_table.iterrows()
        if show_progress:
            iterator = tqdm(
                list(iterator), desc="Extracting trenches", total=n_trenches
            )

        for _, row in iterator:
            tid = row["trench_id"]
            fov = row["fov"]
            y_top = row["y_top"]
            y_bottom = row["y_bottom"]
            x_left = row["x_left"]
            x_right = row["x_right"]
            needs_flip = row["needs_flip"]

            if n_channels > 1:
                for c_idx, ch_name in enumerate(channel_names):
                    stabilized = self.registrator.get_stabilized_data(channel=ch_name)
                    if "P" in stabilized.dims:
                        stack = stabilized.sel(P=fov)
                    else:
                        stack = stabilized

                    crop = stack.isel(
                        Y=slice(y_top, y_bottom), X=slice(x_left, x_right)
                    )
                    crop_np = crop.compute().values

                    if needs_flip:
                        crop_np = crop_np[:, ::-1, :]  # flip Y

                    data_arr[tid, :, c_idx, :, :] = crop_np
            else:
                stabilized = self.registrator.get_stabilized_data()
                if "P" in stabilized.dims:
                    stack = stabilized.sel(P=fov)
                else:
                    stack = stabilized

                crop = stack.isel(
                    Y=slice(y_top, y_bottom), X=slice(x_left, x_right)
                )
                crop_np = crop.compute().values

                if needs_flip:
                    crop_np = crop_np[:, ::-1, :]  # flip Y

                data_arr[tid, :, :, :] = crop_np

        print(f"Extraction complete: {self.output_path}")
        print(f"  Shape: {shape}, Chunks: {chunks}")

    def extract_single_trench(
        self,
        trench_id: int,
        channel: Optional[Union[str, int]] = None,
    ) -> np.ndarray:
        """Extract a single trench for preview (without writing to zarr).

        Parameters
        ----------
        trench_id : int
            Global trench ID.
        channel : str | int | None
            Channel to extract. ``None`` uses the registration channel.

        Returns
        -------
        np.ndarray
            Array of shape ``(T, trench_h, trench_w)``.
        """
        table = self.trench_detector.get_trench_table()
        row = table[table["trench_id"] == trench_id].iloc[0]

        stabilized = self.registrator.get_stabilized_data(channel=channel)

        if "P" in stabilized.dims:
            stack = stabilized.sel(P=row["fov"])
        else:
            stack = stabilized

        crop = stack.isel(
            Y=slice(row["y_top"], row["y_bottom"]),
            X=slice(row["x_left"], row["x_right"]),
        )
        crop_np = crop.compute().values

        if row["needs_flip"]:
            crop_np = crop_np[:, ::-1, :]

        return crop_np

    def __repr__(self) -> str:
        return (
            f"Extractor(output={self.output_path}, "
            f"trenches={self.trench_detector.n_trenches})"
        )
