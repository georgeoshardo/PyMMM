"""Segmenter – Omnipose cell labels for every trench crop in a trench store."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import dask.array as da
import numpy as np
import xarray as xr
import zarr
from tqdm.auto import tqdm

from pymmm.provenance import SCHEMA_VERSION, provenance

_LABEL_DIMS = ("Trench", "T", "Y", "X")

_LABELS_ATTRS = {
    "long_name": (
        "cell labels; 0 is background, 1..n are ordered from the trench's closed "
        "end (Y = 0), so 1 is the mother cell"
    ),
}

DEFAULT_PARAMS: Dict[str, Any] = {
    "lower": None,
    "upper": None,
    "mask_threshold": 0.0,
    "flow_threshold": 0.4,
    "min_size": 15,
}
"""Omnipose settings. ``lower``/``upper`` are the fixed intensity bounds mapped to
0 and 1 before the network; ``None`` means estimate them from the store."""


def _json_native(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


# Omnipose 2.0 logs mask building through the top-level "core" and "gpu" loggers.
_OMNIPOSE_LOGGERS = ("omnipose", "core", "gpu")


@contextmanager
def _quiet_omnipose() -> Iterator[None]:
    """Hide Omnipose's per-frame INFO logging (e.g. one line per empty frame)."""
    loggers = [logging.getLogger(name) for name in _OMNIPOSE_LOGGERS]
    previous = [logger.level for logger in loggers]
    for logger in loggers:
        logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        for logger, level in zip(loggers, previous):
            logger.setLevel(level)


def order_from_closed_end(masks: np.ndarray) -> np.ndarray:
    """Relabel each frame of ``masks`` 1..n by centroid row, smallest Y first.

    Trench crops are flipped so every trench opens downward; Y = 0 is the closed
    end, so label 1 is the mother cell.

    Parameters
    ----------
    masks : np.ndarray
        Integer labels, ``(T, Y, X)``.

    Returns
    -------
    np.ndarray
        ``uint16`` labels of the same shape.
    """
    masks = np.asarray(masks)
    out = np.zeros(masks.shape, dtype=np.uint16)
    rows = np.broadcast_to(
        np.arange(masks.shape[-2], dtype=np.float64)[:, None], masks.shape[-2:],
    ).ravel()
    for t, frame in enumerate(masks):
        flat = frame.ravel()
        counts = np.bincount(flat)
        ids = np.flatnonzero(counts)
        ids = ids[ids > 0]
        if ids.size == 0:
            continue
        centroid = np.bincount(flat, weights=rows)[ids] / counts[ids]
        lut = np.zeros(counts.size, dtype=np.uint16)
        lut[ids[np.argsort(centroid, kind="stable")]] = np.arange(
            1, ids.size + 1, dtype=np.uint16,
        )
        out[t] = lut[frame]
    return out


def label_overlay(images: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Stack ``images`` with the outlines of ``labels`` as a second channel.

    Pixels under an outline are set to the image minimum so the outline stays
    visible under percentile contrast.

    Parameters
    ----------
    images, labels : np.ndarray
        ``(..., Y, X)`` arrays of the same shape.

    Returns
    -------
    np.ndarray
        ``float32`` array ``(..., 2, Y, X)``: image, then outline (0/1).
    """
    from skimage.segmentation import find_boundaries

    images = np.asarray(images, dtype=np.float32)
    labels = np.asarray(labels)
    out = np.empty(images.shape[:-2] + (2,) + images.shape[-2:], dtype=np.float32)
    frames = images.reshape((-1,) + images.shape[-2:])
    frame_labels = labels.reshape((-1,) + labels.shape[-2:])
    flat_out = out.reshape((-1, 2) + images.shape[-2:])
    for i, (image, frame) in enumerate(zip(frames, frame_labels)):
        outline = find_boundaries(frame, mode="inner")
        flat_out[i, 0] = image
        flat_out[i, 0][outline] = image.min()
        flat_out[i, 1] = outline
    return out


class Segmenter:
    """Segment one channel of a trench store with Omnipose.

    Labels are written to ``/segmentation/<name>`` inside the trench store, as
    ``labels(Trench, T, Y, X)`` with the root dataset's coordinates and one chunk
    per trench, so they line up with ``data``::

        tree = xr.open_datatree(store, engine="zarr", consolidated=False)
        labels = tree["segmentation/mcherry"].ds["labels"]

    Parameters
    ----------
    store_path : str | Path
        Trench store written by :class:`~pymmm.extractor.Extractor`.
    channel : str
        Channel to segment, a value of the root ``C`` coordinate.
    model : str
        Omnipose pretrained model name.
    name : str | None
        Segmentation group name; defaults to ``channel.lower()``.
    gpu : bool
        Run Omnipose on the GPU.
    """

    def __init__(
        self,
        store_path: Union[str, Path],
        channel: str = "mCherry",
        model: str = "bact_fluor_omni",
        name: Optional[str] = None,
        gpu: bool = True,
    ) -> None:
        self.store_path = Path(store_path)
        self.root = xr.open_zarr(self.store_path, consolidated=False)
        channels = [str(c) for c in self.root["C"].values]
        if channel not in channels:
            raise ValueError(f"channel {channel!r} not in store channels {channels}")
        self.channel = channel
        self.channel_index = channels.index(channel)
        self.model_name = model
        self.name = name or channel.lower()
        self.group = f"segmentation/{self.name}"
        self.gpu = gpu
        self._model = None

        saved = self._saved_attrs()
        if (
            saved is not None
            and saved.get("channel") == channel
            and saved.get("model") == model
        ):
            self._params = dict(saved["params"])
        else:
            self._params = dict(DEFAULT_PARAMS)
        if self._params["lower"] is None or self._params["upper"] is None:
            self._params["lower"], self._params["upper"] = self.estimate_bounds()

    # ------------------------------------------------------------------
    # Store access
    # ------------------------------------------------------------------

    def _open_store(self, mode: str = "r") -> zarr.Group:
        return zarr.open_group(str(self.store_path), mode=mode, zarr_format=3)

    def _saved_attrs(self) -> Optional[Dict[str, Any]]:
        store = self._open_store()
        if self.group not in store:
            return None
        return dict(store[self.group].attrs)

    @property
    def params(self) -> Dict[str, Any]:
        """Current Omnipose settings (see :data:`DEFAULT_PARAMS`)."""
        return dict(self._params)

    @property
    def is_complete(self) -> bool:
        attrs = self._saved_attrs()
        return bool(attrs and attrs.get("segmentation_complete"))

    def _group_attrs(self) -> Dict[str, Any]:
        """Attrs that identify what produced the labels (checked on resume)."""
        root_provenance = self.root.attrs.get("provenance", {})
        return _json_native({
            "channel": self.channel,
            "model": self.model_name,
            "params": self._params,
            "extraction_created": root_provenance.get("created"),
        })

    def _resume_mismatches(self, attrs: Mapping[str, Any]) -> List[str]:
        current = self._group_attrs()
        return [key for key, value in current.items() if attrs.get(key) != value]

    def _create_group(self) -> None:
        """(Re)write ``/segmentation/<name>`` with empty labels; root is untouched."""
        sizes = self.root.sizes
        shape = tuple(sizes[dim] for dim in _LABEL_DIMS)
        chunks = (1,) + shape[1:]
        dataset = xr.Dataset(
            data_vars={
                "labels": (
                    _LABEL_DIMS,
                    da.zeros(shape, chunks=chunks, dtype=np.uint16),
                    _LABELS_ATTRS,
                ),
            },
            coords={dim: self.root[dim].variable for dim in _LABEL_DIMS},
            attrs={
                "schema_version": SCHEMA_VERSION,
                "layout": "segmentation",
                "description": (
                    f"Omnipose labels of the {self.channel} channel for each trench "
                    "crop in the root dataset."
                ),
                "provenance": provenance(),
                **self._group_attrs(),
                "completed_trenches": [],
                "failed_frames": {},
                "segmentation_complete": False,
            },
        )
        dataset.to_zarr(
            str(self.store_path),
            group=self.group,
            mode="w",
            compute=False,
            consolidated=False,
            zarr_format=3,
            encoding={
                "labels": {
                    "chunks": chunks,
                    "compressors": [
                        zarr.codecs.BloscCodec(
                            cname="zstd", clevel=3, shuffle="bitshuffle",
                        )
                    ],
                },
            },
        )

    # ------------------------------------------------------------------
    # Omnipose
    # ------------------------------------------------------------------

    def estimate_bounds(
        self,
        n_trenches: int = 16,
        percentiles: Tuple[float, float] = (1.0, 99.9),
        seed: int = 0,
    ) -> Tuple[float, float]:
        """Intensity percentiles of the channel over a random sample of trenches.

        One fixed pair of bounds for the whole store keeps frames with no cells
        (e.g. after lysis) dark; per-frame normalisation stretches their noise to
        full range and Omnipose finds cells in it.
        """
        data = self._open_store()["data"]
        n_total = self.root.sizes["Trench"]
        rng = np.random.default_rng(seed)
        picks = np.sort(rng.choice(n_total, size=min(n_trenches, n_total), replace=False))
        sample = np.stack([
            data[int(i), ::5, self.channel_index] for i in picks
        ])
        lower, upper = np.percentile(sample, percentiles)
        return float(lower), float(upper)

    def _get_model(self):
        if self._model is None:
            from omnipose.models import OmniModel

            with _quiet_omnipose():
                self._model = OmniModel(gpu=self.gpu, model_type=self.model_name)
        return self._model

    def _eval_kwargs(self, params: Mapping[str, Any], batch_size: int) -> Dict[str, Any]:
        return {
            "normalize": False,
            "show_progress": False,
            "batch_size": batch_size,
            "mask_threshold": params["mask_threshold"],
            "flow_threshold": params["flow_threshold"],
            "min_size": params["min_size"],
        }

    def segment_stack(
        self,
        stack: np.ndarray,
        params: Optional[Mapping[str, Any]] = None,
        batch_size: int = 64,
    ) -> Tuple[np.ndarray, List[int]]:
        """Segment a ``(T, Y, X)`` stack of one trench.

        Returns the labels ordered from the closed end and the frames Omnipose
        failed on (left empty). Omnipose 2.0 can raise ``IndexError`` while
        building masks for some frames; the stack is then retried frame by frame.
        """
        params = self._params if params is None else params
        lower, upper = float(params["lower"]), float(params["upper"])
        x = (np.asarray(stack, dtype=np.float32) - lower) / (upper - lower)
        model = self._get_model()
        kwargs = self._eval_kwargs(params, batch_size)
        failed: List[int] = []
        with _quiet_omnipose():
            try:
                masks = np.asarray(model.eval(x, **kwargs).masks)
            except IndexError:
                masks = np.zeros(x.shape, dtype=np.int32)
                for t, frame in enumerate(x):
                    try:
                        masks[t] = np.asarray(model.eval(frame[None], **kwargs).masks)[0]
                    except IndexError:
                        failed.append(t)
        return order_from_closed_end(masks), failed

    # ------------------------------------------------------------------
    # Batch segmentation
    # ------------------------------------------------------------------

    def segment(
        self,
        resume: bool = True,
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> None:
        """Segment every trench and write ``/segmentation/<name>/labels``.

        Trenches are done one at a time on the GPU; the next trench is read
        while the current one is segmented. Each finished trench is recorded in
        the group's ``completed_trenches`` attr, so an interrupted run resumes.

        Parameters
        ----------
        resume : bool
            Continue an existing group made with the same channel, model,
            settings and extraction. ``False`` rewrites the group.
        batch_size : int
            Frames per Omnipose GPU batch.
        show_progress : bool
            Show a tqdm progress bar.
        """
        saved = self._saved_attrs()
        if resume and saved is not None:
            mismatches = self._resume_mismatches(saved)
            if mismatches:
                raise RuntimeError(
                    f"Existing {self.group} is not compatible with these settings; "
                    f"these differ: {', '.join(mismatches)}. "
                    "Use resume=False to overwrite it or choose another name."
                )
        else:
            self._create_group()

        store = self._open_store(mode="r+")
        data = store["data"]
        group = store[self.group]
        labels = group["labels"]
        trench_ids = [int(i) for i in self.root["Trench"].values]
        completed = set(group.attrs.get("completed_trenches", []))
        failed_frames = dict(group.attrs.get("failed_frames", {}))
        todo = [i for i, tid in enumerate(trench_ids) if tid not in completed]

        def _read(index: int) -> np.ndarray:
            return data[index, :, self.channel_index]

        progress = tqdm(total=len(todo), desc=f"Segmenting {self.channel}", disable=not show_progress)
        with ThreadPoolExecutor(max_workers=1) as reader:
            pending = reader.submit(_read, todo[0]) if todo else None
            for k, index in enumerate(todo):
                stack = pending.result()
                if k + 1 < len(todo):
                    pending = reader.submit(_read, todo[k + 1])
                masks, failed = self.segment_stack(stack, batch_size=batch_size)
                labels[index] = masks
                tid = trench_ids[index]
                completed.add(tid)
                if failed:
                    failed_frames[str(tid)] = failed
                    group.attrs["failed_frames"] = failed_frames
                group.attrs["completed_trenches"] = [t for t in trench_ids if t in completed]
                progress.update()
        progress.close()

        group.attrs["segmentation_complete"] = completed.issuperset(trench_ids)
        print(f"Segmentation complete: {self.store_path / self.group}")
        if failed_frames:
            n = sum(len(v) for v in failed_frames.values())
            print(f"  Omnipose failed on {n} frames (left empty); see attrs['failed_frames']")

    def labels(self) -> xr.DataArray:
        """Lazy ``labels(Trench, T, Y, X)`` from the store."""
        return xr.open_zarr(
            self.store_path, group=self.group, consolidated=False,
        )["labels"]

    # ------------------------------------------------------------------
    # Interactive tuning and viewing
    # ------------------------------------------------------------------

    def interactive_segment(self, batch_size: int = 64) -> None:
        """Display a widget for tuning the Omnipose settings on single trenches.

        Uses ipywidgets for the controls, ndv for the channel with label
        outlines (all frames of the selected trench), and matplotlib for the
        number of cells per frame. **Apply** keeps the settings for
        :meth:`segment`.
        """
        import ipywidgets as widgets
        import matplotlib.pyplot as plt
        import ndv
        from IPython.display import display

        from pymmm.plotting import NDV_CLIMS

        params = self._params
        trench_ids = [int(i) for i in self.root["Trench"].values]
        data = self._open_store()["data"]
        sizes = self.root.sizes
        slider_style = {"description_width": "initial"}

        trench_slider = widgets.IntSlider(
            value=0, min=0, max=len(trench_ids) - 1,
            description="trench (index)", continuous_update=False, style=slider_style,
        )
        intensity_max = max(int(params["upper"] * 3), 1)
        lower_slider = widgets.IntSlider(
            value=int(params["lower"]), min=0, max=intensity_max,
            description="lower bound", continuous_update=False, style=slider_style,
        )
        upper_slider = widgets.IntSlider(
            value=int(params["upper"]), min=1, max=intensity_max,
            description="upper bound", continuous_update=False, style=slider_style,
        )
        mask_slider = widgets.FloatSlider(
            value=params["mask_threshold"], min=-4.0, max=4.0, step=0.1,
            description="mask threshold", continuous_update=False, style=slider_style,
        )
        flow_slider = widgets.FloatSlider(
            value=params["flow_threshold"], min=0.0, max=2.0, step=0.05,
            description="flow threshold", continuous_update=False, style=slider_style,
        )
        min_size_slider = widgets.IntSlider(
            value=params["min_size"], min=0, max=500,
            description="min size (px)", continuous_update=False, style=slider_style,
        )
        run_btn = widgets.Button(description="Segment trench", button_style="primary")
        apply_btn = widgets.Button(description="Apply", button_style="success")
        preview_label = widgets.HTML(value="")
        status_label = widgets.HTML(value="")
        output = widgets.Output()

        viewer_data = np.zeros(
            (sizes["T"], 2, sizes["Y"], sizes["X"]), dtype=np.float32,
        )
        image_viewer = ndv.ArrayViewer(
            viewer_data,
            channel_axis=1,
            channel_mode="composite",
            luts={
                0: {"name": self.channel, "cmap": "gray", "clims": NDV_CLIMS},
                1: {"name": "cell outlines", "cmap": "red", "clims": (0, 1)},
            },
        )

        def _widget_params() -> Dict[str, Any]:
            return {
                "lower": float(lower_slider.value),
                "upper": float(upper_slider.value),
                "mask_threshold": float(mask_slider.value),
                "flow_threshold": float(flow_slider.value),
                "min_size": int(min_size_slider.value),
            }

        def _update(_change: Any = None) -> None:
            if lower_slider.value >= upper_slider.value:
                status_label.value = "<b>lower bound must be below upper bound</b>"
                return
            index = trench_slider.value
            tid = trench_ids[index]
            status_label.value = f"<b>Segmenting trench {tid}…</b>"
            stack = data[index, :, self.channel_index]
            start = time.perf_counter()
            masks, failed = self.segment_stack(stack, _widget_params(), batch_size)
            elapsed = time.perf_counter() - start
            viewer_data[:] = label_overlay(stack, masks)
            image_viewer.data_wrapper.data_changed.emit()

            n_cells = masks.reshape(masks.shape[0], -1).max(axis=1)
            fov = str(self.root["fov_name"].values[index])
            preview_label.value = (
                f"<b>Trench {tid}</b> ({fov}) — up to {int(n_cells.max())} cells"
            )
            status_label.value = f"Segmented {sizes['T']} frames in {elapsed:.1f} s"
            if failed:
                status_label.value += f"; Omnipose failed on frames {failed}"

            with output:
                output.clear_output(wait=True)
                fig, ax = plt.subplots(1, 1, figsize=(7, 2))
                ax.plot(n_cells, color="tab:red")
                ax.set_xlabel("frame")
                ax.set_ylabel("cells")
                ax.set_title(f"Trench {tid}: cells per frame")
                fig.tight_layout()
                plt.show()

        def _on_apply(_btn: Any) -> None:
            if lower_slider.value >= upper_slider.value:
                status_label.value = "<b>lower bound must be below upper bound</b>"
                return
            self._params = _widget_params()
            status_label.value = "<b>Applied</b> — settings kept for segment()"

        run_btn.on_click(_update)
        apply_btn.on_click(_on_apply)
        trench_slider.observe(_update, names="value")

        controls = widgets.VBox([
            trench_slider, lower_slider, upper_slider,
            mask_slider, flow_slider, min_size_slider,
            widgets.HBox([run_btn, apply_btn]), status_label,
        ])
        image_and_counts = widgets.VBox([
            preview_label, image_viewer.widget(), output,
        ])
        display(widgets.HBox([controls, image_and_counts]))
        _update()

    def view(self):
        """Browse the stored labels as outlines over the channel, any trench and frame."""
        import ndv

        from pymmm.plotting import NDV_CLIMS

        return ndv.imshow(
            _label_overlay_wrapper(self),
            channel_axis=2,
            channel_mode="composite",
            luts={
                0: {"name": self.channel, "cmap": "gray", "clims": NDV_CLIMS},
                1: {"name": "cell outlines", "cmap": "red", "clims": (0, 1)},
            },
        )

    def __repr__(self) -> str:
        attrs = self._saved_attrs()
        if attrs is None:
            state = "not segmented"
        else:
            done = len(attrs.get("completed_trenches", []))
            state = f"{done}/{self.root.sizes['Trench']} trenches segmented"
        return (
            f"Segmenter({self.store_path.name}, {self.group}, channel={self.channel!r}, "
            f"model={self.model_name!r}, {state})"
        )


def _label_overlay_wrapper(segmenter: Segmenter):
    """ndv data ``(Trench, T, C, Y, X)``: the channel and label outlines, read
    from the store only for the frames ndv asks for."""
    import ndv

    store = segmenter._open_store()
    data = store["data"]
    labels = store[segmenter.group]["labels"]
    channel_index = segmenter.channel_index
    trench_ids = [int(i) for i in segmenter.root["Trench"].values]
    sizes = segmenter.root.sizes

    class _LabelOverlay(ndv.DataWrapper):
        @classmethod
        def supports(cls, obj: Any) -> bool:
            return False

        @property
        def dims(self) -> Tuple[str, ...]:
            return ("Trench", "T", "C", "Y", "X")

        @property
        def coords(self) -> Dict[str, Sequence]:
            # ndv's Jupyter sliders only take ranges: Trench is by position.
            return {
                "Trench": range(len(trench_ids)),
                "T": range(sizes["T"]),
                "C": range(2),
                "Y": range(sizes["Y"]),
                "X": range(sizes["X"]),
            }

        def channel_names(self, channel_axis: Optional[int]) -> Dict[int, str]:
            return {0: segmenter.channel, 1: "outlines"}

        @property
        def dtype(self) -> np.dtype:
            return np.dtype(np.float32)

        def isel(self, index: Mapping[int, Union[int, slice]]) -> np.ndarray:
            trench, t, c, y, x = (index.get(axis, slice(None)) for axis in range(5))
            overlay = label_overlay(
                data[trench, t, channel_index], labels[trench, t],
            )
            return overlay[..., c, y, x]

    return _LabelOverlay(segmenter)
