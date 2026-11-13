"""The Segmenter's widget and viewer: :meth:`Segmenter.interactive_segment` and
:meth:`Segmenter.view`.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np
import zarr

from pymmm.steps.segmentation import DEFAULT_PARAMS, overrides, trench_bounds
from pymmm.store.experiment_group import (
    has_segmentation,
    image,
    open_experiment,
    segmentation,
)
from pymmm.widgets._common import imshow, overlay_luts, overlay_viewer, show, slider

if TYPE_CHECKING:
    from pymmm.steps.segmentation import Segmenter

OUTLINES = [("cell outlines", "red")]


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
    out = np.empty((*images.shape[:-2], 2, *images.shape[-2:]), dtype=np.float32)
    frames = images.reshape((-1, *images.shape[-2:]))
    frame_labels = labels.reshape((-1, *labels.shape[-2:]))
    flat_out = out.reshape((-1, 2, *images.shape[-2:]))
    for i, (frame_image, frame) in enumerate(zip(frames, frame_labels)):
        outline = find_boundaries(frame, mode="inner")
        flat_out[i, 0] = frame_image
        flat_out[i, 0][outline] = frame_image.min()
        flat_out[i, 1] = outline
    return out


def segmentation_widget(segmenter: Segmenter, batch_size: int = 64) -> None:
    """Tune the base settings on single trenches (see :meth:`Segmenter.interactive_segment`)."""
    import ipywidgets as widgets
    import matplotlib.pyplot as plt

    params = segmenter.params
    trench_ids = segmenter.trench_ids
    fov_names = [str(f) for f in segmenter.root["trenches/fov_name"][:]]
    n_t, _, n_y, n_x = image(segmenter.root, trench_ids[0]).shape

    hypothesis_select = widgets.Dropdown(
        options=segmenter.hypotheses,
        value=segmenter.hypotheses[0],
        description="hypothesis",
    )
    overrides_label = widgets.HTML(value="")
    trench_slider = slider("int", "trench (index)", 0, 0, len(trench_ids) - 1, 1)
    floor_slider = slider(
        "int",
        "upper-bound floor",
        int(params["floor"]),
        0,
        max(int(params["floor"] * 4), 1),
        1,
    )
    gain_slider = slider("float", "gain", params["gain"], 0.5, 3.0, 0.05)
    rescale_slider = slider("float", "image scale", params["rescale"], 0.5, 1.5, 0.05)
    mask_slider = slider(
        "float", "mask threshold", params["mask_threshold"], -3.0, 3.0, 0.1
    )
    flow_slider = slider(
        "float", "flow threshold", params["flow_threshold"], 0.0, 1.5, 0.05
    )
    area_slider = slider("float", "min area (µm²)", params["min_area"], 0.0, 5.0, 0.05)
    contrast_on = params["min_contrast"] is not None
    contrast_check = widgets.Checkbox(value=contrast_on, description="drop faint masks")
    contrast_slider = slider(
        "float",
        "min contrast (σ)",
        params["min_contrast"] if contrast_on else DEFAULT_PARAMS["min_contrast"],
        0.0,
        20.0,
        0.5,
    )
    run_btn = widgets.Button(description="Segment trench", button_style="primary")
    apply_btn = widgets.Button(description="Apply", button_style="success")
    preview_label = widgets.HTML(value="")
    status_label = widgets.HTML(value="")
    output = widgets.Output()

    viewer_data = np.zeros((n_t, 2, n_y, n_x), dtype=np.float32)
    image_viewer = overlay_viewer(viewer_data, 1, segmenter.channel, OUTLINES)

    def _widget_params() -> dict[str, Any]:
        return {
            **params,
            "floor": float(floor_slider.value),
            "gain": float(gain_slider.value),
            "rescale": float(rescale_slider.value),
            "mask_threshold": float(mask_slider.value),
            "flow_threshold": float(flow_slider.value),
            "min_area": float(area_slider.value),
            "min_contrast": float(contrast_slider.value)
            if contrast_check.value
            else None,
        }

    def _show_overrides(_change: Any = None) -> None:
        spec = segmenter.spec(hypothesis_select.value)
        source = (
            f"checkpoint {spec['checkpoint']}"
            if "checkpoint" in spec
            else f"model {spec['model']}"
        )
        over = overrides(spec)
        overrides_label.value = (
            source
            + "; "
            + (
                "overrides: " + ", ".join(f"{k} {v}" for k, v in over.items())
                if over
                else "uses the sliders as they are"
            )
        )

    def _update(_change: Any = None) -> None:
        _show_overrides()
        index = trench_slider.value
        tid = trench_ids[index]
        hypothesis = hypothesis_select.value
        current = _widget_params()
        status_label.value = f"<b>Segmenting trench {tid} ({hypothesis})…</b>"
        stack = segmenter.read(tid)
        bounds = trench_bounds(stack, current["floor"])
        start = time.perf_counter()
        masks, failed = segmenter.segment_stack(
            stack, bounds, hypothesis, current, batch_size
        )
        elapsed = time.perf_counter() - start
        viewer_data[:] = label_overlay(stack, masks)
        image_viewer.data_wrapper.data_changed.emit()

        n_cells = masks.reshape(masks.shape[0], -1).max(axis=1)
        preview_label.value = (
            f"<b>Trench {tid}</b> ({fov_names[index]}), {hypothesis} — up to "
            f"{int(n_cells.max())} cells; bounds {bounds[0]:.0f}–{bounds[1]:.0f}"
        )
        status_label.value = f"Segmented {n_t} frames in {elapsed:.1f} s"
        if failed:
            status_label.value += f"; Omnipose failed on frames {failed}"

        with output:
            output.clear_output(wait=True)
            fig, ax = plt.subplots(1, 1, figsize=(7, 2))
            ax.plot(n_cells, color="tab:red")
            ax.set_xlabel("frame")
            ax.set_ylabel("cells")
            ax.set_title(f"Trench {tid}, {hypothesis}: cells per frame")
            fig.tight_layout()
            plt.show()

    def _on_apply(_btn: Any) -> None:
        stale = segmenter.set_params(_widget_params())
        status_label.value = "<b>Applied</b> — base settings kept for segment()"
        if stale:
            status_label.value += (
                f"<br>{len(stale)} trenches already have results with the previous "
                "settings; segment() will ask you to re-segment them too."
            )

    run_btn.on_click(_update)
    apply_btn.on_click(_on_apply)
    trench_slider.observe(_update, names="value")
    hypothesis_select.observe(_update, names="value")

    show(
        [
            hypothesis_select,
            overrides_label,
            trench_slider,
            floor_slider,
            gain_slider,
            rescale_slider,
            mask_slider,
            flow_slider,
            area_slider,
            contrast_check,
            contrast_slider,
            widgets.HBox([run_btn, apply_btn]),
            status_label,
        ],
        preview_label,
        image_viewer,
        output,
    )
    _update()


def label_viewer(segmenter: Segmenter) -> Any:
    """An ndv viewer of the written labels as outlines over the channel (see
    :meth:`Segmenter.view`).
    """
    return imshow(
        _label_overlay(segmenter),
        channel_axis=3,
        channel_mode="composite",
        luts=overlay_luts(segmenter.channel, OUTLINES),
    )


def _label_overlay(segmenter: Segmenter) -> Any:
    """Ndv data ``(Hypothesis, Trench, T, C, Y, X)``: the channel and label
    outlines, read from the trench images only for the frames ndv asks for.
    """
    import ndv

    ids = segmenter.trench_ids
    hypotheses = segmenter.hypotheses
    n_t, _, n_y, n_x = image(segmenter.root, ids[0]).shape

    def _labels(trench_id: int) -> list[zarr.Array] | None:
        root = open_experiment(segmenter.store_path)  # fresh: sees labels written since
        if not has_segmentation(root, trench_id, segmenter.name):
            return None
        return segmentation(root, trench_id, segmenter.name)[1]

    class _LabelOverlay(ndv.DataWrapper):
        @classmethod
        def supports(cls, obj: Any) -> bool:
            return False

        @property
        def dims(self) -> tuple[str, ...]:
            return ("Hypothesis", "Trench", "T", "C", "Y", "X")

        @property
        def coords(self) -> dict[str, Sequence]:
            # ndv's Jupyter sliders only take ranges: Hypothesis and Trench are by position.
            return {
                "Hypothesis": range(len(hypotheses)),
                "Trench": range(len(ids)),
                "T": range(n_t),
                "C": range(2),
                "Y": range(n_y),
                "X": range(n_x),
            }

        def channel_names(self, channel_axis: int | None) -> dict[int, str]:
            return {0: segmenter.channel, 1: "outlines"}

        @property
        def dtype(self) -> np.dtype:
            return np.dtype(np.float32)

        def isel(self, index: Mapping[int, int | slice]) -> np.ndarray:
            # ndv asks with a slice on every axis (an index k as k:k+1).
            h, trench, t, c, y, x = (
                slice(i, i + 1) if isinstance(i, (int, np.integer)) else i
                for i in (index.get(axis, slice(None)) for axis in range(6))
            )
            per_trench = []
            for k in range(len(ids))[trench]:
                frames = np.asarray(
                    image(segmenter.root, ids[k])[t, segmenter.channel_index]
                )
                arrays = _labels(ids[k])
                per_trench.append(
                    np.stack(
                        [
                            label_overlay(
                                frames,
                                np.zeros(frames.shape, np.uint16)
                                if arrays is None
                                else np.asarray(arrays[j][t]),
                            )
                            for j in range(len(hypotheses))[h]
                        ]
                    )
                )
            return np.stack(per_trench, axis=1)[..., c, y, x]

    return _LabelOverlay(segmenter)
