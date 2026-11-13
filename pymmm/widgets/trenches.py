"""The trench detection widget: :meth:`TrenchDetector.interactive_detect_trenches`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np

from pymmm.widgets._common import STYLE, MeanImages, overlay_viewer, show, slider

if TYPE_CHECKING:
    from pymmm.steps.trenches import TrenchDetector

# Settings a lane may override (the trench shape is the same for every lane).
LANE_PARAMETERS = {"sigma", "distance", "prominence", "height", "trench_bottom_offset"}


def trench_widget(detector: "TrenchDetector") -> None:
    """Tune trench detection on one FOV at a time, then apply it to every FOV.

    ipywidgets for the controls (global, and per lane of the FOV: a lane's sliders
    follow the global ones until set on their own), ndv for the registered image with
    the candidate trenches' outlines, matplotlib for each lane's x-profile. **Apply to
    all FOVs** runs :meth:`~TrenchDetector.detect_trenches` with the controls'
    settings and the lanes' overrides.
    """
    import ipywidgets as widgets
    import matplotlib.pyplot as plt

    experiment = detector.experiment
    params = detector.detection_params
    mean_image = MeanImages(detector.registrator, detector.channel)

    fov_select = widgets.Dropdown(
        options=list(experiment.fov_names),
        value=experiment.fov_names[0],
        description="FOV",
    )
    lane_controls = widgets.VBox()
    method_select = widgets.Dropdown(
        options=[
            ("peaks (tallest maxima)", "peaks"),
            ("periodic (spacing + symmetry)", "periodic"),
        ],
        value=params.get("method", "peaks"),
        description="method",
        style=STYLE,
    )
    pitch_slider = slider(
        "float",
        "expected pitch (0 = auto)",
        params.get("expected_pitch", 0),
        0,
        experiment.data.sizes["X"] / 2,
        1,
    )
    symmetry_slider = slider(
        "float", "min symmetry", params.get("min_symmetry", 0.1), 0, 1, 0.05
    )
    sigma_slider = slider("float", "sigma", params.get("sigma", 4), 0, 50, 0.5)
    distance_slider = slider(
        "float", "distance", params.get("distance", 100), 1, 500, 1
    )
    prominence_slider = slider(
        "float", "prominence", params.get("prominence", 10), 0, 500, 1
    )
    trench_width_slider = slider(
        "int", "trench width", params.get("trench_width") or 32, 1, 200, 1
    )
    shape = (
        detector.shape_defaults()
    )  # the whole height when the lanes are the image's edge
    trench_length_slider = slider(
        "int",
        "trench length",
        params.get("trench_length", shape["trench_length"]),
        10,
        experiment.data.sizes["Y"],
        5,
    )
    trench_bottom_offset_slider = slider(
        "int",
        "bottom offset",
        params.get("trench_bottom_offset", shape["trench_bottom_offset"]),
        -200,
        200,
        5,
    )
    apply_btn = widgets.Button(description="Apply to all FOVs", button_style="primary")
    preview_label = widgets.HTML(value="")
    status_label = widgets.HTML(value="")
    output = widgets.Output()
    viewer_data = np.zeros(
        (2, experiment.data.sizes["Y"], experiment.data.sizes["X"]),
        dtype=np.float32,
    )
    image_viewer = overlay_viewer(
        viewer_data, 0, detector.channel, [("candidate trenches", "red")]
    )

    lane_widget_state: Dict[int, Dict[str, widgets.Widget]] = {}
    lane_override_values: Dict[int, Dict[str, Any]] = {
        int(lane_idx): {
            key: value for key, value in values.items() if key in LANE_PARAMETERS
        }
        for lane_idx, values in params.get("lane_params", {}).items()
    }
    syncing_lane_widgets = False
    global_widgets = {
        "sigma": sigma_slider,
        "distance": distance_slider,
        "prominence": prominence_slider,
        "trench_bottom_offset": trench_bottom_offset_slider,
    }

    def _lane_params_from_widgets() -> Dict[int, Dict[str, Any]]:
        lane_params = {
            lane_idx: values.copy()
            for lane_idx, values in lane_override_values.items()
            if values
        }
        for lane_idx, widgets_by_name in lane_widget_state.items():
            lane_overrides = lane_params.setdefault(lane_idx, {})
            for param_name, widget in widgets_by_name.items():
                if widget.value == global_widgets[param_name].value:
                    lane_overrides.pop(param_name, None)
                else:
                    lane_overrides[param_name] = widget.value
            if not lane_overrides:
                lane_params.pop(lane_idx, None)
        return lane_params

    def _make_lane_observer(lane_idx: int, param_name: str):
        def _observer(change: Any) -> None:
            if syncing_lane_widgets:
                return
            value = change["new"]
            lane_overrides = lane_override_values.setdefault(lane_idx, {})
            if value == global_widgets[param_name].value:
                lane_overrides.pop(param_name, None)
                if not lane_overrides:
                    lane_override_values.pop(lane_idx, None)
            else:
                lane_overrides[param_name] = value
            _update()

        return _observer

    def _on_global_change(_change: Any = None) -> None:
        nonlocal syncing_lane_widgets
        syncing_lane_widgets = True
        try:
            for lane_idx, widgets_by_name in lane_widget_state.items():
                lane_overrides = lane_override_values.get(lane_idx, {})
                for param_name, widget in widgets_by_name.items():
                    if param_name in lane_overrides:
                        continue
                    target_value = global_widgets[param_name].value
                    if widget.value != target_value:
                        widget.value = target_value
        finally:
            syncing_lane_widgets = False
        _update()

    def _rebuild_lane_controls(_change: Any = None) -> None:
        fov_lanes = detector.lane_detector.lanes.get(fov_select.value, [])
        new_state: Dict[int, Dict[str, widgets.Widget]] = {}
        accordion_children: List[widgets.Widget] = []

        for lane_info in fov_lanes:
            lane_idx = lane_info.lane_index
            overrides = lane_override_values.get(lane_idx, {})
            lane_widgets = {
                "sigma": slider(
                    "float",
                    f"L{lane_idx} sigma",
                    overrides.get("sigma", sigma_slider.value),
                    0,
                    50,
                    0.5,
                ),
                "distance": slider(
                    "float",
                    f"L{lane_idx} distance",
                    overrides.get("distance", distance_slider.value),
                    1,
                    500,
                    1,
                ),
                "prominence": slider(
                    "float",
                    f"L{lane_idx} prominence",
                    overrides.get("prominence", prominence_slider.value),
                    0,
                    500,
                    1,
                ),
                "trench_bottom_offset": slider(
                    "int",
                    f"L{lane_idx} bottom offset",
                    overrides.get(
                        "trench_bottom_offset", trench_bottom_offset_slider.value
                    ),
                    -200,
                    200,
                    5,
                ),
            }
            for param_name, widget in lane_widgets.items():
                widget.observe(_make_lane_observer(lane_idx, param_name), names="value")
            new_state[lane_idx] = lane_widgets
            accordion_children.append(widgets.VBox(list(lane_widgets.values())))

        lane_widget_state.clear()
        lane_widget_state.update(new_state)

        if accordion_children:
            accordion = widgets.Accordion(children=accordion_children)
            for idx, lane_info in enumerate(fov_lanes):
                accordion.set_title(idx, f"Lane {lane_info.lane_index} controls")
            lane_controls.children = (accordion,)
        else:
            lane_controls.children = (
                widgets.HTML("<i>No lanes detected for this FOV.</i>"),
            )

        _update()

    def _sync_method_controls() -> None:
        periodic = method_select.value == "periodic"
        for widget in (distance_slider, prominence_slider):
            widget.disabled = periodic
        for widget in (pitch_slider, symmetry_slider):
            widget.disabled = not periodic

    def _settings() -> Dict[str, Any]:
        return dict(
            sigma=sigma_slider.value,
            distance=distance_slider.value,
            prominence=prominence_slider.value,
            height=params.get("height", 0),
            trench_width=trench_width_slider.value,
            trench_length=trench_length_slider.value,
            trench_bottom_offset=trench_bottom_offset_slider.value,
            lane_params=_lane_params_from_widgets(),
            method=method_select.value,
            expected_pitch=pitch_slider.value,
            min_symmetry=symmetry_slider.value,
        )

    def _update(_change: Any = None) -> None:
        _sync_method_controls()
        fov = fov_select.value
        mean_img = mean_image(fov)
        fov_trenches, lane_profiles, _ = detector.detect_fov(
            mean_img, fov, **_settings()
        )
        n_trenches = len(fov_trenches)
        n_lanes = len(lane_profiles)

        viewer_data[0] = mean_img
        trench_mask = viewer_data[1]
        trench_mask.fill(0)
        image_height, image_width = trench_mask.shape
        for trench in fov_trenches:
            y_start = max(0, trench.y_top)
            y_stop = min(image_height, trench.y_bottom)
            x_start = max(0, trench.x_left)
            x_stop = min(image_width, trench.x_right)
            if y_start >= y_stop or x_start >= x_stop:
                continue
            trench_mask[y_start : min(y_start + 2, y_stop), x_start:x_stop] = 1
            trench_mask[max(y_stop - 2, y_start) : y_stop, x_start:x_stop] = 1
            trench_mask[y_start:y_stop, x_start : min(x_start + 2, x_stop)] = 1
            trench_mask[y_start:y_stop, max(x_stop - 2, x_start) : x_stop] = 1
        # Blank the image under the outlines so they stay visible where the
        # percentile contrast saturates (e.g. the feeding channel).
        viewer_data[0][trench_mask > 0] = mean_img.min()
        image_viewer.data_wrapper.data_changed.emit()
        preview_label.value = f"<b>{fov}</b> — {n_trenches} candidate trench{'es' if n_trenches != 1 else ''}"

        with output:
            output.clear_output(wait=True)
            if n_lanes == 0:
                fig, ax = plt.subplots(1, 1, figsize=(7, 2))
                ax.text(0.5, 0.5, "No lane profiles", ha="center", va="center")
                ax.axis("off")
            else:
                fig, axes = plt.subplots(
                    n_lanes,
                    1,
                    figsize=(7, max(3, 2.5 * n_lanes)),
                    squeeze=False,
                )
                for i, lane_profile in enumerate(lane_profiles):
                    _plot_lane_profile(axes[i, 0], lane_profile, symmetry_slider.value)
            fig.tight_layout()
            plt.show()

    def _on_apply(_btn: Any) -> None:
        status_label.value = "<b>Applying…</b>"
        detector.detect_trenches(**_settings())
        status_label.value = (
            f"<b>Applied</b> — {detector.n_trenches} trenches across "
            f"{len(detector.trenches)} FOVs"
        )

    apply_btn.on_click(_on_apply)
    fov_select.observe(_rebuild_lane_controls, names="value")
    for w in [
        sigma_slider,
        distance_slider,
        prominence_slider,
        trench_width_slider,
        trench_length_slider,
        trench_bottom_offset_slider,
        method_select,
        pitch_slider,
        symmetry_slider,
    ]:
        w.observe(_on_global_change, names="value")

    show(
        [
            fov_select,
            method_select,
            sigma_slider,
            distance_slider,
            prominence_slider,
            pitch_slider,
            symmetry_slider,
            trench_width_slider,
            trench_length_slider,
            trench_bottom_offset_slider,
            lane_controls,
            apply_btn,
            status_label,
        ],
        preview_label,
        image_viewer,
        output,
    )
    _rebuild_lane_controls()


def _plot_lane_profile(
    ax: Any, lane_profile: Dict[str, Any], min_symmetry: float
) -> None:
    """One lane's x-profile, its trench centres and, for the periodic method, the
    symmetry score against its threshold."""
    x_prof, x_smooth = lane_profile["profile"], lane_profile["smooth"]
    centres = lane_profile["centres"]
    xs = np.arange(len(x_prof))
    ax.plot(xs, x_prof, color="grey", alpha=0.5, label="raw")
    ax.plot(xs, x_smooth, color="blue", label="smoothed")
    title = f"Lane {lane_profile['lane_index']} x-profile (trench bodies)"
    if "symmetry" in lane_profile:
        for centre in centres:
            ax.axvline(centre, color="red", lw=1, alpha=0.8)
        ax_sym = ax.twinx()
        ax_sym.plot(
            xs,
            lane_profile["symmetry"],
            color="tab:green",
            alpha=0.6,
            lw=1,
            label="symmetry",
        )
        ax_sym.axhline(min_symmetry, color="tab:green", ls="--", lw=1)
        ax_sym.set_ylim(-1, 1.05)
        ax_sym.set_ylabel("symmetry", color="tab:green")
        title += f", pitch {lane_profile['pitch']:.1f} px"
    elif len(centres) > 0:
        ax.plot(centres, x_smooth[centres], "rv", markersize=8, label="peaks")
    ax.set_xlabel("x pixel")
    ax.set_ylabel("intensity")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
