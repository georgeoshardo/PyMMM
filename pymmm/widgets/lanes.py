"""The lane detection widget: :meth:`LaneDetector.interactive_detect_lanes`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from pymmm.widgets._common import STYLE, MeanImages, overlay_viewer, show, slider

if TYPE_CHECKING:
    from pymmm.steps.lanes import LaneDetector


def lane_widget(detector: "LaneDetector") -> None:
    """Tune lane detection on one FOV at a time, then apply it to every FOV.

    ipywidgets for the controls, ndv for the registered image with the candidate
    lanes (red: trenches above, open down; cyan: below, open up), matplotlib for the
    intensity profile. **Apply to all FOVs** runs :meth:`~LaneDetector.detect_lanes`
    with the controls' settings.
    """
    import ipywidgets as widgets
    import matplotlib.pyplot as plt

    experiment = detector.experiment
    params = detector.detection_params
    mean_image = MeanImages(detector.registrator, detector.channel)

    fov_select = widgets.Dropdown(
        options=list(experiment.fov_names), value=experiment.fov_names[0], description="FOV",
    )
    sigma_slider = slider("float", "sigma", params.get("sigma", 40), 0, 200, 1)
    distance_slider = slider("float", "distance", params.get("distance", 300), 10, 1000, 10)
    height_slider = slider("float", "height", params.get("height", 5000), 0, 50000, 100)
    orientation_slider = slider("float", "orient. window (µm)",
                                params.get("orientation_window_um", 20), 1, 100, 1)
    orientation_select = widgets.Dropdown(
        options=[
            ("auto (compare brightness)", "auto"),
            ("trenches above lane, open down (red)", "down"),
            ("trenches below lane, open up (cyan)", "up"),
        ],
        value=params.get("orientation", "auto"),
        description="orientation",
        style=STYLE,
    )
    orientation_slider.disabled = orientation_select.value != "auto"
    apply_btn = widgets.Button(description="Apply to all FOVs", button_style="primary")
    preview_label = widgets.HTML(value="")
    status_label = widgets.HTML(value="")
    output = widgets.Output()

    viewer_data = np.zeros(
        (3, experiment.data.sizes["Y"], experiment.data.sizes["X"]), dtype=np.float32,
    )
    image_viewer = overlay_viewer(viewer_data, 0, detector.channel,
                                  [("open down", "red"), ("open up", "cyan")])

    def _update(_change: Any = None) -> None:
        fov = fov_select.value
        mean_img = mean_image(fov)
        y_profile, y_profile_smooth, peaks, fov_lanes = detector.detect_fov(
            mean_img,
            sigma_slider.value,
            distance_slider.value,
            height_slider.value,
            orientation_slider.value,
            fov,
            orientation=orientation_select.value,
        )
        orientation_slider.disabled = orientation_select.value != "auto"
        n_lanes = len(fov_lanes)

        viewer_data[0] = mean_img
        viewer_data[1:].fill(0)
        for lane in fov_lanes:
            channel_idx = 1 if lane.orientation == 1 else 2
            y_start = max(0, lane.y_center - 2)
            y_stop = min(viewer_data.shape[1], lane.y_center + 3)
            viewer_data[channel_idx, y_start:y_stop, :] = 1
        # Lanes sit on the bright feeding channel, which the percentile contrast
        # saturates; the composite view adds colours, so blank the image under the
        # lines to keep them visible.
        viewer_data[0][viewer_data[1:].any(axis=0)] = mean_img.min()
        image_viewer.data_wrapper.data_changed.emit()
        preview_label.value = (
            f"<b>{fov}</b> — {n_lanes} candidate lane{'s' if n_lanes != 1 else ''}"
        )

        with output:
            output.clear_output(wait=True)
            fig, ax_prof = plt.subplots(1, 1, figsize=(7, 4))
            xs = np.arange(len(y_profile))
            ax_prof.plot(xs, y_profile, color="grey", alpha=0.5, label="raw")
            ax_prof.plot(xs, y_profile_smooth, color="blue", label="smoothed")
            if len(peaks) > 0:
                ax_prof.plot(peaks, y_profile_smooth[peaks], "rv", markersize=8, label="peaks")
            ax_prof.set_xlabel("y pixel")
            ax_prof.set_ylabel("intensity")
            ax_prof.set_title("Y-profile")
            ax_prof.legend()
            fig.tight_layout()
            plt.show()

    def _on_apply(_btn: Any) -> None:
        status_label.value = "<b>Applying…</b>"
        detector.detect_lanes(
            sigma=sigma_slider.value,
            distance=distance_slider.value,
            height=height_slider.value,
            orientation_window_um=orientation_slider.value,
            orientation=orientation_select.value,
        )
        lanes = detector.lanes
        total = sum(len(v) for v in lanes.values())
        status_label.value = f"<b>Applied</b> — {total} lanes across {len(lanes)} FOVs"

    apply_btn.on_click(_on_apply)
    for w in [fov_select, sigma_slider, distance_slider,
              height_slider, orientation_slider, orientation_select]:
        w.observe(_update, names="value")

    show([fov_select, sigma_slider, distance_slider, height_slider, orientation_select,
          orientation_slider, apply_btn, status_label],
         preview_label, image_viewer, output)
    _update()
