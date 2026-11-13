"""The stitched view: every FOV of an experiment where it sat on the stage, in one napari
view, for looking around the whole chip. Nothing is resampled or saved.

Each FOV is its own image layer of raw frames, placed by the ND2's stage position and
camera matrix (:class:`~pymmm.source.stage.StageLayout`); napari turns and places it.
World coordinates are the Experiment tab's map coordinates (µm), so the two line up
exactly. Frame drift is left out.

Three things keep it light:

- frames are read lazily, one channel at a time, through one cache shared by all FOVs
  (:class:`FrameSource`);
- each FOV is a two-level image, full size and :data:`REDUCTION` times smaller, and
  napari shows the small one when zoomed out;
- FOVs outside the view are hidden, and napari reads nothing for a hidden layer.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import numpy as np

from pymmm.source.stage import fov_frames, rotation

REDUCTION = 8
"""How much smaller the zoomed-out level of each FOV is."""

MARGIN = 0.25
"""FOVs this far (in view widths) outside the view load too, so panning finds them
ready."""


def _batched(viewer: Any):
    """The viewer's batch of layer changes: one scene-graph rebuild and one garbage
    collection at the end, not one per layer.
    """
    return viewer.layers.batched_update()


def reduce(frame: np.ndarray, factor: int) -> np.ndarray:
    """``frame`` shrunk by ``factor`` (mean over each block), in its own dtype."""
    height, width = (n // factor * factor for n in frame.shape)
    blocks = frame[:height, :width].reshape(
        height // factor, factor, width // factor, factor
    )
    return blocks.mean(axis=(1, 3)).astype(frame.dtype)


class FrameSource:
    """One channel's raw frames of every FOV, read on demand and kept in a cache:
    full frames (the last ``cache_frames``) and reduced ones (``cache_reduced``).
    """

    def __init__(
        self,
        experiment: Any,
        channel: str,
        cache_frames: int = 96,
        cache_reduced: int = 64 * 16,
    ) -> None:
        data = (
            experiment.data.sel(C=channel)
            if "C" in experiment.data.dims
            else experiment.data
        )
        if "Z" in data.dims:
            data = data.isel(Z=0)
        self.data = data.transpose("T", "P", "Y", "X") if "P" in data.dims else data
        self.has_p = "P" in data.dims
        self.shape = (int(data.sizes["T"]), int(data.sizes["Y"]), int(data.sizes["X"]))
        self.dtype = np.dtype(data.dtype)
        self._full: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
        self._reduced: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
        self._sizes = (int(cache_frames), int(cache_reduced))
        self._lock = threading.Lock()
        self.reads = 0  # frames read from the file

    @staticmethod
    def _cached(cache: OrderedDict, key: tuple, limit: int, make: Callable):
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        value = make()
        cache[key] = value
        while len(cache) > limit:
            cache.popitem(last=False)
        return value

    def full(self, p: int, t: int) -> np.ndarray:
        """FOV ``p`` (the experiment's index) at frame ``t``, full size."""

        def read() -> np.ndarray:
            self.reads += 1
            frame = self.data[t, p] if self.has_p else self.data[t]
            return np.asarray(frame.values)

        with self._lock:
            return self._cached(self._full, (p, t), self._sizes[0], read)

    def reduced(self, p: int, t: int) -> np.ndarray:
        """FOV ``p`` at frame ``t``, :data:`REDUCTION` times smaller."""
        with self._lock:
            if (p, t) in self._reduced:
                self._reduced.move_to_end((p, t))
                return self._reduced[(p, t)]
        small = reduce(self.full(p, t), REDUCTION)
        with self._lock:
            return self._cached(self._reduced, (p, t), self._sizes[1], lambda: small)


class FovFrames:
    """One FOV of a :class:`FrameSource` as a lazy ``(t, y, x)`` array, full size or
    reduced: a level of the FOV's napari image.
    """

    ndim = 3

    def __init__(self, source: FrameSource, p: int, reduced: bool = False) -> None:
        self.source, self.p, self.is_reduced = source, int(p), bool(reduced)
        n_t, height, width = source.shape
        if reduced:
            height, width = height // REDUCTION, width // REDUCTION
        self.shape = (n_t, height, width)
        self.dtype = source.dtype

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    def __len__(self) -> int:
        return self.shape[0]

    def _frame(self, t: int) -> np.ndarray:
        if self.is_reduced:
            return self.source.reduced(self.p, t)
        return self.source.full(self.p, t)

    def __getitem__(self, key: Any) -> np.ndarray:
        if not isinstance(key, tuple):
            key = (key,)
        if any(k is Ellipsis for k in key):
            i = key.index(Ellipsis)
            key = key[:i] + (slice(None),) * (self.ndim - len(key) + 1) + key[i + 1 :]
        key = key + (slice(None),) * (self.ndim - len(key))
        t_key, *yx = key
        times = range(self.shape[0])[t_key]
        if isinstance(times, int):
            return self._frame(times)[tuple(yx)]
        return np.stack([self._frame(t)[tuple(yx)] for t in times])

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        array = self[...]
        return array if dtype is None else array.astype(dtype)

    def __repr__(self) -> str:
        return f"FovFrames(p={self.p}, shape={self.shape})"


class MosaicView:
    """Every FOV of ``experiment`` in ``viewer`` at its stage position.

    Parameters
    ----------
    viewer : napari.Viewer or ViewerModel
        Where to show it; its other layers are the caller's to hide or remove.
    experiment : ND2Experiment
        With a stage layout (:attr:`~pymmm.ND2Experiment.stage_layout`).
    channel : str
        The channel shown.
    degrees : float
        The registration's rotation: the map's axes (the registered frames upright).
    contrast_limits : (low, high) | None
        Shared by every FOV; by default from :meth:`_contrast` at ``start_t``.
        Contrast, gamma and colormap stay linked across the FOVs, so a change to one
        is a change to all.
    on_view : callable | None
        Called with the view rectangle (x0, y0, x1, y1, µm) after the camera moves.
    """

    def __init__(
        self,
        viewer: Any,
        experiment: Any,
        channel: str,
        degrees: float = 0.0,
        contrast_limits: tuple[float, float] | None = None,
        colormap: str = "gray",
        start_t: int = 0,
        on_view: Callable[[tuple[float, float, float, float]], None] | None = None,
    ) -> None:
        frames = fov_frames(experiment, degrees)
        if frames is None:
            raise ValueError("this ND2 has no stage positions")
        self.viewer = viewer
        self.experiment = experiment
        self.channel = channel
        self.degrees = degrees
        self.on_view = on_view
        self.source = FrameSource(experiment, channel)
        self.outlines = {fov: frames[fov][1] for fov in experiment.fov_names}
        self.layers: dict[str, Any] = {}
        self.overlays: list[Any] = []
        layout = experiment.stage_layout
        y_roi, x_roi = experiment.source_roi
        offset = np.array(
            [(x_roi[0] or 0) if x_roi else 0, (y_roi[0] or 0) if y_roi else 0], float
        )
        c_sensor = np.array([layout.sensor[0] / 2 - 0.5, layout.sensor[1] / 2 - 0.5])
        turn = rotation(degrees).T
        to_map = turn @ np.linalg.inv(layout.camera)
        names = experiment.source_subset_metadata["raw_fov_names"]
        limits = contrast_limits or self._contrast(start_t)
        with _batched(viewer):
            self._add_layers(
                experiment,
                names,
                layout,
                offset,
                c_sensor,
                turn,
                to_map,
                limits,
                colormap,
            )
        viewer.dims.axis_labels = ("T", "y", "x")
        viewer.scene.camera.events.center.connect(self._camera_moved)
        viewer.scene.camera.events.zoom.connect(self._camera_moved)
        viewer.canvas.events.size.connect(self._camera_moved)
        self._closed = False

    def _add_layers(
        self,
        experiment,
        names,
        layout,
        offset,
        c_sensor,
        turn,
        to_map,
        limits,
        colormap,
    ) -> None:
        """One two-level image layer per FOV, hidden until :meth:`update` finds it
        in view.
        """
        channel = self.channel
        for p, fov in enumerate(experiment.fov_names):
            raw_p = names.index(fov)
            # a raw pixel q (x, y) of the crop lies on the map at linear @ q + shift
            linear = layout.pixel_size * turn
            shift = to_map @ layout.positions[raw_p] + linear @ (offset - c_sensor)
            affine = np.eye(4)
            affine[1:3, 1:3] = linear[::-1, ::-1]  # napari's axes are (y, x)
            affine[1:3, 3] = shift[::-1]
            levels = [
                FovFrames(self.source, p),
                FovFrames(self.source, p, reduced=True),
            ]
            self.layers[fov] = self.viewer.add_image(
                levels,
                multiscale=True,
                name=f"{fov} ({channel})",
                affine=affine,
                contrast_limits=limits,
                colormap=colormap,
                blending="additive",
                axis_labels=("T", "y", "x"),
                units=("pixel", "micrometer", "micrometer"),
                visible=False,
            )
        from napari.experimental import link_layers

        # one contrast for the whole chip: a slider moved on any FOV moves them all
        link_layers(
            list(self.layers.values()), ("contrast_limits", "gamma", "colormap")
        )

    # ------------------------------------------------------------------

    def _contrast(self, t: int, n_fovs: int = 8) -> tuple[float, float]:
        """Limits shared by every FOV: the 1st and 99.8th percentiles of ``n_fovs``
        FOVs spread over the experiment, at frame ``t`` (their zoomed-out level).
        """
        n = len(self.experiment.fov_names)
        picks = sorted(set(np.linspace(0, n - 1, min(n, n_fovs)).round().astype(int)))
        values = np.concatenate([self.source.reduced(p, t).ravel() for p in picks])
        low, high = np.percentile(values, [1, 99.8])
        return float(low), float(max(high, low + 1))

    def close(self) -> None:
        """Remove the layers and stop following the camera."""
        if self._closed:
            return
        self._closed = True
        from napari.experimental import unlink_layers

        unlink_layers(list(self.layers.values()))
        self.viewer.scene.camera.events.center.disconnect(self._camera_moved)
        self.viewer.scene.camera.events.zoom.disconnect(self._camera_moved)
        self.viewer.canvas.events.size.disconnect(self._camera_moved)
        with _batched(self.viewer):
            for layer in [*self.layers.values(), *self.overlays]:
                if layer in self.viewer.layers:
                    self.viewer.layers.remove(layer)
        self.layers.clear()
        self.overlays.clear()

    def set_marks(
        self,
        lanes: list[np.ndarray],
        orientations: list[int],
        boxes: list[np.ndarray],
    ) -> None:
        """Lanes ((2, 2) x, y µm ends) and trench boxes ((4, 2) corners) on top."""
        for layer in self.overlays:
            if layer in self.viewer.layers:
                self.viewer.layers.remove(layer)
        self.overlays = []
        common = {"face_color": "transparent", "opacity": 0.9}
        if lanes:
            colours = ["red" if o == 1 else "cyan" for o in orientations]
            self.overlays.append(
                self.viewer.add_shapes(
                    [np.asarray(s)[:, ::-1] for s in lanes],
                    shape_type="line",
                    edge_color=colours,
                    edge_width=1.5,
                    name="lanes (stitched)",
                    units=("micrometer", "micrometer"),
                    **common,
                )
            )
        if boxes:
            self.overlays.append(
                self.viewer.add_shapes(
                    [np.asarray(b)[:, ::-1] for b in boxes],
                    shape_type="polygon",
                    edge_color="red",
                    edge_width=1,
                    name="trenches (stitched)",
                    units=("micrometer", "micrometer"),
                    **common,
                )
            )

    def show_all(self) -> None:
        """Fit every FOV in the view."""
        corners = np.vstack(list(self.outlines.values()))
        self.fit(corners.min(axis=0), corners.max(axis=0))

    def fit(self, low: np.ndarray, high: np.ndarray) -> None:
        """Centre the view on the box from ``low`` to ``high`` (x, y µm) and zoom so
        it fills it.
        """
        width, height = self._canvas()
        span = np.maximum(np.asarray(high) - np.asarray(low), 1e-6)
        self.viewer.scene.camera.center = tuple(
            (np.asarray(low) + np.asarray(high))[::-1] / 2
        )
        self.viewer.scene.camera.zoom = 0.95 * min(width / span[0], height / span[1])
        self.update()

    def go_to(self, x: float, y: float) -> None:
        """Centre the view on (x, y) µm; zoomed out past a few FOVs, zoom to the FOV
        there.
        """
        fov = self.fov_at(x, y)
        x0, _, x1, _ = self.view_rect()
        width = np.ptp(next(iter(self.outlines.values()))[:, 0])
        if fov is not None and (x1 - x0) > 3 * width:
            outline = self.outlines[fov]
            self.fit(outline.min(axis=0), outline.max(axis=0))
            return
        self.viewer.scene.camera.center = (y, x)
        self.update()

    def fov_at(self, x: float, y: float) -> str | None:
        """The FOV whose outline holds (x, y) µm, else ``None``."""
        from matplotlib.path import Path

        for fov, outline in self.outlines.items():
            if Path(outline).contains_point((x, y)):
                return fov
        return None

    def _canvas(self) -> tuple[float, float]:
        """The canvas size, (width, height) in screen pixels."""
        size = self.viewer.canvas.size
        height, width = size  # napari's canvas size is (height, width)
        return float(width), float(height)

    def view_rect(self) -> tuple[float, float, float, float]:
        """What the view shows: (x0, y0, x1, y1) µm."""
        width, height = self._canvas()
        zoom = self.viewer.scene.camera.zoom
        cy, cx = self.viewer.scene.camera.center[-2:]
        half_w, half_h = width / zoom / 2, height / zoom / 2
        return cx - half_w, cy - half_h, cx + half_w, cy + half_h

    def update(self) -> None:
        """Hide the FOVs outside the view (with a margin), then report the view."""
        if self._closed:
            return
        x0, y0, x1, y1 = self.view_rect()
        mx, my = (x1 - x0) * MARGIN, (y1 - y0) * MARGIN
        for fov, layer in self.layers.items():
            outline = self.outlines[fov]
            inside = not (
                outline[:, 0].max() < x0 - mx
                or outline[:, 0].min() > x1 + mx
                or outline[:, 1].max() < y0 - my
                or outline[:, 1].min() > y1 + my
            )
            if layer.visible != inside:
                layer.visible = inside
        if self.on_view is not None:
            self.on_view((x0, y0, x1, y1))

    def _camera_moved(self, _event=None) -> None:
        self.update()

    @property
    def visible_fovs(self) -> list[str]:
        """The FOVs napari reads (those in or near the view)."""
        return [fov for fov, layer in self.layers.items() if layer.visible]
