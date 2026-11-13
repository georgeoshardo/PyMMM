"""Diagnostic figures (pymmm.plotting)."""

from __future__ import annotations

import threading

import matplotlib.pyplot as plt
import numpy as np
from pymmm.plotting import plot_drift_diagnostics, plot_fov_with_lanes


def test_figures_only_saved_stay_out_of_pyplot_and_save_from_any_thread(tmp_path):
    """The napari dock saves them from a background thread, where pyplot's Qt
    backend cannot make figures."""
    before = plt.get_fignums()
    tmats = np.tile(np.eye(3), (5, 1, 1))
    tmats[:, 0, 2] = np.arange(5)
    paths = [tmp_path / "drift.png", tmp_path / "lanes.png"]

    def save() -> None:
        plot_drift_diagnostics(tmats, "f0", save_path=str(paths[0]), show=False)
        plot_fov_with_lanes(
            np.zeros((20, 30)),
            [10],
            orientations=[1],
            save_path=str(paths[1]),
            show=False,
        )

    thread = threading.Thread(target=save)
    thread.start()
    thread.join()
    assert all(p.stat().st_size > 0 for p in paths)
    assert plt.get_fignums() == before
