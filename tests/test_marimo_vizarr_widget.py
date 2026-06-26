from __future__ import annotations

from pathlib import Path


def test_marimo_vizarr_widget_does_not_round_trip_view_state() -> None:
    notebook = Path("notebooks/pymmm_nd2_browser.py").read_text()

    assert "viewStateChange" not in notebook
    assert "model.save_changes();" not in notebook
