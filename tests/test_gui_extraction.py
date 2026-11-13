"""The extraction dock's fields (pymmm.gui.extraction); the dock itself runs on real data
in golden/test_golden_dock.py."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.view


def test_ranges_read_as_typed_and_show_as_saved():
    from pymmm.gui.extraction import format_range, parse_range

    assert parse_range("290:850") == (290, 850)
    assert parse_range(" 0 - 20 ") == (0, 20)
    assert parse_range("250, 1100") == (250, 1100)
    assert parse_range("  ") is None
    for bad in ("12", "1:2:3", "a:b"):
        with pytest.raises(ValueError):
            parse_range(bad)
    assert format_range({"start": 20, "stop": 30, "step": None}) == "20:30"
    assert format_range([250, 1100]) == "250:1100"
    assert format_range({"start": None, "stop": None, "step": None}) == ""
    assert format_range(None) == ""


def test_fovs_by_name_or_index():
    from pymmm.gui.extraction import parse_fovs

    assert parse_fovs("A01, A02 5") == ["A01", "A02", 5]
    assert parse_fovs("") == []


def test_channel_colours_by_fluorophore():
    from pymmm.gui.extraction import channel_colormap

    assert channel_colormap("mCherry") == "magenta"
    assert channel_colormap("mVenus") == "yellow"
    assert channel_colormap("PC") == "gray"


def test_the_dock_is_a_napari_widget_of_pymmm():
    from importlib.resources import files

    import npe2

    manifest = npe2.PluginManifest.from_file(files("pymmm") / "napari.yaml")
    (widget,) = manifest.contributions.widgets
    (command,) = [c for c in manifest.contributions.commands if c.id == widget.command]
    assert command.python_name == "pymmm.gui.extraction:ExtractionDock"
