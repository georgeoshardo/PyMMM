"""``python -m pymmm.gui <experiment.toml or .nd2>``: the extraction dock in a new napari
viewer.

Light on purpose: registration's and extraction's process-pool workers start by
importing this module (the spawn start method imports the main module), so Qt and
napari are imported only in :func:`main`.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pymmm.gui",
        description=(
            "Register an ND2, detect its lanes and trenches, measure its shading, "
            "and preview and extract the trenches in napari: the extraction "
            "notebook's steps."
        ),
    )
    parser.add_argument(
        "experiment",
        type=Path,
        nargs="?",
        help="experiment file (TOML) naming the ND2, or the ND2; or choose one in the dock",
    )
    args = parser.parse_args(argv)

    import napari

    from pymmm.gui.extraction import ExtractionDock

    viewer = napari.Viewer(title="PyMMM extraction")
    dock = ExtractionDock(viewer, experiment_file=args.experiment)
    viewer.window.add_dock_widget(dock, name="PyMMM extraction", area="right")
    napari.run()


if __name__ == "__main__":
    main()
