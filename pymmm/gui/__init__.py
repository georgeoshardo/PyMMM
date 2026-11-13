"""PyMMM's napari front ends (the view extra). :mod:`pymmm.gui.extraction` is the
extraction dock: the extraction notebook's steps in napari, ``python -m pymmm.gui``.

This package imports nothing itself, so a process-pool worker that imports it (the
spawn start method imports the main module) does not start Qt.
"""
