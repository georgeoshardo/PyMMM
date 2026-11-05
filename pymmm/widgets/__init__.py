"""The notebook UIs of the steps: ipywidgets controls, ndv viewers and matplotlib plots
(the notebook extra). The step classes open them: ``LaneDetector.interactive_detect_lanes``,
``TrenchDetector.interactive_detect_trenches``, ``Segmenter.interactive_segment`` and
``Segmenter.view``.

A notebook's own ndv viewers use :func:`imshow`: ``ndv.imshow`` on the canvas that works
in Jupyter (:data:`NDV_CANVAS`)."""

from pymmm.widgets._common import NDV_CANVAS, imshow

__all__ = ["NDV_CANVAS", "imshow"]
