"""What a browser sends an ndv viewer in Jupyter, for checking real viewers without one.

ndv's Jupyter viewers draw on a jupyter_rfb ``RemoteFrameBuffer``. In JupyterLab, its
browser side (jupyter_rfb's ``renderview.js``) sends each resize, mouse and key event as a
custom widget message, which the kernel hands to ``RemoteFrameBuffer._rfb_handle_msg``.
:func:`exercise` sends :data:`EVENTS` the same way, then draws a frame. A failure while
handling an event shows up in the canvas's output, as it would under the viewer in
JupyterLab.

Used in a Jupyter kernel (ndv's Jupyter viewers need one): by ``test_widgets_viewers.py``
and by the main repo's extraction notebook check.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def _pointer(
    kind: str, x: int, y: int, button: int, buttons: List[int], timestamp: int
) -> Dict[str, Any]:
    return {
        "type": kind,
        "x": x,
        "y": y,
        "button": button,
        "buttons": buttons,
        "modifiers": [],
        "ntouches": 0,
        "touches": {},
        "timestamp": timestamp,
    }


EVENTS: List[Dict[str, Any]] = [
    {
        "type": "resize",
        "width": 400,
        "height": 300,
        "pwidth": 400,
        "pheight": 300,
        "ratio": 1.0,
        "timestamp": 0,
    },
    _pointer("pointer_move", 50, 40, 0, [], 1),
    _pointer("pointer_down", 50, 40, 1, [1], 2),
    _pointer("pointer_move", 60, 45, 1, [1], 3),
    _pointer("pointer_up", 60, 45, 1, [], 4),
    {
        "type": "double_click",
        "x": 60,
        "y": 45,
        "button": 1,
        "buttons": [],
        "modifiers": [],
        "timestamp": 5,
    },
    {
        "type": "wheel",
        "x": 60,
        "y": 45,
        "dx": 0,
        "dy": 100,
        "buttons": [],
        "modifiers": [],
        "timestamp": 6,
    },
    {"type": "key_down", "key": "ArrowRight", "modifiers": [], "timestamp": 7},
    {"type": "key_up", "key": "ArrowRight", "modifiers": [], "timestamp": 8},
]
"""A resize, a click-drag, a double click, a scroll and a key press, as ``renderview.js``
builds them."""


def canvases(widget: Any) -> List[Any]:
    """The ``RemoteFrameBuffer`` canvases in an ipywidgets tree (``viewer.widget()``)."""
    from jupyter_rfb import RemoteFrameBuffer

    found = [widget] if isinstance(widget, RemoteFrameBuffer) else []
    for child in getattr(widget, "children", None) or ():
        found += canvases(child)
    return found


def exercise(canvas: Any) -> Dict[str, Any]:
    """Send ``canvas`` :data:`EVENTS` as the browser does, then draw a frame.

    Returns the canvas's class, the last line of each error shown under it, and the
    drawn frame's shape and pixel spread (0 for a blank frame; ``None`` if nothing was
    drawn).
    """
    for event in EVENTS:
        canvas._rfb_handle_msg(canvas, dict(event), [])
    frame = canvas.get_frame()
    if frame is None:  # rendercanvas draws and sends the frame itself
        frame = canvas._rfb_last_frame
    errors = [
        output["text"].strip().splitlines()[-1]
        for output in canvas._output_context.outputs
        if output.get("name") == "stderr"
    ]
    frame = None if frame is None else np.asarray(frame)
    return {
        "canvas": f"{type(canvas).__module__}.{type(canvas).__name__}",
        "errors": errors,
        "frame": None if frame is None else list(frame.shape),
        "spread": None if frame is None else float(frame[..., :3].std()),
    }
