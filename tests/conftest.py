"""Shared fixtures."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterator, List

import pytest


class _Notebook:
    """What a widget showed: the widgets given to ``display()`` and the ndv viewers made."""

    def __init__(self) -> None:
        self.shown: List[Any] = []
        self.viewers: List[Any] = []

    @staticmethod
    def _walk(widget: Any) -> Iterator[Any]:
        yield widget
        for child in getattr(widget, "children", ()):
            yield from _Notebook._walk(child)

    def widgets(self) -> List[Any]:
        return [w for root in self.shown for w in self._walk(root)]

    def control(self, description: str) -> Any:
        """The one control (slider, dropdown, button) with this description."""
        found = [w for w in self.widgets() if getattr(w, "description", None) == description]
        assert len(found) == 1, f"{len(found)} controls named {description!r}"
        return found[0]

    def texts(self) -> List[str]:
        """Every HTML label's text, in display order."""
        import ipywidgets

        return [w.value for w in self.widgets() if isinstance(w, ipywidgets.HTML)]

    def text(self, fragment: str) -> str:
        """The one HTML label containing ``fragment``."""
        found = [t for t in self.texts() if fragment in t]
        assert len(found) == 1, f"{len(found)} labels contain {fragment!r}: {self.texts()}"
        return found[0]


@pytest.fixture
def notebook(monkeypatch):
    """The notebook UI without a browser. ndv's viewers record their data (the widget
    updates the array in place) and how often it changed; ``display()`` keeps what it
    is given, so a test can drive the controls. Needs the notebook extra."""
    ndv = pytest.importorskip("ndv")
    ipywidgets = pytest.importorskip("ipywidgets")
    import IPython.display
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")
    notebook = _Notebook()

    class ArrayViewer:
        def __init__(self, data: Any, **kwargs: Any) -> None:
            self.data, self.kwargs, self.renders = data, kwargs, 0
            self.data_wrapper = SimpleNamespace(data_changed=SimpleNamespace(emit=self._emit))
            notebook.viewers.append(self)

        def _emit(self) -> None:
            self.renders += 1

        def widget(self) -> Any:
            return ipywidgets.HTML("ndv viewer")

    def imshow(data: Any, **kwargs: Any) -> Any:
        viewer = SimpleNamespace(data=data, kwargs=kwargs)
        notebook.viewers.append(viewer)
        return viewer

    monkeypatch.setattr(ndv, "ArrayViewer", ArrayViewer)
    monkeypatch.setattr(ndv, "imshow", imshow)
    monkeypatch.setattr(IPython.display, "display", lambda *objs, **kw: notebook.shown.extend(objs))
    yield notebook
    plt.close("all")
