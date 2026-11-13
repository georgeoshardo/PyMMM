"""Tracking: chaintrack run on an experiment group.

chaintrack follows a trench's cells through time as an ordered chain, from the masks
of a segmentation's hypotheses (no image). It is a program of its own, ``chain.py``,
kept outside PyMMM; :class:`Tracker` runs it on an experiment group, so tracking is
driven from a notebook like the steps before it::

    tracker = Tracker("experiment.zarr", "pc", chaintrack="chaintrack/chain.py")
    tracker.fit([0, 1, 2, 3, 4, 5, 6, 7], recipe="chaintrack/tests/data/recipe.toml")
    tracker.track()                                   # every trench not tracked yet

Two steps, as in chaintrack:

- :meth:`Tracker.fit` measures the experiment's own growth rate, division length and
  noise on a few trenches and records them in the group (``attrs.pymmm.fits``) with a
  frozen copy of the recipe, chaintrack's hand-set constants in µm and minutes.
- :meth:`Tracker.track` tracks trenches with that fit and writes each one's tracks
  beside its pixels: ``trench_NNNN/tracks/<run>.geff`` and the masks it chose,
  ``labels/tracks.<run>``. The run is recorded in ``attrs.pymmm.tracks``.

chaintrack tracks several trenches at once, in processes of its own, as many as fit
in the free memory. It does not use :class:`~pymmm.parallel.Runner` or a Dask cluster.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import zarr

from pymmm.store.experiment_group import has_segmentation, open_experiment, records, trench_ids
from pymmm.store.geff import has_tracks


def trench_ranges(trenches: Sequence[int]) -> List[str]:
    """Trench ids as chaintrack takes them, runs of neighbours joined: ``["0", "2-5", "7"]``."""
    ids = sorted({int(t) for t in trenches})
    out: List[str] = []
    k = 0
    while k < len(ids):
        end = k
        while end + 1 < len(ids) and ids[end + 1] == ids[end] + 1:
            end += 1
        out.append(str(ids[k]) if end == k else f"{ids[k]}-{ids[end]}")
        k = end + 1
    return out


class Tracker:
    """Track an experiment group's trenches with chaintrack.

    Parameters
    ----------
    store_path : str | Path
        Experiment group, already segmented.
    segmentation : str
        The segmentation whose masks are tracked: its base hypothesis, with the others
        as alternatives.
    chaintrack : str | Path
        chaintrack's ``chain.py``.
    run : str
        The tracking run's name: ``tracks/<run>.geff`` in each trench.
    fit : str | None
        The fit's name in the group; defaults to the segmentation's name.
    workers : int | None
        Trenches tracked at once. ``None``: chaintrack takes as many as fit in the free
        memory.
    python : str | Path | None
        The Python that runs chaintrack (it needs numba, scipy and PyMMM); defaults to
        this one.
    """

    def __init__(
        self,
        store_path: Union[str, Path],
        segmentation: str,
        chaintrack: Union[str, Path],
        *,
        run: str = "chain",
        fit: Optional[str] = None,
        workers: Optional[int] = None,
        python: Union[str, Path, None] = None,
    ) -> None:
        self.store_path = Path(store_path).resolve()
        self.segmentation = str(segmentation)
        self.chaintrack = Path(chaintrack).expanduser().resolve()
        self.run = str(run)
        self.fit_name = str(fit or segmentation)
        self.workers = None if workers is None else int(workers)
        self.python = str(python or sys.executable)
        root = self._open()
        if self.segmentation not in records(root, "segmentations"):
            raise ValueError(f"{self.store_path.name} has no segmentation {self.segmentation!r} "
                             f"(its segmentations: {sorted(records(root, 'segmentations')) or 'none'})")
        if not self.chaintrack.is_file():
            raise FileNotFoundError(f"no chaintrack at {self.chaintrack}")
        self.trench_ids = trench_ids(root)

    def _open(self) -> zarr.Group:
        """A fresh handle, so records chaintrack has written since are seen."""
        return open_experiment(self.store_path)

    # ------------------------------------------------------------------
    # What the group holds
    # ------------------------------------------------------------------

    @property
    def fit_record(self) -> Optional[Dict[str, Any]]:
        """The group's record of the fit (values in µm and minutes, the trenches, the
        frozen recipe), or None before :meth:`fit`."""
        return records(self._open(), "fits").get(self.fit_name)

    @property
    def record(self) -> Optional[Dict[str, Any]]:
        """The group's record of the run, or None before :meth:`track`."""
        return records(self._open(), "tracks").get(self.run)

    def segmented(self) -> List[int]:
        """The trenches whose segmentation is finished."""
        root = self._open()
        return [t for t in self.trench_ids if has_segmentation(root, t, self.segmentation)]

    def tracked(self) -> List[int]:
        """The trenches the run has tracked."""
        root = self._open()
        return [t for t in self.trench_ids if has_tracks(root, t, self.run)]

    def is_complete(self) -> bool:
        return len(self.tracked()) == len(self.trench_ids)

    # ------------------------------------------------------------------
    # Running chaintrack
    # ------------------------------------------------------------------

    def command(self, action: str, trenches: Sequence[int], *options: str) -> List[str]:
        """The chaintrack command line for ``action`` (``"fit"`` or ``"run"``)."""
        command = [self.python, str(self.chaintrack), action, *trench_ranges(trenches),
                   "--store", str(self.store_path), "--seg", self.segmentation, "--fit", self.fit_name,
                   *options]
        if self.workers is not None:
            command += ["--workers", str(self.workers)]
        return command

    def _run(self, command: Sequence[str]) -> None:
        """Run chaintrack, printing its lines as they come. Interrupting the cell stops
        it: the interrupt reaches chaintrack too, and this waits for it to stop."""
        process = subprocess.Popen(list(command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
        finally:
            process.stdout.close()
            code = process.wait()
        if code != 0:
            raise RuntimeError(f"chaintrack stopped with an error (exit status {code}); its last lines are above")

    def fit(self, trenches: Sequence[int], recipe: Union[str, Path]) -> Dict[str, float]:
        """Measure the experiment's fitted values on ``trenches`` (all their frames) and
        record them in the group with a frozen copy of ``recipe``; the values, in µm and
        minutes. chaintrack refuses to replace a fit that a run has used: give another
        ``fit`` name, and another ``run``, to track with other constants."""
        recipe = Path(recipe).expanduser().resolve()
        if not recipe.is_file():
            raise FileNotFoundError(f"no recipe at {recipe}")
        self._run(self.command("fit", trenches, "--recipe", str(recipe)))
        return dict(self.fit_record["values"])

    def track(self, trenches: Optional[Sequence[int]] = None, resume: bool = True) -> List[int]:
        """Track ``trenches`` (default: every segmented trench) and write their tracks
        into the group; the trenches the run holds afterwards.

        With ``resume``, trenches the run has already tracked are left as they are, so
        an interrupted run picks up where it stopped. ``resume=False`` tracks them
        again; that replaces the tracker's output only, never a curation.
        """
        if self.fit_record is None:
            raise RuntimeError(f"{self.store_path.name} has no fit {self.fit_name!r}: run fit(trenches, recipe) first")
        wanted = self.segmented() if trenches is None else [int(t) for t in trenches]
        done = set(self.tracked()) if resume else set()
        todo = [t for t in wanted if t not in done]
        if not todo:
            print(f"{self.run}: all {len(wanted)} trenches are tracked already")
            return self.tracked()
        if len(todo) < len(wanted):
            print(f"{self.run}: {len(wanted) - len(todo)} of {len(wanted)} trenches are tracked already", flush=True)
        self._run(self.command("run", todo, "--out", self.run))
        return self.tracked()

    def plot(self, trench_id: int, channel: str = "PC", minutes: Optional[float] = None,
             save_path: Optional[str] = None, show: bool = True) -> None:
        """A trench's kymograph with the run's tracks over it
        (:func:`pymmm.plotting.plot_track_kymograph`)."""
        from pymmm.plotting import plot_track_kymograph

        plot_track_kymograph(self.store_path, trench_id, self.run, channel=channel, minutes=minutes,
                             save_path=save_path, show=show)

    def __repr__(self) -> str:
        return (f"Tracker({self.store_path.name}, segmentation={self.segmentation!r}, run={self.run!r}, "
                f"{len(self.tracked())}/{len(self.trench_ids)} trenches tracked)")
