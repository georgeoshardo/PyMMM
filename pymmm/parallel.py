"""Where batches of work run: the active Dask cluster, a process pool, or this process.

The steps split their work into batches (a few frames of one FOV, say) whose task reads
the ND2 itself, so the same task runs anywhere. A :class:`Runner` decides once where
they run and gives back futures as they finish::

    with Runner(workers=None) as runner:
        reference = runner.share(image)          # sent once, read by every task
        futures = {runner.submit(task, reference, *args): start for start, args in batches}
        for future in runner.as_completed(futures):
            result = future.result()

and the task reads the shared array with ``shared(reference)``.

In a notebook, :func:`local_cluster` makes the Dask cluster. Its workers, and a pool's,
outlive JupyterLab's Interrupt button, which sends SIGINT to the kernel's whole process
group, workers included: interrupting a cell stops the cell, and the :class:`Runner`
then drops the batches it had queued (a batch already running finishes).
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import tempfile
import weakref
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor
from concurrent.futures import as_completed as _futures_completed
from concurrent.futures import wait as _futures_wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Set, Tuple

import numpy as np

# Fresh interpreters: pool workers then share nothing with the parent's Dask client,
# event loops or thread state.
SPAWN = multiprocessing.get_context("spawn")

IGNORE_INTERRUPT = "import signal\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\n"
"""A Dask worker preload (run in each worker process as it starts, restarts included):
ignore SIGINT. Without it, JupyterLab's Interrupt button kills every worker of the
notebook's cluster (it signals the kernel's process group) and the nanny does not start
them again."""


def _ignore_interrupt() -> None:
    """A pool worker's initializer: ignore SIGINT (see :data:`IGNORE_INTERRUPT`)."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def local_cluster(workers: int, **kwargs: Any) -> Tuple[Any, Any]:
    """A Dask cluster of ``workers`` processes on this machine, one thread each, and its
    client: ``cluster, client = local_cluster(16)``. Its workers ignore SIGINT
    (:data:`IGNORE_INTERRUPT`); ``client.close()`` and ``cluster.close()`` stop them.
    ``kwargs`` go to ``LocalCluster`` (``dashboard_address=None``, say)."""
    from dask.distributed import Client, LocalCluster

    cluster = LocalCluster(n_workers=workers, threads_per_worker=1, processes=True,
                           preload=[IGNORE_INTERRUPT], **kwargs)
    return cluster, Client(cluster)


def dask_client():
    """The active ``dask.distributed`` client, or None (also without distributed)."""
    try:
        from dask.distributed import get_client
        return get_client()
    except (ImportError, ValueError):
        return None


@dataclass(frozen=True)
class SharedFile:
    """An array a process pool's tasks read from a file (see :meth:`Runner.share`)."""

    path: str


_loaded: Dict[str, np.ndarray] = {}  # in a pool worker: the shared files it read lately
_KEEP = 4  # a few, for the FOVs in flight at once


def shared(value: Any) -> Any:
    """In a task: the array a :meth:`Runner.share` handle stands for (each pool worker
    reads a shared file once)."""
    if not isinstance(value, SharedFile):
        return value  # Dask has resolved its future; in this process it is the array
    if value.path not in _loaded:
        if len(_loaded) >= _KEEP:
            _loaded.clear()
        _loaded[value.path] = np.load(value.path)
    return _loaded[value.path]


class Runner:
    """Run tasks on the active Dask client, in a process pool, or in this process.

    Parameters
    ----------
    workers : int | None
        Process-pool size (``None``: one per core). ``1`` runs each task in this
        process, when it is submitted.
    dask : bool
        Use the active Dask client when there is one, whatever ``workers`` is.
    """

    def __init__(self, workers: Optional[int] = None, *, dask: bool = True) -> None:
        self.client = dask_client() if dask else None
        self.workers = workers
        self._pool: Optional[ProcessPoolExecutor] = None
        self._folder: Optional[tempfile.TemporaryDirectory] = None
        self._n_shared = 0
        # The futures the caller still holds (Dask's or the pool's): cancelled if the
        # run stops early.
        self._futures: "weakref.WeakSet[Any]" = weakref.WeakSet()

    @property
    def kind(self) -> str:
        """``"dask"``, ``"pool"`` or ``"serial"``."""
        if self.client is not None:
            return "dask"
        return "serial" if self.workers == 1 else "pool"

    def submit(self, fn: Callable[..., Any], /, *args: Any) -> Any:
        """Run ``fn(*args)``; a future of its result. The pool starts at the first task."""
        if self.client is not None:
            future = self.client.submit(fn, *args)
            self._futures.add(future)
            return future
        if self.workers == 1:
            future: Future = Future()
            try:
                future.set_result(fn(*args))
            except BaseException as error:  # raised again by future.result()
                future.set_exception(error)
            return future
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=SPAWN,
                                             initializer=_ignore_interrupt)
        future = self._pool.submit(fn, *args)
        self._futures.add(future)
        return future

    def share(self, array: np.ndarray) -> Any:
        """``array`` once for every task that takes it: scattered to every Dask worker,
        saved to a file each pool worker reads once, or the array itself in this
        process. Tasks read it with :func:`shared`."""
        if self.client is not None:
            future = self.client.scatter(array, broadcast=True, hash=False)
            self._futures.add(future)
            return future
        if self.workers == 1:
            return array
        if self._folder is None:
            self._folder = tempfile.TemporaryDirectory(prefix="pymmm_")
        path = Path(self._folder.name) / f"shared_{self._n_shared}.npy"
        self._n_shared += 1
        np.save(path, array)
        return SharedFile(str(path))

    def release(self, handle: Any) -> None:
        """Done with a shared array: its file goes (a Dask worker drops it with the
        last reference to its future)."""
        if isinstance(handle, SharedFile):
            os.unlink(handle.path)

    @property
    def slots(self) -> int:
        """How many tasks run at once now: the Dask workers' threads (workers may join or
        leave during a run), the pool's processes, or 1."""
        if self.client is not None:
            return max(1, sum(self.client.nthreads().values()))
        if self.workers == 1:
            return 1
        return self.workers or os.cpu_count() or 1

    def as_completed(self, futures: Iterable[Any]) -> Iterator[Any]:
        """The futures, as they finish."""
        if self.client is not None:
            from dask.distributed import as_completed

            return iter(as_completed(list(futures)))
        return _futures_completed(list(futures))

    def wait_first(self, futures: Iterable[Any]) -> Set[Any]:
        """Wait until at least one of the futures is done; the done ones."""
        futures = list(futures)
        if self.client is not None:
            from dask.distributed import wait

            return set(wait(futures, return_when="FIRST_COMPLETED").done)
        return set(_futures_wait(futures, return_when=FIRST_COMPLETED).done)

    def cancel(self) -> None:
        """Drop the tasks not started yet: the queued Dask tasks (and shared arrays) and
        the pool's. A task already running finishes; stopping it would mean killing its
        worker. (The pool's own ``cancel_futures`` is not enough: the pool goes with
        the Runner's reference, before its manager thread cancels anything.)"""
        futures = [future for future in self._futures if not future.done()]
        if self.client is not None:
            if futures:
                self.client.cancel(futures)
        else:
            for future in futures:
                future.cancel()  # a pending one never starts; a started one runs on
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def close(self) -> None:
        """Wait for the pool's tasks and stop it, and remove the shared files (the Dask
        client is not ours to close)."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        if self._folder is not None:
            self._folder.cleanup()
            self._folder = None

    def __enter__(self) -> "Runner":
        return self

    def __exit__(self, exc_type: Any, *exc: Any) -> None:
        """Close; first :meth:`cancel` if the block raised (an interrupted cell, a failed
        batch), so the work queued behind it does not run on."""
        if exc_type is not None:
            self.cancel()
        self.close()

    def __repr__(self) -> str:
        return f"Runner({self.kind}, workers={self.workers})"
