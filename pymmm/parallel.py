"""Where batches of work run: the active Dask cluster, a process pool, or this process.

The steps split their work into batches (a few frames of one FOV, say) whose task reads
the ND2 itself, so the same task runs anywhere. A :class:`Runner` decides once where
they run and gives back futures as they finish::

    with Runner(workers=None) as runner:
        futures = {runner.submit(task, *args): start for start, args in batches}
        for future in runner.as_completed(futures):
            result = future.result()
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures import as_completed as _futures_completed
from typing import Any, Callable, Iterable, Iterator, Optional

# Fresh interpreters: pool workers then share nothing with the parent's Dask client,
# event loops or thread state.
SPAWN = multiprocessing.get_context("spawn")


def dask_client():
    """The active ``dask.distributed`` client, or None (also without distributed)."""
    try:
        from dask.distributed import get_client
        return get_client()
    except (ImportError, ValueError):
        return None


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

    @property
    def kind(self) -> str:
        """``"dask"``, ``"pool"`` or ``"serial"``."""
        if self.client is not None:
            return "dask"
        return "serial" if self.workers == 1 else "pool"

    def submit(self, fn: Callable[..., Any], /, *args: Any) -> Any:
        """Run ``fn(*args)``; a future of its result. The pool starts at the first task."""
        if self.client is not None:
            return self.client.submit(fn, *args)
        if self.workers == 1:
            future: Future = Future()
            try:
                future.set_result(fn(*args))
            except BaseException as error:  # raised again by future.result()
                future.set_exception(error)
            return future
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=SPAWN)
        return self._pool.submit(fn, *args)

    def as_completed(self, futures: Iterable[Any]) -> Iterator[Any]:
        """The futures, as they finish."""
        if self.client is not None:
            from dask.distributed import as_completed

            return iter(as_completed(list(futures)))
        return _futures_completed(list(futures))

    def close(self) -> None:
        """Wait for the pool's tasks and stop it (the Dask client is not ours to close)."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def __enter__(self) -> "Runner":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Runner({self.kind}, workers={self.workers})"
