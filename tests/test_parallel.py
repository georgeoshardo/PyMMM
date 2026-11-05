import operator
from pathlib import Path

import pytest

from pymmm.parallel import Runner, dask_client


def _fail(message):
    raise ValueError(message)


def test_one_worker_runs_each_task_when_submitted():
    assert dask_client() is None
    with Runner(workers=1) as runner:
        assert runner.kind == "serial"
        futures = {runner.submit(operator.add, i, 10): i for i in range(3)}
        assert all(future.done() for future in futures)
        assert sorted(f.result() for f in runner.as_completed(futures)) == [10, 11, 12]
        failed = runner.submit(_fail, "no")
        with pytest.raises(ValueError, match="no"):
            failed.result()


def test_a_pool_starts_at_the_first_task_and_stops_on_close():
    runner = Runner(workers=2)
    assert runner.kind == "pool" and runner._pool is None
    futures = {runner.submit(operator.mul, i, 3): i for i in range(4)}
    assert {futures[f]: f.result() for f in runner.as_completed(futures)} == {
        0: 0, 1: 3, 2: 6, 3: 9}
    runner.close()
    assert runner._pool is None


def test_the_active_dask_client_is_used_unless_told_not_to():
    distributed = pytest.importorskip("dask.distributed")
    with distributed.Client(processes=False, n_workers=1, threads_per_worker=1,
                            dashboard_address=None) as client:
        assert Runner(workers=1).kind == "dask" and Runner().client is client
        assert Runner(workers=1, dask=False).kind == "serial"
        runner = Runner()
        futures = [runner.submit(operator.add, 1, i) for i in range(3)]
        assert sorted(f.result() for f in runner.as_completed(futures)) == [1, 2, 3]
    assert dask_client() is None


def _total(reference, extra):
    from pymmm.parallel import shared

    return float(shared(reference).sum()) + extra


def test_slots_and_wait_first_wherever_tasks_run():
    with Runner(workers=1) as runner:
        future = runner.submit(sum, [1, 2])
        assert runner.slots == 1 and runner.wait_first([future]) == {future}
    with Runner(workers=3, dask=False) as runner:
        futures = [runner.submit(sum, [i]) for i in range(3)]
        done = runner.wait_first(futures)
        assert runner.slots == 3 and done and done <= set(futures)
    distributed = pytest.importorskip("dask.distributed")
    with distributed.Client(processes=False, n_workers=2, threads_per_worker=1,
                            dashboard_address=None):
        with Runner(workers=1) as runner:
            futures = [runner.submit(sum, [i, 1]) for i in range(3)]
            done = runner.wait_first(futures)
            assert runner.slots == 2 and done and done <= set(futures)


def test_a_shared_array_reaches_every_task_wherever_it_runs(tmp_path):
    import numpy as np

    from pymmm.parallel import SharedFile

    array = np.arange(6.0).reshape(2, 3)
    with Runner(workers=1) as serial:
        handle = serial.share(array)
        assert handle is array
        assert serial.submit(_total, handle, 1.0).result() == 16.0
    with Runner(workers=2) as pool:
        handle = pool.share(array)
        assert isinstance(handle, SharedFile)
        futures = [pool.submit(_total, handle, float(k)) for k in range(4)]
        assert sorted(f.result() for f in pool.as_completed(futures)) == [15.0, 16.0, 17.0, 18.0]
        pool.release(handle)
        assert not Path(handle.path).exists()
        folder = Path(handle.path).parent
    assert not folder.exists()  # close() removes the shared files' folder


def test_a_shared_array_is_scattered_to_dask_workers():
    import numpy as np

    distributed = pytest.importorskip("dask.distributed")
    with distributed.Client(processes=False, n_workers=1, threads_per_worker=1,
                            dashboard_address=None):
        runner = Runner()
        handle = runner.share(np.ones(5))
        assert runner.submit(_total, handle, 0.5).result() == 5.5


# JupyterLab's Interrupt button sends SIGINT to the kernel's whole process group, its
# Dask and pool workers included. These scripts run in a session of their own (as a
# kernel does) and send it to their own group, so pytest is not interrupted.

INTERRUPTED = '''
import collections, json, os, signal, sys, time
from pathlib import Path
from pymmm.parallel import Runner, local_cluster

def batch(i, folder):
    time.sleep(0.5)
    Path(folder, f"batch_{i}").touch()
    return i

def outcome(future):
    if future.cancelled():
        return "cancelled"
    if not future.done():
        return "pending"
    return "failed" if future.exception() is not None else "finished"

if __name__ == "__main__":
    kind, folder = sys.argv[1], sys.argv[2]
    if kind == "dask":
        cluster, client = local_cluster(2, dashboard_address=None)
    interrupted = None
    try:
        with Runner(**({} if kind == "dask" else {"workers": 2, "dask": False})) as runner:
            futures = [runner.submit(batch, i, folder) for i in range(16)]
            for future in runner.as_completed(futures):
                future.result()
                interrupted = time.perf_counter()
                os.killpg(os.getpgrp(), signal.SIGINT)  # the Interrupt button
                time.sleep(10)  # KeyboardInterrupt here, as in a running cell
    except KeyboardInterrupt:
        stopped = time.perf_counter() - interrupted
    time.sleep(6)  # long enough for all 16 batches, had they gone on
    report = {"stopped": stopped, "ran": len(list(Path(folder).glob("batch_*"))),
              "batches": collections.Counter(outcome(f) for f in futures)}
    if kind == "dask":
        report["workers"] = len(client.scheduler_info()["workers"])
        try:
            report["task runs"] = client.submit(sum, [1, 2]).result(timeout=10) == 3
        except Exception as error:
            report["task runs"] = repr(error)
    print(json.dumps(report))
'''


def _interrupt(tmp_path: Path, kind: str) -> dict:
    import json
    import subprocess
    import sys

    script = tmp_path / "interrupted.py"
    script.write_text(INTERRUPTED)
    folder = tmp_path / "batches"
    folder.mkdir()
    done = subprocess.run([sys.executable, str(script), kind, str(folder)], text=True,
                          capture_output=True, timeout=120, start_new_session=True)
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_an_interrupt_keeps_the_dask_workers_and_drops_the_queued_batches(tmp_path):
    pytest.importorskip("dask.distributed")
    report = _interrupt(tmp_path, "dask")
    assert report["workers"] == 2 and report["task runs"] is True, report  # it survives
    assert report["ran"] < 8, report  # the batches still queued did not run
    assert "failed" not in report["batches"], report  # those running finished
    assert report["stopped"] < 2, report


def test_an_interrupt_stops_the_pool_without_waiting_for_its_queued_batches(tmp_path):
    report = _interrupt(tmp_path, "pool")
    assert report["ran"] < 8, report
    assert "failed" not in report["batches"], report  # those running finished
    assert report["stopped"] < 2, report  # not after all 16 batches (4 s)
