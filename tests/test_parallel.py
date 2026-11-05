import operator

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
