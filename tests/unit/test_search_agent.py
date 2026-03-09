"""Unit tests for SearchAgent, GridSearchAgent, and RandomSearchAgent."""
from __future__ import annotations

import json
import random
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from unittest.mock import MagicMock

from storegate.agent.search_agent import (
    SearchAgent,
    _RunningJob,
    _execute_task_in_subprocess,
    _shutdown_running_jobs,
)
from storegate.agent.grid_search_agent import GridSearchAgent
from storegate.agent.random_search_agent import RandomSearchAgent


# ---------------------------------------------------------------------------
# Picklable task stubs for parallel (spawn) integration tests.
# Must be defined at module level so child processes can import them.
# ---------------------------------------------------------------------------

class _SumTask:
    """Returns the sum of all HP values as 'score'."""
    def __init__(self) -> None:
        self._hps: dict = {}

    def set_hps(self, hps: dict) -> None:
        self._hps = hps

    def execute(self) -> dict:
        return {'score': sum(v for v in self._hps.values() if isinstance(v, (int, float)))}

    def finalize(self) -> None:
        pass


class _FailingTask:
    """Always raises RuntimeError on execute()."""
    def set_hps(self, hps: dict) -> None:
        pass

    def execute(self) -> dict:
        raise RuntimeError('intentional failure')

    def finalize(self) -> None:
        pass


class _SlowTask:
    """Sleeps long enough to reliably trigger job_timeout."""
    def set_hps(self, hps: dict) -> None:
        pass

    def execute(self) -> dict:
        import time
        time.sleep(0.8)
        return {}

    def finalize(self) -> None:
        pass


class _VerySlowTask:
    """Sleeps long enough that a real hard timeout should stop it early."""
    def set_hps(self, hps: dict) -> None:
        pass

    def execute(self) -> dict:
        time.sleep(5.0)
        return {}

    def finalize(self) -> None:
        pass


class _CrashTask:
    """Kills the worker process via os._exit, causing a non-zero exit in parent."""
    def set_hps(self, hps: dict) -> None:
        pass

    def execute(self) -> dict:
        import os
        os._exit(1)
        return {}

    def finalize(self) -> None:
        pass


class _HpsRecordTask:
    """Records the full hps dict (including injected cuda_id) as result."""
    def __init__(self) -> None:
        self._hps: dict = {}

    def set_hps(self, hps: dict) -> None:
        self._hps = dict(hps)

    def execute(self) -> dict:
        return dict(self._hps)

    def finalize(self) -> None:
        pass


class _LargeResultTask:
    """Returns a payload large enough to block if the parent never drains the pipe."""
    def __init__(self, payload_size: int = 2 * 1024 * 1024) -> None:
        self._payload_size = payload_size

    def set_hps(self, hps: dict) -> None:
        pass

    def execute(self) -> dict:
        return {'blob': 'x' * self._payload_size}

    def finalize(self) -> None:
        pass


class _OutputVarNamesTask:
    """Records output_var_names after SearchAgent-driven HP injection."""
    def __init__(self, output_var_names: object) -> None:
        self._hps: dict = {}
        self._output_var_names = output_var_names

    def set_hps(self, hps: dict) -> None:
        self._hps = dict(hps)
        if 'output_var_names' in hps:
            self._output_var_names = hps['output_var_names']

    def execute(self) -> dict:
        return {'output_var_names': self._output_var_names}

    def finalize(self) -> None:
        pass


class _FakePipe:
    def __init__(
        self,
        *,
        recv_result: dict[str, Any] | None = None,
        poll_result: bool = False,
        recv_error: Exception | None = None,
    ) -> None:
        self._recv_result = recv_result
        self._poll_result = poll_result
        self._recv_error = recv_error
        self.recv_calls = 0
        self.closed = False

    def recv(self) -> dict[str, Any]:
        self.recv_calls += 1
        if self._recv_error is not None:
            raise self._recv_error
        if self._recv_result is None:
            raise RuntimeError('recv_result must be set when recv() is expected.')
        return self._recv_result

    def poll(self) -> bool:
        return self._poll_result

    def close(self) -> None:
        self.closed = True


class _FakeProcess:
    def __init__(self, sentinel: int, exitcode: int = 0) -> None:
        self.sentinel = sentinel
        self.exitcode = exitcode
        self.started = False
        self.closed = False
        self.join_calls: list[float | None] = []

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return False

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)

    def close(self) -> None:
        self.closed = True


class _FakeContext:
    def __init__(
        self,
        *,
        parent_pipe: _FakePipe,
        child_pipe: _FakePipe,
        process: _FakeProcess,
    ) -> None:
        self._parent_pipe = parent_pipe
        self._child_pipe = child_pipe
        self._process = process

    def Pipe(self, duplex: bool = False) -> tuple[_FakePipe, _FakePipe]:
        assert duplex is False
        return self._parent_pipe, self._child_pipe

    def Process(self, target: object, args: tuple[object, ...]) -> _FakeProcess:
        return self._process


def _agent(**kwargs: object) -> SearchAgent:
    task = kwargs.pop('task', MagicMock(spec=['set_hps', 'execute', 'finalize']))
    defaults: dict = dict(task=task, hps=None, num_trials=None, cuda_ids=None)
    defaults.update(kwargs)
    return SearchAgent(**defaults)


class _SendPipe:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.closed = False

    def send(self, value: dict[str, Any]) -> None:
        self.sent.append(value)

    def close(self) -> None:
        self.closed = True


def _execute_pool_jobs_with_fake_worker(
    monkeypatch: pytest.MonkeyPatch,
    *,
    ready_objects: list[object],
    parent_pipe: _FakePipe,
    process: _FakeProcess,
) -> SearchAgent:
    child_pipe = _FakePipe()
    context = _FakeContext(
        parent_pipe=parent_pipe,
        child_pipe=child_pipe,
        process=process,
    )
    wait_results = iter([ready_objects])

    monkeypatch.setattr('storegate.agent.search_agent.mp.get_context', lambda _: context)
    monkeypatch.setattr(
        'storegate.agent.search_agent.wait',
        lambda objects, timeout=None: next(wait_results),
    )

    agent = SearchAgent(task=MagicMock(), hps=None, cuda_ids=[0])
    agent.execute_pool_jobs([[agent._task, {}, 0, None]])
    return agent


# ---------------------------------------------------------------------------
# Initialization — cuda_ids validation
# ---------------------------------------------------------------------------

def test_cuda_ids_as_list_is_accepted() -> None:
    agent = SearchAgent(task=MagicMock(), cuda_ids=[0, 1])
    assert agent._cuda_ids == [0, 1]


def test_cuda_ids_none_is_accepted() -> None:
    agent = _agent()
    assert agent._cuda_ids is None


@pytest.mark.parametrize(
    ('cuda_ids', 'error_type', 'message'),
    [
        (0, TypeError, 'must be a list'),
        ([], ValueError, 'must not be an empty list'),
        ([-1, 0], ValueError, 'non-negative integers'),
        ([0, '1'], TypeError, 'non-negative integers'),
        ((0, 1), TypeError, 'must be a list'),
    ],
)
def test_cuda_ids_invalid_values_raise(
    cuda_ids: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        SearchAgent(task=MagicMock(), cuda_ids=cuda_ids)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Initialization — num_trials validation
# ---------------------------------------------------------------------------

def test_num_trials_none_is_accepted() -> None:
    assert _agent()._num_trials is None


def test_num_trials_positive_int_is_accepted() -> None:
    agent = SearchAgent(task=MagicMock(), num_trials=3)
    assert agent._num_trials == 3


@pytest.mark.parametrize(
    ('num_trials', 'error_type'),
    [
        (True, TypeError),
        (False, TypeError),
        (0, ValueError),
        (-1, ValueError),
        (1.5, TypeError),
        ('3', TypeError),
    ],
)
def test_num_trials_invalid_values_raise(
    num_trials: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match='positive integer or None'):
        SearchAgent(task=MagicMock(), num_trials=num_trials)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Initialization — json_dump validation
# ---------------------------------------------------------------------------

def test_json_dump_none_accepted() -> None:
    assert _agent()._json_dump is None


def test_json_dump_valid_path_accepted(tmp_path: object) -> None:
    path = str(tmp_path / 'result.json')
    agent = SearchAgent(task=MagicMock(), json_dump=path)
    assert agent._json_dump is not None
    assert agent._json_dump.suffix == '.json'


@pytest.mark.parametrize(
    ('json_dump', 'message'),
    [
        ('result.txt', '.json'),
        ('/nonexistent_dir/result.json', 'does not exist'),
    ],
)
def test_json_dump_invalid_values_raise(
    tmp_path: object,
    json_dump: str,
    message: str,
) -> None:
    path = json_dump if json_dump.startswith('/') else str(tmp_path / json_dump)
    with pytest.raises(ValueError, match=message):
        SearchAgent(task=MagicMock(), json_dump=path)


# ---------------------------------------------------------------------------
# all_combinations
# ---------------------------------------------------------------------------

def test_all_combinations_none_returns_single_empty_dict() -> None:
    assert _agent().all_combinations(None) == [{}]


def test_all_combinations_single_param() -> None:
    result = _agent().all_combinations({'lr': [0.1, 0.01]})
    assert result == [{'lr': 0.1}, {'lr': 0.01}]


def test_all_combinations_two_params_cartesian_product() -> None:
    result = _agent().all_combinations({'a': [1, 2], 'b': ['x', 'y']})
    assert len(result) == 4
    assert {'a': 1, 'b': 'x'} in result
    assert {'a': 1, 'b': 'y'} in result
    assert {'a': 2, 'b': 'x'} in result
    assert {'a': 2, 'b': 'y'} in result


def test_all_combinations_count() -> None:
    result = _agent().all_combinations({'a': [1, 2, 3], 'b': [10, 20]})
    assert len(result) == 6


@pytest.mark.parametrize(
    ('hps', 'error_type'),
    [
        ({'lr': []}, ValueError),
        ({'lr': '0.1,0.01'}, TypeError),
        ({'lr': 0.1}, TypeError),
        ({'lr': (0.1, 0.01)}, TypeError),
    ],
)
def test_all_combinations_invalid_candidate_containers_raise(
    hps: dict[str, object],
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match='non-empty list'):
        SearchAgent(task=MagicMock(), hps=hps)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# execute_task — success path
# ---------------------------------------------------------------------------

def test_execute_task_returns_hps_and_job_id() -> None:
    task = MagicMock()
    task.execute.return_value = {'score': 0.9}
    agent = _agent(task=task)

    result = agent.execute_task(task, hps={'lr': 0.1}, job_id=0)

    assert result['hps'] == {'lr': 0.1}
    assert result['job_id'] == 0
    assert result['trial_id'] is None


def test_execute_task_calls_set_hps_execute_finalize_in_order() -> None:
    task = MagicMock()
    task.execute.return_value = {}
    call_order: list[str] = []
    task.set_hps.side_effect = lambda *a, **kw: call_order.append('set_hps')
    task.execute.side_effect = lambda *a, **kw: call_order.append('execute') or {}
    task.finalize.side_effect = lambda *a, **kw: call_order.append('finalize')

    _agent(suffix_job_id=False).execute_task(task, hps={}, job_id=0)

    assert call_order == ['set_hps', 'execute', 'finalize']


def test_execute_task_stores_result() -> None:
    task = MagicMock()
    task.execute.return_value = {'auc': 0.95}
    result = _agent(suffix_job_id=False).execute_task(task, hps={}, job_id=0)
    assert result['result'] == {'auc': 0.95}


def test_execute_task_with_trial_id() -> None:
    task = MagicMock()
    task.execute.return_value = {}
    result = _agent().execute_task(task, hps={}, job_id=3, trial_id=7)
    assert result['trial_id'] == 7


# ---------------------------------------------------------------------------
# execute_task — error handling
# ---------------------------------------------------------------------------

def test_execute_task_catches_exception_and_stores_error() -> None:
    task = MagicMock()
    task.execute.side_effect = RuntimeError('something failed')

    result = _agent(suffix_job_id=False).execute_task(task, hps={}, job_id=0)

    assert 'error' in result
    assert 'RuntimeError' in result['error']
    assert 'something failed' in result['error']


def test_execute_task_on_exception_has_no_result_key() -> None:
    task = MagicMock()
    task.execute.side_effect = ValueError('bad value')

    result = _agent().execute_task(task, hps={}, job_id=0)
    assert 'result' not in result


def test_execute_task_finalize_always_called_on_exception() -> None:
    task = MagicMock()
    task.execute.side_effect = RuntimeError('fail')

    _agent().execute_task(task, hps={}, job_id=0)
    task.finalize.assert_called_once()


def test_execute_task_finalize_failure_is_recorded_when_execute_succeeds() -> None:
    task = MagicMock()
    task.execute.return_value = {'score': 0.9}
    task.finalize.side_effect = RuntimeError('finalize fail')

    result = _agent(suffix_job_id=False).execute_task(task, hps={}, job_id=0)

    assert result['result'] == {'score': 0.9}
    assert result['error'] == 'RuntimeError: finalize fail'
    assert result['finalize_error'] == 'RuntimeError: finalize fail'


def test_execute_task_finalize_failure_does_not_override_execute_error() -> None:
    task = MagicMock()
    task.execute.side_effect = RuntimeError('execute fail')
    task.finalize.side_effect = RuntimeError('finalize fail')

    result = _agent(suffix_job_id=False).execute_task(task, hps={}, job_id=0)
    assert result['error'] == 'RuntimeError: execute fail'
    assert result['finalize_error'] == 'RuntimeError: finalize fail'
    assert 'result' not in result


def test_execute_task_in_subprocess_finalize_failure_is_recorded() -> None:
    class _FinalizeFailTask:
        def set_hps(self, hps: dict[str, Any]) -> None:
            pass

        def execute(self) -> dict[str, Any]:
            return {'score': 1.0}

        def finalize(self) -> None:
            raise RuntimeError('finalize fail')

    result_pipe = _SendPipe()
    _execute_task_in_subprocess(
        _FinalizeFailTask(), {}, 0, None, False, result_pipe
    )

    assert result_pipe.closed is True
    assert result_pipe.sent[0]['result'] == {'score': 1.0}
    assert result_pipe.sent[0]['error'] == 'RuntimeError: finalize fail'
    assert result_pipe.sent[0]['finalize_error'] == 'RuntimeError: finalize fail'


# ---------------------------------------------------------------------------
# finalize — sorting
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ('history', 'expected'),
    [
        (
            [
                {'job_id': 2, 'trial_id': None},
                {'job_id': 0, 'trial_id': None},
                {'job_id': 1, 'trial_id': None},
            ],
            [(0, None), (1, None), (2, None)],
        ),
        (
            [
                {'job_id': 0, 'trial_id': 2},
                {'job_id': 0, 'trial_id': 0},
                {'job_id': 0, 'trial_id': 1},
            ],
            [(0, 0), (0, 1), (0, 2)],
        ),
        (
            [
                {'job_id': 1, 'trial_id': 0},
                {'job_id': 0, 'trial_id': 1},
                {'job_id': 0, 'trial_id': 0},
            ],
            [(0, 0), (0, 1), (1, 0)],
        ),
    ],
)
def test_finalize_sorts_history(
    history: list[dict[str, int | None]],
    expected: list[tuple[int, int | None]],
) -> None:
    agent = _agent()
    agent._history = history
    agent.finalize()
    assert [(r['job_id'], r['trial_id']) for r in agent._history] == expected


# ---------------------------------------------------------------------------
# finalize — json dump
# ---------------------------------------------------------------------------

def test_finalize_writes_json_file(tmp_path: object) -> None:
    path = tmp_path / 'results.json'
    agent = SearchAgent(task=MagicMock(), json_dump=str(path))
    agent._history = [{'job_id': 0, 'trial_id': None, 'result': {'score': 0.9}}]
    agent.finalize()

    assert path.exists()
    data = json.loads(path.read_text(encoding='utf-8'))
    assert data[0]['job_id'] == 0
    assert data[0]['result']['score'] == 0.9


def test_finalize_without_json_dump_does_not_raise() -> None:
    agent = _agent()
    agent._history = [{'job_id': 0, 'trial_id': None}]
    agent.finalize()  # should not raise


def test_finalize_json_is_valid_and_indented(tmp_path: object) -> None:
    path = tmp_path / 'out.json'
    agent = SearchAgent(task=MagicMock(), json_dump=str(path))
    agent._history = [{'job_id': 0, 'trial_id': None}]
    agent.finalize()

    raw = path.read_text(encoding='utf-8')
    json.loads(raw)  # must be valid JSON
    assert '\n' in raw  # indent=2 means multi-line


def test_finalize_json_dump_normalizes_numpy_paths_classes_and_nan(tmp_path: object) -> None:
    path = tmp_path / 'out.json'
    artifact_path = Path(tmp_path / 'artifact.bin')
    agent = SearchAgent(task=MagicMock(), json_dump=str(path))
    agent._history = [{
        'job_id': np.int64(0),
        'trial_id': None,
        'hps': {
            'lr': np.float32(0.5),
            'model': dict,
            'artifact': artifact_path,
        },
        'result': {
            'scores': np.array([1.0, np.nan], dtype=np.float32),
            'loss': np.float64(1.25),
        },
    }]

    agent.finalize()

    data = json.loads(path.read_text(encoding='utf-8'))
    assert data == [{
        'job_id': 0,
        'trial_id': None,
        'hps': {
            'lr': 0.5,
            'model': 'builtins.dict',
            'artifact': str(artifact_path),
        },
        'result': {
            'scores': [1.0, 'nan'],
            'loss': 1.25,
        },
    }]


def test_finalize_json_dump_normalizes_torch_tensors_when_available(
    tmp_path: object,
) -> None:
    torch = pytest.importorskip('torch')
    path = tmp_path / 'out.json'
    agent = SearchAgent(task=MagicMock(), json_dump=str(path))
    agent._history = [{
        'job_id': 0,
        'trial_id': None,
        'result': {'pred': torch.tensor([[1.0, 2.0]])},
    }]

    agent.finalize()

    data = json.loads(path.read_text(encoding='utf-8'))
    assert data[0]['result']['pred'] == [[1.0, 2.0]]


# ---------------------------------------------------------------------------
# GridSearchAgent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ('hps', 'expected_len'),
    [
        ({'a': [1, 2], 'b': [10, 20]}, 4),
        ({'lr': [1e-3, 1e-4, 1e-5]}, 3),
    ],
)
def test_grid_search_agent_counts_combinations(
    hps: dict[str, list[object]],
    expected_len: int,
) -> None:
    agent = GridSearchAgent(task=MagicMock(), hps=hps)
    assert len(agent._hps) == expected_len


# ---------------------------------------------------------------------------
# RandomSearchAgent
# ---------------------------------------------------------------------------

def test_random_search_agent_replace_default_is_false() -> None:
    agent = RandomSearchAgent(num_iter=2, seed=0, task=MagicMock(), hps={'a': [1, 2]})
    assert agent._replace is False


def test_random_search_agent_samples_n_iter() -> None:
    agent = RandomSearchAgent(
        num_iter=5, seed=42, task=MagicMock(),
        hps={'a': [1, 2, 3], 'b': [10, 20, 30]},
    )
    assert len(agent._hps) == 5


def test_random_search_agent_reproducible_with_same_seed() -> None:
    kwargs: dict = dict(task=MagicMock(), hps={'a': [1, 2, 3], 'b': [10, 20]}, num_iter=6)
    assert RandomSearchAgent(seed=0, **kwargs)._hps == RandomSearchAgent(seed=0, **kwargs)._hps


def test_random_search_agent_different_seeds_produce_different_samples() -> None:
    kwargs: dict = dict(task=MagicMock(), hps={'a': list(range(10))}, num_iter=8)
    hps0 = RandomSearchAgent(seed=0, **kwargs)._hps
    hps1 = RandomSearchAgent(seed=1, **kwargs)._hps
    assert hps0 != hps1


@pytest.mark.parametrize('replace', [False, True])
def test_random_search_agent_matches_materialized_sampling(
    replace: bool,
) -> None:
    hps = {'a': [1, 2, 3], 'b': [10, 20]}
    combinations = _agent().all_combinations(hps)
    rng = random.Random(42)
    if replace:
        expected = [dict(rng.choice(combinations)) for _ in range(5)]
    else:
        expected = [dict(combo) for combo in rng.sample(combinations, k=5)]

    agent = RandomSearchAgent(
        num_iter=5,
        seed=42,
        replace=replace,
        task=MagicMock(),
        hps=hps,
    )
    assert agent._hps == expected


@pytest.mark.parametrize('replace', [False, True])
def test_random_search_agent_does_not_call_search_agent_all_combinations(
    monkeypatch: pytest.MonkeyPatch,
    replace: bool,
) -> None:
    def _raise(*args: object, **kwargs: object) -> list[dict[str, object]]:
        raise AssertionError('SearchAgent.all_combinations should not be used')

    monkeypatch.setattr(SearchAgent, 'all_combinations', _raise)
    agent = RandomSearchAgent(
        num_iter=3,
        seed=0,
        replace=replace,
        task=MagicMock(),
        hps={'a': list(range(100)), 'b': list(range(100))},
    )
    assert len(agent._hps) == 3


def test_random_search_agent_none_hps_returns_single_empty_dict() -> None:
    agent = RandomSearchAgent(num_iter=5, seed=0, task=MagicMock(), hps=None)
    assert agent._hps == [{}]


def test_random_search_agent_replace_false_samples_without_duplicates() -> None:
    agent = RandomSearchAgent(
        num_iter=4,
        seed=7,
        task=MagicMock(),
        hps={'a': [1, 2], 'b': [10, 20]},
    )
    assert len(agent._hps) == 4
    assert len({tuple(sorted(combo.items())) for combo in agent._hps}) == 4


def test_random_search_agent_replace_false_num_iter_exceeds_search_space_raises() -> None:
    with pytest.raises(ValueError, match='unique combinations'):
        RandomSearchAgent(
            num_iter=5,
            seed=0,
            task=MagicMock(),
            hps={'a': [1, 2], 'b': [10, 20]},
        )


def test_random_search_agent_replace_true_allows_duplicate_samples() -> None:
    agent = RandomSearchAgent(
        num_iter=5,
        seed=0,
        replace=True,
        task=MagicMock(),
        hps={'a': [1, 2], 'b': [10, 20]},
    )
    assert len(agent._hps) == 5
    assert len({tuple(sorted(combo.items())) for combo in agent._hps}) < 5


def test_random_search_agent_each_sample_uses_valid_values() -> None:
    valid = {'lr': [1e-3, 1e-4], 'bs': [32, 64, 128]}
    agent = RandomSearchAgent(
        num_iter=20,
        seed=7,
        replace=True,
        task=MagicMock(),
        hps=valid,
    )
    for combo in agent._hps:
        assert combo['lr'] in valid['lr']
        assert combo['bs'] in valid['bs']


@pytest.mark.parametrize(
    ('kwargs', 'error_type', 'message'),
    [
        ({'num_iter': 0}, ValueError, 'positive integer'),
        ({'num_iter': 1.5}, TypeError, 'positive integer'),
        ({'replace': 'no'}, TypeError, 'replace must be a bool'),
    ],
)
def test_random_search_agent_invalid_init_args_raise(
    kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    base_kwargs = dict(num_iter=1, seed=0, replace=False, task=MagicMock(), hps={'a': [1]})
    base_kwargs.update(kwargs)
    with pytest.raises(error_type, match=message):
        RandomSearchAgent(**base_kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# execute / execute_pool_jobs
# Default ``cuda_ids=None`` runs serially in-process.
# Parallel tests use explicit ``cuda_ids`` so child processes can pickle tasks.
# ---------------------------------------------------------------------------

def test_execute_single_job_populates_history() -> None:
    """execute() with one HP combo fills _history with one entry."""
    agent = SearchAgent(task=_SumTask(), hps=None)
    agent.execute()
    assert len(agent._history) == 1


def test_execute_all_combos_collected() -> None:
    """execute() collects one result per HP combination."""
    agent = SearchAgent(task=_SumTask(), hps={'a': [1, 2], 'b': [10, 20]})
    agent.execute()
    assert len(agent._history) == 4


def test_execute_result_values_are_correct() -> None:
    """Each result['result'] reflects the HP values passed to that job."""
    agent = SearchAgent(task=_SumTask(), hps={'a': [3, 7]})
    agent.execute()
    scores = {r['result']['score'] for r in agent._history}
    assert scores == {3, 7}


def test_execute_job_ids_cover_full_range() -> None:
    """Every job_id from 0 to n-1 appears exactly once in history."""
    n = 3
    agent = SearchAgent(task=_SumTask(), hps={'a': [1, 2, 3]})
    agent.execute()
    assert {r['job_id'] for r in agent._history} == set(range(n))


def test_execute_with_num_trials_generates_correct_count() -> None:
    """2 HP combos × 3 trials = 6 history entries."""
    agent = SearchAgent(task=_SumTask(), hps={'a': [1, 2]}, num_trials=3)
    agent.execute()
    assert len(agent._history) == 6


def test_execute_with_num_trials_all_trial_ids_present() -> None:
    """trial_id 0, 1, 2 appear for each job_id."""
    agent = SearchAgent(task=_SumTask(), hps={'a': [1]}, num_trials=3)
    agent.execute()
    trial_ids = {r['trial_id'] for r in agent._history}
    assert trial_ids == {0, 1, 2}


def test_execute_failing_task_error_captured_not_raised() -> None:
    """A task that raises must store 'error' in the result without crashing execute()."""
    agent = SearchAgent(task=_FailingTask(), hps=None)
    agent.execute()  # must not raise
    assert len(agent._history) == 1
    assert 'error' in agent._history[0]
    assert 'RuntimeError' in agent._history[0]['error']


def test_execute_failing_task_has_no_result_key() -> None:
    agent = SearchAgent(task=_FailingTask(), hps=None)
    agent.execute()
    assert 'result' not in agent._history[0]


def test_execute_without_cuda_ids_does_not_create_process_pool() -> None:
    class _LocalTask:
        def __init__(self) -> None:
            self._hps: dict = {}
            self._non_picklable = lambda value: value

        def set_hps(self, hps: dict) -> None:
            self._hps = dict(hps)

        def execute(self) -> dict:
            return {'score': self._non_picklable(self._hps['a'])}

        def finalize(self) -> None:
            pass

    agent = SearchAgent(task=_LocalTask(), hps={'a': [3]})
    agent.execute()

    assert agent._history[0]['result'] == {'score': 3}


def test_execute_cuda_ids_injected_into_hps() -> None:
    """cuda_id is added to hps before the task sees them."""
    agent = SearchAgent(task=_HpsRecordTask(), hps=None, cuda_ids=[5])
    agent.execute()
    assert agent._history[0]['result']['cuda_id'] == 5


def test_execute_cuda_ids_round_robin_two_workers() -> None:
    """With 2 workers and 4 jobs, cuda ids are assigned round-robin: 0,1,0,1."""
    agent = SearchAgent(
        task=_HpsRecordTask(),
        hps={'a': [1, 2, 3, 4]},
        cuda_ids=[0, 1],
    )
    agent.execute()
    agent.finalize()  # sort by job_id for deterministic assertion
    assigned = [r['result']['cuda_id'] for r in agent._history]
    assert assigned == [0, 1, 0, 1]


def test_execute_with_cuda_ids_dispatches_to_pool_jobs() -> None:
    agent = SearchAgent(task=_HpsRecordTask(), hps=None, cuda_ids=[5])
    agent.execute_pool_jobs = MagicMock()
    agent.execute_serial_jobs = MagicMock()

    agent.execute()

    agent.execute_pool_jobs.assert_called_once_with([[agent._task, {}, 0, None]])
    agent.execute_serial_jobs.assert_not_called()


def test_suffix_job_id_default_is_true() -> None:
    agent = SearchAgent(task=MagicMock())
    assert agent._suffix_job_id is True


@pytest.mark.parametrize(
    ('output_var_names', 'hps', 'expected'),
    [
        ('pred', None, ['pred_job0_trial0']),
        (['pred', 'score'], None, [['pred_job0_trial0', 'score_job0_trial0']]),
        (
            {'train': None, 'valid': ['val_pred'], 'test': 'pred'},
            None,
            [{'train': None, 'valid': ['val_pred_job0_trial0'], 'test': 'pred_job0_trial0'}],
        ),
        ('pred', {'a': [1, 2]}, ['pred_job0_trial0', 'pred_job1_trial0']),
        ('pred', {'output_var_names': ['custom']}, ['custom_job0_trial0']),
        (
            'pred',
            {'output_var_names': [['custom', 'score']]},
            [['custom_job0_trial0', 'score_job0_trial0']],
        ),
    ],
)
def test_suffix_job_id_updates_output_var_names(
    output_var_names: object,
    hps: dict[str, list[object]] | None,
    expected: list[object],
) -> None:
    agent = SearchAgent(
        task=_OutputVarNamesTask(output_var_names),
        hps=hps,
        suffix_job_id=True,
    )
    agent.execute()
    agent.finalize()
    assert [r['result']['output_var_names'] for r in agent._history] == expected


def test_suffix_job_id_uses_trial_id_when_num_trials_enabled() -> None:
    agent = SearchAgent(
        task=_OutputVarNamesTask('pred'),
        hps=None,
        num_trials=2,
        suffix_job_id=True,
    )

    agent.execute()
    agent.finalize()

    assert [r['result']['output_var_names'] for r in agent._history] == [
        'pred_job0_trial0',
        'pred_job0_trial1',
    ]


def test_execute_then_finalize_writes_json(tmp_path) -> None:
    """Full pipeline: execute() + finalize() produces a valid, sorted JSON file."""
    path = tmp_path / 'results.json'
    agent = SearchAgent(
        task=_SumTask(),
        hps={'v': [10, 20, 30]},
        json_dump=str(path),
    )
    agent.execute()
    agent.finalize()

    assert path.exists()
    data = json.loads(path.read_text(encoding='utf-8'))
    assert len(data) == 3
    job_ids = [entry['job_id'] for entry in data]
    assert job_ids == sorted(job_ids)


def test_execute_history_is_empty_before_execute() -> None:
    agent = SearchAgent(task=_SumTask(), hps={'a': [1, 2]})
    assert agent._history == []


def test_execute_multiple_calls_reset_history() -> None:
    """Calling execute() twice resets _history; only the latest run is kept."""
    agent = SearchAgent(task=_SumTask(), hps={'a': [1]})
    agent.execute()
    agent.execute()
    assert len(agent._history) == 1


# ---------------------------------------------------------------------------
# job_timeout — initialization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ('job_timeout', 'expected'),
    [
        (None, None),
        (5.0, 5.0),
    ],
)
def test_job_timeout_is_stored(job_timeout: float | None, expected: float | None) -> None:
    agent = SearchAgent(task=MagicMock(), job_timeout=job_timeout)
    assert agent._job_timeout == expected


# ---------------------------------------------------------------------------
# job_timeout — timeout fires
# ---------------------------------------------------------------------------

def test_job_timeout_cancels_slow_jobs() -> None:
    """A job that never completes is cancelled and recorded as a timeout error."""
    agent = SearchAgent(task=_SlowTask(), hps={'a': [1, 2]}, job_timeout=0.5)
    agent.execute()
    assert len(agent._history) == 2
    for entry in agent._history:
        assert 'error' in entry
        assert 'TimeoutError' in entry['error']


def test_job_timeout_error_contains_duration() -> None:
    """The error message includes the configured timeout value."""
    agent = SearchAgent(task=_SlowTask(), hps=None, job_timeout=0.5)
    agent.execute()
    assert '0.5' in agent._history[0]['error']


def test_job_timeout_error_entry_has_hps_and_job_id() -> None:
    """Timed-out entries preserve hps and job_id for traceability."""
    agent = SearchAgent(task=_SlowTask(), hps={'lr': [1e-3]}, job_timeout=0.5)
    agent.execute()
    entry = agent._history[0]
    assert 'hps' in entry
    assert 'job_id' in entry
    assert entry['hps'] == {'lr': 1e-3}


def test_job_timeout_returns_promptly() -> None:
    """Timed-out jobs should be terminated instead of running to natural completion."""
    start = time.monotonic()
    agent = SearchAgent(task=_VerySlowTask(), hps=None, job_timeout=0.2)
    agent.execute()
    elapsed = time.monotonic() - start

    assert elapsed < 2.0
    assert 'TimeoutError' in agent._history[0]['error']


def test_job_timeout_none_does_not_interfere_with_fast_jobs() -> None:
    """job_timeout=None (default) lets fast jobs complete normally."""
    agent = SearchAgent(task=_SumTask(), hps={'a': [1, 2]}, job_timeout=None)
    agent.execute()
    assert all('result' in r for r in agent._history)


# ---------------------------------------------------------------------------
# _suffix_output_var_names — direct unit tests (no subprocess)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ('output_var_names', 'job_id', 'trial_id', 'expected'),
    [
        (None, 0, None, None),
        ('pred', 3, None, 'pred_job3_trial0'),
        ('pred', 3, 7, 'pred_job3_trial7'),
        (['a', 'b'], 1, None, ['a_job1_trial0', 'b_job1_trial0']),
        (
            {'train': None, 'valid': ['v'], 'test': 'pred'},
            2,
            4,
            {'train': None, 'valid': ['v_job2_trial4'], 'test': 'pred_job2_trial4'},
        ),
    ],
)
def test_suffix_output_var_names_supported_values(
    output_var_names: object,
    job_id: int,
    trial_id: int | None,
    expected: object,
) -> None:
    assert SearchAgent._suffix_output_var_names(output_var_names, job_id, trial_id) == expected


def test_suffix_output_var_names_unsupported_type_raises() -> None:
    with pytest.raises(TypeError, match='got int'):
        SearchAgent._suffix_output_var_names(42, 0)


# ---------------------------------------------------------------------------
# execute_task with suffix_job_id — direct unit tests (no subprocess)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ('hps', 'job_id', 'trial_id', 'expected'),
    [
        ({}, 5, None, 'pred_job5_trial0'),
        ({'output_var_names': 'custom'}, 0, None, 'custom_job0_trial0'),
        (
            {'output_var_names': ['custom', 'score']},
            2,
            9,
            ['custom_job2_trial9', 'score_job2_trial9'],
        ),
    ],
)
def test_execute_task_suffix_job_id_applies_expected_value(
    hps: dict[str, object],
    job_id: int,
    trial_id: int | None,
    expected: object,
) -> None:
    task = MagicMock()
    task._output_var_names = 'pred'
    task.execute.return_value = {}
    agent = SearchAgent(task=task, suffix_job_id=True)

    result = agent.execute_task(task, hps=hps, job_id=job_id, trial_id=trial_id)

    passed_hps = task.set_hps.call_args[0][0]
    assert passed_hps['output_var_names'] == expected
    assert result['hps']['output_var_names'] == expected


def test_execute_task_suffix_job_id_false_does_not_modify() -> None:
    task = MagicMock()
    task._output_var_names = 'pred'
    task.execute.return_value = {}
    agent = SearchAgent(task=task, suffix_job_id=False)

    agent.execute_task(task, hps={'lr': 0.1}, job_id=0)

    passed_hps = task.set_hps.call_args[0][0]
    assert 'output_var_names' not in passed_hps


def test_execute_task_suffix_job_id_no_attr_skips() -> None:
    """Task without _output_var_names is not modified even with suffix_job_id=True."""
    task = MagicMock(spec=['set_hps', 'execute', 'finalize'])
    task.execute.return_value = {}
    agent = SearchAgent(task=task, suffix_job_id=True)

    agent.execute_task(task, hps={'lr': 0.1}, job_id=0)

    passed_hps = task.set_hps.call_args[0][0]
    assert 'output_var_names' not in passed_hps


# ---------------------------------------------------------------------------
# _shutdown_running_jobs
# ---------------------------------------------------------------------------

def test_shutdown_running_jobs_terminates_alive_and_skips_dead() -> None:
    """Alive processes are terminated; already-dead ones are skipped."""
    proc_alive = MagicMock()
    proc_alive.is_alive.side_effect = [True, False]  # alive → terminate, dead after join
    pipe_alive = MagicMock()

    proc_dead = MagicMock()
    proc_dead.is_alive.return_value = False
    pipe_dead = MagicMock()

    jobs = {
        1: _RunningJob(proc_alive, pipe_alive, {}, 0, None),
        2: _RunningJob(proc_dead, pipe_dead, {}, 1, None),
    }

    _shutdown_running_jobs(jobs)

    pipe_alive.close.assert_called_once()
    pipe_dead.close.assert_called_once()
    proc_alive.terminate.assert_called_once()
    proc_dead.terminate.assert_not_called()
    proc_alive.join.assert_called_once()
    proc_alive.kill.assert_not_called()
    proc_alive.close.assert_called_once()
    proc_dead.close.assert_called_once()


def test_shutdown_running_jobs_kills_stubborn_process() -> None:
    """Processes that survive terminate() are forcefully killed."""
    proc = MagicMock()
    proc.is_alive.side_effect = [True, True]  # alive → terminate, still alive after join → kill
    pipe = MagicMock()

    jobs = {1: _RunningJob(proc, pipe, {}, 0, None)}

    _shutdown_running_jobs(jobs, kill_after=0.01)

    pipe.close.assert_called_once()
    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()
    assert proc.join.call_count == 2
    proc.close.assert_called_once()


def test_shutdown_running_jobs_empty_dict() -> None:
    _shutdown_running_jobs({})  # should not raise


# ---------------------------------------------------------------------------
# _serial_job_timeout — previous timer restoration (line 212)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not hasattr(signal, 'SIGALRM'),
    reason='SIGALRM not available on this platform',
)
def test_serial_job_timeout_restores_previous_timer() -> None:
    """When a pre-existing ITIMER_REAL is active, it is restored after the context exits."""
    agent = SearchAgent(task=MagicMock(), job_timeout=10.0)

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.setitimer(signal.ITIMER_REAL, 50.0)
    try:
        with agent._serial_job_timeout(enabled=True):
            pass
        remaining, _ = signal.getitimer(signal.ITIMER_REAL)
        assert remaining > 0, 'Previous timer should have been restored'
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


# ---------------------------------------------------------------------------
# execute_pool_jobs without cuda_ids (line 250)
# ---------------------------------------------------------------------------

def test_execute_pool_jobs_without_cuda_ids_raises() -> None:
    agent = SearchAgent(task=MagicMock(), cuda_ids=None)
    with pytest.raises(RuntimeError, match='requires cuda_ids'):
        agent.execute_pool_jobs([])


# ---------------------------------------------------------------------------
# execute_pool_jobs — ready pipe/sentinel de-duplication
# ---------------------------------------------------------------------------

def test_execute_pool_jobs_pipe_and_sentinel_ready_once_returns_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = {
        'hps': {'cuda_id': 0},
        'job_id': 0,
        'trial_id': None,
        'result': {'score': 1},
    }
    parent_pipe = _FakePipe(recv_result=result, poll_result=True)
    process = _FakeProcess(sentinel=101, exitcode=0)

    agent = _execute_pool_jobs_with_fake_worker(
        monkeypatch,
        ready_objects=[parent_pipe, process.sentinel],
        parent_pipe=parent_pipe,
        process=process,
    )

    assert agent._history == [result]
    assert parent_pipe.recv_calls == 1
    assert process.join_calls == [None]
    assert process.closed is True


def test_execute_pool_jobs_worker_crash_records_error_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_pipe = _FakePipe(poll_result=False)
    process = _FakeProcess(sentinel=102, exitcode=1)

    agent = _execute_pool_jobs_with_fake_worker(
        monkeypatch,
        ready_objects=[process.sentinel],
        parent_pipe=parent_pipe,
        process=process,
    )

    assert len(agent._history) == 1
    assert 'error' in agent._history[0]
    assert 'ChildProcessError' in agent._history[0]['error']
    assert 'exit code 1' in agent._history[0]['error']
    assert parent_pipe.recv_calls == 0


def test_execute_pool_jobs_pipe_eof_records_error_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_pipe = _FakePipe(poll_result=True, recv_error=EOFError())
    process = _FakeProcess(sentinel=103, exitcode=0)

    agent = _execute_pool_jobs_with_fake_worker(
        monkeypatch,
        ready_objects=[parent_pipe, process.sentinel],
        parent_pipe=parent_pipe,
        process=process,
    )

    assert len(agent._history) == 1
    assert 'error' in agent._history[0]
    assert 'ChildProcessError' in agent._history[0]['error']
    assert 'exit code 0' in agent._history[0]['error']
    assert parent_pipe.recv_calls == 1


# ---------------------------------------------------------------------------
# Pool timeout — pending + remaining jobs (lines 301-332)
# ---------------------------------------------------------------------------

def test_pool_timeout_cancels_pending_and_remaining_jobs() -> None:
    """With 1 worker and 3 slow jobs, timeout cancels the in-flight job
    and records all remaining jobs as timed-out."""
    agent = SearchAgent(
        task=_SlowTask(),
        hps={'a': [1, 2, 3]},
        cuda_ids=[0],
        job_timeout=0.1,
    )
    agent.execute()

    assert len(agent._history) == 3
    for entry in agent._history:
        assert 'error' in entry
        assert 'TimeoutError' in entry['error']
        assert '0.1' in entry['error']


# ---------------------------------------------------------------------------
# Pool worker crash — worker exits without a result
# ---------------------------------------------------------------------------

def test_pool_worker_crash_records_error() -> None:
    """A worker that crashes (os._exit) is caught and recorded as an error."""
    agent = SearchAgent(
        task=_CrashTask(),
        hps=None,
        cuda_ids=[0],
    )
    agent.execute()

    assert len(agent._history) == 1
    assert 'error' in agent._history[0]
    assert 'exit code' in agent._history[0]['error']


def test_pool_large_result_is_received_without_timeout() -> None:
    """Large results must be drained from the pipe before waiting only on process exit."""
    payload_size = 2 * 1024 * 1024
    agent = SearchAgent(
        task=_LargeResultTask(payload_size=payload_size),
        hps=None,
        cuda_ids=[0],
        job_timeout=1.0,
    )
    agent.execute()

    assert len(agent._history) == 1
    assert 'error' not in agent._history[0]
    assert len(agent._history[0]['result']['blob']) == payload_size
