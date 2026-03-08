"""Unit tests for SearchAgent, GridSearchAgent, and RandomSearchAgent."""
from __future__ import annotations

import json
import time
import pytest
from unittest.mock import MagicMock

from storegate.agent.search_agent import SearchAgent
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_task() -> MagicMock:
    task = MagicMock()
    task.execute.return_value = {'score': 0.9}
    return task


def _agent(**kwargs: object) -> SearchAgent:
    task = kwargs.pop('task', MagicMock(spec=['set_hps', 'execute', 'finalize']))
    defaults: dict = dict(task=task, hps=None, num_trials=None, cuda_ids=None)
    defaults.update(kwargs)
    return SearchAgent(**defaults)


# ---------------------------------------------------------------------------
# Initialization — cuda_ids validation
# ---------------------------------------------------------------------------

def test_cuda_ids_as_int_raises() -> None:
    with pytest.raises(TypeError, match='must be a list'):
        SearchAgent(task=MagicMock(), cuda_ids=0)


def test_cuda_ids_as_list_is_accepted() -> None:
    agent = SearchAgent(task=MagicMock(), cuda_ids=[0, 1])
    assert agent._cuda_ids == [0, 1]


def test_cuda_ids_none_is_accepted() -> None:
    agent = _agent()
    assert agent._cuda_ids is None


def test_cuda_ids_empty_list_raises() -> None:
    with pytest.raises(ValueError, match='must not be an empty list'):
        SearchAgent(task=MagicMock(), cuda_ids=[])


def test_cuda_ids_negative_value_raises() -> None:
    with pytest.raises(ValueError, match='non-negative integers'):
        SearchAgent(task=MagicMock(), cuda_ids=[-1, 0])


def test_cuda_ids_non_int_value_raises() -> None:
    with pytest.raises(TypeError, match='non-negative integers'):
        SearchAgent(task=MagicMock(), cuda_ids=[0, '1'])  # type: ignore[list-item]


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


def test_json_dump_non_json_suffix_raises(tmp_path: object) -> None:
    with pytest.raises(ValueError, match='.json'):
        SearchAgent(task=MagicMock(), json_dump=str(tmp_path / 'result.txt'))


def test_json_dump_nonexistent_parent_raises() -> None:
    with pytest.raises(ValueError, match='does not exist'):
        SearchAgent(task=MagicMock(), json_dump='/nonexistent_dir/result.json')


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


def test_execute_task_finalize_failure_does_not_propagate() -> None:
    task = MagicMock()
    task.execute.side_effect = RuntimeError('execute fail')
    task.finalize.side_effect = RuntimeError('finalize fail')

    result = _agent().execute_task(task, hps={}, job_id=0)
    assert 'error' in result  # execute error recorded; finalize error does not override


# ---------------------------------------------------------------------------
# finalize — sorting
# ---------------------------------------------------------------------------

def test_finalize_sorts_history_by_job_id() -> None:
    agent = _agent()
    agent._history = [
        {'job_id': 2, 'trial_id': None},
        {'job_id': 0, 'trial_id': None},
        {'job_id': 1, 'trial_id': None},
    ]
    agent.finalize()
    assert [r['job_id'] for r in agent._history] == [0, 1, 2]


def test_finalize_sorts_by_trial_id_within_same_job() -> None:
    agent = _agent()
    agent._history = [
        {'job_id': 0, 'trial_id': 2},
        {'job_id': 0, 'trial_id': 0},
        {'job_id': 0, 'trial_id': 1},
    ]
    agent.finalize()
    assert [r['trial_id'] for r in agent._history] == [0, 1, 2]


def test_finalize_sorts_job_id_before_trial_id() -> None:
    agent = _agent()
    agent._history = [
        {'job_id': 1, 'trial_id': 0},
        {'job_id': 0, 'trial_id': 1},
        {'job_id': 0, 'trial_id': 0},
    ]
    agent.finalize()
    assert [(r['job_id'], r['trial_id']) for r in agent._history] == [
        (0, 0), (0, 1), (1, 0)
    ]


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


# ---------------------------------------------------------------------------
# GridSearchAgent
# ---------------------------------------------------------------------------

def test_grid_search_agent_exhausts_all_combinations() -> None:
    agent = GridSearchAgent(task=MagicMock(), hps={'a': [1, 2], 'b': [10, 20]})
    assert len(agent._hps) == 4


def test_grid_search_agent_single_param() -> None:
    agent = GridSearchAgent(task=MagicMock(), hps={'lr': [1e-3, 1e-4, 1e-5]})
    assert len(agent._hps) == 3


# ---------------------------------------------------------------------------
# RandomSearchAgent
# ---------------------------------------------------------------------------

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


def test_random_search_agent_none_hps_returns_single_empty_dict() -> None:
    agent = RandomSearchAgent(num_iter=5, seed=0, task=MagicMock(), hps=None)
    assert agent._hps == [{}]


def test_random_search_agent_each_sample_uses_valid_values() -> None:
    valid = {'lr': [1e-3, 1e-4], 'bs': [32, 64, 128]}
    agent = RandomSearchAgent(num_iter=20, seed=7, task=MagicMock(), hps=valid)
    for combo in agent._hps:
        assert combo['lr'] in valid['lr']
        assert combo['bs'] in valid['bs']


# ---------------------------------------------------------------------------
# execute / execute_pool_jobs — parallel integration
#
# These tests exercise the real ProcessPoolExecutor (spawn context).
# Task stubs are defined at module level so child processes can pickle them.
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


def test_suffix_job_id_default_is_true() -> None:
    agent = SearchAgent(task=MagicMock())
    assert agent._suffix_job_id is True


def test_suffix_job_id_appends_to_string_output_var_names() -> None:
    agent = SearchAgent(
        task=_OutputVarNamesTask('pred'),
        hps=None,
        suffix_job_id=True,
    )
    agent.execute()
    assert agent._history[0]['result']['output_var_names'] == 'pred_job0'


def test_suffix_job_id_appends_to_list_output_var_names() -> None:
    agent = SearchAgent(
        task=_OutputVarNamesTask(['pred', 'score']),
        hps=None,
        suffix_job_id=True,
    )
    agent.execute()
    assert agent._history[0]['result']['output_var_names'] == ['pred_job0', 'score_job0']


def test_suffix_job_id_appends_to_phase_dict_output_var_names() -> None:
    agent = SearchAgent(
        task=_OutputVarNamesTask({'train': None, 'valid': ['val_pred'], 'test': 'pred'}),
        hps=None,
        suffix_job_id=True,
    )
    agent.execute()
    assert agent._history[0]['result']['output_var_names'] == {
        'train': None,
        'valid': ['val_pred_job0'],
        'test': 'pred_job0',
    }


def test_suffix_job_id_uses_job_id_for_each_job() -> None:
    agent = SearchAgent(
        task=_OutputVarNamesTask('pred'),
        hps={'a': [1, 2]},
        suffix_job_id=True,
    )
    agent.execute()
    agent.finalize()
    assert [r['result']['output_var_names'] for r in agent._history] == [
        'pred_job0',
        'pred_job1',
    ]


def test_suffix_job_id_suffixes_explicit_hps_string_output_var_names() -> None:
    agent = SearchAgent(
        task=_OutputVarNamesTask('pred'),
        hps={'output_var_names': ['custom']},
        suffix_job_id=True,
    )
    agent.execute()
    assert agent._history[0]['result']['output_var_names'] == 'custom_job0'


def test_suffix_job_id_suffixes_explicit_hps_list_output_var_names() -> None:
    agent = SearchAgent(
        task=_OutputVarNamesTask('pred'),
        hps={'output_var_names': [['custom', 'score']]},
        suffix_job_id=True,
    )
    agent.execute()
    assert agent._history[0]['result']['output_var_names'] == ['custom_job0', 'score_job0']


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

def test_job_timeout_default_is_none() -> None:
    agent = _agent()
    assert agent._job_timeout is None


def test_job_timeout_stored_correctly() -> None:
    agent = SearchAgent(task=MagicMock(), job_timeout=5.0)
    assert agent._job_timeout == 5.0


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

def test_suffix_output_var_names_none_returns_none() -> None:
    assert SearchAgent._suffix_output_var_names(None, 0) is None


def test_suffix_output_var_names_str() -> None:
    assert SearchAgent._suffix_output_var_names('pred', 3) == 'pred_job3'


def test_suffix_output_var_names_list() -> None:
    result = SearchAgent._suffix_output_var_names(['a', 'b'], 1)
    assert result == ['a_job1', 'b_job1']


def test_suffix_output_var_names_dict_with_mixed_values() -> None:
    result = SearchAgent._suffix_output_var_names(
        {'train': None, 'valid': ['v'], 'test': 'pred'}, 2
    )
    assert result == {'train': None, 'valid': ['v_job2'], 'test': 'pred_job2'}


def test_suffix_output_var_names_unsupported_type_raises() -> None:
    with pytest.raises(TypeError, match='got int'):
        SearchAgent._suffix_output_var_names(42, 0)


# ---------------------------------------------------------------------------
# execute_task with suffix_job_id — direct unit tests (no subprocess)
# ---------------------------------------------------------------------------

def test_execute_task_suffix_job_id_injects_suffixed_var_names() -> None:
    task = MagicMock()
    task._output_var_names = 'pred'
    task.execute.return_value = {}
    agent = SearchAgent(task=task, suffix_job_id=True)

    agent.execute_task(task, hps={}, job_id=5)

    passed_hps = task.set_hps.call_args[0][0]
    assert passed_hps['output_var_names'] == 'pred_job5'


def test_execute_task_suffix_job_id_uses_hps_override() -> None:
    task = MagicMock()
    task._output_var_names = 'pred'
    task.execute.return_value = {}
    agent = SearchAgent(task=task, suffix_job_id=True)

    agent.execute_task(task, hps={'output_var_names': 'custom'}, job_id=0)

    passed_hps = task.set_hps.call_args[0][0]
    assert passed_hps['output_var_names'] == 'custom_job0'


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
