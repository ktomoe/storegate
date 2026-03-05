"""Unit tests for SearchAgent, GridSearchAgent, and RandomSearchAgent."""
from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock

from storegate.agent.search_agent import SearchAgent
from storegate.agent.grid_search_agent import GridSearchAgent
from storegate.agent.random_search_agent import RandomSearchAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_task() -> MagicMock:
    task = MagicMock()
    task.execute.return_value = {'score': 0.9}
    return make_task


def _agent(**kwargs: object) -> SearchAgent:
    defaults: dict = dict(task=MagicMock(), hps=None, num_trials=None, cuda_ids=None)
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

    _agent().execute_task(task, hps={}, job_id=0)

    assert call_order == ['set_hps', 'execute', 'finalize']


def test_execute_task_stores_result() -> None:
    task = MagicMock()
    task.execute.return_value = {'auc': 0.95}
    result = _agent().execute_task(task, hps={}, job_id=0)
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

    result = _agent().execute_task(task, hps={}, job_id=0)

    assert 'error' in result
    assert 'RuntimeError' in result['error']
    assert 'something failed' in result['error']


def test_execute_task_on_exception_has_no_result_key() -> None:
    task = MagicMock()
    task.execute.side_effect = ValueError('bad value')

    result = _agent().execute_task(task, hps={}, job_id=0)
    assert 'result' not in result


def test_execute_task_finalize_not_called_on_exception() -> None:
    task = MagicMock()
    task.execute.side_effect = RuntimeError('fail')

    _agent().execute_task(task, hps={}, job_id=0)
    task.finalize.assert_not_called()


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
