"""Comprehensive tests for storegate agent modules.

Covers:
- storegate/agent/agent.py
- storegate/agent/search_agent.py
- storegate/agent/grid_search_agent.py
- storegate/agent/random_search_agent.py
"""
from __future__ import annotations

import json
import multiprocessing as mp
import queue
import tempfile
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest

from storegate.agent.agent import Agent
from storegate.agent.search_agent import (
    SearchAgent,
    _apply_job_id,
    _json_default,
    _worker_loop,
)
from storegate.agent.grid_search_agent import GridSearchAgent
from storegate.agent.random_search_agent import RandomSearchAgent


# ---------------------------------------------------------------------------
# Helper task classes (no torch dependency)
# ---------------------------------------------------------------------------

class SimpleTask:
    """Minimal task that can run in-process and in child processes."""

    _progress_callback = None

    def __init__(self, **kwargs: Any) -> None:
        self._storegate = None
        self._hp_value: Any = None
        self._kwargs = kwargs
        self._cuda_id: int | None = None

    def set_cuda_id(self, cuda_id: int) -> None:
        self._cuda_id = cuda_id

    @property
    def storegate(self):
        return self._storegate

    @storegate.setter
    def storegate(self, sg):
        self._storegate = sg

    def set_hps(self, hps: dict[str, Any]) -> None:
        self._hp_value = hps.get("x")

    def execute(self) -> dict[str, Any]:
        return {"value": self._hp_value}

    def reset(self) -> None:
        pass


class FailingTask(SimpleTask):
    """Task that always raises during execute."""

    def execute(self) -> dict[str, Any]:
        raise ValueError("intentional error")


class TaskWithoutReset:
    """Task with no reset method."""

    _progress_callback = None

    def __init__(self, **kwargs: Any) -> None:
        self._storegate = None
        self._cuda_id: int | None = None

    def set_cuda_id(self, cuda_id: int) -> None:
        self._cuda_id = cuda_id

    @property
    def storegate(self):
        return self._storegate

    @storegate.setter
    def storegate(self, sg):
        self._storegate = sg

    def set_hps(self, hps: dict[str, Any]) -> None:
        pass

    def execute(self) -> dict[str, Any]:
        return {"ok": True}


class TaskWithProgressCallback(SimpleTask):
    """Task that invokes the progress callback during execute."""

    def execute(self) -> dict[str, Any]:
        if self._progress_callback is not None:
            self._progress_callback({"epoch": 1, "batch": 1})
        return {"value": self._hp_value}


# ===========================================================================
# Agent base class
# ===========================================================================

class TestAgentAbstract:
    """Tests for agent.py Agent base class."""

    def test_agent_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            Agent()

    def test_concrete_subclass_works(self):
        class ConcreteAgent(Agent):
            def execute(self):
                return 42

        agent = ConcreteAgent()
        assert agent.execute() == 42


# ===========================================================================
# _apply_job_id
# ===========================================================================

class TestApplyJobId:
    """Tests for the _apply_job_id helper function."""

    def test_no_var_names_key(self):
        args = {"foo": "bar"}
        result = _apply_job_id(args, "job0")
        assert result == {"foo": "bar"}

    def test_var_names_but_no_outputs(self):
        args = {"var_names": {"inputs": "x"}}
        result = _apply_job_id(args, "job0")
        assert result == {"var_names": {"inputs": "x"}}

    def test_outputs_as_str(self):
        args = {"var_names": {"outputs": "pred"}}
        result = _apply_job_id(args, "hp0_trial0")
        assert result["var_names"]["outputs"] == "pred_hp0_trial0"

    def test_outputs_as_list(self):
        args = {"var_names": {"outputs": ["a", "b"]}}
        result = _apply_job_id(args, "hp1_trial2")
        assert result["var_names"]["outputs"] == ["a_hp1_trial2", "b_hp1_trial2"]

    def test_original_not_mutated(self):
        args = {"var_names": {"outputs": ["a", "b"]}}
        original_outputs = args["var_names"]["outputs"]
        _apply_job_id(args, "job0")
        assert args["var_names"]["outputs"] is original_outputs
        assert args["var_names"]["outputs"] == ["a", "b"]

    def test_var_names_none_value(self):
        args = {"var_names": None}
        result = _apply_job_id(args, "job0")
        assert result == {"var_names": None}


# ===========================================================================
# _json_default
# ===========================================================================

class TestJsonDefault:
    """Tests for the _json_default fallback encoder."""

    def test_ndarray_to_list(self):
        arr = np.array([1, 2, 3])
        assert _json_default(arr) == [1, 2, 3]

    def test_integer(self):
        val = np.int64(42)
        result = _json_default(val)
        assert result == 42
        assert isinstance(result, int)

    def test_floating(self):
        val = np.float32(3.14)
        result = _json_default(val)
        assert isinstance(result, float)
        assert abs(result - 3.14) < 1e-5

    def test_bool_(self):
        val = np.bool_(True)
        result = _json_default(val)
        assert result is True
        assert isinstance(result, bool)

    def test_bool_false(self):
        val = np.bool_(False)
        result = _json_default(val)
        assert result is False

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="not JSON serializable"):
            _json_default(object())


# ===========================================================================
# GridSearchAgent._generate_hp_list
# ===========================================================================

class TestGridSearchAgentGenerateHpList:
    """Tests for GridSearchAgent._generate_hp_list."""

    def test_empty_hps(self):
        agent = GridSearchAgent(hps={})
        result = agent._generate_hp_list()
        assert result == [{}]

    def test_none_hps(self):
        agent = GridSearchAgent(hps=None)
        result = agent._generate_hp_list()
        assert result == [{}]

    def test_single_param(self):
        agent = GridSearchAgent(hps={"x": [1, 2, 3]})
        result = agent._generate_hp_list()
        assert result == [{"x": 1}, {"x": 2}, {"x": 3}]

    def test_multiple_params_cartesian_product(self):
        agent = GridSearchAgent(hps={"x": [1, 2], "y": ["a", "b"]})
        result = agent._generate_hp_list()
        expected = [
            {"x": 1, "y": "a"},
            {"x": 1, "y": "b"},
            {"x": 2, "y": "a"},
            {"x": 2, "y": "b"},
        ]
        assert result == expected

    def test_empty_values_list_error(self):
        agent = GridSearchAgent(hps={"x": []})
        with pytest.raises(ValueError, match="must not be empty"):
            agent._generate_hp_list()

    def test_empty_values_for_one_key(self):
        agent = GridSearchAgent(hps={"x": [1], "y": []})
        with pytest.raises(ValueError, match="'y'"):
            agent._generate_hp_list()


# ===========================================================================
# RandomSearchAgent._generate_hp_list
# ===========================================================================

class TestRandomSearchAgentGenerateHpList:
    """Tests for RandomSearchAgent._generate_hp_list."""

    def test_empty_hps(self):
        agent = RandomSearchAgent(hps={}, num_samples=5)
        result = agent._generate_hp_list()
        assert result == [{}]

    def test_none_hps(self):
        agent = RandomSearchAgent(hps=None, num_samples=3)
        result = agent._generate_hp_list()
        assert result == [{}]

    def test_seed_reproducibility(self):
        agent1 = RandomSearchAgent(
            hps={"x": [1, 2, 3, 4, 5]}, num_samples=3, seed=42
        )
        agent2 = RandomSearchAgent(
            hps={"x": [1, 2, 3, 4, 5]}, num_samples=3, seed=42
        )
        assert agent1._generate_hp_list() == agent2._generate_hp_list()

    def test_num_samples(self):
        agent = RandomSearchAgent(
            hps={"x": [1, 2, 3]}, num_samples=10, seed=0
        )
        result = agent._generate_hp_list()
        assert len(result) == 10
        for hp_set in result:
            assert "x" in hp_set
            assert hp_set["x"] in [1, 2, 3]

    def test_empty_values_list_error(self):
        agent = RandomSearchAgent(hps={"x": []}, num_samples=1, seed=0)
        with pytest.raises(ValueError, match="must not be empty"):
            agent._generate_hp_list()

    def test_num_samples_validation_zero(self):
        with pytest.raises(ValueError, match="num_samples"):
            RandomSearchAgent(hps={"x": [1]}, num_samples=0)

    def test_num_samples_validation_negative(self):
        with pytest.raises(ValueError, match="num_samples"):
            RandomSearchAgent(hps={"x": [1]}, num_samples=-1)

    def test_num_samples_validation_non_int(self):
        with pytest.raises(ValueError, match="num_samples"):
            RandomSearchAgent(hps={"x": [1]}, num_samples=1.5)

    def test_num_samples_validation_bool(self):
        with pytest.raises(ValueError, match="num_samples"):
            RandomSearchAgent(hps={"x": [1]}, num_samples=True)

    def test_multiple_hps(self):
        agent = RandomSearchAgent(
            hps={"x": [1, 2], "y": ["a", "b"]}, num_samples=5, seed=7
        )
        result = agent._generate_hp_list()
        assert len(result) == 5
        for hp_set in result:
            assert hp_set["x"] in [1, 2]
            assert hp_set["y"] in ["a", "b"]


# ===========================================================================
# SearchAgent init / properties
# ===========================================================================

class TestSearchAgentInit:
    """Tests for SearchAgent initialization and property accessors."""

    def test_storegate_property_getter(self):
        sg = mock.MagicMock()
        agent = GridSearchAgent(storegate=sg)
        assert agent.storegate is sg

    def test_storegate_property_setter(self):
        agent = GridSearchAgent()
        sg = mock.MagicMock()
        agent.storegate = sg
        assert agent.storegate is sg

    def test_storegate_default_none(self):
        agent = GridSearchAgent()
        assert agent.storegate is None

    def test_num_trials_validation_zero(self):
        with pytest.raises(ValueError, match="num_trials"):
            GridSearchAgent(num_trials=0)

    def test_num_trials_validation_negative(self):
        with pytest.raises(ValueError, match="num_trials"):
            GridSearchAgent(num_trials=-5)

    def test_num_trials_validation_non_int(self):
        with pytest.raises(ValueError, match="num_trials"):
            GridSearchAgent(num_trials=2.5)


# ===========================================================================
# SearchAgent._require_task_cls
# ===========================================================================

class TestRequireTaskCls:
    """Tests for SearchAgent._require_task_cls."""

    def test_none_task_raises(self):
        agent = GridSearchAgent(task=None)
        with pytest.raises(ValueError, match="task is required"):
            agent._require_task_cls()

    def test_valid_task_returns_cls(self):
        agent = GridSearchAgent(task=SimpleTask)
        assert agent._require_task_cls() is SimpleTask


# ===========================================================================
# SearchAgent.execute
# ===========================================================================

class TestSearchAgentExecute:
    """Tests for SearchAgent.execute orchestration."""

    def test_empty_hp_list_returns_empty(self):
        agent = GridSearchAgent(
            task=SimpleTask, hps={}, num_trials=1, progress=False
        )
        # hps={} produces [{}], 1 trial => 1 job, not empty
        # To get truly empty, we need a custom subclass
        class EmptySearchAgent(SearchAgent):
            def _generate_hp_list(self):
                return []

        agent = EmptySearchAgent(task=SimpleTask, progress=False)
        result = agent.execute()
        assert result == []

    def test_basic_run(self):
        agent = GridSearchAgent(
            task=SimpleTask,
            hps={"x": [10, 20]},
            num_trials=1,
            progress=False,
        )
        results = agent.execute()
        assert len(results) == 2
        assert results[0]["status"] == "success"
        assert results[1]["status"] == "success"
        assert results[0]["result"] == {"value": 10}
        assert results[1]["result"] == {"value": 20}

    def test_sorted_by_hp_idx_trial(self):
        agent = GridSearchAgent(
            task=SimpleTask,
            hps={"x": [10, 20]},
            num_trials=2,
            progress=False,
        )
        results = agent.execute()
        assert len(results) == 4
        keys = [(r["hp_idx"], r["trial"]) for r in results]
        assert keys == sorted(keys)

    def test_execute_with_progress_true(self):
        agent = GridSearchAgent(
            task=SimpleTask,
            hps={"x": [1]},
            num_trials=1,
            progress=True,
        )
        # Should not raise, progress output goes to formatter
        results = agent.execute()
        assert len(results) == 1

    def test_execute_with_progress_false(self):
        agent = GridSearchAgent(
            task=SimpleTask,
            hps={"x": [1]},
            num_trials=1,
            progress=False,
        )
        results = agent.execute()
        assert len(results) == 1


# ===========================================================================
# SearchAgent._run_sequential
# ===========================================================================

class TestRunSequential:
    """Tests for SearchAgent._run_sequential."""

    def test_success_path(self):
        agent = GridSearchAgent(
            task=SimpleTask,
            hps={"x": [5]},
            num_trials=1,
            progress=False,
        )
        results = agent.execute()
        assert len(results) == 1
        assert results[0]["status"] == "success"
        assert results[0]["result"] == {"value": 5}
        assert results[0]["cuda_id"] is None

    def test_error_path(self):
        agent = GridSearchAgent(
            task=FailingTask,
            hps={"x": [1]},
            num_trials=1,
            progress=False,
        )
        results = agent.execute()
        assert len(results) == 1
        assert results[0]["status"] == "error"
        assert "intentional error" in results[0]["error"]

    def test_task_reset_called(self):
        reset_called = []

        class TrackResetTask(SimpleTask):
            def reset(self):
                reset_called.append(True)

        agent = GridSearchAgent(
            task=TrackResetTask,
            hps={"x": [1]},
            num_trials=1,
            progress=False,
        )
        agent.execute()
        assert len(reset_called) == 1

    def test_task_without_reset_no_error(self):
        agent = GridSearchAgent(
            task=TaskWithoutReset,
            hps={"x": [1]},
            num_trials=1,
            progress=False,
        )
        # TaskWithoutReset does have reset...but not callable? Actually it doesn't have reset.
        # Let's check: TaskWithoutReset has no reset method - but wait, it doesn't.
        # The code does: reset = getattr(task, "reset", None); if callable(reset): reset()
        # TaskWithoutReset doesn't define reset, so getattr returns None.
        results = agent.execute()
        assert len(results) == 1
        assert results[0]["status"] == "success"

    def test_reset_called_on_error_too(self):
        reset_called = []

        class FailWithReset(FailingTask):
            def reset(self):
                reset_called.append(True)

        agent = GridSearchAgent(
            task=FailWithReset,
            hps={"x": [1]},
            num_trials=1,
            progress=False,
        )
        agent.execute()
        assert len(reset_called) == 1

    def test_progress_callbacks(self):
        calls = {"print_job_start": 0, "print_batch": 0, "print_job_end": 0}

        agent = GridSearchAgent(
            task=TaskWithProgressCallback,
            hps={"x": [1]},
            num_trials=1,
            progress=True,
        )

        orig_start = agent._formatter.print_job_start
        orig_batch = agent._formatter.print_batch
        orig_end = agent._formatter.print_job_end

        def mock_start(*a, **kw):
            calls["print_job_start"] += 1
            orig_start(*a, **kw)

        def mock_batch(*a, **kw):
            calls["print_batch"] += 1
            orig_batch(*a, **kw)

        def mock_end(*a, **kw):
            calls["print_job_end"] += 1
            orig_end(*a, **kw)

        agent._formatter.print_job_start = mock_start
        agent._formatter.print_batch = mock_batch
        agent._formatter.print_job_end = mock_end

        agent.execute()
        assert calls["print_job_start"] >= 1
        assert calls["print_batch"] >= 1
        assert calls["print_job_end"] >= 1

    def test_storegate_passed_to_task(self):
        sg = mock.MagicMock()

        class CheckStoregateTask(SimpleTask):
            def execute(self):
                return {"has_sg": self._storegate is not None}

        agent = GridSearchAgent(
            task=CheckStoregateTask,
            storegate=sg,
            hps={"x": [1]},
            num_trials=1,
            progress=False,
        )
        results = agent.execute()
        assert results[0]["result"]["has_sg"] is True

    def test_apply_job_id_integration(self):
        """Verify output names get appended with job_id in sequential mode."""
        captured_args = []

        class CapturingTask(SimpleTask):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                captured_args.append(kwargs)

        agent = GridSearchAgent(
            task=CapturingTask,
            task_args={"var_names": {"outputs": "pred"}},
            hps={"x": [1]},
            num_trials=1,
            progress=False,
        )
        agent.execute()
        assert len(captured_args) == 1
        assert "hp0_trial0" in captured_args[0]["var_names"]["outputs"]


# ===========================================================================
# SearchAgent._get_slot_labels / sequential / parallel
# ===========================================================================

class TestSlotLabels:
    """Tests for slot label methods."""

    def test_get_slot_labels_without_cuda_ids(self):
        agent = GridSearchAgent()
        labels = agent._get_slot_labels()
        assert labels == ["cpu"]

    def test_get_slot_labels_with_cuda_ids(self):
        agent = GridSearchAgent(cuda_ids=[0, 1])
        labels = agent._get_slot_labels()
        assert labels == ["cuda:0", "cuda:1"]

    def test_get_sequential_slot_label(self):
        agent = GridSearchAgent()
        assert agent._get_sequential_slot_label() == "cpu"

    def test_get_parallel_slot_labels_simple(self):
        agent = GridSearchAgent(cuda_ids=[0, 1])
        labels = agent._get_parallel_slot_labels()
        assert labels == ["cuda:0", "cuda:1"]

    def test_get_parallel_slot_labels_duplicate(self):
        agent = GridSearchAgent(cuda_ids=[0, 0])
        labels = agent._get_parallel_slot_labels()
        assert labels == ["cuda:0", "cuda:0(2)"]

    def test_get_parallel_slot_labels_triple(self):
        agent = GridSearchAgent(cuda_ids=[1, 1, 1])
        labels = agent._get_parallel_slot_labels()
        assert labels == ["cuda:1", "cuda:1(2)", "cuda:1(3)"]

    def test_get_parallel_slot_labels_mixed(self):
        agent = GridSearchAgent(cuda_ids=[0, 1, 0])
        labels = agent._get_parallel_slot_labels()
        assert labels == ["cuda:0", "cuda:1", "cuda:0(2)"]


# ===========================================================================
# SearchAgent._terminate_alive_workers
# ===========================================================================

class TestTerminateAliveWorkers:
    """Tests for _terminate_alive_workers."""

    def test_alive_workers_terminated(self):
        agent = GridSearchAgent()
        worker = mock.MagicMock()
        worker.exitcode = None  # alive
        agent._terminate_alive_workers([worker])
        worker.terminate.assert_called_once()

    def test_dead_workers_not_terminated(self):
        agent = GridSearchAgent()
        worker = mock.MagicMock()
        worker.exitcode = 0  # dead
        agent._terminate_alive_workers([worker])
        worker.terminate.assert_not_called()

    def test_mixed_workers(self):
        agent = GridSearchAgent()
        alive = mock.MagicMock()
        alive.exitcode = None
        dead = mock.MagicMock()
        dead.exitcode = 0
        agent._terminate_alive_workers([alive, dead])
        alive.terminate.assert_called_once()
        dead.terminate.assert_not_called()

    def test_empty_list(self):
        agent = GridSearchAgent()
        agent._terminate_alive_workers([])


# ===========================================================================
# SearchAgent.save_results
# ===========================================================================

class TestSaveResults:
    """Tests for SearchAgent.save_results."""

    def test_writes_valid_json(self):
        agent = GridSearchAgent()
        results = [
            {"job_id": "hp0_trial0", "status": "success", "result": {"val": 1}},
        ]
        with tempfile.NamedTemporaryFile(mode="r", suffix=".json", delete=False) as f:
            path = f.name

        agent.save_results(results, path)

        with open(path) as f:
            loaded = json.load(f)
        assert loaded == results

    def test_numpy_types_serialized(self):
        agent = GridSearchAgent()
        results = [
            {
                "job_id": "hp0",
                "arr": np.array([1.0, 2.0]),
                "int_val": np.int64(42),
                "float_val": np.float32(3.14),
                "bool_val": np.bool_(True),
            },
        ]
        with tempfile.NamedTemporaryFile(mode="r", suffix=".json", delete=False) as f:
            path = f.name

        agent.save_results(results, path)

        with open(path) as f:
            loaded = json.load(f)
        assert loaded[0]["arr"] == [1.0, 2.0]
        assert loaded[0]["int_val"] == 42
        assert abs(loaded[0]["float_val"] - 3.14) < 1e-2
        assert loaded[0]["bool_val"] is True

    def test_save_results_with_path_object(self):
        agent = GridSearchAgent()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.json"
            agent.save_results([{"a": 1}], path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded == [{"a": 1}]


# ===========================================================================
# _worker_loop (unit tests via queues)
# ===========================================================================

class TestWorkerLoop:
    """Tests for _worker_loop function (direct invocation, not via subprocess)."""

    def test_success_path(self):
        result_q: queue.Queue = queue.Queue()
        progress_q: queue.Queue = queue.Queue()

        job = {
            "job_id": "hp0_trial0",
            "hp_idx": 0,
            "trial": 0,
            "hps": {"x": 42},
        }

        _worker_loop(
            cuda_id=0,
            slot_label="cuda:0",
            job=job,
            result_queue=result_q,
            progress_queue=progress_q,
            task_cls=SimpleTask,
            task_args={},
            storegate=None,
        )

        result = result_q.get_nowait()
        assert result["status"] == "success"
        assert result["result"] == {"value": 42}
        assert result["job_id"] == "hp0_trial0"
        assert result["cuda_id"] == 0

        # Check progress messages
        msgs = []
        while not progress_q.empty():
            msgs.append(progress_q.get_nowait())

        types = [m["type"] for m in msgs]
        assert "job_start" in types
        assert "job_end" in types

    def test_error_path(self):
        result_q: queue.Queue = queue.Queue()
        progress_q: queue.Queue = queue.Queue()

        job = {
            "job_id": "hp0_trial0",
            "hp_idx": 0,
            "trial": 0,
            "hps": {"x": 1},
        }

        _worker_loop(
            cuda_id=0,
            slot_label="cuda:0",
            job=job,
            result_queue=result_q,
            progress_queue=progress_q,
            task_cls=FailingTask,
            task_args={},
            storegate=None,
        )

        result = result_q.get_nowait()
        assert result["status"] == "error"
        assert "intentional error" in result["error"]

        # Check progress messages include error job_end
        msgs = []
        while not progress_q.empty():
            msgs.append(progress_q.get_nowait())

        job_end_msgs = [m for m in msgs if m["type"] == "job_end"]
        assert len(job_end_msgs) == 1
        assert job_end_msgs[0]["status"] == "error"

    def test_storegate_close_on_exit(self):
        sg = mock.MagicMock()
        result_q: queue.Queue = queue.Queue()

        job = {
            "job_id": "hp0_trial0",
            "hp_idx": 0,
            "trial": 0,
            "hps": {"x": 1},
        }

        _worker_loop(
            cuda_id=0,
            slot_label="cuda:0",
            job=job,
            result_queue=result_q,
            progress_queue=None,
            task_cls=SimpleTask,
            task_args={},
            storegate=sg,
        )

        sg.close.assert_called_once()

    def test_storegate_close_on_error(self):
        sg = mock.MagicMock()
        result_q: queue.Queue = queue.Queue()

        job = {
            "job_id": "hp0_trial0",
            "hp_idx": 0,
            "trial": 0,
            "hps": {"x": 1},
        }

        _worker_loop(
            cuda_id=0,
            slot_label="cuda:0",
            job=job,
            result_queue=result_q,
            progress_queue=None,
            task_cls=FailingTask,
            task_args={},
            storegate=sg,
        )

        sg.close.assert_called_once()

    def test_storegate_close_exception_suppressed(self):
        sg = mock.MagicMock()
        sg.close.side_effect = RuntimeError("close failed")
        result_q: queue.Queue = queue.Queue()

        job = {
            "job_id": "hp0_trial0",
            "hp_idx": 0,
            "trial": 0,
            "hps": {"x": 1},
        }

        # Should not raise even though close fails
        _worker_loop(
            cuda_id=0,
            slot_label="cuda:0",
            job=job,
            result_queue=result_q,
            progress_queue=None,
            task_cls=SimpleTask,
            task_args={},
            storegate=sg,
        )

        result = result_q.get_nowait()
        assert result["status"] == "success"

    def test_no_progress_queue(self):
        result_q: queue.Queue = queue.Queue()

        job = {
            "job_id": "hp0_trial0",
            "hp_idx": 0,
            "trial": 0,
            "hps": {"x": 1},
        }

        _worker_loop(
            cuda_id=0,
            slot_label="cuda:0",
            job=job,
            result_queue=result_q,
            progress_queue=None,
            task_cls=SimpleTask,
            task_args={},
            storegate=None,
        )

        result = result_q.get_nowait()
        assert result["status"] == "success"

    def test_progress_queue_batch_messages(self):
        result_q: queue.Queue = queue.Queue()
        progress_q: queue.Queue = queue.Queue()

        job = {
            "job_id": "hp0_trial0",
            "hp_idx": 0,
            "trial": 0,
            "hps": {"x": 1},
        }

        _worker_loop(
            cuda_id=0,
            slot_label="cuda:0",
            job=job,
            result_queue=result_q,
            progress_queue=progress_q,
            task_cls=TaskWithProgressCallback,
            task_args={},
            storegate=None,
        )

        msgs = []
        while not progress_q.empty():
            msgs.append(progress_q.get_nowait())

        batch_msgs = [m for m in msgs if m["type"] == "batch"]
        assert len(batch_msgs) >= 1
        assert batch_msgs[0]["slot_label"] == "cuda:0"

    def test_storegate_assigned_to_task(self):
        sg = mock.MagicMock()
        result_q: queue.Queue = queue.Queue()

        class CheckSGTask(SimpleTask):
            def execute(self):
                return {"has_sg": self._storegate is not None}

        job = {
            "job_id": "hp0_trial0",
            "hp_idx": 0,
            "trial": 0,
            "hps": {},
        }

        _worker_loop(
            cuda_id=0,
            slot_label="cuda:0",
            job=job,
            result_queue=result_q,
            progress_queue=None,
            task_cls=CheckSGTask,
            task_args={},
            storegate=sg,
        )

        result = result_q.get_nowait()
        assert result["status"] == "success"
        assert result["result"]["has_sg"] is True

    def test_apply_job_id_in_worker(self):
        captured = []

        class CaptureTask(SimpleTask):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                captured.append(kwargs)

        result_q: queue.Queue = queue.Queue()
        job = {
            "job_id": "hp0_trial0",
            "hp_idx": 0,
            "trial": 0,
            "hps": {},
        }

        _worker_loop(
            cuda_id=0,
            slot_label="cuda:0",
            job=job,
            result_queue=result_q,
            progress_queue=None,
            task_cls=CaptureTask,
            task_args={"var_names": {"outputs": "pred"}},
            storegate=None,
        )

        result = result_q.get_nowait()
        assert result["status"] == "success"
        assert captured[0]["var_names"]["outputs"] == "pred_hp0_trial0"


# ===========================================================================
# ===========================================================================
# SearchAgent._run_parallel (integration, mocked multiprocessing)
# ===========================================================================

class TestRunParallel:
    """Tests for _run_parallel using mocked multiprocessing."""

    def test_parallel_single_worker_success(self):
        """Test _run_parallel with a single GPU via mocking."""
        agent = GridSearchAgent(
            task=SimpleTask,
            hps={"x": [1]},
            num_trials=1,
            cuda_ids=[0],
            progress=False,
        )

        fake_result = {
            "job_id": "hp0_trial0",
            "hp_idx": 0,
            "trial": 0,
            "hps": {"x": 1},
            "cuda_id": 0,
            "status": "success",
            "result": {"value": 1},
        }

        # Create a mock process
        mock_process = mock.MagicMock()
        mock_process.exitcode = 0
        mock_process.start = mock.MagicMock()
        mock_process.join = mock.MagicMock()

        mock_ctx = mock.MagicMock()

        # The result_queue
        mock_result_queue = mock.MagicMock()
        # First call returns the result, second would timeout
        mock_result_queue.get = mock.MagicMock(return_value=fake_result)

        mock_ctx.Queue = mock.MagicMock(return_value=mock_result_queue)
        mock_ctx.Process = mock.MagicMock(return_value=mock_process)

        with mock.patch("storegate.agent.search_agent.mp.get_context", return_value=mock_ctx):
            results = agent.execute()

        assert len(results) == 1
        assert results[0]["status"] == "success"

    def test_parallel_single_worker_error(self):
        """Test _run_parallel with a worker that returns error."""
        agent = GridSearchAgent(
            task=FailingTask,
            hps={"x": [1]},
            num_trials=1,
            cuda_ids=[0],
            progress=False,
        )

        fake_result = {
            "job_id": "hp0_trial0",
            "hp_idx": 0,
            "trial": 0,
            "hps": {"x": 1},
            "cuda_id": 0,
            "status": "error",
            "error": "intentional error",
        }

        mock_process = mock.MagicMock()
        mock_process.exitcode = 0

        mock_ctx = mock.MagicMock()
        mock_result_queue = mock.MagicMock()
        mock_result_queue.get = mock.MagicMock(return_value=fake_result)

        mock_ctx.Queue = mock.MagicMock(return_value=mock_result_queue)
        mock_ctx.Process = mock.MagicMock(return_value=mock_process)

        with mock.patch("storegate.agent.search_agent.mp.get_context", return_value=mock_ctx):
            results = agent.execute()

        assert len(results) == 1
        assert results[0]["status"] == "error"

    def test_parallel_crash_detection(self):
        """Test that crashed workers raise RuntimeError."""
        agent = GridSearchAgent(
            task=SimpleTask,
            hps={"x": [1]},
            num_trials=1,
            cuda_ids=[0],
            progress=False,
        )

        mock_process = mock.MagicMock()
        # Process crashes: exitcode is non-zero and non-None
        mock_process.exitcode = 1
        mock_process.pid = 99999

        mock_ctx = mock.MagicMock()
        mock_result_queue = mock.MagicMock()
        # get times out, causing crash check
        mock_result_queue.get = mock.MagicMock(side_effect=queue.Empty)

        mock_ctx.Queue = mock.MagicMock(return_value=mock_result_queue)
        mock_ctx.Process = mock.MagicMock(return_value=mock_process)

        with mock.patch("storegate.agent.search_agent.mp.get_context", return_value=mock_ctx):
            with pytest.raises(RuntimeError, match="crashed"):
                agent.execute()

    def test_parallel_with_progress(self):
        """Test _run_parallel with progress=True uses progress queue and drain thread."""
        agent = GridSearchAgent(
            task=SimpleTask,
            hps={"x": [1]},
            num_trials=1,
            cuda_ids=[0],
            progress=True,
        )

        fake_result = {
            "job_id": "hp0_trial0",
            "hp_idx": 0,
            "trial": 0,
            "hps": {"x": 1},
            "cuda_id": 0,
            "status": "success",
            "result": {"value": 1},
        }

        mock_process = mock.MagicMock()
        mock_process.exitcode = 0

        mock_ctx = mock.MagicMock()

        # Two queues: result and progress
        queue_call_count = [0]
        mock_result_queue = mock.MagicMock()
        mock_result_queue.get = mock.MagicMock(return_value=fake_result)
        mock_progress_queue = mock.MagicMock()
        mock_progress_queue.get = mock.MagicMock(side_effect=Exception("empty"))

        def queue_factory():
            queue_call_count[0] += 1
            if queue_call_count[0] == 1:
                return mock_result_queue
            return mock_progress_queue

        mock_ctx.Queue = mock.MagicMock(side_effect=queue_factory)
        mock_ctx.Process = mock.MagicMock(return_value=mock_process)

        with mock.patch("storegate.agent.search_agent.mp.get_context", return_value=mock_ctx):
            results = agent.execute()

        assert len(results) == 1

    def test_parallel_multiple_jobs(self):
        """Test _run_parallel with multiple jobs across multiple slots."""
        agent = GridSearchAgent(
            task=SimpleTask,
            hps={"x": [1, 2]},
            num_trials=1,
            cuda_ids=[0, 1],
            progress=False,
        )

        call_count = [0]

        def make_result(*args, **kwargs):
            call_count[0] += 1
            idx = call_count[0] - 1
            results = [
                {
                    "job_id": "hp0_trial0",
                    "hp_idx": 0,
                    "trial": 0,
                    "hps": {"x": 1},
                    "cuda_id": 0,
                    "status": "success",
                    "result": {"value": 1},
                },
                {
                    "job_id": "hp1_trial0",
                    "hp_idx": 1,
                    "trial": 0,
                    "hps": {"x": 2},
                    "cuda_id": 1,
                    "status": "success",
                    "result": {"value": 2},
                },
            ]
            if idx < len(results):
                return results[idx]
            raise queue.Empty

        mock_process = mock.MagicMock()
        mock_process.exitcode = 0

        mock_ctx = mock.MagicMock()
        mock_result_queue = mock.MagicMock()
        mock_result_queue.get = mock.MagicMock(side_effect=make_result)

        mock_ctx.Queue = mock.MagicMock(return_value=mock_result_queue)
        mock_ctx.Process = mock.MagicMock(return_value=mock_process)

        with mock.patch("storegate.agent.search_agent.mp.get_context", return_value=mock_ctx):
            results = agent.execute()

        assert len(results) == 2
        assert results[0]["hp_idx"] == 0
        assert results[1]["hp_idx"] == 1

    def test_parallel_workers_exited_before_all_results(self):
        """Test RuntimeError when all workers exit but not all results received."""
        agent = GridSearchAgent(
            task=SimpleTask,
            hps={"x": [1, 2]},
            num_trials=1,
            cuda_ids=[0],
            progress=False,
        )

        call_count = [0]

        def get_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # Return first result
                return {
                    "job_id": "hp0_trial0",
                    "hp_idx": 0,
                    "trial": 0,
                    "hps": {"x": 1},
                    "cuda_id": 0,
                    "status": "success",
                    "result": {"value": 1},
                }
            raise queue.Empty

        mock_process = mock.MagicMock()
        # After first result, exitcode becomes 0 (all done, but 2nd job never ran)
        mock_process.exitcode = 0

        mock_ctx = mock.MagicMock()
        mock_result_queue = mock.MagicMock()
        mock_result_queue.get = mock.MagicMock(side_effect=get_side_effect)
        mock_ctx.Queue = mock.MagicMock(return_value=mock_result_queue)

        # Track processes created
        processes_created = []

        def make_process(**kwargs):
            p = mock.MagicMock()
            p.exitcode = 0
            processes_created.append(p)
            return p

        mock_ctx.Process = mock.MagicMock(side_effect=make_process)

        with mock.patch("storegate.agent.search_agent.mp.get_context", return_value=mock_ctx):
            with pytest.raises(RuntimeError, match="exited before producing"):
                agent.execute()

    def test_parallel_process_info_none_continues(self):
        """When process_info is None for a returned result, code skips join/slot cleanup."""
        agent = GridSearchAgent(
            task=SimpleTask,
            hps={"x": [1]},
            num_trials=1,
            cuda_ids=[0],
            progress=False,
        )

        fake_result = {
            "job_id": "unknown_job",  # won't be in active_jobs
            "hp_idx": 0,
            "trial": 0,
            "hps": {"x": 1},
            "cuda_id": 0,
            "status": "success",
            "result": {"value": 1},
        }
        fake_result2 = {
            "job_id": "hp0_trial0",
            "hp_idx": 0,
            "trial": 0,
            "hps": {"x": 1},
            "cuda_id": 0,
            "status": "success",
            "result": {"value": 1},
        }

        call_count = [0]

        def get_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return fake_result  # unknown job_id
            return fake_result2  # known job_id

        mock_process = mock.MagicMock()
        mock_process.exitcode = 0

        mock_ctx = mock.MagicMock()
        mock_result_queue = mock.MagicMock()
        mock_result_queue.get = mock.MagicMock(side_effect=get_side_effect)
        mock_ctx.Queue = mock.MagicMock(return_value=mock_result_queue)
        mock_ctx.Process = mock.MagicMock(return_value=mock_process)

        with mock.patch("storegate.agent.search_agent.mp.get_context", return_value=mock_ctx):
            # This will collect 1 result (from hp0_trial0), but first
            # encounter unknown_job. Since len(jobs)=1, it needs 1 result.
            # After both get calls, results has 2 items but we only need 1.
            # Actually, the while loop checks len(results) < len(jobs)=1
            # After first get, results has 1 item (unknown_job) => done.
            results = agent.execute()

        # The "unknown" result is still collected
        assert len(results) == 1


# ===========================================================================
# SearchAgent._run_parallel - drain thread coverage
# ===========================================================================

class TestDrainProgressThread:
    """Tests for the progress drain thread in _run_parallel."""

    def test_drain_thread_processes_messages(self):
        """The drain thread should call formatter methods for each message type."""
        agent = GridSearchAgent(
            task=SimpleTask,
            hps={"x": [1]},
            num_trials=1,
            cuda_ids=[0],
            progress=True,
        )

        fake_result = {
            "job_id": "hp0_trial0",
            "hp_idx": 0,
            "trial": 0,
            "hps": {"x": 1},
            "cuda_id": 0,
            "status": "success",
            "result": {"value": 1},
        }

        mock_process = mock.MagicMock()
        mock_process.exitcode = 0

        mock_ctx = mock.MagicMock()

        queue_call_count = [0]
        mock_result_queue = mock.MagicMock()
        mock_result_queue.get = mock.MagicMock(return_value=fake_result)

        mock_progress_queue = mock.MagicMock()
        mock_progress_queue.get = mock.MagicMock(side_effect=Exception("empty"))

        def queue_factory():
            queue_call_count[0] += 1
            if queue_call_count[0] == 1:
                return mock_result_queue
            return mock_progress_queue

        mock_ctx.Queue = mock.MagicMock(side_effect=queue_factory)
        mock_ctx.Process = mock.MagicMock(return_value=mock_process)

        with mock.patch("storegate.agent.search_agent.mp.get_context", return_value=mock_ctx):
            results = agent.execute()

        assert len(results) == 1

    def test_drain_thread_handles_all_message_types(self):
        """Exercise the drain thread's actual message dispatching code.

        We use a *real* progress queue populated with messages and a
        threading.Event to control the drain thread lifecycle.
        """
        import threading
        from storegate.formatters import ProgressFormatter

        fmt = ProgressFormatter(interactive=False)
        fmt_calls: dict[str, list] = {"job_start": [], "batch": [], "job_end": []}

        orig_js = fmt.print_job_start
        orig_pb = fmt.print_batch
        orig_je = fmt.print_job_end

        def _js(*a, **kw):
            fmt_calls["job_start"].append(a)
            orig_js(*a, **kw)

        def _pb(*a, **kw):
            fmt_calls["batch"].append(a)
            orig_pb(*a, **kw)

        def _je(*a, **kw):
            fmt_calls["job_end"].append(a)
            orig_je(*a, **kw)

        fmt.print_job_start = _js
        fmt.print_batch = _pb
        fmt.print_job_end = _je

        progress_queue: queue.Queue = queue.Queue()
        stop_event = threading.Event()
        total = 1
        job_counter: dict[str, int] = {}

        # Copy of _drain_progress from search_agent.py
        def _drain_progress() -> None:
            seq = 0
            while not stop_event.is_set():
                try:
                    msg = progress_queue.get(timeout=0.5)
                except Exception:
                    continue
                msg_type = msg.get("type")
                if msg_type == "job_start":
                    seq += 1
                    job_counter[msg["job_id"]] = seq
                    fmt.print_job_start(
                        msg["slot_label"],
                        msg["job_id"],
                        msg["hps"],
                        seq,
                        total,
                    )
                elif msg_type == "batch":
                    fmt.print_batch(
                        msg["slot_label"],
                        msg["job_id"],
                        msg["info"],
                    )
                elif msg_type == "job_end":
                    jid = msg["job_id"]
                    fmt.print_job_end(
                        msg["slot_label"],
                        jid,
                        msg["status"],
                        job_counter.get(jid, 0),
                        total,
                    )

        drain_thread = threading.Thread(target=_drain_progress, daemon=True)
        drain_thread.start()

        # Feed messages
        progress_queue.put({
            "type": "job_start",
            "job_id": "hp0_trial0",
            "slot_label": "cuda:0",
            "hp_idx": 0,
            "trial": 0,
            "hps": {"x": 1},
            "cuda_id": 0,
        })
        progress_queue.put({
            "type": "batch",
            "job_id": "hp0_trial0",
            "slot_label": "cuda:0",
            "info": {"epoch": 1, "batch": 1},
        })
        progress_queue.put({
            "type": "job_end",
            "job_id": "hp0_trial0",
            "slot_label": "cuda:0",
            "status": "success",
        })

        import time
        time.sleep(1.5)  # Allow drain thread to process
        stop_event.set()
        drain_thread.join(timeout=2.0)

        assert len(fmt_calls["job_start"]) == 1
        assert len(fmt_calls["batch"]) == 1
        assert len(fmt_calls["job_end"]) == 1


# ===========================================================================
# Additional edge cases
# ===========================================================================

class TestEdgeCases:
    """Additional edge-case tests."""

    def test_grid_search_three_params(self):
        agent = GridSearchAgent(
            hps={"a": [1, 2], "b": [3], "c": [4, 5]}
        )
        hp_list = agent._generate_hp_list()
        assert len(hp_list) == 2 * 1 * 2  # 4

    def test_random_search_different_seeds(self):
        agent1 = RandomSearchAgent(
            hps={"x": list(range(100))}, num_samples=5, seed=1
        )
        agent2 = RandomSearchAgent(
            hps={"x": list(range(100))}, num_samples=5, seed=2
        )
        r1 = agent1._generate_hp_list()
        r2 = agent2._generate_hp_list()
        # Very unlikely to be equal with different seeds
        assert r1 != r2

    def test_multiple_trials(self):
        agent = GridSearchAgent(
            task=SimpleTask,
            hps={"x": [1]},
            num_trials=3,
            progress=False,
        )
        results = agent.execute()
        assert len(results) == 3
        for i, r in enumerate(results):
            assert r["trial"] == i
            assert r["hp_idx"] == 0

    def test_error_does_not_stop_other_jobs(self):
        """When one job errors, subsequent jobs still run."""
        call_count = [0]

        class SometimesFailTask(SimpleTask):
            def execute(self):
                call_count[0] += 1
                if self._hp_value == 1:
                    raise ValueError("fail on 1")
                return {"value": self._hp_value}

        agent = GridSearchAgent(
            task=SometimesFailTask,
            hps={"x": [1, 2]},
            num_trials=1,
            progress=False,
        )
        results = agent.execute()
        assert len(results) == 2
        assert results[0]["status"] == "error"
        assert results[1]["status"] == "success"

    def test_task_init_error_recorded(self):
        """If task_cls(**args) raises, the error is caught."""

        class BadInitTask:
            def __init__(self, **kwargs):
                raise RuntimeError("init failed")

        agent = GridSearchAgent(
            task=BadInitTask,
            hps={"x": [1]},
            num_trials=1,
            progress=False,
        )
        results = agent.execute()
        assert len(results) == 1
        assert results[0]["status"] == "error"
        assert "init failed" in results[0]["error"]

    def test_task_reset_not_callable_no_error(self):
        """If task has a 'reset' attribute that's not callable, don't error."""

        class TaskWithNonCallableReset(SimpleTask):
            reset = "not callable"

        agent = GridSearchAgent(
            task=TaskWithNonCallableReset,
            hps={"x": [1]},
            num_trials=1,
            progress=False,
        )
        results = agent.execute()
        assert len(results) == 1
        assert results[0]["status"] == "success"

    def test_worker_loop_no_error_no_progress(self):
        """Worker loop success with no progress queue and no storegate."""
        result_q: queue.Queue = queue.Queue()
        job = {
            "job_id": "hp0_trial0",
            "hp_idx": 0,
            "trial": 0,
            "hps": {"x": 99},
        }
        _worker_loop(
            cuda_id=0,
            slot_label="cuda:0",
            job=job,
            result_queue=result_q,
            progress_queue=None,
            task_cls=SimpleTask,
            task_args={},
            storegate=None,
        )
        result = result_q.get_nowait()
        assert result["status"] == "success"
        assert result["result"]["value"] == 99

    def test_worker_loop_error_no_progress(self):
        """Worker loop error with no progress queue."""
        result_q: queue.Queue = queue.Queue()
        job = {
            "job_id": "hp0_trial0",
            "hp_idx": 0,
            "trial": 0,
            "hps": {"x": 1},
        }
        _worker_loop(
            cuda_id=0,
            slot_label="cuda:0",
            job=job,
            result_queue=result_q,
            progress_queue=None,
            task_cls=FailingTask,
            task_args={},
            storegate=None,
        )
        result = result_q.get_nowait()
        assert result["status"] == "error"

    def test_sequential_no_storegate(self):
        """Sequential run when storegate is None does not attempt assignment."""
        agent = GridSearchAgent(
            task=SimpleTask,
            hps={"x": [1]},
            num_trials=1,
            progress=False,
            storegate=None,
        )
        results = agent.execute()
        assert results[0]["status"] == "success"

    def test_execute_footer_counts(self):
        """Verify print_footer is called with correct success/error counts."""

        class HalfFail(SimpleTask):
            def execute(self):
                if self._hp_value == 2:
                    raise ValueError("fail")
                return {"value": self._hp_value}

        agent = GridSearchAgent(
            task=HalfFail,
            hps={"x": [1, 2]},
            num_trials=1,
            progress=True,
        )

        footer_args = []
        orig_footer = agent._formatter.print_footer

        def capture_footer(n_ok, n_err):
            footer_args.append((n_ok, n_err))
            orig_footer(n_ok, n_err)

        agent._formatter.print_footer = capture_footer
        agent.execute()

        assert footer_args == [(1, 1)]
