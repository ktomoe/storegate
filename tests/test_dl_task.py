"""Tests for storegate.task.task and storegate.task.dl_task."""

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from storegate.database.numpy_database import NumpyDatabase
from storegate.storegate import StoreGate
from storegate.task.dl_env import DLEnv, LearningMode
from storegate.task.dl_task import DLTask
from storegate.task.pytorch_metrics import BUILTIN_METRICS, ResolvedMetric, make_key_lookup
from storegate.task.task import Task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_storegate(
    *,
    data_id: str = "d1",
    phases: dict[str, dict[str, np.ndarray]] | None = None,
) -> StoreGate:
    """Create a StoreGate with NumpyDatabase, data, and compile it."""
    sg = StoreGate(backends={"numpy": NumpyDatabase()}, backend="numpy", data_id=data_id)
    sg.initialize(data_id)
    if phases:
        for phase, var_dict in phases.items():
            for var_name, arr in var_dict.items():
                sg.add_data(phase, var_name, arr)
    sg.compile()
    return sg


class ConcreteDLTask(DLTask):
    """Minimal concrete DLTask for testing."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._compiled_device = False
        self._compiled_model = False
        self._compiled_optimizer = False
        self._compiled_loss = False
        self._fit_called = False
        self._predict_called = False

    def _compile_device(self) -> None:
        self._dl.device = "cpu"
        self._compiled_device = True

    def _compile_model(self) -> None:
        self._dl.model = "mock_model"
        self._compiled_model = True

    def _compile_optimizer(self) -> None:
        self._dl.optimizer = "mock_opt"
        self._compiled_optimizer = True

    def _compile_loss(self) -> None:
        self._dl.loss = "mock_loss"
        self._compiled_loss = True

    def fit(self) -> dict[str, Any]:
        self._fit_called = True
        return {"fit": "done"}

    def predict(self) -> dict[str, Any]:
        self._predict_called = True
        return {"predict": "done"}


# ---------------------------------------------------------------------------
# Task base class
# ---------------------------------------------------------------------------
class TestTask:
    def test_init(self):
        class ConcreteTask(Task):
            def execute(self):
                return "executed"

        t = ConcreteTask()
        assert t.storegate is None
        assert t._progress_callback is None

    def test_storegate_getter_setter(self):
        class ConcreteTask(Task):
            def execute(self):
                pass

        sg = _make_storegate()
        t = ConcreteTask()
        t.storegate = sg
        assert t.storegate is sg

    def test_reset_no_op(self):
        class ConcreteTask(Task):
            def execute(self):
                pass

        t = ConcreteTask()
        t.reset()  # should be a no-op

    def test_execute_via_subclass(self):
        class ConcreteTask(Task):
            def execute(self):
                return 42

        t = ConcreteTask()
        assert t.execute() == 42


# ---------------------------------------------------------------------------
# DLTask init
# ---------------------------------------------------------------------------
class TestDLTaskInit:
    def test_init_with_all_params(self):
        task = ConcreteDLTask(
            data_id="d1",
            phases="train",
            var_names={"inputs": "x", "true": "y"},
            model="Linear",
            model_args={"a": 1},
            optimizer="SGD",
            optimizer_args={"lr": 0.01},
            loss="MSELoss",
            loss_args={},
            num_epochs=10,
            metrics=["acc"],
        )
        assert task._data_id == "d1"
        assert task._phases == "train"
        assert task._num_epochs == 10

    def test_init_defaults(self):
        task = ConcreteDLTask()
        assert task._data_id is None
        assert task._phases is None
        assert task._var_names is None
        assert task._model is None
        assert task._model_args == {}
        assert task._optimizer is None
        assert task._optimizer_args == {}
        assert task._loss is None
        assert task._loss_args == {}
        assert task._num_epochs == 0
        assert task._metrics is None

    def test_storegate_setter_invalidates_is_compiled(self):
        task = ConcreteDLTask()
        task._dl.is_compiled = True
        sg = _make_storegate()
        task.storegate = sg
        assert task._dl.is_compiled is False


# ---------------------------------------------------------------------------
# set_hps
# ---------------------------------------------------------------------------
class TestSetHps:
    def _make_task_with_sg(self, **kwargs: Any) -> ConcreteDLTask:
        task = ConcreteDLTask(**kwargs)
        task.storegate = _make_storegate()
        return task

    def test_flat_key(self):
        task = self._make_task_with_sg(num_epochs=5)
        task.set_hps({"num_epochs": 10})
        assert task._num_epochs == 10
        assert task._dl.is_compiled is False

    def test_namespaced_model(self):
        task = self._make_task_with_sg(model_args={"a": 1})
        task.set_hps({"model__b": 2})
        assert task._model_args["b"] == 2

    def test_namespaced_optimizer(self):
        task = self._make_task_with_sg()
        task.set_hps({"optimizer__lr": 0.01})
        assert task._optimizer_args["lr"] == 0.01

    def test_namespaced_loss(self):
        task = self._make_task_with_sg()
        task.set_hps({"loss__reduction": "sum"})
        assert task._loss_args["reduction"] == "sum"

    def test_empty_suffix_error(self):
        task = self._make_task_with_sg()
        with pytest.raises(ValueError, match="empty suffix"):
            task.set_hps({"model__": 1})

    def test_unknown_namespace_error(self):
        task = self._make_task_with_sg()
        with pytest.raises(ValueError, match="Unknown hyperparameter namespace"):
            task.set_hps({"unknown__key": 1})

    def test_protected_key_error(self):
        task = self._make_task_with_sg()
        with pytest.raises(AttributeError, match="not a valid hyperparameter"):
            task.set_hps({"storegate": "something"})

    def test_protected_key_dl(self):
        task = self._make_task_with_sg()
        with pytest.raises(AttributeError, match="not a valid hyperparameter"):
            task.set_hps({"dl": "something"})

    def test_undefined_attribute_error(self):
        task = self._make_task_with_sg()
        with pytest.raises(AttributeError, match="not defined"):
            task.set_hps({"nonexistent": 1})

    def test_no_storegate_error(self):
        task = ConcreteDLTask()
        with pytest.raises(RuntimeError, match="storegate is not set"):
            task.set_hps({"num_epochs": 10})


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------
class TestExecute:
    def test_with_train_and_test(self):
        task = ConcreteDLTask(
            phases={"train": "phase1", "test": "phase1"},
        )
        sg = _make_storegate(phases={"phase1": {"x": np.array([1.0, 2.0])}})
        task.storegate = sg
        result = task.execute()
        assert result["fit"] == {"fit": "done"}
        assert result["predict"] == {"predict": "done"}
        assert task._fit_called
        assert task._predict_called

    def test_without_train(self):
        task = ConcreteDLTask(
            phases={"test": "phase1"},
        )
        sg = _make_storegate(phases={"phase1": {"x": np.array([1.0, 2.0])}})
        task.storegate = sg
        result = task.execute()
        assert result["fit"] is None
        assert result["predict"] == {"predict": "done"}

    def test_without_test(self):
        task = ConcreteDLTask(
            phases={"train": "phase1"},
        )
        sg = _make_storegate(phases={"phase1": {"x": np.array([1.0, 2.0])}})
        task.storegate = sg
        result = task.execute()
        assert result["fit"] == {"fit": "done"}
        assert result["predict"] is None


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------
class TestReset:
    def test_clears_dl_env(self):
        task = ConcreteDLTask()
        task._dl.model = "something"
        task._dl.is_compiled = True
        task.reset()
        assert task._dl.model is None
        assert task._dl.is_compiled is False


# ---------------------------------------------------------------------------
# compile
# ---------------------------------------------------------------------------
class TestCompile:
    def test_without_storegate_error(self):
        task = ConcreteDLTask()
        with pytest.raises(RuntimeError, match="storegate is not set"):
            task.compile()

    def test_phases_as_str(self):
        task = ConcreteDLTask(phases="phase1")
        sg = _make_storegate(phases={"phase1": {"x": np.array([1.0])}})
        task.storegate = sg
        task.compile()
        assert task._dl.phase.train == "phase1"
        assert task._dl.phase.valid == "phase1"
        assert task._dl.phase.test == "phase1"
        assert task._dl.is_compiled is True

    def test_phases_as_dict(self):
        task = ConcreteDLTask(phases={"train": "t", "test": "te"})
        sg = _make_storegate(
            phases={
                "t": {"x": np.array([1.0])},
                "te": {"x": np.array([1.0])},
            }
        )
        task.storegate = sg
        task.compile()
        assert task._dl.phase.train == "t"
        assert task._dl.phase.valid is None
        assert task._dl.phase.test == "te"

    def test_phases_invalid_type(self):
        task = ConcreteDLTask(phases=123)
        task.storegate = _make_storegate()
        with pytest.raises(TypeError, match="phases must be a str or dict"):
            task.compile()

    def test_phases_invalid_keys(self):
        task = ConcreteDLTask(phases={"train": "t", "bad_key": "b"})
        task.storegate = _make_storegate()
        with pytest.raises(ValueError, match="Invalid phase keys"):
            task.compile()

    def test_var_names_as_dict_valid(self):
        task = ConcreteDLTask(
            phases="phase1",
            var_names={"inputs": "x", "true": "y"},
        )
        sg = _make_storegate(
            phases={"phase1": {"x": np.array([1.0, 2.0]), "y": np.array([3.0, 4.0])}}
        )
        task.storegate = sg
        task.compile()
        assert task._dl.var_name.inputs == ["x"]
        assert task._dl.var_name.true == ["y"]

    def test_var_names_as_dict_invalid_keys(self):
        task = ConcreteDLTask(
            phases="phase1",
            var_names={"inputs": "x", "bad": "z"},
        )
        task.storegate = _make_storegate()
        with pytest.raises(ValueError, match="Invalid var_name keys"):
            task.compile()

    def test_var_names_as_non_dict_error(self):
        task = ConcreteDLTask(phases="phase1", var_names="not_a_dict")
        task.storegate = _make_storegate()
        with pytest.raises(TypeError, match="var_names must be a dict"):
            task.compile()

    def test_phases_none_no_op(self):
        task = ConcreteDLTask()
        task.storegate = _make_storegate()
        task.compile()
        assert task._dl.phase.train is None


# ---------------------------------------------------------------------------
# _compile_metrics
# ---------------------------------------------------------------------------
class TestCompileMetrics:
    def test_list(self):
        task = ConcreteDLTask(metrics=["acc"])
        task.storegate = _make_storegate()
        task.compile()
        assert len(task._dl.metrics.train) == 1
        assert task._dl.metrics.train[0].name == "acc"
        # list metrics are shared across all phases
        assert len(task._dl.metrics.valid) == 1
        assert len(task._dl.metrics.test) == 1

    def test_dict(self):
        task = ConcreteDLTask(metrics={"train": ["acc"], "test": ["binary_acc"]})
        task.storegate = _make_storegate()
        task.compile()
        assert len(task._dl.metrics.train) == 1
        assert task._dl.metrics.train[0].name == "acc"
        assert len(task._dl.metrics.test) == 1
        assert task._dl.metrics.test[0].name == "binary_acc"
        assert task._dl.metrics.valid == []

    def test_dict_invalid_keys(self):
        task = ConcreteDLTask(metrics={"train": ["acc"], "bad": ["acc"]})
        task.storegate = _make_storegate()
        with pytest.raises(ValueError, match="Invalid metric keys"):
            task.compile()

    def test_none(self):
        task = ConcreteDLTask(metrics=None)
        task.storegate = _make_storegate()
        task.compile()
        assert task._dl.metrics.train == []

    def test_non_list_value_in_dict(self):
        task = ConcreteDLTask(metrics={"train": "acc"})
        task.storegate = _make_storegate()
        with pytest.raises(TypeError, match="metrics\\['train'\\] must be a list"):
            task.compile()

    def test_non_list_dict_none_type_error(self):
        task = ConcreteDLTask(metrics=42)
        task.storegate = _make_storegate()
        with pytest.raises(TypeError, match="metrics must be a list, dict, or None"):
            task.compile()


# ---------------------------------------------------------------------------
# _resolve_metrics
# ---------------------------------------------------------------------------
class TestResolveMetrics:
    def _make_task(self) -> ConcreteDLTask:
        task = ConcreteDLTask()
        task.storegate = _make_storegate()
        return task

    def test_str_builtin_acc(self):
        task = self._make_task()
        resolved = task._resolve_metrics(["acc"])
        assert len(resolved) == 1
        assert resolved[0].name == "acc"
        assert resolved[0].requires_labels is True

    def test_str_builtin_binary_acc(self):
        task = self._make_task()
        resolved = task._resolve_metrics(["binary_acc"])
        assert resolved[0].requires_labels is True

    def test_str_builtin_multi_acc(self):
        task = self._make_task()
        resolved = task._resolve_metrics(["multi_acc"])
        assert resolved[0].requires_labels is True

    def test_str_non_builtin(self):
        task = self._make_task()
        resolved = task._resolve_metrics(["custom_key"])
        assert len(resolved) == 1
        assert resolved[0].name == "custom_key"
        # Test the lookup function
        result = resolved[0].fn({"custom_key": 42})
        assert result == 42

    def test_callable(self):
        task = self._make_task()

        def my_metric(br):
            return br.get("val", 0)

        resolved = task._resolve_metrics([my_metric])
        assert resolved[0].name == "my_metric"
        assert resolved[0].fn is my_metric

    def test_invalid_type(self):
        task = self._make_task()
        with pytest.raises(TypeError, match="metric must be a str or callable"):
            task._resolve_metrics([123])

    def test_non_list_raises_without_phase(self):
        task = self._make_task()
        with pytest.raises(TypeError, match="metrics must be a list"):
            task._resolve_metrics("not_a_list")

    def test_non_list_raises_with_phase(self):
        task = self._make_task()
        with pytest.raises(TypeError, match="metrics\\['train'\\] must be a list"):
            task._resolve_metrics("not_a_list", phase="train")


# ---------------------------------------------------------------------------
# _normalize_var_name
# ---------------------------------------------------------------------------
class TestNormalizeVarName:
    def test_none(self):
        assert DLTask._normalize_var_name(None) is None

    def test_str(self):
        assert DLTask._normalize_var_name("x") == ["x"]

    def test_list(self):
        assert DLTask._normalize_var_name(["x", "y"]) == ["x", "y"]

    def test_list_with_non_str(self):
        with pytest.raises(TypeError, match="must be str"):
            DLTask._normalize_var_name(["x", 123])

    def test_list_with_duplicates(self):
        with pytest.raises(ValueError, match="duplicate"):
            DLTask._normalize_var_name(["x", "x"])

    def test_invalid_type(self):
        with pytest.raises(TypeError, match="var_name value must be"):
            DLTask._normalize_var_name(123)


# ---------------------------------------------------------------------------
# _bind_data_id
# ---------------------------------------------------------------------------
class TestBindDataId:
    def test_with_data_id(self):
        task = ConcreteDLTask(data_id="custom_id")
        sg = _make_storegate(data_id="custom_id")
        task.storegate = sg
        task._bind_data_id()
        assert sg.get_data_id() == "custom_id"

    def test_without_data_id(self):
        task = ConcreteDLTask(data_id=None)
        sg = _make_storegate()
        task.storegate = sg
        task._bind_data_id()  # should be no-op


# ---------------------------------------------------------------------------
# _get_hps_namespace_targets
# ---------------------------------------------------------------------------
class TestGetHpsNamespaceTargets:
    def test_returns_correct_dict(self):
        task = ConcreteDLTask(
            model_args={"a": 1},
            optimizer_args={"b": 2},
            loss_args={"c": 3},
        )
        targets = task._get_hps_namespace_targets()
        assert targets["model"] is task._model_args
        assert targets["optimizer"] is task._optimizer_args
        assert targets["loss"] is task._loss_args


# ---------------------------------------------------------------------------
# _compile_phases
# ---------------------------------------------------------------------------
class TestCompilePhases:
    def test_none_no_op(self):
        task = ConcreteDLTask(phases=None)
        task._compile_phases()
        assert task._dl.phase.train is None

    def test_str(self):
        task = ConcreteDLTask(phases="all_phase")
        task._compile_phases()
        assert task._dl.phase.train == "all_phase"
        assert task._dl.phase.valid == "all_phase"
        assert task._dl.phase.test == "all_phase"

    def test_dict_partial_keys(self):
        task = ConcreteDLTask(phases={"train": "tr"})
        task._compile_phases()
        assert task._dl.phase.train == "tr"
        assert task._dl.phase.valid is None
        assert task._dl.phase.test is None

    def test_invalid_type(self):
        task = ConcreteDLTask(phases=42)
        with pytest.raises(TypeError, match="phases must be a str or dict"):
            task._compile_phases()

    def test_invalid_keys(self):
        task = ConcreteDLTask(phases={"train": "tr", "bad": "b"})
        with pytest.raises(ValueError, match="Invalid phase keys"):
            task._compile_phases()


# ---------------------------------------------------------------------------
# _on_batch_end
# ---------------------------------------------------------------------------
class TestOnBatchEnd:
    def test_with_callback(self):
        task = ConcreteDLTask()
        called_with = {}
        task._progress_callback = lambda info: called_with.update(info)
        task._on_batch_end({"epoch": 1})
        assert called_with["epoch"] == 1

    def test_without_callback(self):
        task = ConcreteDLTask()
        task._on_batch_end({"epoch": 1})  # should not raise


# ---------------------------------------------------------------------------
# _resolve_storegate
# ---------------------------------------------------------------------------
class TestResolveStoregate:
    def test_no_storegate_raises(self):
        task = ConcreteDLTask()
        with pytest.raises(RuntimeError, match="storegate is not set"):
            task._resolve_storegate()

    def test_with_storegate(self):
        task = ConcreteDLTask()
        sg = _make_storegate()
        task.storegate = sg
        assert task._resolve_storegate() is sg


# ---------------------------------------------------------------------------
# _require_compiled
# ---------------------------------------------------------------------------
class TestRequireCompiled:
    def test_delegates_to_dl(self):
        task = ConcreteDLTask()
        with pytest.raises(RuntimeError, match="compile"):
            task._require_compiled("fit")

    def test_passes_when_compiled(self):
        task = ConcreteDLTask()
        task._dl.is_compiled = True
        task._require_compiled("fit")  # should not raise
