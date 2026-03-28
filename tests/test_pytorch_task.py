"""Tests for storegate.task.pytorch_task."""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset

from storegate.database.numpy_database import NumpyDatabase
from storegate.storegate import StoreGate
from storegate.task.dl_env import LearningMode
from storegate.task.pytorch_task import (
    PytorchTask,
    SelfSupervisedPytorchTask,
    UnsupervisedPytorchTask,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_storegate(
    *,
    data_id: str = "d1",
    phases: dict[str, dict[str, np.ndarray]] | None = None,
) -> StoreGate:
    sg = StoreGate(
        backends={"numpy": NumpyDatabase()},
        backend="numpy",
        data_id=data_id,
    )
    sg.initialize(data_id)
    if phases:
        for phase, var_dict in phases.items():
            for var_name, arr in var_dict.items():
                sg.add_data(phase, var_name, arr)
    sg.compile()
    return sg


def _sample_data(n: int = 20, in_features: int = 4, out_features: int = 2) -> dict[str, dict[str, np.ndarray]]:
    rng = np.random.default_rng(42)
    x = rng.standard_normal((n, in_features)).astype(np.float32)
    y = rng.standard_normal((n, out_features)).astype(np.float32)
    return {"train": {"x": x, "y": y}, "test": {"x": x, "y": y}}


def _make_task(
    *,
    task_class: type = PytorchTask,
    in_features: int = 4,
    out_features: int = 2,
    n: int = 20,
    phases: str | dict[str, str] = {"train": "train", "test": "test"},
    var_names: dict[str, Any] | None = None,
    num_epochs: int = 1,
    extra_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, StoreGate]:
    if var_names is None:
        var_names = {"inputs": "x", "true": "y"}

    data = _sample_data(n, in_features, out_features)

    sg = _make_storegate(phases=data)

    kwargs: dict[str, Any] = {
        "phases": phases,
        "var_names": var_names,
        "model": nn.Linear,
        "model_args": {"in_features": in_features, "out_features": out_features},
        "optimizer": optim.SGD,
        "optimizer_args": {"lr": 0.01},
        "loss": nn.MSELoss,
        "num_epochs": num_epochs,
        "device": "cpu",
    }
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    task = task_class(**kwargs)
    task.storegate = sg
    return task, sg


# ---------------------------------------------------------------------------
# PytorchTask init
# ---------------------------------------------------------------------------
class TestPytorchTaskInit:
    def test_default_device_auto(self):
        task = PytorchTask(
            model=nn.Linear,
            model_args={"in_features": 2, "out_features": 1},
            device="auto",
        )
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert task._device == expected

    def test_explicit_device(self):
        task = PytorchTask(
            model=nn.Linear,
            model_args={"in_features": 2, "out_features": 1},
            device="cpu",
        )
        assert task._device == "cpu"

    def test_metric_interval_lt1_error(self):
        with pytest.raises(ValueError, match="metric_interval must be >= 1"):
            PytorchTask(
                model=nn.Linear,
                model_args={"in_features": 2, "out_features": 1},
                metric_interval=0,
            )

    def test_default_args(self):
        task = PytorchTask(
            model=nn.Linear,
            model_args={"in_features": 2, "out_features": 1},
            device="cpu",
        )
        assert task._dataset_args == {}
        assert task._dataloader_args == {}
        assert task._non_blocking is False
        assert task._progress_bar is False
        assert task._metric_interval == 1
        assert task._torchinfo is False
        assert task._torch_compile is False
        assert task._cuda_id is None


# ---------------------------------------------------------------------------
# _compile_device
# ---------------------------------------------------------------------------
class TestCompileDevice:
    def test_auto_cpu(self):
        task, _ = _make_task()
        task.compile()
        assert task._dl.device == torch.device("cpu")

    def test_explicit_device(self):
        task, _ = _make_task()
        task._device = "cpu"
        task.compile()
        assert task._dl.device == torch.device("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_with_cuda_id(self):
        task, _ = _make_task()
        task.set_cuda_id(0)
        task.compile()
        assert task._dl.device == torch.device("cuda:0")


# ---------------------------------------------------------------------------
# _compile_model
# ---------------------------------------------------------------------------
class TestCompileModel:
    def test_from_class(self):
        task, _ = _make_task()
        task.compile()
        assert isinstance(task._dl.model, nn.Linear)

    def test_from_string(self):
        task, _ = _make_task()
        task._model = "Linear"
        task.compile()
        assert isinstance(task._dl.model, nn.Linear)

    def test_from_instance_optimizer_error(self):
        """Passing an Optimizer instance is tested separately in _compile_optimizer."""
        pass

    def test_none_model_error(self):
        task, _ = _make_task()
        task._model = None
        with pytest.raises(ValueError, match="model is required"):
            task.compile()

    def test_torch_compile(self):
        task, _ = _make_task(extra_kwargs={"torch_compile": True})
        task.compile()
        # torch.compile wraps the model
        assert task._dl.model is not None


# ---------------------------------------------------------------------------
# _compile_optimizer
# ---------------------------------------------------------------------------
class TestCompileOptimizer:
    def test_from_class(self):
        task, _ = _make_task()
        task.compile()
        assert isinstance(task._dl.optimizer, optim.SGD)

    def test_from_string(self):
        task, _ = _make_task()
        task._optimizer = "SGD"
        task.compile()
        assert isinstance(task._dl.optimizer, optim.SGD)

    def test_none_skip(self):
        task, _ = _make_task()
        task._optimizer = None
        task.compile()
        assert task._dl.optimizer is None

    def test_instance_error(self):
        task, _ = _make_task()
        # Create an actual optimizer instance
        dummy_model = nn.Linear(4, 2)
        task._optimizer = optim.SGD(dummy_model.parameters(), lr=0.01)
        with pytest.raises(ValueError, match="optimizer instance is not supported"):
            task.compile()


# ---------------------------------------------------------------------------
# _compile_loss
# ---------------------------------------------------------------------------
class TestCompileLoss:
    def test_from_class(self):
        task, _ = _make_task()
        task.compile()
        assert isinstance(task._dl.loss, nn.MSELoss)

    def test_from_string(self):
        task, _ = _make_task()
        task._loss = "MSELoss"
        task.compile()
        assert isinstance(task._dl.loss, nn.MSELoss)

    def test_none_skip(self):
        task, _ = _make_task()
        task._loss = None
        task.compile()
        assert task._dl.loss is None


# ---------------------------------------------------------------------------
# _compile_dataset
# ---------------------------------------------------------------------------
class TestCompileDataset:
    def test_default_storegate_dataset(self):
        task, _ = _make_task()
        task.compile()
        from storegate.task.dataset.storegate_dataset import StoreGateDataset

        assert isinstance(task._dl.dataset.train, StoreGateDataset)
        assert isinstance(task._dl.dataset.test, StoreGateDataset)

    def test_custom_dataset(self):
        class KwargTensorDataset(TensorDataset):
            def __init__(self, tensors):
                super().__init__(*tensors)

        task, _ = _make_task()
        task._dataset = KwargTensorDataset
        task._dataset_args = {
            "tensors": (
                torch.randn(20, 4),
                torch.randn(20, 2),
            )
        }
        task.compile()
        assert isinstance(task._dl.dataset.train, KwargTensorDataset)

    def test_per_phase_dataset_args(self):
        task, _ = _make_task()
        task._dataset_args = {
            "train": {"preload": True},
            "test": {"preload": False},
        }
        task.compile()
        # Should compile without error using per-phase args


# ---------------------------------------------------------------------------
# _compile_dataloader
# ---------------------------------------------------------------------------
class TestCompileDataloader:
    def test_default_dataloader(self):
        task, _ = _make_task()
        task.compile()
        assert isinstance(task._dl.dataloader.train, TorchDataLoader)

    def test_custom_dataloader(self):
        task, _ = _make_task()
        task._dataloader = TorchDataLoader
        task._dataloader_args = {"batch_size": 4}
        task.compile()
        assert isinstance(task._dl.dataloader.train, TorchDataLoader)

    def test_per_phase_dataloader_args(self):
        task, _ = _make_task()
        task._dataloader_args = {
            "train": {"batch_size": 8},
            "test": {"batch_size": 4},
        }
        task.compile()


# ---------------------------------------------------------------------------
# _validate_dataloader_args_for_dataset
# ---------------------------------------------------------------------------
class TestValidateDataloaderArgsForDataset:
    def test_gpu_preload_with_num_workers_error(self):
        task, _ = _make_task()
        ds = MagicMock()
        ds.preload_mode = "gpu"
        with pytest.raises(ValueError, match="num_workers.*must be 0"):
            task._validate_dataloader_args_for_dataset("train", ds, {"num_workers": 2})

    def test_gpu_preload_with_pin_memory_error(self):
        task, _ = _make_task()
        ds = MagicMock()
        ds.preload_mode = "gpu"
        with pytest.raises(ValueError, match="pin_memory.*must be False"):
            task._validate_dataloader_args_for_dataset("train", ds, {"pin_memory": True})

    def test_non_gpu_preload_skip(self):
        task, _ = _make_task()
        ds = MagicMock()
        ds.preload_mode = "cpu"
        # Should not raise
        task._validate_dataloader_args_for_dataset("train", ds, {"num_workers": 4, "pin_memory": True})

    def test_no_preload_mode_attr(self):
        task, _ = _make_task()
        ds = MagicMock(spec=[])  # no preload_mode attr
        task._validate_dataloader_args_for_dataset("train", ds, {"num_workers": 4})

    def test_gpu_preload_with_num_workers_0_ok(self):
        task, _ = _make_task()
        ds = MagicMock()
        ds.preload_mode = "gpu"
        task._validate_dataloader_args_for_dataset("train", ds, {"num_workers": 0})


# ---------------------------------------------------------------------------
# _phase_args
# ---------------------------------------------------------------------------
class TestPhaseArgs:
    def test_flat_args(self):
        task, _ = _make_task()
        args = {"batch_size": 4}
        assert task._phase_args(args, "train") is args

    def test_per_phase_dict_args(self):
        task, _ = _make_task()
        args = {"train": {"batch_size": 8}, "test": {"batch_size": 4}}
        assert task._phase_args(args, "train") == {"batch_size": 8}
        assert task._phase_args(args, "valid") == {}

    def test_mixed_dict_not_all_phase_keys(self):
        task, _ = _make_task()
        # Has non-phase keys -> treated as flat
        args = {"batch_size": 8, "shuffle": True}
        assert task._phase_args(args, "train") is args

    def test_dict_with_non_dict_values(self):
        task, _ = _make_task()
        # Keys are phase keys but values are not dicts -> treated as flat
        args = {"train": 8}
        assert task._phase_args(args, "train") is args


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------
class TestFit:
    def test_basic_training(self):
        task, _ = _make_task(num_epochs=2, extra_kwargs={"dataloader_args": {"batch_size": 10}})
        task.compile()
        result = task.fit()
        assert "train" in result
        assert len(result["train"]) == 2

    def test_num_epochs_0_returns_empty(self):
        task, _ = _make_task(num_epochs=0)
        task.compile()
        result = task.fit()
        assert result == {}

    def test_validation_phase(self):
        task, _ = _make_task(
            phases={"train": "train", "valid": "train", "test": "test"},
            num_epochs=1,
            extra_kwargs={"dataloader_args": {"batch_size": 20}},
        )
        task.compile()
        result = task.fit()
        assert "valid" in result
        assert len(result["valid"]) == 1

    def test_no_validation(self):
        task, _ = _make_task(
            phases={"train": "train"},
            num_epochs=1,
            extra_kwargs={"dataloader_args": {"batch_size": 20}},
        )
        task.compile()
        result = task.fit()
        assert result["valid"] == []

    def test_not_compiled_error(self):
        task, _ = _make_task()
        with pytest.raises(RuntimeError, match="compile"):
            task.fit()


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------
class TestPredict:
    def test_with_output_var_names(self):
        task, sg = _make_task(
            var_names={"inputs": "x", "true": "y", "outputs": ["pred"]},
            extra_kwargs={"dataloader_args": {"batch_size": 20}},
        )
        task.compile()
        result = task.predict()
        assert isinstance(result, dict)

    def test_without_output_var_names(self):
        task, sg = _make_task(
            var_names={"inputs": "x", "true": "y"},
            extra_kwargs={"dataloader_args": {"batch_size": 20}},
        )
        task.compile()
        result = task.predict()
        assert isinstance(result, dict)

    def test_output_count_mismatch_error(self):
        # outputs expects 2 but model outputs 1
        task, sg = _make_task(
            var_names={"inputs": "x", "true": "y", "outputs": ["pred1", "pred2"]},
            extra_kwargs={"dataloader_args": {"batch_size": 20}},
        )
        task.compile()
        with pytest.raises(ValueError, match="Model output count does not match"):
            task.predict()

    def test_not_compiled_error(self):
        task, _ = _make_task()
        with pytest.raises(RuntimeError, match="compile"):
            task.predict()


# ---------------------------------------------------------------------------
# _step_epoch
# ---------------------------------------------------------------------------
class TestStepEpoch:
    def test_batch_processing(self):
        task, _ = _make_task(extra_kwargs={"dataloader_args": {"batch_size": 10}})
        task.compile()
        result = task._step_epoch(1, "train")
        assert isinstance(result, dict)

    def test_progress_bar(self):
        task, _ = _make_task(extra_kwargs={"progress_bar": True, "dataloader_args": {"batch_size": 20}})
        task.compile()
        result = task._step_epoch(1, "train")
        assert isinstance(result, dict)

    def test_progress_callback(self):
        collected = []
        task, _ = _make_task(extra_kwargs={"dataloader_args": {"batch_size": 10}})
        task._progress_callback = lambda info: collected.append(info)
        task.compile()
        task._step_epoch(1, "train")
        assert len(collected) > 0
        assert "epoch" in collected[0]
        assert "phase" in collected[0]

    def test_progress_bar_epoch_0(self):
        """Epoch 0 should use a different desc format."""
        task, _ = _make_task(extra_kwargs={"progress_bar": True, "dataloader_args": {"batch_size": 20}})
        task.compile()
        task._step_epoch(0, "test")


# ---------------------------------------------------------------------------
# _should_sample_metrics
# ---------------------------------------------------------------------------
class TestShouldSampleMetrics:
    def test_first_batch(self):
        task, _ = _make_task()
        assert task._should_sample_metrics(1, 10) is True

    def test_last_batch(self):
        task, _ = _make_task()
        assert task._should_sample_metrics(10, 10) is True

    def test_interval_match(self):
        task, _ = _make_task(extra_kwargs={"metric_interval": 5})
        assert task._should_sample_metrics(5, 20) is True

    def test_interval_miss(self):
        task, _ = _make_task(extra_kwargs={"metric_interval": 5})
        assert task._should_sample_metrics(3, 20) is False

    def test_total_batches_none(self):
        task, _ = _make_task(extra_kwargs={"metric_interval": 5})
        assert task._should_sample_metrics(1, None) is True
        assert task._should_sample_metrics(2, None) is False
        assert task._should_sample_metrics(5, None) is True


# ---------------------------------------------------------------------------
# _step_batch
# ---------------------------------------------------------------------------
class TestStepBatch:
    def test_valid_tuple(self):
        task, _ = _make_task()
        task.compile()
        data = (torch.randn(4, 4), torch.randn(4, 2))
        result = task._step_batch(data, "train")
        assert "batch_size" in result
        assert "preds" in result
        assert "labels" in result
        assert "loss" in result

    def test_valid_list(self):
        task, _ = _make_task()
        task.compile()
        data = [torch.randn(4, 4), torch.randn(4, 2)]
        result = task._step_batch(data, "train")
        assert result["batch_size"] == 4

    def test_invalid_type(self):
        task, _ = _make_task()
        task.compile()
        with pytest.raises(ValueError, match="must be a 2-element tuple or list"):
            task._step_batch(torch.randn(4, 4), "train")

    def test_wrong_length(self):
        task, _ = _make_task()
        task.compile()
        with pytest.raises(ValueError, match="must be a 2-element"):
            task._step_batch((torch.randn(4, 4),), "train")

    def test_loss_none_skipped(self):
        task, _ = _make_task()
        task.compile()
        task._dl.loss = None
        data = (torch.randn(4, 4), torch.randn(4, 2))
        result = task._step_batch(data, "train")
        assert "loss" not in result

    def test_train_phase_optimizer_step(self):
        task, _ = _make_task()
        task.compile()
        data = (torch.randn(4, 4), torch.randn(4, 2))
        result = task._step_batch(data, "train")
        assert "loss" in result

    def test_non_train_phase_skip_optimizer(self):
        task, _ = _make_task()
        task.compile()
        data = (torch.randn(4, 4), torch.randn(4, 2))
        result = task._step_batch(data, "valid")
        assert "loss" in result

    def test_outputs_as_list(self):
        """Test model that returns a list of outputs."""
        task, _ = _make_task()
        task.compile()
        task._dl.loss = None

        # Wrap model to return list
        orig_model = task._dl.model

        class ListModel(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base

            def forward(self, x):
                return [self.base(x)]

        task._dl.model = ListModel(orig_model)
        data = (torch.randn(4, 4), torch.randn(4, 2))
        result = task._step_batch(data, "test")
        assert len(result["preds"]) == 1

    def test_labels_as_list(self):
        task, _ = _make_task()
        task.compile()
        task._dl.loss = None
        data = (torch.randn(4, 4), [torch.randn(4, 2)])
        result = task._step_batch(data, "train")
        assert len(result["labels"]) == 1

    def test_labels_as_tuple(self):
        task, _ = _make_task()
        task.compile()
        task._dl.loss = None
        data = (torch.randn(4, 4), (torch.randn(4, 2),))
        result = task._step_batch(data, "train")
        assert len(result["labels"]) == 1

    def test_loss_none_labels(self):
        """When labels have no batch size (e.g., None-like), loss returns None."""
        task, _ = _make_task()
        task.compile()
        # Empty tuple for labels -> _infer_batch_size returns None -> _step_loss returns None
        data = (torch.randn(4, 4), torch.tensor([]))
        result = task._step_batch(data, "train")
        assert "loss" not in result


# ---------------------------------------------------------------------------
# _step_model
# ---------------------------------------------------------------------------
class TestStepModel:
    def test_single_tensor(self):
        task, _ = _make_task()
        task.compile()
        out = task._step_model(torch.randn(4, 4))
        assert isinstance(out, torch.Tensor)

    def test_list_input(self):
        task, _ = _make_task(in_features=2)
        task.compile()

        class MultiArgModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(2, 2)

            def forward(self, x1, x2):
                return self.fc(x1 + x2)

        task._dl.model = MultiArgModel()
        out = task._step_model([torch.randn(4, 2), torch.randn(4, 2)])
        assert isinstance(out, torch.Tensor)

    def test_tuple_input(self):
        task, _ = _make_task(in_features=2)
        task.compile()

        class MultiArgModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(2, 2)

            def forward(self, x1, x2):
                return self.fc(x1 + x2)

        task._dl.model = MultiArgModel()
        out = task._step_model((torch.randn(4, 2), torch.randn(4, 2)))
        assert isinstance(out, torch.Tensor)


# ---------------------------------------------------------------------------
# _step_loss
# ---------------------------------------------------------------------------
class TestStepLoss:
    def test_with_labels(self):
        task, _ = _make_task()
        task.compile()
        out = torch.randn(4, 2)
        label = torch.randn(4, 2)
        loss = task._step_loss(out, label, torch.randn(4, 4))
        assert isinstance(loss, torch.Tensor)

    def test_without_labels_returns_none(self):
        task, _ = _make_task()
        task.compile()
        out = torch.randn(4, 2)
        loss = task._step_loss(out, torch.tensor([]), torch.randn(4, 4))
        assert loss is None

    def test_unsupervised_override(self):
        task, _ = _make_task(task_class=UnsupervisedPytorchTask, in_features=4, out_features=4)
        task._loss = nn.MSELoss
        task.compile()
        out = torch.randn(4, 4)
        inp = torch.randn(4, 4)
        loss = task._step_loss(out, torch.tensor([]), inp)
        assert isinstance(loss, torch.Tensor)


# ---------------------------------------------------------------------------
# _step_optimizer
# ---------------------------------------------------------------------------
class TestStepOptimizer:
    def test_zero_grad_backward_step(self):
        task, _ = _make_task()
        task.compile()
        inp = torch.randn(4, 4)
        out = task._dl.model(inp)
        label = torch.randn(4, 2)
        loss = task._dl.loss(out, label)
        task._step_optimizer(loss)  # should not raise


# ---------------------------------------------------------------------------
# _add_device
# ---------------------------------------------------------------------------
class TestAddDevice:
    def test_tensor(self):
        task, _ = _make_task()
        task.compile()
        t = torch.randn(4, 4)
        result = task._add_device(t)
        assert result.device == task._dl.device

    def test_list_of_tensors(self):
        task, _ = _make_task()
        task.compile()
        result = task._add_device([torch.randn(4, 4), torch.randn(4, 2)])
        assert isinstance(result, list)
        assert len(result) == 2

    def test_tuple_of_tensors(self):
        task, _ = _make_task()
        task.compile()
        result = task._add_device((torch.randn(4, 4), torch.randn(4, 2)))
        assert isinstance(result, tuple)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _detach_value
# ---------------------------------------------------------------------------
class TestDetachValue:
    def test_tensor(self):
        t = torch.tensor([1.0, 2.0], requires_grad=True)
        result = PytorchTask._detach_value(t)
        assert not result.requires_grad

    def test_non_tensor(self):
        assert PytorchTask._detach_value(42) == 42


# ---------------------------------------------------------------------------
# _detach_numpy
# ---------------------------------------------------------------------------
class TestDetachNumpy:
    def test_tensor(self):
        t = torch.tensor([1.0, 2.0])
        result = PytorchTask._detach_numpy(t)
        assert isinstance(result, np.ndarray)

    def test_non_tensor(self):
        assert PytorchTask._detach_numpy(42) == 42


# ---------------------------------------------------------------------------
# _infer_batch_size
# ---------------------------------------------------------------------------
class TestInferBatchSize:
    def test_tensor_multi_dim(self):
        task, _ = _make_task()
        assert task._infer_batch_size(torch.randn(8, 4)) == 8

    def test_tensor_0d(self):
        task, _ = _make_task()
        assert task._infer_batch_size(torch.tensor(1.0)) == 1

    def test_list_of_tensors_non_empty(self):
        task, _ = _make_task()
        assert task._infer_batch_size([torch.randn(8, 4)]) == 8

    def test_list_of_tensors_empty(self):
        task, _ = _make_task()
        assert task._infer_batch_size([]) is None

    def test_list_non_tensor_first(self):
        task, _ = _make_task()
        assert task._infer_batch_size([42]) is None

    def test_tuple(self):
        task, _ = _make_task()
        assert task._infer_batch_size((torch.randn(5, 2),)) == 5

    def test_none_returns_none(self):
        task, _ = _make_task()
        assert task._infer_batch_size(None) is None

    def test_other_type_returns_none(self):
        task, _ = _make_task()
        assert task._infer_batch_size(42) is None

    def test_list_0d_tensor(self):
        task, _ = _make_task()
        assert task._infer_batch_size([torch.tensor(1.0)]) == 1

    def test_tuple_0d_tensor(self):
        task, _ = _make_task()
        assert task._infer_batch_size((torch.tensor(1.0),)) == 1


# ---------------------------------------------------------------------------
# _get_hps_namespace_targets (PytorchTask override)
# ---------------------------------------------------------------------------
class TestPytorchGetHpsNamespaceTargets:
    def test_includes_dataset_dataloader(self):
        task, _ = _make_task()
        targets = task._get_hps_namespace_targets()
        assert "model" in targets
        assert "optimizer" in targets
        assert "loss" in targets
        assert "dataset" in targets
        assert "dataloader" in targets
        assert targets["dataset"] is task._dataset_args
        assert targets["dataloader"] is task._dataloader_args


# ---------------------------------------------------------------------------
# SelfSupervisedPytorchTask
# ---------------------------------------------------------------------------
class TestSelfSupervisedPytorchTask:
    def test_compile_after_base_sets_learning_mode(self):
        task, _ = _make_task(task_class=SelfSupervisedPytorchTask)
        task.compile()
        assert task._dl.learning_mode == LearningMode.SELF_SUPERVISED


# ---------------------------------------------------------------------------
# UnsupervisedPytorchTask
# ---------------------------------------------------------------------------
class TestUnsupervisedPytorchTask:
    def test_compile_after_base_sets_learning_mode(self):
        task, _ = _make_task(task_class=UnsupervisedPytorchTask, in_features=4, out_features=4)
        task._loss = nn.MSELoss
        task.compile()
        assert task._dl.learning_mode == LearningMode.UNSUPERVISED

    def test_step_loss_ignores_labels(self):
        task, _ = _make_task(task_class=UnsupervisedPytorchTask, in_features=4, out_features=4)
        task._loss = nn.MSELoss
        task.compile()
        inputs = torch.randn(4, 4)
        outputs = torch.randn(4, 4)
        labels = torch.tensor([])
        loss = task._step_loss(outputs, labels, inputs)
        assert isinstance(loss, torch.Tensor)


# ---------------------------------------------------------------------------
# Metrics during training
# ---------------------------------------------------------------------------
class TestMetricsDuringTraining:
    def test_fit_with_metrics(self):
        task, _ = _make_task(
            num_epochs=1,
            extra_kwargs={
                "metrics": ["acc"],
                "dataloader_args": {"batch_size": 20},
            },
        )
        task.compile()
        result = task.fit()
        assert "train" in result
        assert len(result["train"]) == 1

    def test_progress_bar_with_metrics(self):
        task, _ = _make_task(
            num_epochs=1,
            extra_kwargs={
                "metrics": ["acc"],
                "progress_bar": True,
                "dataloader_args": {"batch_size": 20},
            },
        )
        task.compile()
        result = task.fit()
        assert "train" in result


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------
class TestPytorchTaskExecute:
    def test_full_execute(self):
        task, sg = _make_task(
            num_epochs=1,
            extra_kwargs={"dataloader_args": {"batch_size": 20}},
        )
        result = task.execute()
        assert "fit" in result
        assert "predict" in result

    def test_execute_without_train(self):
        task, sg = _make_task(
            phases={"test": "test"},
            num_epochs=0,
            extra_kwargs={"dataloader_args": {"batch_size": 20}},
        )
        result = task.execute()
        assert result["fit"] is None
        assert result["predict"] is not None


# ---------------------------------------------------------------------------
# set_hps with pytorch namespaces
# ---------------------------------------------------------------------------
class TestPytorchSetHps:
    def test_dataset_namespace(self):
        task, _ = _make_task()
        task.set_hps({"dataset__preload": True})
        assert task._dataset_args["preload"] is True

    def test_dataloader_namespace(self):
        task, _ = _make_task()
        task.set_hps({"dataloader__batch_size": 16})
        assert task._dataloader_args["batch_size"] == 16


# ---------------------------------------------------------------------------
# _step_batch with batch_size inference failure
# ---------------------------------------------------------------------------
class TestStepBatchBatchSizeFailure:
    def test_cannot_infer_batch_size(self):
        task, _ = _make_task()
        task.compile()
        # Use a non-tensor, non-list/tuple "input" that can't be inferred
        data = (42, torch.randn(4, 2))
        with pytest.raises(ValueError, match="Could not infer batch_size"):
            task._step_batch(data, "train")
