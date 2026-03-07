"""Unit tests for PytorchTask._output_to_storegate and pytorch_util.build_module."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, call

torch = pytest.importorskip('torch')

from storegate.task.pytorch_task import PytorchTask
from storegate.task.pytorch.pytorch_util import build_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_task(output_var_names):
    """Return a minimal PytorchTask with mocked storegate."""
    task = PytorchTask.__new__(PytorchTask)
    task._output_var_names = output_var_names
    task._storegate = MagicMock()
    return task


def make_tensor(value: float, size: int = 2) -> torch.Tensor:
    return torch.full((size,), value)


def make_step_task() -> PytorchTask:
    """Return a minimal task instance for step_batch tests."""
    task = PytorchTask.__new__(PytorchTask)
    task._device = torch.device('cpu')
    task.step_model = MagicMock(return_value=torch.tensor([[0.1, 0.9]]))
    task.step_loss = MagicMock(return_value={'loss': torch.tensor(0.3)})
    task.step_optimizer = MagicMock()
    return task


# ---------------------------------------------------------------------------
# _output_to_storegate — output_var_names is None
# ---------------------------------------------------------------------------

def test_output_to_storegate_none_var_names_is_noop():
    task = make_task(output_var_names=None)
    task._output_to_storegate(make_tensor(1.0))
    task._storegate.add_data.assert_not_called()


# ---------------------------------------------------------------------------
# _output_to_storegate — single output
# ---------------------------------------------------------------------------

def test_output_to_storegate_single_tensor_calls_add_data():
    task = make_task(output_var_names=['pred'])
    t = make_tensor(1.0)
    task._output_to_storegate(t)
    task._storegate.add_data.assert_called_once()
    args = task._storegate.add_data.call_args
    assert args[0][0] == 'pred'
    assert args[0][2] == 'test'


def test_output_to_storegate_single_tensor_wrapped_in_list():
    """A bare Tensor (not list) must be wrapped before zip."""
    task = make_task(output_var_names=['out'])
    task._output_to_storegate(make_tensor(2.0))
    assert task._storegate.add_data.call_count == 1


# ---------------------------------------------------------------------------
# _output_to_storegate — multiple outputs
# ---------------------------------------------------------------------------

def test_output_to_storegate_list_calls_add_data_for_each():
    task = make_task(output_var_names=['a', 'b'])
    task._output_to_storegate([make_tensor(1.0), make_tensor(2.0)])
    assert task._storegate.add_data.call_count == 2


def test_output_to_storegate_list_maps_names_to_outputs():
    task = make_task(output_var_names=['x', 'y'])
    task._output_to_storegate([make_tensor(1.0), make_tensor(2.0)])
    names_written = [c[0][0] for c in task._storegate.add_data.call_args_list]
    assert names_written == ['x', 'y']


# ---------------------------------------------------------------------------
# _output_to_storegate — length mismatch raises ValueError
# ---------------------------------------------------------------------------

def test_output_to_storegate_more_outputs_than_var_names_raises():
    task = make_task(output_var_names=['only_one'])
    with pytest.raises(ValueError, match='output_var_names'):
        task._output_to_storegate([make_tensor(1.0), make_tensor(2.0)])


def test_output_to_storegate_fewer_outputs_than_var_names_raises():
    task = make_task(output_var_names=['a', 'b', 'c'])
    with pytest.raises(ValueError, match='output_var_names'):
        task._output_to_storegate([make_tensor(1.0)])


def test_output_to_storegate_mismatch_error_contains_counts():
    task = make_task(output_var_names=['a', 'b'])
    with pytest.raises(ValueError, match=r'1.*2|2.*1'):
        task._output_to_storegate([make_tensor(1.0)])


def test_output_to_storegate_mismatch_does_not_write_partial_data():
    """No add_data calls must occur when lengths mismatch."""
    task = make_task(output_var_names=['a', 'b'])
    with pytest.raises(ValueError):
        task._output_to_storegate([make_tensor(1.0)])
    task._storegate.add_data.assert_not_called()


# ---------------------------------------------------------------------------
# build_module — string obj with modules=None raises ValueError
# ---------------------------------------------------------------------------

def test_build_module_string_with_none_modules_raises_value_error():
    with pytest.raises(ValueError, match="'Linear'"):
        build_module('Linear', {}, None)


def test_build_module_string_with_none_modules_error_mentions_class():
    with pytest.raises(ValueError, match='class'):
        build_module('Linear', {}, None)


def test_build_module_string_with_none_modules_error_mentions_torch():
    with pytest.raises(ValueError, match='torch'):
        build_module('Linear', {}, None)


def test_build_module_string_with_valid_modules_succeeds():
    import torch.nn as nn
    model = build_module('Linear', {'in_features': 2, 'out_features': 1}, nn)
    assert isinstance(model, nn.Linear)


def test_build_module_class_with_none_modules_succeeds():
    import torch.nn as nn
    model = build_module(nn.Linear, {'in_features': 2, 'out_features': 1}, None)
    assert isinstance(model, nn.Linear)


# ---------------------------------------------------------------------------
# DLTask.set_hps — empty suffix validation
# ---------------------------------------------------------------------------

def make_dl_task():
    """Return a minimal DLTask with mocked storegate."""
    from storegate.task.dl_task import DLTask
    task = DLTask.__new__(DLTask)
    task._storegate = MagicMock()
    task._data_id = None
    task._model_args = {}
    task._optimizer_args = {}
    task._loss_args = {}
    task._PROTECTED_KEYS = frozenset({'storegate', 'ml'})
    return task


def test_set_hps_model_empty_suffix_raises():
    task = make_dl_task()
    with pytest.raises(ValueError, match="model__"):
        task.set_hps({'model__': 64})


def test_set_hps_optimizer_empty_suffix_raises():
    task = make_dl_task()
    with pytest.raises(ValueError, match="optimizer__"):
        task.set_hps({'optimizer__': 1e-3})


def test_set_hps_loss_empty_suffix_raises():
    task = make_dl_task()
    with pytest.raises(ValueError, match="loss__"):
        task.set_hps({'loss__': 0.1})


def test_set_hps_model_valid_suffix_sets_arg():
    task = make_dl_task()
    task.set_hps({'model__hidden': 128})
    assert task._model_args['hidden'] == 128


def test_set_hps_optimizer_valid_suffix_sets_arg():
    task = make_dl_task()
    task.set_hps({'optimizer__lr': 1e-4})
    assert task._optimizer_args['lr'] == 1e-4


def test_set_hps_loss_valid_suffix_sets_arg():
    task = make_dl_task()
    task.set_hps({'loss__weight': 0.5})
    assert task._loss_args['weight'] == 0.5


def test_step_batch_test_without_labels_skips_loss():
    task = make_step_task()
    result = task.step_batch(torch.tensor([[1.0, 2.0]]), phase='test')

    assert result['labels'] is None
    assert 'loss' not in result
    assert 'pred' in result
    task.step_loss.assert_not_called()
    task.step_optimizer.assert_not_called()


def test_step_batch_train_without_labels_raises():
    task = make_step_task()
    with pytest.raises(ValueError, match='labels are required'):
        task.step_batch(torch.tensor([[1.0, 2.0]]), phase='train')
