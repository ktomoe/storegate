"""Unit tests for PytorchTask._output_to_storegate."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, call

torch = pytest.importorskip('torch')

from storegate.task.pytorch_task import PytorchTask


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
