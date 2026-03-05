"""Unit tests for AgentTask."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from storegate.task.agent_task import AgentTask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SimpleTask(AgentTask):
    """Minimal concrete subclass with extra hyperparameters."""
    def __init__(self, storegate: MagicMock, lr: float = 0.01, epochs: int = 10) -> None:
        super().__init__(storegate)
        self._lr = lr
        self._epochs = epochs


def make_sg() -> MagicMock:
    sg = MagicMock()
    sg.set_data_id = MagicMock()
    return sg


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_init_sets_storegate() -> None:
    sg = make_sg()
    task = AgentTask(sg)
    assert task._storegate is sg


def test_init_data_id_is_none() -> None:
    task = AgentTask(make_sg())
    assert task._data_id is None


def test_storegate_property_getter() -> None:
    sg = make_sg()
    task = AgentTask(sg)
    assert task.storegate is sg


def test_storegate_property_setter() -> None:
    sg1, sg2 = make_sg(), make_sg()
    task = AgentTask(sg1)
    task.storegate = sg2
    assert task.storegate is sg2


# ---------------------------------------------------------------------------
# execute / finalize (base implementations return None)
# ---------------------------------------------------------------------------

def test_execute_returns_none() -> None:
    assert AgentTask(make_sg()).execute() is None


def test_finalize_returns_none() -> None:
    assert AgentTask(make_sg()).finalize() is None


# ---------------------------------------------------------------------------
# set_hps — success paths
# ---------------------------------------------------------------------------

def test_set_hps_updates_attribute() -> None:
    task = SimpleTask(make_sg(), lr=0.01)
    task.set_hps({'lr': 0.001})
    assert task._lr == 0.001


def test_set_hps_updates_multiple_attributes() -> None:
    task = SimpleTask(make_sg(), lr=0.01, epochs=10)
    task.set_hps({'lr': 1e-4, 'epochs': 20})
    assert task._lr == 1e-4
    assert task._epochs == 20


def test_set_hps_empty_dict_is_noop() -> None:
    task = SimpleTask(make_sg(), lr=0.01)
    task.set_hps({})
    assert task._lr == 0.01


# ---------------------------------------------------------------------------
# set_hps — error paths
# ---------------------------------------------------------------------------

def test_set_hps_raises_for_protected_key_storegate() -> None:
    task = SimpleTask(make_sg())
    with pytest.raises(AttributeError, match='not a valid hyperparameter'):
        task.set_hps({'storegate': make_sg()})


def test_set_hps_raises_for_protected_key_ml() -> None:
    task = SimpleTask(make_sg())
    with pytest.raises(AttributeError, match='not a valid hyperparameter'):
        task.set_hps({'ml': None})


def test_set_hps_raises_for_undefined_attribute() -> None:
    task = SimpleTask(make_sg())
    with pytest.raises(AttributeError, match='not defined'):
        task.set_hps({'nonexistent': 99})


# ---------------------------------------------------------------------------
# set_hps — data_id forwarding
# ---------------------------------------------------------------------------

def test_set_hps_calls_set_data_id_when_data_id_is_set() -> None:
    sg = make_sg()
    task = SimpleTask(sg)
    task._data_id = 'my_dataset'
    task.set_hps({'lr': 1e-4})
    sg.set_data_id.assert_called_once_with('my_dataset')


def test_set_hps_does_not_call_set_data_id_when_data_id_is_none() -> None:
    sg = make_sg()
    task = SimpleTask(sg)
    task.set_hps({'lr': 1e-4})
    sg.set_data_id.assert_not_called()
