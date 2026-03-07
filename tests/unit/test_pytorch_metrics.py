"""Unit tests for EpochMetric and get_pbar_metric."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from typing import Any

torch = pytest.importorskip('torch')

from storegate.task.pytorch.pytorch_metrics import EpochMetric, get_pbar_metric  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ml(lr: float = 0.01) -> MagicMock:
    """Minimal DLEnv mock with an optimizer."""
    ml = MagicMock()
    ml.optimizer.param_groups = [{'lr': lr}]
    return ml


def make_batch(
    batch_size: int = 4,
    num_classes: int = 3,
    loss_val: float = 0.5,
    outputs: Any = None,
    labels: Any = None,
) -> dict[str, Any]:
    """Build a minimal batch_result dict."""
    if outputs is None:
        outputs = torch.randn(batch_size, num_classes)
    if labels is None:
        labels = torch.randint(0, num_classes, (batch_size,))
    return {
        'batch_size': batch_size,
        'outputs': outputs,
        'labels': labels,
        'loss': {'loss': torch.tensor(loss_val)},
    }


# ---------------------------------------------------------------------------
# EpochMetric — initialization
# ---------------------------------------------------------------------------

def test_unknown_string_metric_is_filtered() -> None:
    em = EpochMetric(['loss', 'unknown_metric'], make_ml())
    assert 'loss' in em.metrics
    assert 'unknown_metric' not in em.metrics


def test_known_string_metrics_are_accepted() -> None:
    em = EpochMetric(['loss', 'acc', 'lr'], make_ml())
    assert em.metrics == ['loss', 'acc', 'lr']


def test_callable_metric_is_accepted() -> None:
    def my_metric(batch_result: dict) -> float:
        return 1.0

    em = EpochMetric([my_metric], make_ml())
    assert my_metric in em.metrics


def test_empty_metrics_list() -> None:
    em = EpochMetric([], make_ml())
    result = em(make_batch())
    assert result == {}


def test_initial_total_is_zero() -> None:
    em = EpochMetric(['loss'], make_ml())
    assert em.total == 0


# ---------------------------------------------------------------------------
# EpochMetric — loss metric
# ---------------------------------------------------------------------------

def test_loss_first_batch_equals_batch_loss() -> None:
    em = EpochMetric(['loss'], make_ml())
    result = em(make_batch(batch_size=4, loss_val=0.8))
    assert abs(result['loss'] - 0.8) < 1e-5


def test_loss_accumulates_weighted_average() -> None:
    em = EpochMetric(['loss'], make_ml())
    em(make_batch(batch_size=4, loss_val=1.0))   # contributes 4 * 1.0
    result = em(make_batch(batch_size=4, loss_val=0.0))  # contributes 4 * 0.0
    # weighted avg = (4*1.0 + 4*0.0) / 8 = 0.5
    assert abs(result['loss'] - 0.5) < 1e-5


def test_loss_unequal_batch_sizes_weighted_correctly() -> None:
    em = EpochMetric(['loss'], make_ml())
    em(make_batch(batch_size=2, loss_val=1.0))   # 2 * 1.0 = 2.0
    result = em(make_batch(batch_size=8, loss_val=0.0))  # 8 * 0.0 = 0.0
    # weighted avg = 2.0 / 10 = 0.2
    assert abs(result['loss'] - 0.2) < 1e-5


def test_total_increments_with_each_batch() -> None:
    em = EpochMetric(['loss'], make_ml())
    em(make_batch(batch_size=4))
    em(make_batch(batch_size=6))
    assert em.total == 10


# ---------------------------------------------------------------------------
# EpochMetric — acc metric
# ---------------------------------------------------------------------------

def test_acc_perfect_predictions() -> None:
    em = EpochMetric(['acc'], make_ml())
    outputs = torch.tensor([[0.0, 10.0, 0.0],
                            [0.0, 0.0, 10.0]])  # predicts class 1, 2
    labels = torch.tensor([1, 2])
    result = em(make_batch(outputs=outputs, labels=labels, batch_size=2))
    assert abs(result['acc'] - 1.0) < 1e-5


def test_acc_zero_correct_predictions() -> None:
    em = EpochMetric(['acc'], make_ml())
    outputs = torch.tensor([[10.0, 0.0, 0.0],
                            [10.0, 0.0, 0.0]])  # always predicts class 0
    labels = torch.tensor([1, 2])              # correct answers are 1 and 2
    result = em(make_batch(outputs=outputs, labels=labels, batch_size=2))
    assert abs(result['acc'] - 0.0) < 1e-5


def test_acc_with_list_outputs() -> None:
    """acc with multi-head outputs (list of tensors)."""
    em = EpochMetric(['acc'], make_ml())
    # Two heads, each predicting class 0 correctly
    outputs = [
        torch.tensor([[10.0, 0.0]]),  # head 1: predicts class 0
        torch.tensor([[10.0, 0.0]]),  # head 2: predicts class 0
    ]
    labels = [torch.tensor([0]), torch.tensor([0])]
    batch = {
        'batch_size': 1,
        'outputs': outputs,
        'labels': labels,
        'loss': {'loss': torch.tensor(0.0)},
    }
    result = em(batch)
    # result['acc'] is a list of per-head accuracies
    assert isinstance(result['acc'], list)
    assert all(abs(v - 1.0) < 1e-5 for v in result['acc'])


# ---------------------------------------------------------------------------
# EpochMetric — lr metric
# ---------------------------------------------------------------------------

def test_lr_metric_returns_list_of_param_group_lrs() -> None:
    ml = make_ml(lr=0.001)
    em = EpochMetric(['lr'], ml)
    result = em(make_batch())
    assert result['lr'] == [0.001]


def test_lr_metric_multiple_param_groups() -> None:
    ml = MagicMock()
    ml.optimizer.param_groups = [{'lr': 0.1}, {'lr': 0.01}]
    em = EpochMetric(['lr'], ml)
    result = em(make_batch())
    assert result['lr'] == [0.1, 0.01]


# ---------------------------------------------------------------------------
# EpochMetric — callable metric
# ---------------------------------------------------------------------------

def test_callable_metric_is_called_with_batch_result() -> None:
    received: list = []

    def my_metric(batch_result: dict) -> float:
        received.append(batch_result)
        return 0.5

    em = EpochMetric([my_metric], make_ml())
    batch = make_batch(batch_size=4)
    em(batch)
    assert len(received) == 1
    assert received[0] is batch


def test_callable_metric_name_is_function_name() -> None:
    def custom_score(batch_result: dict) -> float:
        return 1.0

    em = EpochMetric([custom_score], make_ml())
    result = em(make_batch(batch_size=4))
    assert 'custom_score' in result


# ---------------------------------------------------------------------------
# EpochMetric — multiple metrics together
# ---------------------------------------------------------------------------

def test_multiple_metrics_all_appear_in_result() -> None:
    em = EpochMetric(['loss', 'acc'], make_ml())
    result = em(make_batch(batch_size=4, loss_val=0.5))
    assert 'loss' in result
    assert 'acc' in result


def test_multiple_batches_update_all_metrics() -> None:
    em = EpochMetric(['loss', 'acc'], make_ml())
    em(make_batch(batch_size=4, loss_val=1.0))
    result = em(make_batch(batch_size=4, loss_val=0.0))
    assert 'loss' in result
    assert 'acc' in result


def test_label_dependent_metrics_skipped_when_labels_none() -> None:
    em = EpochMetric(['loss', 'acc', 'lr'], make_ml(lr=0.02))
    batch = {
        'batch_size': 4,
        'outputs': torch.randn(4, 3),
        'labels': None,
    }
    result = em(batch)
    assert 'loss' not in result
    assert 'acc' not in result
    assert result['lr'] == [0.02]


# ---------------------------------------------------------------------------
# get_pbar_metric
# ---------------------------------------------------------------------------

def test_pbar_metric_float_formatted_as_scientific() -> None:
    result = get_pbar_metric({'loss': 0.123456})
    assert result['loss'] == f'{0.123456:.2e}'


def test_pbar_metric_list_each_element_formatted() -> None:
    result = get_pbar_metric({'lr': [0.001, 0.0001]})
    assert result['lr'] == [f'{0.001:.2e}', f'{0.0001:.2e}']


def test_pbar_metric_non_float_passed_through() -> None:
    result = get_pbar_metric({'step': 10})
    assert result['step'] == 10


def test_pbar_metric_multiple_keys() -> None:
    result = get_pbar_metric({'loss': 0.5, 'step': 3, 'lr': [0.01]})
    assert result['loss'] == f'{0.5:.2e}'
    assert result['step'] == 3
    assert result['lr'] == [f'{0.01:.2e}']
