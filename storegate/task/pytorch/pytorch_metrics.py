from __future__ import annotations

from collections.abc import Callable
from typing import Any, TYPE_CHECKING

import torch

from storegate import logger

if TYPE_CHECKING:
    from storegate.task.dl_env import DLEnv


class EpochMetric:
    """Utility class to manage epoch metrics."""
    def __init__(self, metrics: list[str | Callable[..., Any]], ml: DLEnv) -> None:
        self.ml = ml
        self.total: int = 0
        self.buffs: list[Any] = []
        valid: list[str | Callable[..., Any]] = []
        for metric in metrics:
            if isinstance(metric, str) and not hasattr(self, metric):
                logger.warn(f"Unknown metric '{metric}' will be ignored.")
            elif metric == 'lr' and ml.optimizer is None:
                logger.warn("Metric 'lr' requires an optimizer; it will be ignored.")
            else:
                valid.append(metric)
        self.metrics = valid

    def __call__(self, batch_result: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        batch_size = batch_result['batch_size']
        self.total += batch_size
        buff_index = 0

        for metric in self.metrics:
            if isinstance(metric, str):
                metric_fn = getattr(self, metric)
                metric_name = metric
            else:
                metric_fn = metric
                metric_name = metric_fn.__name__

            # test phase can run without labels; skip label-dependent metrics.
            if batch_result.get('labels') is None and metric_name in ('loss', 'acc'):
                continue

            metric_result = metric_fn(batch_result)

            if self.total == batch_size:  # first batch
                if isinstance(metric_result, list):
                    self.buffs.append([jmetric * batch_size for jmetric in metric_result])
                else:
                    self.buffs.append(metric_result * batch_size)
                result[metric_name] = metric_result

            else:
                if isinstance(metric_result, list):
                    for jj, jmetric in enumerate(metric_result):
                        self.buffs[buff_index][jj] += jmetric * batch_size
                    result[metric_name] = [jmetric / self.total for jmetric in self.buffs[buff_index]]

                else:
                    self.buffs[buff_index] += metric_result * batch_size
                    result[metric_name] = self.buffs[buff_index] / self.total
            buff_index += 1

        return result

    # metrics
    def loss(self, batch_result: dict[str, Any]) -> float:
        return batch_result['loss']['loss'].detach().item()  # type: ignore[no-any-return]

    def acc(self, batch_result: dict[str, Any]) -> float | list[float]:
        outputs = batch_result['outputs']
        labels = batch_result['labels']

        if isinstance(outputs, list):
            result: list[float] = []
            for output, label in zip(outputs, labels):
                _, preds = torch.max(output, 1)
                corrects = torch.sum(preds == label.data)
                result.append(corrects.detach().item() / len(label))
        else:
            _, preds = torch.max(outputs, 1)
            corrects = torch.sum(preds == labels.data)
            result = corrects.detach().item() / len(labels)
        return result

    def lr(self, batch_result: dict[str, Any]) -> list[float]:
        return [p['lr'] for p in self.ml.optimizer.param_groups]


def get_pbar_metric(epoch_result: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in epoch_result.items():

        if isinstance(value, list):
            result[key] = [f'{v:.2e}' for v in value]
        elif isinstance(value, float):
            result[key] = f'{value:.2e}'
        else:
            result[key] = value
    return result
