import numpy as np

import torch
from torch import Tensor


def dummy(*args, **kwargs):
    return None


class EpochMetric:
    """Utility class to manage epoch metrics."""
    def __init__(self, metrics, enable, ml):
        self.metrics = metrics
        self.enable = enable
        self.ml = ml
        self.total = 0
        self.buffs = []
        self.preds = []

    def __call__(self, batch_result):
        result = {}
        if not self.enable:
            return result

        batch_size = batch_result['batch_size']
        self.total += batch_size

        for ii, metric in enumerate(self.metrics):
            if isinstance(metric, str):
                metric_fn = getattr(self, metric, dummy)
            else:
                metric_fn = metric
                metric = metric_fn.__name__

            metric_result = metric_fn(batch_result)

            if self.total == batch_size:  # first batch
                if isinstance(metric_result, list):
                    self.buffs.append([jmetric * batch_size for jmetric in metric_result])
                else:
                    self.buffs.append(metric_result * batch_size)
                result[metric] = metric_result

            else:
                if isinstance(metric_result, list):
                    for jj, jmetric in enumerate(metric_result):
                        self.buffs[ii][jj] += jmetric * batch_size
                    result[metric] = [jmetric / self.total for jmetric in self.buffs[ii]]

                else:
                    self.buffs[ii] += metric_result * batch_size
                    result[metric] = self.buffs[ii] / self.total

        return result

    # metrics
    def loss(self, batch_result):
        return batch_result['loss']['loss'].detach().item()

    def acc(self, batch_result):
        outputs = batch_result['outputs']
        labels = batch_result['labels']

        if isinstance(outputs, list):
            result = []
            for output, label in zip(outputs, labels):
                _, preds = torch.max(output, 1)
                corrects = torch.sum(preds == label.data)
                result.append(corrects.detach().item() / len(label))
        else:
            _, preds = torch.max(outputs, 1)
            corrects = torch.sum(preds == labels.data)
            result = corrects.detach().item() / len(labels)
        return result

    def lr(self, batch_result):
        return [p['lr'] for p in self.ml.optimizer.param_groups]


def get_pbar_metric(epoch_result):
    result = {}
    for key, value in epoch_result.items():

        if isinstance(value, list):
            result[key] = [f'{v:.2e}' for v in value]
        elif isinstance(value, float):
            result[key] = f'{value:.2e}'
        else:
            result[key] = value
    return result
