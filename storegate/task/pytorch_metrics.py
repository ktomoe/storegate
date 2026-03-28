from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


type MetricFn = Callable[[dict[str, Any]], float]


def _as_output_list(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _try_import_torch() -> Any | None:
    try:
        import torch
    except ModuleNotFoundError:
        return None
    return torch


def _is_torch_tensor(value: Any) -> bool:
    torch = _try_import_torch()
    return torch is not None and isinstance(value, torch.Tensor)


def _as_torch_tensor(value: Any, *, device: Any | None = None) -> Any:
    torch = _try_import_torch()
    if torch is None:
        raise ModuleNotFoundError("torch is required for tensor metrics.")

    if isinstance(value, torch.Tensor):
        if device is not None and value.device != device:
            return value.to(device)
        return value
    return torch.as_tensor(value, device=device)


def _get_pred_label_pairs(
    batch_result: dict[str, Any],
    *,
    metric_name: str,
) -> list[tuple[Any, Any]] | None:
    preds = batch_result.get("preds")
    labels = batch_result.get("labels")
    if preds is None or labels is None:
        return None

    pred_list = _as_output_list(preds)
    label_list = _as_output_list(labels)
    if len(pred_list) != len(label_list):
        raise ValueError(
            f"{metric_name} requires preds and labels to have the same number of output heads: "
            f"got preds={len(pred_list)}, labels={len(label_list)}"
        )
    return list(zip(pred_list, label_list))


def _require_single_output_pair(
    batch_result: dict[str, Any],
    *,
    metric_name: str,
) -> tuple[Any, Any] | None:
    pairs = _get_pred_label_pairs(batch_result, metric_name=metric_name)
    if pairs is None:
        return None
    if len(pairs) != 1:
        raise ValueError(
            f"{metric_name} expects a single output head, got {len(pairs)}. "
            "Use multi_acc for multi-output classification."
        )
    return pairs[0]


def _compute_single_acc(pred: Any, label: Any) -> float:
    if _is_torch_tensor(pred) or _is_torch_tensor(label):
        torch = _try_import_torch()
        assert torch is not None
        pred_tensor = _as_torch_tensor(pred)
        label_tensor = _as_torch_tensor(label, device=pred_tensor.device)
        if pred_tensor.ndim > 1:
            pred_tensor = pred_tensor.argmax(dim=-1)
        if label_tensor.ndim > 1:
            label_tensor = label_tensor.argmax(dim=-1)
        return (pred_tensor == label_tensor).to(dtype=torch.float32).mean()

    pred_arr = np.asarray(pred)
    label_arr = np.asarray(label)
    if pred_arr.ndim > 1:
        pred_arr = pred_arr.argmax(axis=-1)
    if label_arr.ndim > 1:
        label_arr = label_arr.argmax(axis=-1)
    return float(np.mean(pred_arr == label_arr))


def _compute_acc(batch_result: dict[str, Any]) -> float:
    """Compute accuracy from preds and labels in batch_result."""
    pair = _require_single_output_pair(batch_result, metric_name="acc")
    if pair is None:
        return 0.0
    pred, label = pair
    return _compute_single_acc(pred, label)


def _compute_multi_acc(batch_result: dict[str, Any]) -> float:
    """Compute mean accuracy across multiple output heads."""
    pairs = _get_pred_label_pairs(batch_result, metric_name="multi_acc")
    if pairs is None:
        return 0.0

    values = [_compute_single_acc(pred, label) for pred, label in pairs]
    torch = _try_import_torch()
    if torch is not None and any(isinstance(value, torch.Tensor) for value in values):
        first_tensor = next(
            value for value in values if isinstance(value, torch.Tensor)
        )
        tensors = [
            value if isinstance(value, torch.Tensor)
            else torch.as_tensor(value, device=first_tensor.device)
            for value in values
        ]
        return torch.stack(tensors).mean()

    return float(np.mean(values))


def _flatten_binary_output(array: Any, *, name: str) -> Any:
    if _is_torch_tensor(array):
        arr = array
        if arr.ndim > 1:
            if arr.shape[-1] != 1:
                raise ValueError(
                    f"binary_acc expects {name} to have shape (N,) or (N, 1), got {tuple(arr.shape)}"
                )
            arr = arr.squeeze(dim=-1)
        return arr

    arr = np.asarray(array)
    if arr.ndim > 1:
        if arr.shape[-1] != 1:
            raise ValueError(
                f"binary_acc expects {name} to have shape (N,) or (N, 1), got {arr.shape}"
            )
        arr = np.squeeze(arr, axis=-1)
    return arr


def _binary_pred_to_label(pred: Any) -> Any:
    arr = _flatten_binary_output(pred, name="preds")
    if _is_torch_tensor(arr):
        torch = _try_import_torch()
        assert torch is not None
        if arr.is_floating_point():
            threshold = torch.where(
                ((0.0 <= arr) & (arr <= 1.0)).all(),
                arr.new_tensor(0.5),
                arr.new_tensor(0.0),
            )
            return (arr >= threshold).to(torch.int64)
        return arr.to(torch.int64)

    if np.issubdtype(arr.dtype, np.floating):
        if np.all((0.0 <= arr) & (arr <= 1.0)):
            return (arr >= 0.5).astype(np.int64)
        return (arr >= 0.0).astype(np.int64)
    return arr.astype(np.int64)


def _binary_true_to_label(label: Any) -> Any:
    arr = _flatten_binary_output(label, name="labels")
    if _is_torch_tensor(arr):
        torch = _try_import_torch()
        assert torch is not None
        if arr.is_floating_point():
            return (arr >= 0.5).to(torch.int64)
        return arr.to(torch.int64)

    if np.issubdtype(arr.dtype, np.floating):
        return (arr >= 0.5).astype(np.int64)
    return arr.astype(np.int64)


def _compute_binary_acc(batch_result: dict[str, Any]) -> float:
    """Compute binary accuracy from probs/logits and binary labels."""
    pair = _require_single_output_pair(batch_result, metric_name="binary_acc")
    if pair is None:
        return 0.0

    pred_raw, label_raw = pair
    pred = _binary_pred_to_label(pred_raw)
    label = _binary_true_to_label(label_raw)
    if tuple(pred.shape) != tuple(label.shape):
        raise ValueError(
            "binary_acc requires preds and labels to have the same shape "
            f"after normalization: got preds={pred.shape}, labels={label.shape}"
        )

    if _is_torch_tensor(pred) or _is_torch_tensor(label):
        torch = _try_import_torch()
        assert torch is not None
        pred_tensor = _as_torch_tensor(pred)
        label_tensor = _as_torch_tensor(label, device=pred_tensor.device)
        return (pred_tensor == label_tensor).to(dtype=torch.float32).mean()

    return float(np.mean(pred == label))


def make_key_lookup(key: str) -> MetricFn:
    """Return a function that looks up *key* in batch_result."""
    def lookup(batch_result: dict[str, Any]) -> Any:
        return batch_result.get(key)

    lookup.__name__ = key
    return lookup


BUILTIN_METRICS: dict[str, MetricFn] = {
    "acc": _compute_acc,
    "binary_acc": _compute_binary_acc,
    "multi_acc": _compute_multi_acc,
}


@dataclass(frozen=True, slots=True)
class ResolvedMetric:
    """A metric whose name and computation function have been resolved."""
    name: str
    fn: MetricFn
    requires_labels: bool = False


def _scalar_numpy_to_float(value: Any) -> float | None:
    if isinstance(value, np.generic):
        return float(value)

    if isinstance(value, np.ndarray) and value.ndim == 0:
        if (
            np.issubdtype(value.dtype, np.number)
            or np.issubdtype(value.dtype, np.bool_)
        ):
            return float(value.item())

    return None


def _scalar_tensor(value: Any) -> Any | None:
    torch = _try_import_torch()
    if torch is None:
        return None

    if isinstance(value, torch.Tensor) and value.dim() == 0:
        return value.detach()
    return None


class EpochMetric:
    """Accumulates batch metrics and computes weighted averages over an epoch."""

    def __init__(self, metrics: list[ResolvedMetric] | None = None) -> None:
        self._metrics = metrics
        self._total_size: int = 0
        self._weighted_sums: dict[str, Any] = {}

    def update(self, batch_result: dict[str, Any]) -> None:
        """Accumulate a single batch result."""
        if self._metrics is None:
            return

        batch_size = batch_result.get("batch_size", 1)
        self._total_size += batch_size

        for metric in self._metrics:
            value = metric.fn(batch_result)
            if value is None:
                continue

            scalar_value: Any | None = None
            if isinstance(value, (int, float)):
                scalar_value = float(value)
            else:
                scalar_value = _scalar_numpy_to_float(value)
                if scalar_value is None:
                    scalar_value = _scalar_tensor(value)

            if scalar_value is not None:
                self._weighted_sums[metric.name] = (
                    self._weighted_sums.get(metric.name, 0.0)
                    + scalar_value * batch_size
                )

    def result(self, *, as_python: bool = True) -> dict[str, Any]:
        """Return epoch-level metrics."""
        results: dict[str, Any] = {}
        if self._total_size > 0:
            for key, wsum in self._weighted_sums.items():
                value = wsum / self._total_size
                if as_python:
                    scalar_value = _scalar_numpy_to_float(value)
                    if scalar_value is None:
                        scalar_tensor = _scalar_tensor(value)
                        if scalar_tensor is not None:
                            scalar_value = float(scalar_tensor.item())

                    if scalar_value is not None:
                        results[key] = scalar_value
                        continue

                results[key] = value
        return results
