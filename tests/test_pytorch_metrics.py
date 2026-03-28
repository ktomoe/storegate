"""Tests for storegate.task.pytorch_metrics."""

import pytest
import numpy as np
import torch

from storegate.task.pytorch_metrics import (
    BUILTIN_METRICS,
    EpochMetric,
    ResolvedMetric,
    _as_output_list,
    _as_torch_tensor,
    _binary_pred_to_label,
    _binary_true_to_label,
    _compute_acc,
    _compute_binary_acc,
    _compute_multi_acc,
    _compute_single_acc,
    _flatten_binary_output,
    _get_pred_label_pairs,
    _is_torch_tensor,
    _require_single_output_pair,
    _scalar_numpy_to_float,
    _scalar_tensor,
    _try_import_torch,
    make_key_lookup,
)


# ---------------------------------------------------------------------------
# _as_output_list
# ---------------------------------------------------------------------------
class TestAsOutputList:
    def test_single_value(self):
        assert _as_output_list(42) == [42]

    def test_list(self):
        assert _as_output_list([1, 2]) == [1, 2]

    def test_tuple(self):
        assert _as_output_list((1, 2)) == [1, 2]

    def test_single_string(self):
        # string is not a list/tuple, so it should be wrapped
        assert _as_output_list("hello") == ["hello"]


# ---------------------------------------------------------------------------
# _try_import_torch
# ---------------------------------------------------------------------------
class TestTryImportTorch:
    def test_returns_torch(self):
        result = _try_import_torch()
        assert result is not None
        assert result is torch


# ---------------------------------------------------------------------------
# _is_torch_tensor
# ---------------------------------------------------------------------------
class TestIsTorchTensor:
    def test_tensor_true(self):
        assert _is_torch_tensor(torch.tensor(1.0)) is True

    def test_ndarray_false(self):
        assert _is_torch_tensor(np.array(1.0)) is False

    def test_int_false(self):
        assert _is_torch_tensor(42) is False


# ---------------------------------------------------------------------------
# _as_torch_tensor
# ---------------------------------------------------------------------------
class TestAsTorchTensor:
    def test_from_tensor_same_device(self):
        t = torch.tensor([1.0, 2.0])
        result = _as_torch_tensor(t, device=torch.device("cpu"))
        assert result is t

    def test_from_tensor_different_device(self):
        t = torch.tensor([1.0, 2.0], device="cpu")
        # We test that it returns a tensor on the same device (cpu->cpu is same)
        result = _as_torch_tensor(t, device=torch.device("cpu"))
        assert result is t

    def test_from_tensor_no_device(self):
        t = torch.tensor([1.0])
        result = _as_torch_tensor(t)
        assert result is t

    def test_from_numpy(self):
        arr = np.array([1.0, 2.0])
        result = _as_torch_tensor(arr)
        assert isinstance(result, torch.Tensor)
        assert torch.equal(result, torch.tensor([1.0, 2.0]))

    def test_from_numpy_with_device(self):
        arr = np.array([1.0])
        result = _as_torch_tensor(arr, device=torch.device("cpu"))
        assert isinstance(result, torch.Tensor)

    def test_without_torch_raises(self):
        import unittest.mock as mock

        with mock.patch("storegate.task.pytorch_metrics._try_import_torch", return_value=None):
            with pytest.raises(ModuleNotFoundError, match="torch is required"):
                _as_torch_tensor(np.array([1.0]))


# ---------------------------------------------------------------------------
# _get_pred_label_pairs
# ---------------------------------------------------------------------------
class TestGetPredLabelPairs:
    def test_normal(self):
        br = {"preds": [torch.tensor([1])], "labels": [torch.tensor([1])]}
        pairs = _get_pred_label_pairs(br, metric_name="test")
        assert len(pairs) == 1

    def test_none_preds(self):
        br = {"labels": [torch.tensor([1])]}
        assert _get_pred_label_pairs(br, metric_name="test") is None

    def test_none_labels(self):
        br = {"preds": [torch.tensor([1])]}
        assert _get_pred_label_pairs(br, metric_name="test") is None

    def test_length_mismatch(self):
        br = {"preds": [torch.tensor([1]), torch.tensor([2])], "labels": [torch.tensor([1])]}
        with pytest.raises(ValueError, match="same number of output heads"):
            _get_pred_label_pairs(br, metric_name="test")

    def test_single_values_wrapped(self):
        # non-list/tuple values get wrapped
        br = {"preds": torch.tensor([1, 2]), "labels": torch.tensor([1, 2])}
        pairs = _get_pred_label_pairs(br, metric_name="test")
        assert len(pairs) == 1


# ---------------------------------------------------------------------------
# _require_single_output_pair
# ---------------------------------------------------------------------------
class TestRequireSingleOutputPair:
    def test_single_head(self):
        br = {"preds": [torch.tensor([1])], "labels": [torch.tensor([1])]}
        pair = _require_single_output_pair(br, metric_name="acc")
        assert pair is not None

    def test_multiple_heads_error(self):
        br = {
            "preds": [torch.tensor([1]), torch.tensor([2])],
            "labels": [torch.tensor([1]), torch.tensor([2])],
        }
        with pytest.raises(ValueError, match="single output head"):
            _require_single_output_pair(br, metric_name="acc")

    def test_none_result(self):
        br = {"preds": None, "labels": [torch.tensor([1])]}
        result = _require_single_output_pair(br, metric_name="acc")
        assert result is None


# ---------------------------------------------------------------------------
# _compute_single_acc
# ---------------------------------------------------------------------------
class TestComputeSingleAcc:
    def test_tensor_with_argmax(self):
        # multi-class predictions (2D), labels (2D one-hot)
        pred = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        label = torch.tensor([[0, 1], [1, 0]])
        acc = _compute_single_acc(pred, label)
        assert float(acc) == pytest.approx(1.0)

    def test_tensor_without_argmax(self):
        # 1D integer labels
        pred = torch.tensor([1, 0, 1])
        label = torch.tensor([1, 0, 0])
        acc = _compute_single_acc(pred, label)
        assert float(acc) == pytest.approx(2.0 / 3.0)

    def test_numpy_with_argmax(self):
        pred = np.array([[0.1, 0.9], [0.8, 0.2]])
        label = np.array([[0, 1], [1, 0]])
        acc = _compute_single_acc(pred, label)
        assert acc == pytest.approx(1.0)

    def test_numpy_without_argmax(self):
        pred = np.array([1, 0, 1])
        label = np.array([1, 0, 0])
        acc = _compute_single_acc(pred, label)
        assert acc == pytest.approx(2.0 / 3.0)


# ---------------------------------------------------------------------------
# _compute_acc
# ---------------------------------------------------------------------------
class TestComputeAcc:
    def test_normal(self):
        br = {
            "preds": [torch.tensor([1, 0, 1])],
            "labels": [torch.tensor([1, 0, 1])],
        }
        assert float(_compute_acc(br)) == pytest.approx(1.0)

    def test_none_pair(self):
        br = {"preds": None}
        assert _compute_acc(br) == 0.0


# ---------------------------------------------------------------------------
# _compute_multi_acc
# ---------------------------------------------------------------------------
class TestComputeMultiAcc:
    def test_multiple_heads_tensor(self):
        br = {
            "preds": [torch.tensor([1, 0]), torch.tensor([1, 1])],
            "labels": [torch.tensor([1, 0]), torch.tensor([1, 0])],
        }
        result = _compute_multi_acc(br)
        # head1: 100%, head2: 50% -> mean 75%
        assert float(result) == pytest.approx(0.75)

    def test_multiple_heads_numpy(self):
        br = {
            "preds": [np.array([1, 0]), np.array([1, 1])],
            "labels": [np.array([1, 0]), np.array([1, 0])],
        }
        result = _compute_multi_acc(br)
        assert result == pytest.approx(0.75)

    def test_mixed_tensor_numpy(self):
        br = {
            "preds": [torch.tensor([1, 0]), np.array([1, 1])],
            "labels": [torch.tensor([1, 0]), np.array([1, 0])],
        }
        result = _compute_multi_acc(br)
        # First head uses tensors -> tensor, second uses numpy -> float
        # The mixed path in _compute_multi_acc checks for any torch tensor
        assert float(result) == pytest.approx(0.75)

    def test_none_pairs(self):
        br = {}
        assert _compute_multi_acc(br) == 0.0


# ---------------------------------------------------------------------------
# _flatten_binary_output
# ---------------------------------------------------------------------------
class TestFlattenBinaryOutput:
    def test_tensor_1d(self):
        t = torch.tensor([0.1, 0.9])
        result = _flatten_binary_output(t, name="preds")
        assert result.shape == (2,)

    def test_tensor_n1(self):
        t = torch.tensor([[0.1], [0.9]])
        result = _flatten_binary_output(t, name="preds")
        assert result.shape == (2,)

    def test_tensor_nk_error(self):
        t = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        with pytest.raises(ValueError, match="shape \\(N,\\) or \\(N, 1\\)"):
            _flatten_binary_output(t, name="preds")

    def test_numpy_1d(self):
        a = np.array([0.1, 0.9])
        result = _flatten_binary_output(a, name="preds")
        assert result.shape == (2,)

    def test_numpy_n1(self):
        a = np.array([[0.1], [0.9]])
        result = _flatten_binary_output(a, name="preds")
        assert result.shape == (2,)

    def test_numpy_nk_error(self):
        a = np.array([[0.1, 0.2], [0.3, 0.4]])
        with pytest.raises(ValueError, match="shape \\(N,\\) or \\(N, 1\\)"):
            _flatten_binary_output(a, name="labels")


# ---------------------------------------------------------------------------
# _binary_pred_to_label
# ---------------------------------------------------------------------------
class TestBinaryPredToLabel:
    def test_float_probs_numpy(self):
        pred = np.array([0.3, 0.7, 0.5])
        result = _binary_pred_to_label(pred)
        np.testing.assert_array_equal(result, np.array([0, 1, 1]))

    def test_float_logits_numpy(self):
        # values outside [0,1] -> threshold 0.0
        pred = np.array([-1.0, 0.5, 2.0])
        result = _binary_pred_to_label(pred)
        np.testing.assert_array_equal(result, np.array([0, 1, 1]))

    def test_int_passthrough_numpy(self):
        pred = np.array([0, 1, 1])
        result = _binary_pred_to_label(pred)
        np.testing.assert_array_equal(result, np.array([0, 1, 1]))

    def test_tensor_float_probs(self):
        pred = torch.tensor([0.3, 0.7, 0.5])
        result = _binary_pred_to_label(pred)
        assert isinstance(result, torch.Tensor)
        assert torch.equal(result, torch.tensor([0, 1, 1], dtype=torch.int64))

    def test_tensor_float_logits(self):
        pred = torch.tensor([-1.0, 0.5, 2.0])
        result = _binary_pred_to_label(pred)
        assert isinstance(result, torch.Tensor)
        assert torch.equal(result, torch.tensor([0, 1, 1], dtype=torch.int64))

    def test_tensor_int_passthrough(self):
        pred = torch.tensor([0, 1, 1])
        result = _binary_pred_to_label(pred)
        assert isinstance(result, torch.Tensor)
        assert torch.equal(result, torch.tensor([0, 1, 1], dtype=torch.int64))


# ---------------------------------------------------------------------------
# _binary_true_to_label
# ---------------------------------------------------------------------------
class TestBinaryTrueToLabel:
    def test_float_threshold_numpy(self):
        label = np.array([0.3, 0.7])
        result = _binary_true_to_label(label)
        np.testing.assert_array_equal(result, np.array([0, 1]))

    def test_int_passthrough_numpy(self):
        label = np.array([0, 1])
        result = _binary_true_to_label(label)
        np.testing.assert_array_equal(result, np.array([0, 1]))

    def test_tensor_float(self):
        label = torch.tensor([0.3, 0.7])
        result = _binary_true_to_label(label)
        assert isinstance(result, torch.Tensor)
        assert torch.equal(result, torch.tensor([0, 1], dtype=torch.int64))

    def test_tensor_int(self):
        label = torch.tensor([0, 1])
        result = _binary_true_to_label(label)
        assert isinstance(result, torch.Tensor)
        assert torch.equal(result, torch.tensor([0, 1], dtype=torch.int64))


# ---------------------------------------------------------------------------
# _compute_binary_acc
# ---------------------------------------------------------------------------
class TestComputeBinaryAcc:
    def test_probs_01(self):
        br = {
            "preds": [torch.tensor([0.1, 0.9])],
            "labels": [torch.tensor([0, 1])],
        }
        assert float(_compute_binary_acc(br)) == pytest.approx(1.0)

    def test_logits(self):
        br = {
            "preds": [torch.tensor([-1.0, 2.0])],
            "labels": [torch.tensor([0, 1])],
        }
        assert float(_compute_binary_acc(br)) == pytest.approx(1.0)

    def test_integers(self):
        br = {
            "preds": [np.array([0, 1, 1])],
            "labels": [np.array([0, 1, 0])],
        }
        assert _compute_binary_acc(br) == pytest.approx(2.0 / 3.0)

    def test_shape_mismatch(self):
        br = {
            "preds": [torch.tensor([0.1, 0.9, 0.5])],
            "labels": [torch.tensor([0, 1])],
        }
        with pytest.raises(ValueError, match="same shape"):
            _compute_binary_acc(br)

    def test_n1_shapes(self):
        br = {
            "preds": [torch.tensor([[0.1], [0.9]])],
            "labels": [torch.tensor([[0], [1]])],
        }
        assert float(_compute_binary_acc(br)) == pytest.approx(1.0)

    def test_none_pair(self):
        br = {}
        assert _compute_binary_acc(br) == 0.0

    def test_numpy_binary_acc(self):
        br = {
            "preds": [np.array([0.1, 0.9])],
            "labels": [np.array([0.0, 1.0])],
        }
        assert _compute_binary_acc(br) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# make_key_lookup
# ---------------------------------------------------------------------------
class TestMakeKeyLookup:
    def test_existing_key(self):
        fn = make_key_lookup("loss")
        result = fn({"loss": 0.5, "acc": 0.9})
        assert result == 0.5

    def test_missing_key(self):
        fn = make_key_lookup("missing")
        result = fn({"loss": 0.5})
        assert result is None

    def test_has_name(self):
        fn = make_key_lookup("my_key")
        assert fn.__name__ == "my_key"


# ---------------------------------------------------------------------------
# BUILTIN_METRICS
# ---------------------------------------------------------------------------
class TestBuiltinMetrics:
    def test_correct_entries(self):
        assert "acc" in BUILTIN_METRICS
        assert "binary_acc" in BUILTIN_METRICS
        assert "multi_acc" in BUILTIN_METRICS
        assert BUILTIN_METRICS["acc"] is _compute_acc
        assert BUILTIN_METRICS["binary_acc"] is _compute_binary_acc
        assert BUILTIN_METRICS["multi_acc"] is _compute_multi_acc


# ---------------------------------------------------------------------------
# ResolvedMetric
# ---------------------------------------------------------------------------
class TestResolvedMetric:
    def test_dataclass(self):
        fn = lambda br: 0.5
        rm = ResolvedMetric(name="test", fn=fn, requires_labels=True)
        assert rm.name == "test"
        assert rm.fn is fn
        assert rm.requires_labels is True

    def test_default_requires_labels(self):
        fn = lambda br: 0.5
        rm = ResolvedMetric(name="test", fn=fn)
        assert rm.requires_labels is False


# ---------------------------------------------------------------------------
# _scalar_numpy_to_float
# ---------------------------------------------------------------------------
class TestScalarNumpyToFloat:
    def test_np_float32(self):
        assert _scalar_numpy_to_float(np.float32(1.5)) == 1.5

    def test_np_int64(self):
        assert _scalar_numpy_to_float(np.int64(3)) == 3.0

    def test_0d_ndarray_number(self):
        assert _scalar_numpy_to_float(np.array(2.5)) == 2.5

    def test_0d_ndarray_bool(self):
        assert _scalar_numpy_to_float(np.array(True)) == 1.0

    def test_non_scalar_ndarray_returns_none(self):
        assert _scalar_numpy_to_float(np.array([1.0, 2.0])) is None

    def test_non_numpy_returns_none(self):
        assert _scalar_numpy_to_float(42) is None
        assert _scalar_numpy_to_float("hello") is None


# ---------------------------------------------------------------------------
# _scalar_tensor
# ---------------------------------------------------------------------------
class TestScalarTensor:
    def test_0d_tensor(self):
        t = torch.tensor(1.5)
        result = _scalar_tensor(t)
        assert result is not None
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0

    def test_non_0d_tensor(self):
        t = torch.tensor([1.5, 2.5])
        assert _scalar_tensor(t) is None

    def test_no_torch_available(self):
        import unittest.mock as mock

        with mock.patch("storegate.task.pytorch_metrics._try_import_torch", return_value=None):
            assert _scalar_tensor(torch.tensor(1.0)) is None


# ---------------------------------------------------------------------------
# EpochMetric
# ---------------------------------------------------------------------------
class TestEpochMetric:
    def test_init_with_none_metrics(self):
        em = EpochMetric(None)
        em.update({"batch_size": 10, "loss": 0.5})
        assert em.result() == {}

    def test_init_with_metrics(self):
        rm = ResolvedMetric(name="loss", fn=make_key_lookup("loss"))
        em = EpochMetric([rm])
        assert em.result() == {}

    def test_update_int_float_value(self):
        rm = ResolvedMetric(name="loss", fn=make_key_lookup("loss"))
        em = EpochMetric([rm])
        em.update({"batch_size": 2, "loss": 1.0})
        em.update({"batch_size": 2, "loss": 3.0})
        result = em.result()
        assert result["loss"] == pytest.approx(2.0)

    def test_update_numpy_scalar(self):
        rm = ResolvedMetric(name="loss", fn=lambda br: np.float32(br.get("loss", 0)))
        em = EpochMetric([rm])
        em.update({"batch_size": 1, "loss": 2.0})
        result = em.result()
        assert "loss" in result

    def test_update_torch_scalar(self):
        rm = ResolvedMetric(name="loss", fn=lambda br: torch.tensor(br.get("loss", 0)))
        em = EpochMetric([rm])
        em.update({"batch_size": 1, "loss": 2.0})
        result = em.result()
        assert "loss" in result

    def test_update_none_value(self):
        rm = ResolvedMetric(name="loss", fn=lambda br: None)
        em = EpochMetric([rm])
        em.update({"batch_size": 1})
        assert em.result() == {}

    def test_update_non_scalar_value_skipped(self):
        rm = ResolvedMetric(name="loss", fn=lambda br: np.array([1.0, 2.0]))
        em = EpochMetric([rm])
        em.update({"batch_size": 1})
        assert em.result() == {}

    def test_result_empty(self):
        rm = ResolvedMetric(name="loss", fn=make_key_lookup("loss"))
        em = EpochMetric([rm])
        assert em.result() == {}

    def test_result_with_data(self):
        rm = ResolvedMetric(name="loss", fn=make_key_lookup("loss"))
        em = EpochMetric([rm])
        em.update({"batch_size": 4, "loss": 1.0})
        em.update({"batch_size": 6, "loss": 2.0})
        result = em.result()
        # weighted average: (4*1 + 6*2) / 10 = 1.6
        assert result["loss"] == pytest.approx(1.6)

    def test_result_as_python_converts_tensors(self):
        rm = ResolvedMetric(name="val", fn=lambda br: torch.tensor(br.get("val", 0.0)))
        em = EpochMetric([rm])
        em.update({"batch_size": 1, "val": 3.0})
        result = em.result(as_python=True)
        assert isinstance(result["val"], float)

    def test_result_as_python_false(self):
        rm = ResolvedMetric(name="val", fn=make_key_lookup("val"))
        em = EpochMetric([rm])
        em.update({"batch_size": 1, "val": 3.0})
        result = em.result(as_python=False)
        assert "val" in result

    def test_result_as_python_with_numpy_scalar(self):
        # Return np.float32 from fn -> stored as float in weighted_sums
        # After division, it's a float, _scalar_numpy_to_float returns None for float
        # but it falls through and gets added as results[key] = value
        rm = ResolvedMetric(name="val", fn=lambda br: np.float32(2.0))
        em = EpochMetric([rm])
        em.update({"batch_size": 1, "val": 2.0})
        result = em.result(as_python=True)
        assert "val" in result

    def test_default_batch_size_is_1(self):
        rm = ResolvedMetric(name="loss", fn=make_key_lookup("loss"))
        em = EpochMetric([rm])
        em.update({"loss": 5.0})  # no batch_size key
        result = em.result()
        assert result["loss"] == pytest.approx(5.0)

    def test_update_int_value(self):
        rm = ResolvedMetric(name="loss", fn=make_key_lookup("loss"))
        em = EpochMetric([rm])
        em.update({"batch_size": 1, "loss": 3})
        result = em.result()
        assert result["loss"] == pytest.approx(3.0)
