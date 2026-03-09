"""Unit tests for StoreGateDataset."""
from __future__ import annotations

import pytest
import numpy as np

torch = pytest.importorskip('torch')

from storegate import StoreGate  # noqa: E402
from storegate.task.pytorch.storegate_dataset import StoreGateDataset  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sg(tmp_path):
    """StoreGate with 10 train samples: x=(10,4), y=(10,1)."""
    store = StoreGate(output_dir=str(tmp_path), mode='w', data_id='ds')
    store.add_data('x', np.arange(40, dtype=np.float32).reshape(10, 4), phase='train')
    store.add_data('y', np.arange(10, dtype=np.int64).reshape(10, 1), phase='train')
    store.compile()
    return store


@pytest.fixture
def sg_multi(tmp_path):
    """StoreGate with two input features a=(5,2) and b=(5,3), label y=(5,1)."""
    store = StoreGate(output_dir=str(tmp_path), mode='w', data_id='ds')
    store.add_data('a', np.ones((5, 2), dtype=np.float32), phase='train')
    store.add_data('b', np.zeros((5, 3), dtype=np.float32), phase='train')
    store.add_data('y', np.arange(5, dtype=np.float32).reshape(5, 1), phase='train')
    store.compile()
    return store


@pytest.fixture
def sg_numpy(tmp_path):
    """StoreGate backed by numpy memory for aliasing tests."""
    store = StoreGate(output_dir=str(tmp_path), mode='w', data_id='ds')
    store.set_backend('numpy')
    store.add_data('x', np.arange(12, dtype=np.float32).reshape(3, 4), phase='train')
    store.add_data('y', np.arange(3, dtype=np.int64).reshape(3, 1), phase='train')
    store.compile()
    return store


# ---------------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------------

def test_len(sg) -> None:
    ds = StoreGateDataset(sg, 'train', input_var_names='x', true_var_names='y')
    assert len(ds) == 10


# ---------------------------------------------------------------------------
# __getitem__ — without preload
# ---------------------------------------------------------------------------

def test_getitem_returns_tuple_of_tensors(sg) -> None:
    ds = StoreGateDataset(sg, 'train', input_var_names='x', true_var_names='y')
    data, target = ds[0]
    assert isinstance(data, torch.Tensor)
    assert isinstance(target, torch.Tensor)


def test_getitem_shape(sg) -> None:
    ds = StoreGateDataset(sg, 'train', input_var_names='x', true_var_names='y')
    data, target = ds[0]
    assert data.shape == (4,)
    assert target.shape == (1,)


def test_getitem_values_match_storegate(sg) -> None:
    ds = StoreGateDataset(sg, 'train', input_var_names='x', true_var_names='y')
    for i in range(len(ds)):
        data, target = ds[i]
        expected_x = torch.as_tensor(sg.get_data('x', 'train', i))
        expected_y = torch.as_tensor(sg.get_data('y', 'train', i))
        assert torch.equal(data, expected_x)
        assert torch.equal(target, expected_y)


def test_getitem_all_indices_accessible(sg) -> None:
    ds = StoreGateDataset(sg, 'train', input_var_names='x', true_var_names='y')
    for i in range(len(ds)):
        data, target = ds[i]
        assert data.shape == (4,)


# ---------------------------------------------------------------------------
# __getitem__ — with preload
# ---------------------------------------------------------------------------

def test_preload_getitem_returns_tensors(sg) -> None:
    ds = StoreGateDataset(sg, 'train', input_var_names='x', true_var_names='y', preload=True)
    data, target = ds[0]
    assert isinstance(data, torch.Tensor)
    assert isinstance(target, torch.Tensor)


def test_preload_getitem_shape(sg) -> None:
    ds = StoreGateDataset(sg, 'train', input_var_names='x', true_var_names='y', preload=True)
    data, target = ds[0]
    assert data.shape == (4,)
    assert target.shape == (1,)


def test_preload_and_lazy_return_identical_data(sg) -> None:
    ds_lazy = StoreGateDataset(sg, 'train', input_var_names='x', true_var_names='y')
    ds_pre = StoreGateDataset(sg, 'train', input_var_names='x', true_var_names='y', preload=True)
    for i in range(len(ds_lazy)):
        d_l, t_l = ds_lazy[i]
        d_p, t_p = ds_pre[i]
        assert torch.equal(d_l, d_p)
        assert torch.equal(t_l, t_p)


# ---------------------------------------------------------------------------
# var_names variations
# ---------------------------------------------------------------------------

def test_single_string_var_name_returns_tensor(sg) -> None:
    ds = StoreGateDataset(sg, 'train', input_var_names='x', true_var_names='y')
    data, _ = ds[0]
    assert isinstance(data, torch.Tensor)


def test_single_element_list_var_name_returns_tensor(sg) -> None:
    """A one-element list should unwrap to a single Tensor, not a list."""
    ds = StoreGateDataset(sg, 'train', input_var_names=['x'], true_var_names='y')
    data, _ = ds[0]
    assert isinstance(data, torch.Tensor)
    assert data.shape == (4,)


def test_multi_element_list_var_names_returns_list_of_tensors(sg_multi) -> None:
    ds = StoreGateDataset(sg_multi, 'train', input_var_names=['a', 'b'], true_var_names='y')
    data, _ = ds[0]
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0].shape == (2,)
    assert data[1].shape == (3,)


def test_none_true_var_names_returns_input_only(sg) -> None:
    ds = StoreGateDataset(sg, 'train', input_var_names='x', true_var_names=None)
    data = ds[0]
    assert isinstance(data, torch.Tensor)
    assert data.shape == (4,)


def test_none_true_var_names_preload_returns_input_only(sg) -> None:
    ds = StoreGateDataset(
        sg, 'train', input_var_names='x', true_var_names=None, preload=True
    )
    data = ds[0]
    assert isinstance(data, torch.Tensor)
    assert data.shape == (4,)


def test_none_input_var_names_returns_none_data(sg) -> None:
    ds = StoreGateDataset(sg, 'train', input_var_names=None, true_var_names='y')
    data, _ = ds[0]
    assert data is None


# ---------------------------------------------------------------------------
# preload with list var_names
# ---------------------------------------------------------------------------

def test_preload_with_multi_var_names_returns_list(sg_multi) -> None:
    ds = StoreGateDataset(
        sg_multi, 'train',
        input_var_names=['a', 'b'], true_var_names='y',
        preload=True,
    )
    data, target = ds[0]
    assert isinstance(data, list)
    assert len(data) == 2


def test_preload_multi_var_values_match_lazy(sg_multi) -> None:
    ds_lazy = StoreGateDataset(
        sg_multi, 'train', input_var_names=['a', 'b'], true_var_names='y'
    )
    ds_pre = StoreGateDataset(
        sg_multi, 'train', input_var_names=['a', 'b'], true_var_names='y', preload=True
    )
    for i in range(len(ds_lazy)):
        d_l, t_l = ds_lazy[i]
        d_p, t_p = ds_pre[i]
        assert torch.equal(d_l[0], d_p[0])
        assert torch.equal(d_l[1], d_p[1])
        assert torch.equal(t_l, t_p)


@pytest.mark.parametrize('preload', [False, True])
def test_tensor_updates_do_not_mutate_numpy_storegate_data(
    sg_numpy: StoreGate,
    preload: bool,
) -> None:
    ds = StoreGateDataset(
        sg_numpy,
        'train',
        input_var_names='x',
        true_var_names='y',
        preload=preload,
    )

    data, target = ds[0]
    assert isinstance(data, torch.Tensor)
    assert isinstance(target, torch.Tensor)

    data.add_(100)
    target.add_(10)

    np.testing.assert_array_equal(
        sg_numpy.get_data('x', 'train', 0),
        np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        sg_numpy.get_data('y', 'train', 0),
        np.array([0], dtype=np.int64),
    )
