"""Shared pytest fixtures for storegate tests."""
import numpy as np
import pytest

from storegate import StoreGate


DATA_ID = 'test_data'


@pytest.fixture
def sg(tmp_path):
    """StoreGate in write mode with no data_id set."""
    return StoreGate(output_dir=str(tmp_path), mode='w')


@pytest.fixture
def sg_with_id(tmp_path):
    """StoreGate in write mode with data_id set."""
    store = StoreGate(output_dir=str(tmp_path), mode='w')
    store.set_data_id(DATA_ID)
    return store


@pytest.fixture
def sg_with_data(tmp_path):
    """StoreGate pre-populated with sample data (zarr backend)."""
    store = StoreGate(output_dir=str(tmp_path), mode='w')
    store.set_data_id(DATA_ID)
    store.add_data('x', np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), phase='train')
    store.add_data('x', np.array([[7.0, 8.0]]), phase='valid')
    store.add_data('x', np.array([[9.0, 10.0], [11.0, 12.0]]), phase='test')
    return store
