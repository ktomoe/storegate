"""Unit tests for HybridDatabase."""
import numpy as np
import pytest

from storegate.database.hybrid_database import HybridDatabase


DATA_ID = 'test_data'


@pytest.fixture
def hybrid_db(tmp_path):
    db = HybridDatabase(output_dir=str(tmp_path), chunk=100, mode='w')
    db.initialize(DATA_ID)
    return db


def test_default_backend_is_zarr(hybrid_db):
    assert hybrid_db.get_backend() == 'zarr'


def test_set_backend_numpy(hybrid_db):
    hybrid_db.set_backend('numpy')
    assert hybrid_db.get_backend() == 'numpy'


def test_set_backend_zarr(hybrid_db):
    hybrid_db.set_backend('numpy')
    hybrid_db.set_backend('zarr')
    assert hybrid_db.get_backend() == 'zarr'


def test_add_and_get_zarr_backend(hybrid_db):
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    hybrid_db.add_data(DATA_ID, 'x', data, 'train')

    result = hybrid_db.get_data(DATA_ID, 'x', 'train', None)
    np.testing.assert_array_equal(result, data)


def test_add_and_get_numpy_backend(hybrid_db):
    hybrid_db.set_backend('numpy')
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    hybrid_db.add_data(DATA_ID, 'x', data, 'train')

    result = hybrid_db.get_data(DATA_ID, 'x', 'train', None)
    np.testing.assert_array_equal(result, data)


def test_backends_store_data_independently(hybrid_db):
    zarr_data = np.array([[1.0, 2.0]])
    numpy_data = np.array([[3.0, 4.0]])

    hybrid_db.set_backend('zarr')
    hybrid_db.add_data(DATA_ID, 'x', zarr_data, 'train')

    hybrid_db.set_backend('numpy')
    hybrid_db.add_data(DATA_ID, 'x', numpy_data, 'train')

    hybrid_db.set_backend('zarr')
    np.testing.assert_array_equal(
        hybrid_db.get_data(DATA_ID, 'x', 'train', None), zarr_data
    )

    hybrid_db.set_backend('numpy')
    np.testing.assert_array_equal(
        hybrid_db.get_data(DATA_ID, 'x', 'train', None), numpy_data
    )


def test_delete_data_zarr(hybrid_db):
    hybrid_db.add_data(DATA_ID, 'x', np.array([[1.0]]), 'train')
    hybrid_db.delete_data(DATA_ID, 'x', 'train')

    metadata = hybrid_db.get_metadata(DATA_ID, 'train')
    assert 'x' not in metadata


def test_delete_data_numpy(hybrid_db):
    hybrid_db.set_backend('numpy')
    hybrid_db.add_data(DATA_ID, 'x', np.array([[1.0]]), 'train')
    hybrid_db.delete_data(DATA_ID, 'x', 'train')

    metadata = hybrid_db.get_metadata(DATA_ID, 'train')
    assert 'x' not in metadata


def test_get_metadata_zarr(hybrid_db):
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    hybrid_db.add_data(DATA_ID, 'x', data, 'train')

    metadata = hybrid_db.get_metadata(DATA_ID, 'train')
    assert 'x' in metadata
    assert metadata['x']['total_events'] == 2
    assert metadata['x']['backend'] == 'zarr'


def test_get_metadata_numpy(hybrid_db):
    hybrid_db.set_backend('numpy')
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    hybrid_db.add_data(DATA_ID, 'x', data, 'train')

    metadata = hybrid_db.get_metadata(DATA_ID, 'train')
    assert 'x' in metadata
    assert metadata['x']['total_events'] == 2
    assert metadata['x']['backend'] == 'numpy'


def test_update_data_zarr(hybrid_db):
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    hybrid_db.add_data(DATA_ID, 'x', data, 'train')
    hybrid_db.update_data(DATA_ID, 'x', np.array([99.0, 99.0]), 'train', 0)

    result = hybrid_db.get_data(DATA_ID, 'x', 'train', 0)
    np.testing.assert_array_equal(result, [99.0, 99.0])
