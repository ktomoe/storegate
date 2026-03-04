"""Unit tests for ZarrDatabase."""
import numpy as np
import pytest

from storegate.database.zarr_database import ZarrDatabase


DATA_ID = 'test_data'


@pytest.fixture
def zarr_db(tmp_path):
    db = ZarrDatabase(output_dir=str(tmp_path), chunk=100, mode='w')
    db.initialize(DATA_ID)
    return db


def test_initialize_creates_groups(zarr_db):
    assert DATA_ID in list(zarr_db._db.group_keys())
    for phase in ['train', 'valid', 'test']:
        assert phase in list(zarr_db._db[DATA_ID].group_keys())


def test_add_data_new_var(zarr_db):
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    zarr_db.add_data(DATA_ID, 'x', data, 'train')

    result = zarr_db._db[DATA_ID]['train']['x'][:]
    np.testing.assert_array_equal(result, data)


def test_add_data_appends_to_existing(zarr_db):
    data1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    data2 = np.array([[5.0, 6.0]])
    zarr_db.add_data(DATA_ID, 'x', data1, 'train')
    zarr_db.add_data(DATA_ID, 'x', data2, 'train')

    expected = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = zarr_db._db[DATA_ID]['train']['x'][:]
    np.testing.assert_array_equal(result, expected)


def test_get_data_all(zarr_db):
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    zarr_db.add_data(DATA_ID, 'x', data, 'train')

    result = zarr_db.get_data(DATA_ID, 'x', 'train', None)
    np.testing.assert_array_equal(result, data)


def test_get_data_by_int_index(zarr_db):
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    zarr_db.add_data(DATA_ID, 'x', data, 'train')

    result = zarr_db.get_data(DATA_ID, 'x', 'train', 1)
    np.testing.assert_array_equal(result, data[1])


def test_get_data_by_slice(zarr_db):
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    zarr_db.add_data(DATA_ID, 'x', data, 'train')

    result = zarr_db.get_data(DATA_ID, 'x', 'train', slice(0, 2))
    np.testing.assert_array_equal(result, data[0:2])


def test_update_data_by_index(zarr_db):
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    zarr_db.add_data(DATA_ID, 'x', data, 'train')

    zarr_db.update_data(DATA_ID, 'x', np.array([99.0, 99.0]), 'train', 0)
    np.testing.assert_array_equal(zarr_db._db[DATA_ID]['train']['x'][0], [99.0, 99.0])


def test_update_data_all(zarr_db):
    data = np.array([[1.0], [2.0]])
    zarr_db.add_data(DATA_ID, 'x', data, 'train')

    new_data = np.array([[9.0], [9.0]])
    zarr_db.update_data(DATA_ID, 'x', new_data, 'train', None)
    np.testing.assert_array_equal(zarr_db._db[DATA_ID]['train']['x'][:], new_data)


def test_delete_data(zarr_db):
    zarr_db.add_data(DATA_ID, 'x', np.array([[1.0]]), 'train')
    zarr_db.delete_data(DATA_ID, 'x', 'train')

    assert 'x' not in list(zarr_db._db[DATA_ID]['train'].array_keys())


def test_delete_data_not_found_raises(zarr_db):
    with pytest.raises(KeyError):
        zarr_db.delete_data(DATA_ID, 'nonexistent', 'train')


def test_get_metadata_returns_correct_structure(zarr_db):
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    zarr_db.add_data(DATA_ID, 'x', data, 'train')

    metadata = zarr_db.get_metadata(DATA_ID, 'train')
    assert 'x' in metadata
    assert metadata['x']['total_events'] == 2
    assert metadata['x']['shape'] == (2,)
    assert metadata['x']['type'] == 'float64'
    assert metadata['x']['backend'] == 'zarr'


def test_get_metadata_empty_phase(zarr_db):
    metadata = zarr_db.get_metadata(DATA_ID, 'train')
    assert metadata == {}


def test_get_metadata_unknown_data_id(tmp_path):
    db = ZarrDatabase(output_dir=str(tmp_path), chunk=100, mode='w')
    metadata = db.get_metadata('unknown', 'train')
    assert metadata == {}


def test_readonly_mode_can_read(tmp_path):
    db_w = ZarrDatabase(output_dir=str(tmp_path), chunk=100, mode='w')
    db_w.initialize(DATA_ID)
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    db_w.add_data(DATA_ID, 'x', data, 'train')

    db_r = ZarrDatabase(output_dir=str(tmp_path), chunk=100, mode='r')
    result = db_r.get_data(DATA_ID, 'x', 'train', None)
    np.testing.assert_array_equal(result, data)


def test_readonly_mode_initialize_is_noop(tmp_path):
    db_w = ZarrDatabase(output_dir=str(tmp_path), chunk=100, mode='w')
    db_w.initialize(DATA_ID)
    db_w.add_data(DATA_ID, 'x', np.array([[1.0]]), 'train')

    db_r = ZarrDatabase(output_dir=str(tmp_path), chunk=100, mode='r')
    db_r.initialize(DATA_ID)  # should not raise
    result = db_r.get_data(DATA_ID, 'x', 'train', None)
    np.testing.assert_array_equal(result, [[1.0]])
