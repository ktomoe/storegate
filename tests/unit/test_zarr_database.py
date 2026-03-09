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


def test_add_data_persists_var_name_registration_order(zarr_db):
    zarr_db.add_data(DATA_ID, 'x', np.array([[1.0]]), 'train')
    zarr_db.add_data(DATA_ID, 'y', np.array([[2.0]]), 'train')

    meta = zarr_db.load_meta_attrs(DATA_ID)
    assert meta['var_names']['train'] == ['x', 'y']


def test_add_data_zarr_incompatible_dtype_raises(zarr_db):
    data = np.array([{'a': 1}, {'b': 2}], dtype=object)

    with pytest.raises(ValueError, match='not persistable to the zarr backend'):
        zarr_db.add_data(DATA_ID, 'x', data, 'train')


def test_add_data_shape_mismatch_raises(zarr_db):
    data1 = np.array([[1.0, 2.0], [3.0, 4.0]])       # shape (2, 2)
    data2 = np.array([[5.0, 6.0, 7.0]])                # shape (1, 3)
    zarr_db.add_data(DATA_ID, 'x', data1, 'train')

    with pytest.raises(ValueError, match='Shape mismatch'):
        zarr_db.add_data(DATA_ID, 'x', data2, 'train')


def test_add_data_shape_mismatch_error_message(zarr_db):
    data1 = np.ones((3, 4, 5))
    data2 = np.ones((2, 4, 6))
    zarr_db.add_data(DATA_ID, 'x', data1, 'train')

    with pytest.raises(ValueError, match=r'\(4, 5\).*\(4, 6\)'):
        zarr_db.add_data(DATA_ID, 'x', data2, 'train')


def test_add_data_dtype_safe_upcast_succeeds(zarr_db):
    # float64 existing + int32 incoming: promoted = float64 == existing → safe
    data1 = np.array([[1.0, 2.0]], dtype=np.float64)
    data2 = np.array([[3, 4]], dtype=np.int32)
    zarr_db.add_data(DATA_ID, 'x', data1, 'train')
    zarr_db.add_data(DATA_ID, 'x', data2, 'train')  # should not raise

    result = zarr_db._db[DATA_ID]['train']['x'][:]
    assert result.dtype == np.float64
    assert result.shape[0] == 2


def test_add_data_dtype_lossy_cast_raises(zarr_db):
    # float32 existing + float64 incoming: promoted = float64 != float32 → ValueError
    data1 = np.array([[1.0, 2.0]], dtype=np.float32)
    data2 = np.array([[3.0, 4.0]], dtype=np.float64)
    zarr_db.add_data(DATA_ID, 'x', data1, 'train')

    with pytest.raises(ValueError, match='dtype mismatch'):
        zarr_db.add_data(DATA_ID, 'x', data2, 'train')


def test_add_data_dtype_lossy_cast_error_message(zarr_db):
    # int32 existing + int64 incoming → promoted = int64 != int32 → raises
    data1 = np.array([[1, 2]], dtype=np.int32)
    data2 = np.array([[3, 4]], dtype=np.int64)
    zarr_db.add_data(DATA_ID, 'x', data1, 'train')

    with pytest.raises(ValueError, match='int32'):
        zarr_db.add_data(DATA_ID, 'x', data2, 'train')


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


def test_update_data_shape_mismatch_raises_value_error(zarr_db):
    zarr_db.add_data(DATA_ID, 'x', np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), 'train')

    with pytest.raises(ValueError, match='Shape mismatch for update'):
        zarr_db.update_data(DATA_ID, 'x', np.array([9.0], dtype=np.float32), 'train', slice(0, 2))


def test_update_data_shape_mismatch_does_not_mutate_existing_data(zarr_db):
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    zarr_db.add_data(DATA_ID, 'x', data, 'train')

    with pytest.raises(ValueError):
        zarr_db.update_data(DATA_ID, 'x', np.array([9.0], dtype=np.float32), 'train', slice(0, 2))

    result = zarr_db.get_data(DATA_ID, 'x', 'train', None)
    np.testing.assert_array_equal(result, data)


def test_update_data_dtype_lossy_cast_raises_value_error(zarr_db):
    zarr_db.add_data(DATA_ID, 'x', np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), 'train')

    with pytest.raises(ValueError, match='dtype mismatch for update'):
        zarr_db.update_data(
            DATA_ID,
            'x',
            np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float64),
            'train',
            None,
        )


def test_update_data_dtype_lossy_cast_does_not_mutate_existing_data(zarr_db):
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    zarr_db.add_data(DATA_ID, 'x', data, 'train')

    with pytest.raises(ValueError):
        zarr_db.update_data(
            DATA_ID,
            'x',
            np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float64),
            'train',
            None,
        )

    result = zarr_db.get_data(DATA_ID, 'x', 'train', None)
    np.testing.assert_array_equal(result, data)
    assert result.dtype == np.float32


def test_update_data_dtype_safe_cast_succeeds(zarr_db):
    zarr_db.add_data(DATA_ID, 'x', np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64), 'train')

    zarr_db.update_data(
        DATA_ID,
        'x',
        np.array([[1, 2], [3, 4]], dtype=np.int32),
        'train',
        None,
    )

    result = zarr_db.get_data(DATA_ID, 'x', 'train', None)
    np.testing.assert_array_equal(result, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
    assert result.dtype == np.float64


def test_delete_data(zarr_db):
    zarr_db.add_data(DATA_ID, 'x', np.array([[1.0]]), 'train')
    zarr_db.delete_data(DATA_ID, 'x', 'train')

    assert 'x' not in list(zarr_db._db[DATA_ID]['train'].array_keys())
    assert zarr_db.load_meta_attrs(DATA_ID)['var_names']['train'] == []


def test_delete_data_not_found_raises(zarr_db):
    with pytest.raises(KeyError):
        zarr_db.delete_data(DATA_ID, 'nonexistent', 'train')


def test_rename_data_streams_in_chunks_and_preserves_order(tmp_path, monkeypatch):
    db = ZarrDatabase(output_dir=str(tmp_path), chunk=2, mode='w')
    db.initialize(DATA_ID)

    x = np.arange(10, dtype=np.float32).reshape(5, 2)
    y = np.arange(5, dtype=np.float32).reshape(5, 1)
    db.add_data(DATA_ID, 'x', x, 'train')
    db.add_data(DATA_ID, 'y', y, 'train')

    original_add_data = db.add_data
    copied_chunk_sizes: list[int] = []

    def wrapped_add_data(data_id, var_name, data, phase):
        if var_name == 'z':
            copied_chunk_sizes.append(len(data))
        return original_add_data(data_id, var_name, data, phase)

    monkeypatch.setattr(db, 'add_data', wrapped_add_data)

    db.rename_data(DATA_ID, 'x', 'z', 'train')

    assert copied_chunk_sizes == [2, 2, 1]
    assert list(db.get_metadata(DATA_ID, 'train')) == ['z', 'y']
    np.testing.assert_array_equal(db.get_data(DATA_ID, 'z', 'train', None), x)


def test_rename_data_existing_destination_raises(zarr_db):
    zarr_db.add_data(DATA_ID, 'x', np.array([[1.0]]), 'train')
    zarr_db.add_data(DATA_ID, 'y', np.array([[2.0]]), 'train')

    with pytest.raises(ValueError, match='already exists'):
        zarr_db.rename_data(DATA_ID, 'x', 'y', 'train')


def test_get_metadata_returns_correct_structure(zarr_db):
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    zarr_db.add_data(DATA_ID, 'x', data, 'train')

    metadata = zarr_db.get_metadata(DATA_ID, 'train')
    assert 'x' in metadata
    assert metadata['x']['total_events'] == 2
    assert metadata['x']['shape'] == (2,)
    assert metadata['x']['type'] == 'float64'
    assert metadata['x']['backend'] == 'zarr'


def test_get_metadata_returns_variables_in_registration_order(zarr_db):
    zarr_db.add_data(DATA_ID, 'x', np.array([[1.0]]), 'train')
    zarr_db.add_data(DATA_ID, 'y', np.array([[2.0]]), 'train')

    metadata = zarr_db.get_metadata(DATA_ID, 'train')
    assert list(metadata) == ['x', 'y']


def test_save_meta_attrs_preserves_var_name_order(zarr_db):
    zarr_db.add_data(DATA_ID, 'x', np.array([[1.0]]), 'train')
    zarr_db.save_meta_attrs(DATA_ID, {'compiled': {'zarr': True}})

    meta = zarr_db.load_meta_attrs(DATA_ID)
    assert meta['compiled']['zarr'] is True
    assert meta['var_names']['train'] == ['x']


def test_get_metadata_empty_phase(zarr_db):
    metadata = zarr_db.get_metadata(DATA_ID, 'train')
    assert metadata == {}


def test_initialize_bootstraps_var_names_for_existing_arrays(tmp_path):
    db = ZarrDatabase(output_dir=str(tmp_path), chunk=100, mode='w')
    db.initialize(DATA_ID)
    db._db[DATA_ID]['train'].create_array(name='b', data=np.array([[1.0]]), chunks=(100, 1))
    db._db[DATA_ID]['train'].create_array(name='a', data=np.array([[2.0]]), chunks=(100, 1))
    db._db[DATA_ID].attrs.clear()

    reopened = ZarrDatabase(output_dir=str(tmp_path), chunk=100, mode='a')
    reopened.initialize(DATA_ID)

    assert reopened.load_meta_attrs(DATA_ID)['var_names']['train'] == ['a', 'b']


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


def test_readonly_mode_initialize_validates_existence(tmp_path):
    db_w = ZarrDatabase(output_dir=str(tmp_path), chunk=100, mode='w')
    db_w.initialize(DATA_ID)
    db_w.add_data(DATA_ID, 'x', np.array([[1.0]]), 'train')

    db_r = ZarrDatabase(output_dir=str(tmp_path), chunk=100, mode='r')
    db_r.initialize(DATA_ID)  # should not raise
    result = db_r.get_data(DATA_ID, 'x', 'train', None)
    np.testing.assert_array_equal(result, [[1.0]])
