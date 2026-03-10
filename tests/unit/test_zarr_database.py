"""Unit tests for ZarrDatabase."""
from unittest.mock import patch

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


# ---------- Line 33: read-only initialize with missing data_id ----------
def test_readonly_initialize_missing_data_id_raises(tmp_path):
    db_w = ZarrDatabase(output_dir=str(tmp_path), chunk=100, mode='w')
    db_w.initialize(DATA_ID)

    db_r = ZarrDatabase(output_dir=str(tmp_path), chunk=100, mode='r')
    with pytest.raises(KeyError, match="does not exist"):
        db_r.initialize('nonexistent')


# ---------- Line 44: _load_storegate_meta for unknown data_id ----------
def test_load_storegate_meta_unknown_data_id_returns_empty(tmp_path):
    db = ZarrDatabase(output_dir=str(tmp_path), chunk=100, mode='w')
    meta = db._load_storegate_meta('nonexistent')
    assert meta == {}


# ---------- Line 65: _require_writable in read-only mode ----------
def test_snapshot_readonly_raises_runtime_error(tmp_path):
    db_w = ZarrDatabase(output_dir=str(tmp_path), chunk=100, mode='w')
    db_w.initialize(DATA_ID)

    db_r = ZarrDatabase(output_dir=str(tmp_path), chunk=100, mode='r')
    with pytest.raises(RuntimeError, match='requires write access'):
        db_r.snapshot_data_id(DATA_ID, 'snap1')


def test_restore_readonly_raises_runtime_error(tmp_path):
    db_w = ZarrDatabase(output_dir=str(tmp_path), chunk=100, mode='w')
    db_w.initialize(DATA_ID)

    db_r = ZarrDatabase(output_dir=str(tmp_path), chunk=100, mode='r')
    with pytest.raises(RuntimeError, match='requires write access'):
        db_r.restore_data_id(DATA_ID, 'snap1')


# ---------- Lines 80-86: _copy_array with empty array ----------
def test_snapshot_with_empty_array(tmp_path):
    db = ZarrDatabase(output_dir=str(tmp_path), chunk=100, mode='w')
    db.initialize(DATA_ID)
    db.add_data(DATA_ID, 'x', np.array([[1.0, 2.0]]), 'train')

    # Make it empty: delete and recreate with 0 events
    del db._db[DATA_ID]['train']['x']
    db._db[DATA_ID]['train'].create_array(
        name='x', data=np.empty((0, 2), dtype=np.float64), chunks=(100, 2),
    )

    db.snapshot_data_id(DATA_ID, 'snap_empty')
    snap = db._db['.storegate_snapshots'][DATA_ID]['snap_empty']['train']['x']
    assert snap.shape == (0, 2)


# ---------- Line 101: _copy_array multi-chunk append ----------
def test_snapshot_multi_chunk_array(tmp_path):
    db = ZarrDatabase(output_dir=str(tmp_path), chunk=2, mode='w')
    db.initialize(DATA_ID)
    data = np.arange(10, dtype=np.float64).reshape(5, 2)
    db.add_data(DATA_ID, 'x', data, 'train')

    db.snapshot_data_id(DATA_ID, 'snap1')
    snap_data = db._db['.storegate_snapshots'][DATA_ID]['snap1']['train']['x'][:]
    np.testing.assert_array_equal(snap_data, data)


# ---------- Line 154: _append_var_name duplicate is no-op ----------
def test_append_var_name_duplicate_is_noop(zarr_db):
    zarr_db.add_data(DATA_ID, 'x', np.array([[1.0]]), 'train')
    # Manually call again — should not duplicate
    zarr_db._append_var_name(DATA_ID, 'train', 'x')
    meta = zarr_db.load_meta_attrs(DATA_ID)
    assert meta['var_names']['train'] == ['x']


# ---------- Line 161: _remove_var_name for nonexistent var ----------
def test_remove_var_name_nonexistent_is_noop(zarr_db):
    zarr_db.add_data(DATA_ID, 'x', np.array([[1.0]]), 'train')
    zarr_db._remove_var_name(DATA_ID, 'train', 'nonexistent')
    meta = zarr_db.load_meta_attrs(DATA_ID)
    assert meta['var_names']['train'] == ['x']


# ---------- Line 187: _rename_var_name when source not in list ----------
def test_rename_var_name_source_missing_appends(zarr_db):
    zarr_db.add_data(DATA_ID, 'x', np.array([[1.0]]), 'train')
    # Rename a var_name that is not registered in meta
    zarr_db._rename_var_name(DATA_ID, 'train', 'ghost', 'z')
    meta = zarr_db.load_meta_attrs(DATA_ID)
    assert 'z' in meta['var_names']['train']


# ---------- Line 197: _event_chunk_size fallback ----------
def test_event_chunk_size_no_chunks_attr(zarr_db):
    """When chunks attribute is not a tuple, falls back to shape[0]."""
    class FakeArr:
        shape = (42, 3)
        chunks = None  # type: ignore[assignment]
    result = zarr_db._event_chunk_size(FakeArr())
    assert result == 42


# ---------- Lines 262-263: iter_data_chunks with empty array ----------
def test_iter_data_chunks_empty_array(tmp_path):
    db = ZarrDatabase(output_dir=str(tmp_path), chunk=100, mode='w')
    db.initialize(DATA_ID)

    db._db[DATA_ID]['train'].create_array(
        name='x', data=np.empty((0, 3), dtype=np.float64), chunks=(100, 3),
    )

    chunks = list(db.iter_data_chunks(DATA_ID, 'x', 'train'))
    assert len(chunks) == 1
    assert chunks[0].shape == (0, 3)


# ---------- Line 286: rename_data same name is no-op ----------
def test_rename_data_same_name_is_noop(zarr_db):
    zarr_db.add_data(DATA_ID, 'x', np.array([[1.0]]), 'train')
    zarr_db.rename_data(DATA_ID, 'x', 'x', 'train')
    assert list(zarr_db.get_metadata(DATA_ID, 'train')) == ['x']


# ---------- Line 295: rename_data with empty array ----------
def test_rename_data_empty_array(tmp_path):
    db = ZarrDatabase(output_dir=str(tmp_path), chunk=100, mode='w')
    db.initialize(DATA_ID)

    db._db[DATA_ID]['train'].create_array(
        name='x', data=np.empty((0, 2), dtype=np.float64), chunks=(100, 2),
    )
    db._append_var_name(DATA_ID, 'train', 'x')

    db.rename_data(DATA_ID, 'x', 'y', 'train')
    assert 'x' not in list(db._db[DATA_ID]['train'].array_keys())
    assert db._db[DATA_ID]['train']['y'].shape == (0, 2)


# ---------- Lines 300-303: rename_data rollback on failure ----------
def test_rename_data_rollback_on_failure(zarr_db):
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    zarr_db.add_data(DATA_ID, 'x', data, 'train')

    with patch.object(zarr_db, 'add_data', side_effect=RuntimeError('injected')):
        with pytest.raises(RuntimeError, match='injected'):
            zarr_db.rename_data(DATA_ID, 'x', 'z', 'train')

    # Source must be intact, destination must not exist
    np.testing.assert_array_equal(
        zarr_db.get_data(DATA_ID, 'x', 'train', None), data,
    )
    assert 'z' not in list(zarr_db._db[DATA_ID]['train'].array_keys())


def test_rename_data_rollback_cleans_partial_destination(tmp_path):
    """If add_data succeeds once then fails, partial destination is removed."""
    db = ZarrDatabase(output_dir=str(tmp_path), chunk=2, mode='w')
    db.initialize(DATA_ID)
    data = np.arange(6, dtype=np.float64).reshape(3, 2)
    db.add_data(DATA_ID, 'x', data, 'train')

    original_add = db.add_data
    call_count = 0

    def fail_on_second(data_id, var_name, d, phase):
        nonlocal call_count
        if var_name == 'z':
            call_count += 1
            if call_count >= 2:
                raise RuntimeError('mid-copy failure')
        return original_add(data_id, var_name, d, phase)

    with patch.object(db, 'add_data', side_effect=fail_on_second):
        with pytest.raises(RuntimeError, match='mid-copy failure'):
            db.rename_data(DATA_ID, 'x', 'z', 'train')

    assert 'z' not in list(db._db[DATA_ID]['train'].array_keys())
    np.testing.assert_array_equal(db.get_data(DATA_ID, 'x', 'train', None), data)


# ---------- Line 339: snapshot_data_id missing data_id ----------
def test_snapshot_missing_data_id_raises(tmp_path):
    db = ZarrDatabase(output_dir=str(tmp_path), chunk=100, mode='w')
    with pytest.raises(KeyError, match='not found'):
        db.snapshot_data_id('nonexistent', 'snap1')


# ---------- Lines 351-353: snapshot_data_id cleanup on failure ----------
def test_snapshot_cleanup_on_copy_failure(tmp_path):
    db = ZarrDatabase(output_dir=str(tmp_path), chunk=100, mode='w')
    db.initialize(DATA_ID)
    db.add_data(DATA_ID, 'x', np.array([[1.0]]), 'train')

    with patch.object(
        db, '_copy_group_contents', side_effect=RuntimeError('copy failed'),
    ):
        with pytest.raises(RuntimeError, match='copy failed'):
            db.snapshot_data_id(DATA_ID, 'snap1')

    # Incomplete snapshot must be cleaned up
    snap_root = db._db['.storegate_snapshots'][DATA_ID]
    assert 'snap1' not in list(snap_root.group_keys())


# ---------- Line 364: restore no snapshot root for data_id ----------
def test_restore_missing_snapshot_data_id_raises(tmp_path):
    db = ZarrDatabase(output_dir=str(tmp_path), chunk=100, mode='w')
    db.initialize(DATA_ID)
    db.add_data(DATA_ID, 'x', np.array([[1.0]]), 'train')
    db.snapshot_data_id(DATA_ID, 'snap1')

    with pytest.raises(KeyError, match='not found'):
        db.restore_data_id('other_id', 'snap1')


# ---------- Line 368: restore missing snapshot_name ----------
def test_restore_missing_snapshot_name_raises(tmp_path):
    db = ZarrDatabase(output_dir=str(tmp_path), chunk=100, mode='w')
    db.initialize(DATA_ID)
    db.add_data(DATA_ID, 'x', np.array([[1.0]]), 'train')
    db.snapshot_data_id(DATA_ID, 'snap1')

    with pytest.raises(KeyError, match='not found'):
        db.restore_data_id(DATA_ID, 'nonexistent_snap')


# ---------- Staging name collision loop ----------
def test_restore_staging_name_collision(tmp_path):
    db = ZarrDatabase(output_dir=str(tmp_path), chunk=100, mode='w')
    db.initialize(DATA_ID)
    db.add_data(DATA_ID, 'x', np.array([[1.0]]), 'train')
    db.snapshot_data_id(DATA_ID, 'snap1')

    # Pre-create colliding staging names
    staging_root = db._db.require_group('.storegate_restore_staging')
    staging_root.require_group(DATA_ID)
    staging_root.require_group(f'{DATA_ID}_1')

    # Restore should still succeed by incrementing suffix
    db.restore_data_id(DATA_ID, 'snap1')
    result = db.get_data(DATA_ID, 'x', 'train', None)
    np.testing.assert_array_equal(result, [[1.0]])


# ---------- Phase 1 failure: staging copy fails, original intact ----------
def test_restore_staging_failure_preserves_original(tmp_path):
    db = ZarrDatabase(output_dir=str(tmp_path), chunk=100, mode='w')
    db.initialize(DATA_ID)
    original_data = np.array([[1.0, 2.0], [3.0, 4.0]])
    db.add_data(DATA_ID, 'x', original_data, 'train')
    db.snapshot_data_id(DATA_ID, 'snap1')

    # Modify data after snapshot
    db.update_data(DATA_ID, 'x', np.array([[9.0, 9.0], [9.0, 9.0]]), 'train', None)

    with patch.object(
        db, '_copy_group_contents', side_effect=RuntimeError('staging failed'),
    ):
        with pytest.raises(RuntimeError, match='staging failed'):
            db.restore_data_id(DATA_ID, 'snap1')

    # Original data must be completely intact (staging-first guarantee)
    result = db.get_data(DATA_ID, 'x', 'train', None)
    np.testing.assert_array_equal(result, [[9.0, 9.0], [9.0, 9.0]])


# ---------- Phase 2 failure: install fails, recovery from staging ----------
def test_restore_install_failure_recovers_from_staging(tmp_path):
    db = ZarrDatabase(output_dir=str(tmp_path), chunk=100, mode='w')
    db.initialize(DATA_ID)
    original_data = np.array([[1.0, 2.0]])
    db.add_data(DATA_ID, 'x', original_data, 'train')
    db.snapshot_data_id(DATA_ID, 'snap1')

    # Mock _copy_array (non-recursive) to avoid recursion counting issues
    # with _copy_group_contents.
    original_copy_array = db._copy_array
    call_count = 0

    def fail_on_install(src_group, src_name, dst_group, dst_name):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError('install failed')
        return original_copy_array(src_group, src_name, dst_group, dst_name)

    with patch.object(db, '_copy_array', side_effect=fail_on_install):
        with pytest.raises(RuntimeError, match='install failed'):
            db.restore_data_id(DATA_ID, 'snap1')

    # Recovery from staging should have restored the snapshot data
    result = db.get_data(DATA_ID, 'x', 'train', None)
    np.testing.assert_array_equal(result, original_data)

    # Staging cleaned up after successful recovery
    staging_root = db._db['.storegate_restore_staging']
    assert len(list(staging_root.group_keys())) == 0


# ---------- Phase 2 double failure: staging preserved for manual recovery ----------
def test_restore_double_failure_preserves_staging(tmp_path):
    db = ZarrDatabase(output_dir=str(tmp_path), chunk=100, mode='w')
    db.initialize(DATA_ID)
    db.add_data(DATA_ID, 'x', np.array([[1.0]]), 'train')
    db.snapshot_data_id(DATA_ID, 'snap1')

    original_copy_array = db._copy_array
    call_count = 0

    def fail_after_staging(src_group, src_name, dst_group, dst_name):
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise RuntimeError('copy always fails')
        return original_copy_array(src_group, src_name, dst_group, dst_name)

    with patch.object(db, '_copy_array', side_effect=fail_after_staging):
        with pytest.raises(RuntimeError, match='copy always fails'):
            db.restore_data_id(DATA_ID, 'snap1')

    # Staging must be preserved for manual recovery
    staging_root = db._db['.storegate_restore_staging']
    assert len(list(staging_root.group_keys())) > 0


# ---------- Restore without existing data_id (no backup needed) ----------
def test_restore_no_existing_data_id(tmp_path):
    """Restore into a data_id that was deleted after snapshot."""
    db = ZarrDatabase(output_dir=str(tmp_path), chunk=100, mode='w')
    db.initialize(DATA_ID)
    data = np.array([[1.0, 2.0]])
    db.add_data(DATA_ID, 'x', data, 'train')
    db.snapshot_data_id(DATA_ID, 'snap1')

    # Delete the data_id entirely
    del db._db[DATA_ID]

    db.restore_data_id(DATA_ID, 'snap1')
    result = db._db[DATA_ID]['train']['x'][:]
    np.testing.assert_array_equal(result, data)


# ---------- No snapshot root at all ----------
def test_restore_no_snapshot_root_raises(tmp_path):
    db = ZarrDatabase(output_dir=str(tmp_path), chunk=100, mode='w')
    db.initialize(DATA_ID)

    with pytest.raises(KeyError, match='not found'):
        db.restore_data_id(DATA_ID, 'snap1')
