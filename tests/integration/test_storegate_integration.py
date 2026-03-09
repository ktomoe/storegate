"""Integration tests for StoreGate end-to-end workflows."""
import numpy as np
import pytest
import zarr

from storegate import StoreGate


DATA_ID = 'dataset'


def _make_store(tmp_path, *, backend: str | None = None):
    store = StoreGate(output_dir=str(tmp_path), mode='w')
    store.set_data_id(DATA_ID)
    if backend is not None:
        store.set_backend(backend)
    return store


def _reopen_store(tmp_path):
    store = StoreGate(output_dir=str(tmp_path), mode='r')
    store.set_data_id(DATA_ID)
    return store


def _assert_lengths(sg, **phase_lengths):
    for phase, expected in phase_lengths.items():
        assert len(sg[phase]) == expected


@pytest.fixture
def sg(tmp_path):
    return _make_store(tmp_path)


# ---------------------------------------------------------------------------
# Full workflow tests
# ---------------------------------------------------------------------------

def test_full_workflow_zarr_backend(sg):
    x_train = np.random.rand(100, 5).astype(np.float32)
    x_valid = np.random.rand(20, 5).astype(np.float32)
    x_test = np.random.rand(10, 5).astype(np.float32)

    sg.add_data_splits('x', train=x_train, valid=x_valid, test=x_test)
    sg.compile()

    _assert_lengths(sg, train=100, valid=20, test=10)

    np.testing.assert_array_equal(sg.get_data('x', 'train', None), x_train)
    np.testing.assert_array_equal(sg.get_data('x', 'valid', None), x_valid)
    np.testing.assert_array_equal(sg.get_data('x', 'test', None), x_test)


def test_full_workflow_numpy_backend(tmp_path):
    sg = _make_store(tmp_path, backend='numpy')

    x_train = np.random.rand(50, 3).astype(np.float64)
    sg.add_data('x', x_train, phase='train')
    sg.compile()

    np.testing.assert_array_equal(sg.get_data('x', 'train', None), x_train)
    assert len(sg['train']) == 50


def test_full_workflow_multiple_variables(sg):
    x = np.random.rand(80, 4).astype(np.float32)
    y = np.random.randint(0, 10, size=(80, 1))

    sg.add_data('x', x, phase='train')
    sg.add_data('y', y, phase='train')
    sg.compile()

    np.testing.assert_array_equal(sg.get_data('x', 'train', None), x)
    np.testing.assert_array_equal(sg.get_data('y', 'train', None), y)
    assert len(sg['train']) == 80


# ---------------------------------------------------------------------------
# Indexed access
# ---------------------------------------------------------------------------

def test_indexed_get_via_bracket_notation(sg):
    data = np.arange(30).reshape(10, 3).astype(np.float32)
    sg.add_data('x', data, phase='train')

    np.testing.assert_array_equal(sg['train']['x'][0], data[0])
    np.testing.assert_array_equal(sg['train']['x'][2:5], data[2:5])


def test_indexed_update_via_bracket_notation(sg):
    data = np.zeros((5, 2), dtype=np.float32)
    sg.add_data('x', data, phase='train')

    new_row = np.array([9.0, 9.0], dtype=np.float32)
    sg['train']['x'][2] = new_row

    result = sg.get_data('x', 'train', 2)
    np.testing.assert_array_equal(result, new_row)


# ---------------------------------------------------------------------------
# Backend switching
# ---------------------------------------------------------------------------

def test_zarr_and_numpy_backends_are_independent(sg):
    zarr_data = np.array([[1.0, 2.0], [3.0, 4.0]])
    numpy_data = np.array([[5.0, 6.0], [7.0, 8.0]])

    sg.set_backend('zarr')
    sg.add_data('x', zarr_data, phase='train')

    sg.set_backend('numpy')
    sg.add_data('x', numpy_data, phase='train')

    sg.set_backend('zarr')
    np.testing.assert_array_equal(sg.get_data('x', 'train', None), zarr_data)

    sg.set_backend('numpy')
    np.testing.assert_array_equal(sg.get_data('x', 'train', None), numpy_data)


def test_using_backend_context_manager_in_workflow(sg):
    zarr_data = np.array([[1.0], [2.0], [3.0]])
    sg.add_data('x', zarr_data, phase='train')

    with sg.using_backend('numpy'):
        sg.add_data('x', np.array([[9.0]]), phase='train')
        assert sg.get_backend() == 'numpy'

    assert sg.get_backend() == 'zarr'
    np.testing.assert_array_equal(sg.get_data('x', 'train', None), zarr_data)


# ---------------------------------------------------------------------------
# copy_to_memory / copy_to_storage
# ---------------------------------------------------------------------------

def test_copy_to_memory_round_trip(sg):
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    sg.add_data('x', data, phase='train')
    sg.copy_to_memory('x', phase='train')

    with sg.using_backend('numpy'):
        result = sg.get_data('x', 'train', None)
    np.testing.assert_array_equal(result, data)


def test_copy_to_storage_round_trip(sg):
    sg.set_backend('numpy')
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    sg.add_data('x', data, phase='train')
    sg.copy_to_storage('x', phase='train')

    with sg.using_backend('zarr'):
        result = sg.get_data('x', 'train', None)
    np.testing.assert_array_equal(result, data)


def test_copy_to_memory_with_rename(sg):
    data = np.array([[1.0], [2.0]])
    sg.add_data('x', data, phase='train')
    sg.copy_to_memory('x', phase='train', output_var_name='x_mem')

    with sg.using_backend('numpy'):
        assert 'x_mem' in sg.get_var_names('train')
        np.testing.assert_array_equal(sg.get_data('x_mem', 'train', None), data)


# ---------------------------------------------------------------------------
# Persistence (write → read)
# ---------------------------------------------------------------------------

def test_zarr_data_persists_across_instances(tmp_path):
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    sg_write = _make_store(tmp_path)
    sg_write.add_data('x', data, phase='train')

    sg_read = _reopen_store(tmp_path)
    result = sg_read.get_data('x', 'train', None)
    np.testing.assert_array_equal(result, data)


def test_zarr_data_persists_multiple_phases(tmp_path):
    x_train = np.random.rand(10, 3).astype(np.float32)
    x_valid = np.random.rand(5, 3).astype(np.float32)

    sg_write = _make_store(tmp_path)
    sg_write.add_data_splits('x', train=x_train, valid=x_valid)

    sg_read = _reopen_store(tmp_path)
    np.testing.assert_array_equal(sg_read.get_data('x', 'train', None), x_train)
    np.testing.assert_array_equal(sg_read.get_data('x', 'valid', None), x_valid)


# ---------------------------------------------------------------------------
# Snapshot / restore
# ---------------------------------------------------------------------------

def test_snapshot_restore_round_trip_restores_data_and_metadata(tmp_path):
    sg = _make_store(tmp_path)
    train = np.arange(12, dtype=np.float32).reshape(6, 2)
    valid = np.arange(4, dtype=np.float32).reshape(2, 2)

    sg.add_data('x', train, phase='train')
    sg.add_data('x', valid, phase='valid')
    sg.compile()
    sg.snapshot('baseline')

    sg.update_data('x', np.full((2,), 99.0, dtype=np.float32), phase='train', index=0)
    sg.add_data('y', np.array([[1.0], [2.0]], dtype=np.float32), phase='valid')
    sg.compile()

    sg.restore('baseline')

    assert sg.get_backend() == 'zarr'
    np.testing.assert_array_equal(sg.get_data('x', 'train', None), train)
    np.testing.assert_array_equal(sg.get_data('x', 'valid', None), valid)
    assert 'y' not in sg.get_var_names('valid')
    assert len(sg['train']) == 6
    assert len(sg['valid']) == 2


def test_snapshot_restore_clears_numpy_backend_for_current_data_id(tmp_path):
    sg = _make_store(tmp_path)
    train = np.arange(6, dtype=np.float32).reshape(3, 2)

    sg.add_data('x', train, phase='train')
    sg.compile()
    sg.snapshot('baseline')
    sg.copy_to_memory('x', phase='train')
    with sg.using_backend('numpy'):
        sg.add_data('cache_only', np.ones((3, 1), dtype=np.float32), phase='train')

    sg.update_data('x', np.full((2,), 77.0, dtype=np.float32), phase='train', index=0)
    sg.restore('baseline')

    np.testing.assert_array_equal(sg.get_data('x', 'train', None), train)
    assert sg.get_backend() == 'zarr'
    with sg.using_backend('numpy'):
        assert sg.get_var_names('train') == []


def test_snapshot_duplicate_name_raises(tmp_path):
    sg = _make_store(tmp_path)
    sg.add_data('x', np.array([[1.0]], dtype=np.float32), phase='train')
    sg.snapshot('baseline')

    with pytest.raises(ValueError, match='already exists'):
        sg.snapshot('baseline')


def test_restore_missing_snapshot_raises(tmp_path):
    sg = _make_store(tmp_path)
    sg.add_data('x', np.array([[1.0]], dtype=np.float32), phase='train')

    with pytest.raises(KeyError, match='missing'):
        sg.restore('missing')


# ---------------------------------------------------------------------------
# Multiple data_id
# ---------------------------------------------------------------------------

def test_multiple_data_ids_are_isolated(tmp_path):
    sg = StoreGate(output_dir=str(tmp_path), mode='w')

    sg.set_data_id('dataset_a')
    sg.add_data('x', np.array([[1.0], [2.0]]), phase='train')
    sg.compile()

    sg.set_data_id('dataset_b')
    sg.add_data('x', np.array([[3.0], [4.0], [5.0]]), phase='train')
    sg.compile()

    sg.set_data_id('dataset_a')
    assert len(sg['train']) == 2

    sg.set_data_id('dataset_b')
    assert len(sg['train']) == 3


def test_switching_data_id_preserves_each_metadata(tmp_path):
    sg = StoreGate(output_dir=str(tmp_path), mode='w')

    sg.set_data_id('a')
    sg.add_data('x', np.array([[1.0]]), phase='train')

    sg.set_data_id('b')
    sg.add_data('y', np.array([[2.0]]), phase='train')

    sg.set_data_id('a')
    assert 'x' in sg.get_var_names('train')
    assert 'y' not in sg.get_var_names('train')

    sg.set_data_id('b')
    assert 'y' in sg.get_var_names('train')
    assert 'x' not in sg.get_var_names('train')


# ---------------------------------------------------------------------------
# Incremental data addition
# ---------------------------------------------------------------------------

def test_incremental_add_and_compile(sg):
    chunk1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    chunk2 = np.array([[5.0, 6.0], [7.0, 8.0]])

    sg.add_data('x', chunk1, phase='train')
    sg.add_data('x', chunk2, phase='train')
    sg.compile()

    expected = np.concatenate([chunk1, chunk2], axis=0)
    np.testing.assert_array_equal(sg.get_data('x', 'train', None), expected)
    assert len(sg['train']) == 4


# ---------------------------------------------------------------------------
# Metadata persistence (zarr attrs)
# ---------------------------------------------------------------------------

def test_metadata_restored_after_reopen(tmp_path):
    """After compile(), reopening the store allows len() without calling compile() again."""
    sg = _make_store(tmp_path)
    sg.add_data('x', np.array([[1.0], [2.0], [3.0]]), phase='train')
    sg.add_data('x', np.array([[4.0]]), phase='valid')
    sg.compile()

    sg2 = _reopen_store(tmp_path)
    _assert_lengths(sg2, train=3, valid=1)


def test_metadata_restored_multi_data_id(tmp_path):
    """Each data_id's metadata is independently restored in a store with multiple data_ids."""
    sg = StoreGate(output_dir=str(tmp_path), mode='w')

    sg.set_data_id('a')
    sg.add_data('x', np.array([[1.0], [2.0]]), phase='train')
    sg.compile()

    sg.set_data_id('b')
    sg.add_data('x', np.array([[3.0], [4.0], [5.0]]), phase='train')
    sg.compile()

    sg2 = StoreGate(output_dir=str(tmp_path), mode='r')
    sg2.set_data_id('a')
    assert len(sg2['train']) == 2

    sg2.set_data_id('b')
    assert len(sg2['train']) == 3


def test_metadata_invalidated_after_add_data(tmp_path):
    """add_data after compile() persists compiled=False so reopening correctly requires compile()."""
    sg = _make_store(tmp_path)
    sg.add_data('x', np.array([[1.0], [2.0]]), phase='train')
    sg.compile()
    sg.add_data('x', np.array([[3.0]]), phase='train')  # invalidates compiled flag

    sg2 = _reopen_store(tmp_path)

    with pytest.raises(ValueError, match='compile'):
        len(sg2['train'])


def test_metadata_not_present_requires_compile(tmp_path):
    """A store written without compile() still requires compile() after reopening."""
    sg = _make_store(tmp_path)
    sg.add_data('x', np.array([[1.0]]), phase='train')
    # compile() not called

    sg2 = _reopen_store(tmp_path)

    with pytest.raises(ValueError, match='compile'):
        len(sg2['train'])


def test_stale_metadata_invalidated_after_external_zarr_mutation(tmp_path):
    """Reopen must distrust persisted compiled metadata if zarr data changed externally."""
    sg = _make_store(tmp_path)
    sg.add_data('x', np.array([[1.0], [2.0]], dtype=np.float32), phase='train')
    sg.compile()

    root = zarr.open(str(tmp_path), mode='a')
    root[DATA_ID]['train']['x'].append(np.array([[3.0]], dtype=np.float32))

    sg2 = _reopen_store(tmp_path)

    with pytest.raises(ValueError, match='compile'):
        len(sg2['train'])
    assert sg2.get_data('x', 'train', None).shape[0] == 3
    assert sg2._metadata[DATA_ID]['compiled']['zarr'] is False
    assert sg2._metadata[DATA_ID]['sizes']['zarr'] == {}


def test_metadata_compiled_false_does_not_persist(tmp_path):
    """Re-compiling after add_data saves compiled=True so len() works again after reopen."""
    sg = _make_store(tmp_path)
    sg.add_data('x', np.array([[1.0], [2.0]]), phase='train')
    sg.compile()
    sg.add_data('x', np.array([[3.0]]), phase='train')
    sg.compile()  # recompile: saves compiled=True, size=3

    sg2 = _reopen_store(tmp_path)
    _assert_lengths(sg2, train=3)
