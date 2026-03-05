"""Unit tests for NumpyDatabase."""
import numpy as np
import pytest

from storegate.database.numpy_database import NumpyDatabase


DATA_ID = 'test_data'


@pytest.fixture
def db():
    d = NumpyDatabase()
    d.initialize(DATA_ID)
    return d


def test_initialize_creates_structure(db):
    for phase in ['train', 'valid', 'test']:
        assert db._chunks[DATA_ID][phase] == {}
        assert db._metadata[DATA_ID][phase] == {}


def test_initialize_idempotent(db):
    db.initialize(DATA_ID)  # second call should not overwrite
    db.add_data(DATA_ID, 'x', np.array([[1.0]]), 'train')
    db.initialize(DATA_ID)  # should not clear existing data
    assert 'x' in db._chunks[DATA_ID]['train']


def test_add_data_new_var(db):
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    db.add_data(DATA_ID, 'x', data, 'train')

    np.testing.assert_array_equal(db.get_data(DATA_ID, 'x', 'train', None), data)
    meta = db._metadata[DATA_ID]['train']['x']
    assert meta['backend'] == 'numpy'
    assert meta['type'] == 'float64'
    assert meta['shape'] == (2,)
    assert meta['total_events'] == 2


def test_add_data_appends_to_existing(db):
    data1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    data2 = np.array([[5.0, 6.0]])
    db.add_data(DATA_ID, 'x', data1, 'train')
    db.add_data(DATA_ID, 'x', data2, 'train')

    expected = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    np.testing.assert_array_equal(db.get_data(DATA_ID, 'x', 'train', None), expected)
    assert db._metadata[DATA_ID]['train']['x']['total_events'] == 3


def test_add_data_accumulates_chunks_without_concat(db):
    """Chunks are stored as a list; concatenation is deferred until get_data."""
    db.add_data(DATA_ID, 'x', np.array([[1.0]]), 'train')
    db.add_data(DATA_ID, 'x', np.array([[2.0]]), 'train')
    db.add_data(DATA_ID, 'x', np.array([[3.0]]), 'train')

    assert len(db._chunks[DATA_ID]['train']['x']) == 3
    assert db._cache[DATA_ID]['train']['x'] is None  # not yet materialized


def test_get_data_materializes_and_caches(db):
    """After get_data the cache is populated; a second call does not replace it."""
    db.add_data(DATA_ID, 'x', np.array([[1.0]]), 'train')
    db.add_data(DATA_ID, 'x', np.array([[2.0]]), 'train')

    db.get_data(DATA_ID, 'x', 'train', None)
    cache_obj = db._cache[DATA_ID]['train']['x']
    assert cache_obj is not None

    db.get_data(DATA_ID, 'x', 'train', None)
    assert db._cache[DATA_ID]['train']['x'] is cache_obj  # not re-concatenated


def test_materialize_collapses_chunks_to_single_element(db):
    """After get_data, the chunk list is collapsed to one element to free original arrays."""
    db.add_data(DATA_ID, 'x', np.array([[1.0]]), 'train')
    db.add_data(DATA_ID, 'x', np.array([[2.0]]), 'train')
    db.add_data(DATA_ID, 'x', np.array([[3.0]]), 'train')
    assert len(db._chunks[DATA_ID]['train']['x']) == 3

    db.get_data(DATA_ID, 'x', 'train', None)
    assert len(db._chunks[DATA_ID]['train']['x']) == 1


def test_add_data_after_materialize_appends_correctly(db):
    """Appending after materialization still produces the correct concatenated result."""
    db.add_data(DATA_ID, 'x', np.array([[1.0], [2.0]]), 'train')
    db.get_data(DATA_ID, 'x', 'train', None)  # materializes and collapses

    db.add_data(DATA_ID, 'x', np.array([[3.0]]), 'train')
    result = db.get_data(DATA_ID, 'x', 'train', None)
    np.testing.assert_array_equal(result, [[1.0], [2.0], [3.0]])


def test_add_data_invalidates_cache(db):
    """add_data after a get_data call must invalidate the cache."""
    db.add_data(DATA_ID, 'x', np.array([[1.0]]), 'train')
    db.get_data(DATA_ID, 'x', 'train', None)  # populate cache
    assert db._cache[DATA_ID]['train']['x'] is not None

    db.add_data(DATA_ID, 'x', np.array([[2.0]]), 'train')
    assert db._cache[DATA_ID]['train']['x'] is None  # invalidated


def test_add_data_independent_phases(db):
    db.add_data(DATA_ID, 'x', np.array([[1.0]]), 'train')
    db.add_data(DATA_ID, 'x', np.array([[2.0], [3.0]]), 'valid')

    assert db._metadata[DATA_ID]['train']['x']['total_events'] == 1
    assert db._metadata[DATA_ID]['valid']['x']['total_events'] == 2


def test_get_data_all(db):
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    db.add_data(DATA_ID, 'x', data, 'train')

    result = db.get_data(DATA_ID, 'x', 'train', None)
    np.testing.assert_array_equal(result, data)


def test_get_data_by_int_index(db):
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    db.add_data(DATA_ID, 'x', data, 'train')

    result = db.get_data(DATA_ID, 'x', 'train', 1)
    np.testing.assert_array_equal(result, data[1])


def test_get_data_by_slice(db):
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    db.add_data(DATA_ID, 'x', data, 'train')

    result = db.get_data(DATA_ID, 'x', 'train', slice(0, 2))
    np.testing.assert_array_equal(result, data[0:2])


def test_update_data_by_index(db):
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    db.add_data(DATA_ID, 'x', data, 'train')

    db.update_data(DATA_ID, 'x', np.array([99.0, 99.0]), 'train', 0)
    np.testing.assert_array_equal(db.get_data(DATA_ID, 'x', 'train', 0), [99.0, 99.0])
    np.testing.assert_array_equal(db.get_data(DATA_ID, 'x', 'train', 1), data[1])


def test_update_data_by_slice(db):
    data = np.array([[1.0], [2.0], [3.0]])
    db.add_data(DATA_ID, 'x', data, 'train')

    db.update_data(DATA_ID, 'x', np.array([[9.0], [9.0]]), 'train', slice(0, 2))
    np.testing.assert_array_equal(db.get_data(DATA_ID, 'x', 'train', 0), [9.0])
    np.testing.assert_array_equal(db.get_data(DATA_ID, 'x', 'train', 2), [3.0])


def test_update_data_all(db):
    data = np.array([[1.0], [2.0]])
    db.add_data(DATA_ID, 'x', data, 'train')

    new_data = np.array([[9.0], [9.0]])
    db.update_data(DATA_ID, 'x', new_data, 'train', None)
    np.testing.assert_array_equal(db.get_data(DATA_ID, 'x', 'train', None), new_data)


def test_update_data_collapses_chunks(db):
    """After update_data the chunk list must be collapsed to a single element."""
    db.add_data(DATA_ID, 'x', np.array([[1.0]]), 'train')
    db.add_data(DATA_ID, 'x', np.array([[2.0]]), 'train')
    assert len(db._chunks[DATA_ID]['train']['x']) == 2

    db.update_data(DATA_ID, 'x', np.array([9.0]), 'train', 0)
    assert len(db._chunks[DATA_ID]['train']['x']) == 1


def test_delete_data(db):
    db.add_data(DATA_ID, 'x', np.array([[1.0]]), 'train')
    db.delete_data(DATA_ID, 'x', 'train')

    assert 'x' not in db._chunks[DATA_ID]['train']
    assert 'x' not in db._metadata[DATA_ID]['train']


def test_delete_data_not_found_raises(db):
    with pytest.raises(KeyError):
        db.delete_data(DATA_ID, 'nonexistent', 'train')


def test_get_metadata_returns_correct_structure(db):
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    db.add_data(DATA_ID, 'x', data, 'train')

    metadata = db.get_metadata(DATA_ID, 'train')
    assert 'x' in metadata
    assert metadata['x']['total_events'] == 2
    assert metadata['x']['shape'] == (2,)
    assert metadata['x']['type'] == 'float64'
    assert metadata['x']['backend'] == 'numpy'


def test_get_metadata_empty_phase(db):
    metadata = db.get_metadata(DATA_ID, 'train')
    assert metadata == {}


def test_get_metadata_unknown_data_id():
    db = NumpyDatabase()
    metadata = db.get_metadata('unknown', 'train')
    assert metadata == {}
