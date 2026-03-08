"""Unit tests for StoreGate class."""
import logging

import numpy as np
import pytest

from storegate import StoreGate
from storegate.storegate import _AllPhaseAccessor, _PhaseAccessor


DATA_ID = 'test_data'


class _ListHandler(logging.Handler):
    """Handler that collects formatted messages into a list."""
    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(self.format(record))


@pytest.fixture
def capture_log():
    """Capture storegate logger output (propagate=False, so caplog won't work)."""
    from storegate import logger as sg_logger
    handler = _ListHandler()
    handler.setLevel(logging.DEBUG)
    sg_logger._logger.addHandler(handler)
    yield handler.messages
    sg_logger._logger.removeHandler(handler)


@pytest.fixture
def sg(tmp_path):
    return StoreGate(output_dir=str(tmp_path), mode='w')


@pytest.fixture
def sg_id(tmp_path):
    store = StoreGate(output_dir=str(tmp_path), mode='w')
    store.set_data_id(DATA_ID)
    return store


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_init_data_id_is_none(sg):
    assert sg.data_id is None


# ---------------------------------------------------------------------------
# output_dir validation
# ---------------------------------------------------------------------------

def test_init_invalid_mode_raises(tmp_path):
    with pytest.raises(ValueError, match="Invalid mode"):
        StoreGate(output_dir=str(tmp_path), mode='x')


def test_init_empty_output_dir_raises(tmp_path):
    with pytest.raises(ValueError, match="non-empty string"):
        StoreGate(output_dir='', mode='w')


def test_init_readonly_nonexistent_path_raises(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        StoreGate(output_dir=str(tmp_path / 'missing'), mode='r')


def test_init_write_nonexistent_parent_raises(tmp_path):
    with pytest.raises(ValueError, match="Parent directory"):
        StoreGate(output_dir=str(tmp_path / 'missing' / 'store'), mode='w')


def test_init_with_data_id(tmp_path):
    sg = StoreGate(output_dir=str(tmp_path), mode='w', data_id=DATA_ID)
    assert sg.data_id == DATA_ID


def test_set_data_id(sg):
    sg.set_data_id(DATA_ID)
    assert sg.data_id == DATA_ID


@pytest.mark.parametrize('valid_id', [
    'abc',
    'ABC',
    'test_data',
    'experiment-01',
    'a' * 128,
    'mix_AND-123',
])
def test_set_data_id_valid_values(sg, valid_id):
    sg.set_data_id(valid_id)
    assert sg.data_id == valid_id


@pytest.mark.parametrize('invalid_id', [
    '',
    'a' * 129,
    '../secret',
    'a/b',
    'hello world',
    'tab\there',
    'new\nline',
    'semi;colon',
])
def test_set_data_id_invalid_values_raise(sg, invalid_id):
    with pytest.raises(ValueError, match='Invalid data_id'):
        sg.set_data_id(invalid_id)


def test_init_with_invalid_data_id_raises(tmp_path):
    with pytest.raises(ValueError, match='Invalid data_id'):
        StoreGate(output_dir=str(tmp_path), mode='w', data_id='bad/id')


def test_set_data_id_initializes_metadata(sg):
    sg.set_data_id(DATA_ID)
    assert DATA_ID in sg._metadata
    assert 'compiled' in sg._metadata[DATA_ID]
    assert 'sizes' in sg._metadata[DATA_ID]


def test_set_data_id_twice_does_not_reset_metadata(sg):
    sg.set_data_id(DATA_ID)
    sg.add_data('x', np.array([[1.0]]), phase='train')
    sg.set_data_id(DATA_ID)  # second call
    # metadata should still exist
    assert DATA_ID in sg._metadata


def test_repr_no_data_id(sg):
    assert 'None' in repr(sg)


def test_repr_with_data_id(sg_id):
    r = repr(sg_id)
    assert DATA_ID in r


def test_context_manager(tmp_path):
    with StoreGate(output_dir=str(tmp_path), mode='w') as sg:
        sg.set_data_id(DATA_ID)
        sg.add_data('x', np.array([[1.0]]), phase='train')
    # __exit__ returns False (does not suppress exceptions)


# ---------------------------------------------------------------------------
# require_data_id decorator
# ---------------------------------------------------------------------------

def test_add_data_requires_data_id(sg):
    with pytest.raises(RuntimeError, match='set_data_id'):
        sg.add_data('x', np.array([[1.0]]), phase='train')


def test_get_data_requires_data_id(sg):
    with pytest.raises(RuntimeError):
        sg.get_data('x', 'train')


def test_compile_requires_data_id(sg):
    with pytest.raises(RuntimeError):
        sg.compile()


# ---------------------------------------------------------------------------
# Backend management
# ---------------------------------------------------------------------------

def test_default_backend_is_zarr(sg_id):
    assert sg_id.get_backend() == 'zarr'


def test_set_backend_numpy(sg_id):
    sg_id.set_backend('numpy')
    assert sg_id.get_backend() == 'numpy'


def test_set_backend_invalid_raises(sg_id):
    with pytest.raises(ValueError, match='Unsupported backend'):
        sg_id.set_backend('invalid')


def test_using_backend_switches_and_restores(sg_id):
    assert sg_id.get_backend() == 'zarr'
    with sg_id.using_backend('numpy'):
        assert sg_id.get_backend() == 'numpy'
    assert sg_id.get_backend() == 'zarr'


def test_using_backend_restores_on_exception(sg_id):
    try:
        with sg_id.using_backend('numpy'):
            raise ValueError('intentional error')
    except ValueError:
        pass
    assert sg_id.get_backend() == 'zarr'


def test_using_backend_invalid_raises(sg_id):
    with pytest.raises(ValueError):
        with sg_id.using_backend('invalid'):
            pass


# ---------------------------------------------------------------------------
# add_data / get_data / update_data / delete_data
# ---------------------------------------------------------------------------

def test_add_and_get_data(sg_id):
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    sg_id.add_data('x', data, phase='train')

    result = sg_id.get_data('x', 'train', index=None)
    np.testing.assert_array_equal(result, data)


def test_add_data_invalid_phase_raises(sg_id):
    with pytest.raises(ValueError, match='Invalid phase'):
        sg_id.add_data('x', np.array([[1.0]]), phase='bad_phase')


def test_add_data_converts_to_ndarray(sg_id):
    sg_id.add_data('x', [[1.0, 2.0], [3.0, 4.0]], phase='train')
    result = sg_id.get_data('x', 'train', index=None)
    assert isinstance(result, np.ndarray)


def test_add_data_splits_all_phases(sg_id):
    train = np.array([[1.0], [2.0]])
    valid = np.array([[3.0]])
    test = np.array([[4.0], [5.0]])
    sg_id.add_data_splits('x', train=train, valid=valid, test=test)

    np.testing.assert_array_equal(sg_id.get_data('x', 'train', None), train)
    np.testing.assert_array_equal(sg_id.get_data('x', 'valid', None), valid)
    np.testing.assert_array_equal(sg_id.get_data('x', 'test', None), test)


def test_add_data_splits_partial(sg_id):
    train = np.array([[1.0], [2.0]])
    sg_id.add_data_splits('x', train=train)

    np.testing.assert_array_equal(sg_id.get_data('x', 'train', None), train)
    assert 'x' not in sg_id.get_var_names('valid')
    assert 'x' not in sg_id.get_var_names('test')


def test_update_data_by_index(sg_id):
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    sg_id.add_data('x', data, phase='train')
    sg_id.update_data('x', np.array([99.0, 99.0]), phase='train', index=0)

    result = sg_id.get_data('x', 'train', 0)
    np.testing.assert_array_equal(result, [99.0, 99.0])


def test_update_data_shape_mismatch_raises_value_error(sg_id):
    sg_id.add_data('x', np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), phase='train')

    with pytest.raises(ValueError, match='Shape mismatch for update'):
        sg_id.update_data('x', np.array([9.0], dtype=np.float32), phase='train', index=slice(0, 2))


def test_update_data_dtype_lossy_cast_raises_value_error_on_numpy_backend(sg_id):
    with sg_id.using_backend('numpy'):
        sg_id.add_data('x', np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), phase='train')

        with pytest.raises(ValueError, match='dtype mismatch for update'):
            sg_id.update_data(
                'x',
                np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float64),
                phase='train',
                index=None,
            )


def test_delete_data_single_phase(sg_id):
    sg_id.add_data('x', np.array([[1.0]]), phase='train')
    sg_id.delete_data('x', phase='train')

    assert 'x' not in sg_id.get_var_names('train')


def test_delete_data_all_phases(sg_id):
    sg_id.add_data('x', np.array([[1.0]]), phase='train')
    sg_id.add_data('x', np.array([[2.0]]), phase='valid')
    sg_id.add_data('x', np.array([[3.0]]), phase='test')
    sg_id.delete_data('x', phase='all')

    assert 'x' not in sg_id.get_var_names('train')
    assert 'x' not in sg_id.get_var_names('valid')
    assert 'x' not in sg_id.get_var_names('test')


def test_delete_data_all_phases_partial_existence(sg_id):
    # 'x' exists only in train and valid, not in test
    sg_id.add_data('x', np.array([[1.0]]), phase='train')
    sg_id.add_data('x', np.array([[2.0]]), phase='valid')
    sg_id.delete_data('x', phase='all')  # should not raise KeyError

    assert 'x' not in sg_id.get_var_names('train')
    assert 'x' not in sg_id.get_var_names('valid')
    assert 'x' not in sg_id.get_var_names('test')


def test_get_var_names(sg_id):
    sg_id.add_data('x', np.array([[1.0]]), phase='train')
    sg_id.add_data('y', np.array([[2.0]]), phase='train')

    names = sg_id.get_var_names('train')
    assert 'x' in names
    assert 'y' in names


def test_get_var_names_empty_phase(sg_id):
    names = sg_id.get_var_names('train')
    assert names == []


# ---------------------------------------------------------------------------
# compile
# ---------------------------------------------------------------------------

def test_compile_succeeds_with_consistent_data(sg_id):
    sg_id.add_data('x', np.array([[1.0], [2.0]]), phase='train')
    sg_id.add_data('y', np.array([[3.0], [4.0]]), phase='train')
    sg_id.compile()  # should not raise


def test_compile_raises_on_inconsistent_event_counts(sg_id):
    sg_id.add_data('x', np.array([[1.0], [2.0]]), phase='train')
    sg_id.add_data('y', np.array([[3.0]]), phase='train')

    with pytest.raises(ValueError, match='Inconsistent event counts'):
        sg_id.compile()


def test_compile_sets_compiled_flag(sg_id):
    sg_id.add_data('x', np.array([[1.0], [2.0]]), phase='train')
    sg_id.compile()

    backend = sg_id.get_backend()
    assert sg_id._metadata[DATA_ID]['compiled'][backend] is True


def test_compile_sets_size(sg_id):
    sg_id.add_data('x', np.array([[1.0], [2.0], [3.0]]), phase='train')
    sg_id.compile()

    backend = sg_id.get_backend()
    assert sg_id._metadata[DATA_ID]['sizes'][backend]['train'] == 3


def test_add_data_after_compile_resets_flag(sg_id):
    sg_id.add_data('x', np.array([[1.0]]), phase='train')
    sg_id.compile()
    sg_id.add_data('x', np.array([[2.0]]), phase='train')

    backend = sg_id.get_backend()
    assert sg_id._metadata[DATA_ID]['compiled'][backend] is False


# ---------------------------------------------------------------------------
# compile — cross_backend
# ---------------------------------------------------------------------------

def test_compile_cross_backend_default_false_does_not_check(sg_id):
    """Default compile() ignores cross-backend differences."""
    data = np.array([[1.0], [2.0]])
    sg_id.add_data('x', data, phase='train')                      # zarr: 2 events
    with sg_id.using_backend('numpy'):
        sg_id.add_data('x', np.array([[9.0]]), phase='train')     # numpy: 1 event
    sg_id.compile()  # should not raise (cross_backend=False)


def test_compile_cross_backend_passes_when_no_common_vars(sg_id):
    """Variables in only one backend are not compared."""
    sg_id.add_data('x', np.array([[1.0], [2.0]]), phase='train')  # zarr only
    with sg_id.using_backend('numpy'):
        sg_id.add_data('y', np.array([[9.0]]), phase='train')      # numpy only
    sg_id.compile(cross_backend=True)  # should not raise


def test_compile_cross_backend_passes_when_metadata_match(sg_id):
    data = np.array([[1.0], [2.0]])
    sg_id.add_data('x', data, phase='train')                      # zarr: 2
    sg_id.copy_to_memory('x', phase='train')                      # numpy: 2
    sg_id.compile(cross_backend=True)  # should not raise


def test_compile_cross_backend_raises_on_count_mismatch(sg_id):
    sg_id.add_data('x', np.array([[1.0], [2.0]]), phase='train')  # zarr: 2
    with sg_id.using_backend('numpy'):
        sg_id.add_data('x', np.array([[9.0]]), phase='train')     # numpy: 1
    with pytest.raises(ValueError, match='Cross-backend inconsistency'):
        sg_id.compile(cross_backend=True)


def test_compile_cross_backend_error_contains_var_name(sg_id):
    sg_id.add_data('score', np.array([[1.0], [2.0]]), phase='train')
    with sg_id.using_backend('numpy'):
        sg_id.add_data('score', np.array([[9.0]]), phase='train')
    with pytest.raises(ValueError, match="'score'"):
        sg_id.compile(cross_backend=True)


def test_compile_cross_backend_error_contains_phase(sg_id):
    sg_id.add_data('x', np.array([[1.0], [2.0]]), phase='valid')
    with sg_id.using_backend('numpy'):
        sg_id.add_data('x', np.array([[9.0]]), phase='valid')
    with pytest.raises(ValueError, match="'valid'"):
        sg_id.compile(cross_backend=True)


def test_compile_cross_backend_error_contains_counts(sg_id):
    sg_id.add_data('x', np.array([[1.0], [2.0], [3.0]]), phase='train')  # zarr: 3
    with sg_id.using_backend('numpy'):
        sg_id.add_data('x', np.array([[9.0]]), phase='train')             # numpy: 1
    with pytest.raises(ValueError, match=r'zarr=3.*numpy=1|numpy=1.*zarr=3'):
        sg_id.compile(cross_backend=True)


def test_compile_cross_backend_raises_on_type_mismatch(sg_id):
    sg_id.add_data('x', np.array([[1.0], [2.0]], dtype=np.float32), phase='train')
    with sg_id.using_backend('numpy'):
        sg_id.add_data('x', np.array([[1], [2]], dtype=np.int64), phase='train')
    with pytest.raises(ValueError, match=r'type: zarr=float32, numpy=int64'):
        sg_id.compile(cross_backend=True)


def test_compile_cross_backend_raises_on_shape_mismatch(sg_id):
    sg_id.add_data('x', np.array([[1.0], [2.0]], dtype=np.float32), phase='train')
    with sg_id.using_backend('numpy'):
        sg_id.add_data('x', np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), phase='train')
    with pytest.raises(ValueError, match=r'shape: zarr=\(1,\), numpy=\(2,\)'):
        sg_id.compile(cross_backend=True)


def test_compile_cross_backend_reports_all_mismatches_at_once(sg_id):
    """All mismatching variables are collected before raising."""
    sg_id.add_data('x', np.array([[1.0], [2.0]]), phase='train')
    sg_id.add_data('y', np.array([[3.0], [4.0]]), phase='train')
    with sg_id.using_backend('numpy'):
        sg_id.add_data('x', np.array([[9.0]]), phase='train')  # mismatch
        sg_id.add_data('y', np.array([[8.0]]), phase='train')  # mismatch
    with pytest.raises(ValueError) as exc_info:
        sg_id.compile(cross_backend=True)
    msg = str(exc_info.value)
    assert "'x'" in msg
    assert "'y'" in msg


def test_compile_cross_backend_restores_backend_after_check(sg_id):
    """Active backend must be unchanged after cross_backend check."""
    sg_id.set_backend('numpy')
    sg_id.add_data('x', np.array([[1.0]]), phase='train')
    sg_id.compile(cross_backend=True)
    assert sg_id.get_backend() == 'numpy'


def test_compile_cross_backend_failure_does_not_mark_backend_compiled(sg_id):
    """Failed cross-backend validation must not leave the active backend compiled."""
    sg_id.add_data('x', np.array([[1.0], [2.0]]), phase='train')
    with sg_id.using_backend('numpy'):
        sg_id.add_data('x', np.array([[9.0]]), phase='train')
        with pytest.raises(ValueError, match='Cross-backend inconsistency'):
            sg_id.compile(cross_backend=True)
        assert sg_id._metadata[DATA_ID]['compiled']['numpy'] is False
        assert 'train' not in sg_id._metadata[DATA_ID]['sizes']['numpy']


def test_compile_cross_backend_failure_keeps_len_unavailable(sg_id):
    """len() must remain unavailable after a failed cross-backend compile."""
    sg_id.add_data('x', np.array([[1.0], [2.0]]), phase='train')
    with sg_id.using_backend('numpy'):
        sg_id.add_data('x', np.array([[9.0]]), phase='train')
        with pytest.raises(ValueError, match='Cross-backend inconsistency'):
            sg_id.compile(cross_backend=True)
        with pytest.raises(ValueError, match='supported only after compile'):
            len(sg_id['train'])


# ---------------------------------------------------------------------------
# __getitem__ / _PhaseAccessor / _VarAccessor
# ---------------------------------------------------------------------------

def test_getitem_returns_phase_accessor(sg_id):
    accessor = sg_id['train']
    assert isinstance(accessor, _PhaseAccessor)


def test_getitem_all_returns_all_phase_accessor(sg_id):
    accessor = sg_id['all']
    assert isinstance(accessor, _AllPhaseAccessor)


def test_all_phase_accessor_delitem_removes_from_all_phases(sg_id):
    sg_id.add_data('x', np.array([[1.0]]), phase='train')
    sg_id.add_data('x', np.array([[2.0]]), phase='valid')
    del sg_id['all']['x']
    assert 'x' not in sg_id.get_var_names('train')
    assert 'x' not in sg_id.get_var_names('valid')


def test_all_phase_accessor_getitem_raises(sg_id):
    with pytest.raises(NotImplementedError, match="sg\\['all'\\]"):
        _ = sg_id['all']['x']


def test_all_phase_accessor_setitem_raises(sg_id):
    with pytest.raises(NotImplementedError, match="sg\\['all'\\]"):
        sg_id['all']['x'] = np.array([[1.0]])


def test_all_phase_accessor_contains_raises(sg_id):
    with pytest.raises(NotImplementedError, match="sg\\['all'\\]"):
        _ = 'x' in sg_id['all']


def test_all_phase_accessor_iter_raises(sg_id):
    with pytest.raises(NotImplementedError, match="sg\\['all'\\]"):
        _ = list(sg_id['all'])


def test_all_phase_accessor_len_raises(sg_id):
    with pytest.raises(NotImplementedError, match="sg\\['all'\\]"):
        len(sg_id['all'])


def test_getitem_invalid_phase_raises(sg_id):
    with pytest.raises(NotImplementedError):
        sg_id['bad_phase']


def test_phase_accessor_setitem(sg_id):
    data = np.array([[1.0, 2.0]])
    sg_id['train']['x'] = data

    result = sg_id.get_data('x', 'train', None)
    np.testing.assert_array_equal(result, data)


def test_phase_accessor_delitem(sg_id):
    sg_id.add_data('x', np.array([[1.0]]), phase='train')
    del sg_id['train']['x']

    assert 'x' not in sg_id.get_var_names('train')


def test_phase_accessor_contains(sg_id):
    sg_id.add_data('x', np.array([[1.0]]), phase='train')

    assert 'x' in sg_id['train']
    assert 'y' not in sg_id['train']


def test_phase_accessor_iter(sg_id):
    sg_id.add_data('x', np.array([[1.0]]), phase='train')
    sg_id.add_data('y', np.array([[2.0]]), phase='train')

    # Use a comprehension to avoid Python calling __len__ (which requires compile)
    names = [n for n in sg_id['train']]
    assert 'x' in names
    assert 'y' in names


def test_phase_accessor_items(sg_id):
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    sg_id.add_data('x', data, phase='train')

    items = list(sg_id['train'].items())
    assert len(items) == 1
    assert items[0][0] == 'x'
    np.testing.assert_array_equal(items[0][1], data)


def test_phase_accessor_len_after_compile(sg_id):
    sg_id.add_data('x', np.array([[1.0], [2.0], [3.0]]), phase='train')
    sg_id.compile()

    assert len(sg_id['train']) == 3


def test_phase_accessor_len_before_compile_raises(sg_id):
    sg_id.add_data('x', np.array([[1.0], [2.0]]), phase='train')

    with pytest.raises(ValueError, match='compile'):
        len(sg_id['train'])


def test_phase_accessor_len_empty_phase_returns_zero(sg_id):
    # train has data, valid has none; compile() sets sizes['valid'] = None
    sg_id.add_data('x', np.array([[1.0], [2.0]]), phase='train')
    sg_id.compile()

    assert len(sg_id['valid']) == 0


def test_var_accessor_getitem_int(sg_id):
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    sg_id.add_data('x', data, phase='train')

    result = sg_id['train']['x'][0]
    np.testing.assert_array_equal(result, data[0])


def test_var_accessor_getitem_slice(sg_id):
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    sg_id.add_data('x', data, phase='train')

    result = sg_id['train']['x'][1:3]
    np.testing.assert_array_equal(result, data[1:3])


def test_var_accessor_setitem(sg_id):
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    sg_id.add_data('x', data, phase='train')

    sg_id['train']['x'][0] = np.array([99.0, 99.0])
    result = sg_id.get_data('x', 'train', 0)
    np.testing.assert_array_equal(result, [99.0, 99.0])


def test_var_accessor_getitem_invalid_type_raises(sg_id):
    sg_id.add_data('x', np.array([[1.0]]), phase='train')

    with pytest.raises(NotImplementedError):
        _ = sg_id['train']['x']['bad_index']


def test_var_accessor_setitem_invalid_type_raises(sg_id):
    sg_id.add_data('x', np.array([[1.0]]), phase='train')

    with pytest.raises(ValueError):
        sg_id['train']['x']['bad_index'] = np.array([1.0])


# ---------------------------------------------------------------------------
# copy_to_memory / copy_to_storage
# ---------------------------------------------------------------------------

def test_copy_to_memory(sg_id):
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    sg_id.add_data('x', data, phase='train')
    sg_id.copy_to_memory('x', phase='train')

    with sg_id.using_backend('numpy'):
        result = sg_id.get_data('x', 'train', None)
    np.testing.assert_array_equal(result, data)


def test_copy_to_memory_already_exists_raises(sg_id):
    data = np.array([[1.0]])
    sg_id.add_data('x', data, phase='train')
    sg_id.copy_to_memory('x', phase='train')

    with pytest.raises(ValueError, match='already exists'):
        sg_id.copy_to_memory('x', phase='train')


def test_copy_to_memory_with_output_var_name(sg_id):
    data = np.array([[1.0, 2.0]])
    sg_id.add_data('x', data, phase='train')
    sg_id.copy_to_memory('x', phase='train', output_var_name='x_mem')

    with sg_id.using_backend('numpy'):
        result = sg_id.get_data('x_mem', 'train', None)
    np.testing.assert_array_equal(result, data)


def test_copy_to_storage(sg_id):
    sg_id.set_backend('numpy')
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    sg_id.add_data('x', data, phase='train')
    sg_id.copy_to_storage('x', phase='train')

    with sg_id.using_backend('zarr'):
        result = sg_id.get_data('x', 'train', None)
    np.testing.assert_array_equal(result, data)


# ---------------------------------------------------------------------------
# show_info
# ---------------------------------------------------------------------------

def test_show_info_requires_data_id(sg):
    with pytest.raises(RuntimeError, match='set_data_id'):
        sg.show_info()


def test_show_info_runs_without_error(sg_id):
    sg_id.add_data('x', np.array([[1.0, 2.0], [3.0, 4.0]]), phase='train')
    sg_id.compile()
    sg_id.show_info()  # should not raise


def test_show_info_displays_var_name(sg_id, capture_log):
    sg_id.add_data('x', np.array([[1.0], [2.0]]), phase='train')
    sg_id.compile()
    sg_id.show_info()
    text = '\n'.join(capture_log)
    assert 'x' in text


def test_show_info_displays_data_id(sg_id, capture_log):
    sg_id.add_data('x', np.array([[1.0]]), phase='train')
    sg_id.compile()
    sg_id.show_info()
    text = '\n'.join(capture_log)
    assert DATA_ID in text


def test_show_info_displays_compiled_status(sg_id, capture_log):
    sg_id.add_data('x', np.array([[1.0]]), phase='train')
    sg_id.compile()
    sg_id.show_info()
    text = '\n'.join(capture_log)
    assert 'True' in text


def test_show_info_before_compile_shows_false(sg_id, capture_log):
    sg_id.add_data('x', np.array([[1.0]]), phase='train')
    sg_id.show_info()
    text = '\n'.join(capture_log)
    assert 'False' in text


def test_show_info_multiple_phases(sg_id, capture_log):
    sg_id.add_data('x', np.array([[1.0]]), phase='train')
    sg_id.add_data('x', np.array([[2.0]]), phase='valid')
    sg_id.compile()
    sg_id.show_info()
    text = '\n'.join(capture_log)
    assert 'train' in text
    assert 'valid' in text


def test_show_info_multiple_variables(sg_id, capture_log):
    sg_id.add_data('x', np.array([[1.0], [2.0]]), phase='train')
    sg_id.add_data('y', np.array([[3.0], [4.0]]), phase='train')
    sg_id.compile()
    sg_id.show_info()
    text = '\n'.join(capture_log)
    assert 'x' in text
    assert 'y' in text


def test_show_info_empty_storegate(sg_id, capture_log):
    sg_id.compile()
    sg_id.show_info()
    text = '\n'.join(capture_log)
    assert DATA_ID in text


def test_show_info_displays_dtype(sg_id, capture_log):
    sg_id.add_data('x', np.array([[1.0]], dtype=np.float32), phase='train')
    sg_id.compile()
    sg_id.show_info()
    text = '\n'.join(capture_log)
    assert 'float32' in text


def test_show_info_displays_total_events(sg_id, capture_log):
    sg_id.add_data('x', np.array([[1.0], [2.0], [3.0]]), phase='train')
    sg_id.compile()
    sg_id.show_info()
    text = '\n'.join(capture_log)
    assert '3' in text


def test_show_info_displays_shape(sg_id, capture_log):
    sg_id.add_data('x', np.array([[1.0, 2.0], [3.0, 4.0]]), phase='train')
    sg_id.compile()
    sg_id.show_info()
    text = '\n'.join(capture_log)
    assert '(2,)' in text


def test_show_info_displays_backend(sg_id, capture_log):
    sg_id.add_data('x', np.array([[1.0]]), phase='train')
    sg_id.compile()
    sg_id.show_info()
    text = '\n'.join(capture_log)
    assert 'zarr' in text


def test_compile_show_info_true_calls_show_info(sg_id, capture_log):
    sg_id.add_data('x', np.array([[1.0]]), phase='train')
    sg_id.compile(show_info=True)
    text = '\n'.join(capture_log)
    assert DATA_ID in text
    assert 'x' in text


def test_compile_show_info_false_does_not_call_show_info(sg_id, capture_log):
    sg_id.add_data('x', np.array([[1.0]]), phase='train')
    sg_id.compile(show_info=False)
    text = '\n'.join(capture_log)
    # No table header columns should appear
    assert 'var_name' not in text


# ---------------------------------------------------------------------------
# copy_to_storage (continued)
# ---------------------------------------------------------------------------

def test_copy_to_storage_already_exists_raises(tmp_path):
    # Start with numpy-only data, copy to storage, then try to copy again
    sg = StoreGate(output_dir=str(tmp_path), mode='w')
    sg.set_data_id(DATA_ID)
    sg.set_backend('numpy')

    data = np.array([[1.0]])
    sg.add_data('x', data, phase='train')
    sg.copy_to_storage('x', phase='train')

    with pytest.raises(ValueError, match='already exists'):
        sg.copy_to_storage('x', phase='train')


# ---------------------------------------------------------------------------
# Additional coverage tests — uncovered lines
# ---------------------------------------------------------------------------

def test_add_data_scalar_raises(sg_id):
    """Line 407: scalar (0-dim) data must be rejected."""
    with pytest.raises(ValueError, match='at least 1-dimensional'):
        sg_id.add_data('x', 42, phase='train')


def test_add_data_scalar_numpy_raises(sg_id):
    """Line 407: 0-dim numpy array must be rejected."""
    with pytest.raises(ValueError, match='at least 1-dimensional'):
        sg_id.add_data('x', np.float64(3.14), phase='train')


def test_invalid_var_name_raises(sg_id):
    """Line 172: invalid var_name triggers ValueError."""
    data = np.array([[1.0]])
    with pytest.raises(ValueError, match='Invalid var_name'):
        sg_id.add_data('bad name!', data, phase='train')


def test_invalid_var_name_empty_raises(sg_id):
    """Line 172: empty string var_name triggers ValueError."""
    data = np.array([[1.0]])
    with pytest.raises(ValueError, match='Invalid var_name'):
        sg_id.add_data('', data, phase='train')


def test_close_warns_unsaved_numpy_data(tmp_path, capture_log):
    """Lines 302-304: close() warns about unsaved numpy data."""
    sg = StoreGate(output_dir=str(tmp_path), mode='w')
    sg.set_data_id(DATA_ID)
    sg.set_backend('numpy')
    sg.add_data('x', np.array([[1.0]]), phase='train')
    sg.close()
    text = '\n'.join(capture_log)
    assert 'discarding unsaved numpy data' in text
    assert 'train' in text
    assert 'x' in text


def test_phase_accessor_setitem_non_str_raises(sg_id):
    """Line 37: _PhaseAccessor.__setitem__ with non-str key."""
    accessor = sg_id['train']
    with pytest.raises(ValueError, match='must be str'):
        accessor[123] = np.array([[1.0]])


def test_phase_accessor_delitem_non_str_raises(sg_id):
    """Line 42: _PhaseAccessor.__delitem__ with non-str key."""
    accessor = sg_id['train']
    with pytest.raises(ValueError, match='must be str'):
        del accessor[123]


def test_all_phase_accessor_delitem_non_str_raises(sg_id):
    """Line 85: _AllPhaseAccessor.__delitem__ with non-str key."""
    accessor = _AllPhaseAccessor(sg_id)
    with pytest.raises(ValueError, match='must be str'):
        del accessor[123]
