"""Unit tests for StoreGate class."""
import logging
from unittest.mock import patch

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
def sg_id(tmp_path):
    store = StoreGate(output_dir=str(tmp_path), mode='w')
    store.set_data_id(DATA_ID)
    return store


def _log_text(messages: list[str]) -> str:
    return '\n'.join(messages)


def _seed_var(
    sg: StoreGate,
    phase: str,
    *,
    var_name: str = 'x',
    data: np.ndarray | None = None,
) -> np.ndarray:
    data = np.array([[1.0]]) if data is None else data
    sg.add_data(var_name, data, phase=phase)
    return data


def _seed_var_in_phases(
    sg: StoreGate,
    phases: tuple[str, ...],
    *,
    var_name: str = 'x',
) -> None:
    for value, phase in enumerate(phases, start=1):
        _seed_var(sg, phase, var_name=var_name, data=np.array([[float(value)]]))


def _configure_cross_backend(
    sg: StoreGate,
    *,
    phase: str = 'train',
    zarr_var_name: str | None = None,
    zarr_data: np.ndarray | None = None,
    numpy_var_name: str | None = None,
    numpy_data: np.ndarray | None = None,
) -> None:
    if zarr_var_name is not None and zarr_data is not None:
        sg.add_data(zarr_var_name, zarr_data, phase=phase)
    if numpy_var_name is not None and numpy_data is not None:
        with sg.using_backend('numpy'):
            sg.add_data(numpy_var_name, numpy_data, phase=phase)


def _show_info_text(sg: StoreGate, capture_log: list[str]) -> str:
    sg.show_info()
    return _log_text(capture_log)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_init_data_id_is_none(sg):
    assert sg.data_id is None


# ---------------------------------------------------------------------------
# output_dir validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ('output_dir', 'mode', 'message'),
    [
        ('__TMP__', 'x', 'Invalid mode'),
        ('', 'w', 'non-empty string'),
        ('__TMP__/missing', 'r', 'does not exist'),
        ('__TMP__/missing/store', 'w', 'Parent directory'),
    ],
)
def test_init_invalid_arguments_raise(tmp_path, output_dir, mode, message):
    resolved_output_dir = output_dir.replace('__TMP__', str(tmp_path))
    with pytest.raises(ValueError, match=message):
        StoreGate(output_dir=resolved_output_dir, mode=mode)


def test_init_with_data_id(tmp_path):
    sg = StoreGate(output_dir=str(tmp_path), mode='w', data_id=DATA_ID)
    assert sg.data_id == DATA_ID


@pytest.mark.parametrize('chunk', [0, -1, 1.5, '100', False])
def test_init_invalid_chunk_raises(tmp_path, chunk):
    with pytest.raises(ValueError, match='chunk must be a positive integer'):
        StoreGate(output_dir=str(tmp_path), mode='w', chunk=chunk)


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


def test_set_data_id_rolls_back_on_initialize_failure(sg):
    sg.set_data_id(DATA_ID)

    with patch.object(sg._db, 'initialize', side_effect=RuntimeError('boom')):
        with pytest.raises(RuntimeError, match='boom'):
            sg.set_data_id('other_data')

    assert sg.data_id == DATA_ID
    assert DATA_ID in sg._metadata
    assert 'other_data' not in sg._metadata


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

@pytest.mark.parametrize(
    ('operation', 'message'),
    [
        (lambda sg: sg.add_data('x', np.array([[1.0]]), phase='train'), 'set_data_id'),
        (lambda sg: sg.get_data('x', 'train'), None),
        (lambda sg: sg.compile(), None),
    ],
)
def test_methods_require_data_id(sg, operation, message):
    if message is None:
        with pytest.raises(RuntimeError):
            operation(sg)
        return
    with pytest.raises(RuntimeError, match=message):
        operation(sg)


# ---------------------------------------------------------------------------
# Backend management
# ---------------------------------------------------------------------------

def test_default_backend_is_zarr(sg_id):
    assert sg_id.get_backend() == 'zarr'


@pytest.mark.parametrize(
    ('backend', 'expected', 'message'),
    [
        ('numpy', 'numpy', None),
        ('invalid', None, 'Unsupported backend'),
    ],
)
def test_set_backend_behaviour(sg_id, backend, expected, message):
    if message is not None:
        with pytest.raises(ValueError, match=message):
            sg_id.set_backend(backend)
        return
    sg_id.set_backend(backend)
    assert sg_id.get_backend() == expected


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


def test_backend_view_reports_fixed_backend_without_mutating_root(sg_id):
    assert sg_id.get_backend() == 'zarr'
    assert sg_id.numpy.get_backend() == 'numpy'
    assert sg_id.zarr.get_backend() == 'zarr'
    assert sg_id.get_backend() == 'zarr'


def test_backend_view_add_and_get_without_mutating_root(sg_id):
    data = np.array([[1.0, 2.0], [3.0, 4.0]])

    sg_id.numpy.add_data('x', data, phase='train')

    assert sg_id.get_backend() == 'zarr'
    assert 'x' not in sg_id.zarr.get_var_names('train')
    np.testing.assert_array_equal(sg_id.numpy.get_data('x', 'train', None), data)


def test_backend_view_bracket_access_uses_fixed_backend(sg_id):
    data = np.array([[1.0, 2.0], [3.0, 4.0]])

    sg_id.numpy['train']['x'] = data

    assert sg_id.get_backend() == 'zarr'
    assert 'x' not in sg_id.zarr['train']
    np.testing.assert_array_equal(sg_id.numpy['train']['x'][0], data[0])


def test_backend_view_compile_and_len_are_backend_specific(sg_id):
    sg_id.zarr.add_data('x', np.array([[1.0], [2.0]]), phase='train')
    sg_id.numpy.add_data('x', np.array([[3.0]]), phase='train')

    sg_id.zarr.compile()
    sg_id.numpy.compile()

    assert len(sg_id.zarr['train']) == 2
    assert len(sg_id.numpy['train']) == 1
    assert sg_id.get_backend() == 'zarr'


def test_backend_view_methods_require_data_id(sg):
    with pytest.raises(RuntimeError, match='set_data_id'):
        sg.numpy.add_data('x', np.array([[1.0]]), phase='train')


# ---------------------------------------------------------------------------
# add_data / get_data / update_data / delete_data
# ---------------------------------------------------------------------------

def test_add_and_get_data(sg_id):
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    sg_id.add_data('x', data, phase='train')

    result = sg_id.get_data('x', 'train', index=None)
    np.testing.assert_array_equal(result, data)


def test_add_data_numpy_backend_rejects_non_zarr_compatible_dtype(sg_id):
    sg_id.set_backend('numpy')
    data = np.array([{'a': 1}, {'b': 2}], dtype=object)

    with pytest.raises(ValueError, match='not persistable to the zarr backend'):
        sg_id.add_data('x', data, phase='train')


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


@pytest.mark.parametrize(
    'phases',
    [
        ('train', 'valid', 'test'),
        ('train', 'valid'),
    ],
)
def test_delete_data_all_phases(sg_id, phases):
    _seed_var_in_phases(sg_id, phases)
    sg_id.delete_data('x', phase='all')  # should not raise KeyError

    for phase in ('train', 'valid', 'test'):
        assert 'x' not in sg_id.get_var_names(phase)


def test_rename_data_preserves_data_order_and_compiled_state(sg_id):
    x = np.array([[1.0], [2.0]])
    y = np.array([[3.0], [4.0]])
    sg_id.add_data('x', x, phase='train')
    sg_id.add_data('y', y, phase='train')
    sg_id.compile()

    sg_id.rename_data('x', 'z', phase='train')

    assert sg_id.get_var_names('train') == ['z', 'y']
    np.testing.assert_array_equal(sg_id.get_data('z', 'train', None), x)
    assert len(sg_id['train']) == 2


def test_rename_data_numpy_backend(tmp_path):
    sg = StoreGate(output_dir=str(tmp_path), mode='w', data_id=DATA_ID)
    sg.set_backend('numpy')
    sg.add_data('x', np.array([[1.0], [2.0]]), phase='train')
    sg.add_data('y', np.array([[3.0], [4.0]]), phase='train')
    sg.compile()

    sg.rename_data('x', 'z', phase='train')

    assert sg.get_var_names('train') == ['z', 'y']
    np.testing.assert_array_equal(
        sg.get_data('z', 'train', None),
        np.array([[1.0], [2.0]]),
    )
    assert len(sg['train']) == 2


@pytest.mark.parametrize(
    ('seed_names', 'expected_names'),
    [
        (('x', 'y'), ('x', 'y')),
        ((), ()),
    ],
)
def test_get_var_names(sg_id, seed_names, expected_names):
    for index, name in enumerate(seed_names, start=1):
        _seed_var(sg_id, 'train', var_name=name, data=np.array([[float(index)]]))

    assert tuple(sg_id.get_var_names('train')) == expected_names


def test_get_var_names_preserves_registration_order_after_reopen(tmp_path):
    sg = StoreGate(output_dir=str(tmp_path), mode='w', data_id=DATA_ID)
    sg.add_data('x', np.array([[1.0]]), phase='train')
    sg.add_data('y', np.array([[2.0]]), phase='train')

    reopened = StoreGate(output_dir=str(tmp_path), mode='r', data_id=DATA_ID)
    assert reopened.get_var_names('train') == ['x', 'y']


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


def test_compile_cross_backend_passes_when_metadata_match(sg_id):
    data = np.array([[1.0], [2.0]])
    sg_id.add_data('x', data, phase='train')                      # zarr: 2
    sg_id.copy_to_memory('x', phase='train')                      # numpy: 2
    sg_id.compile(cross_backend=True)  # should not raise


@pytest.mark.parametrize(
    ('phase', 'zarr_var_name', 'zarr_data', 'numpy_var_name', 'numpy_data', 'message'),
    [
        (
            'train',
            'x',
            np.array([[1.0], [2.0]]),
            None,
            None,
            r"'x'.*missing in numpy",
        ),
        (
            'train',
            None,
            None,
            'y',
            np.array([[9.0]]),
            r"'y'.*missing in zarr",
        ),
        (
            'valid',
            'x',
            np.array([[1.0], [2.0]]),
            'x',
            np.array([[9.0]]),
            r"'x'.*'valid'.*zarr=2.*numpy=1|zarr=2.*numpy=1.*'x'.*'valid'",
        ),
        (
            'train',
            'score',
            np.array([[1.0], [2.0]], dtype=np.float32),
            'score',
            np.array([[9.0]], dtype=np.float32),
            r"'score'.*zarr=2.*numpy=1|zarr=2.*numpy=1.*'score'",
        ),
        (
            'train',
            'x',
            np.array([[1.0], [2.0]], dtype=np.float32),
            'x',
            np.array([[1], [2]], dtype=np.int64),
            r'type: zarr=float32, numpy=int64',
        ),
        (
            'train',
            'x',
            np.array([[1.0], [2.0]], dtype=np.float32),
            'x',
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            r'shape: zarr=\(1,\), numpy=\(2,\)',
        ),
    ],
)
def test_compile_cross_backend_detects_mismatches(
    sg_id,
    phase,
    zarr_var_name,
    zarr_data,
    numpy_var_name,
    numpy_data,
    message,
):
    _configure_cross_backend(
        sg_id,
        phase=phase,
        zarr_var_name=zarr_var_name,
        zarr_data=zarr_data,
        numpy_var_name=numpy_var_name,
        numpy_data=numpy_data,
    )
    with pytest.raises(ValueError, match=message):
        sg_id.compile(cross_backend=True)


def test_compile_cross_backend_reports_all_mismatches_at_once(sg_id):
    """All mismatching variables are collected before raising."""
    sg_id.add_data('x', np.array([[1.0], [2.0]]), phase='train')
    sg_id.add_data('y', np.array([[3.0], [4.0]]), phase='train')
    with sg_id.using_backend('numpy'):
        sg_id.add_data('x', np.array([[9.0]]), phase='train')  # mismatch
        sg_id.add_data('z', np.array([[8.0]]), phase='train')  # missing in zarr
    with pytest.raises(ValueError) as exc_info:
        sg_id.compile(cross_backend=True)
    msg = str(exc_info.value)
    assert "'x'" in msg
    assert "'y'" in msg
    assert "'z'" in msg
    assert 'missing in numpy' in msg
    assert 'missing in zarr' in msg


def test_compile_cross_backend_restores_backend_after_check(sg_id):
    """Active backend must be unchanged after cross_backend check."""
    sg_id.set_backend('numpy')
    sg_id.add_data('x', np.array([[1.0]]), phase='train')
    sg_id.copy_to_storage('x', phase='train')
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
# verify_backend_data
# ---------------------------------------------------------------------------

def test_verify_backend_data_passes_when_values_match(sg_id):
    data = np.array([[1.0], [2.0]], dtype=np.float32)
    sg_id.add_data('x', data, phase='train')
    sg_id.copy_to_memory('x', phase='train')

    sg_id.verify_backend_data()


def test_verify_backend_data_treats_nan_values_as_equal(sg_id):
    data = np.array([[np.nan], [1.0]], dtype=np.float32)
    sg_id.add_data('x', data, phase='train')
    sg_id.copy_to_memory('x', phase='train')

    sg_id.verify_backend_data()


def test_verify_backend_data_detects_value_mismatch_even_when_metadata_match(sg_id):
    data = np.array([[1.0], [2.0]], dtype=np.float32)
    sg_id.add_data('x', data, phase='train')
    sg_id.copy_to_memory('x', phase='train')

    with sg_id.using_backend('numpy'):
        sg_id.update_data(
            'x',
            np.array([9.0], dtype=np.float32),
            phase='train',
            index=1,
        )

    with pytest.raises(ValueError, match='data values differ'):
        sg_id.verify_backend_data()


def test_verify_backend_data_phase_allows_scoped_verification(sg_id):
    train_data = np.array([[1.0], [2.0]], dtype=np.float32)
    test_data = np.array([[3.0]], dtype=np.float32)
    sg_id.add_data('x', train_data, phase='train')
    sg_id.add_data('x', test_data, phase='test')
    sg_id.copy_to_memory('x', phase='train')
    sg_id.copy_to_memory('x', phase='test')

    with sg_id.using_backend('numpy'):
        sg_id.update_data(
            'x',
            np.array([7.0], dtype=np.float32),
            phase='test',
            index=0,
        )

    sg_id.verify_backend_data(phase='train')
    with pytest.raises(ValueError, match="'x' in 'test': data values differ"):
        sg_id.verify_backend_data(phase='test')


def test_verify_backend_data_reports_metadata_mismatch(sg_id):
    sg_id.add_data('x', np.array([[1.0], [2.0]]), phase='train')

    with pytest.raises(ValueError, match='missing in numpy'):
        sg_id.verify_backend_data()


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


@pytest.mark.parametrize(
    'operation',
    [
        lambda accessor: accessor['x'],
        lambda accessor: accessor.__setitem__('x', np.array([[1.0]])),
        lambda accessor: 'x' in accessor,
        lambda accessor: list(accessor),
        lambda accessor: len(accessor),
    ],
)
def test_all_phase_accessor_unsupported_operations_raise(sg_id, operation):
    with pytest.raises(NotImplementedError, match=r"sg\['all'\]"):
        operation(sg_id['all'])


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


@pytest.mark.parametrize(
    ('operation', 'error_type'),
    [
        (lambda accessor: accessor['bad_index'], NotImplementedError),
        (lambda accessor: accessor.__setitem__('bad_index', np.array([1.0])), ValueError),
    ],
)
def test_var_accessor_invalid_type_raises(sg_id, operation, error_type):
    _seed_var(sg_id, 'train')
    with pytest.raises(error_type):
        operation(sg_id['train']['x'])


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


def test_copy_to_memory_streams_without_full_source_load(tmp_path):
    sg = StoreGate(output_dir=str(tmp_path), mode='w', chunk=2, data_id=DATA_ID)
    data = np.arange(12, dtype=np.float32).reshape(6, 2)
    sg.add_data('x', data, phase='train')

    with patch.object(
        sg._db._db['zarr'],
        'get_data',
        side_effect=AssertionError('copy_to_memory should not full-load the source variable'),
    ):
        sg.copy_to_memory('x', phase='train')

    with sg.using_backend('numpy'):
        np.testing.assert_array_equal(sg.get_data('x', 'train', None), data)


def test_copy_to_storage_streams_without_full_source_load(tmp_path):
    sg = StoreGate(output_dir=str(tmp_path), mode='w', data_id=DATA_ID)
    data = np.arange(12, dtype=np.float32).reshape(6, 2)

    with sg.using_backend('numpy'):
        sg.add_data('x', data[:2], phase='train')
        sg.add_data('x', data[2:], phase='train')

        with patch.object(
            sg._db._db['numpy'],
            'get_data',
            side_effect=AssertionError('copy_to_storage should not full-load the source variable'),
        ):
            sg.copy_to_storage('x', phase='train')

    with sg.using_backend('zarr'):
        np.testing.assert_array_equal(sg.get_data('x', 'train', None), data)


def test_copy_to_memory_rolls_back_partial_destination_on_failure(tmp_path):
    sg = StoreGate(output_dir=str(tmp_path), mode='w', chunk=2, data_id=DATA_ID)
    data = np.arange(12, dtype=np.float32).reshape(6, 2)
    sg.add_data('x', data, phase='train')

    numpy_db = sg._db._db['numpy']
    original_add_data = numpy_db.add_data
    add_calls = 0

    def flaky_add_data(*args, **kwargs):
        nonlocal add_calls
        add_calls += 1
        if add_calls == 2:
            raise RuntimeError('boom')
        return original_add_data(*args, **kwargs)

    with patch.object(numpy_db, 'add_data', side_effect=flaky_add_data):
        with pytest.raises(RuntimeError, match='boom'):
            sg.copy_to_memory('x', phase='train')

    with sg.using_backend('numpy'):
        assert 'x' not in sg.get_var_names('train')


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


@pytest.mark.parametrize(
    ('setup', 'expected_tokens'),
    [
        (lambda sg: (_seed_var(sg, 'train', data=np.array([[1.0], [2.0]])), sg.compile()), ('x', DATA_ID, 'True')),
        (lambda sg: _seed_var(sg, 'train'), ('False',)),
        (lambda sg: (_seed_var(sg, 'train'), _seed_var(sg, 'valid')), ('train', 'valid')),
        (
            lambda sg: (
                _seed_var(sg, 'train', var_name='x', data=np.array([[1.0], [2.0]])),
                _seed_var(sg, 'train', var_name='y', data=np.array([[3.0], [4.0]])),
                sg.compile(),
            ),
            ('x', 'y'),
        ),
        (lambda sg: sg.compile(), (DATA_ID,)),
        (lambda sg: (_seed_var(sg, 'train', data=np.array([[1.0]], dtype=np.float32)), sg.compile()), ('float32',)),
        (lambda sg: (_seed_var(sg, 'train', data=np.array([[1.0], [2.0], [3.0]])), sg.compile()), ('3',)),
        (lambda sg: (_seed_var(sg, 'train', data=np.array([[1.0, 2.0], [3.0, 4.0]])), sg.compile()), ('(2,)',)),
        (lambda sg: (_seed_var(sg, 'train'), sg.compile()), ('zarr',)),
    ],
)
def test_show_info_outputs_expected_tokens(sg_id, capture_log, setup, expected_tokens):
    setup(sg_id)
    text = _show_info_text(sg_id, capture_log)
    for token in expected_tokens:
        assert token in text


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

@pytest.mark.parametrize(
    ('var_name', 'data', 'message'),
    [
        ('x', 42, 'at least 1-dimensional'),
        ('x', np.float64(3.14), 'at least 1-dimensional'),
        ('bad name!', np.array([[1.0]]), 'Invalid var_name'),
        ('', np.array([[1.0]]), 'Invalid var_name'),
    ],
)
def test_add_data_input_validation(sg_id, var_name, data, message):
    with pytest.raises(ValueError, match=message):
        sg_id.add_data(var_name, data, phase='train')


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


@pytest.mark.parametrize(
    'accessor_factory',
    [
        lambda sg: sg['train'],
        lambda sg: _AllPhaseAccessor(sg),
    ],
)
def test_accessor_delitem_non_str_raises(sg_id, accessor_factory):
    accessor = accessor_factory(sg_id)
    with pytest.raises(ValueError, match='must be str'):
        del accessor[123]


def test_phase_accessor_setitem_non_str_raises(sg_id):
    with pytest.raises(ValueError, match='must be str'):
        sg_id['train'][123] = np.array([[1.0]])
