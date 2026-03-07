"""Unit tests for DLTask."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

import numpy as np

from storegate.task.dl_task import DLTask
from storegate.task.dl_env import DLEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class ConcreteDLTask(DLTask):
    """Minimal concrete subclass for testing."""
    pass


def make_sg(existing_var_names: dict[str, list[str]] | None = None) -> MagicMock:
    """Create a mock StoreGate."""
    sg = MagicMock()
    sg.set_data_id = MagicMock()
    sg.compile = MagicMock()
    if existing_var_names is None:
        sg.get_var_names = MagicMock(return_value=[])
    else:
        sg.get_var_names = MagicMock(
            side_effect=lambda phase: list(existing_var_names.get(phase, []))
        )
    sg.using_backend = MagicMock()
    return sg


def make_task(**overrides) -> ConcreteDLTask:
    """Create a ConcreteDLTask with sensible defaults."""
    kwargs = dict(storegate=make_sg())
    kwargs.update(overrides)
    return ConcreteDLTask(**kwargs)


# ---------------------------------------------------------------------------
# __init__ defaults
# ---------------------------------------------------------------------------

def test_init_default_values() -> None:
    task = make_task()
    assert task._input_var_names is None
    assert task._output_var_names is None
    assert task._true_var_names is None
    assert task._model is None
    assert task._optimizer is None
    assert task._loss is None
    assert task._metrics is None
    assert task._num_epochs == 10
    assert task._batch_size == 64
    assert task._preload is False


def test_init_none_args_become_empty_dicts() -> None:
    task = make_task()
    assert task._model_args == {}
    assert task._optimizer_args == {}
    assert task._loss_args == {}


def test_init_preserves_explicit_args() -> None:
    m_args = {'hidden': 128}
    o_args = {'lr': 0.01}
    l_args = {'weight': 0.5}
    task = make_task(model_args=m_args, optimizer_args=o_args, loss_args=l_args)
    assert task._model_args is m_args
    assert task._optimizer_args is o_args
    assert task._loss_args is l_args


def test_init_creates_dl_env() -> None:
    task = make_task()
    assert isinstance(task._ml, DLEnv)
    assert task._ml.model is None
    assert task._ml.optimizer is None
    assert task._ml.loss is None


def test_init_custom_values() -> None:
    task = make_task(
        input_var_names=['x'],
        output_var_names=['pred'],
        true_var_names=['y'],
        model='Linear',
        num_epochs=20,
        batch_size=128,
        preload=True,
    )
    assert task._input_var_names == ['x']
    assert task._output_var_names == ['pred']
    assert task._true_var_names == ['y']
    assert task._model == 'Linear'
    assert task._num_epochs == 20
    assert task._batch_size == 128
    assert task._preload is True


# ---------------------------------------------------------------------------
# set_hps — model__ prefix
# ---------------------------------------------------------------------------

def test_set_hps_model_prefix_sets_model_args() -> None:
    task = make_task()
    task.set_hps({'model__hidden_size': 256})
    assert task._model_args['hidden_size'] == 256


def test_set_hps_model_prefix_multiple_keys() -> None:
    task = make_task()
    task.set_hps({'model__hidden_size': 256, 'model__dropout': 0.5})
    assert task._model_args['hidden_size'] == 256
    assert task._model_args['dropout'] == 0.5


def test_set_hps_model_prefix_overwrites_existing() -> None:
    task = make_task(model_args={'hidden_size': 64})
    task.set_hps({'model__hidden_size': 256})
    assert task._model_args['hidden_size'] == 256


def test_set_hps_model_prefix_empty_suffix_raises() -> None:
    task = make_task()
    with pytest.raises(ValueError, match='empty suffix'):
        task.set_hps({'model__': 1})


# ---------------------------------------------------------------------------
# set_hps — optimizer__ prefix
# ---------------------------------------------------------------------------

def test_set_hps_optimizer_prefix_sets_optimizer_args() -> None:
    task = make_task()
    task.set_hps({'optimizer__lr': 0.001})
    assert task._optimizer_args['lr'] == 0.001


def test_set_hps_optimizer_prefix_empty_suffix_raises() -> None:
    task = make_task()
    with pytest.raises(ValueError, match='empty suffix'):
        task.set_hps({'optimizer__': 1})


# ---------------------------------------------------------------------------
# set_hps — loss__ prefix
# ---------------------------------------------------------------------------

def test_set_hps_loss_prefix_sets_loss_args() -> None:
    task = make_task()
    task.set_hps({'loss__weight': 0.5})
    assert task._loss_args['weight'] == 0.5


def test_set_hps_loss_prefix_empty_suffix_raises() -> None:
    task = make_task()
    with pytest.raises(ValueError, match='empty suffix'):
        task.set_hps({'loss__': 1})


# ---------------------------------------------------------------------------
# set_hps — plain keys (task attributes)
# ---------------------------------------------------------------------------

def test_set_hps_plain_key_updates_task_attribute() -> None:
    task = make_task()
    task.set_hps({'batch_size': 128})
    assert task._batch_size == 128


def test_set_hps_plain_key_num_epochs() -> None:
    task = make_task()
    task.set_hps({'num_epochs': 50})
    assert task._num_epochs == 50


def test_set_hps_protected_key_storegate_raises() -> None:
    task = make_task()
    with pytest.raises(AttributeError, match='not a valid hyperparameter'):
        task.set_hps({'storegate': None})


def test_set_hps_protected_key_ml_raises() -> None:
    task = make_task()
    with pytest.raises(AttributeError, match='not a valid hyperparameter'):
        task.set_hps({'ml': None})


def test_set_hps_undefined_key_raises() -> None:
    task = make_task()
    with pytest.raises(AttributeError, match='not defined'):
        task.set_hps({'nonexistent': 99})


# ---------------------------------------------------------------------------
# set_hps — mixed prefixed and plain keys
# ---------------------------------------------------------------------------

def test_set_hps_mixed_keys() -> None:
    task = make_task()
    task.set_hps({
        'model__hidden': 128,
        'optimizer__lr': 1e-3,
        'loss__reduction': 'mean',
        'batch_size': 32,
    })
    assert task._model_args['hidden'] == 128
    assert task._optimizer_args['lr'] == 1e-3
    assert task._loss_args['reduction'] == 'mean'
    assert task._batch_size == 32


# ---------------------------------------------------------------------------
# set_hps — data_id forwarding
# ---------------------------------------------------------------------------

def test_set_hps_calls_set_data_id_when_data_id_set() -> None:
    sg = make_sg()
    task = ConcreteDLTask(storegate=sg)
    task._data_id = 'exp01'
    task.set_hps({'batch_size': 32})
    sg.set_data_id.assert_called_once_with('exp01')


def test_set_hps_does_not_call_set_data_id_when_none() -> None:
    sg = make_sg()
    task = ConcreteDLTask(storegate=sg)
    task.set_hps({'batch_size': 32})
    sg.set_data_id.assert_not_called()


# ---------------------------------------------------------------------------
# compile_var_names
# ---------------------------------------------------------------------------

def test_compile_var_names_str_to_list() -> None:
    task = make_task(
        storegate=make_sg({
            'train': ['x', 'y'],
            'valid': ['x', 'y'],
            'test': ['x', 'y'],
        }),
        input_var_names='x',
        output_var_names='pred',
        true_var_names='y',
    )
    task.compile_var_names()
    assert task._input_var_names == {'train': ['x'], 'valid': ['x'], 'test': ['x']}
    assert task._output_var_names == {'train': ['pred'], 'valid': ['pred'], 'test': ['pred']}
    assert task._true_var_names == {'train': ['y'], 'valid': ['y'], 'test': ['y']}


def test_compile_var_names_list_unchanged() -> None:
    task = make_task(
        storegate=make_sg({
            'train': ['x', 'z', 'y'],
            'valid': ['x', 'z', 'y'],
            'test': ['x', 'z', 'y'],
        }),
        input_var_names=['x', 'z'],
        output_var_names=['pred'],
        true_var_names=['y'],
    )
    task.compile_var_names()
    assert task._input_var_names == {
        'train': ['x', 'z'],
        'valid': ['x', 'z'],
        'test': ['x', 'z'],
    }
    assert task._output_var_names == {
        'train': ['pred'],
        'valid': ['pred'],
        'test': ['pred'],
    }
    assert task._true_var_names == {'train': ['y'], 'valid': ['y'], 'test': ['y']}


def test_compile_var_names_none_normalizes_to_phase_dict() -> None:
    task = make_task()
    task.compile_var_names()
    assert task._input_var_names == {'train': None, 'valid': None, 'test': None}
    assert task._output_var_names == {'train': None, 'valid': None, 'test': None}
    assert task._true_var_names == {'train': None, 'valid': None, 'test': None}


def test_compile_var_names_phase_dict_normalizes_values() -> None:
    task = make_task(
        storegate=make_sg({
            'train': ['x0', 'y0'],
            'valid': ['x1', 'x2', 'y1'],
            'test': [],
        }),
        input_var_names={'train': 'x0', 'valid': ['x1', 'x2'], 'test': None},
        output_var_names={'test': 'pred'},
        true_var_names={'train': 'y0', 'valid': 'y1'},
    )

    task.compile_var_names()

    assert task._input_var_names == {
        'train': ['x0'],
        'valid': ['x1', 'x2'],
        'test': None,
    }
    assert task._output_var_names == {
        'train': None,
        'valid': None,
        'test': ['pred'],
    }
    assert task._true_var_names == {
        'train': ['y0'],
        'valid': ['y1'],
        'test': None,
    }


def test_compile_var_names_phase_dict_rejects_invalid_phase() -> None:
    task = make_task(input_var_names={'train': 'x', 'dev': 'x_dev'})

    with pytest.raises(ValueError, match='invalid phases'):
        task.compile_var_names()


def test_compile_var_names_raises_when_required_input_missing() -> None:
    task = make_task(
        storegate=make_sg({
            'train': ['x', 'y'],
            'valid': ['y'],
            'test': ['x', 'y'],
        }),
        input_var_names='x',
        true_var_names='y',
    )

    with pytest.raises(ValueError, match="phase='valid' missing input_var_names=\\['x'\\]"):
        task.compile_var_names()


def test_compile_var_names_raises_when_required_true_missing_in_phase_dict() -> None:
    task = make_task(
        storegate=make_sg({
            'train': ['x_train', 'y_train'],
            'valid': ['x_valid'],
            'test': ['x_test'],
        }),
        input_var_names={'train': 'x_train', 'valid': 'x_valid', 'test': 'x_test'},
        true_var_names={'train': 'y_train', 'valid': 'y_valid', 'test': None},
    )

    with pytest.raises(ValueError, match="phase='valid' missing true_var_names=\\['y_valid'\\]"):
        task.compile_var_names()


def test_compile_var_names_does_not_require_output_var_names_to_exist() -> None:
    task = make_task(
        storegate=make_sg({
            'train': ['x', 'y'],
            'valid': ['x', 'y'],
            'test': ['x', 'y'],
        }),
        input_var_names='x',
        output_var_names='pred',
        true_var_names='y',
    )

    task.compile_var_names()

    assert task._output_var_names == {
        'train': ['pred'],
        'valid': ['pred'],
        'test': ['pred'],
    }


def test_compile_var_names_raises_when_output_overlaps_input() -> None:
    task = make_task(
        storegate=make_sg({
            'train': ['x', 'y'],
            'valid': ['x', 'y'],
            'test': ['x', 'y'],
        }),
        input_var_names='x',
        output_var_names='x',
        true_var_names='y',
    )

    with pytest.raises(
        ValueError,
        match=r"phase='train' output_var_names overlap with input_var_names=\['x'\]",
    ):
        task.compile_var_names()


def test_compile_var_names_raises_when_output_overlaps_true_in_phase_dict() -> None:
    task = make_task(
        storegate=make_sg({
            'train': ['x_train', 'y_train'],
            'valid': ['x_valid', 'y_valid'],
            'test': ['x_test'],
        }),
        input_var_names={'train': 'x_train', 'valid': 'x_valid', 'test': 'x_test'},
        output_var_names={'valid': 'y_valid'},
        true_var_names={'train': 'y_train', 'valid': 'y_valid', 'test': None},
    )

    with pytest.raises(
        ValueError,
        match=r"phase='valid' output_var_names overlap with true_var_names=\['y_valid'\]",
    ):
        task.compile_var_names()


def test_get_var_names_for_phase_resolves_phase_specific_values() -> None:
    task = make_task(
        storegate=make_sg({
            'train': ['x_train'],
            'valid': ['x_valid'],
            'test': [],
        }),
        input_var_names={'train': 'x_train', 'valid': ['x_valid'], 'test': None},
    )
    task.compile_var_names()

    assert task._get_var_names_for_phase(task._input_var_names, 'train') == ['x_train']
    assert task._get_var_names_for_phase(task._input_var_names, 'valid') == ['x_valid']
    assert task._get_var_names_for_phase(task._input_var_names, 'test') is None


# ---------------------------------------------------------------------------
# compile — calls sub-compile methods in order
# ---------------------------------------------------------------------------

def test_compile_clears_dl_env() -> None:
    task = make_task()
    task._ml.model = 'dummy_model'
    task._ml.optimizer = 'dummy_opt'
    task._ml.loss = 'dummy_loss'
    task.compile()
    assert task._ml.model is None
    assert task._ml.optimizer is None
    assert task._ml.loss is None


def test_compile_calls_storegate_compile() -> None:
    sg = make_sg()
    task = ConcreteDLTask(storegate=sg)
    task.compile()
    sg.compile.assert_called_once()


def test_compile_calls_sub_compiles_in_order() -> None:
    task = make_task()
    call_order = []
    task.compile_var_names = lambda: call_order.append('var_names')
    task.compile_model = lambda: call_order.append('model')
    task.compile_loss = lambda: call_order.append('loss')
    task.compile_optimizer = lambda: call_order.append('optimizer')
    task._storegate.compile = lambda: call_order.append('storegate')
    task.compile()
    assert call_order == ['var_names', 'model', 'loss', 'optimizer', 'storegate']


# ---------------------------------------------------------------------------
# fit / predict — base implementations
# ---------------------------------------------------------------------------

def test_fit_returns_empty_dict() -> None:
    task = make_task()
    assert task.fit() == {}


def test_predict_returns_empty_dict() -> None:
    task = make_task()
    assert task.predict() == {}


# ---------------------------------------------------------------------------
# execute — preload=False
# ---------------------------------------------------------------------------

def test_execute_no_preload_calls_compile_fit_predict() -> None:
    task = make_task(preload=False)
    call_order = []
    task.compile = lambda: call_order.append('compile')
    task.fit = lambda: (call_order.append('fit'), {})[1]
    task.predict = lambda: (call_order.append('predict'), {})[1]
    task.execute()
    assert call_order == ['compile', 'fit', 'predict']


def test_execute_no_preload_merges_fit_and_predict() -> None:
    task = make_task(preload=False)
    task.compile = lambda: None
    task.fit = lambda: {'train_loss': 0.1}
    task.predict = lambda: {'test_acc': 0.9}
    result = task.execute()
    assert result == {'train_loss': 0.1, 'test_acc': 0.9}


# ---------------------------------------------------------------------------
# execute — preload=True
# ---------------------------------------------------------------------------

def test_execute_preload_copies_vars_to_memory(tmp_path) -> None:
    """Integration-style test: preload copies zarr data to numpy backend."""
    from storegate import StoreGate

    sg = StoreGate(output_dir=str(tmp_path), mode='w', data_id='test')
    sg.add_data('x', np.arange(10).reshape(5, 2), phase='train')
    sg.add_data('y', np.arange(5), phase='train')
    sg.compile()

    task = ConcreteDLTask(
        storegate=sg,
        input_var_names={'train': 'x', 'valid': None, 'test': None},
        true_var_names={'train': 'y', 'valid': None, 'test': None},
        preload=True,
    )

    task.compile_var_names()
    task.compile = lambda: None
    task.execute()

    # After preload, data should exist in numpy backend
    sg.set_backend('numpy')
    assert 'x' in sg.get_var_names('train')
    assert 'y' in sg.get_var_names('train')
    np.testing.assert_array_equal(sg.get_data('x', 'train'), np.arange(10).reshape(5, 2))
    np.testing.assert_array_equal(sg.get_data('y', 'train'), np.arange(5))


def test_execute_preload_skips_missing_vars(tmp_path) -> None:
    """Preload skips variables that don't exist in zarr for a given phase."""
    from storegate import StoreGate

    sg = StoreGate(output_dir=str(tmp_path), mode='w', data_id='test')
    sg.add_data('x', np.ones((3, 2)), phase='train')
    # 'y' only in train, not in valid
    sg.add_data('y', np.ones(3), phase='train')
    sg.add_data('x', np.ones((2, 2)), phase='valid')
    sg.compile()

    task = ConcreteDLTask(
        storegate=sg,
        input_var_names={'train': 'x', 'valid': 'x', 'test': None},
        true_var_names={'train': 'y', 'valid': None, 'test': None},
        preload=True,
    )

    task.compile_var_names()
    task.compile = lambda: None
    task.execute()  # should not raise

    sg.set_backend('numpy')
    assert 'x' in sg.get_var_names('valid')
    assert 'y' not in sg.get_var_names('valid')


def test_execute_preload_deletes_existing_numpy_before_copy(tmp_path) -> None:
    """Preload deletes existing numpy data for a var before copying from zarr."""
    from storegate import StoreGate

    sg = StoreGate(output_dir=str(tmp_path), mode='w', data_id='test')
    sg.add_data('x', np.ones((3, 2)), phase='train')
    sg.compile()

    # Pre-populate numpy backend with stale data
    sg.set_backend('numpy')
    sg.add_data('x', np.zeros((3, 2)), phase='train')
    sg.set_backend('zarr')

    task = ConcreteDLTask(
        storegate=sg,
        input_var_names={'train': 'x', 'valid': None, 'test': None},
        preload=True,
    )

    task.compile_var_names()
    task.compile = lambda: None
    task.execute()

    sg.set_backend('numpy')
    np.testing.assert_array_equal(sg.get_data('x', 'train'), np.ones((3, 2)))


def test_execute_preload_runs_fit_predict_under_numpy_backend(tmp_path) -> None:
    """When preload=True, fit() and predict() run with numpy backend active."""
    from storegate import StoreGate

    sg = StoreGate(output_dir=str(tmp_path), mode='w', data_id='test')
    sg.add_data('x', np.ones((3, 2)), phase='train')
    sg.compile()

    backends_during_fit: list[str] = []

    class TrackerTask(ConcreteDLTask):
        def fit(self):
            backends_during_fit.append(self._storegate.get_backend())
            return {}
        def predict(self):
            backends_during_fit.append(self._storegate.get_backend())
            return {}

    task = TrackerTask(
        storegate=sg,
        input_var_names={'train': 'x', 'valid': None, 'test': None},
        preload=True,
    )
    task.compile_var_names()
    task.compile = lambda: None
    task.execute()

    assert backends_during_fit == ['numpy', 'numpy']


def test_execute_preload_uses_phase_specific_var_names(tmp_path) -> None:
    from storegate import StoreGate

    sg = StoreGate(output_dir=str(tmp_path), mode='w', data_id='test')
    sg.add_data('x_train', np.arange(6).reshape(3, 2), phase='train')
    sg.add_data('y_train', np.arange(3), phase='train')
    sg.add_data('x_valid', np.arange(4).reshape(2, 2), phase='valid')
    sg.add_data('y_valid', np.arange(2), phase='valid')
    sg.add_data('x_test', np.arange(2).reshape(1, 2), phase='test')
    sg.compile()

    task = ConcreteDLTask(
        storegate=sg,
        input_var_names={'train': 'x_train', 'valid': 'x_valid', 'test': 'x_test'},
        true_var_names={'train': 'y_train', 'valid': 'y_valid', 'test': None},
        preload=True,
    )

    task.compile_var_names()
    task.compile = lambda: None
    task.execute()

    sg.set_backend('numpy')
    assert 'x_train' in sg.get_var_names('train')
    assert 'y_train' in sg.get_var_names('train')
    assert 'x_valid' in sg.get_var_names('valid')
    assert 'y_valid' in sg.get_var_names('valid')
    assert 'x_test' in sg.get_var_names('test')
    assert 'y_train' not in sg.get_var_names('test')
    assert 'y_valid' not in sg.get_var_names('test')


# ---------------------------------------------------------------------------
# DLEnv
# ---------------------------------------------------------------------------

def test_dl_env_clear() -> None:
    env = DLEnv(model='m', optimizer='o', loss='l')
    env.clear()
    assert env.model is None
    assert env.optimizer is None
    assert env.loss is None


def test_dl_env_defaults() -> None:
    env = DLEnv()
    assert env.model is None
    assert env.optimizer is None
    assert env.loss is None
