"""Unit tests for PytorchTask and pytorch_util.build_module."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from storegate import StoreGate  # noqa: E402
from storegate.task.dl_env import DLEnv  # noqa: E402
from storegate.task.dl_task import DLTask  # noqa: E402
from storegate.task.pytorch.pytorch_util import build_module  # noqa: E402
from storegate.task.pytorch_task import PytorchTask  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compile_var_names(var_names):
    if isinstance(var_names, dict):
        return {
            'train': [var_names['train']] if isinstance(var_names.get('train'), str) else var_names.get('train'),
            'valid': [var_names['valid']] if isinstance(var_names.get('valid'), str) else var_names.get('valid'),
            'test': [var_names['test']] if isinstance(var_names.get('test'), str) else var_names.get('test'),
        }
    if isinstance(var_names, str):
        resolved = [var_names]
    else:
        resolved = var_names
    return {'train': resolved, 'valid': resolved, 'test': resolved}


def make_task(output_var_names):
    """Return a minimal PytorchTask with mocked storegate."""
    task = PytorchTask.__new__(PytorchTask)
    task._output_var_names = compile_var_names(output_var_names)
    task._storegate = MagicMock()
    return task


def make_tensor(value: float, size: int = 2) -> torch.Tensor:
    return torch.full((size,), value)


def make_step_task() -> PytorchTask:
    """Return a minimal task instance for step_batch tests."""
    task = PytorchTask.__new__(PytorchTask)
    task._device = torch.device('cpu')
    task.step_model = MagicMock(return_value=torch.tensor([[0.1, 0.9]]))
    task.step_loss = MagicMock(return_value={'loss': torch.tensor(0.3)})
    task.step_optimizer = MagicMock()
    return task


def make_runtime_task(**overrides: Any) -> PytorchTask:
    """Return a fully initialized task with lightweight defaults."""
    kwargs = dict(
        storegate=MagicMock(),
        input_var_names=['x'],
        true_var_names=['y'],
        model=torch.nn.Linear,
        model_args={'in_features': 2, 'out_features': 1},
        optimizer=torch.optim.SGD,
        optimizer_args={'lr': 0.1},
        loss=torch.nn.MSELoss,
        device='cpu',
    )
    kwargs.update(overrides)
    task = PytorchTask(**kwargs)

    input_var_names = compile_var_names(task._input_var_names)
    true_var_names = compile_var_names(task._true_var_names)
    task._storegate.get_var_names.side_effect = lambda phase: [
        *(input_var_names[phase] or []),
        *(true_var_names[phase] or []),
    ]
    task.compile_var_names()
    return task


def make_loop_task(
    *,
    input_var_names: Any = None,
    true_var_names: Any = None,
    output_var_names: Any = None,
    storegate: Any = None,
    num_epochs: int = 1,
) -> PytorchTask:
    """Return a minimal task instance for fit()/predict() tests."""
    task = PytorchTask.__new__(PytorchTask)
    task._storegate = MagicMock() if storegate is None else storegate
    if input_var_names is not None:
        task._input_var_names = compile_var_names(input_var_names)
    if true_var_names is not None:
        task._true_var_names = compile_var_names(true_var_names)
    if output_var_names is not None:
        task._output_var_names = compile_var_names(output_var_names)
    task.get_dataloader = MagicMock()
    task.step_epoch = MagicMock()
    task._ml = DLEnv(model=MagicMock())
    task._num_epochs = num_epochs
    return task


class _FakeTqdm:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.descriptions: list[str] = []
        self.postfixes: list[dict[str, Any]] = []
        self.updates: list[int] = []

    def __enter__(self) -> _FakeTqdm:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    def set_description(self, desc: str) -> None:
        self.descriptions.append(desc)

    def set_postfix(self, metrics: dict[str, Any]) -> None:
        self.postfixes.append(metrics)

    def update(self, value: int) -> None:
        self.updates.append(value)


class _PredictOnlyTask(PytorchTask):
    def fit(self) -> dict[str, Any]:
        return {}


def _internal_tmp_name(index: int = 0, suffix: int | None = None) -> str:
    base = f'__storegate_predict_tmp_{index}'
    if suffix is None:
        return base
    return f'{base}_{suffix}'


# ---------------------------------------------------------------------------
# __init__ / compile
# ---------------------------------------------------------------------------


def test_init_auto_device_sets_cpu_and_default_metrics(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    task = PytorchTask(storegate=MagicMock())
    assert task._device == torch.device('cpu')
    assert task._metrics == ['loss']
    assert task._is_gpu is False


def test_init_copies_dataset_and_dataloader_args() -> None:
    dataset_args = {'preload': False}
    dataloader_args = {'num_workers': 2}
    task = PytorchTask(
        storegate=MagicMock(),
        dataset_args=dataset_args,
        dataloader_args=dataloader_args,
    )
    assert task._dataset_args == dataset_args
    assert task._dataloader_args == dataloader_args
    assert task._dataset_args is not dataset_args
    assert task._dataloader_args is not dataloader_args


def test_compile_calls_compile_device_then_super_compile() -> None:
    task = PytorchTask.__new__(PytorchTask)
    calls: list[str] = []
    task.compile_device = lambda: calls.append('compile_device')  # type: ignore[method-assign]

    with patch.object(
        DLTask,
        'compile',
        autospec=True,
        side_effect=lambda self: calls.append('super_compile'),
    ):
        task.compile()

    assert calls == ['compile_device', 'super_compile']


def test_compile_device_uses_cuda_id() -> None:
    task = PytorchTask.__new__(PytorchTask)
    task._device = torch.device('cpu')
    task._cuda_id = 2
    task.compile_device()
    assert task._device == torch.device('cuda:2')
    assert task._is_gpu is True


# ---------------------------------------------------------------------------
# compile_model / compile_optimizer / compile_loss
# ---------------------------------------------------------------------------


def test_compile_model_raises_when_model_missing() -> None:
    task = make_runtime_task(model=None, optimizer=None, loss=torch.nn.MSELoss)
    with pytest.raises(ValueError, match='model is required'):
        task.compile_model()


def test_compile_model_raises_when_cuda_device_unavailable() -> None:
    task = make_runtime_task(model=torch.nn.Linear, optimizer=None, loss=torch.nn.MSELoss)
    task._device = torch.device('cuda:0')
    task._is_gpu = True
    with patch('storegate.task.pytorch_task.torch.cuda.is_available', return_value=False):
        with pytest.raises(ValueError, match='cuda:0 is not available'):
            task.compile_model()


def test_compile_model_builds_model_and_applies_optional_features(monkeypatch) -> None:
    task = make_runtime_task(
        model=MagicMock(name='model_factory'),
        optimizer=None,
        loss=torch.nn.MSELoss,
        torch_compile=True,
        torchinfo=True,
    )
    fake_model = MagicMock()
    summary = MagicMock()
    monkeypatch.setitem(sys.modules, 'torchinfo', SimpleNamespace(summary=summary))

    with patch('storegate.task.pytorch_task.util.build_module', return_value=fake_model) as build_module:
        with patch('storegate.task.pytorch_task.torch.compile', return_value='compiled-model') as torch_compile:
            task.compile_model()

    build_module.assert_called_once_with(task._model, task._model_args, torch.nn)
    fake_model.to.assert_called_once_with(task._device)
    summary.assert_called_once_with(fake_model)
    torch_compile.assert_called_once_with(fake_model)
    assert task._ml.model == 'compiled-model'


def test_compile_model_supports_string_model_names_from_torch_nn() -> None:
    task = make_runtime_task(
        model='Linear',
        model_args={'in_features': 2, 'out_features': 1},
        optimizer=None,
        loss=torch.nn.MSELoss,
    )

    task.compile_model()

    assert isinstance(task._ml.model, torch.nn.Linear)
    assert task._ml.model.in_features == 2
    assert task._ml.model.out_features == 1


def test_compile_optimizer_skips_when_optimizer_missing() -> None:
    task = make_runtime_task(optimizer=None, loss=torch.nn.MSELoss)
    task._ml.model = MagicMock()
    task.compile_optimizer()
    assert task._ml.optimizer is None


def test_compile_optimizer_builds_optimizer_from_model_params() -> None:
    task = make_runtime_task()
    params = [torch.nn.Parameter(torch.ones(1))]
    task._ml.model = MagicMock()
    task._ml.model.parameters.return_value = params

    with patch('storegate.task.pytorch_task.util.build_module', return_value='optimizer') as build_module:
        task.compile_optimizer()

    build_args = build_module.call_args.args
    assert build_args[0] is task._optimizer
    assert build_args[1]['lr'] == 0.1
    assert build_args[1]['params'] == params
    assert task._optimizer_args == {'lr': 0.1}
    assert task._ml.optimizer == 'optimizer'


def test_compile_loss_raises_when_loss_missing() -> None:
    task = make_runtime_task(loss=None)
    with pytest.raises(ValueError, match='loss is required'):
        task.compile_loss()


def test_compile_loss_builds_and_moves_loss() -> None:
    task = make_runtime_task()
    fake_loss = MagicMock()
    fake_loss.to.return_value = 'loss-on-device'

    with patch('storegate.task.pytorch_task.util.build_module', return_value=fake_loss) as build_module:
        task.compile_loss()

    build_args = build_module.call_args.args
    assert build_args[0] is task._loss
    assert build_args[1] == task._loss_args
    fake_loss.to.assert_called_once_with(task._device)
    assert task._ml.loss == 'loss-on-device'


# ---------------------------------------------------------------------------
# get_dataloader / fit / predict / step_epoch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('phase', 'dataset_args', 'dataloader_args', 'batch_size', 'expected_input', 'expected_true', 'expected_loader_kwargs'),
    [
        (
            'train',
            {'preload': False},
            {'num_workers': 2},
            16,
            ['x_train'],
            ['y_train'],
            {'shuffle': True, 'num_workers': 2},
        ),
        (
            'valid',
            None,
            None,
            8,
            ['x_valid'],
            ['y_valid'],
            {'shuffle': False},
        ),
        (
            'test',
            None,
            None,
            8,
            ['x_test'],
            None,
            {'shuffle': False},
        ),
        (
            'test',
            None,
            {'shuffle': True, 'num_workers': 2},
            8,
            ['x_test'],
            None,
            {'shuffle': False, 'num_workers': 2},
        ),
    ],
)
def test_get_dataloader_builds_loader_for_phase(
    phase: str,
    dataset_args: dict[str, Any] | None,
    dataloader_args: dict[str, Any] | None,
    batch_size: int,
    expected_input: list[str],
    expected_true: list[str] | None,
    expected_loader_kwargs: dict[str, Any],
) -> None:
    task = make_runtime_task(
        batch_size=batch_size,
        input_var_names={'train': 'x_train', 'valid': 'x_valid', 'test': 'x_test'},
        true_var_names={'train': 'y_train', 'valid': 'y_valid', 'test': None},
        dataset_args=dataset_args,
        dataloader_args=dataloader_args,
    )
    fake_dataset = object()

    with patch('storegate.task.pytorch_task.StoreGateDataset', return_value=fake_dataset) as dataset_cls:
        with patch('storegate.task.pytorch_task.DataLoader', return_value='loader') as dataloader_cls:
            loader = task.get_dataloader(phase)

    assert loader == 'loader'
    assert dataset_cls.call_args.args == (task._storegate, phase)
    assert dataset_cls.call_args.kwargs['input_var_names'] == expected_input
    assert dataset_cls.call_args.kwargs['true_var_names'] == expected_true
    assert dataset_cls.call_args.kwargs.get('preload', True) is (dataset_args or {}).get('preload', True)

    kwargs = dataloader_cls.call_args.kwargs
    assert kwargs['dataset'] is fake_dataset
    assert kwargs['batch_size'] == batch_size
    for key, value in expected_loader_kwargs.items():
        assert kwargs[key] == value


def test_get_dataloader_with_sampler_does_not_inject_shuffle() -> None:
    sampler = torch.utils.data.SequentialSampler(range(8))
    task = make_runtime_task(
        dataloader_args={'sampler': sampler},
        batch_size=16,
    )
    fake_dataset = object()

    with patch('storegate.task.pytorch_task.StoreGateDataset', return_value=fake_dataset):
        with patch('storegate.task.pytorch_task.DataLoader', return_value='loader') as dataloader_cls:
            loader = task.get_dataloader('train')

    assert loader == 'loader'
    kwargs = dataloader_cls.call_args.kwargs
    assert kwargs['dataset'] is fake_dataset
    assert kwargs['sampler'] is sampler
    assert kwargs['batch_size'] == 16
    assert 'shuffle' not in kwargs


def test_get_dataloader_with_batch_sampler_does_not_inject_batch_size_or_shuffle() -> None:
    batch_sampler = torch.utils.data.BatchSampler(
        torch.utils.data.SequentialSampler(range(8)),
        batch_size=2,
        drop_last=False,
    )
    task = make_runtime_task(
        dataloader_args={'batch_sampler': batch_sampler},
        batch_size=16,
    )
    fake_dataset = object()

    with patch('storegate.task.pytorch_task.StoreGateDataset', return_value=fake_dataset):
        with patch('storegate.task.pytorch_task.DataLoader', return_value='loader') as dataloader_cls:
            loader = task.get_dataloader('train')

    assert loader == 'loader'
    kwargs = dataloader_cls.call_args.kwargs
    assert kwargs['dataset'] is fake_dataset
    assert kwargs['batch_sampler'] is batch_sampler
    assert 'batch_size' not in kwargs
    assert 'shuffle' not in kwargs
    assert 'sampler' not in kwargs


def test_fit_without_valid_returns_train_history_only() -> None:
    task = make_loop_task(
        input_var_names={'train': 'x', 'valid': None, 'test': 'x_test'},
        true_var_names={'train': 'y', 'valid': None, 'test': None},
        num_epochs=2,
    )
    task.get_dataloader.return_value = 'train-loader'
    task.step_epoch.side_effect = [{'loss': 0.1}, {'loss': 0.2}]

    result = task.fit()

    assert result == {'train': [{'loss': 0.1}, {'loss': 0.2}]}
    assert task.get_dataloader.call_args_list == [call('train')]
    assert task._ml.model.train.call_count == 2
    task._ml.model.eval.assert_not_called()


def test_fit_with_valid_runs_both_phases_each_epoch() -> None:
    task = make_loop_task(
        input_var_names={'train': 'x_train', 'valid': 'x_valid', 'test': 'x_test'},
        true_var_names={'train': 'y_train', 'valid': 'y_valid', 'test': None},
        num_epochs=2,
    )
    task.get_dataloader.side_effect = ['train-loader', 'valid-loader']
    task.step_epoch.side_effect = [
        {'train': 1},
        {'valid': 1},
        {'train': 2},
        {'valid': 2},
    ]

    result = task.fit()

    assert result == {
        'train': [{'train': 1}, {'train': 2}],
        'valid': [{'valid': 1}, {'valid': 2}],
    }
    assert task.step_epoch.call_args_list == [
        call(1, 'train', 'train-loader'),
        call(1, 'valid', 'valid-loader'),
        call(2, 'train', 'train-loader'),
        call(2, 'valid', 'valid-loader'),
    ]
    assert task._ml.model.train.call_count == 2
    assert task._ml.model.eval.call_count == 2


def test_fit_ignores_valid_phase_when_only_unrelated_storegate_vars_exist() -> None:
    task = make_loop_task(
        input_var_names={'train': 'x', 'valid': None, 'test': 'x_test'},
        true_var_names={'train': 'y', 'valid': None, 'test': None},
    )
    task.get_dataloader.return_value = 'train-loader'
    task.step_epoch.side_effect = [{'loss': 0.1}]

    result = task.fit()

    assert result == {'train': [{'loss': 0.1}]}
    assert task.get_dataloader.call_args_list == [call('train')]
    task._ml.model.eval.assert_not_called()


def test_fit_with_zero_epochs_skips_train_and_valid_dataloaders() -> None:
    task = make_loop_task(
        input_var_names={'train': 'x_train', 'valid': 'x_valid', 'test': 'x_test'},
        true_var_names={'train': 'y_train', 'valid': 'y_valid', 'test': None},
        num_epochs=0,
    )

    result = task.fit()

    assert result == {}
    task.get_dataloader.assert_not_called()
    task.step_epoch.assert_not_called()
    task._ml.model.train.assert_not_called()
    task._ml.model.eval.assert_not_called()


def test_predict_skips_when_test_phase_empty() -> None:
    task = make_loop_task(output_var_names=['pred'])
    task._storegate.get_var_names.return_value = []

    with patch('storegate.task.pytorch_task.logger.warn') as warn:
        result = task.predict()

    assert result == {'test': {}}
    warn.assert_called_once()
    task.get_dataloader.assert_not_called()
    task.step_epoch.assert_not_called()


def test_predict_raises_when_test_input_var_names_are_missing() -> None:
    task = make_loop_task(
        input_var_names={'test': 'x'},
        output_var_names={'test': 'pred'},
    )
    task._storegate.get_var_names.return_value = ['pred']

    with pytest.raises(
        ValueError,
        match=r"missing input_var_names=\['x'\]",
    ):
        task.predict()

    task.get_dataloader.assert_not_called()
    task.step_epoch.assert_not_called()


def test_predict_replaces_existing_outputs_after_running() -> None:
    tmp_pred = _internal_tmp_name()
    task = make_loop_task(
        input_var_names={'test': 'x'},
        output_var_names={'train': ['ignored'], 'test': 'pred'},
    )
    task._storegate.get_var_names.side_effect = [
        ['pred', 'x'],
        ['pred', 'x'],
        ['pred', 'x', tmp_pred],
        ['pred', 'x', 'backup_pred'],
    ]
    task.get_dataloader.return_value = 'test-loader'
    task.step_epoch.return_value = {'acc': 0.9}

    result = task.predict()

    assert result == {'test': {'acc': 0.9}}
    assert task._storegate.delete_data.call_args_list == [call('backup_pred', 'test')]
    assert task._storegate.rename_data.call_args_list == [
        call('pred', 'backup_pred', 'test'),
        call(tmp_pred, 'pred', 'test'),
    ]
    task._storegate.compile.assert_called_once()
    task._ml.model.eval.assert_called_once()
    task.get_dataloader.assert_called_once_with('test')
    task.step_epoch.assert_called_once_with(0, 'test', 'test-loader')


def test_predict_recompiles_after_writing_new_outputs() -> None:
    tmp_pred = _internal_tmp_name()
    task = make_loop_task(
        input_var_names={'test': 'x'},
        output_var_names={'test': 'pred'},
    )
    task._storegate.get_var_names.side_effect = [
        ['x'],
        ['x'],
        ['x', tmp_pred],
        ['x', 'pred'],
    ]
    task.get_dataloader.return_value = 'test-loader'
    task.step_epoch.return_value = {'acc': 0.9}

    result = task.predict()

    assert result == {'test': {'acc': 0.9}}
    task._storegate.delete_data.assert_not_called()
    task._storegate.rename_data.assert_called_once_with(tmp_pred, 'pred', 'test')
    task._storegate.compile.assert_called_once()
    task._ml.model.eval.assert_called_once()
    task.get_dataloader.assert_called_once_with('test')
    task.step_epoch.assert_called_once_with(0, 'test', 'test-loader')


def test_predict_restores_storegate_compiled_state_after_writing_outputs(tmp_path) -> None:
    sg = StoreGate(output_dir=str(tmp_path), mode='w', data_id='exp')
    sg.add_data('x', np.arange(8, dtype=np.float32).reshape(4, 2), phase='test')
    sg.compile()

    task = make_loop_task(
        input_var_names={'test': 'x'},
        output_var_names={'test': 'pred'},
        storegate=sg,
    )
    task.get_dataloader.return_value = 'test-loader'
    tmp_pred = _internal_tmp_name()

    def write_predictions(epoch: int, phase: str, dataloader: object) -> dict[str, float]:
        assert (epoch, phase, dataloader) == (0, 'test', 'test-loader')
        sg.add_data(tmp_pred, np.ones((4, 1), dtype=np.float32), phase='test')
        return {'acc': 1.0}

    task.step_epoch.side_effect = write_predictions

    result = task.predict()

    assert result == {'test': {'acc': 1.0}}
    assert len(sg['test']) == 4
    np.testing.assert_array_equal(
        sg.get_data('pred', 'test'),
        np.ones((4, 1), dtype=np.float32),
    )
    assert tmp_pred not in sg.get_var_names('test')


def test_predict_failure_preserves_existing_outputs_and_restores_compiled_state(tmp_path) -> None:
    sg = StoreGate(output_dir=str(tmp_path), mode='w', data_id='exp')
    sg.add_data('x', np.arange(8, dtype=np.float32).reshape(4, 2), phase='test')
    sg.add_data('pred', np.zeros((4, 1), dtype=np.float32), phase='test')
    sg.compile()

    task = make_loop_task(
        input_var_names={'test': 'x'},
        output_var_names={'test': 'pred'},
        storegate=sg,
    )
    task.get_dataloader.return_value = 'test-loader'
    tmp_pred = _internal_tmp_name()

    def fail_predictions(epoch: int, phase: str, dataloader: object) -> dict[str, float]:
        assert (epoch, phase, dataloader) == (0, 'test', 'test-loader')
        sg.add_data(tmp_pred, np.ones((2, 1), dtype=np.float32), phase='test')
        raise RuntimeError('boom')

    task.step_epoch.side_effect = fail_predictions

    with pytest.raises(RuntimeError, match='boom'):
        task.predict()

    np.testing.assert_array_equal(
        sg.get_data('pred', 'test'),
        np.zeros((4, 1), dtype=np.float32),
    )
    assert tmp_pred not in sg.get_var_names('test')
    assert len(sg['test']) == 4


def test_predict_promotion_failure_preserves_existing_outputs_and_restores_compiled_state(tmp_path) -> None:
    sg = StoreGate(output_dir=str(tmp_path), mode='w', data_id='exp')
    sg.add_data('x', np.arange(8, dtype=np.float32).reshape(4, 2), phase='test')
    sg.add_data('pred', np.zeros((4, 1), dtype=np.float32), phase='test')
    sg.compile()

    task = make_loop_task(
        input_var_names={'test': 'x'},
        output_var_names={'test': 'pred'},
        storegate=sg,
    )
    task.get_dataloader.return_value = 'test-loader'
    tmp_pred = _internal_tmp_name()

    def write_predictions(epoch: int, phase: str, dataloader: object) -> dict[str, float]:
        assert (epoch, phase, dataloader) == (0, 'test', 'test-loader')
        sg.add_data(tmp_pred, np.ones((2, 1), dtype=np.float32), phase='test')
        return {'acc': 1.0}

    original_rename_data = sg.rename_data

    def fail_promoting_predictions(var_name: str, output_var_name: str, phase: str) -> None:
        if (var_name, output_var_name, phase) == (tmp_pred, 'pred', 'test'):
            raise RuntimeError('promote failed')
        original_rename_data(var_name, output_var_name, phase)

    task.step_epoch.side_effect = write_predictions

    with patch.object(sg, 'rename_data', side_effect=fail_promoting_predictions):
        with pytest.raises(RuntimeError, match='promote failed'):
            task.predict()

    np.testing.assert_array_equal(
        sg.get_data('pred', 'test'),
        np.zeros((4, 1), dtype=np.float32),
    )
    assert tmp_pred not in sg.get_var_names('test')
    assert len(sg['test']) == 4


def test_predict_missing_tmp_outputs_preserves_existing_outputs_and_restores_compiled_state(tmp_path) -> None:
    sg = StoreGate(output_dir=str(tmp_path), mode='w', data_id='exp')
    sg.add_data('x', np.arange(8, dtype=np.float32).reshape(4, 2), phase='test')
    sg.add_data('pred', np.zeros((4, 1), dtype=np.float32), phase='test')
    sg.compile()

    task = make_loop_task(
        input_var_names={'test': 'x'},
        output_var_names={'test': 'pred'},
        storegate=sg,
    )
    task.get_dataloader.return_value = 'test-loader'
    task.step_epoch.return_value = {'acc': 1.0}
    tmp_pred = _internal_tmp_name()

    with pytest.raises(KeyError, match=tmp_pred):
        task.predict()

    np.testing.assert_array_equal(
        sg.get_data('pred', 'test'),
        np.zeros((4, 1), dtype=np.float32),
    )
    assert tmp_pred not in sg.get_var_names('test')
    assert len(sg['test']) == 4


def test_predict_preserves_user_tmp_named_variable(tmp_path) -> None:
    sg = StoreGate(output_dir=str(tmp_path), mode='w', data_id='exp')
    sg.add_data('x', np.arange(8, dtype=np.float32).reshape(4, 2), phase='test')
    sg.add_data('pred', np.zeros((4, 1), dtype=np.float32), phase='test')
    user_tmp_pred = np.full((4, 1), 7.0, dtype=np.float32)
    sg.add_data('tmp_pred', user_tmp_pred, phase='test')
    sg.compile()

    task = make_loop_task(
        input_var_names={'test': 'x'},
        output_var_names={'test': 'pred'},
        storegate=sg,
    )
    task.get_dataloader.return_value = 'test-loader'
    internal_tmp_pred = _internal_tmp_name()

    def write_predictions(epoch: int, phase: str, dataloader: object) -> dict[str, float]:
        assert (epoch, phase, dataloader) == (0, 'test', 'test-loader')
        sg.add_data(internal_tmp_pred, np.ones((4, 1), dtype=np.float32), phase='test')
        return {'acc': 1.0}

    task.step_epoch.side_effect = write_predictions

    result = task.predict()

    assert result == {'test': {'acc': 1.0}}
    np.testing.assert_array_equal(
        sg.get_data('pred', 'test'),
        np.ones((4, 1), dtype=np.float32),
    )
    np.testing.assert_array_equal(sg.get_data('tmp_pred', 'test'), user_tmp_pred)
    assert internal_tmp_pred not in sg.get_var_names('test')


def test_predict_preserves_test_output_order_even_when_shuffle_requested(tmp_path) -> None:
    sg = StoreGate(output_dir=str(tmp_path), mode='w', data_id='exp')
    inputs = np.arange(6, dtype=np.float32).reshape(6, 1)
    sg.add_data('x', inputs, phase='test')
    sg.compile()

    task = _PredictOnlyTask(
        storegate=sg,
        input_var_names={'train': None, 'valid': None, 'test': 'x'},
        true_var_names=None,
        output_var_names={'test': 'pred'},
        model=torch.nn.Identity,
        loss=torch.nn.MSELoss,
        optimizer=None,
        num_epochs=1,
        batch_size=1,
        device='cpu',
        dataloader_args={
            'shuffle': True,
            'generator': torch.Generator().manual_seed(0),
        },
    )
    task._pbar_args = {**task._pbar_args, 'disable': True}

    task.execute()

    np.testing.assert_array_equal(sg.get_data('pred', 'test'), inputs)


def test_step_epoch_writes_test_outputs_and_updates_progress() -> None:
    task = PytorchTask.__new__(PytorchTask)
    task._metrics = ['loss']
    task._ml = DLEnv()
    task._pbar_args = {'ncols': 20}
    task._num_epochs = 3
    task.step_batch = MagicMock(
        side_effect=[
            {'batch_size': 1, 'outputs': torch.tensor([[1.0]]), 'labels': None},
            {'batch_size': 1, 'outputs': torch.tensor([[2.0]]), 'labels': None},
        ]
    )
    task._output_to_storegate = MagicMock()

    fake_bars: list[_FakeTqdm] = []

    def make_bar(**kwargs: Any) -> _FakeTqdm:
        bar = _FakeTqdm(**kwargs)
        fake_bars.append(bar)
        return bar

    metric = MagicMock(side_effect=[{'loss': 0.1}, {'loss': 0.2}])

    with patch('storegate.task.pytorch_task.EpochMetric', return_value=metric) as metric_cls:
        with patch('storegate.task.pytorch_task.get_pbar_metric', side_effect=lambda result: {'loss': result['loss']}):
            with patch('storegate.task.pytorch_task.tqdm', side_effect=make_bar):
                result = task.step_epoch(2, 'test', ['batch0', 'batch1'])

    assert result == {'loss': 0.2}
    metric_cls.assert_called_once_with(['loss'], task._ml)
    assert task._output_to_storegate.call_count == 2
    assert torch.equal(task._output_to_storegate.call_args_list[0].args[0], torch.tensor([[1.0]]))
    assert torch.equal(task._output_to_storegate.call_args_list[1].args[0], torch.tensor([[2.0]]))
    assert fake_bars[0].descriptions == ['Epoch [   2/3] test ']
    assert fake_bars[0].postfixes == [{'loss': 0.1}, {'loss': 0.2}]
    assert fake_bars[0].updates == [1, 1]


# ---------------------------------------------------------------------------
# _output_to_storegate
# ---------------------------------------------------------------------------


def test_output_to_storegate_none_var_names_is_noop():
    task = make_task(output_var_names=None)
    task._output_to_storegate(make_tensor(1.0))
    task._storegate.add_data.assert_not_called()


@pytest.mark.parametrize(
    ('output_var_names', 'payload', 'expected_names'),
    [
        ({'train': ['train_pred'], 'test': 'pred'}, make_tensor(1.0), ['pred']),
        (['out'], make_tensor(2.0), ['out']),
        (['x', 'y'], [make_tensor(1.0), make_tensor(2.0)], ['x', 'y']),
    ],
)
def test_output_to_storegate_writes_expected_names(
    output_var_names: Any,
    payload: torch.Tensor | list[torch.Tensor],
    expected_names: list[str],
) -> None:
    task = make_task(output_var_names=output_var_names)
    task._output_to_storegate(payload)

    names_written = [call_.args[0] for call_ in task._storegate.add_data.call_args_list]
    assert names_written == expected_names
    assert all(call_.args[2] == 'test' for call_ in task._storegate.add_data.call_args_list)


@pytest.mark.parametrize(
    ('output_var_names', 'payload', 'message'),
    [
        (['only_one'], [make_tensor(1.0), make_tensor(2.0)], 'output_var_names'),
        (['a', 'b', 'c'], [make_tensor(1.0)], 'output_var_names'),
        (['a', 'b'], [make_tensor(1.0)], r'1.*2|2.*1'),
    ],
)
def test_output_to_storegate_length_mismatch_raises_without_partial_writes(
    output_var_names: list[str],
    payload: list[torch.Tensor],
    message: str,
) -> None:
    task = make_task(output_var_names=output_var_names)
    with pytest.raises(ValueError, match=message):
        task._output_to_storegate(payload)
    task._storegate.add_data.assert_not_called()


# ---------------------------------------------------------------------------
# build_module — string obj with modules=None raises ValueError
# ---------------------------------------------------------------------------


def test_build_module_string_with_none_modules_raises_actionable_error():
    with pytest.raises(ValueError) as exc_info:
        build_module('Linear', {}, None)

    message = str(exc_info.value)
    assert "'Linear'" in message
    assert 'class' in message
    assert 'torch' in message


def test_build_module_string_with_valid_modules_succeeds():
    import torch.nn as nn

    model = build_module('Linear', {'in_features': 2, 'out_features': 1}, nn)
    assert isinstance(model, nn.Linear)


def test_build_module_class_with_none_modules_succeeds():
    import torch.nn as nn

    model = build_module(nn.Linear, {'in_features': 2, 'out_features': 1}, None)
    assert isinstance(model, nn.Linear)


def test_build_module_instance_returns_deepcopy():
    import torch.nn as nn

    original = nn.Linear(2, 1)
    result = build_module(original, {}, None)
    assert isinstance(result, nn.Linear)
    assert result is not original


def test_build_module_instance_with_args_warns():
    import logging
    import logging.handlers
    import torch.nn as nn
    from storegate import logger as sg_logger

    handler = logging.handlers.MemoryHandler(capacity=100)
    handler.setLevel(logging.WARNING)
    sg_logger._logger.addHandler(handler)
    try:
        original = nn.Linear(2, 1)
        build_module(original, {'lr': 0.1}, None)
        messages = [h.getMessage() for h in handler.buffer]
        assert any('instance object is given but args is also provided' in m for m in messages)
    finally:
        sg_logger._logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# inputs_size
# ---------------------------------------------------------------------------

from storegate.task.pytorch.pytorch_util import inputs_size  # noqa: E402


@pytest.mark.parametrize(
    ('inputs', 'expected'),
    [
        (torch.randn(8, 3), 8),
        ([torch.randn(4, 3), torch.randn(4, 5)], 4),
        ((torch.randn(6, 2),), 6),
    ],
)
def test_inputs_size_tensor_like_inputs(inputs: Any, expected: int):
    assert inputs_size(inputs) == expected


def test_inputs_size_batch_size_attr():
    class FakeInput:
        batch_size = 16
    assert inputs_size(FakeInput()) == 16


def test_inputs_size_len_fallback():
    class FakeInput:
        def __len__(self):
            return 7
    assert inputs_size(FakeInput()) == 7


# ---------------------------------------------------------------------------
# DLTask.set_hps — empty suffix validation
# ---------------------------------------------------------------------------


def make_dl_task():
    """Return a minimal DLTask with mocked storegate."""
    task = DLTask.__new__(DLTask)
    task._storegate = MagicMock()
    task._data_id = None
    task._model_args = {}
    task._optimizer_args = {}
    task._loss_args = {}
    task._PROTECTED_KEYS = frozenset({'storegate', 'ml'})
    return task


@pytest.mark.parametrize(
    ('params', 'message'),
    [
        ({'model__': 64}, 'model__'),
        ({'optimizer__': 1e-3}, 'optimizer__'),
        ({'loss__': 0.1}, 'loss__'),
    ],
)
def test_set_hps_empty_suffix_raises(params: dict[str, Any], message: str):
    task = make_dl_task()
    with pytest.raises(ValueError, match=message):
        task.set_hps(params)


@pytest.mark.parametrize(
    ('params', 'target_attr', 'expected'),
    [
        ({'model__hidden': 128}, '_model_args', {'hidden': 128}),
        ({'optimizer__lr': 1e-4}, '_optimizer_args', {'lr': 1e-4}),
        ({'loss__weight': 0.5}, '_loss_args', {'weight': 0.5}),
    ],
)
def test_set_hps_valid_suffix_sets_arg(
    params: dict[str, Any],
    target_attr: str,
    expected: dict[str, Any],
):
    task = make_dl_task()
    task.set_hps(params)
    assert getattr(task, target_attr) == expected


def test_step_batch_test_phase_without_labels_skips_loss():
    task = make_step_task()
    task._true_var_names = compile_var_names(
        {'train': 'y_train', 'valid': 'y_valid', 'test': None}
    )
    result = task.step_batch(torch.tensor([[1.0, 2.0]]), phase='test')

    assert result['labels'] is None
    assert 'loss' not in result
    assert 'pred' in result
    task.step_loss.assert_not_called()
    task.step_optimizer.assert_not_called()


def test_step_batch_test_phase_with_multi_inputs_and_no_labels_keeps_all_inputs():
    task = make_step_task()
    task._true_var_names = compile_var_names(
        {'train': 'y_train', 'valid': 'y_valid', 'test': None}
    )
    x0 = torch.tensor([[1.0, 2.0]])
    x1 = torch.tensor([[3.0, 4.0]])

    result = task.step_batch([x0, x1], phase='test')

    assert result['labels'] is None
    passed_inputs = task.step_model.call_args.args[0]
    assert isinstance(passed_inputs, list)
    assert len(passed_inputs) == 2
    assert torch.equal(passed_inputs[0], x0)
    assert torch.equal(passed_inputs[1], x1)
    task.step_loss.assert_not_called()
    task.step_optimizer.assert_not_called()


def test_step_batch_test_phase_with_labels_still_unpacks_supervised_batch():
    task = make_step_task()
    task._true_var_names = compile_var_names(
        {'train': 'y_train', 'valid': 'y_valid', 'test': 'y_test'}
    )
    inputs = torch.tensor([[1.0, 2.0]])
    labels = torch.tensor([[1.0]])

    result = task.step_batch([inputs, labels], phase='test')

    assert torch.equal(result['labels'], labels)
    task.step_loss.assert_called_once()
    task.step_optimizer.assert_not_called()


def test_step_batch_train_without_labels_raises():
    task = make_step_task()
    with pytest.raises(ValueError, match='labels are required'):
        task.step_batch(torch.tensor([[1.0, 2.0]]), phase='train')


def test_step_batch_train_with_labels_calls_loss_and_optimizer():
    task = make_step_task()
    inputs = torch.tensor([[1.0, 2.0]])
    labels = torch.tensor([[1.0]])

    result = task.step_batch((inputs, labels), phase='train')

    assert torch.equal(result['labels'], labels)
    assert 'loss' in result
    assert torch.equal(result['outputs'], task.step_model.return_value)
    task.step_loss.assert_called_once()
    task.step_optimizer.assert_called_once_with(task.step_loss.return_value['loss'])


def test_step_batch_train_with_labels_and_missing_optimizer_raises_value_error():
    task = PytorchTask.__new__(PytorchTask)
    task._device = torch.device('cpu')
    task._ml = DLEnv(model=MagicMock(return_value=torch.tensor([[0.1]])), optimizer=None)
    task.step_model = MagicMock(return_value=torch.tensor([[0.1]]))
    task.step_loss = MagicMock(return_value={'loss': torch.tensor(0.3)})

    with pytest.raises(ValueError, match='optimizer is required for training'):
        task.step_batch((torch.tensor([[1.0, 2.0]]), torch.tensor([[1.0]])), phase='train')


def test_step_model_calls_underlying_model() -> None:
    task = PytorchTask.__new__(PytorchTask)
    task._ml = DLEnv(model=MagicMock(return_value=torch.tensor([1.0])))
    inputs = torch.tensor([2.0])

    result = task.step_model(inputs)

    assert torch.equal(result, torch.tensor([1.0]))
    task._ml.model.assert_called_once_with(inputs)


def test_step_loss_returns_loss_dict() -> None:
    task = PytorchTask.__new__(PytorchTask)
    task._ml = DLEnv(loss=MagicMock(return_value=torch.tensor(2.0)))

    result = task.step_loss(torch.tensor([1.0]), torch.tensor([1.0]))

    assert result['subloss'] == []
    assert torch.equal(result['loss'], torch.tensor(2.0))


def test_step_optimizer_runs_zero_grad_backward_and_step() -> None:
    task = PytorchTask.__new__(PytorchTask)
    optimizer = MagicMock()
    task._ml = DLEnv(optimizer=optimizer)
    loss = MagicMock()

    task.step_optimizer(loss)

    optimizer.zero_grad.assert_called_once_with(set_to_none=True)
    loss.backward.assert_called_once()
    optimizer.step.assert_called_once()


def test_step_optimizer_raises_when_optimizer_missing() -> None:
    task = PytorchTask.__new__(PytorchTask)
    task._ml = DLEnv(optimizer=None)

    with pytest.raises(ValueError, match='optimizer is required for training'):
        task.step_optimizer(MagicMock())


def test_add_device_recurses_over_lists() -> None:
    task = PytorchTask.__new__(PytorchTask)
    tensors = [torch.tensor([1.0]), torch.tensor([2.0])]

    result = task.add_device(tensors, torch.device('cpu'))

    assert isinstance(result, list)
    assert all(t.device.type == 'cpu' for t in result)


def test_get_pbar_description_formats_epoch_and_phase() -> None:
    task = PytorchTask.__new__(PytorchTask)
    task._num_epochs = 12
    assert task._get_pbar_description(3, 'train') == 'Epoch [   3/12] train'


# ---------------------------------------------------------------------------
# Coverage gap: _tmp_test_output_var_names — None input & collision
# ---------------------------------------------------------------------------


def test_tmp_test_output_var_names_returns_none_when_output_var_names_is_none() -> None:
    result = PytorchTask._tmp_test_output_var_names(None, set())
    assert result is None


def test_tmp_test_output_var_names_handles_collision_with_reserved_names() -> None:
    reserved = {'__storegate_predict_tmp_0'}
    result = PytorchTask._tmp_test_output_var_names(['pred'], reserved)
    assert result == ['__storegate_predict_tmp_0_1']
    assert '__storegate_predict_tmp_0_1' in reserved


# ---------------------------------------------------------------------------
# Coverage gap: _backup_test_output_var_name — collision
# ---------------------------------------------------------------------------


def test_backup_test_output_var_name_handles_collision() -> None:
    reserved = {'backup_pred'}
    result = PytorchTask._backup_test_output_var_name('pred', reserved)
    assert result == 'backup_pred_1'
    assert 'backup_pred_1' in reserved


# ---------------------------------------------------------------------------
# Coverage gap: _delete_test_outputs — None input
# ---------------------------------------------------------------------------


def test_delete_test_outputs_returns_false_when_none() -> None:
    task = make_task(output_var_names=['pred'])
    assert task._delete_test_outputs(None) is False
    task._storegate.get_var_names.assert_not_called()


# ---------------------------------------------------------------------------
# Coverage gap: _promote_tmp_test_outputs — rollback deletes promoted vars
# ---------------------------------------------------------------------------


def test_promote_tmp_test_outputs_rollback_deletes_partially_promoted_vars(tmp_path) -> None:
    sg = StoreGate(output_dir=str(tmp_path), mode='w', data_id='exp')
    sg.add_data('x', np.arange(8, dtype=np.float32).reshape(4, 2), phase='test')
    sg.add_data('pred0', np.zeros((4, 1), dtype=np.float32), phase='test')
    sg.add_data('pred1', np.zeros((4, 1), dtype=np.float32), phase='test')

    tmp0 = '__storegate_predict_tmp_0'
    tmp1 = '__storegate_predict_tmp_1'
    sg.add_data(tmp0, np.ones((4, 1), dtype=np.float32), phase='test')
    sg.add_data(tmp1, np.full((4, 1), 2.0, dtype=np.float32), phase='test')
    sg.compile()

    task = make_loop_task(
        input_var_names={'test': 'x'},
        output_var_names={'test': ['pred0', 'pred1']},
        storegate=sg,
    )

    original_rename = sg.rename_data
    call_count = 0

    def fail_on_second_promote(var_name, output_var_name, phase):
        nonlocal call_count
        call_count += 1
        # Let backup renames and first promote succeed, fail on second promote
        if call_count == 4:  # 2 backups + 1 promote succeed, 4th fails
            raise RuntimeError('promote boom')
        original_rename(var_name, output_var_name, phase)

    with patch.object(sg, 'rename_data', side_effect=fail_on_second_promote):
        with pytest.raises(RuntimeError, match='promote boom'):
            task._promote_tmp_test_outputs([tmp0, tmp1], ['pred0', 'pred1'])

    # Originals should be restored
    np.testing.assert_array_equal(
        sg.get_data('pred0', 'test'),
        np.zeros((4, 1), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        sg.get_data('pred1', 'test'),
        np.zeros((4, 1), dtype=np.float32),
    )
    assert tmp0 not in sg.get_var_names('test')
    assert tmp1 not in sg.get_var_names('test')


# ---------------------------------------------------------------------------
# Coverage gap: _validate_test_inputs_exist — not dict / empty
# ---------------------------------------------------------------------------


def test_validate_test_inputs_exist_raises_when_not_dict() -> None:
    task = PytorchTask.__new__(PytorchTask)
    task._input_var_names = 'x'  # not a dict (pre-compile state)
    with pytest.raises(ValueError, match="requires input_var_names"):
        task._validate_test_inputs_exist(set())


def test_validate_test_inputs_exist_raises_when_test_phase_is_none() -> None:
    task = PytorchTask.__new__(PytorchTask)
    task._input_var_names = {'train': ['x'], 'valid': ['x'], 'test': None}
    with pytest.raises(ValueError, match="requires input_var_names"):
        task._validate_test_inputs_exist(set())


# ---------------------------------------------------------------------------
# Coverage gap: step_batch — valid phase without labels raises
# ---------------------------------------------------------------------------


def test_step_batch_valid_without_labels_raises():
    task = make_step_task()
    task._true_var_names = compile_var_names(
        {'train': 'y_train', 'valid': 'y_valid', 'test': None}
    )
    # valid phase expects labels; passing non-tuple data triggers the error
    with pytest.raises(ValueError, match='labels are required'):
        task.step_batch(torch.tensor([[1.0, 2.0]]), phase='valid')


# ---------------------------------------------------------------------------
# Coverage gap: _phase_expects_labels — true_var_names not a dict
# ---------------------------------------------------------------------------


def test_phase_expects_labels_returns_false_when_true_var_names_not_dict() -> None:
    task = PytorchTask.__new__(PytorchTask)
    task._true_var_names = 'y'  # string, not dict (pre-compile state)
    assert task._phase_expects_labels('test') is False
