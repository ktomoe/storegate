"""Unit tests for PytorchTask and pytorch_util.build_module."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

torch = pytest.importorskip('torch')

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


# ---------------------------------------------------------------------------
# __init__ / compile
# ---------------------------------------------------------------------------


def test_init_auto_device_sets_cpu_and_default_metrics(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    task = PytorchTask(storegate=MagicMock())
    assert task._device == torch.device('cpu')
    assert task._metrics == ['loss']
    assert task._is_gpu is False


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

    build_module.assert_called_once_with(task._model, task._model_args, None)
    fake_model.to.assert_called_once_with(task._device)
    summary.assert_called_once_with(fake_model)
    torch_compile.assert_called_once_with(fake_model)
    assert task._ml.model == 'compiled-model'


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


def test_get_dataloader_builds_train_loader_with_defaults() -> None:
    task = make_runtime_task(
        input_var_names={'train': 'x_train', 'valid': 'x_valid', 'test': 'x_test'},
        true_var_names={'train': 'y_train', 'valid': 'y_valid', 'test': None},
        dataset_args={'preload': False},
        dataloader_args={'num_workers': 2},
        batch_size=16,
    )
    fake_dataset = object()

    with patch('storegate.task.pytorch_task.StoreGateDataset', return_value=fake_dataset) as dataset_cls:
        with patch('storegate.task.pytorch_task.DataLoader', return_value='loader') as dataloader_cls:
            loader = task.get_dataloader('train')

    assert loader == 'loader'
    dataset_cls.assert_called_once_with(
        task._storegate,
        'train',
        input_var_names=['x_train'],
        true_var_names=['y_train'],
        preload=False,
    )
    kwargs = dataloader_cls.call_args.kwargs
    assert kwargs['dataset'] is fake_dataset
    assert kwargs['batch_size'] == 16
    assert kwargs['shuffle'] is True
    assert kwargs['num_workers'] == 2


def test_get_dataloader_defaults_shuffle_false_for_valid() -> None:
    task = make_runtime_task(
        batch_size=8,
        input_var_names={'train': 'x_train', 'valid': 'x_valid', 'test': 'x_test'},
        true_var_names={'train': 'y_train', 'valid': 'y_valid', 'test': None},
    )

    with patch('storegate.task.pytorch_task.StoreGateDataset', return_value='dataset') as dataset_cls:
        with patch('storegate.task.pytorch_task.DataLoader', return_value='loader') as dataloader_cls:
            task.get_dataloader('valid')

    assert dataset_cls.call_args.kwargs['input_var_names'] == ['x_valid']
    assert dataset_cls.call_args.kwargs['true_var_names'] == ['y_valid']
    assert dataloader_cls.call_args.kwargs['shuffle'] is False


def test_get_dataloader_uses_test_phase_var_names() -> None:
    task = make_runtime_task(
        input_var_names={'train': 'x_train', 'valid': 'x_valid', 'test': 'x_test'},
        true_var_names={'train': 'y_train', 'valid': 'y_valid', 'test': None},
    )

    with patch('storegate.task.pytorch_task.StoreGateDataset', return_value='dataset') as dataset_cls:
        with patch('storegate.task.pytorch_task.DataLoader', return_value='loader'):
            task.get_dataloader('test')

    assert dataset_cls.call_args.kwargs['input_var_names'] == ['x_test']
    assert dataset_cls.call_args.kwargs['true_var_names'] is None


def test_fit_without_valid_returns_train_history_only() -> None:
    task = PytorchTask.__new__(PytorchTask)
    task._storegate = MagicMock()
    task._input_var_names = compile_var_names({'train': 'x', 'valid': None, 'test': 'x_test'})
    task._true_var_names = compile_var_names({'train': 'y', 'valid': None, 'test': None})
    task.get_dataloader = MagicMock(return_value='train-loader')
    task.step_epoch = MagicMock(side_effect=[{'loss': 0.1}, {'loss': 0.2}])
    task._ml = DLEnv(model=MagicMock())
    task._num_epochs = 2

    result = task.fit()

    assert result == {'train': [{'loss': 0.1}, {'loss': 0.2}]}
    assert task.get_dataloader.call_args_list == [call('train')]
    assert task._ml.model.train.call_count == 2
    task._ml.model.eval.assert_not_called()


def test_fit_with_valid_runs_both_phases_each_epoch() -> None:
    task = PytorchTask.__new__(PytorchTask)
    task._storegate = MagicMock()
    task._input_var_names = compile_var_names({'train': 'x_train', 'valid': 'x_valid', 'test': 'x_test'})
    task._true_var_names = compile_var_names({'train': 'y_train', 'valid': 'y_valid', 'test': None})
    task.get_dataloader = MagicMock(side_effect=['train-loader', 'valid-loader'])
    task.step_epoch = MagicMock(
        side_effect=[
            {'train': 1},
            {'valid': 1},
            {'train': 2},
            {'valid': 2},
        ]
    )
    task._ml = DLEnv(model=MagicMock())
    task._num_epochs = 2

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
    task = PytorchTask.__new__(PytorchTask)
    task._storegate = MagicMock()
    task._input_var_names = compile_var_names({'train': 'x', 'valid': None, 'test': 'x_test'})
    task._true_var_names = compile_var_names({'train': 'y', 'valid': None, 'test': None})
    task.get_dataloader = MagicMock(return_value='train-loader')
    task.step_epoch = MagicMock(side_effect=[{'loss': 0.1}])
    task._ml = DLEnv(model=MagicMock())
    task._num_epochs = 1

    result = task.fit()

    assert result == {'train': [{'loss': 0.1}]}
    assert task.get_dataloader.call_args_list == [call('train')]
    task._ml.model.eval.assert_not_called()


def test_predict_skips_when_test_phase_empty() -> None:
    task = PytorchTask.__new__(PytorchTask)
    task._output_var_names = compile_var_names(['pred'])
    task._storegate = MagicMock()
    task._storegate.get_var_names.return_value = []
    task._ml = DLEnv(model=MagicMock())
    task.get_dataloader = MagicMock()
    task.step_epoch = MagicMock()

    with patch('storegate.task.pytorch_task.logger.warn') as warn:
        result = task.predict()

    assert result == {'test': {}}
    warn.assert_called_once()
    task.get_dataloader.assert_not_called()
    task.step_epoch.assert_not_called()


def test_predict_deletes_existing_outputs_before_running() -> None:
    task = PytorchTask.__new__(PytorchTask)
    task._output_var_names = compile_var_names({'train': ['ignored'], 'test': 'pred'})
    task._storegate = MagicMock()
    task._storegate.get_var_names.side_effect = [['pred', 'x'], ['x']]
    task._ml = DLEnv(model=MagicMock())
    task.get_dataloader = MagicMock(return_value='test-loader')
    task.step_epoch = MagicMock(return_value={'acc': 0.9})

    result = task.predict()

    assert result == {'test': {'acc': 0.9}}
    task._storegate.delete_data.assert_called_once_with('pred', 'test')
    task._storegate.compile.assert_called_once()
    task._ml.model.eval.assert_called_once()
    task.get_dataloader.assert_called_once_with('test')
    task.step_epoch.assert_called_once_with(0, 'test', 'test-loader')


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
# _output_to_storegate — output_var_names is None
# ---------------------------------------------------------------------------


def test_output_to_storegate_none_var_names_is_noop():
    task = make_task(output_var_names=None)
    task._output_to_storegate(make_tensor(1.0))
    task._storegate.add_data.assert_not_called()


# ---------------------------------------------------------------------------
# _output_to_storegate — single output
# ---------------------------------------------------------------------------


def test_output_to_storegate_single_tensor_calls_add_data():
    task = make_task(output_var_names={'train': ['train_pred'], 'test': 'pred'})
    t = make_tensor(1.0)
    task._output_to_storegate(t)
    task._storegate.add_data.assert_called_once()
    args = task._storegate.add_data.call_args
    assert args[0][0] == 'pred'
    assert args[0][2] == 'test'


def test_output_to_storegate_single_tensor_wrapped_in_list():
    """A bare Tensor (not list) must be wrapped before zip."""
    task = make_task(output_var_names=['out'])
    task._output_to_storegate(make_tensor(2.0))
    assert task._storegate.add_data.call_count == 1


# ---------------------------------------------------------------------------
# _output_to_storegate — multiple outputs
# ---------------------------------------------------------------------------


def test_output_to_storegate_list_calls_add_data_for_each():
    task = make_task(output_var_names=['a', 'b'])
    task._output_to_storegate([make_tensor(1.0), make_tensor(2.0)])
    assert task._storegate.add_data.call_count == 2


def test_output_to_storegate_list_maps_names_to_outputs():
    task = make_task(output_var_names=['x', 'y'])
    task._output_to_storegate([make_tensor(1.0), make_tensor(2.0)])
    names_written = [c[0][0] for c in task._storegate.add_data.call_args_list]
    assert names_written == ['x', 'y']


# ---------------------------------------------------------------------------
# _output_to_storegate — length mismatch raises ValueError
# ---------------------------------------------------------------------------


def test_output_to_storegate_more_outputs_than_var_names_raises():
    task = make_task(output_var_names=['only_one'])
    with pytest.raises(ValueError, match='output_var_names'):
        task._output_to_storegate([make_tensor(1.0), make_tensor(2.0)])


def test_output_to_storegate_fewer_outputs_than_var_names_raises():
    task = make_task(output_var_names=['a', 'b', 'c'])
    with pytest.raises(ValueError, match='output_var_names'):
        task._output_to_storegate([make_tensor(1.0)])


def test_output_to_storegate_mismatch_error_contains_counts():
    task = make_task(output_var_names=['a', 'b'])
    with pytest.raises(ValueError, match=r'1.*2|2.*1'):
        task._output_to_storegate([make_tensor(1.0)])


def test_output_to_storegate_mismatch_does_not_write_partial_data():
    """No add_data calls must occur when lengths mismatch."""
    task = make_task(output_var_names=['a', 'b'])
    with pytest.raises(ValueError):
        task._output_to_storegate([make_tensor(1.0)])
    task._storegate.add_data.assert_not_called()


# ---------------------------------------------------------------------------
# build_module — string obj with modules=None raises ValueError
# ---------------------------------------------------------------------------


def test_build_module_string_with_none_modules_raises_value_error():
    with pytest.raises(ValueError, match="'Linear'"):
        build_module('Linear', {}, None)


def test_build_module_string_with_none_modules_error_mentions_class():
    with pytest.raises(ValueError, match='class'):
        build_module('Linear', {}, None)


def test_build_module_string_with_none_modules_error_mentions_torch():
    with pytest.raises(ValueError, match='torch'):
        build_module('Linear', {}, None)


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


def test_inputs_size_tensor():
    t = torch.randn(8, 3)
    assert inputs_size(t) == 8


def test_inputs_size_list_of_tensors():
    ts = [torch.randn(4, 3), torch.randn(4, 5)]
    assert inputs_size(ts) == 4


def test_inputs_size_tuple_of_tensors():
    ts = (torch.randn(6, 2),)
    assert inputs_size(ts) == 6


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


def test_set_hps_model_empty_suffix_raises():
    task = make_dl_task()
    with pytest.raises(ValueError, match='model__'):
        task.set_hps({'model__': 64})


def test_set_hps_optimizer_empty_suffix_raises():
    task = make_dl_task()
    with pytest.raises(ValueError, match='optimizer__'):
        task.set_hps({'optimizer__': 1e-3})


def test_set_hps_loss_empty_suffix_raises():
    task = make_dl_task()
    with pytest.raises(ValueError, match='loss__'):
        task.set_hps({'loss__': 0.1})


def test_set_hps_model_valid_suffix_sets_arg():
    task = make_dl_task()
    task.set_hps({'model__hidden': 128})
    assert task._model_args['hidden'] == 128


def test_set_hps_optimizer_valid_suffix_sets_arg():
    task = make_dl_task()
    task.set_hps({'optimizer__lr': 1e-4})
    assert task._optimizer_args['lr'] == 1e-4


def test_set_hps_loss_valid_suffix_sets_arg():
    task = make_dl_task()
    task.set_hps({'loss__weight': 0.5})
    assert task._loss_args['weight'] == 0.5


def test_step_batch_test_phase_without_labels_skips_loss():
    task = make_step_task()
    result = task.step_batch(torch.tensor([[1.0, 2.0]]), phase='test')

    assert result['labels'] is None
    assert 'loss' not in result
    assert 'pred' in result
    task.step_loss.assert_not_called()
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
