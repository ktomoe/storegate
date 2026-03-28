import copy
import io
from typing import Any

import torch
from torch import optim
from torch.nn.modules import loss as tl
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm

from storegate.task import DLTask
from storegate.task.pytorch_metrics import EpochMetric
from storegate import utilities as util
from storegate.task.dataset.storegate_dataset import StoreGateDataset
from storegate.task.dl_env import LearningMode


class PytorchTask(DLTask):
    """DL task class for the default functions."""

    def __init__(self,
                 device: str = 'auto',
                 dataset: Any = None,
                 dataset_args: dict[str, Any] | None = None,
                 dataloader: Any = None,
                 dataloader_args: dict[str, Any] | None = None,
                 non_blocking: bool = False,
                 progress_bar: bool = False,
                 metric_interval: int = 1,
                 torchinfo: bool = False,
                 torch_compile: bool = False,
                 **kwargs: Any):
        """Initialize the pytorch base task."""
        super().__init__(**kwargs)

        if dataset_args is None:
            dataset_args = {}

        if dataloader_args is None:
            dataloader_args = {}

        if metric_interval < 1:
            raise ValueError("metric_interval must be >= 1.")

        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._device = device
        self._dataset = dataset
        self._dataset_args = dataset_args
        self._dataloader = dataloader
        self._dataloader_args = dataloader_args
        self._non_blocking = non_blocking
        self._progress_bar = progress_bar
        self._metric_interval = metric_interval
        self._torchinfo = torchinfo
        self._torch_compile = torch_compile

        self._cuda_id: int | None = None

    def set_cuda_id(self, cuda_id: int) -> None:
        """Set the CUDA device ID used during compilation."""
        self._cuda_id = cuda_id

    def _compile_after_base(self) -> None:
        self._compile_dataset()
        self._compile_dataloader()

    def _compile_device(self) -> None:
        """Compile device."""
        if self._cuda_id is not None:
            self._dl.device = torch.device(f'cuda:{self._cuda_id}')
        else:
            self._dl.device = torch.device(self._device)

    def _compile_model(self) -> None:
        """Compile model."""
        if self._model is None:
            raise ValueError('model is required. Pass a class, instance, or string name via model=..., '
                             'or override _step_model() in a subclass.')

        self._dl.model = util.build_module(self._model, self._model_args, torch.nn)
        self._dl.model.to(self._dl.device)

        if self._torch_compile:
            self._dl.model = torch.compile(self._dl.model)

        if self._torchinfo:
            from torchinfo import summary
            summary(self._dl.model)

    def _compile_optimizer(self) -> None:
        """Compile optimizer."""
        if self._optimizer is None:
            return

        if isinstance(self._optimizer, optim.Optimizer):
            raise ValueError(
                "optimizer instance is not supported. "
                "Pass an optimizer class or string name via optimizer=..., "
                "and configure it with optimizer_args=...."
            )

        optimizer_args = copy.deepcopy(self._optimizer_args)
        optimizer_args['params'] = list(self._dl.model.parameters())
        self._dl.optimizer = util.build_module(self._optimizer, optimizer_args, optim)

    def _compile_loss(self) -> None:
        """Compile loss."""
        if self._loss is None:
            return

        self._dl.loss = util.build_module(self._loss, self._loss_args, tl)
        self._dl.loss = self._dl.loss.to(self._dl.device)

    def _compile_dataset(self) -> None:
        """Compile dataset."""
        for phase_name, phase_value in self._dl.phase.iter_active_items():
            phase_args = self._phase_args(self._dataset_args, phase_name)

            if self._dataset is None:
                base_args = {
                    'storegate': self._storegate,
                    'phase': phase_value,
                    'input_var_names': self._dl.var_name.inputs,
                    'true_var_names': self._dl.var_name.true,
                    'device': self._dl.device,
                }
                base_args.update(phase_args)
                dataset = StoreGateDataset(**base_args)
            else:
                dataset = util.build_module(self._dataset, phase_args, None)

            setattr(self._dl.dataset, phase_name, dataset)

    def _compile_dataloader(self) -> None:
        """Compile dataloader."""
        for phase_name, _phase_value in self._dl.phase.iter_active_items():
            phase_dataset = getattr(self._dl.dataset, phase_name)
            phase_args = self._phase_args(self._dataloader_args, phase_name)
            self._validate_dataloader_args_for_dataset(phase_name, phase_dataset, phase_args)

            if self._dataloader is None:
                base_args = {'dataset': phase_dataset}
                base_args.update(phase_args)
                dataloader = TorchDataLoader(**base_args)
            else:
                dl_args = copy.deepcopy(phase_args)
                dl_args['dataset'] = phase_dataset
                dataloader = util.build_module(self._dataloader, dl_args, None)

            setattr(self._dl.dataloader, phase_name, dataloader)

    def _validate_dataloader_args_for_dataset(
        self,
        phase_name: str,
        dataset: Any,
        phase_args: dict[str, Any],
    ) -> None:
        if getattr(dataset, 'preload_mode', None) != 'gpu':
            return

        num_workers = phase_args.get('num_workers', 0)
        if num_workers != 0:
            raise ValueError(
                f"dataloader_args['{phase_name}']['num_workers'] must be 0 when preload='gpu'."
            )

        if phase_args.get('pin_memory', False):
            raise ValueError(
                f"dataloader_args['{phase_name}']['pin_memory'] must be False when preload='gpu'."
            )

    def _phase_args(
        self,
        args: dict[str, Any],
        phase_name: str,
    ) -> dict[str, Any]:
        """Return phase-specific args when train/valid/test mapping is used."""
        if (
            isinstance(args, dict)
            and args.keys() <= self._PHASE_KEYS
            and all(isinstance(value, dict) for value in args.values())
        ):
            return args.get(phase_name, {})
        return args

    def fit(self) -> dict[str, Any]:
        """Train model over epoch."""
        self._dl.require_compiled("fit")
        if self._num_epochs == 0:
            return {}
        self._dl.require_fit_ready()

        rtn_history: dict[str, list[dict[str, Any]]] = {'train': [], 'valid': []}

        for epoch in range(1, self._num_epochs + 1):
            self._dl.model.train()
            rtn_train = self._step_epoch(epoch, 'train')
            rtn_history['train'].append(rtn_train)

            if self._dl.phase.valid:
                self._dl.model.eval()
                with torch.inference_mode():
                    rtn_valid = self._step_epoch(epoch, 'valid')
                rtn_history['valid'].append(rtn_valid)

        return rtn_history

    def predict(self) -> dict[str, Any]:
        """Predict and upload outputs to storegate."""
        self._dl.require_compiled("predict")
        self._dl.require_predict_ready()
        test_phase = self._dl.phase.test
        if test_phase is None:
            raise RuntimeError("predict() requires a configured test phase.")
        output_var_names = self._dl.var_name.outputs

        self._dl.model.eval()

        if output_var_names:
            sg = self._resolve_storegate()
            with sg.staged_add(test_phase, output_var_names) as tx:
                with torch.inference_mode():
                    rtn_result = self._step_epoch(0, 'test', staged_tx=tx)
        else:
            with torch.inference_mode():
                rtn_result = self._step_epoch(0, 'test')

        return rtn_result

    def _step_epoch(
        self,
        epoch: int,
        phase: str,
        staged_tx: Any = None,
    ) -> dict[str, Any]:
        """Process model for given epoch and phase."""
        dataloader = getattr(self._dl.dataloader, phase)
        phase_metrics = getattr(self._dl.metrics, phase)
        epoch_metric = EpochMetric(phase_metrics or None)

        try:
            total_batches: int | None = len(dataloader)
        except TypeError:
            total_batches = None

        if self._progress_bar:
            desc = f"Epoch {epoch}/{self._num_epochs} [{phase}]" if epoch > 0 else f"[{phase}]"
            pbar = tqdm(
                dataloader,
                desc=desc,
                leave=True,
            )
        else:
            pbar = tqdm(dataloader, file=io.StringIO(), leave=False)

        last_progress_metrics: dict[str, Any] = {}

        for batch_idx, data in enumerate(pbar, 1):
            rtn_batch = self._step_batch(data, phase)
            epoch_metric.update(rtn_batch)
            current_metrics = last_progress_metrics
            metrics_sampled = False

            if self._progress_bar or self._progress_callback is not None:
                metrics_sampled = self._should_sample_metrics(
                    batch_idx,
                    total_batches,
                )
                if metrics_sampled:
                    current_metrics = epoch_metric.result()
                    last_progress_metrics = current_metrics

                if self._progress_bar and metrics_sampled:
                    display = {k: f"{v:.4f}" for k, v in current_metrics.items()
                               if isinstance(v, (int, float))}
                    pbar.set_postfix(display)

            if self._progress_callback is not None:
                batch_info: dict[str, Any] = {
                    'epoch': epoch,
                    'num_epochs': self._num_epochs,
                    'phase': phase,
                    'batch': batch_idx,
                    'total_batches': total_batches,
                    'metrics': current_metrics,
                    'metrics_sampled': metrics_sampled,
                }
                rate = pbar.format_dict.get("rate")
                if rate is not None:
                    batch_info['batch_rate'] = rate
                self._on_batch_end(batch_info)

            if staged_tx is not None and self._dl.var_name.outputs:
                expected_outputs = len(self._dl.var_name.outputs)
                actual_outputs = len(rtn_batch['preds'])
                if actual_outputs != expected_outputs:
                    raise ValueError(
                        "Model output count does not match configured output var_names: "
                        f"expected {expected_outputs}, got {actual_outputs}. "
                        f"output_var_names={self._dl.var_name.outputs}"
                    )
                for name, pred in zip(self._dl.var_name.outputs, rtn_batch['preds']):
                    staged_tx.add_data(name, self._detach_numpy(pred))

        epoch_result = epoch_metric.result()

        return epoch_result

    def _should_sample_metrics(
        self,
        batch_idx: int,
        total_batches: int | None,
    ) -> bool:
        """Sample metrics on the first batch, every N batches, and the last batch."""
        if batch_idx == 1:
            return True
        if total_batches is not None and batch_idx >= total_batches:
            return True
        return batch_idx % self._metric_interval == 0

    def _step_batch(self, data: Any, phase: str) -> dict[str, Any]:
        if not isinstance(data, (tuple, list)):
            raise ValueError(
                "Each batch must be a 2-element tuple or list: (data, label). "
                f"Got {type(data).__name__}."
            )
        if len(data) != 2:
            raise ValueError(
                "Each batch must be a 2-element tuple or list: (data, label). "
                f"Got length {len(data)}."
            )

        inputs, labels = data

        batch_size = self._infer_batch_size(inputs)
        if batch_size is None:
            raise ValueError("Could not infer batch_size from inputs. "
                             "Expected a Tensor or a non-empty list/tuple of Tensors.")

        rtn_result: dict[str, Any] = {'batch_size': batch_size}

        inputs = self._add_device(inputs)
        labels = self._add_device(labels)

        outputs = self._step_model(inputs)

        if self._dl.loss is not None:
            loss = self._step_loss(outputs, labels, inputs)

            if loss is not None:
                if phase == 'train':
                    self._step_optimizer(loss)

                rtn_result['loss'] = self._detach_value(loss)

        if isinstance(outputs, (list, tuple)):
            rtn_result['preds'] = [self._detach_value(o) for o in outputs]
        else:
            rtn_result['preds'] = [self._detach_value(outputs)]

        if isinstance(labels, (list, tuple)):
            rtn_result['labels'] = [self._detach_value(label) for label in labels]
        else:
            rtn_result['labels'] = [self._detach_value(labels)]

        return rtn_result

    def _step_model(
        self,
        inputs: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]:
        """Process model."""
        if isinstance(inputs, (list, tuple)):
            outputs = self._dl.model(*inputs)
        else:
            outputs = self._dl.model(inputs)
        return outputs

    def _step_loss(
        self,
        outputs: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        labels: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        inputs: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> torch.Tensor | None:
        """Process loss function.

        Returns ``None`` when *labels* are absent (e.g. during prediction),
        which causes ``_step_batch`` to skip optimizer and loss recording.

        Override this method to customize loss computation.
        For example, unsupervised tasks can ignore *labels* and
        compute reconstruction loss from *outputs* and *inputs*.
        """
        if not self._infer_batch_size(labels):
            return None
        return self._dl.loss(outputs, labels)

    def _step_optimizer(self, loss: torch.Tensor) -> None:
        """Process optimizer."""
        self._dl.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._dl.optimizer.step()

    def _add_device(
        self,
        data: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]:
        """Add data to device."""
        if isinstance(data, list):
            return [
                idata.to(self._dl.device, non_blocking=self._non_blocking)
                for idata in data
            ]
        if isinstance(data, tuple):
            return tuple(
                idata.to(self._dl.device, non_blocking=self._non_blocking)
                for idata in data
            )
        return data.to(self._dl.device, non_blocking=self._non_blocking)

    @staticmethod
    def _detach_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach()
        return value

    @staticmethod
    def _detach_numpy(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return value

    ################################################################################
    def _get_hps_namespace_targets(self) -> dict[str, dict[str, Any]]:
        targets = super()._get_hps_namespace_targets()
        targets['dataset'] = self._dataset_args
        targets['dataloader'] = self._dataloader_args
        return targets

    def _infer_batch_size(self, data: Any) -> int | None:
        """Infer batch size from a Tensor or a non-empty Tensor sequence."""
        if isinstance(data, torch.Tensor):
            if data.ndim == 0:
                return 1
            return int(data.shape[0])

        if isinstance(data, (list, tuple)):
            if not data:
                return None
            first = data[0]
            if not isinstance(first, torch.Tensor):
                return None
            if first.ndim == 0:
                return 1
            return int(first.shape[0])

        return None


class SelfSupervisedPytorchTask(PytorchTask):
    """DL task class for self-supervised learning."""

    def _compile_after_base(self) -> None:
        self._compile_dataset()
        self._compile_dataloader()
        self._dl.learning_mode = LearningMode.SELF_SUPERVISED


class UnsupervisedPytorchTask(PytorchTask):
    """DL task class for unsupervised learning."""

    def _compile_after_base(self) -> None:
        self._compile_dataset()
        self._compile_dataloader()
        self._dl.learning_mode = LearningMode.UNSUPERVISED

    def _step_loss(
        self,
        outputs: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        labels: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        inputs: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Compute loss from outputs and inputs (ignoring labels)."""
        return self._dl.loss(outputs, inputs)
