import contextlib
import copy
from typing import Any, cast

import torch
from tqdm import tqdm  # type: ignore[import-untyped]

from storegate import logger, const
from storegate.task import DLTask
from storegate.task.dl_task import _CompiledVarNames
from storegate.task.pytorch import pytorch_util as util
from storegate.task.pytorch.storegate_dataset import StoreGateDataset
from storegate.task.pytorch.pytorch_metrics import EpochMetric, get_pbar_metric

from torch import optim
from torch.nn.modules import loss as tl
from torch.utils.data import DataLoader


class PytorchTask(DLTask):
    """DL task class for the default functions."""
    def __init__(self,
                 device: str = 'auto',
                 torch_compile: bool = False,
                 torchinfo: bool = False,
                 dataset_args: dict[str, Any] | None = None,
                 dataloader_args: dict[str, Any] | None = None,
                 **kwargs: Any):
        """Initialize the pytorch base task."""

        super().__init__(**kwargs)

        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._device: torch.device = torch.device(device)
        self._torch_compile: bool = torch_compile
        self._torchinfo: bool = torchinfo
        self._dataset_args: dict[str, Any] | None = dataset_args
        self._dataloader_args: dict[str, Any] | None = dataloader_args

        self._pbar_args: dict[str, Any] = const.PBAR_ARGS

        self._is_gpu: bool = 'cuda' in self._device.type
        self._cuda_id: int | None = None

        if self._metrics is None:
            self._metrics = ['loss']

    def _phase_var_names(self, attr_name: str, phase: str) -> list[str] | None:
        return self._get_var_names_for_phase(
            cast(_CompiledVarNames, getattr(self, attr_name)),
            phase,
        )

    @staticmethod
    def _tmp_test_output_var_names(
        output_var_names: list[str] | None,
    ) -> list[str] | None:
        if output_var_names is None:
            return None
        return [f'tmp_{var_name}' for var_name in output_var_names]

    @staticmethod
    def _backup_test_output_var_name(
        output_var_name: str,
        reserved_names: set[str],
    ) -> str:
        candidate = f'backup_{output_var_name}'
        suffix = 0
        while candidate in reserved_names:
            suffix += 1
            candidate = f'backup_{output_var_name}_{suffix}'
        reserved_names.add(candidate)
        return candidate

    def _delete_test_outputs(self, output_var_names: list[str] | None) -> bool:
        if output_var_names is None:
            return False
        existing_var_names = set(self._storegate.get_var_names('test'))
        deleted = False
        for var_name in output_var_names:
            if var_name in existing_var_names:
                self._storegate.delete_data(var_name, 'test')
                deleted = True
        return deleted

    def _promote_tmp_test_outputs(
        self,
        tmp_output_var_names: list[str],
        output_var_names: list[str],
    ) -> None:
        reserved_names = (
            set(self._storegate.get_var_names('test'))
            | set(tmp_output_var_names)
            | set(output_var_names)
        )
        backup_var_names: dict[str, str] = {}
        promoted_var_names: set[str] = set()

        try:
            for output_var_name in output_var_names:
                existing_var_names = set(self._storegate.get_var_names('test'))
                if output_var_name not in existing_var_names:
                    continue

                backup_var_name = self._backup_test_output_var_name(
                    output_var_name,
                    reserved_names,
                )
                self._storegate.rename_data(output_var_name, backup_var_name, 'test')
                backup_var_names[output_var_name] = backup_var_name

            for tmp_output_var_name, output_var_name in zip(
                tmp_output_var_names,
                output_var_names,
            ):
                self._storegate.rename_data(tmp_output_var_name, output_var_name, 'test')
                promoted_var_names.add(output_var_name)

        except Exception:
            existing_var_names = set(self._storegate.get_var_names('test'))

            for output_var_name in promoted_var_names:
                if output_var_name in existing_var_names:
                    self._storegate.delete_data(output_var_name, 'test')
                    existing_var_names.remove(output_var_name)

            for tmp_output_var_name in tmp_output_var_names:
                if tmp_output_var_name in existing_var_names:
                    self._storegate.delete_data(tmp_output_var_name, 'test')
                    existing_var_names.remove(tmp_output_var_name)

            for output_var_name, backup_var_name in backup_var_names.items():
                if backup_var_name in existing_var_names:
                    self._storegate.rename_data(backup_var_name, output_var_name, 'test')
                    existing_var_names.remove(backup_var_name)
                    existing_var_names.add(output_var_name)

            self._storegate.compile()
            raise

        existing_var_names = set(self._storegate.get_var_names('test'))
        for backup_var_name in backup_var_names.values():
            if backup_var_name in existing_var_names:
                self._storegate.delete_data(backup_var_name, 'test')
                existing_var_names.remove(backup_var_name)

        self._storegate.compile()

    def compile(self) -> None:
        """Compile pytorch ml objects."""
        self.compile_device()
        super().compile()


    def compile_device(self) -> None:
        """Compile device."""
        if self._cuda_id is not None:
            self._device = torch.device(f'cuda:{self._cuda_id}')
        self._is_gpu = 'cuda' in self._device.type

    def compile_model(self) -> None:
        """Compile model."""
        if self._model is None:
            raise ValueError(
                'model is required. Pass a class, instance, or string name via model=..., '
                'or override step_model() in a subclass.'
            )

        if self._is_gpu and (not torch.cuda.is_available()):
            raise ValueError(f'{self._device} is not available')

        self._ml.model = util.build_module(self._model, self._model_args, None)
        self._ml.model.to(self._device)

        if self._torchinfo:
            from torchinfo import summary
            summary(self._ml.model)

        if self._torch_compile:
            self._ml.model = torch.compile(self._ml.model)


    def compile_optimizer(self) -> None:
        """Compile optimizer."""
        if self._optimizer is None:
            return

        optimizer_args = copy.deepcopy(self._optimizer_args)
        optimizer_args['params'] = list(self._ml.model.parameters())
        self._ml.optimizer = util.build_module(self._optimizer, optimizer_args, optim)


    def compile_loss(self) -> None:
        """Compile loss."""
        if self._loss is None:
            raise ValueError(
                'loss is required. Pass a class, instance, or string name via loss=..., '
                'or override step_loss() in a subclass.'
            )

        self._ml.loss = util.build_module(self._loss, self._loss_args, tl)
        self._ml.loss = self._ml.loss.to(self._device)


    def get_dataloader(self, phase: str) -> DataLoader:  # type: ignore[type-arg]
        """Return dataloader."""
        dataset_kwargs = dict(self._dataset_args or {})
        dataset = StoreGateDataset(self._storegate,
                                   phase,
                                   input_var_names=self._phase_var_names('_input_var_names', phase),
                                   true_var_names=self._phase_var_names('_true_var_names', phase),
                                   **dataset_kwargs)

        dataloader_args: dict[str, Any] = {'dataset': dataset, **(self._dataloader_args or {})}

        # Prediction writes outputs back to StoreGate in dataloader order,
        # so the test phase must keep the original sample order.
        dataloader_args['shuffle'] = (
            False if phase == 'test'
            else dataloader_args.get('shuffle', phase == 'train')
        )

        return DataLoader(batch_size=self._batch_size,
                          **dataloader_args)

    def _phase_has_supervised_data(self, phase: str) -> bool:
        return bool(self._phase_var_names('_input_var_names', phase)) and bool(
            self._phase_var_names('_true_var_names', phase)
        )

    def fit(self) -> dict[str, Any]:
        """Train model over epoch."""
        phases = ['train', *(['valid'] if self._phase_has_supervised_data('valid') else [])]
        dataloaders: dict[str, DataLoader] = {phase: self.get_dataloader(phase) for phase in phases}  # type: ignore[type-arg]
        rtn_history: dict[str, list[dict[str, Any]]] = {phase: [] for phase in phases}

        for epoch in range(1, self._num_epochs + 1):
            for phase in phases:
                (self._ml.model.train if phase == 'train' else self._ml.model.eval)()
                rtn_history[phase].append(self.step_epoch(epoch, phase, dataloaders[phase]))

        return rtn_history


    def predict(self) -> dict[str, Any]:
        """Predict and upload outputs to storegate."""
        output_var_names = self._phase_var_names('_output_var_names', 'test')
        tmp_output_var_names = self._tmp_test_output_var_names(output_var_names)

        if self._delete_test_outputs(tmp_output_var_names):
            self._storegate.compile()

        if not self._storegate.get_var_names('test'):
            logger.warn("predict() skipped: no variables found in the 'test' phase.")
            return {'test': {}}

        self._ml.model.eval()
        dataloader = self.get_dataloader('test')

        if tmp_output_var_names is not None:
            self._active_test_output_var_names = tmp_output_var_names

        try:
            rtn_result = self.step_epoch(0, 'test', dataloader)
            if output_var_names is not None and tmp_output_var_names is not None:
                self._promote_tmp_test_outputs(tmp_output_var_names, output_var_names)
        except Exception:
            if self._delete_test_outputs(tmp_output_var_names):
                self._storegate.compile()
            raise
        finally:
            if hasattr(self, '_active_test_output_var_names'):
                del self._active_test_output_var_names

        return {'test': rtn_result}


    def step_epoch(self, epoch: int, phase: str, dataloader: DataLoader) -> dict[str, Any]:  # type: ignore[type-arg]
        """Process model for given epoch and phase."""
        metrics = self._metrics if self._metrics is not None else ['loss']
        epoch_metric = EpochMetric(metrics, self._ml)
        num_batches = len(dataloader)
        pbar_args: dict[str, Any] = dict(total=num_batches, disable=False)
        pbar_args.update(self._pbar_args)
        pbar_desc = self._get_pbar_description(epoch, phase)

        rtn_results: dict[str, Any] = {}
        with tqdm(**pbar_args) as pbar:
            pbar.set_description(pbar_desc)
            for ii, data in enumerate(dataloader):
                batch_result = self.step_batch(data, phase)

                if phase == 'test':
                    self._output_to_storegate(batch_result['outputs'])

                rtn_results.update(epoch_metric(batch_result))
                pbar_metrics = get_pbar_metric(rtn_results)
                pbar.set_postfix(pbar_metrics)
                pbar.update(1)

        return rtn_results


    def step_batch(self, data: Any, phase: str) -> dict[str, Any]:
        """Process batch data and update weights."""
        expects_labels = self._phase_expects_labels(phase)
        if expects_labels:
            if not isinstance(data, (tuple, list)) or len(data) != 2:
                raise ValueError(f'labels are required for phase={phase!r}.')
            inputs, labels = data
        else:
            inputs, labels = data, None

        if phase != 'test' and labels is None:
            raise ValueError(f'labels are required for phase={phase!r}.')

        inputs = self.add_device(inputs, self._device)
        if labels is not None:
            labels = self.add_device(labels, self._device)

        rtn_result: dict[str, Any] = {'batch_size': util.inputs_size(inputs)}

        ctx = contextlib.nullcontext() if phase == 'train' else torch.inference_mode()
        with ctx:
            outputs = self.step_model(inputs)

            if labels is not None:
                loss_result = self.step_loss(outputs, labels)
                rtn_result['loss'] = loss_result

                if phase == 'train':
                    self.step_optimizer(loss_result['loss'])

            if phase == 'test':
                rtn_result['pred'] = outputs

            rtn_result.update(dict(outputs=outputs, labels=labels))

        return rtn_result

    def _phase_expects_labels(self, phase: str) -> bool:
        if phase != 'test':
            return True

        true_var_names = getattr(self, '_true_var_names', None)
        if not isinstance(true_var_names, dict):
            return False

        return bool(
            self._get_var_names_for_phase(
                cast(_CompiledVarNames, true_var_names), phase
            )
        )


    def step_model(self, inputs: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        """Process model.

        Note:
            For multi-input data, StoreGate passes ``inputs`` as
            ``list[torch.Tensor]``. The default implementation forwards that list
            unchanged as ``model(inputs)``. Override this method if your model
            expects positional arguments such as ``model(x0, x1)`` instead.
        """
        outputs: torch.Tensor = self._ml.model(inputs)
        return outputs


    def step_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
        """Process loss function.

        Note:
            For multi-output or multi-label data, StoreGate may pass list-based
            containers through the execution path. The default implementation
            calls ``loss(outputs, labels)`` unchanged, so override this method or
            wrap the loss function if it needs positional arguments or per-head
            aggregation.
        """
        loss_result: dict[str, Any] = {'loss': 0, 'subloss': []}
        loss_result['loss'] += self._ml.loss(outputs, labels)
        return loss_result


    def step_optimizer(self, loss: torch.Tensor) -> None:
        """Process optimizer."""
        if self._ml.optimizer is None:
            raise ValueError(
                'optimizer is required for training. '
                'Pass optimizer=..., or override step_optimizer() in a subclass.'
            )
        self._ml.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._ml.optimizer.step()


    def add_device(
        self,
        data: torch.Tensor | list[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Add data to device."""
        if isinstance(data, list):
            return [idata.to(device) for idata in data]

        return data.to(device)


    ##########################################################################
    # Internal methods
    ##########################################################################
    def _get_pbar_description(self, epoch: int, phase: str) -> str:
        return f'Epoch [{epoch: >4}/{self._num_epochs}] {phase.ljust(5)}'


    def _output_to_storegate(self, outputs: torch.Tensor | list[torch.Tensor]) -> None:
        output_var_names = cast(
            list[str] | None,
            getattr(self, '_active_test_output_var_names', None),
        )
        if output_var_names is None:
            output_var_names = self._get_var_names_for_phase(
                cast(_CompiledVarNames, self._output_var_names), 'test'
            )
        if output_var_names is None:
            return

        if not isinstance(outputs, list):
            outputs = [outputs]

        if len(outputs) != len(output_var_names):
            raise ValueError(
                f'Number of model outputs ({len(outputs)}) does not match '
                f'output_var_names ({len(output_var_names)}).'
            )

        for output_var_name, output in zip(output_var_names, outputs):
            self._storegate.add_data(output_var_name, output.detach().cpu().numpy(), 'test')
