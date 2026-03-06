from __future__ import annotations

import contextlib
import copy
from typing import Any

import torch
import numpy as np
from tqdm import tqdm

from storegate import logger, const
from storegate.task import DLTask
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
            return

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
            return

        self._ml.loss = util.build_module(self._loss, self._loss_args, tl)
        self._ml.loss = self._ml.loss.to(self._device)


    def get_dataloader(self, phase: str) -> DataLoader:  # type: ignore[type-arg]
        """Return dataloader."""
        dataset_kwargs: dict[str, Any] = {}
        if self._dataset_args is not None:
            dataset_kwargs.update(self._dataset_args)

        dataset = StoreGateDataset(self._storegate,
                                   phase,
                                   input_var_names=self._input_var_names,
                                   true_var_names=self._true_var_names,
                                   **dataset_kwargs)

        dataloader_args: dict[str, Any] = dict(dataset=dataset)

        if self._dataloader_args is not None:
            dataloader_args.update(self._dataloader_args)

        dataloader_args.setdefault('shuffle', phase == 'train')

        return DataLoader(batch_size=self._batch_size,
                          **dataloader_args)


    def fit(self) -> dict[str, Any]:
        """Train model over epoch."""
        has_valid = bool(self._storegate.get_var_names('valid'))

        dataloaders: dict[str, DataLoader] = {'train': self.get_dataloader('train')}  # type: ignore[type-arg]
        if has_valid:
            dataloaders['valid'] = self.get_dataloader('valid')

        rtn_history: dict[str, list[dict[str, Any]]] = {'train': []}
        if has_valid:
            rtn_history['valid'] = []

        for epoch in range(1, self._num_epochs + 1):
            # train
            self._ml.model.train()
            rtn_result = self.step_epoch(epoch, 'train', dataloaders['train'])
            rtn_history['train'].append(rtn_result)

            # valid
            if has_valid:
                self._ml.model.eval()
                rtn_result = self.step_epoch(epoch, 'valid', dataloaders['valid'])
                rtn_history['valid'].append(rtn_result)

        return rtn_history


    def predict(self) -> dict[str, Any]:
        """Predict and upload outputs to storegate."""
        deleted = False
        if self._output_var_names is not None:
            for var_name in self._output_var_names:
                if var_name in self._storegate.get_var_names('test'):
                    self._storegate.delete_data(var_name, 'test')
                    deleted = True
        if deleted:
            self._storegate.compile()

        self._ml.model.eval()
        dataloader = self.get_dataloader('test')

        rtn_result = self.step_epoch(0, 'test', dataloader)

        return {'test': rtn_result}


    def step_epoch(self, epoch: int, phase: str, dataloader: DataLoader) -> dict[str, Any]:  # type: ignore[type-arg]
        """Process model for given epoch and phase."""
        epoch_metric = EpochMetric(self._metrics, self._ml)
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
        inputs, labels = data
        inputs = self.add_device(inputs, self._device)
        labels = self.add_device(labels, self._device)

        rtn_result: dict[str, Any] = {'batch_size': util.inputs_size(inputs)}

        ctx = contextlib.nullcontext() if phase == 'train' else torch.inference_mode()
        with ctx:
            outputs = self.step_model(inputs)

            loss_result = self.step_loss(outputs, labels)

            if phase == 'train':
                self.step_optimizer(loss_result['loss'])

            elif phase == 'test':
                rtn_result['pred'] = outputs

            rtn_result.update(dict(outputs=outputs, labels=labels, loss=loss_result))

        return rtn_result


    def step_model(self, inputs: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        """Process model."""
        outputs: torch.Tensor = self._ml.model(inputs)
        return outputs


    def step_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
        """Process loss function."""
        loss_result: dict[str, Any] = {'loss': 0, 'subloss': []}
        loss_result['loss'] += self._ml.loss(outputs, labels)
        return loss_result


    def step_optimizer(self, loss: torch.Tensor) -> None:
        """Process optimizer."""
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
            return [self.add_device(idata, device) for idata in data]  # type: ignore[return-value]

        return data.to(device)  # type: ignore[union-attr]


    ##########################################################################
    # Internal methods
    ##########################################################################
    def _get_pbar_description(self, epoch: int, phase: str) -> str:
        return f'Epoch [{epoch: >4}/{self._num_epochs}] {phase.ljust(5)}'


    def _output_to_storegate(self, outputs: torch.Tensor | list[torch.Tensor]) -> None:
        if self._output_var_names is None:
            return

        if not isinstance(outputs, list):
            outputs = [outputs]

        if len(outputs) != len(self._output_var_names):
            raise ValueError(
                f'Number of model outputs ({len(outputs)}) does not match '
                f'output_var_names ({len(self._output_var_names)}).'
            )

        for output_var_name, output in zip(self._output_var_names, outputs):
            self._storegate.add_data(output_var_name, output.detach().cpu().numpy(), 'test')
