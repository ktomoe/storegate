from __future__ import annotations

from typing import Any

from storegate import logger, const
from storegate.task.agent_task import AgentTask
from storegate.task.dl_env import DLEnv

_VarNames = str | list[str] | None


class DLTask(AgentTask):
    """DL task class for the default functions."""
    def __init__(self,
                 input_var_names: _VarNames = None,
                 output_var_names: _VarNames = None,
                 true_var_names: _VarNames = None,
                 model: Any = None,
                 model_args: dict[str, Any] | None = None,
                 optimizer: Any = None,
                 optimizer_args: dict[str, Any] | None = None,
                 loss: Any = None,
                 loss_args: dict[str, Any] | None = None,
                 metrics: list[Any] | None = None,
                 num_epochs: int = 10,
                 batch_size: int = 64,
                 preload: bool = False,
                 **kwargs: Any):
        """Initialize a DL task.

        Args:
            preload (bool): If ``True``, input and label variables are copied
                from the zarr store into memory before training begins.
                This avoids per-sample zarr reads during each epoch and is
                strongly recommended when the dataset fits in RAM.
                Default: ``False``.
        """

        super().__init__(**kwargs)

        if model_args is None:
            model_args = {}

        if optimizer_args is None:
            optimizer_args = {}

        if loss_args is None:
            loss_args = {}

        self._ml = DLEnv()

        self._input_var_names: _VarNames = input_var_names
        self._output_var_names: _VarNames = output_var_names
        self._true_var_names: _VarNames = true_var_names

        self._model: Any = model
        self._model_args: dict[str, Any] = model_args
        self._optimizer: Any = optimizer
        self._optimizer_args: dict[str, Any] = optimizer_args
        self._loss: Any = loss
        self._loss_args: dict[str, Any] = loss_args
        self._metrics: list[Any] | None = metrics

        self._num_epochs: int = num_epochs
        self._batch_size: int = batch_size
        self._preload: bool = preload


    def set_hps(self, params: dict[str, Any]) -> None:
        """Set hyperparameters to this task."""
        for key, value in params.items():

            if key.startswith('model__'):
                suffix = key.removeprefix('model__')
                if not suffix:
                    raise ValueError("Hyperparameter key 'model__' has an empty suffix. Use 'model__<arg_name>'.")
                self._model_args[suffix] = value

            elif key.startswith('optimizer__'):
                suffix = key.removeprefix('optimizer__')
                if not suffix:
                    raise ValueError("Hyperparameter key 'optimizer__' has an empty suffix. Use 'optimizer__<arg_name>'.")
                self._optimizer_args[suffix] = value

            elif key.startswith('loss__'):
                suffix = key.removeprefix('loss__')
                if not suffix:
                    raise ValueError("Hyperparameter key 'loss__' has an empty suffix. Use 'loss__<arg_name>'.")
                self._loss_args[suffix] = value

            else:
                if key in self._PROTECTED_KEYS:
                    raise AttributeError(f'{key} is not a valid hyperparameter.')
                if '_' + key not in self.__dict__:
                    raise AttributeError(f'{key} is not defined.')

                setattr(self, '_' + key, value)

        if self._data_id is not None:
            self._storegate.set_data_id(self._data_id)

    def execute(self) -> dict[str, Any]:
        """Execute a task.

        Note:
            When ``preload=True``, model predictions written via
            ``output_var_names`` are stored in the numpy (memory) backend.
            To persist them to disk, call
            ``storegate.copy_to_storage(var_name, phase='test')``
            after ``execute()`` returns.
        """

        self.compile()

        if self._preload:
            for phase in const.PHASES:
                for var_name in (self._input_var_names or []) + (self._true_var_names or []):
                    with self._storegate.using_backend('zarr'):
                        zarr_vars = self._storegate.get_var_names(phase)
                    if var_name not in zarr_vars:
                        continue
                    with self._storegate.using_backend('numpy'):
                        if var_name in self._storegate.get_var_names(phase):
                            self._storegate.delete_data(var_name, phase=phase)
                    self._storegate.copy_to_memory(var_name, phase=phase)

            with self._storegate.using_backend('numpy'):
                self._storegate.compile()
                rtn_fit = self.fit()
                rtn_predict = self.predict()
        else:
            rtn_fit = self.fit()
            rtn_predict = self.predict()

        return rtn_fit | rtn_predict


    def fit(self) -> dict[str, Any]:
        """Fit model."""
        return {}


    def predict(self) -> dict[str, Any]:
        """Predict model."""
        return {}


    def compile(self) -> None:
        """Compile model, optimizer and loss."""
        self._ml.clear()

        self.compile_var_names()
        self.compile_model()
        self.compile_loss()
        self.compile_optimizer()

        self.storegate.compile()

    def compile_var_names(self) -> None:
        """Compile variable names."""
        if isinstance(self._input_var_names, str):
            self._input_var_names = [self._input_var_names]

        if isinstance(self._output_var_names, str):
            self._output_var_names = [self._output_var_names]

        if isinstance(self._true_var_names, str):
            self._true_var_names = [self._true_var_names]

    def compile_model(self) -> None:
        """Compile model."""
        pass

    def compile_optimizer(self) -> None:
        """Compile optimizer."""
        pass

    def compile_loss(self) -> None:
        """Compile loss."""
        pass
