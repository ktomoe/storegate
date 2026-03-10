import copy
from abc import abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, cast

from storegate import const
from storegate.task.agent_task import AgentTask
from storegate.task.dl_env import DLEnv

type _SinglePhaseVarNames = str | list[str] | None
type _PhaseVarNames = Mapping[str, _SinglePhaseVarNames]
type _VarNames = _SinglePhaseVarNames | _PhaseVarNames
type _ResolvedVarNames = list[str] | None
type _CompiledVarNames = dict[str, _ResolvedVarNames]


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
                 metrics: list[str | Callable[..., Any]] | None = None,
                 num_epochs: int = 10,
                 batch_size: int = 64,
                 execute_in_memory: bool = False,
                 **kwargs: Any):
        """Initialize a DL task.

        Args:
            execute_in_memory (bool): If ``True``, required input and label
                variables are copied from zarr into the numpy backend before
                execution, and ``fit()`` / ``predict()`` run with the numpy
                backend active.
                Default: ``False``.
        """

        super().__init__(**kwargs)

        if model_args is None:
            model_args = {}
        else:
            model_args = copy.deepcopy(model_args)

        if optimizer_args is None:
            optimizer_args = {}
        else:
            optimizer_args = copy.deepcopy(optimizer_args)

        if loss_args is None:
            loss_args = {}
        else:
            loss_args = copy.deepcopy(loss_args)

        self._ml = DLEnv()

        self._input_var_names: _VarNames = copy.deepcopy(input_var_names)
        self._output_var_names: _VarNames = copy.deepcopy(output_var_names)
        self._true_var_names: _VarNames = copy.deepcopy(true_var_names)

        self._model: Any = model
        self._model_args: dict[str, Any] = model_args
        self._optimizer: Any = optimizer
        self._optimizer_args: dict[str, Any] = optimizer_args
        self._loss: Any = loss
        self._loss_args: dict[str, Any] = loss_args
        self._metrics: list[str | Callable[..., Any]] | None = copy.deepcopy(metrics)

        self._num_epochs: int = num_epochs
        self._batch_size: int = batch_size
        self._execute_in_memory: bool = execute_in_memory


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
            When ``execute_in_memory=True``, model predictions written via
            ``output_var_names`` are stored in the numpy (memory) backend.
            To persist them to disk, call
            ``storegate.copy_to_storage(var_name, phase='test')``
            after ``execute()`` returns.

            Required input/label variables are validated against the zarr
            backend before the zarr -> numpy copy starts, so this mode does
            not depend on the caller's currently selected backend.
        """

        if self._execute_in_memory:
            with self._storegate.using_backend('zarr'):
                self.compile()

            for phase in const.PHASES:
                phase_var_names = (
                    (
                        self._get_var_names_for_phase(
                            cast(_CompiledVarNames, self._input_var_names), phase
                        ) or []
                    )
                    + (
                        self._get_var_names_for_phase(
                            cast(_CompiledVarNames, self._true_var_names), phase
                        ) or []
                    )
                )
                for var_name in phase_var_names:
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
            self.compile()
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
        self._input_var_names = self._compile_var_name_groups(self._input_var_names)
        self._output_var_names = self._compile_var_name_groups(self._output_var_names)
        self._true_var_names = self._compile_var_name_groups(self._true_var_names)
        self._validate_required_var_names_exist()
        self._validate_output_var_names_do_not_overlap()

    @abstractmethod
    def compile_model(self) -> None:
        """Compile model.

        Concrete subclasses must prepare the model here.
        """
        raise NotImplementedError

    def compile_optimizer(self) -> None:
        """Compile optimizer."""
        pass

    @abstractmethod
    def compile_loss(self) -> None:
        """Compile loss.

        Concrete subclasses must prepare the loss here.
        """
        raise NotImplementedError

    def _compile_var_name_groups(
        self,
        var_names: _VarNames,
    ) -> _CompiledVarNames:
        if isinstance(var_names, Mapping):
            self._validate_phase_var_names(var_names)
            return {
                phase: self._compile_single_phase_var_names(var_names.get(phase))
                for phase in const.PHASES
            }
        compiled = self._compile_single_phase_var_names(var_names)
        return {phase: compiled for phase in const.PHASES}

    def _compile_single_phase_var_names(
        self,
        var_names: _SinglePhaseVarNames,
    ) -> _ResolvedVarNames:
        if isinstance(var_names, str):
            return [var_names]
        return var_names

    def _get_var_names_for_phase(
        self,
        var_names: _CompiledVarNames,
        phase: str,
    ) -> _ResolvedVarNames:
        if phase not in const.PHASES:
            raise ValueError(f'phase must be one of {const.PHASES}, got {phase!r}.')
        return var_names[phase]

    def _validate_phase_var_names(self, var_names: _PhaseVarNames) -> None:
        invalid_phases = sorted(set(var_names) - set(const.PHASES))
        if invalid_phases:
            raise ValueError(
                f'var_names dict contains invalid phases {invalid_phases}. '
                f'Expected only {const.PHASES}.'
            )

    def _validate_required_var_names_exist(self) -> None:
        errors: list[str] = []
        for label, var_names in (
            ('input', cast(_CompiledVarNames, self._input_var_names)),
            ('true', cast(_CompiledVarNames, self._true_var_names)),
        ):
            errors.extend(self._missing_required_var_name_errors(label, var_names))

        if errors:
            raise ValueError(
                'Required StoreGate variables are missing:\n' + '\n'.join(errors)
            )

    def _missing_required_var_name_errors(
        self,
        label: str,
        var_names: _CompiledVarNames,
    ) -> list[str]:
        errors: list[str] = []
        for phase in const.PHASES:
            phase_var_names = var_names[phase]
            if phase_var_names is None:
                continue

            existing_var_names = set(self._storegate.get_var_names(phase))
            missing_var_names = [
                var_name for var_name in phase_var_names
                if var_name not in existing_var_names
            ]
            if missing_var_names:
                errors.append(
                    f"  phase='{phase}' missing {label}_var_names={missing_var_names}"
                )

        return errors

    @staticmethod
    def _duplicate_var_names(var_names: list[str]) -> list[str]:
        seen: set[str] = set()
        duplicates: list[str] = []
        for var_name in var_names:
            if var_name in seen and var_name not in duplicates:
                duplicates.append(var_name)
            seen.add(var_name)
        return duplicates

    def _validate_output_var_names_do_not_overlap(self) -> None:
        errors: list[str] = []

        for phase in const.PHASES:
            input_var_names = cast(_CompiledVarNames, self._input_var_names)[phase] or []
            true_var_names = cast(_CompiledVarNames, self._true_var_names)[phase] or []
            output_var_names = cast(_CompiledVarNames, self._output_var_names)[phase]

            if output_var_names is None:
                continue

            duplicate_output_var_names = self._duplicate_var_names(output_var_names)
            input_overlap = sorted(set(output_var_names) & set(input_var_names))
            true_overlap = sorted(set(output_var_names) & set(true_var_names))

            if duplicate_output_var_names:
                errors.append(
                    f"  phase='{phase}' output_var_names contain duplicates={duplicate_output_var_names}"
                )
            if input_overlap:
                errors.append(
                    f"  phase='{phase}' output_var_names overlap with input_var_names={input_overlap}"
                )
            if true_overlap:
                errors.append(
                    f"  phase='{phase}' output_var_names overlap with true_var_names={true_overlap}"
                )

        if errors:
            raise ValueError(
                'output_var_names must be unique within each phase and must not '
                'overlap with input_var_names or true_var_names:\n'
                + '\n'.join(errors)
            )
