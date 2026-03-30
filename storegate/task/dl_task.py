from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from storegate.task.task import Task
from storegate.task.dl_env import DLEnv, Metrics
from storegate.task.pytorch_metrics import (
    BUILTIN_METRICS,
    MetricFn,
    ResolvedMetric,
    make_key_lookup,
)

if TYPE_CHECKING:
    from storegate.storegate import StoreGate


class DLTask(Task):
    """DL task class for the default functions."""
    _PROTECTED_KEYS: frozenset[str] = frozenset({'storegate', 'dl'})
    _PHASE_NAMES: tuple[str, str, str] = ("train", "valid", "test")
    _PHASE_KEYS: frozenset[str] = frozenset(_PHASE_NAMES)

    def __init__(self, data_id: str | None = None,
                       phases: str | dict[str, str] | None = None,
                       var_names: dict[str, str | list[str]] | None = None,
                       model: Any = None,
                       model_args: dict[str, Any] | None = None,
                       optimizer: Any = None,
                       optimizer_args: dict[str, Any] | None = None,
                       loss: Any = None,
                       loss_args: dict[str, Any] | None = None,
                       num_epochs: int = 0,
                       metrics: list[str | MetricFn]
                       | dict[str, list[str | MetricFn]]
                       | None = None,
                       **kwargs: Any) -> None:
        """Initialize a DL task."""
        super().__init__(**kwargs)

        if model_args is None:
            model_args = {}

        if optimizer_args is None:
            optimizer_args = {}

        if loss_args is None:
            loss_args = {}

        self._dl = DLEnv()

        self._data_id = data_id
        self._phases = phases
        self._var_names = var_names
        self._model = model
        self._model_args = model_args
        self._optimizer = optimizer
        self._optimizer_args = optimizer_args
        self._loss = loss
        self._loss_args = loss_args
        self._num_epochs = num_epochs
        self._metrics = metrics

    @property
    def storegate(self) -> "StoreGate | None":
        return self._storegate

    @storegate.setter
    def storegate(self, storegate: "StoreGate | None") -> None:
        self._storegate = storegate
        self._dl.is_compiled = False

    def set_hps(self, params: dict[str, Any]) -> None:
        """Set hyperparameters to this task."""
        if self._storegate is None:
            raise RuntimeError("storegate is not set. Assign a StoreGate instance before calling set_hps().")

        namespace_targets = self._get_hps_namespace_targets()

        for key, value in params.items():
            namespace, sep, suffix = key.partition('__')

            if sep:
                target = namespace_targets.get(namespace)
                if target is None:
                    supported = ", ".join(
                        sorted(f"{name}__" for name in namespace_targets)
                    )
                    raise ValueError(
                        f"Unknown hyperparameter namespace '{namespace}__'. "
                        f"Supported namespaces are: {supported}."
                    )
                if not suffix:
                    raise ValueError(
                        f"Hyperparameter key '{namespace}__' has an empty suffix. "
                        f"Use '{namespace}__<arg_name>'."
                    )
                target[suffix] = value
                continue

            if key in self._PROTECTED_KEYS:
                raise AttributeError(f'{key} is not a valid hyperparameter.')
            if '_' + key not in self.__dict__:
                raise AttributeError(f'{key} is not defined.')

            setattr(self, '_' + key, value)

        self._dl.is_compiled = False

    def execute(self) -> dict[str, Any]:
        self.compile()

        rtn_fit = None
        rtn_predict = None

        if self._dl.phase.train is not None:
            rtn_fit = self.fit()

        if self._dl.phase.test is not None:
            rtn_predict = self.predict()

        return {'fit': rtn_fit, 'predict': rtn_predict}

    def reset(self) -> None:
        self._dl.clear()

    def compile(self) -> None:
        if self._storegate is None:
            raise RuntimeError("storegate is not set. Assign a StoreGate instance before calling compile().")

        self._storegate.initialize()
        self._bind_data_id()
        self._dl.clear()
        self._storegate.compile()
        self._compile_device()
        self._compile_phases()
        self._compile_var_names()
        self._compile_model()
        self._compile_optimizer()
        self._compile_loss()
        self._compile_metrics()
        self._compile_after_base()
        self._dl.is_compiled = True

    def _compile_phases(self) -> None:
        if self._phases is None:
            return

        if isinstance(self._phases, str):
            self._dl.phase.train = self._phases
            self._dl.phase.valid = self._phases
            self._dl.phase.test  = self._phases

        elif isinstance(self._phases, dict):
            _invalid_keys = self._phases.keys() - self._PHASE_KEYS
            if _invalid_keys:
                raise ValueError(
                    f"Invalid phase keys: {_invalid_keys}. "
                    f"Allowed keys are {self._PHASE_KEYS}."
                )

            self._dl.phase.train = self._phases.get('train')
            self._dl.phase.valid = self._phases.get('valid')
            self._dl.phase.test  = self._phases.get('test')

        else:
            raise TypeError(f"phases must be a str or dict, got {type(self._phases).__name__}")

    def _compile_var_names(self) -> None:
        if self._var_names is None:
            return

        if not isinstance(self._var_names, dict):
            raise TypeError(f"var_names must be a dict, got {type(self._var_names).__name__}")

        _valid_keys = {'inputs', 'outputs', 'true'}
        _invalid_keys = self._var_names.keys() - _valid_keys
        if _invalid_keys:
            raise ValueError(f"Invalid var_name keys: {_invalid_keys}. Allowed keys are {_valid_keys}.")

        self._dl.var_name.inputs  = self._normalize_var_name(self._var_names.get('inputs'))
        self._dl.var_name.outputs = self._normalize_var_name(self._var_names.get('outputs'))
        self._dl.var_name.true    = self._normalize_var_name(self._var_names.get('true'))

    def _compile_metrics(self) -> None:
        """Resolve raw metric specifications into ``ResolvedMetric`` objects.

        String names are resolved to either a built-in metric function
        (e.g. ``'acc'``) or a key-lookup function that reads the value
        directly from ``batch_result``.  Callables are wrapped as-is.
        """
        if self._metrics is None:
            return

        if isinstance(self._metrics, list):
            resolved = self._resolve_metrics(self._metrics)
            self._dl.metrics = Metrics(
                train=list(resolved),
                valid=list(resolved),
                test=list(resolved),
            )
        elif isinstance(self._metrics, dict):
            invalid_keys = self._metrics.keys() - self._PHASE_KEYS
            if invalid_keys:
                raise ValueError(
                    f"Invalid metric keys: {invalid_keys}. "
                    f"Allowed keys are {self._PHASE_KEYS}."
                )
            self._dl.metrics = Metrics(
                train=self._resolve_metrics(self._metrics.get("train", []), phase="train"),
                valid=self._resolve_metrics(self._metrics.get("valid", []), phase="valid"),
                test=self._resolve_metrics(self._metrics.get("test", []), phase="test"),
            )
        else:
            raise TypeError(
                "metrics must be a list, dict, or None, "
                f"got {type(self._metrics).__name__}"
            )

    @abstractmethod
    def _compile_device(self) -> None:
        """Compile device."""

    @abstractmethod
    def _compile_model(self) -> None:
        """Compile model."""

    @abstractmethod
    def _compile_optimizer(self) -> None:
        """Compile optimizer."""

    @abstractmethod
    def _compile_loss(self) -> None:
        """Compile loss."""

    @abstractmethod
    def fit(self) -> dict[str, Any]:
        """Fit model."""

    @abstractmethod
    def predict(self) -> dict[str, Any]:
        """Predict model."""

    def _on_batch_end(self, info: dict[str, Any]) -> None:
        """Called at the end of each batch with progress info.

        If ``_progress_callback`` is set (e.g. by a SearchAgent),
        the info dict is forwarded to it.
        """
        if self._progress_callback is not None:
            self._progress_callback(info)

    def _compile_after_base(self) -> None:
        """Optional subclass hook called at the end of ``compile()``."""

    def _bind_data_id(self) -> None:
        if self._data_id is not None:
            self._resolve_storegate().set_data_id(self._data_id)

    def _require_compiled(self, action: str) -> None:
        self._dl.require_compiled(action)

    def _resolve_storegate(self) -> "StoreGate":
        if self._storegate is None:
            raise RuntimeError(
                "storegate is not set. "
                "Assign a StoreGate instance before this operation."
            )
        return self._storegate

    ##########################################################################################
    def _get_hps_namespace_targets(self) -> dict[str, dict[str, Any]]:
        """Return namespace-to-target mappings for namespaced hyperparameters."""
        return {
            'model': self._model_args,
            'optimizer': self._optimizer_args,
            'loss': self._loss_args,
        }

    def _resolve_metrics(
        self,
        metrics: list[str | MetricFn],
        *,
        phase: str | None = None,
    ) -> list[ResolvedMetric]:
        if not isinstance(metrics, list):
            if phase is None:
                raise TypeError(f"metrics must be a list, got {type(metrics).__name__}")
            raise TypeError(f"metrics['{phase}'] must be a list, got {type(metrics).__name__}")

        resolved: list[ResolvedMetric] = []
        for metric in metrics:
            if callable(metric) and not isinstance(metric, str):
                name = getattr(metric, '__name__', str(metric))
                resolved.append(ResolvedMetric(name=name, fn=metric))
            elif isinstance(metric, str):
                if metric in BUILTIN_METRICS:
                    resolved.append(ResolvedMetric(name=metric,
                                                   fn=BUILTIN_METRICS[metric],
                                                   requires_labels=metric in {"acc", "binary_acc", "multi_acc"},))
                else:
                    resolved.append(ResolvedMetric(name=metric, fn=make_key_lookup(metric)))
            else:
                raise TypeError(
                    f"metric must be a str or callable, got {type(metric).__name__}"
                )
        return resolved

    @staticmethod
    def _normalize_var_name(
        value: str | list[str] | None,
    ) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            if not all(isinstance(name, str) for name in value):
                raise TypeError("var_name list elements must be str")
            if len(value) != len(set(value)):
                raise ValueError("var_name list contains duplicate entries.")
            return value
        raise TypeError(f"var_name value must be a str, list, or None, got {type(value).__name__}")
