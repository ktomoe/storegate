from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Generic, Iterator, TypeVar

from storegate.task.pytorch_metrics import ResolvedMetric

PhaseValueT = TypeVar("PhaseValueT")
MetricT = TypeVar("MetricT")


class LearningMode(StrEnum):
    SUPERVISED = "supervised"
    SELF_SUPERVISED = "self_supervised"
    UNSUPERVISED = "unsupervised"


@dataclass
class _PhaseSlots(Generic[PhaseValueT]):
    """Common train/valid/test slot container."""
    train: PhaseValueT | None = None
    valid: PhaseValueT | None = None
    test: PhaseValueT | None = None

    def iter_active_items(self) -> Iterator[tuple[str, PhaseValueT]]:
        """Yield non-None train/valid/test items."""
        for phase_name in ("train", "valid", "test"):
            value = getattr(self, phase_name)
            if value is not None:
                yield phase_name, value


@dataclass
class _PhaseLists(Generic[MetricT]):
    """Common train/valid/test list container."""
    train: list[MetricT] = field(default_factory=list)
    valid: list[MetricT] = field(default_factory=list)
    test: list[MetricT] = field(default_factory=list)


@dataclass
class Phase(_PhaseSlots[Any]):
    """Data class to store phase objects."""
    pass


@dataclass
class VarName:
    """Data class to store variable name objects."""
    inputs: Any = None
    outputs: Any = None
    true:    Any = None


@dataclass
class DataSet(_PhaseSlots[Any]):
    """Data class to store phase objects."""
    pass


@dataclass
class DataLoader(_PhaseSlots[Any]):
    """Data class to store phase objects."""
    pass


@dataclass
class Metrics(_PhaseLists[ResolvedMetric]):
    """Data class to store phase-specific resolved metrics."""
    pass


@dataclass
class DLEnv:
    """Data class to store compiled DL objects."""
    phase:      Phase = field(default_factory=Phase)
    var_name:   VarName = field(default_factory=VarName)
    learning_mode: LearningMode = LearningMode.SUPERVISED
    model:      Any = None
    optimizer:  Any = None
    loss:       Any = None
    device:     Any = None
    dataset:    DataSet = field(default_factory=DataSet)
    dataloader: DataLoader = field(default_factory=DataLoader)
    metrics:    Metrics = field(default_factory=Metrics)
    is_compiled: bool = False

    def clear(self) -> None:
        self.phase = Phase()
        self.var_name = VarName()
        self.learning_mode = LearningMode.SUPERVISED
        self.model = None
        self.optimizer = None
        self.loss = None
        self.device = None
        self.dataset = DataSet()
        self.dataloader = DataLoader()
        self.metrics = Metrics()
        self.is_compiled = False

    def require_compiled(self, action: str) -> None:
        if not self.is_compiled:
            raise RuntimeError(f"Call compile() before {action}().")

    def require_fit_ready(self) -> None:
        required_fields: list[tuple[str, Any]] = [
            ("model", self.model),
            ("optimizer", self.optimizer),
            ("loss", self.loss),
            ("device", self.device),
            ("phase.train", self.phase.train),
            ("dataloader.train", self.dataloader.train),
        ]

        if self.learning_mode in (
            LearningMode.SUPERVISED,
            LearningMode.SELF_SUPERVISED,
        ):
            required_fields.append(("var_name.true", self.var_name.true))
        elif self.learning_mode == LearningMode.UNSUPERVISED:
            pass
        else:
            raise ValueError(f"Unsupported learning_mode: {self.learning_mode!r}")

        self._require_ready("fit", tuple(required_fields))

    def require_predict_ready(self) -> None:
        required_fields: list[tuple[str, Any]] = [
            ("model", self.model),
            ("device", self.device),
            ("phase.test", self.phase.test),
            ("dataloader.test", self.dataloader.test),
        ]

        if self.learning_mode in (
            LearningMode.SUPERVISED,
            LearningMode.SELF_SUPERVISED,
            LearningMode.UNSUPERVISED,
        ):
            pass
        else:
            raise ValueError(f"Unsupported learning_mode: {self.learning_mode!r}")

        self._require_ready("predict", tuple(required_fields))

    def _require_ready(
        self,
        action: str,
        required_fields: tuple[tuple[str, Any], ...],
    ) -> None:
        missing = [
            field_name
            for field_name, value in required_fields
            if value is None
        ]
        if missing:
            raise ValueError(f"{action}() requires: {', '.join(missing)}")
