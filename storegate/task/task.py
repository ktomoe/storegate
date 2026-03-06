from abc import ABCMeta, abstractmethod
from typing import Any


class Task(metaclass=ABCMeta):
    """Tasks need be inherited this base class.

    Multi-ai agents assume that initialize, execute, finalize, set_hps, methods are available.
    """
    @abstractmethod
    def execute(self) -> Any:
        """Execute the task."""

    @abstractmethod
    def finalize(self) -> None:
        """Finalize the task."""

    @abstractmethod
    def set_hps(self, params: dict[str, Any]) -> None:
        """Set hyperparameters of this task."""
