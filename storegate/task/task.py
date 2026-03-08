from abc import ABCMeta, abstractmethod
from typing import Any


class Task(metaclass=ABCMeta):
    """Abstract base class for all tasks.

    Search agents assume that execute, finalize, and set_hps methods are available.
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
