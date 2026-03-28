from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from typing import Any

from storegate.storegate import StoreGate


class Task(metaclass=ABCMeta):
    """Abstract base class for all tasks.
    """
    def __init__(self, storegate: StoreGate | None = None) -> None:
        self._storegate = storegate
        self._progress_callback: Callable[[dict[str, Any]], None] | None = None

    @abstractmethod
    def execute(self) -> Any:
        """Execute the task."""

    def reset(self) -> None:
        """Release per-run state.

        The default implementation is a no-op so generic tasks do not need
        to implement cleanup unless they keep mutable run-local state.
        """

    @property
    def storegate(self) -> StoreGate | None:
        """Return storegate of task."""
        return self._storegate

    @storegate.setter
    def storegate(self, storegate: StoreGate) -> None:
        """Set storegate."""
        self._storegate = storegate
