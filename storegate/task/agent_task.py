from __future__ import annotations

from typing import TYPE_CHECKING, Any

from storegate import logger
from storegate.task import Task

if TYPE_CHECKING:
    from storegate.storegate import StoreGate


class AgentTask(Task):
    """Agent task class for the default functions."""
    _PROTECTED_KEYS: frozenset[str] = frozenset()

    def __init__(self, storegate: StoreGate) -> None:
        self._storegate = storegate
        self._data_id: str | None = None

    def set_hps(self, params: dict[str, Any]) -> None:
        """Set hyperparameters to this task."""
        for key, value in params.items():
            if key in self._PROTECTED_KEYS:
                raise AttributeError(f'{key} is not a valid hyperparameter.')
            if not hasattr(self, '_' + key):
                raise AttributeError(f'{key} is not defined.')

            setattr(self, '_' + key, value)

        if self._data_id is not None:
            self._storegate.set_data_id(self._data_id)

    def execute(self) -> Any:
        """Execute base task.

        Users implement their algorithms.
        """

    def finalize(self) -> None:
        """Finalize base task.

        Users implement their algorithms.
        """

    @property
    def storegate(self) -> StoreGate:
        """Return storegate of task."""
        return self._storegate

    @storegate.setter
    def storegate(self, storegate: StoreGate) -> None:
        """Set storegate."""
        self._storegate = storegate
