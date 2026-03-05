from __future__ import annotations

from typing import TYPE_CHECKING, Any

from storegate import logger
from storegate.task import Task

if TYPE_CHECKING:
    from storegate.storegate import StoreGate


class AgentTask(Task):
    """Agent task class for the default functions."""
    _PROTECTED_KEYS: frozenset[str] = frozenset({'storegate', 'ml'})

    def __init__(self, storegate: StoreGate) -> None:
        self._storegate = storegate
        self._data_id: str | None = None

    def set_hps(self, params: dict[str, Any]) -> None:
        """Set hyperparameters to this task.

        Each key maps to a task attribute ``_<key>``. For example, ``'data_id'``
        updates ``_data_id`` and automatically calls ``storegate.set_data_id()``,
        allowing each hyperparameter trial to operate on a different dataset.

        Protected keys (``storegate``, ``ml``) and undefined attributes raise
        ``AttributeError``.
        """
        for key, value in params.items():
            if key in self._PROTECTED_KEYS:
                raise AttributeError(f'{key} is not a valid hyperparameter.')
            if '_' + key not in self.__dict__:
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
