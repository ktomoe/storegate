"""Module to define Database abstraction."""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any

import numpy as np


class Database(metaclass=ABCMeta):
    """Base class of Database."""

    @staticmethod
    def _normalize_index(index: int | slice | None) -> int | slice:
        """Convert None to a full-range slice."""
        return slice(0, None) if index is None else index

    @abstractmethod
    def initialize(self, data_id: str) -> None:
        """Initialize database, and set data_id."""

    @abstractmethod
    def add_data(self, data_id: str, var_name: str, data: np.ndarray, phase: str) -> None:
        """Add data to database for given data_id, var_name and phase.

        If var_name already exists, data need to be appended
        """

    @abstractmethod
    def update_data(self, data_id: str, var_name: str, data: np.ndarray, phase: str, index: int | slice | None) -> None:
        """Update (replace) data in database for given data_id, var_name, phase and index."""

    @abstractmethod
    def get_data(self, data_id: str, var_name: str, phase: str, index: int | slice | None) -> np.ndarray:
        """Get data for given data_id, var_name, phase and index from database."""

    @abstractmethod
    def delete_data(self, data_id: str, var_name: str, phase: str) -> None:
        """Delete data for given data_id, var_name and phase from database."""

    @abstractmethod
    def get_metadata(self, data_id: str, phase: str) -> dict[str, Any]:
        """Returns a dictionary of metadata for a given data_id and phase.

        The dict contains: {'var_name': {'type': type of variable,
                                         'total_events': number of samples}
        """

    def load_meta_attrs(self, data_id: str) -> dict[str, Any]:
        """Load persisted metadata for the given data_id.

        Returns an empty dict by default (in-memory backends have no persistence).
        Override in backends that support metadata persistence (e.g. zarr).
        """
        return {}

    def save_meta_attrs(self, data_id: str, meta: dict[str, Any]) -> None:
        """Persist metadata for the given data_id.

        No-op by default (in-memory backends have no persistence).
        Override in backends that support metadata persistence (e.g. zarr).
        """

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the database."""
