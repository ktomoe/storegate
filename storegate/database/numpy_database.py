"""NumpyDatabase module."""
from __future__ import annotations

from typing import Any

import numpy as np

from storegate import const
from storegate.database.database import Database


class NumpyDatabase(Database):
    """Base class of Numpy database."""
    def __init__(self) -> None:
        self._db: dict[str, Any] = {}
        self._metadata: dict[str, Any] = {}

    def initialize(self, data_id: str) -> None:
        if data_id not in self._db:
            self._db[data_id] = {}
            self._metadata[data_id] = {}
            for phase in const.PHASES:
                self._db[data_id][phase] = {}
                self._metadata[data_id][phase] = {}

    def add_data(self, data_id: str, var_name: str, data: np.ndarray, phase: str) -> None:
        if var_name in self._db[data_id][phase]:
            tmp_data = self._db[data_id][phase][var_name]
            self._db[data_id][phase][var_name] = np.concatenate((tmp_data, data), axis=0)
            self._metadata[data_id][phase][var_name]['type'] = data.dtype.name
            self._metadata[data_id][phase][var_name]['shape'] = data.shape[1:]
            self._metadata[data_id][phase][var_name]['total_events'] += len(data)

        else:
            self._db[data_id][phase][var_name] = data
            self._metadata[data_id][phase][var_name] = {
                'backend': 'numpy',
                'type': data.dtype.name,
                'shape': data.shape[1:],
                'total_events': len(data)
            }

    def update_data(self, data_id: str, var_name: str, data: np.ndarray, phase: str, index: int | slice | None) -> None:
        if index is None:
            index = slice(0, None)

        self._db[data_id][phase][var_name][index] = data

    def get_data(self, data_id: str, var_name: str, phase: str, index: int | slice | None) -> np.ndarray:
        if index is None:
            index = slice(0, None)

        return self._db[data_id][phase][var_name][index]  # type: ignore[no-any-return]

    def delete_data(self, data_id: str, var_name: str, phase: str) -> None:
        if var_name not in self._db[data_id][phase]:
            raise KeyError(f'"{var_name}" not found in {phase} phase.')
        del self._db[data_id][phase][var_name]
        del self._metadata[data_id][phase][var_name]

    def get_metadata(self, data_id: str, phase: str) -> dict[str, Any]:
        if data_id not in self._metadata:
            return {}
        return self._metadata[data_id][phase]  # type: ignore[no-any-return]
