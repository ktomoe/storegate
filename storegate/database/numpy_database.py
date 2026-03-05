"""NumpyDatabase module."""
from __future__ import annotations

from typing import Any

import numpy as np

from storegate import const
from storegate.database.database import Database


class NumpyDatabase(Database):
    """Base class of Numpy database."""
    def __init__(self) -> None:
        # data_id -> phase -> var_name -> list[np.ndarray]
        self._chunks: dict[str, Any] = {}
        # data_id -> phase -> var_name -> np.ndarray | None  (None = dirty, needs concat)
        self._cache: dict[str, Any] = {}
        self._metadata: dict[str, Any] = {}

    def initialize(self, data_id: str) -> None:
        if data_id not in self._chunks:
            self._chunks[data_id] = {}
            self._cache[data_id] = {}
            self._metadata[data_id] = {}
            for phase in const.PHASES:
                self._chunks[data_id][phase] = {}
                self._cache[data_id][phase] = {}
                self._metadata[data_id][phase] = {}

    def _materialize(self, data_id: str, var_name: str, phase: str) -> np.ndarray:
        """Concatenate pending chunks into a single array, caching the result.

        After concatenation the chunk list is collapsed to one element so the
        original per-chunk arrays can be garbage-collected, halving peak memory.
        """
        if self._cache[data_id][phase][var_name] is None:
            arr = np.concatenate(self._chunks[data_id][phase][var_name], axis=0)
            self._cache[data_id][phase][var_name] = arr
            self._chunks[data_id][phase][var_name] = [arr]
        return self._cache[data_id][phase][var_name]  # type: ignore[return-value]

    def add_data(self, data_id: str, var_name: str, data: np.ndarray, phase: str) -> None:
        if var_name in self._chunks[data_id][phase]:
            self._chunks[data_id][phase][var_name].append(data)
            self._cache[data_id][phase][var_name] = None  # invalidate
            meta = self._metadata[data_id][phase][var_name]
            meta['total_events'] += len(data)
            promoted = np.result_type(np.dtype(meta['type']), data.dtype).name
            if promoted != meta['type']:
                from storegate import logger
                logger.warn(
                    f"dtype promotion for '{var_name}' in '{phase}': "
                    f"{meta['type']} + {data.dtype.name} -> {promoted}"
                )
            meta['type'] = promoted
        else:
            self._chunks[data_id][phase][var_name] = [data]
            self._cache[data_id][phase][var_name] = None
            self._metadata[data_id][phase][var_name] = {
                'backend': 'numpy',
                'type': data.dtype.name,
                'shape': data.shape[1:],
                'total_events': len(data),
            }

    def update_data(self, data_id: str, var_name: str, data: np.ndarray, phase: str, index: int | slice | None) -> None:
        arr = self._materialize(data_id, var_name, phase)
        arr[self._normalize_index(index)] = data
        # Collapse to single chunk so the updated array is the source of truth
        self._chunks[data_id][phase][var_name] = [arr]

    def get_data(self, data_id: str, var_name: str, phase: str, index: int | slice | None) -> np.ndarray:
        return self._materialize(data_id, var_name, phase)[self._normalize_index(index)]  # type: ignore[no-any-return]

    def delete_data(self, data_id: str, var_name: str, phase: str) -> None:
        if var_name not in self._chunks[data_id][phase]:
            raise KeyError(f'"{var_name}" not found in {phase} phase.')
        del self._chunks[data_id][phase][var_name]
        del self._cache[data_id][phase][var_name]
        del self._metadata[data_id][phase][var_name]

    def get_metadata(self, data_id: str, phase: str) -> dict[str, Any]:
        if data_id not in self._metadata:
            return {}
        return {k: dict(v) for k, v in self._metadata[data_id][phase].items()}

    def close(self) -> None:
        pass
