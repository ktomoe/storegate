"""ZarrDatabase module."""
from __future__ import annotations

from typing import Any

import numpy as np
import zarr

from storegate import const
from storegate.database.database import Database


class ZarrDatabase(Database):
    """Base class of Zarr database."""
    def __init__(self, output_dir: str, chunk: int = 1000, mode: str = 'r') -> None:
        self._output_dir = output_dir
        self._chunk = chunk
        self._mode = mode
        self._db: Any = zarr.open(self._output_dir, mode=mode)

    def initialize(self, data_id: str) -> None:
        """Initialize the store for the given data_id.

        In read-only mode ('r'), data_id must already exist in the store.
        In writable modes, groups are created if they do not exist.
        """
        if self._mode == 'r':
            if data_id not in self._db.group_keys():
                raise KeyError(
                    f"data_id '{data_id}' does not exist in '{self._output_dir}' (mode='r')."
                )
            return
        db_data_id = self._db.require_group(data_id)
        for phase in const.PHASES:
            db_data_id.require_group(phase)

    def add_data(self, data_id: str, var_name: str, data: np.ndarray, phase: str) -> None:
        db = self._db[data_id][phase]

        if var_name in db.array_keys():
            db[var_name].append(data)
        else:
            shape = data.shape
            chunks = (self._chunk, ) + tuple(shape[1:])
            db.create_array(name=var_name, data=data, chunks=chunks)

    def update_data(self, data_id: str, var_name: str, data: np.ndarray, phase: str, index: int | slice | None) -> None:
        self._db[data_id][phase][var_name][self._normalize_index(index)] = data

    def get_data(self, data_id: str, var_name: str, phase: str, index: int | slice | None) -> np.ndarray:
        return self._db[data_id][phase][var_name][self._normalize_index(index)]  # type: ignore[no-any-return]

    def delete_data(self, data_id: str, var_name: str, phase: str) -> None:
        if var_name not in self._db[data_id][phase].array_keys():
            raise KeyError(f'"{var_name}" not found in {phase} phase.')
        del self._db[data_id][phase][var_name]

    def get_metadata(self, data_id: str, phase: str) -> dict[str, Any]:
        results: dict[str, Any] = {}
        if data_id not in self._db.group_keys():
            return results

        db = self._db[data_id][phase]

        for var_name in db.array_keys():
            arr = db[var_name]
            results[var_name] = {
                'backend': 'zarr',
                'type': arr.dtype.name,
                'shape': arr.shape[1:],
                'total_events': arr.shape[0]
            }
        return results

    def close(self) -> None:
        self._db.store.close()
