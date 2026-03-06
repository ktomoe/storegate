"""HybridDatabase module."""
from typing import Any

import numpy as np

from storegate.database.database import Database
from storegate.database.zarr_database import ZarrDatabase
from storegate.database.numpy_database import NumpyDatabase


class HybridDatabase(Database):
    """Hybrid database that combines zarr (storage) and numpy (memory) backends."""
    def __init__(self, output_dir: str, mode: str = 'r', chunk: int = 1000) -> None:
        self._output_dir = output_dir
        self._chunk = chunk
        self._mode = mode

        self._db: dict[str, Database] = {}
        self._db['zarr'] = ZarrDatabase(output_dir=output_dir,
                                        chunk=chunk,
                                        mode=mode)
        self._db['numpy'] = NumpyDatabase()
        self._backend: str = 'zarr'

    def initialize(self, data_id: str) -> None:
        self._db['zarr'].initialize(data_id)
        self._db['numpy'].initialize(data_id)

    def set_backend(self, backend: str) -> None:
        if backend not in ('zarr', 'numpy'):
            raise ValueError(f'Unsupported backend: "{backend}". Use "numpy" or "zarr".')
        self._backend = backend

    def get_backend(self) -> str:
        return self._backend

    def add_data(self, data_id: str, var_name: str, data: np.ndarray, phase: str) -> None:
        self._db[self._backend].add_data(data_id, var_name, data, phase)

    def update_data(self, data_id: str, var_name: str, data: np.ndarray, phase: str, index: int | slice | None) -> None:
        self._db[self._backend].update_data(data_id, var_name, data, phase, index)

    def get_data(self, data_id: str, var_name: str, phase: str, index: int | slice | None) -> np.ndarray:
        return self._db[self._backend].get_data(data_id, var_name, phase, index)

    def delete_data(self, data_id: str, var_name: str, phase: str) -> None:
        self._db[self._backend].delete_data(data_id, var_name, phase)

    def get_metadata(self, data_id: str, phase: str) -> dict[str, Any]:
        return self._db[self._backend].get_metadata(data_id, phase)

    def load_meta_attrs(self, data_id: str) -> dict[str, Any]:
        """Delegate to the zarr backend's load_meta_attrs."""
        return self._db['zarr'].load_meta_attrs(data_id)

    def save_meta_attrs(self, data_id: str, meta: dict[str, Any]) -> None:
        """Delegate to the zarr backend's save_meta_attrs."""
        self._db['zarr'].save_meta_attrs(data_id, meta)

    def close(self) -> None:
        for db in self._db.values():
            db.close()
