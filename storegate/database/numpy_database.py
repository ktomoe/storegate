"""NumpyDatabase module."""
from typing import Any

import numpy as np

from storegate import const
from storegate.database.database import Database


class NumpyDatabase(Database):
    """In-memory database backed by numpy arrays.

    Warning:
        This class is **not thread-safe**.  Concurrent access from
        multiple threads without external locking may corrupt internal
        chunk lists, caches, or metadata.
    """
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
            meta = self._metadata[data_id][phase][var_name]
            if data.shape[1:] != meta['shape']:
                raise ValueError(
                    f"Shape mismatch for '{var_name}' in '{phase}': "
                    f"expected {meta['shape']}, got {data.shape[1:]}"
                )
            existing_dtype = np.dtype(meta['type'])
            if data.dtype != existing_dtype:
                promoted = np.result_type(existing_dtype, data.dtype)
                if promoted != existing_dtype:
                    raise ValueError(
                        f"dtype mismatch for '{var_name}' in '{phase}': "
                        f"existing={existing_dtype}, incoming={data.dtype.name}. "
                        f"Appending would require a lossy cast to {existing_dtype} "
                        f"(safe promotion would be {promoted.name}). "
                        f"Cast your data to {existing_dtype} explicitly before adding."
                    )
            self._chunks[data_id][phase][var_name].append(data)
            self._cache[data_id][phase][var_name] = None  # invalidate
            meta['total_events'] += len(data)
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
        meta = self._metadata[data_id][phase][var_name]
        data = self._prepare_update_data(
            data=data,
            total_events=meta['total_events'],
            sample_shape=meta['shape'],
            existing_dtype=np.dtype(meta['type']),
            index=index,
            var_name=var_name,
            phase=phase,
        )
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

    def get_pending_var_names(self) -> dict[str, dict[str, list[str]]]:
        """Return variable names currently held in memory, grouped by data_id and phase.

        Returns:
            ``{data_id: {phase: [var_name, ...]}}``.
            Only non-empty phases are included.
        """
        result: dict[str, dict[str, list[str]]] = {}
        for data_id, phases in self._chunks.items():
            for phase, vars_ in phases.items():
                if vars_:
                    result.setdefault(data_id, {})[phase] = list(vars_.keys())
        return result

    def close(self) -> None:
        """Release all in-memory data held by this database.

        Warning:
            This operation is **destructive and irreversible**.  All data
            stored in the numpy (memory) backend — including every chunk,
            cached array, and metadata entry — is permanently discarded.
            Any data that has not been copied to the zarr (disk) backend via
            ``StoreGate.copy_to_storage()`` will be lost.
        """
        self._chunks.clear()
        self._cache.clear()
        self._metadata.clear()
