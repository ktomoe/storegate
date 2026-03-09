"""Module to define Database abstraction."""
from abc import ABCMeta, abstractmethod
from collections.abc import Iterator
from typing import Any

import numpy as np
from zarr.dtype import parse_dtype


class Database(metaclass=ABCMeta):
    """Base class of Database."""

    @staticmethod
    def _validate_zarr_compatible_dtype(
        data: np.ndarray,
        *,
        var_name: str,
        phase: str,
    ) -> None:
        """Reject dtypes that StoreGate cannot persist to the zarr backend."""
        try:
            parse_dtype(data.dtype, zarr_format=3)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"dtype '{data.dtype.name}' for '{var_name}' in '{phase}' is not "
                'persistable to the zarr backend. '
                'Cast the array to a zarr-compatible dtype before adding it.'
            ) from exc

    @staticmethod
    def _normalize_index(index: int | slice | None) -> int | slice:
        """Convert None to a full-range slice."""
        return slice(0, None) if index is None else index

    @classmethod
    def _selection_shape(
        cls,
        total_events: int,
        sample_shape: tuple[int, ...],
        index: int | slice | None,
    ) -> tuple[int, ...]:
        """Return the exact shape required to update the selected region."""
        normalized = cls._normalize_index(index)
        if isinstance(normalized, int):
            resolved_index = normalized if normalized >= 0 else total_events + normalized
            if resolved_index < 0 or resolved_index >= total_events:
                raise IndexError(
                    f'index {normalized} is out of bounds for axis 0 with size {total_events}'
                )
            return sample_shape

        start, stop, step = normalized.indices(total_events)
        return (len(range(start, stop, step)),) + sample_shape

    @classmethod
    def _prepare_update_data(
        cls,
        *,
        data: np.ndarray,
        total_events: int,
        sample_shape: tuple[int, ...],
        existing_dtype: np.dtype[Any],
        index: int | slice | None,
        var_name: str,
        phase: str,
    ) -> np.ndarray:
        """Validate update payload shape/dtype and cast safely when allowed."""
        expected_shape = cls._selection_shape(total_events, sample_shape, index)

        if data.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch for update of '{var_name}' in '{phase}': "
                f"expected {expected_shape}, got {data.shape}"
            )

        existing_dtype = np.dtype(existing_dtype)
        if data.dtype != existing_dtype:
            promoted = np.result_type(existing_dtype, data.dtype)
            if promoted != existing_dtype:
                raise ValueError(
                    f"dtype mismatch for update of '{var_name}' in '{phase}': "
                    f"existing={existing_dtype}, incoming={data.dtype.name}. "
                    f"Updating would require a lossy cast to {existing_dtype} "
                    f"(safe promotion would be {promoted.name}). "
                    f"Cast your data to {existing_dtype} explicitly before updating."
                )
            data = data.astype(existing_dtype, copy=False)

        return data

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
    def rename_data(
        self,
        data_id: str,
        var_name: str,
        output_var_name: str,
        phase: str,
    ) -> None:
        """Rename data for given data_id, var_name and phase within one backend."""

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

    def get_pending_var_names(self) -> dict[str, dict[str, list[str]]]:
        """Return in-memory-only variable names, grouped by data_id and phase.

        Backends without a pending/in-memory concept return an empty result.
        """
        return {}

    def iter_data_chunks(
        self,
        data_id: str,
        var_name: str,
        phase: str,
    ) -> Iterator[np.ndarray]:
        """Yield one or more event-axis chunks for a variable.

        Backends override this to stream data without materializing the full
        variable in memory. The default implementation falls back to a single
        full-array read for compatibility.
        """
        yield self.get_data(data_id, var_name, phase, None)

    def clear_data_id(self, data_id: str) -> None:
        """Discard any backend state for ``data_id`` when supported."""

    def snapshot_data_id(self, data_id: str, snapshot_name: str) -> None:
        """Persist a point-in-time snapshot for ``data_id`` when supported."""
        raise NotImplementedError

    def restore_data_id(self, data_id: str, snapshot_name: str) -> None:
        """Restore ``data_id`` from a previously saved snapshot when supported."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the database."""
