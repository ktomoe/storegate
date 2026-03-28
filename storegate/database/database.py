from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, Iterator, Self, TypeAlias

import numpy as np
from zarr.core.dtype import parse_dtype

from storegate import utilities as util

if TYPE_CHECKING:
    from storegate.database.staged_add import _StagedAddTransaction

GetIndex: TypeAlias = int | np.integer | slice | list[int | np.integer] | np.ndarray
UpdateIndex: TypeAlias = int | np.integer | slice


class Database(metaclass=ABCMeta):
    """Base class of Database."""

    ################################################################################
    @abstractmethod
    def initialize(self, data_id: str | None = None) -> None:
        """Initialize backend resources.

        Must be called before any other operation. Calling ``initialize()``
        on an already-initialized backend is idempotent.

        Args:
            data_id: If given, the namespace is created (or verified to
                exist for read-only backends) as part of initialization.

        Raises:
            ValueError: If *data_id* fails identifier validation.
            RuntimeError: If the backend cannot be initialized (e.g.
                missing store directory for a read-only Zarr backend).
        """

    @abstractmethod
    def get_name(self) -> str:
        """Return a short, human-readable name identifying this backend type.

        The name is used in error messages and compile reports.

        Returns:
            A lowercase string such as ``"numpy"`` or ``"zarr"``.
        """

    @abstractmethod
    def is_writable(self) -> bool:
        """Return ``True`` if this backend accepts mutating operations.

        Read-only backends must raise on any method decorated with
        ``@requires_write_mode`` or equivalent guard.
        """

    @abstractmethod
    def set_data_id(self, data_id: str) -> None:
        """Ensure the given *data_id* namespace exists on this backend.

        If *data_id* already exists this is a no-op.

        Preconditions:
            The backend must be initialized.

        Args:
            data_id: Identifier (1-128 chars: alphanumeric, ``_``, ``-``).

        Raises:
            ValueError: If *data_id* fails identifier validation.
            RuntimeError: If the backend is not initialized.
        """

    @abstractmethod
    def delete_data_id(self, data_id: str) -> None:
        """Delete the entire namespace for the given *data_id*.

        All phases and variables under *data_id* are removed.

        Preconditions:
            The backend must be initialized and *data_id* must exist.

        Args:
            data_id: The namespace to remove.

        Raises:
            ValueError: If *data_id* does not exist or fails validation.
            RuntimeError: If the backend is not initialized.
        """

    @abstractmethod
    def set_phase(self, data_id: str, phase: str) -> None:
        """Ensure the given *phase* namespace exists under *data_id*.

        If *phase* already exists this is a no-op.

        Preconditions:
            The backend must be initialized and *data_id* must exist.

        Args:
            data_id: Parent namespace.
            phase: Phase name to create.

        Raises:
            ValueError: If *data_id* does not exist or identifiers fail
                validation.
            RuntimeError: If the backend is not initialized.
        """

    @abstractmethod
    def delete_phase(self, data_id: str, phase: str) -> None:
        """Delete the entire namespace for *phase* under *data_id*.

        All variables under *phase* are removed.

        Preconditions:
            The backend must be initialized, *data_id* and *phase* must
            exist.

        Args:
            data_id: Parent namespace.
            phase: Phase to remove.

        Raises:
            ValueError: If *data_id* or *phase* does not exist, or
                identifiers fail validation.
            RuntimeError: If the backend is not initialized.
        """

    @abstractmethod
    def clear(self) -> None:
        """Delete all data_ids and their contents.

        After this call the backend is still initialized but empty.

        Preconditions:
            The backend must be initialized.

        Raises:
            RuntimeError: If the backend is not initialized.
        """

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the database.

        After ``close()`` the backend is no longer usable; calling most
        methods will raise ``RuntimeError``.  Calling ``close()`` on an
        already-closed backend is idempotent.
        """

    def __enter__(self) -> Self:
        self.initialize()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    @abstractmethod
    def add_data(
        self, data_id: str, phase: str, var_name: str, data: np.ndarray
    ) -> None:
        """Add data to the database.

        If *var_name* already exists under the given *phase*, *data* is
        appended along the event axis (axis 0).  On append the dtype and
        feature shape (axes 1..N) must match the existing data.
        If *phase* does not yet exist, it is created automatically.

        The implementation must store a defensive copy of *data* so that
        caller-side mutation cannot corrupt stored data.

        Preconditions:
            The backend must be initialized and *data_id* must exist.

        Args:
            data_id: Target namespace.
            phase: Target phase (created if missing).
            var_name: Variable name.
            data: Array with ``ndim >= 1``.  Must be a plain
                ``numpy.ndarray`` with a Zarr V3-compatible numeric dtype.

        Raises:
            ValueError: If identifiers fail validation, *data_id* does
                not exist, *data* has incompatible dtype/shape on append,
                or ``ndim < 1``.
            TypeError: If *data* is not ``numpy.ndarray``, is a
                ``MaskedArray``, or has a structured/subarray dtype.
            RuntimeError: If the backend is not initialized.
        """

    @abstractmethod
    def update_data(
        self,
        data_id: str,
        phase: str,
        var_name: str,
        data: np.ndarray,
        index: UpdateIndex,
    ) -> None:
        """Replace existing data at *index* for a variable.

        For a scalar index (``int`` / ``np.integer``), *data* must match
        the feature shape (no event axis).  For a ``slice`` index, *data*
        must have ``ndim >= 1`` with matching feature shape and an event
        count equal to the slice length.

        Preconditions:
            The backend must be initialized and *data_id*, *phase*, and
            *var_name* must exist.

        Args:
            data_id: Target namespace.
            phase: Target phase.
            var_name: Variable to update.
            data: Replacement array with compatible dtype and shape.
            index: ``int``, ``np.integer``, or ``slice`` (step must be
                ``None``).

        Raises:
            ValueError: If identifiers fail validation, the target does
                not exist, *index* is invalid, or dtype/shape/event-count
                mismatch.
            TypeError: If *data* is not a valid ``numpy.ndarray``.
            RuntimeError: If the backend is not initialized.
        """

    @abstractmethod
    def get_data(
        self, data_id: str, phase: str, var_name: str, index: GetIndex = slice(None)
    ) -> np.ndarray:
        """Retrieve data for a variable.

        The returned array is always a defensive copy; modifying it does
        not affect the stored data.

        Preconditions:
            The backend must be initialized and *data_id*, *phase*, and
            *var_name* must exist.

        Args:
            data_id: Target namespace.
            phase: Target phase.
            var_name: Variable to read.
            index: ``int``, ``np.integer``, ``slice`` (step must be
                ``None``), ``list[int | np.integer]``, or 1-D integer
                ``np.ndarray``.  Defaults to ``slice(None)`` (all events).

        Returns:
            A new ``numpy.ndarray`` containing the requested slice.

        Raises:
            ValueError: If identifiers fail validation, the target does
                not exist, or *index* is invalid.
            RuntimeError: If the backend is not initialized.
        """

    @abstractmethod
    def delete_data(self, data_id: str, phase: str, var_name: str) -> None:
        """Delete a single variable from a phase.

        The phase itself is retained even if it becomes empty.

        Preconditions:
            The backend must be initialized and *data_id*, *phase*, and
            *var_name* must exist.

        Args:
            data_id: Target namespace.
            phase: Target phase.
            var_name: Variable to delete.

        Raises:
            ValueError: If identifiers fail validation or the target does
                not exist.
            RuntimeError: If the backend is not initialized.
        """

    @abstractmethod
    def stream_data(
        self, data_id: str, phase: str, var_name: str
    ) -> Iterator[np.ndarray]:
        """Yield one or more event-axis chunks for a variable.

        Each yielded array is a defensive copy.  Chunk boundaries depend
        on the backend: ``ZarrDatabase`` yields storage-aligned chunks,
        while ``NumpyDatabase`` yields a single full-array snapshot.

        Variables with zero events must still yield exactly one empty
        chunk (shape ``(0, ...)``) so callers can treat ``stream_data()``
        consistently across backends.

        Preconditions:
            The backend must be initialized and *data_id*, *phase*, and
            *var_name* must exist.

        Args:
            data_id: Target namespace.
            phase: Target phase.
            var_name: Variable to stream.

        Yields:
            ``numpy.ndarray`` chunks along the event axis.

        Raises:
            ValueError: If identifiers fail validation or the target does
                not exist.
            RuntimeError: If the backend is not initialized or is closed
                during iteration.
        """

    @abstractmethod
    def copy_data(
        self, data_id: str, phase: str, var_name: str, output_var_name: str
    ) -> None:
        """Copy a variable within the same phase under a new name.

        If *var_name* equals *output_var_name* this is a no-op.

        Preconditions:
            The backend must be initialized, the source (*data_id*,
            *phase*, *var_name*) must exist, and *output_var_name* must
            not already exist in the same phase.

        Args:
            data_id: Target namespace.
            phase: Target phase.
            var_name: Source variable.
            output_var_name: Destination variable name.

        Raises:
            ValueError: If identifiers fail validation, the source does
                not exist, or the destination already exists.
            RuntimeError: If the backend is not initialized.
        """

    @abstractmethod
    def rename_data(
        self, data_id: str, phase: str, var_name: str, output_var_name: str
    ) -> None:
        """Rename a variable within the same phase.

        If *var_name* equals *output_var_name* this is a no-op.
        After a successful rename, the original *var_name* no longer
        exists.

        Preconditions:
            The backend must be initialized, the source (*data_id*,
            *phase*, *var_name*) must exist, and *output_var_name* must
            not already exist in the same phase.

        Args:
            data_id: Target namespace.
            phase: Target phase.
            var_name: Current variable name.
            output_var_name: New variable name.

        Raises:
            ValueError: If identifiers fail validation, the source does
                not exist, or the destination already exists.
            RuntimeError: If the backend is not initialized.
        """

    @abstractmethod
    def get_data_ids(self) -> list[str]:
        """Return a list of all data_id namespaces.

        The returned order is unspecified and must not be relied on.

        Preconditions:
            The backend must be initialized.

        Returns:
            A list of data_id strings (may be empty).

        Raises:
            RuntimeError: If the backend is not initialized.
        """

    @abstractmethod
    def get_phases(self, data_id: str) -> list[str]:
        """Return a list of phases under *data_id*.

        The returned order is unspecified and must not be relied on.

        Preconditions:
            The backend must be initialized and *data_id* must exist.

        Args:
            data_id: Target namespace.

        Returns:
            A list of phase strings (may be empty).

        Raises:
            ValueError: If *data_id* does not exist or fails validation.
            RuntimeError: If the backend is not initialized.
        """

    @abstractmethod
    def get_var_names(self, data_id: str, phase: str) -> list[str]:
        """Return a list of variable names under *data_id* / *phase*.

        The returned order is unspecified and must not be relied on.

        Preconditions:
            The backend must be initialized and *data_id* / *phase* must
            exist.

        Args:
            data_id: Target namespace.
            phase: Target phase.

        Returns:
            A list of variable name strings (may be empty).

        Raises:
            ValueError: If *data_id* or *phase* does not exist, or
                identifiers fail validation.
            RuntimeError: If the backend is not initialized.
        """

    @abstractmethod
    def get_data_info(self, data_id: str, phase: str, var_name: str) -> dict[str, Any]:
        """Return metadata for a single variable.

        Preconditions:
            The backend must be initialized and *data_id*, *phase*, and
            *var_name* must exist.

        Args:
            data_id: Target namespace.
            phase: Target phase.
            var_name: Target variable.

        Returns:
            A dictionary with the following keys:

            - ``"dtype"`` (``str``): NumPy dtype name (e.g. ``"float32"``).
            - ``"shape"`` (``tuple[int, ...]``): Feature shape (axes 1..N,
              excluding the event axis).
            - ``"num_events"`` (``int``): Number of events (axis-0 length).

        Raises:
            ValueError: If identifiers fail validation or the target does
                not exist.
            RuntimeError: If the backend is not initialized.
        """

    @abstractmethod
    def compile(self, data_id: str) -> dict[str, Any]:
        """Inspect event-count alignment for one *data_id*.

        A phase is "compiled" when it contains at least one variable and
        all its variables share the same event count.  The top-level
        ``is_compiled`` is ``True`` only when every phase is compiled.

        Preconditions:
            The backend must be initialized and *data_id* must exist.

        Args:
            data_id: The namespace to inspect.

        Returns:
            A dictionary with the following structure::

                {
                    "is_compiled": bool,
                    "data_id": str,
                    "phases": {
                        phase: {
                            "is_compiled": bool,
                            "num_events": int | None,
                            "vars": {var_name: num_events},
                        },
                    },
                }

            ``num_events`` at the phase level is ``None`` when the phase
            is not compiled (i.e. event counts disagree or the phase is
            empty).

        Raises:
            ValueError: If *data_id* does not exist or fails validation.
            RuntimeError: If the backend is not initialized.
        """

    ################################################################################
    def _validate_identifiers(
        self,
        data_id: str,
        phase: str | None = None,
        var_name: str | None = None,
        output_var_name: str | None = None,
    ) -> None:
        util.ensure_str(data_id, "data_id")
        if phase is not None:
            util.ensure_str(phase, "phase")
        if var_name is not None:
            util.ensure_str(var_name, "var_name")
        if output_var_name is not None:
            util.ensure_str(output_var_name, "output_var_name")

    def _validate_data(self, data: np.ndarray) -> None:
        if np.ma.isMaskedArray(data):
            raise TypeError(
                "numpy.ma.MaskedArray is not allowed. "
                "Convert it to a plain numpy.ndarray explicitly before storing."
            )
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(data).__name__}")
        if data.dtype.fields is not None:
            raise TypeError("Structured dtype is not allowed")
        if data.dtype.subdtype is not None:  # pragma: no cover
            raise TypeError("Subarray dtype is not allowed")
        if data.dtype.kind in {"U", "S"}:
            raise ValueError(
                f'dtype "{data.dtype.name}" is not persistable under the storegate contract. '
                "Text and bytes dtypes are rejected because their Zarr V3 "
                "representations are unstable."
            )

        try:
            parse_dtype(data.dtype, zarr_format=3)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f'dtype "{data.dtype.name}" is not persistable under the storegate contract. '
                "Cast the array to a zarr-compatible dtype before adding it."
            ) from exc

    def _validate_add_data_new(
        self, data_id: str, phase: str, var_name: str, data: np.ndarray
    ) -> None:
        self._is_batch_data(data_id, phase, var_name, data)

    def _validate_add_data_append(
        self, data_id: str, phase: str, var_name: str, data: np.ndarray
    ) -> None:
        self._is_batch_data(data_id, phase, var_name, data)
        self._match_data_info(data_id, phase, var_name, data, ignore_first_axis=True)

    def _validate_update_data_single(
        self, data_id: str, phase: str, var_name: str, data: np.ndarray
    ) -> None:
        self._match_data_info(data_id, phase, var_name, data)

    def _validate_update_data_batch(
        self, data_id: str, phase: str, var_name: str, data: np.ndarray, index: slice
    ) -> None:
        self._is_batch_data(data_id, phase, var_name, data)
        self._match_data_info(data_id, phase, var_name, data, ignore_first_axis=True)

        expected_events = len(
            range(
                *index.indices(
                    self.get_data_info(data_id, phase, var_name)["num_events"]
                )
            )
        )
        if data.shape[0] != expected_events:
            raise ValueError(
                f"backend:{self.get_name()}, data_id:{data_id}, phase:{phase}, var_name:{var_name}| expected {expected_events} events for index {index}, got {data.shape[0]}"
            )

    def _validate_get_data_index(self, index: GetIndex) -> None:
        if isinstance(index, bool):
            raise ValueError("index must not be bool")
        if isinstance(index, (int, np.integer)):
            return
        if isinstance(index, slice) and index.step is None:
            return
        if isinstance(index, list):
            if not index:
                raise ValueError("index must not be an empty list")
            if all(isinstance(v, bool) for v in index):
                raise ValueError("index must not be list[bool]")
            if all(
                isinstance(v, (int, np.integer)) and not isinstance(v, bool)
                for v in index
            ):
                return
        if (
            isinstance(index, np.ndarray)
            and index.ndim == 1
            and np.issubdtype(index.dtype, np.integer)
        ):
            if index.size == 0:
                raise ValueError("index must not be an empty numpy.ndarray")
            return
        raise ValueError(
            "index must be: int, np.integer, slice without step, list[int|np.integer], or 1D integer np.ndarray"
        )

    def _validate_update_data_index(self, index: UpdateIndex) -> None:
        if isinstance(index, bool):
            raise ValueError("index must not be bool")
        if isinstance(index, (int, np.integer)):
            return
        if isinstance(index, slice) and index.step is None:
            return
        raise ValueError("index must be: int, np.integer, or slice without step")

    ################################################################################
    def _is_batch_data(
        self, data_id: str, phase: str, var_name: str, data: np.ndarray
    ) -> None:
        if data.ndim < 1:
            raise ValueError(
                f"backend:{self.get_name()}, data_id:{data_id}, phase:{phase}, var_name:{var_name}| len(data.shape) must be >= 1"
            )

    def _match_data_info(
        self,
        data_id: str,
        phase: str,
        var_name: str,
        data: np.ndarray,
        ignore_first_axis: bool = False,
    ) -> None:
        metadata = self.get_data_info(data_id, phase, var_name)

        actual_dtype = data.dtype.name
        actual_shape = tuple(data.shape[1:] if ignore_first_axis else data.shape)

        if actual_dtype != metadata["dtype"]:
            raise ValueError(
                f"backend:{self.get_name()}, data_id:{data_id}, phase:{phase}, var_name:{var_name}| expected {metadata['dtype']}, got {actual_dtype}"
            )

        if actual_shape != metadata["shape"]:
            raise ValueError(
                f"backend:{self.get_name()}, data_id:{data_id}, phase:{phase}, var_name:{var_name}| expected {metadata['shape']}, got {actual_shape}"
            )

    def staged_add(
        self, data_id: str, phase: str, var_names: list[str]
    ) -> "_StagedAddTransaction":
        """Return a context manager that stages ``add_data`` calls.

        * ``__enter__``: allocates a temporary variable for each name in
          *var_names* and validates that none of the public names already
          exist.
        * ``tx.add_data(name, data)``: appends *data* to the corresponding
          temporary variable via ``add_data()``.
        * On **normal** exit the temporary variables are published to their
          public names.
        * If publication fails part-way through, already-published variables
          are renamed back to their temporary names and all staged variables
          are deleted (best-effort).
        * On **exception** exit all temporary variables are deleted
          (best-effort).

        Args:
            data_id: Target namespace (must already exist).
            phase: Target phase (created automatically on first ``add_data``).
            var_names: Public variable names to stage.

        Returns:
            A :class:`_StagedAddTransaction` context manager.
        """
        from storegate.database import staged_add as staged_add_module

        return staged_add_module._StagedAddTransaction(
            self, data_id, phase, var_names,
        )
