"""StoreGate module."""
from __future__ import annotations

import functools
import re
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, TypeVar

import numpy as np

from storegate import logger, const
from storegate.database import HybridDatabase

_F = TypeVar('_F', bound=Callable[..., Any])


class _PhaseAccessor:
    """Intermediate accessor returned by StoreGate[phase].

    Holds phase context without mutating the parent StoreGate instance.
    """

    def __init__(self, storegate: StoreGate, phase: str) -> None:
        self._storegate = storegate
        self._phase = phase

    def __getitem__(self, item: str) -> _VarAccessor:
        if isinstance(item, str):
            return _VarAccessor(self._storegate, self._phase, item)
        raise NotImplementedError(f'item {item} is not supported')

    def __setitem__(self, item: str, data: Any) -> None:
        if not isinstance(item, str):
            raise ValueError(f'item {item} must be str')
        self._storegate.add_data(item, data, phase=self._phase)

    def __delitem__(self, item: str) -> None:
        if not isinstance(item, str):
            raise ValueError(f'item {item} must be str')
        self._storegate.delete_data(item, phase=self._phase)

    def __iter__(self) -> Iterator[str]:
        return iter(self._storegate.get_var_names(self._phase))

    def items(self) -> Generator[tuple[str, np.ndarray], None, None]:
        """Yield (var_name, data) pairs for all variables in this phase."""
        for var_name in self:
            yield var_name, self._storegate.get_data(var_name, self._phase, index=None)

    def __contains__(self, item: object) -> bool:
        return item in self._storegate.get_var_names(self._phase)

    def __len__(self) -> int:
        data_id = self._storegate._require_current_data_id()
        backend = self._storegate.get_backend()
        if not self._storegate._metadata[data_id]['compiled'][backend]:
            raise ValueError('len() is supported only after compile')
        size = self._storegate._metadata[data_id]['sizes'][backend][self._phase]
        return 0 if size is None else size


class _AllPhaseAccessor:
    """Intermediate accessor returned by StoreGate['all'].

    Only deletion is supported.  Use ``del sg['all']['x']`` to remove a
    variable from every phase in a single call.

    All other operations (read, write, iteration, membership test, len) raise
    ``NotImplementedError`` because ``'all'`` has no single meaningful
    semantics for those operations — use phase-specific accessors instead::

        for name in sg['train']:   ...  # iterate one phase
        'x' in sg['valid']             # membership in one phase
        len(sg['test'])                # size of one phase
    """

    def __init__(self, storegate: StoreGate) -> None:
        self._storegate = storegate

    def __delitem__(self, item: str) -> None:
        if not isinstance(item, str):
            raise ValueError(f'item {item} must be str')
        self._storegate.delete_data(item, phase='all')

    def __getitem__(self, item: object) -> Any:
        raise NotImplementedError(
            "Read access via sg['all'][var] is not supported. "
            "Use sg['train'][var], sg['valid'][var], or sg['test'][var] instead."
        )

    def __setitem__(self, item: object, data: Any) -> None:
        raise NotImplementedError(
            "Write access via sg['all'][var] = data is not supported. "
            "Use sg['train'][var] = data (etc.) per phase instead."
        )

    def __contains__(self, item: object) -> bool:
        raise NotImplementedError(
            "Membership test via sg['all'] is not supported. "
            "Use 'x' in sg['train'] (etc.) per phase instead."
        )

    def __iter__(self) -> Iterator[str]:
        raise NotImplementedError(
            "Iteration via sg['all'] is not supported. "
            "Iterate over sg['train'], sg['valid'], or sg['test'] instead."
        )

    def __len__(self) -> int:
        raise NotImplementedError(
            "len(sg['all']) is not supported. "
            "Use len(sg['train']), len(sg['valid']), or len(sg['test']) instead."
        )


class _VarAccessor:
    """Intermediate accessor returned by StoreGate[phase][var_name].

    Holds phase and var_name context without mutating the parent StoreGate instance.
    """

    def __init__(self, storegate: StoreGate, phase: str, var_name: str) -> None:
        self._storegate = storegate
        self._phase = phase
        self._var_name = var_name

    def __getitem__(self, item: int | slice) -> np.ndarray:
        if not isinstance(item, (int, slice)):
            raise NotImplementedError(f'item {item} is not supported')
        return self._storegate.get_data(
            var_name=self._var_name, phase=self._phase, index=item
        )

    def __setitem__(self, item: int | slice, data: Any) -> None:
        if not isinstance(item, (int, slice)):
            raise ValueError(f'item {item} must be int or slice')
        self._storegate.update_data(
            var_name=self._var_name, data=data, phase=self._phase, index=item
        )


def require_data_id(method: _F) -> _F:
    """Raise RuntimeError if self._data_id is not set."""
    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if self._data_id is None:
            raise RuntimeError(
                f'{method.__name__}() requires data_id to be set. Call set_data_id() first.'
            )
        return method(self, *args, **kwargs)
    return wrapper  # type: ignore[return-value]


_VALID_IDENTIFIER = re.compile(r'^[a-zA-Z0-9_-]{1,128}$')
_SUPPORTED_BACKENDS = ('numpy', 'zarr')


def _validate_data_id(data_id: str) -> None:
    """Raise ValueError if data_id contains invalid characters."""
    if not isinstance(data_id, str) or not _VALID_IDENTIFIER.match(data_id):
        raise ValueError(
            f"Invalid data_id {data_id!r}. "
            "Must be 1-128 characters: alphanumeric, underscore, or hyphen only."
        )


def _validate_var_name(var_name: str) -> None:
    """Raise ValueError if var_name contains invalid characters."""
    if not isinstance(var_name, str) or not _VALID_IDENTIFIER.match(var_name):
        raise ValueError(
            f"Invalid var_name {var_name!r}. "
            "Must be 1-128 characters: alphanumeric, underscore, or hyphen only."
        )


def _validate_snapshot_name(snapshot_name: str) -> None:
    """Raise ValueError if snapshot_name contains invalid characters."""
    if not isinstance(snapshot_name, str) or not _VALID_IDENTIFIER.match(snapshot_name):
        raise ValueError(
            f"Invalid snapshot_name {snapshot_name!r}. "
            "Must be 1-128 characters: alphanumeric, underscore, or hyphen only."
        )


def _validate_phase(phase: str, allow_all: bool = False) -> None:
    """Raise ValueError if phase is not a valid phase name."""
    valid = const.PHASES + ('all',) if allow_all else const.PHASES
    if phase not in valid:
        raise ValueError(f"Invalid phase '{phase}'. Must be one of {valid}.")


def _validate_backend(backend: str) -> None:
    if backend not in _SUPPORTED_BACKENDS:
        raise ValueError(f'Unsupported backend: "{backend}". Use "numpy" or "zarr".')


def _validate_output_dir(output_dir: str, mode: str) -> None:
    """Raise ValueError if output_dir is invalid for the given mode.

    - ``mode='r'``: the path must already exist.
    - ``mode='w'`` / ``mode='a'``: the parent directory must exist.
    """
    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError(
            f"output_dir must be a non-empty string, got: {output_dir!r}"
        )
    if mode not in ('r', 'w', 'a'):
        raise ValueError(
            f"Invalid mode {mode!r}. Must be 'r', 'w', or 'a'."
        )
    path = Path(output_dir)
    if mode == 'r' and not path.exists():
        raise ValueError(
            f"output_dir {output_dir!r} does not exist (mode='r')."
        )
    if mode in ('w', 'a') and not path.parent.exists():
        raise ValueError(
            f"Parent directory of output_dir does not exist: {path.parent!r}"
        )


def _validate_chunk(chunk: int) -> None:
    """Raise ValueError if chunk is not a positive integer."""
    if not isinstance(chunk, int) or isinstance(chunk, bool):
        raise ValueError(
            f'chunk must be a positive integer, got: {chunk!r}'
        )
    if chunk <= 0:
        raise ValueError(
            f'chunk must be a positive integer, got: {chunk!r}'
        )


class StoreGate:
    """Data management class.

    Warning:
        This class is **not thread-safe**.  All access to a single
        ``StoreGate`` instance (and its underlying databases) must be
        serialised by the caller.  Concurrent reads/writes from multiple
        threads without external locking may cause data corruption or
        inconsistent metadata.  Use one instance per thread, or protect
        shared instances with a ``threading.Lock``.

        **Concurrent processes** may write to the same zarr store only
        when each process writes to a different ``data_id`` or a
        different set of variable names (``output_var_names``).
        Concurrent writes to the *same* ``data_id`` and variable name
        will corrupt data because zarr has no built-in file locking.
    """

    def __init__(self, output_dir: str, mode: str = 'r', chunk: int = 1000, data_id: str | None = None) -> None:
        """Initialize the storegate and the zarr architecture.

        Args:
            output_dir (str): Directory path for the zarr store.
            mode (str): File open mode passed to zarr.
                ``'w'`` creates or overwrites the store.
                ``'a'`` opens for appending (creates if absent).
                ``'r'`` opens read-only; the store must already exist.
            chunk (int): Chunk size along the first (event) axis for zarr arrays.
            data_id (str or None): If provided, calls ``set_data_id()`` immediately.

        Note:
            ``compile()`` persists the zarr backend's compiled flag and phase sizes
            in the store attrs. When reopening an existing store (``mode='r'`` or
            ``mode='a'``), that zarr metadata is restored automatically, so
            ``len(sg[phase])`` works immediately if the store was last left in a
            compiled zarr state. Metadata for the numpy backend remains
            in-memory-only and is not restored across sessions.

        Examples:
            Write, then reopen and read::

                sg = StoreGate(output_dir='./store', mode='w', data_id='exp01')
                sg.add_data('x', x_train, phase='train')

                # Reopen in read-only mode
                sg = StoreGate(output_dir='./store', mode='r', data_id='exp01')
                x = sg.get_data('x', phase='train')
                # Call sg.compile() only if the store was never compiled, or was
                # modified after the last successful compile().
        """

        _validate_output_dir(output_dir, mode)
        _validate_chunk(chunk)
        output_dir = str(Path(output_dir).resolve())
        self._db: HybridDatabase = HybridDatabase(output_dir=output_dir, mode=mode, chunk=chunk)
        self._data_id: str | None = None
        self._metadata: dict[str, Any] = {}

        if data_id is not None:
            self.set_data_id(data_id)


    def __repr__(self) -> str:
        if self._data_id is None:
            return 'StoreGate(data_id=None)'
        backend = self.get_backend()
        compiled = self._metadata[self._data_id]['compiled'][backend]
        return f'StoreGate(data_id={self._data_id!r}, backend={backend!r}, compiled={compiled})'

    def __enter__(self) -> StoreGate:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        self.close()
        return False

    def close(self) -> None:
        """Release resources held by the underlying databases.

        Warning:
            Any data stored **only** in the numpy (memory) backend is
            permanently discarded.  Persist it first with
            ``copy_to_storage()`` if you need to keep it on disk.
            This method is called automatically when using StoreGate as a
            context manager (``with StoreGate(...) as sg:``).
        """
        try:
            for data_id, phases in self._db.get_pending_var_names().items():
                for phase, var_names in phases.items():
                    var_list = ', '.join(var_names)
                    logger.warning(
                        f"close(): discarding unsaved numpy data."
                        f" data_id='{data_id}', phase='{phase}', vars=[{var_list}]"
                    )
        finally:
            self._db.close()

    @require_data_id
    def __getitem__(self, item: str) -> _PhaseAccessor | _AllPhaseAccessor:
        """Return a phase accessor for chained access: sg[phase][var_name][index].

        For ``item='all'``, returns an :class:`_AllPhaseAccessor` that only
        supports deletion (``del sg['all']['x']``).  All other operations on
        that accessor raise ``NotImplementedError``.
        """
        if item in const.PHASES:
            return _PhaseAccessor(self, item)
        if item == 'all':
            return _AllPhaseAccessor(self)
        raise NotImplementedError(f'item {item} is not supported')


    ##########################################################################
    # Public user APIs
    ##########################################################################
    @property
    def data_id(self) -> str | None:
        """Return the current data_id."""
        return self._data_id

    def _require_current_data_id(self) -> str:
        data_id = self._data_id
        if data_id is None:
            raise RuntimeError('data_id is not set. Call set_data_id() first.')
        return data_id

    def set_data_id(self, data_id: str) -> None:
        """Set the default ``data_id`` and initialize the zarr.

        Note:
            Compiled state and phase sizes are automatically restored from the
            zarr store if ``compile()`` was previously called and the store was
            saved.  Re-calling ``compile()`` is only necessary when new data has
            been added since the last ``compile()``.
        """
        _validate_data_id(data_id)
        previous_data_id = self._data_id
        has_metadata = data_id in self._metadata

        if not has_metadata:
            self._metadata[data_id] = {
                'compiled': {backend: False for backend in _SUPPORTED_BACKENDS},
                'sizes': {backend: {} for backend in _SUPPORTED_BACKENDS},
            }

        try:
            self._db.initialize(data_id)
            self._data_id = data_id
            if not has_metadata:
                self._load_meta(data_id)
        except Exception:
            self._data_id = previous_data_id
            if not has_metadata:
                del self._metadata[data_id]
            raise


    @require_data_id
    def set_backend(self, backend: str) -> None:
        """Set backend mode of hybrid architecture.

        Note:
            The numpy backend is purely in-memory.  Its data and compiled
            metadata are **not persisted** across sessions.  When the
            ``StoreGate`` instance is closed or garbage-collected, all
            numpy-only data is lost.  Use ``copy_to_storage()`` to persist
            numpy data to the zarr backend before closing.
        """
        _validate_backend(backend)
        self._db.set_backend(backend)


    @require_data_id
    def get_backend(self) -> str:
        return self._db.get_backend()

    @require_data_id
    @contextmanager
    def using_backend(self, backend: str) -> Generator[StoreGate, None, None]:
        """Context manager that temporarily switches to ``backend``, restoring the original on exit."""
        _validate_backend(backend)
        old = self.get_backend()
        self.set_backend(backend)
        try:
            yield self
        finally:
            self.set_backend(old)

    @staticmethod
    def _target_phases(phase: str) -> tuple[str, ...]:
        return const.PHASES if phase == 'all' else (phase,)


    @require_data_id
    def add_data(self, var_name: str, data: Any, phase: str) -> None:
        """Append data for the given variable and phase.

        Args:
            var_name (str): Variable name to register.
            data: Array-like data to append. Converted to ``np.ndarray`` internally.
            phase (str): One of ``'train'``, ``'valid'``, or ``'test'``.

        Note:
            The array dtype is fixed at creation time.  Appending data whose
            dtype would require promotion (e.g. ``float32`` + ``float64``)
            raises ``ValueError``.  Cast your data explicitly before calling
            ``add_data``, e.g. ``data.astype(existing_dtype)``.
            Appending a narrower dtype that can be safely represented by the
            existing dtype (e.g. ``int32`` into ``float64``) is allowed.
        """
        _validate_var_name(var_name)
        _validate_phase(phase)
        data = np.asarray(data)
        if data.ndim == 0:
            raise ValueError(
                'data must be at least 1-dimensional. '
                'Wrap scalar values in a list or array, e.g. np.array([value]).'
            )
        data_id = self._require_current_data_id()
        self._db.add_data(data_id, var_name, data, phase)
        self._invalidate_compiled(phase)


    @require_data_id
    def add_data_splits(self, var_name: str, *, train: Any = None, valid: Any = None, test: Any = None) -> None:
        """Add data for multiple phases in a single call.

        Each phase argument is optional; omitted phases are skipped.

        Args:
            var_name (str): Variable name to register.
            train: Data for the train phase.
            valid: Data for the valid phase.
            test:  Data for the test phase.

        Examples:
            >>> sg.add_data_splits('x', train=x_train, valid=x_valid, test=x_test)
        """
        for phase, data in zip(const.PHASES, (train, valid, test)):
            if data is not None:
                self.add_data(var_name, data, phase)


    @require_data_id
    def update_data(self, var_name: str, data: Any, phase: str, index: int | slice | None = None) -> None:
        """Update data for the selected region of an existing variable.

        The incoming payload must match the selected region exactly:
        - ``index=None`` updates the full array and requires an identical shape.
        - ``index=int`` updates one event and requires the per-event shape.
        - ``index=slice`` updates multiple events and requires the exact
          ``(num_selected, *sample_shape)`` shape.

        Unlike raw numpy assignment, broadcast updates are not allowed.
        The existing dtype is preserved; lossy casts raise ``ValueError``.
        """
        _validate_var_name(var_name)
        _validate_phase(phase)
        data = np.asarray(data)
        data_id = self._require_current_data_id()
        self._db.update_data(data_id, var_name, data, phase, index)
        self._invalidate_compiled(phase)


    @require_data_id
    def get_data(self, var_name: str, phase: str, index: int | slice | None = None) -> np.ndarray:
        """Retrieve data from storegate with given options."""
        _validate_var_name(var_name)
        _validate_phase(phase)
        data_id = self._require_current_data_id()
        return self._db.get_data(data_id, var_name, phase, index)


    @require_data_id
    def delete_data(self, var_name: str, phase: str) -> None:
        """Delete data associated with var_names."""
        _validate_var_name(var_name)
        _validate_phase(phase, allow_all=True)
        data_id = self._require_current_data_id()
        for iphase in self._target_phases(phase):
            if phase == 'all' and var_name not in self.get_var_names(iphase):
                continue
            self._db.delete_data(data_id, var_name, iphase)
        self._invalidate_compiled(phase)

    @require_data_id
    def rename_data(self, var_name: str, output_var_name: str, phase: str) -> None:
        """Rename a variable within the currently active backend."""
        _validate_var_name(var_name)
        _validate_var_name(output_var_name)
        _validate_phase(phase)
        data_id = self._require_current_data_id()
        self._db.rename_data(data_id, var_name, output_var_name, phase)


    @require_data_id
    def get_var_names(self, phase: str) -> list[str]:
        """Returns registered var_names for given phase."""
        _validate_phase(phase)
        data_id = self._require_current_data_id()
        metadata = self._db.get_metadata(data_id, phase)
        return list(metadata.keys())


    @require_data_id
    def copy_to_memory(self, var_name: str, phase: str, output_var_name: str | None = None) -> None:
        """Copy data from storage to memory.

        Note:
            The copied data resides in the numpy (in-memory) backend and is
            **not persisted** across sessions.  It will be discarded when the
            ``StoreGate`` instance is closed.
        """
        self._copy_between_backends(
            var_name=var_name,
            phase=phase,
            src_backend='zarr',
            dst_backend='numpy',
            output_var_name=output_var_name,
            dst_label='memory',
        )

    def _copy_between_backends(
        self,
        *,
        var_name: str,
        phase: str,
        src_backend: str,
        dst_backend: str,
        output_var_name: str | None,
        dst_label: str,
    ) -> None:
        """Copy a variable between backends with shared validation and checks."""
        _validate_var_name(var_name)
        _validate_phase(phase)
        if output_var_name is not None:
            _validate_var_name(output_var_name)
        output_var_name = output_var_name or var_name
        data_id = self._require_current_data_id()

        logger.debug(
            "copy_to_%s: data_id='%s', var='%s' -> '%s', phase='%s'",
            dst_label, data_id, var_name, output_var_name, phase,
        )

        with self.using_backend(dst_backend):
            if output_var_name in self.get_var_names(phase):
                raise ValueError(f'{output_var_name} already exists in {dst_label}. Delete first or use a different output_var_name.')

        try:
            with self.using_backend(src_backend):
                for chunk in self._db.iter_data_chunks(data_id, var_name, phase):
                    with self.using_backend(dst_backend):
                        self.add_data(output_var_name, chunk, phase)
        except Exception:
            with self.using_backend(dst_backend):
                if output_var_name in self.get_var_names(phase):
                    self.delete_data(output_var_name, phase)
            raise

        logger.debug("copy_to_%s: done", dst_label)


    @require_data_id
    def copy_to_storage(self, var_name: str, phase: str, output_var_name: str | None = None) -> None:
        """Copy data from memory to storage."""
        self._copy_between_backends(
            var_name=var_name,
            phase=phase,
            src_backend='numpy',
            dst_backend='zarr',
            output_var_name=output_var_name,
            dst_label='storage',
        )

    @require_data_id
    def snapshot(self, snapshot_name: str) -> None:
        """Save the current zarr-backed state of ``data_id`` as ``snapshot_name``."""
        _validate_snapshot_name(snapshot_name)
        data_id = self._require_current_data_id()
        logger.debug("snapshot: data_id='%s', snapshot_name='%s'", data_id, snapshot_name)
        self._db.snapshot_data_id(data_id, snapshot_name)
        logger.debug("snapshot: done")

    @require_data_id
    def restore(self, snapshot_name: str) -> None:
        """Restore the current ``data_id`` from a previously saved zarr snapshot.

        Note:
            Restore replaces only the zarr backend contents. Any numpy
            (in-memory) backend data for the current ``data_id`` is discarded,
            and the active backend is switched to ``'zarr'`` afterwards.
        """
        _validate_snapshot_name(snapshot_name)
        data_id = self._require_current_data_id()
        logger.debug("restore: data_id='%s', snapshot_name='%s'", data_id, snapshot_name)
        self._db.restore_data_id(data_id, snapshot_name)
        self._db.clear_data_id(data_id)
        self._db.set_backend('zarr')
        self._metadata[data_id] = {
            'compiled': {backend: False for backend in _SUPPORTED_BACKENDS},
            'sizes': {backend: {} for backend in _SUPPORTED_BACKENDS},
        }
        self._load_meta(data_id)
        logger.debug("restore: done")


    @require_data_id
    def compile(self, show_info: bool = False, cross_backend: bool = False) -> None:
        """Check if registered data are valid.

        Validates that all variables within each phase have the same number of events.
        Note: consistency across phases (e.g. train vs valid) is intentionally not checked,
        as each phase may have a different number of events by design.

        This method also updates size metadata used by ``len()``. For the zarr
        backend, the compiled flag and phase sizes are persisted in the store
        attrs and restored on reopen. Re-run ``compile()`` after reopening only
        if the store has never been compiled, or if data was added, updated, or
        deleted after the last successful compile(). Metadata for the numpy
        backend is in-memory-only and is not persisted across sessions.

        Args:
            show_info (bool): Print a summary table after compilation.
            cross_backend (bool): When True, also verify that any variable present in
                either backend (zarr or numpy) exists in the other backend and
                has the same number of events, dtype, and per-event shape.
                Raises ValueError if a mismatch is found.
        """

        data_id = self._require_current_data_id()
        backend = self.get_backend()
        logger.debug(
            "compile: data_id='%s', backend='%s', cross_backend=%s",
            data_id, backend, cross_backend,
        )
        next_sizes = {
            phase: self._phase_total_events(self._db.get_metadata(data_id, phase), phase)
            for phase in const.PHASES
        }

        if cross_backend:
            self._check_cross_backend_consistency()

        self._metadata[data_id]['compiled'][backend] = True
        self._metadata[data_id]['sizes'][backend] = next_sizes
        self._save_meta(data_id)
        logger.debug("compile: done, sizes=%s", next_sizes)

        if show_info:
            self.show_info()

    @staticmethod
    def _phase_total_events(metadata: dict[str, Any], phase: str) -> int | None:
        phase_events = [data['total_events'] for data in metadata.values()]
        if len(set(phase_events)) > 1:
            detail = '\n'.join(
                f'  {name}: {info["total_events"]} events' for name, info in metadata.items()
            )
            raise ValueError(f"Inconsistent event counts in '{phase}' phase:\n{detail}")
        return phase_events[0] if phase_events else None

    @staticmethod
    def _cross_backend_metadata_errors(
        zarr_meta: dict[str, Any],
        numpy_meta: dict[str, Any],
        phase: str,
    ) -> list[str]:
        """Return metadata mismatch messages for one phase."""
        errors: list[str] = []

        for var_name in sorted(set(zarr_meta) | set(numpy_meta)):
            if var_name not in zarr_meta:
                errors.append(f"  '{var_name}' in '{phase}': missing in zarr")
                continue
            if var_name not in numpy_meta:
                errors.append(f"  '{var_name}' in '{phase}': missing in numpy")
                continue

            zarr_info = zarr_meta[var_name]
            numpy_info = numpy_meta[var_name]
            diffs: list[str] = []

            if zarr_info['total_events'] != numpy_info['total_events']:
                diffs.append(
                    f"events: zarr={zarr_info['total_events']}, "
                    f"numpy={numpy_info['total_events']}"
                )
            if zarr_info['type'] != numpy_info['type']:
                diffs.append(
                    f"type: zarr={zarr_info['type']}, "
                    f"numpy={numpy_info['type']}"
                )
            if zarr_info['shape'] != numpy_info['shape']:
                diffs.append(
                    f"shape: zarr={zarr_info['shape']}, "
                    f"numpy={numpy_info['shape']}"
                )

            if diffs:
                errors.append(
                    f"  '{var_name}' in '{phase}': " + ', '.join(diffs)
                )

        return errors

    def _check_cross_backend_consistency(self) -> None:
        """Verify that variables present in either backend agree on metadata.

        Called from compile(cross_backend=True). Raises ValueError listing all
        mismatches found across phases before aborting.
        """
        data_id = self._require_current_data_id()
        errors: list[str] = []
        for phase in const.PHASES:
            with self.using_backend('zarr'):
                zarr_meta = self._db.get_metadata(data_id, phase)
            with self.using_backend('numpy'):
                numpy_meta = self._db.get_metadata(data_id, phase)
            errors.extend(
                self._cross_backend_metadata_errors(zarr_meta, numpy_meta, phase)
            )

        if errors:
            raise ValueError(
                "Cross-backend inconsistency detected:\n" + "\n".join(errors)
            )

    @require_data_id
    def verify_backend_data(self, phase: str = 'all') -> None:
        """Verify that zarr and numpy backends contain identical array values.

        This method first checks cross-backend metadata parity (variable names,
        event counts, dtypes, and shapes), then loads matching variables from
        both backends and compares their contents with
        ``np.array_equal(..., equal_nan=True)``.

        Note:
            This is stricter and potentially more expensive than
            ``compile(cross_backend=True)``, which validates metadata only.
        """
        _validate_phase(phase, allow_all=True)
        data_id = self._require_current_data_id()
        errors: list[str] = []

        for iphase in self._target_phases(phase):
            with self.using_backend('zarr'):
                zarr_meta = self._db.get_metadata(data_id, iphase)
            with self.using_backend('numpy'):
                numpy_meta = self._db.get_metadata(data_id, iphase)

            phase_errors = self._cross_backend_metadata_errors(
                zarr_meta,
                numpy_meta,
                iphase,
            )
            errors.extend(phase_errors)
            if phase_errors:
                continue

            for var_name in sorted(zarr_meta):
                with self.using_backend('zarr'):
                    zarr_data = self.get_data(var_name, iphase)
                with self.using_backend('numpy'):
                    numpy_data = self.get_data(var_name, iphase)

                if not np.array_equal(zarr_data, numpy_data, equal_nan=True):
                    errors.append(
                        f"  '{var_name}' in '{iphase}': data values differ"
                    )

        if errors:
            raise ValueError(
                "Cross-backend data mismatch detected:\n" + "\n".join(errors)
            )


    def _load_meta(self, data_id: str) -> None:
        """Restore compiled state and sizes from zarr attrs (zarr backend only).

        Saved zarr metadata is treated as advisory. If the underlying zarr arrays
        were modified outside StoreGate after the last successful compile(), the
        persisted compiled flag and phase sizes may be stale. In that case we
        invalidate the restored compiled state so callers are forced to
        recompile before using ``len()``.
        """
        saved = self._db.load_meta_attrs(data_id)
        if not saved:
            return
        meta = self._metadata[data_id]
        meta['compiled']['zarr'] = saved.get('compiled', {}).get('zarr', False)
        meta['sizes']['zarr'] = dict(saved.get('sizes', {}).get('zarr', {}))

        if not meta['compiled']['zarr']:
            return

        try:
            actual_sizes = self._current_zarr_sizes(data_id)
        except ValueError:
            actual_sizes = None

        if actual_sizes != meta['sizes']['zarr']:
            logger.warning(
                f"Invalidating stale compiled zarr metadata for data_id='{data_id}'. "
                'Call compile() again before using len().'
            )
            meta['compiled']['zarr'] = False
            meta['sizes']['zarr'] = {}
            self._save_meta(data_id)

    def _current_zarr_sizes(self, data_id: str) -> dict[str, int | None]:
        """Return phase sizes derived from the current zarr arrays."""
        with self.using_backend('zarr'):
            return {
                phase: self._phase_total_events(self._db.get_metadata(data_id, phase), phase)
                for phase in const.PHASES
            }

    def _save_meta(self, data_id: str) -> None:
        """Persist compiled state and sizes to zarr attrs (zarr backend only)."""
        meta = self._metadata[data_id]
        self._db.save_meta_attrs(data_id, {
            'compiled': {'zarr': meta['compiled']['zarr']},
            'sizes':    {'zarr': meta['sizes']['zarr']},
        })

    def _invalidate_compiled(self, phase: str) -> None:
        """Mark the current backend as not compiled and clear stale size entries."""
        data_id = self._require_current_data_id()
        backend = self.get_backend()
        meta = self._metadata[data_id]
        meta['compiled'][backend] = False
        sizes = meta['sizes'][backend]
        if phase == 'all':
            for iphase in const.PHASES:
                sizes.pop(iphase, None)
        else:
            sizes.pop(phase, None)
        if backend == 'zarr':
            self._save_meta(data_id)

    @require_data_id
    def show_info(self) -> None:
        """Show information currently registered in storegate."""
        data_id = self._require_current_data_id()
        is_compiled = self._metadata[data_id]['compiled'][self.get_backend()]
        header = f'StoreGate data_id : {data_id}, compiled : {is_compiled}'

        names = ['phase', 'backend', 'var_name', 'var_type', 'total_events', 'var_shape']

        table_data: list[list[str] | str] = []
        for phase in const.PHASES:
            metadata = self._db.get_metadata(data_id, phase)
            if not metadata:
                continue

            for var_name, data in metadata.items():
                backend = data['backend']
                dtype = data['type']
                total_events = str(data["total_events"])
                shape = f'{data["shape"]}'

                table_data.append([phase, backend, var_name, dtype, total_events, shape])
            table_data.append('-')

        logger.table(names, table_data, header=header)
