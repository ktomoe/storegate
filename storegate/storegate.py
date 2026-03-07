"""StoreGate module."""
from __future__ import annotations

import functools
import re
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Any, TypeVar

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
        data_id = self._storegate._data_id
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


def _validate_phase(phase: str, allow_all: bool = False) -> None:
    """Raise ValueError if phase is not a valid phase name."""
    valid = const.PHASES + ('all',) if allow_all else const.PHASES
    if phase not in valid:
        raise ValueError(f"Invalid phase '{phase}'. Must be one of {valid}.")


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


class StoreGate:
    """Data management class."""

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
            In-memory metadata (compiled flags, phase sizes) is **not** persisted to
            disk.  When reopening an existing store (``mode='r'`` or ``mode='a'``),
            the metadata starts empty, so ``len(sg[phase])`` will raise until
            ``compile()`` is called.

        Examples:
            Write, then reopen and read::

                sg = StoreGate(output_dir='./store', mode='w', data_id='exp01')
                sg.add_data('x', x_train, phase='train')

                # Reopen in read-only mode
                sg = StoreGate(output_dir='./store', mode='r', data_id='exp01')
                sg.compile()          # required to enable len() and size queries
                x = sg.get_data('x', phase='train')
        """

        _validate_output_dir(output_dir, mode)
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
    ) -> bool:
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
        numpy_chunks = self._db._db['numpy']._chunks
        for data_id, phases in numpy_chunks.items():
            for phase, vars_ in phases.items():
                if vars_:
                    var_list = ', '.join(vars_.keys())
                    logger.debug(
                        f"close(): discarding numpy data —"
                        f" data_id='{data_id}', phase='{phase}', vars=[{var_list}]"
                    )
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

    def set_data_id(self, data_id: str) -> None:
        """Set the default ``data_id`` and initialize the zarr.

        Note:
            Compiled state and phase sizes are automatically restored from the
            zarr store if ``compile()`` was previously called and the store was
            saved.  Re-calling ``compile()`` is only necessary when new data has
            been added since the last ``compile()``.
        """
        _validate_data_id(data_id)
        self._data_id = data_id
        self._db.initialize(data_id)

        if data_id not in self._metadata:
            self._metadata[self._data_id] = {'compiled': {'zarr': False, 'numpy': False},
                                             'sizes': {'zarr': {}, 'numpy': {}}}
            self._load_meta(data_id)


    @require_data_id
    def set_backend(self, backend: str) -> None:
        """Set backend mode of hybrid architecture."""
        if backend not in ['numpy', 'zarr']:
            raise ValueError(f'Unsupported backend: "{backend}". Use "numpy" or "zarr".')

        self._db.set_backend(backend)


    @require_data_id
    def get_backend(self) -> str:
        return self._db.get_backend()

    @require_data_id
    @contextmanager
    def using_backend(self, backend: str) -> Generator[StoreGate, None, None]:
        """Context manager that temporarily switches to ``backend``, restoring the original on exit."""
        if backend not in ['numpy', 'zarr']:
            raise ValueError(f'Unsupported backend: "{backend}". Use "numpy" or "zarr".')
        old = self.get_backend()
        self.set_backend(backend)
        try:
            yield self
        finally:
            self.set_backend(old)


    @require_data_id
    def add_data(self, var_name: str, data: Any, phase: str) -> None:
        """Append data for the given variable and phase.

        Args:
            var_name (str): Variable name to register.
            data: Array-like data to append. Converted to ``np.ndarray`` internally.
            phase (str): One of ``'train'``, ``'valid'``, or ``'test'``.

        Note:
            **dtype handling differs by backend:**

            - **zarr** (disk): the array dtype is fixed at creation time.
              Appending data whose dtype would require promotion raises
              ``ValueError``.  Cast your data explicitly before calling
              ``add_data``, e.g. ``data.astype(existing_dtype)``.
            - **numpy** (memory): dtype promotion is performed automatically
              (e.g. ``int32`` + ``float64`` → ``float64``) and a warning is
              logged.  No data is lost.
        """
        _validate_var_name(var_name)
        _validate_phase(phase)
        data = np.asarray(data)
        if data.ndim == 0:
            raise ValueError(
                'data must be at least 1-dimensional. '
                'Wrap scalar values in a list or array, e.g. np.array([value]).'
            )
        self._db.add_data(self._data_id, var_name, data, phase)
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
        """Update data in storegate with given options."""
        _validate_var_name(var_name)
        _validate_phase(phase)
        data = np.asarray(data)
        self._db.update_data(self._data_id, var_name, data, phase, index)
        self._invalidate_compiled(phase)


    @require_data_id
    def get_data(self, var_name: str, phase: str, index: int | slice | None = None) -> np.ndarray:
        """Retrieve data from storegate with given options."""
        _validate_var_name(var_name)
        _validate_phase(phase)
        return self._db.get_data(self._data_id, var_name, phase, index)


    @require_data_id
    def delete_data(self, var_name: str, phase: str) -> None:
        """Delete data associated with var_names."""
        _validate_var_name(var_name)
        _validate_phase(phase, allow_all=True)
        if phase == 'all':
            for iphase in const.PHASES:
                if var_name in self.get_var_names(iphase):
                    self._db.delete_data(self._data_id, var_name, iphase)

        else:
            self._db.delete_data(self._data_id, var_name, phase)
        self._invalidate_compiled(phase)


    @require_data_id
    def get_var_names(self, phase: str) -> list[str]:
        """Returns registered var_names for given phase."""
        _validate_phase(phase)
        metadata = self._db.get_metadata(self._data_id, phase)
        return list(metadata.keys())


    @require_data_id
    def copy_to_memory(self, var_name: str, phase: str, output_var_name: str | None = None) -> None:
        """Copy data from storage to memory."""
        _validate_var_name(var_name)
        _validate_phase(phase)
        if output_var_name is not None:
            _validate_var_name(output_var_name)
        if output_var_name is None:
            output_var_name = var_name

        with self.using_backend('zarr'):
            tmp_data = self.get_data(var_name, phase=phase)

        with self.using_backend('numpy'):
            if output_var_name in self.get_var_names(phase):
                raise ValueError(f'{output_var_name} already exists in memory. Delete first or use a different output_var_name.')
            self.add_data(output_var_name, tmp_data, phase)


    @require_data_id
    def copy_to_storage(self, var_name: str, phase: str, output_var_name: str | None = None) -> None:
        """Copy data from memory to storage."""
        _validate_var_name(var_name)
        _validate_phase(phase)
        if output_var_name is not None:
            _validate_var_name(output_var_name)
        if output_var_name is None:
            output_var_name = var_name

        with self.using_backend('numpy'):
            tmp_data = self.get_data(var_name, phase=phase)

        with self.using_backend('zarr'):
            if output_var_name in self.get_var_names(phase):
                raise ValueError(f'{output_var_name} already exists in storage. Delete first or use a different output_var_name.')
            self.add_data(output_var_name, tmp_data, phase)


    @require_data_id
    def compile(self, show_info: bool = False, cross_backend: bool = False) -> None:
        """Check if registered data are valid.

        Validates that all variables within each phase have the same number of events.
        Note: consistency across phases (e.g. train vs valid) is intentionally not checked,
        as each phase may have a different number of events by design.

        This method also populates in-memory size metadata (used by ``len()``).
        Because this metadata is not persisted to disk, ``compile()`` must be
        called after reopening an existing store before ``len(sg[phase])`` or
        any other size-dependent operation is used.

        Args:
            show_info (bool): Print a summary table after compilation.
            cross_backend (bool): When True, also verify that any variable present in
                both backends (zarr and numpy) has the same number of events.
                Raises ValueError if a mismatch is found.
        """

        num_events: list[int | None] = []
        for phase in const.PHASES:
            metadata = self._db.get_metadata(self._data_id, phase)

            phase_events: list[int] = []
            for data in metadata.values():
                phase_events.append(data['total_events'])

            if len(set(phase_events)) > 1:
                detail = '\n'.join(
                    f'  {k}: {v["total_events"]} events' for k, v in metadata.items()
                )
                raise ValueError(
                    f"Inconsistent event counts in '{phase}' phase:\n{detail}"
                )

            if phase_events:
                num_events.append(phase_events[0])
            else:
                num_events.append(None)

        self._metadata[self._data_id]['compiled'][self.get_backend()] = True

        for phase, events in zip(const.PHASES, num_events):
            self._metadata[self._data_id]['sizes'][self.get_backend()][phase] = events

        if cross_backend:
            self._check_cross_backend_consistency()

        self._save_meta(self._data_id)

        if show_info:
            self.show_info()

    def _check_cross_backend_consistency(self) -> None:
        """Verify that variables present in both backends agree on event counts.

        Called from compile(cross_backend=True). Raises ValueError listing all
        mismatches found across phases before aborting.
        """
        errors: list[str] = []
        for phase in const.PHASES:
            with self.using_backend('zarr'):
                zarr_meta = self._db.get_metadata(self._data_id, phase)
            with self.using_backend('numpy'):
                numpy_meta = self._db.get_metadata(self._data_id, phase)

            for var_name in sorted(set(zarr_meta) & set(numpy_meta)):
                z = zarr_meta[var_name]['total_events']
                n = numpy_meta[var_name]['total_events']
                if z != n:
                    errors.append(
                        f"  '{var_name}' in '{phase}': zarr={z} events, numpy={n} events"
                    )

        if errors:
            raise ValueError(
                "Cross-backend inconsistency detected:\n" + "\n".join(errors)
            )


    def _load_meta(self, data_id: str) -> None:
        """Restore compiled state and sizes from zarr attrs (zarr backend only)."""
        saved = self._db.load_meta_attrs(data_id)
        if not saved:
            return
        meta = self._metadata[data_id]
        meta['compiled']['zarr'] = saved.get('compiled', {}).get('zarr', False)
        meta['sizes']['zarr'] = dict(saved.get('sizes', {}).get('zarr', {}))

    def _save_meta(self, data_id: str) -> None:
        """Persist compiled state and sizes to zarr attrs (zarr backend only)."""
        meta = self._metadata[data_id]
        self._db.save_meta_attrs(data_id, {
            'compiled': {'zarr': meta['compiled']['zarr']},
            'sizes':    {'zarr': meta['sizes']['zarr']},
        })

    def _invalidate_compiled(self, phase: str) -> None:
        """Mark the current backend as not compiled and clear stale size entries."""
        backend = self.get_backend()
        meta = self._metadata[self._data_id]
        meta['compiled'][backend] = False
        sizes = meta['sizes'][backend]
        if phase == 'all':
            for iphase in const.PHASES:
                sizes.pop(iphase, None)
        else:
            sizes.pop(phase, None)
        if backend == 'zarr':
            self._save_meta(self._data_id)

    @require_data_id
    def show_info(self) -> None:
        """Show information currently registered in storegate."""
        is_compiled = self._metadata[self._data_id]['compiled'][self.get_backend()]
        header = f'StoreGate data_id : {self._data_id}, compiled : {is_compiled}'

        names = ['phase', 'backend', 'var_name', 'var_type', 'total_events', 'var_shape']

        table_data: list[list[str] | str] = []
        for phase in const.PHASES:
            metadata = self._db.get_metadata(self._data_id, phase)
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
