from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, Iterator, Self

import numpy as np

from storegate.database.database import Database, GetIndex, UpdateIndex


class _BackendProxy:
    """Proxy that forwards method calls to one or more backends."""

    _ALL_ALLOWED_METHODS = frozenset(
        {
            "set_data_id",
            "set_phase",
        }
    )

    def __init__(
        self,
        hybrid: "HybridDatabase",
        backends: list[str],
        *,
        write_only: bool = False,
    ) -> None:
        self._hybrid = hybrid
        self._backends = backends
        self._write_only = write_only

    def initialize(self, data_id: str | None = None) -> None:
        if not self._write_only:
            self._invoke("initialize", data_id)
            return

        previous = self._hybrid.get_backend()
        initialized: list[str] = []

        try:
            for backend in self._backends:
                with self._hybrid.using_backend(backend):
                    self._hybrid.initialize(data_id)
                initialized.append(backend)
        except Exception:
            self._close_backends_best_effort(
                initialized,
                annotate_errors=False,
            )
            self._hybrid._backend = previous
            raise

        self._hybrid._backend = previous

    def compile(self, data_id: str) -> Any:
        if not self._write_only:
            return self._invoke("compile", data_id)

        reports, errors = self._compile_backends_best_effort(
            data_id,
            annotate_errors=True,
        )
        if errors:
            raise ExceptionGroup("One or more backends failed to compile.", errors)

        return reports

    def close(self) -> None:
        if not self._write_only:
            self._invoke("close")
            return

        errors = self._close_backends_best_effort(
            self._backends,
            annotate_errors=True,
        )
        if errors:
            raise ExceptionGroup(
                "One or more backends failed to close.",
                errors,
            )

    def set_data_id(self, data_id: str) -> None:
        if not self._write_only:
            self._invoke("set_data_id", data_id)
            return

        self._broadcast_create_with_rollback(
            apply=lambda: self._hybrid.set_data_id(data_id),
            exists=lambda: data_id in self._hybrid.get_data_ids(),
            rollback=lambda: self._hybrid.delete_data_id(data_id),
            action=f"set_data_id(data_id={data_id!r})",
        )

    def set_phase(self, data_id: str, phase: str) -> None:
        if not self._write_only:
            self._invoke("set_phase", data_id, phase)
            return

        self._broadcast_create_with_rollback(
            apply=lambda: self._hybrid.set_phase(data_id, phase),
            exists=lambda: phase in self._hybrid.get_phases(data_id),
            rollback=lambda: self._hybrid.delete_phase(data_id, phase),
            action=f"set_phase(data_id={data_id!r}, phase={phase!r})",
        )

    def __getattr__(self, name: str) -> Any:
        if self._write_only and name not in self._ALL_ALLOWED_METHODS:
            def unsupported_method(*args: Any, **kwargs: Any) -> Any:
                raise RuntimeError(
                    'HybridDatabase["all"] only supports broadcast-safe methods; '
                    f"{name}() is not supported."
                )

            return unsupported_method

        def proxy_method(*args: Any, **kwargs: Any) -> Any:
            return self._invoke(name, *args, **kwargs)

        return proxy_method

    def _invoke(self, name: str, *args: Any, **kwargs: Any) -> Any:
        result: Any = None
        for backend in self._backends:
            with self._hybrid.using_backend(backend):
                result = getattr(self._hybrid, name)(*args, **kwargs)
        return result

    def _broadcast_create_with_rollback(
        self,
        *,
        apply: Callable[[], None],
        exists: Callable[[], bool],
        rollback: Callable[[], None],
        action: str,
    ) -> None:
        previous = self._hybrid.get_backend()
        existed: dict[str, bool] = {}
        created: list[str] = []
        current_backend: str | None = None

        try:
            for backend in self._backends:
                with self._hybrid.using_backend(backend):
                    existed[backend] = exists()

            for backend in self._backends:
                current_backend = backend
                with self._hybrid.using_backend(backend):
                    apply()
                if not existed[backend]:
                    created.append(backend)
        except Exception as exc:
            if (
                current_backend is not None
                and not existed.get(current_backend, True)
            ):
                with self._hybrid.using_backend(current_backend):
                    try:
                        if exists() and current_backend not in created:
                            created.append(current_backend)
                    except Exception:
                        pass

            rollback_errors = self._rollback_backends_best_effort(
                created,
                rollback=rollback,
                action=action,
            )
            if rollback_errors:
                raise ExceptionGroup(
                    f'Broadcast "{action}" failed and rollback was incomplete.',
                    [exc, *rollback_errors],
                ) from None
            raise
        finally:
            self._hybrid._backend = previous

    def _rollback_backends_best_effort(
        self,
        backends: list[str],
        *,
        rollback: Callable[[], None],
        action: str,
    ) -> list[Exception]:
        previous = self._hybrid.get_backend()
        errors: list[Exception] = []

        try:
            for backend in reversed(backends):
                with self._hybrid.using_backend(backend):
                    try:
                        rollback()
                    except Exception as exc:
                        exc.add_note(
                            f'while rolling back broadcast "{action}" on backend:{backend}'
                        )
                        errors.append(exc)
        finally:
            self._hybrid._backend = previous

        return errors

    def _close_backends_best_effort(
        self,
        backends: list[str],
        *,
        annotate_errors: bool,
    ) -> list[Exception]:
        previous = self._hybrid.get_backend()
        errors: list[Exception] = []

        try:
            for backend in backends:
                with self._hybrid.using_backend(backend):
                    try:
                        self._hybrid.close()
                    except Exception as exc:
                        if annotate_errors:
                            exc.add_note(f"while closing backend:{backend}")
                        errors.append(exc)
        finally:
            self._hybrid._backend = previous

        return errors

    def _compile_backends_best_effort(
        self,
        data_id: str,
        *,
        annotate_errors: bool,
    ) -> tuple[dict[str, Any], list[Exception]]:
        previous = self._hybrid.get_backend()
        reports: dict[str, Any] = {}
        errors: list[Exception] = []

        try:
            for backend in self._backends:
                with self._hybrid.using_backend(backend):
                    try:
                        reports[backend] = self._hybrid.compile(data_id)
                    except Exception as exc:
                        if annotate_errors:
                            exc.add_note(f"while compiling backend:{backend}")
                        errors.append(exc)
        finally:
            self._hybrid._backend = previous

        return reports, errors


class HybridDatabase(Database):
    """Hybrid database that combines several backends."""

    def __init__(self, backends: dict[str, Database] | None = None) -> None:
        self._db: dict[str, Database] = {}
        self._backend: str | None = None

        if backends is not None:
            for backend, backend_obj in backends.items():
                self.register_backend(backend, backend_obj)

    ################################################################################
    def register_backend(self, backend: str, backend_obj: Database) -> None:
        if backend == "all":
            raise ValueError('backend name "all" is reserved')
        if backend in self._db:
            raise ValueError(f'backend: {backend} already exists')

        self._db[backend] = backend_obj

    def delete_backend(self, backend: str) -> None:
        backend_obj = self._db[backend]
        backend_obj.close()
        del self._db[backend]

        if self._backend == backend:
            self._backend = None

    def set_backend(self, backend: str) -> None:
        if backend not in self._db:
            raise ValueError(f'backend: {backend} does not exist, register first')
        self._backend = backend

    def get_backend(self) -> str | None:
        return self._backend

    @contextmanager
    def using_backend(self, backend: str) -> Iterator[None]:
        """Temporarily switch the active backend, restoring the previous one on exit."""
        previous = self._backend
        self.set_backend(backend)
        try:
            yield
        finally:
            self._backend = previous

    def __getitem__(self, key: str) -> _BackendProxy:
        """Return a backend proxy for a single backend or for all backends."""
        if key == "all":
            if not self._db:
                raise RuntimeError("No backends are registered.")
            return _BackendProxy(self, list(self._db.keys()), write_only=True)

        if key not in self._db:
            raise KeyError(f"backend: {key} does not exist")

        return _BackendProxy(self, [key])

    def __enter__(self) -> Self:
        if not self._db:
            return self

        self["all"].initialize()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if not self._db:
            return

        if exc_val is None:
            self["all"].close()
            return

        try:
            self["all"].close()
        except ExceptionGroup as close_errors:
            if isinstance(exc_val, BaseException):
                raise BaseExceptionGroup(
                    "An exception occurred in the context body and one or more "
                    "backends failed to close.",
                    [exc_val, *close_errors.exceptions],
                ) from None

    def initialize(self, data_id: str | None = None) -> None:
        self._resolve_backend().initialize(data_id)

    def get_name(self) -> str:
        return "hybrid"

    def is_writable(self) -> bool:
        return self._resolve_backend().is_writable()

    def set_data_id(self, data_id: str) -> None:
        self._resolve_backend().set_data_id(data_id)

    def delete_data_id(self, data_id: str) -> None:
        self._resolve_backend().delete_data_id(data_id)

    def set_phase(self, data_id: str, phase: str) -> None:
        self._resolve_backend().set_phase(data_id, phase)

    def delete_phase(self, data_id: str, phase: str) -> None:
        self._resolve_backend().delete_phase(data_id, phase)

    def clear(self) -> None:
        self._resolve_backend().clear()

    def close(self) -> None:
        self._resolve_backend().close()

    def add_data(
        self, data_id: str, phase: str, var_name: str, data: np.ndarray
    ) -> None:
        self._resolve_backend().add_data(data_id, phase, var_name, data)

    def update_data(
        self,
        data_id: str,
        phase: str,
        var_name: str,
        data: np.ndarray,
        index: UpdateIndex,
    ) -> None:
        self._resolve_backend().update_data(data_id, phase, var_name, data, index)

    def get_data(
        self, data_id: str, phase: str, var_name: str, index: GetIndex = slice(None)
    ) -> np.ndarray:
        return self._resolve_backend().get_data(data_id, phase, var_name, index)

    def delete_data(self, data_id: str, phase: str, var_name: str) -> None:
        self._resolve_backend().delete_data(data_id, phase, var_name)

    def stream_data(
        self, data_id: str, phase: str, var_name: str
    ) -> Iterator[np.ndarray]:
        return self._resolve_backend().stream_data(data_id, phase, var_name)

    def copy_data(
        self, data_id: str, phase: str, var_name: str, output_var_name: str
    ) -> None:
        self._resolve_backend().copy_data(data_id, phase, var_name, output_var_name)

    def rename_data(
        self, data_id: str, phase: str, var_name: str, output_var_name: str
    ) -> None:
        self._resolve_backend().rename_data(data_id, phase, var_name, output_var_name)

    def get_data_ids(self) -> list[str]:
        return self._resolve_backend().get_data_ids()

    def get_phases(self, data_id: str) -> list[str]:
        return self._resolve_backend().get_phases(data_id)

    def get_var_names(self, data_id: str, phase: str) -> list[str]:
        return self._resolve_backend().get_var_names(data_id, phase)

    def get_data_info(self, data_id: str, phase: str, var_name: str) -> dict[str, Any]:
        return self._resolve_backend().get_data_info(data_id, phase, var_name)

    def compile(self, data_id: str) -> dict[str, Any]:
        return self._resolve_backend().compile(data_id)

    def transfer_data(
        self,
        src_backend: str,
        dst_backend: str,
        data_id: str,
        phase: str,
        var_name: str,
        output_var_name: str | None = None,
    ) -> None:
        """Copy a variable from one backend to another.

        Contract:
        - both backends must already be registered and initialized
        - source ``data_id`` / ``phase`` / ``var_name`` must already exist
        - destination backend must support writes
        - destination ``data_id`` must already exist
        - destination ``phase`` is created automatically if missing
        - destination output var_name must not already exist

        Uses ``stream_data()`` to avoid destination-side full-array rewrites.
        Backends that implement chunked streaming, such as ``ZarrDatabase``,
        are transferred incrementally. ``NumpyDatabase`` emits a single
        full-array snapshot, so NumPy-backed sources are not chunked in
        practice.
        Writes are staged into a temporary destination var_name and only made
        visible at the final var_name after a successful rename.
        If ``output_var_name`` is None, the same ``var_name`` is used in the
        destination.
        """
        dst_var = output_var_name if output_var_name is not None else var_name
        self._validate_identifiers(data_id, phase, var_name, dst_var)

        if src_backend not in self._db:
            raise ValueError(f"src_backend: {src_backend} does not exist, register first")
        if dst_backend not in self._db:
            raise ValueError(f"dst_backend: {dst_backend} does not exist, register first")

        src = self._db[src_backend]
        dst = self._db[dst_backend]

        if src is dst and var_name == dst_var:
            return

        self._require_transfer_backend_initialized(src_backend, src)
        self._require_transfer_backend_initialized(dst_backend, dst)
        self._require_transfer_backend_writable(dst_backend, dst)

        if data_id not in dst.get_data_ids():
            raise ValueError(
                f"backend:{dst_backend}| data_id:{data_id} does not exist. "
                "Create it first with set_data_id() before transfer_data()."
            )

        with dst.staged_add(data_id, phase, [dst_var]) as tx:
            chunk_received = False
            for chunk in src.stream_data(data_id, phase, var_name):
                chunk_received = True
                tx.add_data(dst_var, chunk)

            if not chunk_received:
                raise RuntimeError(
                    f"backend:{src_backend}, data_id:{data_id}, phase:{phase}| "
                    f"stream_data() produced no chunks for var_name:{var_name}."
                )

    ################################################################################
    def _require_transfer_backend_initialized(
        self, backend_name: str, db: Database
    ) -> None:
        try:
            db.get_data_ids()
        except RuntimeError as exc:
            raise RuntimeError(
                f"backend:{backend_name} is not initialized. "
                "Call initialize() before transfer_data()."
            ) from exc

    def _require_transfer_backend_writable(
        self, backend_name: str, db: Database
    ) -> None:
        if not db.is_writable():
            raise ValueError(
                f"backend:{backend_name} does not support writes in its current mode. "
                "Open the destination backend in a write-capable mode before transfer_data()."
            )

    def _resolve_backend(self) -> Database:
        """Return the active backend, raising if none is set."""
        backend = self._backend
        if backend is None:
            raise RuntimeError("requires backend to be set. Call set_backend() first.")
        if backend not in self._db:
            raise RuntimeError(f"backend: {backend} was set but has been deleted. "
                                "Call set_backend() with a registered backend.")
        return self._db[backend]
