import copy
import sys
from contextlib import contextmanager
from typing import Any, Iterator, Self, TextIO, TypeAlias

import numpy as np

from storegate import utilities as util
from storegate.formatters import ShowInfoRow, format_show_info_table
from storegate.database.database import Database, GetIndex, UpdateIndex
from storegate.database.hybrid_database import HybridDatabase
from storegate.database.staged_add import _StagedAddTransaction

CompiledKey: TypeAlias = tuple[str, str]
CompiledReport: TypeAlias = dict[str, Any] | None


class _CompiledReportUnset:
    __slots__ = ()


_COMPILED_REPORT_UNSET = _CompiledReportUnset()


class _AccessContext:
    """Bound backend/data_id context shared by proxy/view objects."""

    __slots__ = ("gate", "backend", "data_id", "compiled_report")

    def __init__(
        self,
        gate: "StoreGate",
        *,
        backend: str | None = None,
        data_id: str | None = None,
        compiled_report: CompiledReport | _CompiledReportUnset = _COMPILED_REPORT_UNSET,
    ) -> None:
        self.gate = gate
        self.backend = backend
        self.data_id = data_id
        self.compiled_report = compiled_report

    def resolve_data_id(self) -> str:
        if self.data_id is not None:
            return self.data_id
        return self.gate._resolve_data_id()

    def resolve_compiled_backend(self) -> str:
        return self.gate._resolve_compiled_backend(self.backend)

    def get_compiled_report(self) -> CompiledReport:
        compiled_report = self.compiled_report
        if not isinstance(compiled_report, _CompiledReportUnset):
            return compiled_report

        backend = self.resolve_compiled_backend()
        data_id = self.resolve_data_id()
        return self.gate._compiled.get(
            self.gate._compiled_key(backend, data_id)
        )

    def dispatch(
        self,
        op: str,
        *args: Any,
        invalidate: bool = False,
    ) -> Any:
        return self.gate._dispatch(
            op,
            *args,
            backend=self.backend,
            data_id=self.data_id,
            invalidate=invalidate,
        )


class _ReadOnlyVarView:
    """Read-only variable view."""

    __slots__ = ("_ctx", "_phase", "_var_name")

    def __init__(
        self,
        ctx: _AccessContext,
        phase: str,
        var_name: str,
    ) -> None:
        self._ctx = ctx
        self._phase = phase
        self._var_name = var_name

    def __getitem__(self, index: GetIndex) -> np.ndarray:
        return self._ctx.dispatch(
            "get_data",
            self._phase,
            self._var_name,
            index,
        )


class _VarView(_ReadOnlyVarView):
    """Mutable variable view."""

    __slots__ = ()

    def __setitem__(self, index: UpdateIndex, data: np.ndarray) -> None:
        self._ctx.dispatch(
            "update_data",
            self._phase,
            self._var_name,
            data,
            index,
            invalidate=True,
        )

    def append(self, data: np.ndarray) -> None:
        self._ctx.dispatch(
            "add_data",
            self._phase,
            self._var_name,
            data,
            invalidate=True,
        )


class _BasePhaseView:
    """Common phase view logic."""

    __slots__ = ("_ctx", "_phase")

    def __init__(
        self,
        ctx: _AccessContext,
        phase: str,
    ) -> None:
        self._ctx = ctx
        self._phase = phase

    def _make_var_view(self, var_name: str) -> _ReadOnlyVarView:
        raise NotImplementedError

    def __getitem__(self, var_name: str) -> _ReadOnlyVarView:
        return self._make_var_view(var_name)

    def __contains__(self, var_name: str) -> bool:
        return var_name in self._ctx.dispatch(
            "get_var_names",
            self._phase,
        )

    def __len__(self) -> int:
        backend = self._ctx.resolve_compiled_backend()
        data_id = self._ctx.resolve_data_id()
        report = self._ctx.get_compiled_report()
        if report is None:
            if self._ctx.data_id is None:
                raise RuntimeError(
                    f"Backend '{backend}' is not compiled. "
                    "Call compile() first."
                )
            raise RuntimeError(
                f"Backend '{backend}' / data_id '{data_id}' "
                "is not compiled. Call compile() first."
            )

        phase_report = report["phases"].get(self._phase)
        if phase_report is None:
            if self._ctx.data_id is None:
                raise RuntimeError(
                    f"Phase '{self._phase}' not found in compiled report "
                    f"for backend '{backend}'."
                )
            raise RuntimeError(
                f"Phase '{self._phase}' not found in compiled report for "
                f"backend '{backend}' / data_id '{data_id}'."
            )

        if not phase_report["is_compiled"]:
            if self._ctx.data_id is None:
                raise RuntimeError(
                    f"Phase '{self._phase}' is not compiled "
                    f"on backend '{backend}': "
                    "event counts are mismatched."
                )
            raise RuntimeError(
                f"Phase '{self._phase}' is not compiled on backend "
                f"'{backend}' / data_id '{data_id}'."
            )

        return phase_report["num_events"]


class _PhaseView(_BasePhaseView):
    """Mutable phase view."""

    __slots__ = ()

    def _make_var_view(self, var_name: str) -> _VarView:
        return _VarView(self._ctx, self._phase, var_name)


class _ReadOnlyPhaseView(_BasePhaseView):
    """Read-only phase view."""

    __slots__ = ()

    def _make_var_view(self, var_name: str) -> _ReadOnlyVarView:
        return _ReadOnlyVarView(self._ctx, self._phase, var_name)


class _BaseBoundView:
    """Common backend/data_id-bound view logic."""

    __slots__ = ("_ctx",)

    def __init__(self, ctx: _AccessContext) -> None:
        self._ctx = ctx

    def _make_phase_view(self, phase: str) -> _BasePhaseView:
        raise NotImplementedError

    def __getitem__(self, phase: str) -> _BasePhaseView:
        return self._make_phase_view(phase)

    def __contains__(self, phase: str) -> bool:
        return phase in self._ctx.dispatch("get_phases")

    def get_data(
        self, phase: str, var_name: str, index: GetIndex = slice(None)
    ) -> np.ndarray:
        return self._ctx.dispatch("get_data", phase, var_name, index)

    def get_phases(self) -> list[str]:
        return self._ctx.dispatch("get_phases")

    def get_var_names(self, phase: str) -> list[str]:
        return self._ctx.dispatch("get_var_names", phase)

    def get_data_info(self, phase: str, var_name: str) -> dict[str, Any]:
        return self._ctx.dispatch("get_data_info", phase, var_name)

    def compile(self) -> dict[str, Any]:
        data_id = self._ctx.resolve_data_id()
        report = self._ctx.dispatch("compile")
        if self._ctx.compiled_report is not _COMPILED_REPORT_UNSET:
            self._ctx.compiled_report = copy.deepcopy(report)
        self._ctx.gate._compiled[
            self._ctx.gate._compiled_key(
                self._ctx.resolve_compiled_backend(),
                data_id,
            )
        ] = report
        return report


class _BackendView(_BaseBoundView):
    """Backend-pinned mutable view returned by ``sg.<backend_name>``."""

    __slots__ = ()

    def __init__(self, gate: "StoreGate", backend: str) -> None:
        super().__init__(_AccessContext(gate, backend=backend))

    def _make_phase_view(self, phase: str) -> _PhaseView:
        return _PhaseView(self._ctx, phase)

    ############################################################################
    def add_data(self, phase: str, var_name: str, data: np.ndarray) -> None:
        self._ctx.dispatch("add_data", phase, var_name, data, invalidate=True)

    def update_data(
        self, phase: str, var_name: str, data: np.ndarray, index: UpdateIndex
    ) -> None:
        self._ctx.dispatch(
            "update_data", phase, var_name, data, index, invalidate=True
        )

    def delete_data(self, phase: str, var_name: str) -> None:
        self._ctx.dispatch("delete_data", phase, var_name, invalidate=True)

    def stream_data(self, phase: str, var_name: str) -> Iterator[np.ndarray]:
        return self._ctx.dispatch("stream_data", phase, var_name)

    def copy_data(
        self, phase: str, var_name: str, output_var_name: str
    ) -> None:
        self._ctx.dispatch(
            "copy_data", phase, var_name, output_var_name, invalidate=True
        )

    def rename_data(
        self, phase: str, var_name: str, output_var_name: str
    ) -> None:
        self._ctx.dispatch(
            "rename_data", phase, var_name, output_var_name, invalidate=True
        )

    def staged_add(
        self, phase: str, var_names: list[str]
    ) -> _StagedAddTransaction:
        backend = self._ctx.backend
        assert backend is not None
        self._ctx.gate._invalidate_compiled(backend)
        db = self._ctx.gate._hybrid._db[backend]
        return db.staged_add(self._ctx.resolve_data_id(), phase, var_names)

    ############################################################################
    def set_phase(self, phase: str) -> None:
        self._ctx.dispatch("set_phase", phase, invalidate=True)

    def delete_phase(self, phase: str) -> None:
        self._ctx.dispatch("delete_phase", phase, invalidate=True)

    def is_writable(self) -> bool:
        backend = self._ctx.backend
        assert backend is not None
        with self._ctx.gate._hybrid.using_backend(backend):
            return self._ctx.gate._hybrid.is_writable()

    def show_info(
        self, file: TextIO | None = None, *, compile: bool = False
    ) -> str:
        """Print a formatted table for this backend."""
        if file is None:
            file = sys.stdout
        backend = self._ctx.backend
        assert backend is not None
        with self._ctx.gate._hybrid.using_backend(backend):
            return self._ctx.gate.show_info(file=file, compile=compile)


class StoreGateReadView(_BaseBoundView):
    """Read-only view pinned to one backend/data_id pair."""

    __slots__ = ()

    def __init__(
        self,
        gate: "StoreGate",
        backend: str,
        data_id: str,
        *,
        compiled_report: CompiledReport | _CompiledReportUnset = _COMPILED_REPORT_UNSET,
    ) -> None:
        super().__init__(
            _AccessContext(
                gate,
                backend=backend,
                data_id=data_id,
                compiled_report=compiled_report,
            )
        )

    def _make_phase_view(self, phase: str) -> _ReadOnlyPhaseView:
        return _ReadOnlyPhaseView(self._ctx, phase)

    def get_backend(self) -> str:
        return self._ctx.resolve_compiled_backend()

    def get_data_id(self) -> str:
        return self._ctx.resolve_data_id()

    def is_writable(self) -> bool:
        return False


class _AllBackendView:
    """Proxy returned by ``sg.all``, broadcasting operations to every backend."""

    __slots__ = ("_gate",)

    def __init__(self, gate: "StoreGate") -> None:
        self._gate = gate

    def initialize(self, data_id: str | None = None) -> None:
        self._gate._initialize_with_data_id(data_id)

    def close(self) -> None:
        try:
            if not self._gate._hybrid._db:
                return

            self._gate._hybrid["all"].close()
        finally:
            self._gate._compiled.clear()

    def set_data_id(self, data_id: str) -> None:
        self._gate._hybrid["all"].set_data_id(data_id)
        self._gate._data_id = data_id

    def set_phase(self, phase: str) -> None:
        data_id = self._gate._resolve_data_id()
        self._gate._invalidate_compiled_data_id(data_id)
        self._gate._hybrid["all"].set_phase(
            data_id, phase
        )

    def compile(self) -> dict[str, Any]:
        data_id = self._gate._resolve_data_id()
        reports = self._gate._hybrid["all"].compile(data_id)
        for backend, report in reports.items():
            self._gate._compiled[self._gate._compiled_key(backend, data_id)] = report
        return reports


class StoreGate:
    """High-level facade for deep-learning data management.

    Wraps ``HybridDatabase`` and binds a single ``data_id`` so that
    callers can use ``sg['phase']['var_name'][index]`` syntax instead
    of passing ``data_id`` explicitly to every call.
    """

    def __init__(
        self,
        backends: dict[str, Database] | None = None,
        backend: str | None = None,
        data_id: str | None = None,
    ) -> None:
        self._hybrid = HybridDatabase()
        self._data_id: str | None = None
        self._compiled: dict[CompiledKey, dict[str, Any]] = {}

        if data_id is not None:
            util.ensure_str(data_id, "data_id")
            self._data_id = data_id

        registered_backends: list[str] = []
        try:
            if backends is not None:
                for backend_name, backend_obj in backends.items():
                    self.register_backend(backend_name, backend_obj)
                    registered_backends.append(backend_name)

            if backend is not None:
                self._hybrid.set_backend(backend)
        except Exception:
            for registered_backend in reversed(registered_backends):
                try:
                    self._hybrid.delete_backend(registered_backend)
                except Exception:
                    pass
            raise

    ############################################################################
    #  Context manager
    ############################################################################
    def __enter__(self) -> Self:
        self.initialize()
        return self

    def __exit__(
        self, exc_type: object, exc_val: object, exc_tb: object
    ) -> None:
        try:
            self._hybrid.__exit__(exc_type, exc_val, exc_tb)
        finally:
            self._compiled.clear()

    def initialize(self, data_id: str | None = None) -> None:
        self._initialize_with_data_id(data_id)

    def close(self) -> None:
        try:
            if not self._hybrid._db:
                return

            self._hybrid["all"].close()
        finally:
            self._compiled.clear()

    ############################################################################
    #  Subscript / attribute access
    ############################################################################
    def __getitem__(self, phase: str) -> _PhaseView:
        return _PhaseView(_AccessContext(self), phase)

    def __contains__(self, phase: str) -> bool:
        return phase in self._dispatch("get_phases")

    def __getattr__(self, name: str) -> _BackendView | _AllBackendView:
        if name.startswith("_"):
            raise AttributeError(name)

        if name == "all":
            return _AllBackendView(self)

        try:
            return self.get_backend_view(name)
        except ValueError:
            pass

        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}' "
            f"and no backend named '{name}' is registered"
        )

    def get_backend_view(self, backend: str) -> _BackendView:
        """Return a backend-pinned view with a precise static type.

        Prefer this explicit accessor to dynamic attribute access when using
        static type checkers.
        """
        util.ensure_str(backend, "backend")
        if backend == "all":
            raise ValueError(
                'backend name "all" refers to the broadcast view. '
                "Use sg.all for broadcast operations."
            )
        if backend not in self._hybrid._db:
            raise ValueError(
                f"backend: {backend} does not exist, register first"
            )
        return _BackendView(self, backend)

    ############################################################################
    #  Backend management
    ############################################################################
    def register_backend(self, backend: str, backend_obj: Database) -> None:
        if backend in self._reserved_backend_names():
            raise ValueError(
                f'backend name "{backend}" is reserved by StoreGate and cannot be registered'
            )
        if backend in self._hybrid._db:
            raise ValueError(f"backend: {backend} already exists")

        try:
            backend_obj.initialize(data_id=self._data_id)
            self._hybrid.register_backend(backend, backend_obj)
        except Exception:
            try:
                backend_obj.close()
            except Exception:
                pass
            raise

    @classmethod
    def _reserved_backend_names(cls) -> set[str]:
        reserved = {name for name in dir(cls) if not name.startswith("_")}
        reserved.add("all")
        return reserved

    def delete_backend(self, backend: str) -> None:
        self._invalidate_compiled_backend(backend)
        self._hybrid.delete_backend(backend)

    def set_backend(self, backend: str) -> None:
        self._hybrid.set_backend(backend)

    def get_backend(self) -> str | None:
        return self._hybrid.get_backend()

    @contextmanager
    def using_backend(self, backend: str) -> Iterator[None]:
        with self._hybrid.using_backend(backend):
            yield

    ############################################################################
    #  data_id management
    ############################################################################
    def set_data_id(self, data_id: str) -> None:
        self._hybrid.set_data_id(data_id)
        self._data_id = data_id

    def delete_data_id(self, data_id: str) -> None:
        self._dispatch(
            "delete_data_id",
            data_id=data_id,
            invalidate=True,
        )
        if self._data_id == data_id:
            self._data_id = None

    def get_data_id(self) -> str | None:
        return self._data_id

    def get_data_ids(self) -> list[str]:
        return self._hybrid.get_data_ids()

    ############################################################################
    #  Phase management
    ############################################################################
    def set_phase(self, phase: str) -> None:
        self._dispatch("set_phase", phase, invalidate=True)

    def delete_phase(self, phase: str) -> None:
        self._dispatch("delete_phase", phase, invalidate=True)

    def get_phases(self) -> list[str]:
        return self._dispatch("get_phases")

    ############################################################################
    #  Data operations
    ############################################################################
    def add_data(self, phase: str, var_name: str, data: np.ndarray) -> None:
        self._dispatch("add_data", phase, var_name, data, invalidate=True)

    def update_data(
        self, phase: str, var_name: str, data: np.ndarray, index: UpdateIndex
    ) -> None:
        self._dispatch(
            "update_data",
            phase,
            var_name,
            data,
            index,
            invalidate=True,
        )

    def get_data(
        self, phase: str, var_name: str, index: GetIndex = slice(None)
    ) -> np.ndarray:
        return self._dispatch("get_data", phase, var_name, index)

    def delete_data(self, phase: str, var_name: str) -> None:
        self._dispatch("delete_data", phase, var_name, invalidate=True)

    def stream_data(self, phase: str, var_name: str) -> Iterator[np.ndarray]:
        return self._dispatch("stream_data", phase, var_name)

    def copy_data(
        self, phase: str, var_name: str, output_var_name: str
    ) -> None:
        self._dispatch(
            "copy_data",
            phase,
            var_name,
            output_var_name,
            invalidate=True,
        )

    def rename_data(
        self, phase: str, var_name: str, output_var_name: str
    ) -> None:
        self._dispatch(
            "rename_data",
            phase,
            var_name,
            output_var_name,
            invalidate=True,
        )

    def staged_add(
        self, phase: str, var_names: list[str]
    ) -> _StagedAddTransaction:
        self._invalidate_compiled()
        return self._hybrid._resolve_backend().staged_add(
            self._resolve_data_id(), phase, var_names
        )

    ############################################################################
    #  Query
    ############################################################################
    def get_var_names(self, phase: str) -> list[str]:
        return self._dispatch("get_var_names", phase)

    def get_data_info(
        self, phase: str, var_name: str
    ) -> dict[str, Any]:
        return self._dispatch("get_data_info", phase, var_name)

    def compile(self) -> dict[str, Any]:
        data_id = self._resolve_data_id()
        report = self._dispatch("compile", data_id=data_id)
        backend = self._hybrid.get_backend()
        if backend is not None:
            self._compiled[self._compiled_key(backend, data_id)] = report
        return report

    def is_writable(self) -> bool:
        return self._hybrid.is_writable()

    def show_info(
        self, file: TextIO | None = None, *, compile: bool = False
    ) -> str:
        """Print a formatted table of phases, variables, and metadata.

        Returns the table string and also writes it to *file*
        (defaults to ``sys.stdout``).

        When *compile* is ``True``, ``compile()`` is called internally to
        determine alignment status.  The default (``False``) uses the
        cached compile report if available; otherwise the status is
        shown as "Not Compiled".
        """
        if file is None:
            file = sys.stdout

        data_id = self._resolve_data_id()

        if compile:
            self.compile()

        backend = self._hybrid.get_backend()
        if backend is not None:
            cached = self._compiled.get(self._compiled_key(backend, data_id))
        else:
            cached = None
        all_compiled = cached["is_compiled"] if cached is not None else False
        backend_label = self._hybrid.get_backend()

        rows: list[ShowInfoRow] = []
        for phase in sorted(self._hybrid.get_phases(data_id)):
            var_names = sorted(
                self._hybrid.get_var_names(data_id, phase)
            )
            if not var_names:
                rows.append((phase, "(empty)", "", "", ""))
                continue
            for var_name in var_names:
                info = self._hybrid.get_data_info(data_id, phase, var_name)
                rows.append((
                    phase,
                    var_name,
                    str(info["num_events"]),
                    info["dtype"],
                    str(info["shape"]),
                ))

        table = format_show_info_table(
            data_id=data_id,
            backend_label=backend_label,
            all_compiled=all_compiled,
            rows=rows,
        )
        file.write(table + "\n")
        return table

    ############################################################################
    #  Other
    ############################################################################
    def clear(self) -> None:
        current_backend = self._hybrid.get_backend()
        self._invalidate_compiled_backend(current_backend)
        self._hybrid.clear()
        self._data_id = None

    def transfer_data(
        self,
        src_backend: str,
        dst_backend: str,
        phase: str,
        var_name: str,
        output_var_name: str | None = None,
    ) -> None:
        data_id = self._resolve_data_id()
        self._invalidate_compiled(dst_backend, data_id)
        self._hybrid.transfer_data(
            src_backend,
            dst_backend,
            data_id,
            phase,
            var_name,
            output_var_name,
        )

    def pin(
        self,
        *,
        backend: str | None = None,
        data_id: str | None = None,
    ) -> StoreGateReadView:
        target_backend = self._resolve_compiled_backend(backend)
        if data_id is None:
            target_data_id = self._resolve_data_id()
        else:
            target_data_id = util.ensure_str(data_id, "data_id")

        with self._hybrid.using_backend(target_backend):
            if target_data_id not in self._hybrid.get_data_ids():
                raise ValueError(
                    f"backend:{target_backend}| data_id:{target_data_id} "
                    "does not exist."
                )

        compiled_report = self._compiled.get(
            self._compiled_key(target_backend, target_data_id)
        )

        return StoreGateReadView(
            self,
            target_backend,
            target_data_id,
            compiled_report=copy.deepcopy(compiled_report),
        )

    ############################################################################
    #  Private
    ############################################################################
    def _initialize_with_data_id(self, data_id: str | None = None) -> None:
        target_data_id = self._data_id
        if data_id is not None:
            util.ensure_str(data_id, "data_id")
            target_data_id = data_id

        if not self._hybrid._db:
            self._data_id = target_data_id
            return

        previous_data_id = self._data_id
        try:
            self._hybrid["all"].initialize(data_id=target_data_id)
        except Exception:
            self._data_id = previous_data_id
            raise

        self._data_id = target_data_id

    def _resolve_data_id(self) -> str:
        if self._data_id is None:
            raise RuntimeError("data_id is not set. Call set_data_id() first.")
        return self._data_id

    def _dispatch(
        self,
        op: str,
        *args: Any,
        backend: str | None = None,
        data_id: str | None = None,
        invalidate: bool = False,
    ) -> Any:
        """Call a HybridDatabase method whose first argument is data_id."""
        target_data_id = self._resolve_data_id() if data_id is None else data_id

        if invalidate:
            self._invalidate_compiled(backend, target_data_id)

        method = getattr(self._hybrid, op)
        if backend is None:
            return method(target_data_id, *args)

        with self._hybrid.using_backend(backend):
            return method(target_data_id, *args)

    def _compiled_key(self, backend: str, data_id: str) -> CompiledKey:
        return (backend, data_id)

    def _invalidate_compiled(
        self, backend: str | None = None, data_id: str | None = None
    ) -> None:
        """Clear the cached compile report for one backend/data_id pair.

        If *backend* or *data_id* is ``None``, the current active value is used.
        """
        target_backend = backend if backend is not None else self._hybrid.get_backend()
        target_data_id = data_id if data_id is not None else self._data_id

        if target_backend is None or target_data_id is None:
            return

        self._compiled.pop(self._compiled_key(target_backend, target_data_id), None)

    def _invalidate_compiled_backend(self, backend: str | None) -> None:
        if backend is None:
            return

        self._compiled = {key: report
                          for key, report in self._compiled.items()
                          if key[0] != backend}

    def _invalidate_compiled_data_id(self, data_id: str | None) -> None:
        if data_id is None:
            return

        self._compiled = {key: report
                          for key, report in self._compiled.items()
                          if key[1] != data_id}

    def _resolve_compiled_backend(self, backend: str | None) -> str:
        """Return the backend name to use for a compiled-report lookup."""
        if backend is not None:
            return backend
        current = self._hybrid.get_backend()
        if current is None:
            raise RuntimeError("No active backend. Call set_backend() first.")
        return current
