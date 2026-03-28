from collections.abc import Callable
from functools import wraps
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Literal, cast
import os
import shutil

import numpy as np
import zarr
from zarr.core.group import Group
from zarr.storage import LocalStore

from storegate import utilities as util
from storegate.database.database import Database, GetIndex, UpdateIndex
from storegate.database.staged_add import _is_staged_var_name

_AUTO_CHUNK_BYTES = 16 * 1024 * 1024
_STOREGATE_SCHEMA_KEY = "_storegate_schema"
_STOREGATE_SCHEMA_MARKER = "storegate:v1"


def requires_write_mode(method: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(method)
    def wrapper(self: "ZarrDatabase", *args: Any, **kwargs: Any) -> Any:
        if self._mode == "r":
            raise ValueError("zarr is read only mode.")
        return method(self, *args, **kwargs)

    return wrapper


class ZarrDatabase(Database):
    """Zarr implementation of Database."""

    _ALLOWED_MODES = frozenset({"r", "r+", "a", "w-"})

    def __init__(
        self,
        output_dir: str,
        mode: Literal["r", "r+", "a", "w-"] = "r",
        chunk: int | Literal["auto"] = "auto",
        auto_chunk_bytes: int = _AUTO_CHUNK_BYTES,
        strict_schema: bool = True,
    ) -> None:
        """Create a zarr-backed database handle."""
        if mode not in self._ALLOWED_MODES:
            raise ValueError(
                f"Invalid mode: {mode!r}. Must be one of {sorted(self._ALLOWED_MODES)}."
            )

        if not strict_schema and mode != "r":
            raise ValueError(
                "strict_schema=False is supported only with mode='r'. "
                "Relaxed schema mode is read-only to prevent writes to foreign "
                "or legacy zarr stores."
            )

        self._output_dir = output_dir
        self._mode = mode

        self._chunk: int | Literal["auto"]
        if chunk == "auto":
            self._chunk = chunk
        else:
            self._chunk = util.ensure_positive_int(chunk, "chunk")

        self._auto_chunk_bytes = util.ensure_positive_int(
            auto_chunk_bytes, "auto_chunk_bytes"
        )
        self._strict_schema = strict_schema
        self._db: Group | None = None

    ################################################################################
    def initialize(self, data_id: str | None = None) -> None:
        if data_id is not None:
            self._validate_identifiers(data_id)

        if self._db is None:
            store_path = Path(self._output_dir)
            store_preexisted = store_path.exists() and (
                not store_path.is_dir() or any(store_path.iterdir())
            )

            if self._strict_schema and store_preexisted and self._mode != "w-":
                existing_db = zarr.open_group(self._output_dir, mode="r")
                try:
                    self._validate_schema(existing_db)
                finally:
                    existing_db.store.close()

            db = zarr.open_group(self._output_dir, mode=self._mode)
            if self._mode != "r" and not store_preexisted:
                self._set_marker(db)
            self._db = db

        if data_id is not None:
            if self._mode == "r":
                self._resolve_data_id(data_id)
            else:
                self.set_data_id(data_id)

    def get_name(self) -> str:
        return "zarr"

    def is_writable(self) -> bool:
        return self._mode != "r"

    @requires_write_mode
    def set_data_id(self, data_id: str) -> None:
        self._validate_identifiers(data_id)

        if not self._exist_data_id(data_id):
            db = self._resolve_db()
            group = db.require_group(data_id)
            self._set_marker(group)

    @requires_write_mode
    def delete_data_id(self, data_id: str) -> None:
        self._validate_identifiers(data_id)

        db = self._resolve_db()
        self._resolve_data_id(data_id)
        del db[data_id]

    @requires_write_mode
    def set_phase(self, data_id: str, phase: str) -> None:
        self._validate_identifiers(data_id, phase)

        if not self._exist_phase(data_id, phase):
            did_group = self._resolve_data_id(data_id)
            phase_group = did_group.require_group(phase)
            self._set_marker(phase_group)

    @requires_write_mode
    def delete_phase(self, data_id: str, phase: str) -> None:
        self._validate_identifiers(data_id, phase)

        db = self._resolve_data_id(data_id)
        self._resolve_phase(data_id, phase)
        del db[phase]

    @requires_write_mode
    def clear(self) -> None:
        db = self._resolve_db()
        for data_id in list(db.group_keys()):
            self.delete_data_id(data_id)

    def close(self) -> None:
        db = self._db
        if db is None:
            return

        try:
            db.store.close()
        finally:
            self._db = None

    @requires_write_mode
    def add_data(
        self, data_id: str, phase: str, var_name: str, data: np.ndarray
    ) -> None:
        self._validate_identifiers(data_id, phase, var_name)
        self._validate_data(data)

        db = self._resolve_data_id(data_id)

        if self._exist_var_name(data_id, phase, var_name):
            # append
            self._validate_add_data_append(data_id, phase, var_name, data)
            phase_group = cast(Group, db[phase])
            arr = cast(zarr.Array, phase_group[var_name])
            arr.append(data)
        else:
            # new
            self._validate_add_data_new(data_id, phase, var_name, data)

            chunk = self._resolve_chunk(data)
            self.set_phase(data_id, phase)
            phase_group = cast(Group, db[phase])
            phase_group.create_array(name=var_name, data=data, chunks=chunk)
            self._set_marker(phase_group[var_name])

    @requires_write_mode
    def update_data(
        self,
        data_id: str,
        phase: str,
        var_name: str,
        data: np.ndarray,
        index: UpdateIndex,
    ) -> None:
        self._validate_identifiers(data_id, phase, var_name)
        self._validate_data(data)
        self._validate_update_data_index(index)

        arr = self._resolve_var_name(data_id, phase, var_name)
        if isinstance(index, (int, np.integer)):
            self._validate_update_data_single(data_id, phase, var_name, data)
        else:
            self._validate_update_data_batch(data_id, phase, var_name, data, index)

        arr[cast(int | slice, index)] = data

    def get_data(
        self, data_id: str, phase: str, var_name: str, index: GetIndex = slice(None)
    ) -> np.ndarray:
        self._validate_identifiers(data_id, phase, var_name)
        self._validate_get_data_index(index)

        arr = self._resolve_var_name(data_id, phase, var_name)
        result = arr[cast(int | slice | list[int] | np.ndarray, index)]
        return np.array(result, copy=True)

    @requires_write_mode
    def delete_data(self, data_id: str, phase: str, var_name: str) -> None:
        self._validate_identifiers(data_id, phase, var_name)

        phase_group = self._resolve_phase(data_id, phase)
        self._resolve_var_name(data_id, phase, var_name)
        del phase_group[var_name]

    def stream_data(
        self, data_id: str, phase: str, var_name: str
    ) -> Iterator[np.ndarray]:
        self._validate_identifiers(data_id, phase, var_name)

        arr = self._resolve_var_name(data_id, phase, var_name)
        chunk_size = arr.chunks[0]

        def iterator() -> Iterator[np.ndarray]:
            if arr.shape[0] == 0:
                if self._db is None:
                    raise RuntimeError("Database was closed while streaming.")
                yield np.array(arr[:0], copy=True)
                return

            for start in range(0, arr.shape[0], chunk_size):
                if self._db is None:
                    raise RuntimeError("Database was closed while streaming.")
                stop = min(start + chunk_size, arr.shape[0])
                yield np.array(arr[start:stop], copy=True)

        return iterator()

    @requires_write_mode
    def copy_data(
        self, data_id: str, phase: str, var_name: str, output_var_name: str
    ) -> None:
        self._validate_identifiers(data_id, phase, var_name, output_var_name)

        phase_group = self._resolve_phase(data_id, phase)
        if var_name not in phase_group.array_keys():
            raise ValueError(
                f"backend:{self.get_name()}, data_id:{data_id}, phase:{phase}| var_name:{var_name} does not exist."
            )

        if var_name == output_var_name:
            return

        if output_var_name in phase_group.array_keys():
            raise ValueError(
                f"backend:{self.get_name()}, data_id:{data_id}, phase:{phase}| var_name:{output_var_name} already exists."
            )

        source_path = self._get_local_array_path(phase_group, var_name)
        dest_path = self._get_local_array_path(phase_group, output_var_name)
        temp_path = self._get_local_copy_temp_path(dest_path)

        if not source_path.exists():  # pragma: no cover
            raise FileNotFoundError(
                f"backend:{self.get_name()}, data_id:{data_id}, phase:{phase}| array path for var_name:{var_name} was not found: {source_path}"
            )
        if dest_path.exists():
            raise ValueError(
                f"backend:{self.get_name()}, data_id:{data_id}, phase:{phase}| var_name:{output_var_name} already exists."
            )

        try:
            shutil.copytree(
                source_path,
                temp_path,
                symlinks=True,
                copy_function=shutil.copy2,
            )
            os.replace(temp_path, dest_path)
        except Exception:
            shutil.rmtree(temp_path, ignore_errors=True)
            raise

        self._fsync_directory_best_effort(dest_path.parent)

    @requires_write_mode
    def rename_data(
        self, data_id: str, phase: str, var_name: str, output_var_name: str
    ) -> None:
        self._validate_identifiers(data_id, phase, var_name, output_var_name)

        phase_group = self._resolve_phase(data_id, phase)
        if var_name not in phase_group.array_keys():
            raise ValueError(
                f"backend:{self.get_name()}, data_id:{data_id}, phase:{phase}| var_name:{var_name} does not exist."
            )

        if var_name == output_var_name:
            return

        if output_var_name in phase_group.array_keys():
            raise ValueError(
                f"backend:{self.get_name()}, data_id:{data_id}, phase:{phase}| var_name:{output_var_name} already exists."
            )

        source_path = self._get_local_array_path(phase_group, var_name)
        dest_path = self._get_local_array_path(phase_group, output_var_name)

        if not source_path.exists():  # pragma: no cover
            raise FileNotFoundError(
                f"backend:{self.get_name()}, data_id:{data_id}, phase:{phase}| array path for var_name:{var_name} was not found: {source_path}"
            )
        if dest_path.exists():
            raise ValueError(
                f"backend:{self.get_name()}, data_id:{data_id}, phase:{phase}| var_name:{output_var_name} already exists."
            )

        os.replace(source_path, dest_path)
        self._fsync_directory_best_effort(dest_path.parent)

    def get_data_ids(self) -> list[str]:
        db = self._resolve_db()
        return list(db.group_keys())

    def get_phases(self, data_id: str) -> list[str]:
        self._validate_identifiers(data_id)
        db = self._resolve_data_id(data_id)
        return list(cast(Group, db).group_keys())

    def get_var_names(self, data_id: str, phase: str) -> list[str]:
        self._validate_identifiers(data_id, phase)
        db = self._resolve_phase(data_id, phase)
        return list(cast(Group, db).array_keys())

    def get_data_info(self, data_id: str, phase: str, var_name: str) -> dict[str, Any]:
        self._validate_identifiers(data_id, phase, var_name)
        arr = self._resolve_var_name(data_id, phase, var_name)
        return {
            "dtype": arr.dtype.name,
            "shape": tuple(arr.shape[1:]),
            "num_events": arr.shape[0],
        }

    def compile(self, data_id: str) -> dict[str, Any]:
        self._validate_identifiers(data_id)

        did_group = self._resolve_data_id(data_id)
        phase_reports: dict[str, dict[str, Any]] = {}

        for phase in sorted(did_group.group_keys()):
            phase_group = cast(Group, did_group[phase])
            vars_report: dict[str, int] = {}

            for var_name in sorted(phase_group.array_keys()):
                if _is_staged_var_name(var_name):
                    continue
                arr = cast(zarr.Array, phase_group[var_name])
                vars_report[var_name] = int(arr.shape[0])

            event_counts = set(vars_report.values())
            phase_is_compiled = bool(vars_report) and len(event_counts) == 1
            phase_num_events = next(iter(event_counts)) if phase_is_compiled else None

            phase_reports[phase] = {
                "is_compiled": phase_is_compiled,
                "num_events": phase_num_events,
                "vars": vars_report,
            }

        return {
            "is_compiled": bool(phase_reports)
            and all(report["is_compiled"] for report in phase_reports.values()),
            "data_id": data_id,
            "phases": phase_reports,
        }

    ################################################################################
    def _validate_schema(self, db: Group) -> None:
        self._validate_marker(db, "root")

        root_members = db.members(max_depth=0)
        unexpected = [
            name for name, member in root_members if not isinstance(member, Group)
        ]
        if unexpected:
            raise ValueError(
                f"Root contains unexpected arrays: {unexpected}. "
                "Only data_id groups are allowed at root level."
            )

        for data_id, did_member in root_members:
            if not isinstance(did_member, Group):  # pragma: no cover
                continue

            util.ensure_str(data_id, "data_id")
            did_group = did_member
            self._validate_marker(did_group, f"data_id:{data_id}")

            did_members = did_group.members(max_depth=0)
            unexpected = [
                name for name, member in did_members if not isinstance(member, Group)
            ]
            if unexpected:
                raise ValueError(
                    f"data_id:{data_id} contains unexpected arrays: {unexpected}. "
                    "Only phase groups are allowed under a data_id."
                )

            for phase, phase_member in did_members:
                if not isinstance(phase_member, Group):  # pragma: no cover
                    continue

                util.ensure_str(phase, "phase")
                ph_group = phase_member
                self._validate_marker(ph_group, f"data_id:{data_id}, phase:{phase}")

                phase_members = ph_group.members(max_depth=0)
                unexpected = [
                    name for name, member in phase_members if isinstance(member, Group)
                ]
                if unexpected:
                    raise ValueError(
                        f"data_id:{data_id}, phase:{phase} contains unexpected groups: {unexpected}. "
                        "Only arrays are allowed under a phase."
                    )

                for var_name, var_member in phase_members:
                    if isinstance(var_member, Group):  # pragma: no cover
                        continue

                    util.ensure_str(var_name, "var_name")
                    arr = cast(zarr.Array, var_member)
                    self._validate_marker(
                        arr,
                        f"data_id:{data_id}, phase:{phase}, var_name:{var_name}",
                    )
                    if arr.ndim < 1:
                        raise ValueError(
                            f"data_id:{data_id}, phase:{phase}, var_name:{var_name} "
                            "must have ndim >= 1."
                        )

    def _set_marker(self, db: Group | zarr.Array) -> None:
        marker = db.attrs.get(_STOREGATE_SCHEMA_KEY)
        if marker is None:
            db.attrs[_STOREGATE_SCHEMA_KEY] = _STOREGATE_SCHEMA_MARKER

    def _validate_marker(self, db: Group | zarr.Array, name: str) -> None:
        marker = db.attrs.get(_STOREGATE_SCHEMA_KEY)

        if marker is None:
            raise ValueError(
                f"{name} is a zarr store but is missing the storegate schema marker."
            )

        if marker != _STOREGATE_SCHEMA_MARKER:
            raise ValueError(
                f"{name} has unsupported {_STOREGATE_SCHEMA_KEY}={marker!r}."
            )

    def _resolve_db(self) -> Group:
        """Return database if exists"""
        db = self._db
        if db is None:
            raise RuntimeError("Database is not initialized. Call initialize() first.")
        return db

    def _resolve_data_id(self, data_id: str) -> Group:
        """Return data_id database if exists"""
        db = self._resolve_db()
        member = db.get(data_id)
        if member is None or not isinstance(member, Group):
            raise ValueError(
                f"backend:{self.get_name()}| data_id:{data_id} does not exist."
            )
        return member

    def _resolve_phase(self, data_id: str, phase: str) -> Group:
        """Return phase database if exists"""
        db = self._resolve_data_id(data_id)
        member = db.get(phase)
        if member is None or not isinstance(member, Group):
            raise ValueError(
                f"backend:{self.get_name()}, data_id:{data_id}| phase:{phase} does not exist."
            )
        return member

    def _resolve_var_name(self, data_id: str, phase: str, var_name: str) -> zarr.Array:
        """Return arr if exists"""
        db = self._resolve_phase(data_id, phase)
        member = db.get(var_name)
        if member is None or isinstance(member, Group):
            raise ValueError(
                f"backend:{self.get_name()}, data_id:{data_id}, phase:{phase}| var_name:{var_name} does not exist."
            )
        return cast(zarr.Array, member)

    def _exist_db(self) -> bool:
        try:
            self._resolve_db()
            return True
        except RuntimeError:
            return False

    def _exist_data_id(self, data_id: str) -> bool:
        try:
            self._resolve_data_id(data_id)
            return True
        except (RuntimeError, ValueError):
            return False

    def _exist_phase(self, data_id: str, phase: str) -> bool:
        try:
            self._resolve_phase(data_id, phase)
            return True
        except (RuntimeError, ValueError):
            return False

    def _exist_var_name(self, data_id: str, phase: str, var_name: str) -> bool:
        try:
            self._resolve_var_name(data_id, phase, var_name)
            return True
        except (RuntimeError, ValueError):
            return False

    def _resolve_chunk(self, data: np.ndarray) -> tuple[int, ...]:
        """Return a chunk shape that only splits the event axis."""
        if self._chunk == "auto":
            if data.ndim > 1:
                sample_items = int(np.prod(data.shape[1:], dtype=np.int64))
            else:
                sample_items = 1
            sample_bytes = sample_items * data.dtype.itemsize
            if sample_bytes == 0:
                # Zero-width feature axes have no per-event payload bytes.
                chunk_events = max(1, data.shape[0])
            else:
                chunk_events = max(1, self._auto_chunk_bytes // sample_bytes)

        else:
            chunk_events = self._chunk

        return (chunk_events,) + tuple(data.shape[1:])

    def _get_local_array_path(self, phase_group: Group, var_name: str) -> Path:
        """Resolve the filesystem path for one array inside a LocalStore."""
        store = phase_group.store
        if not isinstance(store, LocalStore):
            raise TypeError("Atomic rename requires a zarr LocalStore backend.")

        return store.root.joinpath(*PurePosixPath(phase_group.path).parts, var_name)

    @staticmethod
    def _get_local_copy_temp_path(dest_path: Path) -> Path:
        """Return a non-existing temporary sibling path for filesystem copies."""
        candidate = dest_path.with_name(f".{dest_path.name}.storegate-copy-tmp")
        suffix = 0

        while candidate.exists() or candidate.is_symlink():
            suffix += 1
            candidate = dest_path.with_name(
                f".{dest_path.name}.storegate-copy-tmp-{suffix}"
            )

        return candidate

    @staticmethod
    def _fsync_directory_best_effort(path: Path) -> None:
        """Best-effort directory fsync after rename on POSIX filesystems.

        This helper must never turn a completed rename into an API-level
        failure. Any ``OSError`` during directory open/fsync is ignored.
        """
        if os.name == "nt":
            return

        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY

        try:
            fd = os.open(path, flags)
        except OSError:
            return

        try:
            try:
                os.fsync(fd)
            except OSError:
                return
        finally:
            os.close(fd)
