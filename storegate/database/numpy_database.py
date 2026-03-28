import copy
from typing import Any, Iterator, cast

import numpy as np

from storegate.database.database import Database, GetIndex, UpdateIndex
from storegate.database.staged_add import _is_staged_var_name


class NumpyDatabase(Database):
    """Numpy implementation of Database."""

    def __init__(self) -> None:
        """Create a numpy-backed database handle."""
        self._db: dict[str, dict[str, dict[str, dict]]] | None = None

    ################################################################################
    def initialize(self, data_id: str | None = None) -> None:
        if data_id is not None:
            self._validate_identifiers(data_id)

        if self._db is None:
            self._db = {}

        if data_id is not None:
            if data_id not in self._db:
                self._db[data_id] = {}

    def get_name(self) -> str:
        return "numpy"

    def is_writable(self) -> bool:
        return True

    def set_data_id(self, data_id: str) -> None:
        self._validate_identifiers(data_id)

        if not self._exist_data_id(data_id):
            db = self._resolve_db()
            db[data_id] = {}

    def delete_data_id(self, data_id: str) -> None:
        self._validate_identifiers(data_id)

        db = self._resolve_db()
        self._resolve_data_id(data_id)
        del db[data_id]

    def set_phase(self, data_id: str, phase: str) -> None:
        self._validate_identifiers(data_id, phase)

        if not self._exist_phase(data_id, phase):
            db = self._resolve_data_id(data_id)
            db[phase] = {}

    def delete_phase(self, data_id: str, phase: str) -> None:
        self._validate_identifiers(data_id, phase)

        db = self._resolve_data_id(data_id)
        self._resolve_phase(data_id, phase)
        del db[phase]

    def clear(self) -> None:
        self._resolve_db()
        self._db = {}

    def close(self) -> None:
        self._db = None

    def add_data(
        self, data_id: str, phase: str, var_name: str, data: np.ndarray
    ) -> None:
        self._validate_identifiers(data_id, phase, var_name)
        self._validate_data(data)

        db = self._resolve_data_id(data_id)

        if self._exist_var_name(data_id, phase, var_name):
            # append
            self._validate_add_data_append(data_id, phase, var_name, data)
            # Keep an owned copy so caller-side mutation cannot corrupt stored data.
            db[phase][var_name]['chunks'].append(data.copy())
            db[phase][var_name]['cache'] = None
        else:
            # new
            self._validate_add_data_new(data_id, phase, var_name, data)
            self.set_phase(data_id, phase)
            db[phase][var_name] = {'chunks': [], 'cache': None}
            db[phase][var_name]['chunks'] = [data.copy()]

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
        entry = self._resolve_var_entry(data_id, phase, var_name)
        entry["chunks"] = [arr]

    def get_data(
        self, data_id: str, phase: str, var_name: str, index: GetIndex = slice(None)
    ) -> np.ndarray:
        self._validate_identifiers(data_id, phase, var_name)
        self._validate_get_data_index(index)

        arr = self._resolve_var_name(data_id, phase, var_name)
        result = arr[cast(int | slice | list[int] | np.ndarray, index)]
        return np.array(result, copy=True)

    def delete_data(self, data_id: str, phase: str, var_name: str) -> None:
        self._validate_identifiers(data_id, phase, var_name)

        phase_group = self._resolve_phase(data_id, phase)
        self._resolve_var_name(data_id, phase, var_name)
        del phase_group[var_name]

    def stream_data(
        self, data_id: str, phase: str, var_name: str
    ) -> Iterator[np.ndarray]:
        """Yield one owned full-array snapshot for the requested variable.

        This backend does not preserve append-time chunk boundaries for
        streaming. Callers receive a single defensive copy of the full array,
        so NumPy-backed sources are not chunked in practice.
        """
        self._validate_identifiers(data_id, phase, var_name)

        # Stream one owned snapshot so callers cannot mutate internal storage.
        arr = self._resolve_var_name(data_id, phase, var_name)
        yield np.array(arr, copy=True)

    def copy_data(
        self, data_id: str, phase: str, var_name: str, output_var_name: str
    ) -> None:
        self._validate_identifiers(data_id, phase, var_name, output_var_name)

        phase_group = self._resolve_phase(data_id, phase)
        if var_name not in phase_group:
            raise ValueError(
                f"backend:{self.get_name()}, data_id:{data_id}, phase:{phase}| var_name:{var_name} does not exist."
            )

        if var_name == output_var_name:
            return

        if output_var_name in phase_group:
            raise ValueError(
                f"backend:{self.get_name()}, data_id:{data_id}, phase:{phase}| var_name:{output_var_name} already exists."
            )

        phase_group[output_var_name] = copy.deepcopy(phase_group[var_name])

    def rename_data(
        self, data_id: str, phase: str, var_name: str, output_var_name: str
    ) -> None:
        self._validate_identifiers(data_id, phase, var_name, output_var_name)

        phase_group = self._resolve_phase(data_id, phase)
        if var_name not in phase_group:
            raise ValueError(
                f"backend:{self.get_name()}, data_id:{data_id}, phase:{phase}| var_name:{var_name} does not exist."
            )

        if var_name == output_var_name:
            return

        if output_var_name in phase_group:
            raise ValueError(
                f"backend:{self.get_name()}, data_id:{data_id}, phase:{phase}| var_name:{output_var_name} already exists."
            )

        phase_group[output_var_name] = phase_group.pop(var_name)

    def get_data_ids(self) -> list[str]:
        db = self._resolve_db()
        return list(db.keys())

    def get_phases(self, data_id: str) -> list[str]:
        self._validate_identifiers(data_id)
        db = self._resolve_data_id(data_id)
        return list(db.keys())

    def get_var_names(self, data_id: str, phase: str) -> list[str]:
        self._validate_identifiers(data_id, phase)
        db = self._resolve_phase(data_id, phase)
        return list(db.keys())

    def get_data_info(self, data_id: str, phase: str, var_name: str) -> dict[str, Any]:
        self._validate_identifiers(data_id, phase, var_name)
        entry = self._resolve_var_entry(data_id, phase, var_name)
        cache = entry["cache"]

        if cache is None:
            first_chunk = cast(np.ndarray, entry["chunks"][0])
            dtype = first_chunk.dtype.name
            shape = tuple(first_chunk.shape[1:])
            num_events = sum(chunk.shape[0] for chunk in entry["chunks"])
        else:
            cached_arr = cast(np.ndarray, cache)
            dtype = cached_arr.dtype.name
            shape = tuple(cached_arr.shape[1:])
            num_events = cached_arr.shape[0]

        return {
            "dtype": dtype,
            "shape": shape,
            "num_events": int(num_events),
        }

    def compile(self, data_id: str) -> dict[str, Any]:
        self._validate_identifiers(data_id)

        did_group = self._resolve_data_id(data_id)
        phase_reports: dict[str, dict[str, Any]] = {}

        for phase in sorted(did_group.keys()):
            phase_group = did_group[phase]
            vars_report: dict[str, int] = {}

            for var_name in sorted(phase_group.keys()):
                if _is_staged_var_name(var_name):
                    continue
                entry = phase_group[var_name]
                cache = entry["cache"]
                if cache is None:
                    num_events = sum(chunk.shape[0] for chunk in entry["chunks"])
                else:
                    num_events = cache.shape[0]
                vars_report[var_name] = int(num_events)

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
    def _resolve_db(self) -> dict:
        """Return database if exists"""
        db = self._db
        if db is None:
            raise RuntimeError("Database is not initialized. Call initialize() first.")
        return db

    def _resolve_data_id(self, data_id: str) -> dict:
        """Return data_id database if exists"""
        db = self._resolve_db()
        member = db.get(data_id)
        if member is None or not isinstance(member, dict):
            raise ValueError(
                f"backend:{self.get_name()}| data_id:{data_id} does not exist."
            )
        return member

    def _resolve_phase(self, data_id: str, phase: str) -> dict:
        """Return phase database if exists"""
        db = self._resolve_data_id(data_id)
        member = db.get(phase)
        if member is None or not isinstance(member, dict):
            raise ValueError(
                f"backend:{self.get_name()}, data_id:{data_id}| phase:{phase} does not exist."
            )
        return member

    def _resolve_var_entry(
        self, data_id: str, phase: str, var_name: str
    ) -> dict:
        """Return variable backing record if exists."""
        db = self._resolve_phase(data_id, phase)
        member = db.get(var_name)
        if member is None or not isinstance(member, dict):
            raise ValueError(
                f"backend:{self.get_name()}, data_id:{data_id}, phase:{phase}| var_name:{var_name} does not exist."
            )
        return member

    def _resolve_var_name(self, data_id: str, phase: str, var_name: str) -> np.ndarray:
        """Return arr if exists"""
        return self._materialize(self._resolve_var_entry(data_id, phase, var_name))

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
            self._resolve_var_entry(data_id, phase, var_name)
            return True
        except (RuntimeError, ValueError):
            return False

    def _materialize(self, entry: dict) -> np.ndarray:
        """Concatenate pending chunks into a single array, caching the result.

        After concatenation the chunk list is collapsed to one element so the
        original per-chunk arrays can be garbage-collected, halving peak memory.
        """
        if entry["cache"] is None:
            arr = np.concatenate(entry["chunks"], axis=0)
            entry["cache"] = arr
            entry["chunks"] = [arr]
        return cast(np.ndarray, entry["cache"])
