from __future__ import annotations

import numpy as np

from storegate.database.database import Database

_TEMP_PREFIX = "_db_staged"


def _is_staged_var_name(name: str) -> bool:
    """Return whether *name* is an internal staged_add temporary variable."""
    return name.startswith(f"{_TEMP_PREFIX}_")


class _StagedAddTransaction:
    """Transaction object returned by :meth:`Database.staged_add`."""

    def __init__(
        self,
        db: Database,
        data_id: str,
        phase: str,
        var_names: list[str],
    ) -> None:
        self._db = db
        self._data_id = data_id
        self._phase = phase
        self._var_names = list(var_names)
        self._temp_map: dict[str, str] = {}
        self._written: set[str] = set()
        self._active = False

    # ------------------------------------------------------------------
    def __enter__(self) -> _StagedAddTransaction:
        db = self._db
        data_id, phase = self._data_id, self._phase

        if not self._var_names:
            raise ValueError("var_names must not be empty.")

        for name in self._var_names:
            db._validate_identifiers(data_id, phase, name)

        if len(self._var_names) != len(set(self._var_names)):
            raise ValueError(
                f"backend:{db.get_name()}, data_id:{data_id}, phase:{phase}| "
                "var_names contains duplicate entries."
            )

        try:
            existing = set(db.get_var_names(data_id, phase))
        except ValueError:
            existing = set()

        for name in self._var_names:
            if name in existing:
                raise ValueError(
                    f"backend:{db.get_name()}, data_id:{data_id}, "
                    f"phase:{phase}| var_name:{name} already exists."
                )

        forbidden = existing | set(self._var_names)
        for name in self._var_names:
            temp = _allocate_temp_name(name, forbidden)
            self._temp_map[name] = temp
            forbidden.add(temp)

        self._active = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self._active = False

        if exc_val is None:
            published: list[str] = []
            try:
                for name in self._var_names:
                    if name in self._written:
                        self._db.rename_data(
                            self._data_id,
                            self._phase,
                            self._temp_map[name],
                            name,
                        )
                        published.append(name)
            except Exception as exc:
                rollback_errors = self._rollback_published(published)
                self._cleanup()
                if rollback_errors:
                    raise ExceptionGroup(
                        "staged_add commit failed and rollback was incomplete.",
                        [exc, *rollback_errors],
                    ) from None
                raise
        else:
            self._cleanup()

    # ------------------------------------------------------------------
    def add_data(self, name: str, data: np.ndarray) -> None:
        """Append *data* to the staged variable *name*."""
        if not self._active:
            raise RuntimeError("add_data() called outside of staged_add context.")
        if name not in self._temp_map:
            raise ValueError(
                f"var_name:{name} is not declared in this staged_add transaction. "
                f"Declared names: {self._var_names}"
            )
        # Mark before add_data() so _cleanup covers partial writes
        # from backends with non-atomic write behavior.
        self._written.add(name)
        self._db.add_data(
            self._data_id, self._phase, self._temp_map[name], data,
        )

    # ------------------------------------------------------------------
    def _rollback_published(self, published: list[str]) -> list[Exception]:
        errors: list[Exception] = []
        for name in reversed(published):
            try:
                self._db.rename_data(
                    self._data_id,
                    self._phase,
                    name,
                    self._temp_map[name],
                )
            except Exception as exc:
                exc.add_note(
                    "while rolling back staged_add commit "
                    f"for var_name:{name}"
                )
                errors.append(exc)

        return errors

    # ------------------------------------------------------------------
    def _cleanup(self, exclude: set[str] | None = None) -> None:
        excluded = exclude or set()
        for name in self._var_names:
            if name in self._written and name not in excluded:
                try:
                    self._db.delete_data(
                        self._data_id, self._phase, self._temp_map[name],
                    )
                except Exception:
                    pass


def _allocate_temp_name(name: str, forbidden: set[str]) -> str:
    base = f"{_TEMP_PREFIX}_{name}"
    candidate = base
    suffix = 0
    while candidate in forbidden:
        suffix += 1
        candidate = f"{base}_{suffix}"
    return candidate
