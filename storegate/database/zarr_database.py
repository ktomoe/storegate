"""ZarrDatabase module."""
from collections.abc import Iterator
from typing import Any, Literal, cast

import numpy as np
import zarr

from storegate import const
from storegate.database.database import Database

_STOREGATE_META_KEY = '_storegate_meta'
_SNAPSHOT_ROOT_GROUP = '.storegate_snapshots'
_RESTORE_TMP_ROOT_GROUP = '.storegate_restore_tmp'


class ZarrDatabase(Database):
    """Base class of Zarr database."""
    def __init__(self, output_dir: str, chunk: int = 1000, mode: str = 'r') -> None:
        self._output_dir = output_dir
        self._chunk = chunk
        self._mode = mode
        zarr_mode = cast(Literal['r', 'r+', 'a', 'w', 'w-'], mode)
        self._db: Any = zarr.open(self._output_dir, mode=zarr_mode)

    def initialize(self, data_id: str) -> None:
        """Initialize the store for the given data_id.

        In read-only mode ('r'), data_id must already exist in the store.
        In writable modes, groups are created if they do not exist.
        """
        if self._mode == 'r':
            if data_id not in self._db.group_keys():
                raise KeyError(
                    f"data_id '{data_id}' does not exist in '{self._output_dir}' (mode='r')."
                )
            return
        db_data_id = self._db.require_group(data_id)
        for phase in const.PHASES:
            db_data_id.require_group(phase)
        self._bootstrap_var_names(data_id)

    def _load_storegate_meta(self, data_id: str) -> dict[str, Any]:
        if data_id not in self._db.group_keys():
            return {}

        raw = self._db[data_id].attrs.get(_STOREGATE_META_KEY, {})
        meta = dict(raw) if isinstance(raw, dict) else {}

        raw_var_names = meta.get('var_names', {})
        var_names = {
            phase: list(raw_var_names.get(phase, []))
            if isinstance(raw_var_names, dict) else []
            for phase in const.PHASES
        }
        meta['var_names'] = var_names
        return meta

    def _save_storegate_meta(self, data_id: str, meta: dict[str, Any]) -> None:
        if self._mode == 'r':
            return
        self._db[data_id].attrs[_STOREGATE_META_KEY] = meta

    def _require_writable(self, operation: str) -> None:
        if self._mode == 'r':
            raise RuntimeError(
                f'{operation}() requires write access to the zarr store; reopen with mode="w" or mode="a".'
            )

    @staticmethod
    def _copy_attrs(src: Any, dst: Any) -> None:
        for key, value in dict(src.attrs).items():
            dst.attrs[key] = value

    def _copy_array(self, src_group: Any, src_name: str, dst_group: Any, dst_name: str) -> None:
        src_arr = src_group[src_name]
        chunk_size = self._event_chunk_size(src_arr)
        chunks = getattr(src_arr, 'chunks', None)

        if src_arr.shape[0] == 0:
            dst_arr = dst_group.create_array(
                name=dst_name,
                data=src_arr[0:0],
                chunks=chunks,
            )
            self._copy_attrs(src_arr, dst_arr)
            return

        first = True
        for start in range(0, src_arr.shape[0], chunk_size):
            stop = min(start + chunk_size, src_arr.shape[0])
            chunk = src_arr[start:stop]
            if first:
                dst_arr = dst_group.create_array(
                    name=dst_name,
                    data=chunk,
                    chunks=chunks,
                )
                self._copy_attrs(src_arr, dst_arr)
                first = False
            else:
                dst_group[dst_name].append(chunk)

    def _copy_group_contents(self, src_group: Any, dst_group: Any) -> None:
        self._copy_attrs(src_group, dst_group)

        for group_name in src_group.group_keys():
            child_dst = dst_group.require_group(group_name)
            self._copy_group_contents(src_group[group_name], child_dst)

        for array_name in src_group.array_keys():
            self._copy_array(src_group, array_name, dst_group, array_name)

    @staticmethod
    def _delete_child_if_exists(parent: Any, child_name: str) -> None:
        if child_name in parent.group_keys() or child_name in parent.array_keys():
            del parent[child_name]

    def _bootstrap_var_names(self, data_id: str) -> None:
        raw_meta = self._db[data_id].attrs.get(_STOREGATE_META_KEY, {})
        meta = self._load_storegate_meta(data_id)
        changed = not isinstance(raw_meta, dict) or 'var_names' not in raw_meta

        for phase in const.PHASES:
            current_names = sorted(self._db[data_id][phase].array_keys())
            saved_names = meta['var_names'].get(phase, [])
            ordered_names = [
                name for name in saved_names
                if name in current_names
            ]
            extras = [name for name in current_names if name not in ordered_names]
            if ordered_names + extras != saved_names:
                meta['var_names'][phase] = ordered_names + extras
                changed = True

        if changed:
            self._save_storegate_meta(data_id, meta)

    def _ordered_var_names(self, data_id: str, phase: str) -> list[str]:
        current_names = set(self._db[data_id][phase].array_keys())
        if not current_names:
            return []

        meta = self._load_storegate_meta(data_id)
        ordered_names = [
            name for name in meta['var_names'].get(phase, [])
            if name in current_names
        ]
        extras = sorted(current_names - set(ordered_names))
        return ordered_names + extras

    def _append_var_name(self, data_id: str, phase: str, var_name: str) -> None:
        meta = self._load_storegate_meta(data_id)
        if var_name in meta['var_names'][phase]:
            return
        meta['var_names'][phase].append(var_name)
        self._save_storegate_meta(data_id, meta)

    def _remove_var_name(self, data_id: str, phase: str, var_name: str) -> None:
        meta = self._load_storegate_meta(data_id)
        if var_name not in meta['var_names'][phase]:
            return
        meta['var_names'][phase] = [
            name for name in meta['var_names'][phase]
            if name != var_name
        ]
        self._save_storegate_meta(data_id, meta)

    def _rename_var_name(
        self,
        data_id: str,
        phase: str,
        var_name: str,
        output_var_name: str,
    ) -> None:
        meta = self._load_storegate_meta(data_id)
        renamed: list[str] = []
        replaced = False

        for name in meta['var_names'][phase]:
            if name == var_name:
                renamed.append(output_var_name)
                replaced = True
            elif name != output_var_name:
                renamed.append(name)

        if not replaced:
            renamed.append(output_var_name)

        meta['var_names'][phase] = renamed
        self._save_storegate_meta(data_id, meta)

    @staticmethod
    def _event_chunk_size(arr: Any) -> int:
        chunks = getattr(arr, 'chunks', None)
        if isinstance(chunks, tuple) and chunks and chunks[0] is not None:
            return max(1, int(chunks[0]))
        return max(1, int(arr.shape[0]))

    def add_data(self, data_id: str, var_name: str, data: np.ndarray, phase: str) -> None:
        self._validate_zarr_compatible_dtype(
            data,
            var_name=var_name,
            phase=phase,
        )
        db = self._db[data_id][phase]
        is_new_var = var_name not in db.array_keys()

        if not is_new_var:
            existing_shape = db[var_name].shape[1:]
            if data.shape[1:] != existing_shape:
                raise ValueError(
                    f"Shape mismatch for '{var_name}' in '{phase}': "
                    f"expected {existing_shape}, got {data.shape[1:]}"
                )
            existing_dtype = db[var_name].dtype
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
            db[var_name].append(data)
        else:
            shape = data.shape
            chunks = (self._chunk, ) + tuple(shape[1:])
            db.create_array(name=var_name, data=data, chunks=chunks)

        if is_new_var:
            self._append_var_name(data_id, phase, var_name)

    def update_data(self, data_id: str, var_name: str, data: np.ndarray, phase: str, index: int | slice | None) -> None:
        arr = self._db[data_id][phase][var_name]
        data = self._prepare_update_data(
            data=data,
            total_events=arr.shape[0],
            sample_shape=tuple(arr.shape[1:]),
            existing_dtype=arr.dtype,
            index=index,
            var_name=var_name,
            phase=phase,
        )
        arr[self._normalize_index(index)] = data

    def get_data(self, data_id: str, var_name: str, phase: str, index: int | slice | None) -> np.ndarray:
        return self._db[data_id][phase][var_name][self._normalize_index(index)]  # type: ignore[no-any-return]

    def iter_data_chunks(
        self,
        data_id: str,
        var_name: str,
        phase: str,
    ) -> Iterator[np.ndarray]:
        """Yield zarr-backed chunks along the event axis."""
        arr = self._db[data_id][phase][var_name]
        chunk_size = self._event_chunk_size(arr)

        if arr.shape[0] == 0:
            yield arr[0:0]
            return

        for start in range(0, arr.shape[0], chunk_size):
            stop = min(start + chunk_size, arr.shape[0])
            yield arr[start:stop]

    def delete_data(self, data_id: str, var_name: str, phase: str) -> None:
        if var_name not in self._db[data_id][phase].array_keys():
            raise KeyError(f'"{var_name}" not found in {phase} phase.')
        del self._db[data_id][phase][var_name]
        self._remove_var_name(data_id, phase, var_name)

    def rename_data(
        self,
        data_id: str,
        var_name: str,
        output_var_name: str,
        phase: str,
    ) -> None:
        db = self._db[data_id][phase]
        if var_name not in db.array_keys():
            raise KeyError(f'"{var_name}" not found in {phase} phase.')
        if var_name == output_var_name:
            return
        if output_var_name in db.array_keys():
            raise ValueError(f'"{output_var_name}" already exists in {phase} phase.')

        arr = db[var_name]
        chunk_size = self._event_chunk_size(arr)

        try:
            if arr.shape[0] == 0:
                self.add_data(data_id, output_var_name, arr[0:0], phase)
            else:
                for start in range(0, arr.shape[0], chunk_size):
                    stop = min(start + chunk_size, arr.shape[0])
                    self.add_data(data_id, output_var_name, arr[start:stop], phase)
        except Exception:
            if output_var_name in db.array_keys():
                self.delete_data(data_id, output_var_name, phase)
            raise

        del db[var_name]
        self._rename_var_name(data_id, phase, var_name, output_var_name)

    def get_metadata(self, data_id: str, phase: str) -> dict[str, Any]:
        results: dict[str, Any] = {}
        if data_id not in self._db.group_keys():
            return results

        db = self._db[data_id][phase]

        for var_name in self._ordered_var_names(data_id, phase):
            arr = db[var_name]
            results[var_name] = {
                'backend': 'zarr',
                'type': arr.dtype.name,
                'shape': arr.shape[1:],
                'total_events': arr.shape[0]
            }
        return results

    def load_meta_attrs(self, data_id: str) -> dict[str, Any]:
        """Load persisted metadata from the attrs of the data_id group."""
        return self._load_storegate_meta(data_id)

    def save_meta_attrs(self, data_id: str, meta: dict[str, Any]) -> None:
        """Persist metadata to the attrs of the data_id group. No-op in read-only mode."""
        saved_meta = self._load_storegate_meta(data_id)
        saved_meta.update(meta)
        self._save_storegate_meta(data_id, saved_meta)

    def snapshot_data_id(self, data_id: str, snapshot_name: str) -> None:
        """Save the current zarr state of ``data_id`` under ``snapshot_name``."""
        self._require_writable('snapshot')
        if data_id not in self._db.group_keys():
            raise KeyError(f"data_id '{data_id}' not found.")

        snapshot_root = self._db.require_group(_SNAPSHOT_ROOT_GROUP)
        data_snapshots = snapshot_root.require_group(data_id)
        if snapshot_name in data_snapshots.group_keys():
            raise ValueError(
                f"snapshot '{snapshot_name}' already exists for data_id '{data_id}'."
            )

        snapshot_group = data_snapshots.require_group(snapshot_name)
        try:
            self._copy_group_contents(self._db[data_id], snapshot_group)
        except Exception:
            self._delete_child_if_exists(data_snapshots, snapshot_name)
            raise

    def restore_data_id(self, data_id: str, snapshot_name: str) -> None:
        """Replace ``data_id`` in the zarr store with ``snapshot_name``."""
        self._require_writable('restore')

        if _SNAPSHOT_ROOT_GROUP not in self._db.group_keys():
            raise KeyError(f"snapshot '{snapshot_name}' not found for data_id '{data_id}'.")

        snapshot_root = self._db[_SNAPSHOT_ROOT_GROUP]
        if data_id not in snapshot_root.group_keys():
            raise KeyError(f"snapshot '{snapshot_name}' not found for data_id '{data_id}'.")

        data_snapshots = snapshot_root[data_id]
        if snapshot_name not in data_snapshots.group_keys():
            raise KeyError(f"snapshot '{snapshot_name}' not found for data_id '{data_id}'.")

        snapshot_group = data_snapshots[snapshot_name]
        restore_tmp_root = self._db.require_group(_RESTORE_TMP_ROOT_GROUP)
        backup_name = f'{data_id}_{snapshot_name}'
        suffix = 0
        while backup_name in restore_tmp_root.group_keys():
            suffix += 1
            backup_name = f'{data_id}_{snapshot_name}_{suffix}'

        had_existing = data_id in self._db.group_keys()
        try:
            if had_existing:
                backup_group = restore_tmp_root.require_group(backup_name)
                self._copy_group_contents(self._db[data_id], backup_group)

            self._delete_child_if_exists(self._db, data_id)
            restored_group = self._db.require_group(data_id)
            self._copy_group_contents(snapshot_group, restored_group)
        except Exception:
            self._delete_child_if_exists(self._db, data_id)

            if had_existing and backup_name in restore_tmp_root.group_keys():
                restored_group = self._db.require_group(data_id)
                self._copy_group_contents(restore_tmp_root[backup_name], restored_group)
            raise
        finally:
            self._delete_child_if_exists(restore_tmp_root, backup_name)

    def close(self) -> None:
        self._db.store.close()
