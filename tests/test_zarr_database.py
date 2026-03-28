"""Comprehensive tests for storegate/database/zarr_database.py."""

import os
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import zarr
from zarr.core.group import Group
from zarr.storage import LocalStore

from storegate.database.zarr_database import (
    ZarrDatabase,
    _AUTO_CHUNK_BYTES,
    _STOREGATE_SCHEMA_KEY,
    _STOREGATE_SCHEMA_MARKER,
    requires_write_mode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_db(
    tmp_path: Path,
    mode: str = "a",
    data_id: str | None = "ds1",
    chunk: int | str = "auto",
    auto_chunk_bytes: int = _AUTO_CHUNK_BYTES,
    strict_schema: bool = True,
) -> ZarrDatabase:
    """Create and initialize a ZarrDatabase."""
    db = ZarrDatabase(
        str(tmp_path / "store.zarr"),
        mode=mode,
        chunk=chunk,
        auto_chunk_bytes=auto_chunk_bytes,
        strict_schema=strict_schema,
    )
    db.initialize(data_id)
    return db


def _populated_db(tmp_path: Path) -> ZarrDatabase:
    """Return a db with ds1/train/x (10, 3) float32."""
    db = _make_db(tmp_path)
    db.add_data("ds1", "train", "x", np.ones((10, 3), dtype=np.float32))
    return db


# ===========================================================================
# Constructor validation
# ===========================================================================
class TestConstructor:
    def test_valid_modes(self, tmp_path):
        for mode in ("r", "r+", "a", "w-"):
            if mode == "r":
                # need existing store for read mode
                store_path = tmp_path / f"store_{mode}.zarr"
                store_path.mkdir(parents=True)
                ZarrDatabase(str(store_path), mode=mode, strict_schema=False)
            else:
                ZarrDatabase(str(tmp_path / f"store_{mode}.zarr"), mode=mode)

    def test_invalid_mode(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid mode"):
            ZarrDatabase(str(tmp_path / "store.zarr"), mode="x")  # type: ignore

    def test_strict_schema_false_non_read_mode(self, tmp_path):
        with pytest.raises(ValueError, match="strict_schema=False is supported only"):
            ZarrDatabase(str(tmp_path / "store.zarr"), mode="a", strict_schema=False)

    def test_chunk_auto(self, tmp_path):
        db = ZarrDatabase(str(tmp_path / "store.zarr"), mode="a", chunk="auto")
        assert db._chunk == "auto"

    def test_chunk_int(self, tmp_path):
        db = ZarrDatabase(str(tmp_path / "store.zarr"), mode="a", chunk=1024)
        assert db._chunk == 1024

    def test_chunk_invalid(self, tmp_path):
        with pytest.raises(ValueError, match="must be a positive integer"):
            ZarrDatabase(str(tmp_path / "store.zarr"), mode="a", chunk=0)

    def test_auto_chunk_bytes_invalid(self, tmp_path):
        with pytest.raises(ValueError, match="must be a positive integer"):
            ZarrDatabase(str(tmp_path / "store.zarr"), mode="a", auto_chunk_bytes=-1)


# ===========================================================================
# requires_write_mode decorator
# ===========================================================================
class TestRequiresWriteMode:
    def test_blocks_in_read_mode(self, tmp_path):
        # Create a store first in write mode
        db_w = _make_db(tmp_path, mode="a", data_id="ds1")
        db_w.add_data("ds1", "train", "x", np.ones((5,), dtype=np.float32))
        db_w.close()

        db = ZarrDatabase(str(tmp_path / "store.zarr"), mode="r")
        db.initialize()
        with pytest.raises(ValueError, match="read only"):
            db.set_data_id("ds2")
        db.close()

    def test_allows_in_write_mode(self, tmp_path):
        db = _make_db(tmp_path, mode="a")
        db.set_data_id("ds2")  # should not raise
        db.close()


# ===========================================================================
# Initialize
# ===========================================================================
class TestInitialize:
    def test_initialize_no_data_id(self, tmp_path):
        db = ZarrDatabase(str(tmp_path / "store.zarr"), mode="a")
        db.initialize()
        assert db.get_data_ids() == []
        db.close()

    def test_initialize_with_data_id(self, tmp_path):
        db = _make_db(tmp_path)
        assert "ds1" in db.get_data_ids()
        db.close()

    def test_initialize_already_initialized_idempotent(self, tmp_path):
        db = _make_db(tmp_path)
        db.add_data("ds1", "train", "x", np.ones((5,), dtype=np.float32))
        db.initialize("ds1")
        assert db.get_var_names("ds1", "train") == ["x"]
        db.close()

    def test_initialize_new_data_id(self, tmp_path):
        db = _make_db(tmp_path)
        db.initialize("ds2")
        assert set(db.get_data_ids()) == {"ds1", "ds2"}
        db.close()

    def test_initialize_store_does_not_exist_read_mode(self, tmp_path):
        db = ZarrDatabase(str(tmp_path / "missing.zarr"), mode="r")
        with pytest.raises(Exception):
            db.initialize()

    def test_initialize_store_exists_valid_schema(self, tmp_path):
        # Create a valid store
        db1 = _make_db(tmp_path, data_id="ds1")
        db1.close()

        # Re-open in append mode
        db2 = ZarrDatabase(str(tmp_path / "store.zarr"), mode="a")
        db2.initialize()
        assert "ds1" in db2.get_data_ids()
        db2.close()

    def test_initialize_store_exists_invalid_schema(self, tmp_path):
        # Create a zarr store without storegate marker
        store_path = tmp_path / "bad.zarr"
        zarr.open_group(str(store_path), mode="w-")

        db = ZarrDatabase(str(store_path), mode="a")
        with pytest.raises(ValueError, match="missing the storegate schema marker"):
            db.initialize()

    def test_initialize_strict_schema_false(self, tmp_path):
        # Create a zarr store without storegate marker
        store_path = tmp_path / "foreign.zarr"
        zarr.open_group(str(store_path), mode="w-")

        db = ZarrDatabase(str(store_path), mode="r", strict_schema=False)
        db.initialize()
        db.close()

    def test_initialize_read_mode_data_id_must_exist(self, tmp_path):
        db_w = _make_db(tmp_path, data_id="ds1")
        db_w.close()

        db = ZarrDatabase(str(tmp_path / "store.zarr"), mode="r")
        with pytest.raises(ValueError, match="does not exist"):
            db.initialize("missing")

    def test_initialize_w_minus_mode_existing_store_raises(self, tmp_path):
        # Create a store with bad schema
        store_path = tmp_path / "wm.zarr"
        zarr.open_group(str(store_path), mode="w-")

        # zarr mode='w-' refuses any existing store, regardless of schema
        db = ZarrDatabase(str(store_path), mode="w-")
        with pytest.raises(FileExistsError):
            db.initialize()

    def test_initialize_store_not_preexisted_new_dir(self, tmp_path):
        store_path = tmp_path / "new_store.zarr"
        db = ZarrDatabase(str(store_path), mode="a")
        db.initialize()
        # Should set marker on new store
        db.close()

    def test_initialize_read_mode_no_marker_needed_for_nonexistent_dir(self, tmp_path):
        """Read mode with non-existent dir should raise."""
        db = ZarrDatabase(str(tmp_path / "noexist.zarr"), mode="r")
        with pytest.raises(Exception):
            db.initialize()


# ===========================================================================
# Schema validation
# ===========================================================================
class TestSchemaValidation:
    def test_missing_root_marker(self, tmp_path):
        store_path = tmp_path / "bad.zarr"
        zarr.open_group(str(store_path), mode="w-")

        db = ZarrDatabase(str(store_path), mode="a")
        with pytest.raises(ValueError, match="missing the storegate schema marker"):
            db.initialize()

    def test_wrong_root_marker(self, tmp_path):
        store_path = tmp_path / "bad.zarr"
        g = zarr.open_group(str(store_path), mode="w-")
        g.attrs[_STOREGATE_SCHEMA_KEY] = "wrong_version"

        db = ZarrDatabase(str(store_path), mode="a")
        with pytest.raises(ValueError, match="unsupported"):
            db.initialize()

    def test_unexpected_arrays_at_root(self, tmp_path):
        store_path = tmp_path / "bad.zarr"
        g = zarr.open_group(str(store_path), mode="w-")
        g.attrs[_STOREGATE_SCHEMA_KEY] = _STOREGATE_SCHEMA_MARKER
        g.create_array("bad_arr", data=np.ones(5))

        db = ZarrDatabase(str(store_path), mode="a")
        with pytest.raises(ValueError, match="unexpected arrays"):
            db.initialize()

    def test_invalid_data_id_identifier(self, tmp_path):
        store_path = tmp_path / "bad.zarr"
        g = zarr.open_group(str(store_path), mode="w-")
        g.attrs[_STOREGATE_SCHEMA_KEY] = _STOREGATE_SCHEMA_MARKER
        bad_group = g.require_group("bad name with spaces")
        bad_group.attrs[_STOREGATE_SCHEMA_KEY] = _STOREGATE_SCHEMA_MARKER

        db = ZarrDatabase(str(store_path), mode="a")
        with pytest.raises(ValueError, match="Invalid"):
            db.initialize()

    def test_missing_data_id_marker(self, tmp_path):
        store_path = tmp_path / "bad.zarr"
        g = zarr.open_group(str(store_path), mode="w-")
        g.attrs[_STOREGATE_SCHEMA_KEY] = _STOREGATE_SCHEMA_MARKER
        g.require_group("ds1")  # no marker

        db = ZarrDatabase(str(store_path), mode="a")
        with pytest.raises(ValueError, match="missing the storegate schema marker"):
            db.initialize()

    def test_unexpected_arrays_at_data_id_level(self, tmp_path):
        store_path = tmp_path / "bad.zarr"
        g = zarr.open_group(str(store_path), mode="w-")
        g.attrs[_STOREGATE_SCHEMA_KEY] = _STOREGATE_SCHEMA_MARKER
        did = g.require_group("ds1")
        did.attrs[_STOREGATE_SCHEMA_KEY] = _STOREGATE_SCHEMA_MARKER
        did.create_array("bad_arr", data=np.ones(5))

        db = ZarrDatabase(str(store_path), mode="a")
        with pytest.raises(ValueError, match="unexpected arrays"):
            db.initialize()

    def test_invalid_phase_identifier(self, tmp_path):
        store_path = tmp_path / "bad.zarr"
        g = zarr.open_group(str(store_path), mode="w-")
        g.attrs[_STOREGATE_SCHEMA_KEY] = _STOREGATE_SCHEMA_MARKER
        did = g.require_group("ds1")
        did.attrs[_STOREGATE_SCHEMA_KEY] = _STOREGATE_SCHEMA_MARKER
        bad_phase = did.require_group("bad phase")
        bad_phase.attrs[_STOREGATE_SCHEMA_KEY] = _STOREGATE_SCHEMA_MARKER

        db = ZarrDatabase(str(store_path), mode="a")
        with pytest.raises(ValueError, match="Invalid"):
            db.initialize()

    def test_missing_phase_marker(self, tmp_path):
        store_path = tmp_path / "bad.zarr"
        g = zarr.open_group(str(store_path), mode="w-")
        g.attrs[_STOREGATE_SCHEMA_KEY] = _STOREGATE_SCHEMA_MARKER
        did = g.require_group("ds1")
        did.attrs[_STOREGATE_SCHEMA_KEY] = _STOREGATE_SCHEMA_MARKER
        did.require_group("train")  # no marker

        db = ZarrDatabase(str(store_path), mode="a")
        with pytest.raises(ValueError, match="missing the storegate schema marker"):
            db.initialize()

    def test_unexpected_groups_at_phase_level(self, tmp_path):
        store_path = tmp_path / "bad.zarr"
        g = zarr.open_group(str(store_path), mode="w-")
        g.attrs[_STOREGATE_SCHEMA_KEY] = _STOREGATE_SCHEMA_MARKER
        did = g.require_group("ds1")
        did.attrs[_STOREGATE_SCHEMA_KEY] = _STOREGATE_SCHEMA_MARKER
        phase = did.require_group("train")
        phase.attrs[_STOREGATE_SCHEMA_KEY] = _STOREGATE_SCHEMA_MARKER
        bad = phase.require_group("subgroup")
        bad.attrs[_STOREGATE_SCHEMA_KEY] = _STOREGATE_SCHEMA_MARKER

        db = ZarrDatabase(str(store_path), mode="a")
        with pytest.raises(ValueError, match="unexpected groups"):
            db.initialize()

    def test_invalid_var_name_identifier(self, tmp_path):
        store_path = tmp_path / "bad.zarr"
        g = zarr.open_group(str(store_path), mode="w-")
        g.attrs[_STOREGATE_SCHEMA_KEY] = _STOREGATE_SCHEMA_MARKER
        did = g.require_group("ds1")
        did.attrs[_STOREGATE_SCHEMA_KEY] = _STOREGATE_SCHEMA_MARKER
        phase = did.require_group("train")
        phase.attrs[_STOREGATE_SCHEMA_KEY] = _STOREGATE_SCHEMA_MARKER
        arr = phase.create_array("bad.name", data=np.ones(5))
        arr.attrs[_STOREGATE_SCHEMA_KEY] = _STOREGATE_SCHEMA_MARKER

        db = ZarrDatabase(str(store_path), mode="a")
        with pytest.raises(ValueError, match="Invalid"):
            db.initialize()

    def test_missing_var_marker(self, tmp_path):
        store_path = tmp_path / "bad.zarr"
        g = zarr.open_group(str(store_path), mode="w-")
        g.attrs[_STOREGATE_SCHEMA_KEY] = _STOREGATE_SCHEMA_MARKER
        did = g.require_group("ds1")
        did.attrs[_STOREGATE_SCHEMA_KEY] = _STOREGATE_SCHEMA_MARKER
        phase = did.require_group("train")
        phase.attrs[_STOREGATE_SCHEMA_KEY] = _STOREGATE_SCHEMA_MARKER
        phase.create_array("x", data=np.ones(5))  # no marker

        db = ZarrDatabase(str(store_path), mode="a")
        with pytest.raises(ValueError, match="missing the storegate schema marker"):
            db.initialize()

    def test_ndim_lt_1(self, tmp_path):
        store_path = tmp_path / "bad.zarr"
        g = zarr.open_group(str(store_path), mode="w-")
        g.attrs[_STOREGATE_SCHEMA_KEY] = _STOREGATE_SCHEMA_MARKER
        did = g.require_group("ds1")
        did.attrs[_STOREGATE_SCHEMA_KEY] = _STOREGATE_SCHEMA_MARKER
        phase = did.require_group("train")
        phase.attrs[_STOREGATE_SCHEMA_KEY] = _STOREGATE_SCHEMA_MARKER
        # Create a 0-dim array
        arr = phase.create_array("x", shape=(), dtype=np.float32)
        arr.attrs[_STOREGATE_SCHEMA_KEY] = _STOREGATE_SCHEMA_MARKER

        db = ZarrDatabase(str(store_path), mode="a")
        with pytest.raises(ValueError, match="ndim >= 1"):
            db.initialize()


# ===========================================================================
# Basic properties
# ===========================================================================
class TestBasicProperties:
    def test_get_name(self, tmp_path):
        db = _make_db(tmp_path)
        assert db.get_name() == "zarr"
        db.close()

    def test_is_writable_true(self, tmp_path):
        db = _make_db(tmp_path, mode="a")
        assert db.is_writable() is True
        db.close()

    def test_is_writable_false(self, tmp_path):
        db_w = _make_db(tmp_path, data_id="ds1")
        db_w.close()
        db = ZarrDatabase(str(tmp_path / "store.zarr"), mode="r")
        assert db.is_writable() is False


# ===========================================================================
# __enter__ / __exit__
# ===========================================================================
class TestContextManager:
    def test_enter_exit(self, tmp_path):
        db = ZarrDatabase(str(tmp_path / "store.zarr"), mode="a")
        with db as d:
            assert d is db
            d.set_data_id("ds1")
            assert "ds1" in d.get_data_ids()
        # After exit, db is closed
        with pytest.raises(RuntimeError, match="not initialized"):
            db.get_data_ids()

    def test_exit_on_exception(self, tmp_path):
        db = ZarrDatabase(str(tmp_path / "store.zarr"), mode="a")
        with pytest.raises(ValueError):
            with db as d:
                d.set_data_id("ds1")
                raise ValueError("boom")
        with pytest.raises(RuntimeError, match="not initialized"):
            db.get_data_ids()


# ===========================================================================
# set_data_id / delete_data_id
# ===========================================================================
class TestDataIdOps:
    def test_set_data_id(self, tmp_path):
        db = _make_db(tmp_path)
        db.set_data_id("ds2")
        assert "ds2" in db.get_data_ids()
        db.close()

    def test_set_data_id_already_exists(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        db.set_data_id("ds1")  # no error
        assert db.get_data_ids() == ["ds1"]
        db.close()

    def test_set_data_id_invalid(self, tmp_path):
        db = _make_db(tmp_path)
        with pytest.raises(ValueError, match="Invalid"):
            db.set_data_id("")
        db.close()

    def test_delete_data_id(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        db.delete_data_id("ds1")
        assert db.get_data_ids() == []
        db.close()

    def test_delete_data_id_missing(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        with pytest.raises(ValueError, match="does not exist"):
            db.delete_data_id("missing")
        db.close()


# ===========================================================================
# set_phase / delete_phase
# ===========================================================================
class TestPhaseOps:
    def test_set_phase(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        db.set_phase("ds1", "train")
        assert "train" in db.get_phases("ds1")
        db.close()

    def test_set_phase_already_exists(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        db.set_phase("ds1", "train")
        db.set_phase("ds1", "train")
        assert db.get_phases("ds1") == ["train"]
        db.close()

    def test_set_phase_data_id_missing(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        with pytest.raises(ValueError, match="does not exist"):
            db.set_phase("missing", "train")
        db.close()

    def test_delete_phase(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        db.set_phase("ds1", "train")
        db.delete_phase("ds1", "train")
        assert db.get_phases("ds1") == []
        db.close()

    def test_delete_phase_missing(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        with pytest.raises(ValueError, match="does not exist"):
            db.delete_phase("ds1", "train")
        db.close()


# ===========================================================================
# clear / close
# ===========================================================================
class TestClearClose:
    def test_clear(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        db.add_data("ds1", "train", "x", np.ones((5,), dtype=np.float32))
        db.clear()
        assert db.get_data_ids() == []
        db.close()

    def test_clear_not_initialized(self, tmp_path):
        db = ZarrDatabase(str(tmp_path / "store.zarr"), mode="a")
        with pytest.raises(RuntimeError, match="not initialized"):
            db.clear()

    def test_close(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        db.close()
        with pytest.raises(RuntimeError, match="not initialized"):
            db.get_data_ids()

    def test_close_already_closed_idempotent(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        db.close()
        db.close()  # should not raise

    def test_close_not_initialized(self, tmp_path):
        db = ZarrDatabase(str(tmp_path / "store.zarr"), mode="a")
        db.close()  # should not raise


# ===========================================================================
# add_data
# ===========================================================================
class TestAddData:
    def test_add_new(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        data = np.ones((10, 3), dtype=np.float32)
        db.add_data("ds1", "train", "x", data)
        result = db.get_data("ds1", "train", "x")
        np.testing.assert_array_equal(result, data)
        db.close()

    def test_add_creates_phase(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        db.add_data("ds1", "train", "x", np.ones((5,), dtype=np.float32))
        assert "train" in db.get_phases("ds1")
        db.close()

    def test_add_append(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        db.add_data("ds1", "train", "x", np.ones((5, 3), dtype=np.float32))
        db.add_data("ds1", "train", "x", np.zeros((3, 3), dtype=np.float32))
        result = db.get_data("ds1", "train", "x")
        assert result.shape == (8, 3)
        np.testing.assert_array_equal(result[:5], 1.0)
        np.testing.assert_array_equal(result[5:], 0.0)
        db.close()

    def test_add_append_dtype_mismatch(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        db.add_data("ds1", "train", "x", np.ones((5, 3), dtype=np.float32))
        with pytest.raises(ValueError, match="expected float32"):
            db.add_data("ds1", "train", "x", np.ones((3, 3), dtype=np.float64))
        db.close()

    def test_add_append_shape_mismatch(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        db.add_data("ds1", "train", "x", np.ones((5, 3), dtype=np.float32))
        with pytest.raises(ValueError, match="expected"):
            db.add_data("ds1", "train", "x", np.ones((3, 4), dtype=np.float32))
        db.close()

    def test_add_0dim_rejected(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        with pytest.raises(ValueError, match="must be >= 1"):
            db.add_data("ds1", "train", "x", np.array(1.0, dtype=np.float32))
        db.close()

    def test_add_various_dtypes(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        for dtype in [np.float32, np.float64, np.int32, np.int64, bool, np.complex64]:
            name = f"v_{np.dtype(dtype).name}"
            db.add_data("ds1", "train", name, np.ones((3,), dtype=dtype))
            info = db.get_data_info("ds1", "train", name)
            assert info["dtype"] == np.dtype(dtype).name
        db.close()

    def test_add_masked_array(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        data = np.ma.MaskedArray(np.ones((5,)))
        with pytest.raises(TypeError, match="MaskedArray"):
            db.add_data("ds1", "train", "x", data)
        db.close()

    def test_add_not_ndarray(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        with pytest.raises(TypeError, match="Expected numpy.ndarray"):
            db.add_data("ds1", "train", "x", [1, 2, 3])  # type: ignore
        db.close()

    def test_add_structured_dtype(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        dt = np.dtype([("a", "f4"), ("b", "i4")])
        data = np.zeros(5, dtype=dt)
        with pytest.raises(TypeError, match="Structured dtype"):
            db.add_data("ds1", "train", "x", data)
        db.close()

    def test_add_string_dtype(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        with pytest.raises(ValueError, match="not persistable"):
            db.add_data("ds1", "train", "x", np.array(["a", "b"]))
        db.close()

    def test_add_bytes_dtype(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        with pytest.raises(ValueError, match="not persistable"):
            db.add_data("ds1", "train", "x", np.array([b"a", b"b"]))
        db.close()

    def test_add_missing_data_id(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        with pytest.raises(ValueError, match="does not exist"):
            db.add_data("missing", "train", "x", np.ones((5,), dtype=np.float32))
        db.close()

    def test_add_read_only(self, tmp_path):
        db_w = _make_db(tmp_path, data_id="ds1")
        db_w.close()
        db = ZarrDatabase(str(tmp_path / "store.zarr"), mode="r")
        db.initialize()
        with pytest.raises(ValueError, match="read only"):
            db.add_data("ds1", "train", "x", np.ones((5,), dtype=np.float32))
        db.close()


# ===========================================================================
# update_data
# ===========================================================================
class TestUpdateData:
    def test_update_single_int(self, tmp_path):
        db = _populated_db(tmp_path)
        new_row = np.zeros((3,), dtype=np.float32)
        db.update_data("ds1", "train", "x", new_row, 0)
        result = db.get_data("ds1", "train", "x", 0)
        np.testing.assert_array_equal(result, 0.0)
        db.close()

    def test_update_single_np_integer(self, tmp_path):
        db = _populated_db(tmp_path)
        new_row = np.zeros((3,), dtype=np.float32)
        db.update_data("ds1", "train", "x", new_row, np.int64(2))
        result = db.get_data("ds1", "train", "x", 2)
        np.testing.assert_array_equal(result, 0.0)
        db.close()

    def test_update_slice(self, tmp_path):
        db = _populated_db(tmp_path)
        new_data = np.zeros((3, 3), dtype=np.float32)
        db.update_data("ds1", "train", "x", new_data, slice(0, 3))
        result = db.get_data("ds1", "train", "x")
        np.testing.assert_array_equal(result[:3], 0.0)
        np.testing.assert_array_equal(result[3:], 1.0)
        db.close()

    def test_update_slice_wrong_events(self, tmp_path):
        db = _populated_db(tmp_path)
        new_data = np.zeros((5, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="expected 3 events"):
            db.update_data("ds1", "train", "x", new_data, slice(0, 3))
        db.close()

    def test_update_dtype_mismatch(self, tmp_path):
        db = _populated_db(tmp_path)
        with pytest.raises(ValueError, match="expected float32"):
            db.update_data("ds1", "train", "x", np.zeros((3,), dtype=np.float64), 0)
        db.close()

    def test_update_shape_mismatch(self, tmp_path):
        db = _populated_db(tmp_path)
        with pytest.raises(ValueError, match="expected"):
            db.update_data("ds1", "train", "x", np.zeros((4,), dtype=np.float32), 0)
        db.close()

    def test_update_invalid_index(self, tmp_path):
        db = _populated_db(tmp_path)
        with pytest.raises(ValueError, match="int, np.integer, or slice"):
            db.update_data("ds1", "train", "x", np.zeros((3,), dtype=np.float32), [0])  # type: ignore
        db.close()

    def test_update_read_only(self, tmp_path):
        db_w = _make_db(tmp_path, data_id="ds1")
        db_w.add_data("ds1", "train", "x", np.ones((5, 3), dtype=np.float32))
        db_w.close()
        db = ZarrDatabase(str(tmp_path / "store.zarr"), mode="r")
        db.initialize()
        with pytest.raises(ValueError, match="read only"):
            db.update_data("ds1", "train", "x", np.zeros((3,), dtype=np.float32), 0)
        db.close()


# ===========================================================================
# get_data
# ===========================================================================
class TestGetData:
    def test_get_all(self, tmp_path):
        db = _populated_db(tmp_path)
        result = db.get_data("ds1", "train", "x")
        assert result.shape == (10, 3)
        db.close()

    def test_get_int_index(self, tmp_path):
        db = _populated_db(tmp_path)
        result = db.get_data("ds1", "train", "x", 0)
        assert result.shape == (3,)
        db.close()

    def test_get_np_integer_index(self, tmp_path):
        db = _populated_db(tmp_path)
        result = db.get_data("ds1", "train", "x", np.int64(0))
        assert result.shape == (3,)
        db.close()

    def test_get_slice_index(self, tmp_path):
        db = _populated_db(tmp_path)
        result = db.get_data("ds1", "train", "x", slice(0, 5))
        assert result.shape == (5, 3)
        db.close()

    def test_get_list_index(self, tmp_path):
        db = _populated_db(tmp_path)
        result = db.get_data("ds1", "train", "x", [0, 2, 4])
        assert result.shape == (3, 3)
        db.close()

    def test_get_ndarray_index(self, tmp_path):
        db = _populated_db(tmp_path)
        idx = np.array([0, 2, 4], dtype=np.int64)
        result = db.get_data("ds1", "train", "x", idx)
        assert result.shape == (3, 3)
        db.close()

    def test_get_defensive_copy(self, tmp_path):
        db = _populated_db(tmp_path)
        result = db.get_data("ds1", "train", "x")
        result[:] = 999
        result2 = db.get_data("ds1", "train", "x")
        np.testing.assert_array_equal(result2, 1.0)
        db.close()

    def test_get_missing_var(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        db.set_phase("ds1", "train")
        with pytest.raises(ValueError, match="does not exist"):
            db.get_data("ds1", "train", "x")
        db.close()

    def test_get_bool_index_rejected(self, tmp_path):
        db = _populated_db(tmp_path)
        with pytest.raises(ValueError, match="must not be bool"):
            db.get_data("ds1", "train", "x", True)  # type: ignore
        db.close()

    def test_get_slice_with_step_rejected(self, tmp_path):
        db = _populated_db(tmp_path)
        with pytest.raises(ValueError, match="int, np.integer, slice without step"):
            db.get_data("ds1", "train", "x", slice(0, 5, 2))
        db.close()

    def test_get_empty_list_rejected(self, tmp_path):
        db = _populated_db(tmp_path)
        with pytest.raises(ValueError, match="must not be an empty list"):
            db.get_data("ds1", "train", "x", [])
        db.close()

    def test_get_list_of_bools_rejected(self, tmp_path):
        db = _populated_db(tmp_path)
        with pytest.raises(ValueError, match="must not be list"):
            db.get_data("ds1", "train", "x", [True, False])
        db.close()

    def test_get_empty_ndarray_rejected(self, tmp_path):
        db = _populated_db(tmp_path)
        with pytest.raises(ValueError, match="must not be an empty numpy.ndarray"):
            db.get_data("ds1", "train", "x", np.array([], dtype=np.int64))
        db.close()

    def test_get_2d_ndarray_rejected(self, tmp_path):
        db = _populated_db(tmp_path)
        with pytest.raises(ValueError, match="int, np.integer, slice without step"):
            db.get_data("ds1", "train", "x", np.array([[0, 1]], dtype=np.int64))
        db.close()

    def test_get_float_ndarray_rejected(self, tmp_path):
        db = _populated_db(tmp_path)
        with pytest.raises(ValueError, match="int, np.integer, slice without step"):
            db.get_data("ds1", "train", "x", np.array([1.0, 2.0]))
        db.close()


# ===========================================================================
# delete_data
# ===========================================================================
class TestDeleteData:
    def test_delete(self, tmp_path):
        db = _populated_db(tmp_path)
        db.delete_data("ds1", "train", "x")
        assert db.get_var_names("ds1", "train") == []
        db.close()

    def test_delete_missing(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        db.set_phase("ds1", "train")
        with pytest.raises(ValueError, match="does not exist"):
            db.delete_data("ds1", "train", "x")
        db.close()

    def test_delete_read_only(self, tmp_path):
        db_w = _make_db(tmp_path, data_id="ds1")
        db_w.add_data("ds1", "train", "x", np.ones((5,), dtype=np.float32))
        db_w.close()
        db = ZarrDatabase(str(tmp_path / "store.zarr"), mode="r")
        db.initialize()
        with pytest.raises(ValueError, match="read only"):
            db.delete_data("ds1", "train", "x")
        db.close()


# ===========================================================================
# stream_data
# ===========================================================================
class TestStreamData:
    def test_stream_chunked(self, tmp_path):
        """With small chunk size, data is split into multiple chunks."""
        db = _make_db(tmp_path, chunk=3, data_id="ds1")
        db.add_data("ds1", "train", "x", np.ones((10,), dtype=np.float32))
        chunks = list(db.stream_data("ds1", "train", "x"))
        total = sum(c.shape[0] for c in chunks)
        assert total == 10
        assert len(chunks) == 4  # 3 + 3 + 3 + 1
        db.close()

    def test_stream_zero_events(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        db.add_data("ds1", "train", "x", np.ones((0, 3), dtype=np.float32))
        chunks = list(db.stream_data("ds1", "train", "x"))
        assert len(chunks) == 1
        assert chunks[0].shape == (0, 3)
        db.close()

    def test_stream_defensive_copy(self, tmp_path):
        db = _populated_db(tmp_path)
        for chunk in db.stream_data("ds1", "train", "x"):
            chunk[:] = 999
        result = db.get_data("ds1", "train", "x")
        np.testing.assert_array_equal(result, 1.0)
        db.close()

    def test_stream_closed_during_iteration(self, tmp_path):
        db = _make_db(tmp_path, chunk=3, data_id="ds1")
        db.add_data("ds1", "train", "x", np.ones((10,), dtype=np.float32))
        gen = db.stream_data("ds1", "train", "x")
        next(gen)  # get first chunk
        db.close()
        with pytest.raises(RuntimeError, match="closed while streaming"):
            next(gen)

    def test_stream_closed_during_zero_events(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        db.add_data("ds1", "train", "x", np.ones((0, 3), dtype=np.float32))
        gen = db.stream_data("ds1", "train", "x")
        db.close()
        with pytest.raises(RuntimeError, match="closed while streaming"):
            next(gen)

    def test_stream_missing_var(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        db.set_phase("ds1", "train")
        with pytest.raises(ValueError, match="does not exist"):
            list(db.stream_data("ds1", "train", "x"))
        db.close()


# ===========================================================================
# copy_data
# ===========================================================================
class TestCopyData:
    def test_copy(self, tmp_path):
        db = _populated_db(tmp_path)
        db.copy_data("ds1", "train", "x", "y")
        result = db.get_data("ds1", "train", "y")
        np.testing.assert_array_equal(result, np.ones((10, 3), dtype=np.float32))
        db.close()

    def test_copy_same_name_noop(self, tmp_path):
        db = _populated_db(tmp_path)
        db.copy_data("ds1", "train", "x", "x")
        assert "x" in db.get_var_names("ds1", "train")
        db.close()

    def test_copy_dest_exists_in_array_keys(self, tmp_path):
        db = _populated_db(tmp_path)
        db.add_data("ds1", "train", "y", np.ones((5,), dtype=np.float32))
        with pytest.raises(ValueError, match="already exists"):
            db.copy_data("ds1", "train", "x", "y")
        db.close()

    def test_copy_src_missing(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        db.set_phase("ds1", "train")
        with pytest.raises(ValueError, match="does not exist"):
            db.copy_data("ds1", "train", "missing", "y")
        db.close()

    def test_copy_dest_exists_at_filesystem(self, tmp_path):
        """Filesystem path already exists even though not in array_keys."""
        db = _populated_db(tmp_path)
        # Manually create the dest directory to simulate collision
        store_path = Path(db._output_dir)
        dest = store_path / "ds1" / "train" / "y"
        dest.mkdir(parents=True, exist_ok=True)
        with pytest.raises(ValueError, match="already exists"):
            db.copy_data("ds1", "train", "x", "y")
        db.close()

    def test_copy_read_only(self, tmp_path):
        db_w = _make_db(tmp_path, data_id="ds1")
        db_w.add_data("ds1", "train", "x", np.ones((5,), dtype=np.float32))
        db_w.close()
        db = ZarrDatabase(str(tmp_path / "store.zarr"), mode="r")
        db.initialize()
        with pytest.raises(ValueError, match="read only"):
            db.copy_data("ds1", "train", "x", "y")
        db.close()


# ===========================================================================
# rename_data
# ===========================================================================
class TestRenameData:
    def test_rename(self, tmp_path):
        db = _populated_db(tmp_path)
        db.rename_data("ds1", "train", "x", "y")
        assert "y" in db.get_var_names("ds1", "train")
        assert "x" not in db.get_var_names("ds1", "train")
        db.close()

    def test_rename_same_name_noop(self, tmp_path):
        db = _populated_db(tmp_path)
        db.rename_data("ds1", "train", "x", "x")
        assert "x" in db.get_var_names("ds1", "train")
        db.close()

    def test_rename_dest_exists_in_array_keys(self, tmp_path):
        db = _populated_db(tmp_path)
        db.add_data("ds1", "train", "y", np.ones((5,), dtype=np.float32))
        with pytest.raises(ValueError, match="already exists"):
            db.rename_data("ds1", "train", "x", "y")
        db.close()

    def test_rename_src_missing(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        db.set_phase("ds1", "train")
        with pytest.raises(ValueError, match="does not exist"):
            db.rename_data("ds1", "train", "missing", "y")
        db.close()

    def test_rename_dest_exists_at_filesystem(self, tmp_path):
        db = _populated_db(tmp_path)
        store_path = Path(db._output_dir)
        dest = store_path / "ds1" / "train" / "y"
        dest.mkdir(parents=True, exist_ok=True)
        with pytest.raises(ValueError, match="already exists"):
            db.rename_data("ds1", "train", "x", "y")
        db.close()

    def test_rename_read_only(self, tmp_path):
        db_w = _make_db(tmp_path, data_id="ds1")
        db_w.add_data("ds1", "train", "x", np.ones((5,), dtype=np.float32))
        db_w.close()
        db = ZarrDatabase(str(tmp_path / "store.zarr"), mode="r")
        db.initialize()
        with pytest.raises(ValueError, match="read only"):
            db.rename_data("ds1", "train", "x", "y")
        db.close()


# ===========================================================================
# get_data_ids / get_phases / get_var_names
# ===========================================================================
class TestListOps:
    def test_get_data_ids(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        db.set_data_id("ds2")
        ids = db.get_data_ids()
        assert set(ids) == {"ds1", "ds2"}
        db.close()

    def test_get_data_ids_empty(self, tmp_path):
        db = _make_db(tmp_path, data_id=None)
        assert db.get_data_ids() == []
        db.close()

    def test_get_phases(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        db.set_phase("ds1", "train")
        db.set_phase("ds1", "test")
        phases = db.get_phases("ds1")
        assert set(phases) == {"train", "test"}
        db.close()

    def test_get_phases_missing_data_id(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        with pytest.raises(ValueError, match="does not exist"):
            db.get_phases("missing")
        db.close()

    def test_get_var_names(self, tmp_path):
        db = _populated_db(tmp_path)
        db.add_data("ds1", "train", "y", np.ones((10, 3), dtype=np.float32))
        names = db.get_var_names("ds1", "train")
        assert set(names) == {"x", "y"}
        db.close()

    def test_get_var_names_missing_phase(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        with pytest.raises(ValueError, match="does not exist"):
            db.get_var_names("ds1", "train")
        db.close()


# ===========================================================================
# get_data_info
# ===========================================================================
class TestGetDataInfo:
    def test_basic(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        db.add_data("ds1", "train", "x", np.ones((7, 4), dtype=np.float64))
        info = db.get_data_info("ds1", "train", "x")
        assert info["dtype"] == "float64"
        assert info["shape"] == (4,)
        assert info["num_events"] == 7
        db.close()

    def test_1d_data(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        db.add_data("ds1", "train", "x", np.ones((5,), dtype=np.int32))
        info = db.get_data_info("ds1", "train", "x")
        assert info["dtype"] == "int32"
        assert info["shape"] == ()
        assert info["num_events"] == 5
        db.close()


# ===========================================================================
# compile
# ===========================================================================
class TestCompile:
    def test_all_compiled(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        db.add_data("ds1", "train", "x", np.ones((10, 3), dtype=np.float32))
        db.add_data("ds1", "train", "y", np.ones((10, 2), dtype=np.float64))
        report = db.compile("ds1")
        assert report["is_compiled"] is True
        assert report["phases"]["train"]["is_compiled"] is True
        assert report["phases"]["train"]["num_events"] == 10
        db.close()

    def test_not_compiled(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        db.add_data("ds1", "train", "x", np.ones((10,), dtype=np.float32))
        db.add_data("ds1", "train", "y", np.ones((5,), dtype=np.float32))
        report = db.compile("ds1")
        assert report["is_compiled"] is False
        assert report["phases"]["train"]["is_compiled"] is False
        assert report["phases"]["train"]["num_events"] is None
        db.close()

    def test_empty_phase(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        db.set_phase("ds1", "train")
        report = db.compile("ds1")
        assert report["is_compiled"] is False
        assert report["phases"]["train"]["is_compiled"] is False
        db.close()

    def test_no_phases(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        report = db.compile("ds1")
        assert report["is_compiled"] is False
        assert report["phases"] == {}
        db.close()


# ===========================================================================
# _resolve_* methods
# ===========================================================================
class TestResolveMethods:
    def test_resolve_db_not_initialized(self, tmp_path):
        db = ZarrDatabase(str(tmp_path / "store.zarr"), mode="a")
        with pytest.raises(RuntimeError, match="not initialized"):
            db._resolve_db()

    def test_resolve_data_id_missing(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        with pytest.raises(ValueError, match="does not exist"):
            db._resolve_data_id("missing")
        db.close()

    def test_resolve_phase_missing(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        with pytest.raises(ValueError, match="does not exist"):
            db._resolve_phase("ds1", "missing")
        db.close()

    def test_resolve_var_name_missing(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        db.set_phase("ds1", "train")
        with pytest.raises(ValueError, match="does not exist"):
            db._resolve_var_name("ds1", "train", "missing")
        db.close()


# ===========================================================================
# _exist_* methods
# ===========================================================================
class TestExistMethods:
    def test_exist_db_false(self, tmp_path):
        db = ZarrDatabase(str(tmp_path / "store.zarr"), mode="a")
        assert db._exist_db() is False

    def test_exist_db_true(self, tmp_path):
        db = _make_db(tmp_path)
        assert db._exist_db() is True
        db.close()

    def test_exist_data_id_false(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        assert db._exist_data_id("ds2") is False
        db.close()

    def test_exist_data_id_true(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        assert db._exist_data_id("ds1") is True
        db.close()

    def test_exist_phase_false(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        assert db._exist_phase("ds1", "train") is False
        db.close()

    def test_exist_phase_true(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        db.set_phase("ds1", "train")
        assert db._exist_phase("ds1", "train") is True
        db.close()

    def test_exist_var_name_false(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        db.set_phase("ds1", "train")
        assert db._exist_var_name("ds1", "train", "x") is False
        db.close()

    def test_exist_var_name_true(self, tmp_path):
        db = _populated_db(tmp_path)
        assert db._exist_var_name("ds1", "train", "x") is True
        db.close()

    def test_exist_data_id_not_initialized(self, tmp_path):
        db = ZarrDatabase(str(tmp_path / "store.zarr"), mode="a")
        assert db._exist_data_id("ds1") is False

    def test_exist_phase_not_initialized(self, tmp_path):
        db = ZarrDatabase(str(tmp_path / "store.zarr"), mode="a")
        assert db._exist_phase("ds1", "train") is False

    def test_exist_var_name_not_initialized(self, tmp_path):
        db = ZarrDatabase(str(tmp_path / "store.zarr"), mode="a")
        assert db._exist_var_name("ds1", "train", "x") is False


# ===========================================================================
# _resolve_chunk
# ===========================================================================
class TestResolveChunk:
    def test_auto_1d(self, tmp_path):
        db = _make_db(tmp_path, chunk="auto", auto_chunk_bytes=16)
        data = np.ones((100,), dtype=np.float32)  # 4 bytes per event
        chunk = db._resolve_chunk(data)
        assert chunk == (4,)  # 16 // 4 = 4
        db.close()

    def test_auto_2d(self, tmp_path):
        db = _make_db(tmp_path, chunk="auto", auto_chunk_bytes=32)
        data = np.ones((100, 4), dtype=np.float32)  # 16 bytes per event
        chunk = db._resolve_chunk(data)
        assert chunk == (2, 4)  # 32 // 16 = 2
        db.close()

    def test_auto_zero_width_features(self, tmp_path):
        """Zero-width feature axes => sample_bytes=0 => chunk_events=max(1, shape[0])."""
        db = _make_db(tmp_path, chunk="auto")
        data = np.ones((10, 0), dtype=np.float32)
        chunk = db._resolve_chunk(data)
        assert chunk == (10, 0)
        db.close()

    def test_auto_zero_events_zero_width(self, tmp_path):
        db = _make_db(tmp_path, chunk="auto")
        data = np.ones((0, 0), dtype=np.float32)
        chunk = db._resolve_chunk(data)
        # max(1, 0) = 1
        assert chunk == (1, 0)
        db.close()

    def test_manual_chunk(self, tmp_path):
        db = _make_db(tmp_path, chunk=7)
        data = np.ones((100, 3), dtype=np.float32)
        chunk = db._resolve_chunk(data)
        assert chunk == (7, 3)
        db.close()


# ===========================================================================
# _get_local_array_path / _get_local_copy_temp_path
# ===========================================================================
class TestLocalPaths:
    def test_get_local_array_path(self, tmp_path):
        db = _populated_db(tmp_path)
        phase_group = db._resolve_phase("ds1", "train")
        path = db._get_local_array_path(phase_group, "x")
        assert path.name == "x"
        assert path.exists()
        db.close()

    def test_get_local_array_path_non_local_store(self, tmp_path):
        db = _populated_db(tmp_path)
        phase_group = db._resolve_phase("ds1", "train")
        # Mock the store to a non-LocalStore
        mock_group = MagicMock(spec=Group)
        mock_group.store = MagicMock()
        mock_group.store.__class__ = type("FakeStore", (), {})
        with pytest.raises(TypeError, match="Atomic rename requires"):
            db._get_local_array_path(mock_group, "x")
        db.close()

    def test_get_local_copy_temp_path(self, tmp_path):
        dest = tmp_path / "myvar"
        result = ZarrDatabase._get_local_copy_temp_path(dest)
        assert ".storegate-copy-tmp" in result.name
        assert not result.exists()

    def test_get_local_copy_temp_path_collision(self, tmp_path):
        dest = tmp_path / "myvar"
        # Create the default temp path to force suffix increment
        collision = dest.with_name(f".{dest.name}.storegate-copy-tmp")
        collision.mkdir(parents=True, exist_ok=True)
        result = ZarrDatabase._get_local_copy_temp_path(dest)
        assert "-1" in result.name

    def test_get_local_copy_temp_path_multiple_collisions(self, tmp_path):
        dest = tmp_path / "myvar"
        # Create multiple collisions
        c0 = dest.with_name(f".{dest.name}.storegate-copy-tmp")
        c0.mkdir(parents=True, exist_ok=True)
        c1 = dest.with_name(f".{dest.name}.storegate-copy-tmp-1")
        c1.mkdir(parents=True, exist_ok=True)
        result = ZarrDatabase._get_local_copy_temp_path(dest)
        assert "-2" in result.name


# ===========================================================================
# _fsync_directory_best_effort
# ===========================================================================
class TestFsyncDirectory:
    def test_fsync_normal(self, tmp_path):
        # Should not raise
        ZarrDatabase._fsync_directory_best_effort(tmp_path)

    def test_fsync_nonexistent_path(self, tmp_path):
        # Should not raise (OSError on open is caught)
        ZarrDatabase._fsync_directory_best_effort(tmp_path / "nonexistent")

    @patch("os.name", "nt")
    def test_fsync_windows_returns_early(self, tmp_path):
        # Should just return on Windows
        ZarrDatabase._fsync_directory_best_effort(tmp_path)

    @patch("os.fsync", side_effect=OSError("fsync failed"))
    def test_fsync_oserror_on_fsync(self, mock_fsync, tmp_path):
        # Should not raise even if fsync fails
        ZarrDatabase._fsync_directory_best_effort(tmp_path)


# ===========================================================================
# _set_marker / _validate_marker
# ===========================================================================
class TestMarker:
    def test_set_marker_idempotent(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        root = db._resolve_db()
        # Already has marker from initialize
        original = root.attrs.get(_STOREGATE_SCHEMA_KEY)
        db._set_marker(root)
        assert root.attrs.get(_STOREGATE_SCHEMA_KEY) == original
        db.close()

    def test_validate_marker_missing(self, tmp_path):
        store_path = tmp_path / "test.zarr"
        g = zarr.open_group(str(store_path), mode="w-")
        db = ZarrDatabase(str(store_path), mode="a")
        with pytest.raises(ValueError, match="missing the storegate schema marker"):
            db._validate_marker(g, "test")

    def test_validate_marker_wrong(self, tmp_path):
        store_path = tmp_path / "test.zarr"
        g = zarr.open_group(str(store_path), mode="w-")
        g.attrs[_STOREGATE_SCHEMA_KEY] = "wrong"
        db = ZarrDatabase(str(store_path), mode="a")
        with pytest.raises(ValueError, match="unsupported"):
            db._validate_marker(g, "test")


# ===========================================================================
# Modes: r+, w-
# ===========================================================================
class TestModes:
    def test_r_plus_mode(self, tmp_path):
        """r+ mode allows read/write on existing store."""
        db_w = _make_db(tmp_path, mode="a", data_id="ds1")
        db_w.close()

        db = ZarrDatabase(str(tmp_path / "store.zarr"), mode="r+")
        db.initialize()
        db.set_data_id("ds2")
        assert set(db.get_data_ids()) == {"ds1", "ds2"}
        db.close()

    def test_w_minus_mode(self, tmp_path):
        """w- mode creates fresh store."""
        db = _make_db(tmp_path, mode="w-", data_id="ds1")
        db.add_data("ds1", "train", "x", np.ones((5,), dtype=np.float32))
        result = db.get_data("ds1", "train", "x")
        assert result.shape == (5,)
        db.close()


# ===========================================================================
# Edge cases
# ===========================================================================
class TestEdgeCases:
    def test_3d_data(self, tmp_path):
        db = _make_db(tmp_path, data_id="ds1")
        data = np.ones((5, 3, 4), dtype=np.float32)
        db.add_data("ds1", "train", "x", data)
        result = db.get_data("ds1", "train", "x")
        assert result.shape == (5, 3, 4)
        info = db.get_data_info("ds1", "train", "x")
        assert info["shape"] == (3, 4)
        db.close()

    def test_large_auto_chunk(self, tmp_path):
        """When data is small, auto chunk covers all events."""
        db = _make_db(tmp_path, chunk="auto", auto_chunk_bytes=_AUTO_CHUNK_BYTES)
        data = np.ones((5, 3), dtype=np.float32)
        db.add_data("ds1", "train", "x", data)
        chunks = list(db.stream_data("ds1", "train", "x"))
        assert len(chunks) == 1
        db.close()

    def test_copy_data_then_modify_original(self, tmp_path):
        db = _populated_db(tmp_path)
        db.copy_data("ds1", "train", "x", "y")
        db.update_data("ds1", "train", "x", np.zeros((3,), dtype=np.float32), 0)
        # y should be independent
        result_y = db.get_data("ds1", "train", "y", 0)
        np.testing.assert_array_equal(result_y, 1.0)
        db.close()

    def test_initialize_empty_dir(self, tmp_path):
        """An empty directory is not considered pre-existing."""
        store_path = tmp_path / "empty.zarr"
        store_path.mkdir()
        db = ZarrDatabase(str(store_path), mode="a")
        db.initialize()  # Should not try to validate schema
        db.close()

    def test_copy_data_exception_during_copytree_cleanup(self, tmp_path):
        """If copytree fails, temp path is cleaned up."""
        db = _populated_db(tmp_path)
        with patch("shutil.copytree", side_effect=OSError("copy failed")):
            with pytest.raises(OSError, match="copy failed"):
                db.copy_data("ds1", "train", "x", "y")
        db.close()
