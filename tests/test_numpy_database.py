"""Comprehensive tests for storegate/database/numpy_database.py and base database.py."""

import copy
from typing import Any

import numpy as np
import pytest

from storegate.database.numpy_database import NumpyDatabase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_db(data_id: str = "ds1") -> NumpyDatabase:
    """Create and initialize a NumpyDatabase with one data_id."""
    db = NumpyDatabase()
    db.initialize(data_id)
    return db


def _populated_db() -> NumpyDatabase:
    """Return a db with ds1/train/x (10, 3) float32."""
    db = _make_db("ds1")
    db.add_data("ds1", "train", "x", np.ones((10, 3), dtype=np.float32))
    return db


# ===========================================================================
# Initialize
# ===========================================================================
class TestInitialize:
    def test_initialize_no_data_id(self):
        db = NumpyDatabase()
        db.initialize()
        assert db.get_data_ids() == []

    def test_initialize_with_data_id(self):
        db = NumpyDatabase()
        db.initialize("ds1")
        assert "ds1" in db.get_data_ids()

    def test_initialize_already_initialized_idempotent(self):
        db = _make_db("ds1")
        db.add_data("ds1", "train", "x", np.ones((5,), dtype=np.float32))
        db.initialize("ds1")  # should not clear existing data
        assert db.get_var_names("ds1", "train") == ["x"]

    def test_initialize_with_new_data_id_on_existing_db(self):
        db = _make_db("ds1")
        db.initialize("ds2")
        assert set(db.get_data_ids()) == {"ds1", "ds2"}

    def test_not_initialized_error(self):
        db = NumpyDatabase()
        with pytest.raises(RuntimeError, match="not initialized"):
            db.get_data_ids()


# ===========================================================================
# get_name / is_writable
# ===========================================================================
class TestBasicProperties:
    def test_get_name(self):
        assert NumpyDatabase().get_name() == "numpy"

    def test_is_writable(self):
        assert NumpyDatabase().is_writable() is True


# ===========================================================================
# __enter__ / __exit__
# ===========================================================================
class TestContextManager:
    def test_enter_exit(self):
        db = NumpyDatabase()
        with db as d:
            assert d is db
            d.set_data_id("ds1")
            assert "ds1" in d.get_data_ids()
        # After exit, db is closed
        with pytest.raises(RuntimeError, match="not initialized"):
            db.get_data_ids()

    def test_exit_on_exception(self):
        db = NumpyDatabase()
        with pytest.raises(ValueError):
            with db as d:
                d.set_data_id("ds1")
                raise ValueError("boom")
        # Still closed after exception
        with pytest.raises(RuntimeError, match="not initialized"):
            db.get_data_ids()


# ===========================================================================
# set_data_id / delete_data_id
# ===========================================================================
class TestDataIdOps:
    def test_set_data_id(self):
        db = _make_db()
        db.set_data_id("ds2")
        assert "ds2" in db.get_data_ids()

    def test_set_data_id_already_exists_noop(self):
        db = _make_db("ds1")
        db.set_data_id("ds1")  # no error
        assert db.get_data_ids() == ["ds1"]

    def test_set_data_id_invalid(self):
        db = _make_db()
        with pytest.raises(ValueError, match="Invalid"):
            db.set_data_id("")

    def test_delete_data_id(self):
        db = _make_db("ds1")
        db.delete_data_id("ds1")
        assert db.get_data_ids() == []

    def test_delete_data_id_not_found(self):
        db = _make_db("ds1")
        with pytest.raises(ValueError, match="does not exist"):
            db.delete_data_id("ds_missing")


# ===========================================================================
# set_phase / delete_phase
# ===========================================================================
class TestPhaseOps:
    def test_set_phase(self):
        db = _make_db("ds1")
        db.set_phase("ds1", "train")
        assert "train" in db.get_phases("ds1")

    def test_set_phase_already_exists_noop(self):
        db = _make_db("ds1")
        db.set_phase("ds1", "train")
        db.set_phase("ds1", "train")  # no error
        assert db.get_phases("ds1") == ["train"]

    def test_set_phase_data_id_missing(self):
        db = _make_db("ds1")
        with pytest.raises(ValueError, match="does not exist"):
            db.set_phase("missing", "train")

    def test_delete_phase(self):
        db = _make_db("ds1")
        db.set_phase("ds1", "train")
        db.delete_phase("ds1", "train")
        assert db.get_phases("ds1") == []

    def test_delete_phase_missing(self):
        db = _make_db("ds1")
        with pytest.raises(ValueError, match="does not exist"):
            db.delete_phase("ds1", "train")


# ===========================================================================
# clear / close
# ===========================================================================
class TestClearClose:
    def test_clear(self):
        db = _make_db("ds1")
        db.add_data("ds1", "train", "x", np.ones((5,), dtype=np.float32))
        db.clear()
        assert db.get_data_ids() == []

    def test_clear_not_initialized(self):
        db = NumpyDatabase()
        with pytest.raises(RuntimeError, match="not initialized"):
            db.clear()

    def test_close(self):
        db = _make_db("ds1")
        db.close()
        with pytest.raises(RuntimeError, match="not initialized"):
            db.get_data_ids()

    def test_close_twice(self):
        db = _make_db("ds1")
        db.close()
        db.close()  # idempotent


# ===========================================================================
# _validate_data (from base Database)
# ===========================================================================
class TestValidateData:
    def _validate(self, data):
        db = _make_db("ds1")
        db._validate_data(data)

    def test_valid_float32(self):
        self._validate(np.ones((5, 3), dtype=np.float32))

    def test_valid_float64(self):
        self._validate(np.ones((5,), dtype=np.float64))

    def test_valid_int32(self):
        self._validate(np.ones((5,), dtype=np.int32))

    def test_valid_int64(self):
        self._validate(np.ones((5,), dtype=np.int64))

    def test_valid_bool(self):
        self._validate(np.ones((5,), dtype=bool))

    def test_valid_complex64(self):
        self._validate(np.ones((5,), dtype=np.complex64))

    def test_masked_array_rejected(self):
        data = np.ma.MaskedArray(np.ones((5,)))
        with pytest.raises(TypeError, match="MaskedArray"):
            self._validate(data)

    def test_not_ndarray_rejected(self):
        with pytest.raises(TypeError, match="Expected numpy.ndarray"):
            self._validate([1, 2, 3])  # type: ignore

    def test_structured_dtype_rejected(self):
        dt = np.dtype([("a", "f4"), ("b", "i4")])
        data = np.zeros(5, dtype=dt)
        with pytest.raises(TypeError, match="Structured dtype"):
            self._validate(data)

    def test_string_dtype_rejected(self):
        data = np.array(["hello", "world"])
        with pytest.raises(ValueError, match="not persistable"):
            self._validate(data)

    def test_bytes_dtype_rejected(self):
        data = np.array([b"hello", b"world"])
        with pytest.raises(ValueError, match="not persistable"):
            self._validate(data)


# ===========================================================================
# _validate_identifiers (from base Database)
# ===========================================================================
class TestValidateIdentifiers:
    def test_valid_all(self):
        db = _make_db()
        db._validate_identifiers("ds1", "train", "x", "y")

    def test_only_data_id(self):
        db = _make_db()
        db._validate_identifiers("ds1")

    def test_invalid_data_id(self):
        db = _make_db()
        with pytest.raises(ValueError, match="Invalid"):
            db._validate_identifiers("")

    def test_invalid_phase(self):
        db = _make_db()
        with pytest.raises(ValueError, match="Invalid"):
            db._validate_identifiers("ds1", "bad phase")

    def test_invalid_var_name(self):
        db = _make_db()
        with pytest.raises(ValueError, match="Invalid"):
            db._validate_identifiers("ds1", "train", "bad.name")

    def test_invalid_output_var_name(self):
        db = _make_db()
        with pytest.raises(ValueError, match="Invalid"):
            db._validate_identifiers("ds1", "train", "x", "bad/name")


# ===========================================================================
# _validate_get_data_index
# ===========================================================================
class TestValidateGetDataIndex:
    def _v(self, index):
        db = _make_db()
        db._validate_get_data_index(index)

    def test_int_valid(self):
        self._v(0)

    def test_np_integer_valid(self):
        self._v(np.int64(3))

    def test_slice_valid(self):
        self._v(slice(None))

    def test_slice_with_start_stop(self):
        self._v(slice(1, 5))

    def test_list_int_valid(self):
        self._v([0, 1, 2])

    def test_list_np_integer_valid(self):
        self._v([np.int64(0), np.int32(1)])

    def test_ndarray_1d_int_valid(self):
        self._v(np.array([0, 1, 2], dtype=np.int64))

    def test_bool_rejected(self):
        with pytest.raises(ValueError, match="must not be bool"):
            self._v(True)

    def test_slice_with_step_rejected(self):
        with pytest.raises(ValueError, match="int, np.integer, slice without step"):
            self._v(slice(0, 5, 2))

    def test_empty_list_rejected(self):
        with pytest.raises(ValueError, match="must not be an empty list"):
            self._v([])

    def test_list_of_bools_rejected(self):
        with pytest.raises(ValueError, match="must not be list\\[bool\\]"):
            self._v([True, False])

    def test_empty_ndarray_rejected(self):
        with pytest.raises(ValueError, match="must not be an empty numpy.ndarray"):
            self._v(np.array([], dtype=np.int64))

    def test_2d_ndarray_rejected(self):
        with pytest.raises(ValueError, match="int, np.integer, slice without step"):
            self._v(np.array([[0, 1]], dtype=np.int64))

    def test_float_ndarray_rejected(self):
        with pytest.raises(ValueError, match="int, np.integer, slice without step"):
            self._v(np.array([1.0, 2.0]))

    def test_string_rejected(self):
        with pytest.raises(ValueError, match="int, np.integer, slice without step"):
            self._v("abc")

    def test_list_of_floats_rejected(self):
        with pytest.raises(ValueError, match="int, np.integer, slice without step"):
            self._v([1.0, 2.0])

    def test_list_mixed_bool_int_rejected(self):
        # A list of all bools triggers bool check; mixed triggers fallthrough
        with pytest.raises(ValueError, match="int, np.integer, slice without step"):
            self._v([True, 1])


# ===========================================================================
# _validate_update_data_index
# ===========================================================================
class TestValidateUpdateDataIndex:
    def _v(self, index):
        db = _make_db()
        db._validate_update_data_index(index)

    def test_int_valid(self):
        self._v(0)

    def test_np_integer_valid(self):
        self._v(np.int64(3))

    def test_slice_valid(self):
        self._v(slice(None))

    def test_bool_rejected(self):
        with pytest.raises(ValueError, match="must not be bool"):
            self._v(True)

    def test_slice_with_step_rejected(self):
        with pytest.raises(ValueError, match="int, np.integer, or slice without step"):
            self._v(slice(0, 5, 2))

    def test_list_rejected(self):
        with pytest.raises(ValueError, match="int, np.integer, or slice without step"):
            self._v([0, 1])

    def test_string_rejected(self):
        with pytest.raises(ValueError, match="int, np.integer, or slice without step"):
            self._v("abc")


# ===========================================================================
# add_data
# ===========================================================================
class TestAddData:
    def test_add_new_data(self):
        db = _make_db("ds1")
        data = np.ones((10, 3), dtype=np.float32)
        db.add_data("ds1", "train", "x", data)
        result = db.get_data("ds1", "train", "x")
        np.testing.assert_array_equal(result, data)

    def test_add_data_creates_phase_automatically(self):
        db = _make_db("ds1")
        db.add_data("ds1", "train", "x", np.ones((5,), dtype=np.float32))
        assert "train" in db.get_phases("ds1")

    def test_add_data_append(self):
        db = _make_db("ds1")
        db.add_data("ds1", "train", "x", np.ones((5, 3), dtype=np.float32))
        db.add_data("ds1", "train", "x", np.zeros((3, 3), dtype=np.float32))
        result = db.get_data("ds1", "train", "x")
        assert result.shape == (8, 3)
        np.testing.assert_array_equal(result[:5], 1.0)
        np.testing.assert_array_equal(result[5:], 0.0)

    def test_add_data_append_dtype_mismatch(self):
        db = _make_db("ds1")
        db.add_data("ds1", "train", "x", np.ones((5, 3), dtype=np.float32))
        with pytest.raises(ValueError, match="expected float32"):
            db.add_data("ds1", "train", "x", np.ones((3, 3), dtype=np.float64))

    def test_add_data_append_shape_mismatch(self):
        db = _make_db("ds1")
        db.add_data("ds1", "train", "x", np.ones((5, 3), dtype=np.float32))
        with pytest.raises(ValueError, match="expected"):
            db.add_data("ds1", "train", "x", np.ones((3, 4), dtype=np.float32))

    def test_add_data_0dim_rejected(self):
        db = _make_db("ds1")
        with pytest.raises(ValueError, match="must be >= 1"):
            db.add_data("ds1", "train", "x", np.array(1.0, dtype=np.float32))

    def test_add_data_defensive_copy(self):
        db = _make_db("ds1")
        data = np.ones((5,), dtype=np.float32)
        db.add_data("ds1", "train", "x", data)
        data[:] = 999  # mutate original
        result = db.get_data("ds1", "train", "x")
        np.testing.assert_array_equal(result, 1.0)

    def test_add_data_invalid_identifiers(self):
        db = _make_db("ds1")
        with pytest.raises(ValueError, match="Invalid"):
            db.add_data("ds1", "bad phase", "x", np.ones((5,), dtype=np.float32))

    def test_add_data_missing_data_id(self):
        db = _make_db("ds1")
        with pytest.raises(ValueError, match="does not exist"):
            db.add_data("missing", "train", "x", np.ones((5,), dtype=np.float32))

    def test_add_data_various_dtypes(self):
        db = _make_db("ds1")
        for dtype in [np.float32, np.float64, np.int32, np.int64, bool, np.complex64]:
            name = f"v_{np.dtype(dtype).name}"
            db.add_data("ds1", "train", name, np.ones((3,), dtype=dtype))
            info = db.get_data_info("ds1", "train", name)
            assert info["dtype"] == np.dtype(dtype).name

    def test_add_data_masked_array(self):
        db = _make_db("ds1")
        data = np.ma.MaskedArray(np.ones((5,)))
        with pytest.raises(TypeError, match="MaskedArray"):
            db.add_data("ds1", "train", "x", data)

    def test_add_data_not_ndarray(self):
        db = _make_db("ds1")
        with pytest.raises(TypeError, match="Expected numpy.ndarray"):
            db.add_data("ds1", "train", "x", [1, 2, 3])  # type: ignore

    def test_add_data_structured_dtype(self):
        db = _make_db("ds1")
        dt = np.dtype([("a", "f4"), ("b", "i4")])
        data = np.zeros(5, dtype=dt)
        with pytest.raises(TypeError, match="Structured dtype"):
            db.add_data("ds1", "train", "x", data)

    def test_add_data_string_dtype(self):
        db = _make_db("ds1")
        with pytest.raises(ValueError, match="not persistable"):
            db.add_data("ds1", "train", "x", np.array(["a", "b"]))


# ===========================================================================
# update_data
# ===========================================================================
class TestUpdateData:
    def test_update_single_int(self):
        db = _populated_db()
        new_row = np.zeros((3,), dtype=np.float32)
        db.update_data("ds1", "train", "x", new_row, 0)
        result = db.get_data("ds1", "train", "x", 0)
        np.testing.assert_array_equal(result, 0.0)

    def test_update_single_np_integer(self):
        db = _populated_db()
        new_row = np.zeros((3,), dtype=np.float32)
        db.update_data("ds1", "train", "x", new_row, np.int64(2))
        result = db.get_data("ds1", "train", "x", 2)
        np.testing.assert_array_equal(result, 0.0)

    def test_update_slice(self):
        db = _populated_db()
        new_data = np.zeros((3, 3), dtype=np.float32)
        db.update_data("ds1", "train", "x", new_data, slice(0, 3))
        result = db.get_data("ds1", "train", "x")
        np.testing.assert_array_equal(result[:3], 0.0)
        np.testing.assert_array_equal(result[3:], 1.0)

    def test_update_slice_wrong_events(self):
        db = _populated_db()
        new_data = np.zeros((5, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="expected 3 events"):
            db.update_data("ds1", "train", "x", new_data, slice(0, 3))

    def test_update_single_dtype_mismatch(self):
        db = _populated_db()
        new_row = np.zeros((3,), dtype=np.float64)
        with pytest.raises(ValueError, match="expected float32"):
            db.update_data("ds1", "train", "x", new_row, 0)

    def test_update_single_shape_mismatch(self):
        db = _populated_db()
        new_row = np.zeros((4,), dtype=np.float32)
        with pytest.raises(ValueError, match="expected"):
            db.update_data("ds1", "train", "x", new_row, 0)

    def test_update_batch_dtype_mismatch(self):
        db = _populated_db()
        new_data = np.zeros((3, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="expected float32"):
            db.update_data("ds1", "train", "x", new_data, slice(0, 3))

    def test_update_batch_shape_mismatch(self):
        db = _populated_db()
        new_data = np.zeros((3, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="expected"):
            db.update_data("ds1", "train", "x", new_data, slice(0, 3))

    def test_update_invalid_index_type(self):
        db = _populated_db()
        with pytest.raises(ValueError, match="int, np.integer, or slice without step"):
            db.update_data("ds1", "train", "x", np.zeros((3,), dtype=np.float32), [0])  # type: ignore

    def test_update_bool_index_rejected(self):
        db = _populated_db()
        with pytest.raises(ValueError, match="must not be bool"):
            db.update_data("ds1", "train", "x", np.zeros((3,), dtype=np.float32), True)  # type: ignore

    def test_update_slice_with_step_rejected(self):
        db = _populated_db()
        with pytest.raises(ValueError, match="int, np.integer, or slice without step"):
            db.update_data("ds1", "train", "x", np.zeros((3, 3), dtype=np.float32), slice(0, 3, 1))

    def test_update_missing_var(self):
        db = _make_db("ds1")
        db.set_phase("ds1", "train")
        with pytest.raises(ValueError, match="does not exist"):
            db.update_data("ds1", "train", "x", np.zeros((3,), dtype=np.float32), 0)

    def test_update_masked_array(self):
        db = _populated_db()
        data = np.ma.MaskedArray(np.zeros((3,), dtype=np.float32))
        with pytest.raises(TypeError, match="MaskedArray"):
            db.update_data("ds1", "train", "x", data, 0)


# ===========================================================================
# get_data
# ===========================================================================
class TestGetData:
    def test_get_all(self):
        db = _populated_db()
        result = db.get_data("ds1", "train", "x")
        assert result.shape == (10, 3)

    def test_get_int_index(self):
        db = _populated_db()
        result = db.get_data("ds1", "train", "x", 0)
        assert result.shape == (3,)

    def test_get_np_integer_index(self):
        db = _populated_db()
        result = db.get_data("ds1", "train", "x", np.int64(0))
        assert result.shape == (3,)

    def test_get_slice_index(self):
        db = _populated_db()
        result = db.get_data("ds1", "train", "x", slice(0, 5))
        assert result.shape == (5, 3)

    def test_get_list_index(self):
        db = _populated_db()
        result = db.get_data("ds1", "train", "x", [0, 2, 4])
        assert result.shape == (3, 3)

    def test_get_ndarray_index(self):
        db = _populated_db()
        idx = np.array([0, 2, 4], dtype=np.int64)
        result = db.get_data("ds1", "train", "x", idx)
        assert result.shape == (3, 3)

    def test_get_defensive_copy(self):
        db = _populated_db()
        result = db.get_data("ds1", "train", "x")
        result[:] = 999
        result2 = db.get_data("ds1", "train", "x")
        np.testing.assert_array_equal(result2, 1.0)

    def test_get_missing_var(self):
        db = _make_db("ds1")
        db.set_phase("ds1", "train")
        with pytest.raises(ValueError, match="does not exist"):
            db.get_data("ds1", "train", "x")


# ===========================================================================
# delete_data
# ===========================================================================
class TestDeleteData:
    def test_delete(self):
        db = _populated_db()
        db.delete_data("ds1", "train", "x")
        assert db.get_var_names("ds1", "train") == []

    def test_delete_missing(self):
        db = _make_db("ds1")
        db.set_phase("ds1", "train")
        with pytest.raises(ValueError, match="does not exist"):
            db.delete_data("ds1", "train", "x")


# ===========================================================================
# stream_data
# ===========================================================================
class TestStreamData:
    def test_stream(self):
        db = _populated_db()
        chunks = list(db.stream_data("ds1", "train", "x"))
        assert len(chunks) == 1
        np.testing.assert_array_equal(chunks[0], np.ones((10, 3), dtype=np.float32))

    def test_stream_defensive_copy(self):
        db = _populated_db()
        for chunk in db.stream_data("ds1", "train", "x"):
            chunk[:] = 999
        result = db.get_data("ds1", "train", "x")
        np.testing.assert_array_equal(result, 1.0)


# ===========================================================================
# copy_data
# ===========================================================================
class TestCopyData:
    def test_copy(self):
        db = _populated_db()
        db.copy_data("ds1", "train", "x", "y")
        result = db.get_data("ds1", "train", "y")
        np.testing.assert_array_equal(result, np.ones((10, 3), dtype=np.float32))

    def test_copy_same_name_noop(self):
        db = _populated_db()
        db.copy_data("ds1", "train", "x", "x")  # no error
        assert db.get_var_names("ds1", "train") == ["x"]

    def test_copy_dest_exists(self):
        db = _populated_db()
        db.add_data("ds1", "train", "y", np.ones((5,), dtype=np.float32))
        with pytest.raises(ValueError, match="already exists"):
            db.copy_data("ds1", "train", "x", "y")

    def test_copy_src_missing(self):
        db = _make_db("ds1")
        db.set_phase("ds1", "train")
        with pytest.raises(ValueError, match="does not exist"):
            db.copy_data("ds1", "train", "missing", "y")

    def test_copy_is_deep(self):
        db = _populated_db()
        db.copy_data("ds1", "train", "x", "y")
        # Modify original; copy should be unchanged
        db.update_data("ds1", "train", "x", np.zeros((3,), dtype=np.float32), 0)
        result = db.get_data("ds1", "train", "y", 0)
        np.testing.assert_array_equal(result, 1.0)


# ===========================================================================
# rename_data
# ===========================================================================
class TestRenameData:
    def test_rename(self):
        db = _populated_db()
        db.rename_data("ds1", "train", "x", "y")
        assert "y" in db.get_var_names("ds1", "train")
        assert "x" not in db.get_var_names("ds1", "train")

    def test_rename_same_name_noop(self):
        db = _populated_db()
        db.rename_data("ds1", "train", "x", "x")
        assert db.get_var_names("ds1", "train") == ["x"]

    def test_rename_dest_exists(self):
        db = _populated_db()
        db.add_data("ds1", "train", "y", np.ones((5,), dtype=np.float32))
        with pytest.raises(ValueError, match="already exists"):
            db.rename_data("ds1", "train", "x", "y")

    def test_rename_src_missing(self):
        db = _make_db("ds1")
        db.set_phase("ds1", "train")
        with pytest.raises(ValueError, match="does not exist"):
            db.rename_data("ds1", "train", "missing", "y")


# ===========================================================================
# get_data_ids / get_phases / get_var_names
# ===========================================================================
class TestListOps:
    def test_get_data_ids(self):
        db = _make_db("ds1")
        db.set_data_id("ds2")
        ids = db.get_data_ids()
        assert set(ids) == {"ds1", "ds2"}

    def test_get_phases(self):
        db = _make_db("ds1")
        db.set_phase("ds1", "train")
        db.set_phase("ds1", "test")
        phases = db.get_phases("ds1")
        assert set(phases) == {"train", "test"}

    def test_get_phases_missing_data_id(self):
        db = _make_db("ds1")
        with pytest.raises(ValueError, match="does not exist"):
            db.get_phases("missing")

    def test_get_var_names(self):
        db = _populated_db()
        db.add_data("ds1", "train", "y", np.ones((10, 3), dtype=np.float32))
        names = db.get_var_names("ds1", "train")
        assert set(names) == {"x", "y"}

    def test_get_var_names_missing_phase(self):
        db = _make_db("ds1")
        with pytest.raises(ValueError, match="does not exist"):
            db.get_var_names("ds1", "train")


# ===========================================================================
# get_data_info
# ===========================================================================
class TestGetDataInfo:
    def test_from_chunks(self):
        db = _make_db("ds1")
        db.add_data("ds1", "train", "x", np.ones((5, 3), dtype=np.float32))
        db.add_data("ds1", "train", "x", np.ones((3, 3), dtype=np.float32))
        # Before materialization, cache is None - info from chunks
        info = db.get_data_info("ds1", "train", "x")
        assert info["dtype"] == "float32"
        assert info["shape"] == (3,)
        assert info["num_events"] == 8

    def test_from_cache(self):
        db = _make_db("ds1")
        db.add_data("ds1", "train", "x", np.ones((5, 3), dtype=np.float32))
        db.add_data("ds1", "train", "x", np.ones((3, 3), dtype=np.float32))
        # Access data to materialize/cache
        _ = db.get_data("ds1", "train", "x")
        info = db.get_data_info("ds1", "train", "x")
        assert info["dtype"] == "float32"
        assert info["shape"] == (3,)
        assert info["num_events"] == 8

    def test_1d_data(self):
        db = _make_db("ds1")
        db.add_data("ds1", "train", "x", np.ones((7,), dtype=np.int64))
        info = db.get_data_info("ds1", "train", "x")
        assert info["dtype"] == "int64"
        assert info["shape"] == ()
        assert info["num_events"] == 7


# ===========================================================================
# compile
# ===========================================================================
class TestCompile:
    def test_all_compiled(self):
        db = _make_db("ds1")
        db.add_data("ds1", "train", "x", np.ones((10, 3), dtype=np.float32))
        db.add_data("ds1", "train", "y", np.ones((10, 2), dtype=np.float64))
        report = db.compile("ds1")
        assert report["is_compiled"] is True
        assert report["data_id"] == "ds1"
        assert report["phases"]["train"]["is_compiled"] is True
        assert report["phases"]["train"]["num_events"] == 10

    def test_not_compiled_different_events(self):
        db = _make_db("ds1")
        db.add_data("ds1", "train", "x", np.ones((10, 3), dtype=np.float32))
        db.add_data("ds1", "train", "y", np.ones((5, 2), dtype=np.float64))
        report = db.compile("ds1")
        assert report["is_compiled"] is False
        assert report["phases"]["train"]["is_compiled"] is False
        assert report["phases"]["train"]["num_events"] is None

    def test_empty_phase(self):
        db = _make_db("ds1")
        db.set_phase("ds1", "train")
        report = db.compile("ds1")
        assert report["is_compiled"] is False
        assert report["phases"]["train"]["is_compiled"] is False
        assert report["phases"]["train"]["num_events"] is None

    def test_no_phases(self):
        db = _make_db("ds1")
        report = db.compile("ds1")
        assert report["is_compiled"] is False
        assert report["phases"] == {}

    def test_compile_with_cache(self):
        db = _make_db("ds1")
        db.add_data("ds1", "train", "x", np.ones((5, 3), dtype=np.float32))
        db.add_data("ds1", "train", "x", np.ones((5, 3), dtype=np.float32))
        # Materialize cache
        _ = db.get_data("ds1", "train", "x")
        report = db.compile("ds1")
        assert report["phases"]["train"]["vars"]["x"] == 10

    def test_compile_multiple_phases(self):
        db = _make_db("ds1")
        db.add_data("ds1", "train", "x", np.ones((10,), dtype=np.float32))
        db.add_data("ds1", "test", "x", np.ones((5,), dtype=np.float32))
        report = db.compile("ds1")
        assert report["is_compiled"] is True
        assert report["phases"]["train"]["num_events"] == 10
        assert report["phases"]["test"]["num_events"] == 5

    def test_compile_partial_compiled(self):
        """One phase compiled, another not -> overall not compiled."""
        db = _make_db("ds1")
        db.add_data("ds1", "train", "x", np.ones((10,), dtype=np.float32))
        db.set_phase("ds1", "test")  # empty phase
        report = db.compile("ds1")
        assert report["is_compiled"] is False


# ===========================================================================
# _resolve_* methods
# ===========================================================================
class TestResolveMethods:
    def test_resolve_db_not_initialized(self):
        db = NumpyDatabase()
        with pytest.raises(RuntimeError, match="not initialized"):
            db._resolve_db()

    def test_resolve_data_id_missing(self):
        db = _make_db("ds1")
        with pytest.raises(ValueError, match="does not exist"):
            db._resolve_data_id("missing")

    def test_resolve_phase_missing(self):
        db = _make_db("ds1")
        with pytest.raises(ValueError, match="does not exist"):
            db._resolve_phase("ds1", "missing")

    def test_resolve_var_entry_missing(self):
        db = _make_db("ds1")
        db.set_phase("ds1", "train")
        with pytest.raises(ValueError, match="does not exist"):
            db._resolve_var_entry("ds1", "train", "missing")

    def test_resolve_var_name_missing(self):
        db = _make_db("ds1")
        db.set_phase("ds1", "train")
        with pytest.raises(ValueError, match="does not exist"):
            db._resolve_var_name("ds1", "train", "missing")


# ===========================================================================
# _exist_* methods
# ===========================================================================
class TestExistMethods:
    def test_exist_db_false(self):
        db = NumpyDatabase()
        assert db._exist_db() is False

    def test_exist_db_true(self):
        db = _make_db()
        assert db._exist_db() is True

    def test_exist_data_id_false(self):
        db = _make_db("ds1")
        assert db._exist_data_id("ds2") is False

    def test_exist_data_id_true(self):
        db = _make_db("ds1")
        assert db._exist_data_id("ds1") is True

    def test_exist_phase_false(self):
        db = _make_db("ds1")
        assert db._exist_phase("ds1", "train") is False

    def test_exist_phase_true(self):
        db = _make_db("ds1")
        db.set_phase("ds1", "train")
        assert db._exist_phase("ds1", "train") is True

    def test_exist_var_name_false(self):
        db = _make_db("ds1")
        db.set_phase("ds1", "train")
        assert db._exist_var_name("ds1", "train", "x") is False

    def test_exist_var_name_true(self):
        db = _populated_db()
        assert db._exist_var_name("ds1", "train", "x") is True

    def test_exist_data_id_not_initialized(self):
        db = NumpyDatabase()
        assert db._exist_data_id("ds1") is False

    def test_exist_phase_not_initialized(self):
        db = NumpyDatabase()
        assert db._exist_phase("ds1", "train") is False

    def test_exist_var_name_not_initialized(self):
        db = NumpyDatabase()
        assert db._exist_var_name("ds1", "train", "x") is False


# ===========================================================================
# _materialize
# ===========================================================================
class TestMaterialize:
    def test_materialize_cache_miss(self):
        db = _make_db("ds1")
        db.add_data("ds1", "train", "x", np.ones((3,), dtype=np.float32))
        db.add_data("ds1", "train", "x", np.zeros((2,), dtype=np.float32))
        entry = db._resolve_var_entry("ds1", "train", "x")
        assert entry["cache"] is None
        result = db._materialize(entry)
        assert result.shape == (5,)
        assert entry["cache"] is not None

    def test_materialize_cache_hit(self):
        db = _make_db("ds1")
        db.add_data("ds1", "train", "x", np.ones((3,), dtype=np.float32))
        entry = db._resolve_var_entry("ds1", "train", "x")
        r1 = db._materialize(entry)
        r2 = db._materialize(entry)
        assert r1 is r2  # same cached object

    def test_add_data_invalidates_cache(self):
        db = _make_db("ds1")
        db.add_data("ds1", "train", "x", np.ones((3,), dtype=np.float32))
        _ = db.get_data("ds1", "train", "x")  # materialize
        entry = db._resolve_var_entry("ds1", "train", "x")
        assert entry["cache"] is not None
        db.add_data("ds1", "train", "x", np.zeros((2,), dtype=np.float32))
        entry = db._resolve_var_entry("ds1", "train", "x")
        assert entry["cache"] is None


# ===========================================================================
# staged_add
# ===========================================================================
class TestStagedAdd:
    def test_staged_add_basic(self):
        db = _make_db("ds1")
        with db.staged_add("ds1", "train", ["x", "y"]) as tx:
            tx.add_data("x", np.ones((5, 3), dtype=np.float32))
            tx.add_data("y", np.ones((5, 2), dtype=np.float64))
        assert set(db.get_var_names("ds1", "train")) == {"x", "y"}

    def test_staged_add_rollback_on_exception(self):
        db = _make_db("ds1")
        with pytest.raises(ValueError, match="boom"):
            with db.staged_add("ds1", "train", ["x"]) as tx:
                tx.add_data("x", np.ones((5,), dtype=np.float32))
                raise ValueError("boom")
        # x should not be published
        assert db.get_phases("ds1") == [] or "x" not in db.get_var_names("ds1", "train")

    def test_staged_add_outside_context(self):
        db = _make_db("ds1")
        tx = db.staged_add("ds1", "train", ["x"])
        with pytest.raises(RuntimeError, match="outside of staged_add"):
            tx.add_data("x", np.ones((5,), dtype=np.float32))

    def test_staged_add_undeclared_name(self):
        db = _make_db("ds1")
        with db.staged_add("ds1", "train", ["x"]) as tx:
            with pytest.raises(ValueError, match="not declared"):
                tx.add_data("y", np.ones((5,), dtype=np.float32))

    def test_staged_add_empty_var_names(self):
        db = _make_db("ds1")
        with pytest.raises(ValueError, match="must not be empty"):
            with db.staged_add("ds1", "train", []) as tx:
                pass

    def test_staged_add_duplicate_var_names(self):
        db = _make_db("ds1")
        with pytest.raises(ValueError, match="duplicate"):
            with db.staged_add("ds1", "train", ["x", "x"]) as tx:
                pass

    def test_staged_add_existing_var(self):
        db = _make_db("ds1")
        db.add_data("ds1", "train", "x", np.ones((5,), dtype=np.float32))
        with pytest.raises(ValueError, match="already exists"):
            with db.staged_add("ds1", "train", ["x"]) as tx:
                pass


# ===========================================================================
# _is_batch_data / _match_data_info (from base Database)
# ===========================================================================
class TestBaseValidation:
    def test_is_batch_data_0dim(self):
        db = _make_db("ds1")
        with pytest.raises(ValueError, match="must be >= 1"):
            db._is_batch_data("ds1", "train", "x", np.float32(1.0))

    def test_match_data_info_dtype_mismatch(self):
        db = _populated_db()
        with pytest.raises(ValueError, match="expected float32"):
            db._match_data_info("ds1", "train", "x", np.ones((3,), dtype=np.float64))

    def test_match_data_info_shape_mismatch(self):
        db = _populated_db()
        with pytest.raises(ValueError, match="expected"):
            db._match_data_info("ds1", "train", "x", np.ones((4,), dtype=np.float32))

    def test_match_data_info_ignore_first_axis(self):
        db = _populated_db()
        # shape is (3,) when ignoring first axis of (N, 3)
        db._match_data_info(
            "ds1", "train", "x", np.ones((99, 3), dtype=np.float32), ignore_first_axis=True
        )

    def test_match_data_info_ignore_first_axis_shape_mismatch(self):
        db = _populated_db()
        with pytest.raises(ValueError, match="expected"):
            db._match_data_info(
                "ds1", "train", "x", np.ones((99, 4), dtype=np.float32), ignore_first_axis=True
            )
