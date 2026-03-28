"""Tests for storegate/database/staged_add.py."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from storegate.database.numpy_database import NumpyDatabase
from storegate.database.zarr_database import ZarrDatabase
from storegate.database.staged_add import _StagedAddTransaction, _allocate_temp_name


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def numpy_db():
    db = NumpyDatabase()
    db.initialize("d1")
    db.set_phase("d1", "train")
    return db


@pytest.fixture
def zarr_db(tmp_path):
    db = ZarrDatabase(str(tmp_path / "zarr_store"), mode="a")
    db.initialize("d1")
    db.set_phase("d1", "train")
    return db


# ---------------------------------------------------------------------------
# _allocate_temp_name
# ---------------------------------------------------------------------------
class TestAllocateTempName:
    def test_no_collision(self):
        result = _allocate_temp_name("x", set())
        assert result == "_db_staged_x"

    def test_collision_handling(self):
        forbidden = {"_db_staged_x"}
        result = _allocate_temp_name("x", forbidden)
        assert result == "_db_staged_x_1"

    def test_multiple_collisions(self):
        forbidden = {"_db_staged_x", "_db_staged_x_1", "_db_staged_x_2"}
        result = _allocate_temp_name("x", forbidden)
        assert result == "_db_staged_x_3"


# ---------------------------------------------------------------------------
# _StagedAddTransaction with NumpyDatabase
# ---------------------------------------------------------------------------
class TestStagedAddTransactionNumpy:
    def test_normal_commit_flow(self, numpy_db):
        data = np.array([[1.0, 2.0]])
        with numpy_db.staged_add("d1", "train", ["x", "y"]) as tx:
            tx.add_data("x", data)
            tx.add_data("y", data * 2)

        assert "x" in numpy_db.get_var_names("d1", "train")
        assert "y" in numpy_db.get_var_names("d1", "train")
        np.testing.assert_array_equal(
            numpy_db.get_data("d1", "train", "x", slice(None)),
            data,
        )

    def test_exception_during_body_cleanup(self, numpy_db):
        data = np.array([[1.0]])
        with pytest.raises(ValueError, match="test error"):
            with numpy_db.staged_add("d1", "train", ["x"]) as tx:
                tx.add_data("x", data)
                raise ValueError("test error")

        # Temp data should be cleaned up
        assert "x" not in numpy_db.get_var_names("d1", "train")

    def test_commit_failure_and_rollback(self, numpy_db):
        """If rename fails during commit, rollback and cleanup happen."""
        data = np.array([[1.0]])
        original_rename = numpy_db.rename_data

        call_count = [0]
        def failing_rename(data_id, phase, src, dst):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("rename failed")
            return original_rename(data_id, phase, src, dst)

        with pytest.raises(RuntimeError, match="rename failed"):
            with numpy_db.staged_add("d1", "train", ["a", "b"]) as tx:
                tx.add_data("a", data)
                tx.add_data("b", data)
                numpy_db.rename_data = failing_rename

        numpy_db.rename_data = original_rename

    def test_commit_failure_rollback_failure_exception_group(self, numpy_db):
        """If rename fails AND rollback fails, an ExceptionGroup is raised."""
        data = np.array([[1.0]])
        original_rename = numpy_db.rename_data

        call_count = [0]
        def failing_rename(data_id, phase, src, dst):
            call_count[0] += 1
            if call_count[0] == 1:
                return original_rename(data_id, phase, src, dst)
            raise RuntimeError(f"rename fail {call_count[0]}")

        with pytest.raises(ExceptionGroup):
            with numpy_db.staged_add("d1", "train", ["a", "b"]) as tx:
                tx.add_data("a", data)
                tx.add_data("b", data)
                numpy_db.rename_data = failing_rename

        numpy_db.rename_data = original_rename

    def test_empty_var_names_error(self, numpy_db):
        with pytest.raises(ValueError, match="var_names must not be empty"):
            with numpy_db.staged_add("d1", "train", []) as tx:
                pass

    def test_duplicate_var_names(self, numpy_db):
        with pytest.raises(ValueError, match="duplicate"):
            with numpy_db.staged_add("d1", "train", ["x", "x"]) as tx:
                pass

    def test_var_already_exists(self, numpy_db):
        numpy_db.add_data("d1", "train", "x", np.array([[1.0]]))
        with pytest.raises(ValueError, match="already exists"):
            with numpy_db.staged_add("d1", "train", ["x"]) as tx:
                pass

    def test_add_data_outside_context(self, numpy_db):
        tx = numpy_db.staged_add("d1", "train", ["x"])
        with pytest.raises(RuntimeError, match="outside of staged_add context"):
            tx.add_data("x", np.array([[1.0]]))

    def test_add_data_for_undeclared_name(self, numpy_db):
        with pytest.raises(ValueError, match="not declared"):
            with numpy_db.staged_add("d1", "train", ["x"]) as tx:
                tx.add_data("z", np.array([[1.0]]))

    def test_multiple_add_data_calls(self, numpy_db):
        """Multiple add_data calls should append."""
        d1 = np.array([[1.0]])
        d2 = np.array([[2.0]])
        with numpy_db.staged_add("d1", "train", ["x"]) as tx:
            tx.add_data("x", d1)
            tx.add_data("x", d2)

        result = numpy_db.get_data("d1", "train", "x", slice(None))
        assert result.shape[0] == 2

    def test_phase_autocreated(self, numpy_db):
        """Phase is created automatically if not existing (via add_data in the backend)."""
        data = np.array([[1.0]])
        with numpy_db.staged_add("d1", "newphase", ["x"]) as tx:
            tx.add_data("x", data)
        assert "x" in numpy_db.get_var_names("d1", "newphase")


# ---------------------------------------------------------------------------
# _StagedAddTransaction with ZarrDatabase
# ---------------------------------------------------------------------------
class TestStagedAddTransactionZarr:
    def test_normal_commit_flow(self, zarr_db):
        data = np.array([[1.0, 2.0]], dtype=np.float32)
        with zarr_db.staged_add("d1", "train", ["x"]) as tx:
            tx.add_data("x", data)

        assert "x" in zarr_db.get_var_names("d1", "train")
        np.testing.assert_array_equal(
            zarr_db.get_data("d1", "train", "x", slice(None)),
            data,
        )

    def test_exception_during_body_cleanup(self, zarr_db):
        data = np.array([[1.0]], dtype=np.float32)
        with pytest.raises(RuntimeError, match="zarr test"):
            with zarr_db.staged_add("d1", "train", ["x"]) as tx:
                tx.add_data("x", data)
                raise RuntimeError("zarr test")

        # temp data cleaned up
        var_names = zarr_db.get_var_names("d1", "train")
        assert "x" not in var_names
        # temp names should also be cleaned up
        assert not any(n.startswith("_db_staged") for n in var_names)

    def test_empty_var_names_error(self, zarr_db):
        with pytest.raises(ValueError, match="var_names must not be empty"):
            with zarr_db.staged_add("d1", "train", []) as tx:
                pass

    def test_duplicate_var_names_zarr(self, zarr_db):
        with pytest.raises(ValueError, match="duplicate"):
            with zarr_db.staged_add("d1", "train", ["x", "x"]) as tx:
                pass

    def test_var_already_exists_zarr(self, zarr_db):
        zarr_db.add_data("d1", "train", "x", np.array([[1.0]], dtype=np.float32))
        with pytest.raises(ValueError, match="already exists"):
            with zarr_db.staged_add("d1", "train", ["x"]) as tx:
                pass

    def test_add_data_outside_context_zarr(self, zarr_db):
        tx = zarr_db.staged_add("d1", "train", ["x"])
        with pytest.raises(RuntimeError, match="outside of staged_add context"):
            tx.add_data("x", np.array([[1.0]], dtype=np.float32))


# ---------------------------------------------------------------------------
# _StagedAddTransaction edge cases
# ---------------------------------------------------------------------------
class TestStagedAddEdgeCases:
    def test_cleanup_with_delete_failure(self, numpy_db):
        """If delete_data fails during cleanup, it should be silently ignored."""
        data = np.array([[1.0]])

        original_delete = numpy_db.delete_data
        def failing_delete(data_id, phase, var_name):
            raise RuntimeError("delete failed")

        with pytest.raises(ValueError, match="test"):
            with numpy_db.staged_add("d1", "train", ["x"]) as tx:
                tx.add_data("x", data)
                numpy_db.delete_data = failing_delete
                raise ValueError("test")

        numpy_db.delete_data = original_delete

    def test_unwritten_var_not_published(self, numpy_db):
        """If a declared var has no add_data calls, it is not published or cleaned."""
        data = np.array([[1.0]])
        with numpy_db.staged_add("d1", "train", ["x", "y"]) as tx:
            tx.add_data("x", data)
            # y is never written

        assert "x" in numpy_db.get_var_names("d1", "train")
        # y was never written so it should not exist
        assert "y" not in numpy_db.get_var_names("d1", "train")
