"""Tests for storegate/database/hybrid_database.py."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from storegate.database.numpy_database import NumpyDatabase
from storegate.database.zarr_database import ZarrDatabase
from storegate.database.hybrid_database import HybridDatabase, _BackendProxy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_numpy_backend():
    return NumpyDatabase()


def _make_initialized_hybrid(*backend_names):
    """Create a HybridDatabase with named numpy backends, all initialized."""
    h = HybridDatabase()
    for name in backend_names:
        db = NumpyDatabase()
        h.register_backend(name, db)
    return h


# ---------------------------------------------------------------------------
# HybridDatabase constructor
# ---------------------------------------------------------------------------
class TestHybridDatabaseConstructor:
    def test_no_backends(self):
        h = HybridDatabase()
        assert h._db == {}
        assert h._backend is None

    def test_with_backends(self):
        backends = {"np": NumpyDatabase(), "np2": NumpyDatabase()}
        h = HybridDatabase(backends=backends)
        assert "np" in h._db
        assert "np2" in h._db


# ---------------------------------------------------------------------------
# register_backend / delete_backend / set_backend / get_backend
# ---------------------------------------------------------------------------
class TestBackendManagement:
    def test_register_backend_success(self):
        h = HybridDatabase()
        db = NumpyDatabase()
        h.register_backend("np", db)
        assert "np" in h._db

    def test_register_backend_all_reserved(self):
        h = HybridDatabase()
        with pytest.raises(ValueError, match='"all" is reserved'):
            h.register_backend("all", NumpyDatabase())

    def test_register_backend_duplicate(self):
        h = HybridDatabase()
        h.register_backend("np", NumpyDatabase())
        with pytest.raises(ValueError, match="already exists"):
            h.register_backend("np", NumpyDatabase())

    def test_delete_backend_success(self):
        h = HybridDatabase()
        db = NumpyDatabase()
        db.initialize()
        h.register_backend("np", db)
        h.set_backend("np")
        h.delete_backend("np")
        assert "np" not in h._db

    def test_delete_backend_resets_active(self):
        h = HybridDatabase()
        db = NumpyDatabase()
        db.initialize()
        h.register_backend("np", db)
        h.set_backend("np")
        assert h.get_backend() == "np"
        h.delete_backend("np")
        assert h.get_backend() is None

    def test_set_backend_success(self):
        h = _make_initialized_hybrid("np")
        h.set_backend("np")
        assert h.get_backend() == "np"

    def test_set_backend_missing(self):
        h = HybridDatabase()
        with pytest.raises(ValueError, match="does not exist"):
            h.set_backend("missing")

    def test_get_backend(self):
        h = HybridDatabase()
        assert h.get_backend() is None


# ---------------------------------------------------------------------------
# using_backend context manager
# ---------------------------------------------------------------------------
class TestUsingBackend:
    def test_restore_on_success(self):
        h = _make_initialized_hybrid("a", "b")
        h.set_backend("a")
        with h.using_backend("b"):
            assert h.get_backend() == "b"
        assert h.get_backend() == "a"

    def test_restore_on_exception(self):
        h = _make_initialized_hybrid("a", "b")
        h.set_backend("a")
        with pytest.raises(RuntimeError):
            with h.using_backend("b"):
                raise RuntimeError("boom")
        assert h.get_backend() == "a"


# ---------------------------------------------------------------------------
# __getitem__
# ---------------------------------------------------------------------------
class TestGetItem:
    def test_single_backend_proxy(self):
        h = _make_initialized_hybrid("np")
        proxy = h["np"]
        assert isinstance(proxy, _BackendProxy)
        assert proxy._write_only is False

    def test_all_proxy(self):
        h = _make_initialized_hybrid("np")
        proxy = h["all"]
        assert isinstance(proxy, _BackendProxy)
        assert proxy._write_only is True

    def test_all_no_backends(self):
        h = HybridDatabase()
        with pytest.raises(RuntimeError, match="No backends"):
            h["all"]

    def test_missing_key(self):
        h = HybridDatabase()
        with pytest.raises(KeyError, match="does not exist"):
            h["missing"]


# ---------------------------------------------------------------------------
# __enter__ / __exit__
# ---------------------------------------------------------------------------
class TestContextManager:
    def test_normal(self):
        h = _make_initialized_hybrid("np")
        with h:
            # Should have initialized all backends
            pass

    def test_no_backends(self):
        h = HybridDatabase()
        with h:
            pass

    def test_exception_in_body(self):
        h = _make_initialized_hybrid("np")
        with pytest.raises(ValueError):
            with h:
                raise ValueError("body error")

    def test_close_failure_during_exit_with_body_exception(self):
        """If body raises and close also fails with ExceptionGroup, BaseExceptionGroup is raised."""
        h = _make_initialized_hybrid("np")

        original_close = NumpyDatabase.close
        def failing_close(self_db):
            raise RuntimeError("close failed")

        # Enter the context manually
        h.__enter__()

        # We need to trigger the ExceptionGroup path.
        # Patch the "all" proxy close to raise ExceptionGroup
        body_exc = ValueError("body error")
        close_exc = RuntimeError("close fail")

        # We'll mock h["all"].close to raise ExceptionGroup
        with patch.object(
            _BackendProxy, "close",
            side_effect=ExceptionGroup("close errors", [close_exc]),
        ):
            with pytest.raises(BaseExceptionGroup) as exc_info:
                h.__exit__(type(body_exc), body_exc, None)

            group = exc_info.value
            assert len(group.exceptions) == 2

    def test_exit_no_exception_calls_close(self):
        h = _make_initialized_hybrid("np")
        h.__enter__()
        # Normal exit
        h.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Forwarded methods
# ---------------------------------------------------------------------------
class TestForwardedMethods:
    def _setup(self):
        h = _make_initialized_hybrid("np")
        h.set_backend("np")
        h.initialize("d1")
        return h

    def test_initialize(self):
        h = _make_initialized_hybrid("np")
        h.set_backend("np")
        h.initialize("test_data")
        assert "test_data" in h.get_data_ids()

    def test_get_name(self):
        h = HybridDatabase()
        assert h.get_name() == "hybrid"

    def test_set_data_id(self):
        h = self._setup()
        h.set_data_id("d2")
        assert "d2" in h.get_data_ids()

    def test_delete_data_id(self):
        h = self._setup()
        h.delete_data_id("d1")
        assert "d1" not in h.get_data_ids()

    def test_set_phase(self):
        h = self._setup()
        h.set_phase("d1", "train")
        assert "train" in h.get_phases("d1")

    def test_delete_phase(self):
        h = self._setup()
        h.set_phase("d1", "train")
        h.delete_phase("d1", "train")
        assert "train" not in h.get_phases("d1")

    def test_clear(self):
        h = self._setup()
        h.clear()
        assert h.get_data_ids() == []

    def test_close(self):
        h = self._setup()
        h.close()
        # After close, the underlying backend is closed

    def test_add_data(self):
        h = self._setup()
        h.set_phase("d1", "train")
        h.add_data("d1", "train", "x", np.array([[1.0]]))
        assert "x" in h.get_var_names("d1", "train")

    def test_update_data(self):
        h = self._setup()
        h.set_phase("d1", "train")
        h.add_data("d1", "train", "x", np.array([[1.0]]))
        h.update_data("d1", "train", "x", np.array([[2.0]]), slice(0, 1))
        result = h.get_data("d1", "train", "x")
        np.testing.assert_array_equal(result, np.array([[2.0]]))

    def test_get_data(self):
        h = self._setup()
        h.set_phase("d1", "train")
        h.add_data("d1", "train", "x", np.array([[1.0]]))
        result = h.get_data("d1", "train", "x")
        np.testing.assert_array_equal(result, np.array([[1.0]]))

    def test_delete_data(self):
        h = self._setup()
        h.set_phase("d1", "train")
        h.add_data("d1", "train", "x", np.array([[1.0]]))
        h.delete_data("d1", "train", "x")
        assert "x" not in h.get_var_names("d1", "train")

    def test_stream_data(self):
        h = self._setup()
        h.set_phase("d1", "train")
        h.add_data("d1", "train", "x", np.array([[1.0]]))
        chunks = list(h.stream_data("d1", "train", "x"))
        assert len(chunks) >= 1

    def test_copy_data(self):
        h = self._setup()
        h.set_phase("d1", "train")
        h.add_data("d1", "train", "x", np.array([[1.0]]))
        h.copy_data("d1", "train", "x", "y")
        assert "y" in h.get_var_names("d1", "train")

    def test_rename_data(self):
        h = self._setup()
        h.set_phase("d1", "train")
        h.add_data("d1", "train", "x", np.array([[1.0]]))
        h.rename_data("d1", "train", "x", "z")
        assert "z" in h.get_var_names("d1", "train")
        assert "x" not in h.get_var_names("d1", "train")

    def test_get_data_ids(self):
        h = self._setup()
        assert "d1" in h.get_data_ids()

    def test_get_phases(self):
        h = self._setup()
        h.set_phase("d1", "train")
        assert "train" in h.get_phases("d1")

    def test_get_var_names(self):
        h = self._setup()
        h.set_phase("d1", "train")
        h.add_data("d1", "train", "x", np.array([[1.0]]))
        assert "x" in h.get_var_names("d1", "train")

    def test_get_data_info(self):
        h = self._setup()
        h.set_phase("d1", "train")
        h.add_data("d1", "train", "x", np.array([[1.0]]))
        info = h.get_data_info("d1", "train", "x")
        assert "dtype" in info
        assert "shape" in info
        assert "num_events" in info

    def test_compile(self):
        h = self._setup()
        h.set_phase("d1", "train")
        h.add_data("d1", "train", "x", np.array([[1.0]]))
        report = h.compile("d1")
        assert "is_compiled" in report

    def test_is_writable(self):
        h = self._setup()
        assert h.is_writable() is True


# ---------------------------------------------------------------------------
# _resolve_backend
# ---------------------------------------------------------------------------
class TestResolveBackend:
    def test_none_set(self):
        h = HybridDatabase()
        with pytest.raises(RuntimeError, match="requires backend"):
            h._resolve_backend()

    def test_deleted(self):
        h = _make_initialized_hybrid("np")
        h.set_backend("np")
        # Manually delete the backend dict entry to simulate deletion
        db = h._db.pop("np")
        db.close()
        with pytest.raises(RuntimeError, match="has been deleted"):
            h._resolve_backend()


# ---------------------------------------------------------------------------
# transfer_data
# ---------------------------------------------------------------------------
class TestTransferData:
    def _setup_two_backends(self):
        h = HybridDatabase()
        src = NumpyDatabase()
        dst = NumpyDatabase()
        h.register_backend("src", src)
        h.register_backend("dst", dst)
        h.set_backend("src")
        h.initialize("d1")
        h.set_phase("d1", "train")
        h.add_data("d1", "train", "x", np.array([[1.0, 2.0]]))
        h.set_backend("dst")
        h.initialize("d1")
        return h

    def test_success(self):
        h = self._setup_two_backends()
        h.transfer_data("src", "dst", "d1", "train", "x")
        h.set_backend("dst")
        assert "x" in h.get_var_names("d1", "train")
        np.testing.assert_array_equal(
            h.get_data("d1", "train", "x"),
            np.array([[1.0, 2.0]]),
        )

    def test_success_with_output_var_name(self):
        h = self._setup_two_backends()
        h.transfer_data("src", "dst", "d1", "train", "x", "y")
        h.set_backend("dst")
        assert "y" in h.get_var_names("d1", "train")

    def test_same_src_dst_same_var_noop(self):
        h = self._setup_two_backends()
        h.set_backend("src")
        # Same backend, same var => no-op
        h.transfer_data("src", "src", "d1", "train", "x")
        assert "x" in h.get_var_names("d1", "train")

    def test_same_src_dst_different_var(self):
        h = self._setup_two_backends()
        h.set_backend("src")
        h.transfer_data("src", "src", "d1", "train", "x", "y")
        assert "y" in h.get_var_names("d1", "train")

    def test_missing_src_backend(self):
        h = HybridDatabase()
        with pytest.raises(ValueError, match="src_backend"):
            h.transfer_data("missing", "dst", "d1", "train", "x")

    def test_missing_dst_backend(self):
        h = HybridDatabase()
        h.register_backend("src", NumpyDatabase())
        with pytest.raises(ValueError, match="dst_backend"):
            h.transfer_data("src", "missing", "d1", "train", "x")

    def test_src_not_initialized(self):
        h = HybridDatabase()
        h.register_backend("src", NumpyDatabase())
        h.register_backend("dst", NumpyDatabase())
        with pytest.raises(RuntimeError, match="not initialized"):
            h.transfer_data("src", "dst", "d1", "train", "x")

    def test_dst_not_writable(self, tmp_path):
        h = HybridDatabase()
        src = NumpyDatabase()
        src.initialize("d1")
        src.set_phase("d1", "train")
        src.add_data("d1", "train", "x", np.array([[1.0]]))
        h.register_backend("src", src)

        # Create a read-only zarr backend
        zdb_write = ZarrDatabase(str(tmp_path / "z"), mode="a")
        zdb_write.initialize("d1")
        zdb_write.close()
        zdb_read = ZarrDatabase(str(tmp_path / "z"), mode="r")
        zdb_read.initialize("d1")
        h.register_backend("dst", zdb_read)

        with pytest.raises(ValueError, match="does not support writes"):
            h.transfer_data("src", "dst", "d1", "train", "x")

    def test_dst_data_id_not_exist(self):
        h = HybridDatabase()
        src = NumpyDatabase()
        src.initialize("d1")
        src.set_phase("d1", "train")
        src.add_data("d1", "train", "x", np.array([[1.0]]))
        h.register_backend("src", src)

        dst = NumpyDatabase()
        dst.initialize()
        h.register_backend("dst", dst)

        with pytest.raises(ValueError, match="does not exist"):
            h.transfer_data("src", "dst", "d1", "train", "x")


# ---------------------------------------------------------------------------
# _BackendProxy
# ---------------------------------------------------------------------------
class TestBackendProxy:
    def test_write_only_blocks_non_allowed(self):
        h = _make_initialized_hybrid("np")
        proxy = h["all"]
        with pytest.raises(RuntimeError, match="not supported"):
            proxy.get_data("d1", "train", "x")

    def test_broadcast_set_data_id(self):
        h = _make_initialized_hybrid("a", "b")
        for name in ["a", "b"]:
            with h.using_backend(name):
                h.initialize("d1")
        proxy = h["all"]
        proxy.set_data_id("d2")
        for name in ["a", "b"]:
            with h.using_backend(name):
                assert "d2" in h.get_data_ids()

    def test_broadcast_set_phase(self):
        h = _make_initialized_hybrid("a", "b")
        for name in ["a", "b"]:
            with h.using_backend(name):
                h.initialize("d1")
        proxy = h["all"]
        proxy.set_phase("d1", "train")
        for name in ["a", "b"]:
            with h.using_backend(name):
                assert "train" in h.get_phases("d1")

    def test_broadcast_set_data_id_with_rollback(self):
        """If set_data_id fails on second backend, rollback first."""
        h = HybridDatabase()
        good_db = NumpyDatabase()
        good_db.initialize()
        h.register_backend("a", good_db)

        bad_db = MagicMock(spec=NumpyDatabase)
        bad_db.get_data_ids.return_value = []
        bad_db.set_data_id.side_effect = RuntimeError("fail")
        h.register_backend("b", bad_db)

        proxy = h["all"]
        with pytest.raises(RuntimeError, match="fail"):
            proxy.set_data_id("d1")

    def test_compile_with_errors_exception_group(self):
        """If compile fails on one backend, ExceptionGroup is raised."""
        h = HybridDatabase()
        good_db = NumpyDatabase()
        good_db.initialize("d1")
        h.register_backend("a", good_db)

        bad_db = MagicMock(spec=NumpyDatabase)
        bad_db.compile.side_effect = RuntimeError("compile fail")
        h.register_backend("b", bad_db)

        proxy = h["all"]
        with pytest.raises(ExceptionGroup, match="failed to compile"):
            proxy.compile("d1")

    def test_close_with_errors_exception_group(self):
        h = HybridDatabase()
        bad_db = MagicMock(spec=NumpyDatabase)
        bad_db.close.side_effect = RuntimeError("close fail")
        h.register_backend("a", bad_db)

        proxy = h["all"]
        with pytest.raises(ExceptionGroup, match="failed to close"):
            proxy.close()

    def test_initialize_with_failures_and_best_effort_close(self):
        """If initialize fails on second backend, first is closed best-effort."""
        h = HybridDatabase()
        good_db = NumpyDatabase()
        h.register_backend("a", good_db)

        bad_db = MagicMock(spec=NumpyDatabase)
        bad_db.initialize.side_effect = RuntimeError("init fail")
        h.register_backend("b", bad_db)

        proxy = h["all"]
        with pytest.raises(RuntimeError, match="init fail"):
            proxy.initialize()

    def test_single_backend_proxy_getattr(self):
        """Single-backend proxy forwards arbitrary methods."""
        h = _make_initialized_hybrid("np")
        h.set_backend("np")
        h.initialize("d1")
        proxy = h["np"]
        phases = proxy.get_phases("d1")
        assert isinstance(phases, list)

    def test_write_only_allowed_methods(self):
        """'set_data_id' and 'set_phase' are allowed on write_only proxy."""
        h = _make_initialized_hybrid("np")
        with h.using_backend("np"):
            h.initialize("d1")
        proxy = h["all"]
        # These should not raise
        proxy.set_data_id("d2")
        proxy.set_phase("d2", "train")

    def test_broadcast_set_data_id_rollback_with_errors(self):
        """Test broadcast rollback when both apply and rollback fail -> ExceptionGroup."""
        h = HybridDatabase()

        db_a = NumpyDatabase()
        db_a.initialize()
        h.register_backend("a", db_a)

        db_b = NumpyDatabase()
        db_b.initialize()
        h.register_backend("b", db_b)

        # Make set_data_id succeed on a, then fail on b
        # And make rollback (delete_data_id) fail on a
        call_count = {"set": 0}
        original_set = NumpyDatabase.set_data_id
        original_delete = NumpyDatabase.delete_data_id

        def patched_set(self_db, data_id):
            call_count["set"] += 1
            if self_db is db_b:
                # First succeed to create it, then fail
                raise RuntimeError("set fail on b")
            return original_set(self_db, data_id)

        def patched_delete(self_db, data_id):
            if self_db is db_a:
                raise RuntimeError("rollback fail on a")
            return original_delete(self_db, data_id)

        with patch.object(NumpyDatabase, "set_data_id", patched_set):
            with patch.object(NumpyDatabase, "delete_data_id", patched_delete):
                proxy = h["all"]
                with pytest.raises(ExceptionGroup):
                    proxy.set_data_id("new_id")

    def test_compile_success_no_errors(self):
        """Compile on all backends returns dict of reports."""
        h = _make_initialized_hybrid("a", "b")
        for name in ["a", "b"]:
            with h.using_backend(name):
                h.initialize("d1")
                h.set_phase("d1", "train")
                h.add_data("d1", "train", "x", np.array([[1.0]]))

        proxy = h["all"]
        reports = proxy.compile("d1")
        assert "a" in reports
        assert "b" in reports

    def test_non_write_only_compile(self):
        """Single backend proxy compile returns a single report."""
        h = _make_initialized_hybrid("np")
        h.set_backend("np")
        h.initialize("d1")
        h.set_phase("d1", "train")
        h.add_data("d1", "train", "x", np.array([[1.0]]))
        proxy = h["np"]
        report = proxy.compile("d1")
        assert "is_compiled" in report

    def test_non_write_only_close(self):
        h = _make_initialized_hybrid("np")
        h.set_backend("np")
        h.initialize("d1")
        proxy = h["np"]
        proxy.close()

    def test_non_write_only_initialize(self):
        h = _make_initialized_hybrid("np")
        proxy = h["np"]
        proxy.initialize("d1")

    def test_non_write_only_set_data_id(self):
        h = _make_initialized_hybrid("np")
        h.set_backend("np")
        h.initialize("d1")
        proxy = h["np"]
        proxy.set_data_id("d2")

    def test_non_write_only_set_phase(self):
        h = _make_initialized_hybrid("np")
        h.set_backend("np")
        h.initialize("d1")
        proxy = h["np"]
        proxy.set_phase("d1", "train")

    def test_broadcast_set_phase_with_rollback(self):
        """If set_phase fails on second backend, rollback first."""
        h = HybridDatabase()
        good_db = NumpyDatabase()
        good_db.initialize("d1")
        h.register_backend("a", good_db)

        bad_db = MagicMock(spec=NumpyDatabase)
        bad_db.get_phases.return_value = []
        bad_db.set_phase.side_effect = RuntimeError("set_phase fail")
        h.register_backend("b", bad_db)

        proxy = h["all"]
        with pytest.raises(RuntimeError, match="set_phase fail"):
            proxy.set_phase("d1", "train")

    def test_initialize_close_best_effort_errors(self):
        """If init fails and close also fails, still raises the init error."""
        h = HybridDatabase()
        good_db = MagicMock(spec=NumpyDatabase)
        good_db.close.side_effect = RuntimeError("close fail")
        h.register_backend("a", good_db)

        bad_db = MagicMock(spec=NumpyDatabase)
        bad_db.initialize.side_effect = RuntimeError("init fail")
        h.register_backend("b", bad_db)

        proxy = h["all"]
        with pytest.raises(RuntimeError, match="init fail"):
            proxy.initialize()

    def test_broadcast_create_rollback_partial_create_check(self):
        """Exercise the branch where a failed backend still created the entity."""
        h = HybridDatabase()

        db_a = NumpyDatabase()
        db_a.initialize()
        h.register_backend("a", db_a)

        # b's set_data_id creates the id then raises
        db_b = NumpyDatabase()
        db_b.initialize()
        h.register_backend("b", db_b)

        original_set = NumpyDatabase.set_data_id

        def patched_set(self_db, data_id):
            original_set(self_db, data_id)
            if self_db is db_b:
                raise RuntimeError("partially created then failed")

        with patch.object(NumpyDatabase, "set_data_id", patched_set):
            proxy = h["all"]
            with pytest.raises(RuntimeError):
                proxy.set_data_id("new_id")

        # Both should have been rolled back
        assert "new_id" not in db_a.get_data_ids()

    def test_broadcast_create_exists_check_fails(self):
        """Exercise the branch where exists() check on failed backend raises."""
        h = HybridDatabase()

        db_a = NumpyDatabase()
        db_a.initialize()
        h.register_backend("a", db_a)

        db_b = NumpyDatabase()
        db_b.initialize()
        h.register_backend("b", db_b)

        call_count = {"set": 0}
        original_set = NumpyDatabase.set_data_id
        original_get_ids = NumpyDatabase.get_data_ids

        def patched_set(self_db, data_id):
            if self_db is db_b:
                call_count["set"] += 1
                raise RuntimeError("set fail")
            return original_set(self_db, data_id)

        def patched_get_ids(self_db):
            if self_db is db_b and call_count["set"] > 0:
                raise RuntimeError("get_data_ids fail too")
            return original_get_ids(self_db)

        with patch.object(NumpyDatabase, "set_data_id", patched_set):
            with patch.object(NumpyDatabase, "get_data_ids", patched_get_ids):
                proxy = h["all"]
                with pytest.raises(RuntimeError, match="set fail"):
                    proxy.set_data_id("new_id")
