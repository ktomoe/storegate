"""Tests for storegate/storegate.py."""
import io
import copy
from contextlib import contextmanager
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from storegate.database.numpy_database import NumpyDatabase
from storegate.database.hybrid_database import HybridDatabase, _BackendProxy
from storegate.database.staged_add import _StagedAddTransaction
from storegate.storegate import (
    StoreGate,
    StoreGateReadView,
    _AccessContext,
    _AllBackendView,
    _BackendView,
    _BasePhaseView,
    _CompiledReportUnset,
    _COMPILED_REPORT_UNSET,
    _PhaseView,
    _ReadOnlyPhaseView,
    _ReadOnlyVarView,
    _VarView,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_sg(*backend_names, data_id=None, backend=None):
    """Create a StoreGate with named NumpyDatabase backends."""
    backends = {name: NumpyDatabase() for name in backend_names}
    sg = StoreGate(backends=backends, backend=backend, data_id=data_id)
    return sg


def _make_sg_with_data():
    """Create a StoreGate with one backend, initialized with data."""
    sg = _make_sg("np", backend="np", data_id="d1")
    sg.initialize()
    sg.set_phase("train")
    sg.add_data("train", "x", np.array([[1.0, 2.0]]))
    sg.add_data("train", "y", np.array([[3.0]]))
    return sg


# ---------------------------------------------------------------------------
# StoreGate constructor
# ---------------------------------------------------------------------------
class TestStoreGateConstructor:
    def test_no_args(self):
        sg = StoreGate()
        assert sg._data_id is None
        assert sg._hybrid._db == {}

    def test_with_backends_dict(self):
        sg = _make_sg("np1", "np2")
        assert "np1" in sg._hybrid._db
        assert "np2" in sg._hybrid._db

    def test_with_backend_selection(self):
        sg = _make_sg("np", backend="np")
        assert sg._hybrid.get_backend() == "np"

    def test_with_data_id(self):
        sg = _make_sg("np", data_id="d1")
        assert sg._data_id == "d1"

    def test_registration_failure_cleanup(self):
        """If backend selection fails, registered backends are cleaned up."""
        backends = {"np": NumpyDatabase()}
        with pytest.raises(ValueError):
            StoreGate(backends=backends, backend="nonexistent")


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------
class TestContextManager:
    def test_enter_exit(self):
        sg = _make_sg("np", backend="np", data_id="d1")
        with sg:
            assert sg._data_id == "d1"

    def test_exit_clears_compiled(self):
        sg = _make_sg("np", backend="np", data_id="d1")
        with sg:
            sg.set_phase("train")
            sg.add_data("train", "x", np.array([[1.0]]))
            sg.compile()
            assert len(sg._compiled) > 0
        assert len(sg._compiled) == 0


# ---------------------------------------------------------------------------
# __getitem__ / __contains__
# ---------------------------------------------------------------------------
class TestSubscript:
    def test_getitem_returns_phase_view(self):
        sg = _make_sg_with_data()
        view = sg["train"]
        assert isinstance(view, _PhaseView)

    def test_contains(self):
        sg = _make_sg_with_data()
        assert "train" in sg
        assert "nonexistent" not in sg


# ---------------------------------------------------------------------------
# __getattr__
# ---------------------------------------------------------------------------
class TestGetattr:
    def test_valid_backend_name(self):
        sg = _make_sg("np", backend="np")
        view = sg.np
        assert isinstance(view, _BackendView)

    def test_all_returns_all_backend_view(self):
        sg = _make_sg("np")
        view = sg.all
        assert isinstance(view, _AllBackendView)

    def test_invalid_name_raises_attribute_error(self):
        sg = _make_sg("np")
        with pytest.raises(AttributeError, match="no attribute"):
            _ = sg.nonexistent_backend

    def test_underscore_prefixed_raises(self):
        sg = _make_sg("np")
        with pytest.raises(AttributeError):
            _ = sg._private_attr


# ---------------------------------------------------------------------------
# get_backend_view
# ---------------------------------------------------------------------------
class TestGetBackendView:
    def test_valid(self):
        sg = _make_sg("np")
        view = sg.get_backend_view("np")
        assert isinstance(view, _BackendView)

    def test_all_rejected(self):
        sg = _make_sg("np")
        with pytest.raises(ValueError, match="broadcast view"):
            sg.get_backend_view("all")

    def test_missing(self):
        sg = _make_sg("np")
        with pytest.raises(ValueError, match="does not exist"):
            sg.get_backend_view("missing")


# ---------------------------------------------------------------------------
# register_backend / delete_backend / set_backend / get_backend / using_backend
# ---------------------------------------------------------------------------
class TestBackendManagement:
    def test_register_backend_success(self):
        sg = StoreGate()
        sg.register_backend("np", NumpyDatabase())
        assert "np" in sg._hybrid._db

    def test_register_backend_reserved_name(self):
        sg = StoreGate()
        with pytest.raises(ValueError, match="reserved"):
            sg.register_backend("compile", NumpyDatabase())

    def test_register_backend_already_exists(self):
        sg = _make_sg("np")
        with pytest.raises(ValueError, match="already exists"):
            sg.register_backend("np", NumpyDatabase())

    def test_register_backend_init_failure_cleanup(self):
        """If backend init fails, close is called on the backend."""
        bad_db = MagicMock(spec=NumpyDatabase)
        bad_db.initialize.side_effect = RuntimeError("init fail")
        sg = StoreGate()
        with pytest.raises(RuntimeError, match="init fail"):
            sg.register_backend("bad", bad_db)
        bad_db.close.assert_called_once()

    def test_register_backend_init_failure_close_failure(self):
        """If backend init fails and close also fails, still propagates init error."""
        bad_db = MagicMock(spec=NumpyDatabase)
        bad_db.initialize.side_effect = RuntimeError("init fail")
        bad_db.close.side_effect = RuntimeError("close fail")
        sg = StoreGate()
        with pytest.raises(RuntimeError, match="init fail"):
            sg.register_backend("bad", bad_db)

    def test_delete_backend(self):
        sg = _make_sg("np")
        sg.delete_backend("np")
        assert "np" not in sg._hybrid._db

    def test_set_backend(self):
        sg = _make_sg("np")
        sg.set_backend("np")
        assert sg.get_backend() == "np"

    def test_get_backend(self):
        sg = StoreGate()
        assert sg.get_backend() is None

    def test_using_backend(self):
        sg = _make_sg("a", "b", backend="a")
        with sg.using_backend("b"):
            assert sg.get_backend() == "b"
        assert sg.get_backend() == "a"


# ---------------------------------------------------------------------------
# data_id management
# ---------------------------------------------------------------------------
class TestDataIdManagement:
    def test_set_data_id(self):
        sg = _make_sg("np", backend="np")
        sg.initialize()
        sg.set_data_id("d1")
        assert sg._data_id == "d1"

    def test_delete_data_id_clears_current(self):
        sg = _make_sg("np", backend="np", data_id="d1")
        sg.initialize()
        sg.delete_data_id("d1")
        assert sg._data_id is None

    def test_delete_data_id_non_current(self):
        sg = _make_sg("np", backend="np", data_id="d1")
        sg.initialize()
        sg.set_data_id("d2")
        sg.delete_data_id("d1")
        assert sg._data_id == "d2"

    def test_get_data_id(self):
        sg = _make_sg("np", data_id="d1")
        assert sg.get_data_id() == "d1"

    def test_get_data_ids(self):
        sg = _make_sg("np", backend="np", data_id="d1")
        sg.initialize()
        sg.set_data_id("d2")
        ids = sg.get_data_ids()
        assert "d1" in ids
        assert "d2" in ids


# ---------------------------------------------------------------------------
# Phase management
# ---------------------------------------------------------------------------
class TestPhaseManagement:
    def test_set_phase(self):
        sg = _make_sg_with_data()
        sg.set_phase("val")
        assert "val" in sg.get_phases()

    def test_delete_phase(self):
        sg = _make_sg_with_data()
        sg.set_phase("val")
        sg.delete_phase("val")
        assert "val" not in sg.get_phases()

    def test_get_phases(self):
        sg = _make_sg_with_data()
        assert "train" in sg.get_phases()


# ---------------------------------------------------------------------------
# Data operations
# ---------------------------------------------------------------------------
class TestDataOperations:
    def test_add_data(self):
        sg = _make_sg_with_data()
        sg.add_data("train", "z", np.array([[1.0]]))
        assert "z" in sg.get_var_names("train")

    def test_update_data(self):
        sg = _make_sg_with_data()
        sg.update_data("train", "x", np.array([[9.0, 8.0]]), slice(0, 1))
        result = sg.get_data("train", "x")
        np.testing.assert_array_equal(result, np.array([[9.0, 8.0]]))

    def test_get_data(self):
        sg = _make_sg_with_data()
        result = sg.get_data("train", "x")
        np.testing.assert_array_equal(result, np.array([[1.0, 2.0]]))

    def test_delete_data(self):
        sg = _make_sg_with_data()
        sg.delete_data("train", "y")
        assert "y" not in sg.get_var_names("train")

    def test_stream_data(self):
        sg = _make_sg_with_data()
        chunks = list(sg.stream_data("train", "x"))
        assert len(chunks) >= 1

    def test_copy_data(self):
        sg = _make_sg_with_data()
        sg.copy_data("train", "x", "x_copy")
        assert "x_copy" in sg.get_var_names("train")

    def test_rename_data(self):
        sg = _make_sg_with_data()
        sg.rename_data("train", "x", "x_renamed")
        assert "x_renamed" in sg.get_var_names("train")
        assert "x" not in sg.get_var_names("train")

    def test_staged_add(self):
        sg = _make_sg_with_data()
        with sg.staged_add("train", ["z"]) as tx:
            tx.add_data("z", np.array([[1.0]]))
        assert "z" in sg.get_var_names("train")


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
class TestQuery:
    def test_get_var_names(self):
        sg = _make_sg_with_data()
        names = sg.get_var_names("train")
        assert "x" in names
        assert "y" in names

    def test_get_data_info(self):
        sg = _make_sg_with_data()
        info = sg.get_data_info("train", "x")
        assert info["dtype"] == "float64"
        assert info["num_events"] == 1

    def test_compile_caches_report(self):
        sg = _make_sg_with_data()
        report = sg.compile()
        assert report["is_compiled"] is True
        # Check it's cached
        key = sg._compiled_key("np", "d1")
        assert key in sg._compiled

    def test_compile_no_backend(self):
        sg = _make_sg("np", data_id="d1")
        sg.initialize()
        # No backend set, set one first
        sg.set_backend("np")
        report = sg.compile()
        assert "is_compiled" in report

    def test_is_writable(self):
        sg = _make_sg("np", backend="np")
        assert sg.is_writable() is True


# ---------------------------------------------------------------------------
# show_info
# ---------------------------------------------------------------------------
class TestShowInfo:
    def test_with_compile(self):
        sg = _make_sg_with_data()
        buf = io.StringIO()
        result = sg.show_info(file=buf, compile=True)
        assert "Compiled" in result
        assert "train" in result

    def test_without_compile(self):
        sg = _make_sg_with_data()
        buf = io.StringIO()
        result = sg.show_info(file=buf)
        assert "Not Compiled" in result

    def test_with_backend(self):
        sg = _make_sg_with_data()
        buf = io.StringIO()
        result = sg.show_info(file=buf)
        assert "np" in result

    def test_without_backend_raises(self):
        sg = _make_sg_with_data()
        sg._hybrid._backend = None
        buf = io.StringIO()
        with pytest.raises(RuntimeError, match="requires backend"):
            sg.show_info(file=buf)

    def test_default_file(self):
        sg = _make_sg_with_data()
        buf = io.StringIO()
        import sys
        with patch.object(sys, "stdout", buf):
            result = sg.show_info()
        assert "d1" in result

    def test_empty_phase(self):
        sg = _make_sg("np", backend="np", data_id="d1")
        sg.initialize()
        sg.set_phase("empty_phase")
        buf = io.StringIO()
        result = sg.show_info(file=buf)
        assert "(empty)" in result

    def test_with_cached_compile(self):
        sg = _make_sg_with_data()
        sg.compile()
        buf = io.StringIO()
        result = sg.show_info(file=buf)
        assert "Compiled" in result


# ---------------------------------------------------------------------------
# Other methods
# ---------------------------------------------------------------------------
class TestOther:
    def test_clear(self):
        sg = _make_sg_with_data()
        sg.compile()
        sg.clear()
        assert sg._data_id is None
        assert sg._compiled == {}

    def test_transfer_data(self):
        sg = StoreGate()
        src = NumpyDatabase()
        dst = NumpyDatabase()
        sg.register_backend("src", src)
        sg.register_backend("dst", dst)
        sg.set_backend("src")
        sg.set_data_id("d1")
        sg.initialize()
        sg.set_phase("train")
        sg.add_data("train", "x", np.array([[1.0]]))
        sg.set_backend("dst")
        sg.set_data_id("d1")
        sg.transfer_data("src", "dst", "train", "x")
        assert "x" in sg.get_var_names("train")

    def test_pin_success(self):
        sg = _make_sg_with_data()
        sg.compile()
        view = sg.pin(backend="np", data_id="d1")
        assert isinstance(view, StoreGateReadView)

    def test_pin_missing_data_id(self):
        sg = _make_sg("np", backend="np", data_id="d1")
        sg.initialize()
        with pytest.raises(ValueError, match="does not exist"):
            sg.pin(backend="np", data_id="missing")

    def test_pin_with_compiled_report(self):
        sg = _make_sg_with_data()
        sg.compile()
        view = sg.pin(backend="np")
        assert view._ctx.compiled_report is not None

    def test_pin_without_compiled_report(self):
        sg = _make_sg_with_data()
        view = sg.pin(backend="np")
        assert view._ctx.compiled_report is None

    def test_close(self):
        sg = _make_sg("np", backend="np", data_id="d1")
        sg.initialize()
        sg.compile()
        sg.close()
        assert sg._compiled == {}

    def test_close_no_backends(self):
        sg = StoreGate()
        sg.close()  # Should not raise

    def test_initialize_with_data_id(self):
        sg = _make_sg("np", backend="np")
        sg.initialize("d1")
        assert sg._data_id == "d1"

    def test_initialize_without_data_id(self):
        sg = _make_sg("np", backend="np", data_id="d1")
        sg.initialize()
        assert sg._data_id == "d1"

    def test_initialize_no_backends(self):
        sg = StoreGate(data_id="d1")
        sg.initialize()
        assert sg._data_id == "d1"

    def test_initialize_failure_preserves_previous_data_id(self):
        """If initialization fails, _data_id reverts to previous value."""
        sg = _make_sg("np", backend="np", data_id="d1")
        sg.initialize()

        # Make initialize fail on the "all" proxy
        original_init = NumpyDatabase.initialize
        def failing_init(self_db, data_id=None):
            raise RuntimeError("init fail")

        with patch.object(NumpyDatabase, "initialize", failing_init):
            with pytest.raises(RuntimeError):
                sg.initialize("d2")

        assert sg._data_id == "d1"


# ---------------------------------------------------------------------------
# _AllBackendView
# ---------------------------------------------------------------------------
class TestAllBackendView:
    def test_initialize(self):
        sg = _make_sg("np")
        sg.all.initialize("d1")
        assert sg._data_id == "d1"

    def test_close_no_backends(self):
        sg = StoreGate()
        sg.all.close()

    def test_close_with_backends(self):
        sg = _make_sg("np", backend="np", data_id="d1")
        sg.initialize()
        sg.compile()
        sg.all.close()
        assert sg._compiled == {}

    def test_set_data_id(self):
        sg = _make_sg("np", backend="np")
        sg.initialize()
        sg.all.set_data_id("d1")
        assert sg._data_id == "d1"

    def test_set_phase(self):
        sg = _make_sg("np", backend="np", data_id="d1")
        sg.initialize()
        sg.all.set_phase("train")
        assert "train" in sg.get_phases()

    def test_compile(self):
        sg = _make_sg_with_data()
        reports = sg.all.compile()
        assert "np" in reports


# ---------------------------------------------------------------------------
# _BackendView
# ---------------------------------------------------------------------------
class TestBackendView:
    def _make_bv(self):
        sg = _make_sg_with_data()
        return sg, sg.get_backend_view("np")

    def test_add_data(self):
        sg, bv = self._make_bv()
        bv.add_data("train", "z", np.array([[1.0]]))
        assert "z" in sg.get_var_names("train")

    def test_update_data(self):
        sg, bv = self._make_bv()
        bv.update_data("train", "x", np.array([[9.0, 8.0]]), slice(0, 1))

    def test_delete_data(self):
        sg, bv = self._make_bv()
        bv.delete_data("train", "y")
        assert "y" not in sg.get_var_names("train")

    def test_get_data(self):
        sg, bv = self._make_bv()
        result = bv.get_data("train", "x")
        np.testing.assert_array_equal(result, np.array([[1.0, 2.0]]))

    def test_stream_data(self):
        sg, bv = self._make_bv()
        chunks = list(bv.stream_data("train", "x"))
        assert len(chunks) >= 1

    def test_copy_data(self):
        sg, bv = self._make_bv()
        bv.copy_data("train", "x", "x_copy")
        assert "x_copy" in sg.get_var_names("train")

    def test_rename_data(self):
        sg, bv = self._make_bv()
        bv.rename_data("train", "x", "x_new")
        assert "x_new" in sg.get_var_names("train")

    def test_staged_add(self):
        sg, bv = self._make_bv()
        with bv.staged_add("train", ["z"]) as tx:
            tx.add_data("z", np.array([[1.0]]))
        assert "z" in sg.get_var_names("train")

    def test_set_phase(self):
        sg, bv = self._make_bv()
        bv.set_phase("val")
        assert "val" in sg.get_phases()

    def test_delete_phase(self):
        sg, bv = self._make_bv()
        bv.set_phase("val")
        bv.delete_phase("val")
        assert "val" not in sg.get_phases()

    def test_is_writable(self):
        sg, bv = self._make_bv()
        assert bv.is_writable() is True

    def test_show_info(self):
        sg, bv = self._make_bv()
        buf = io.StringIO()
        result = bv.show_info(file=buf, compile=True)
        assert "Compiled" in result

    def test_show_info_default_file(self):
        sg, bv = self._make_bv()
        buf = io.StringIO()
        import sys
        with patch.object(sys, "stdout", buf):
            result = bv.show_info()
        assert "d1" in result

    def test_getitem_returns_phase_view(self):
        sg, bv = self._make_bv()
        pv = bv["train"]
        assert isinstance(pv, _PhaseView)

    def test_contains(self):
        sg, bv = self._make_bv()
        assert "train" in bv
        assert "nonexistent" not in bv

    def test_get_phases(self):
        sg, bv = self._make_bv()
        assert "train" in bv.get_phases()

    def test_get_var_names(self):
        sg, bv = self._make_bv()
        assert "x" in bv.get_var_names("train")

    def test_get_data_info(self):
        sg, bv = self._make_bv()
        info = bv.get_data_info("train", "x")
        assert "dtype" in info

    def test_compile(self):
        sg, bv = self._make_bv()
        report = bv.compile()
        assert report["is_compiled"] is True
        # Check it's cached in gate
        key = sg._compiled_key("np", "d1")
        assert key in sg._compiled


# ---------------------------------------------------------------------------
# StoreGateReadView
# ---------------------------------------------------------------------------
class TestStoreGateReadView:
    def test_read_only(self):
        sg = _make_sg_with_data()
        view = sg.pin(backend="np")
        result = view.get_data("train", "x")
        np.testing.assert_array_equal(result, np.array([[1.0, 2.0]]))

    def test_get_backend(self):
        sg = _make_sg_with_data()
        view = sg.pin(backend="np")
        assert view.get_backend() == "np"

    def test_get_data_id(self):
        sg = _make_sg_with_data()
        view = sg.pin(backend="np")
        assert view.get_data_id() == "d1"

    def test_is_writable_returns_false(self):
        sg = _make_sg_with_data()
        view = sg.pin(backend="np")
        assert view.is_writable() is False

    def test_getitem_returns_readonly_phase_view(self):
        sg = _make_sg_with_data()
        view = sg.pin(backend="np")
        pv = view["train"]
        assert isinstance(pv, _ReadOnlyPhaseView)

    def test_compile_updates_ctx_report(self):
        sg = _make_sg_with_data()
        view = StoreGateReadView(
            sg, "np", "d1", compiled_report=None
        )
        report = view.compile()
        assert view._ctx.compiled_report is not None


# ---------------------------------------------------------------------------
# _VarView / _ReadOnlyVarView
# ---------------------------------------------------------------------------
class TestVarView:
    def test_setitem(self):
        sg = _make_sg_with_data()
        pv = sg["train"]
        vv = pv["x"]
        assert isinstance(vv, _VarView)
        vv[slice(0, 1)] = np.array([[9.0, 8.0]])
        result = sg.get_data("train", "x")
        np.testing.assert_array_equal(result, np.array([[9.0, 8.0]]))

    def test_append(self):
        sg = _make_sg_with_data()
        vv = sg["train"]["x"]
        vv.append(np.array([[5.0, 6.0]]))
        result = sg.get_data("train", "x")
        assert result.shape[0] == 2

    def test_readonly_getitem(self):
        sg = _make_sg_with_data()
        view = sg.pin(backend="np")
        rv = view["train"]["x"]
        assert isinstance(rv, _ReadOnlyVarView)
        result = rv[slice(None)]
        np.testing.assert_array_equal(result, np.array([[1.0, 2.0]]))


# ---------------------------------------------------------------------------
# _PhaseView / _ReadOnlyPhaseView
# ---------------------------------------------------------------------------
class TestPhaseView:
    def test_getitem(self):
        sg = _make_sg_with_data()
        pv = sg["train"]
        vv = pv["x"]
        assert isinstance(vv, _VarView)

    def test_contains(self):
        sg = _make_sg_with_data()
        pv = sg["train"]
        assert "x" in pv
        assert "nonexistent" not in pv

    def test_len_compiled(self):
        sg = _make_sg_with_data()
        sg.compile()
        pv = sg["train"]
        assert len(pv) == 1

    def test_len_not_compiled_no_data_id(self):
        sg = _make_sg("np", backend="np", data_id="d1")
        sg.initialize()
        sg.set_phase("train")
        sg.add_data("train", "x", np.array([[1.0]]))
        ctx = _AccessContext(sg)
        pv = _PhaseView(ctx, "train")
        with pytest.raises(RuntimeError, match="not compiled.*compile"):
            len(pv)

    def test_len_not_compiled_with_data_id(self):
        sg = _make_sg("np", backend="np", data_id="d1")
        sg.initialize()
        sg.set_phase("train")
        sg.add_data("train", "x", np.array([[1.0]]))
        ctx = _AccessContext(sg, data_id="d1")
        pv = _PhaseView(ctx, "train")
        with pytest.raises(RuntimeError, match="not compiled"):
            len(pv)

    def test_len_phase_not_found_no_data_id(self):
        sg = _make_sg("np", backend="np", data_id="d1")
        sg.initialize()
        sg.set_phase("train")
        sg.add_data("train", "x", np.array([[1.0]]))
        sg.compile()
        ctx = _AccessContext(sg)
        pv = _PhaseView(ctx, "nonexistent")
        with pytest.raises(RuntimeError, match="not found"):
            len(pv)

    def test_len_phase_not_found_with_data_id(self):
        sg = _make_sg("np", backend="np", data_id="d1")
        sg.initialize()
        sg.set_phase("train")
        sg.add_data("train", "x", np.array([[1.0]]))
        sg.compile()
        ctx = _AccessContext(sg, data_id="d1")
        pv = _PhaseView(ctx, "nonexistent")
        with pytest.raises(RuntimeError, match="not found.*data_id"):
            len(pv)

    def test_len_phase_not_compiled_no_data_id(self):
        sg = _make_sg("np", backend="np", data_id="d1")
        sg.initialize()
        sg.set_phase("train")
        # Add vars with different event counts
        sg.add_data("train", "x", np.array([[1.0]]))
        sg.add_data("train", "y", np.array([[1.0], [2.0]]))
        sg.compile()
        ctx = _AccessContext(sg)
        pv = _PhaseView(ctx, "train")
        with pytest.raises(RuntimeError, match="not compiled.*mismatched"):
            len(pv)

    def test_len_phase_not_compiled_with_data_id(self):
        sg = _make_sg("np", backend="np", data_id="d1")
        sg.initialize()
        sg.set_phase("train")
        sg.add_data("train", "x", np.array([[1.0]]))
        sg.add_data("train", "y", np.array([[1.0], [2.0]]))
        sg.compile()
        ctx = _AccessContext(sg, data_id="d1")
        pv = _PhaseView(ctx, "train")
        with pytest.raises(RuntimeError, match="not compiled.*data_id"):
            len(pv)


class TestReadOnlyPhaseView:
    def test_getitem(self):
        sg = _make_sg_with_data()
        view = sg.pin(backend="np")
        pv = view["train"]
        vv = pv["x"]
        assert isinstance(vv, _ReadOnlyVarView)

    def test_contains(self):
        sg = _make_sg_with_data()
        view = sg.pin(backend="np")
        pv = view["train"]
        assert "x" in pv

    def test_len(self):
        sg = _make_sg_with_data()
        sg.compile()
        view = sg.pin(backend="np")
        pv = view["train"]
        assert len(pv) == 1


# ---------------------------------------------------------------------------
# _dispatch
# ---------------------------------------------------------------------------
class TestDispatch:
    def test_without_backend(self):
        sg = _make_sg_with_data()
        # _dispatch uses the active backend
        result = sg._dispatch("get_phases", data_id="d1")
        assert "train" in result

    def test_with_backend(self):
        sg = _make_sg_with_data()
        result = sg._dispatch("get_phases", backend="np", data_id="d1")
        assert "train" in result

    def test_with_invalidate(self):
        sg = _make_sg_with_data()
        sg.compile()
        assert len(sg._compiled) > 0
        sg._dispatch("set_phase", "val", data_id="d1", invalidate=True)
        # Compiled report should be invalidated
        key = sg._compiled_key("np", "d1")
        assert key not in sg._compiled

    def test_without_invalidate(self):
        sg = _make_sg_with_data()
        sg.compile()
        sg._dispatch("get_phases", data_id="d1", invalidate=False)
        # Compiled report should still be cached
        key = sg._compiled_key("np", "d1")
        assert key in sg._compiled


# ---------------------------------------------------------------------------
# _invalidate_compiled*
# ---------------------------------------------------------------------------
class TestInvalidateCompiled:
    def test_invalidate_compiled(self):
        sg = _make_sg_with_data()
        sg.compile()
        sg._invalidate_compiled("np", "d1")
        assert sg._compiled_key("np", "d1") not in sg._compiled

    def test_invalidate_compiled_no_backend(self):
        sg = _make_sg_with_data()
        sg.compile()
        # When backend is None and no active backend
        sg._hybrid._backend = None
        sg._invalidate_compiled()
        # Should not crash, and should not clear anything (both None)

    def test_invalidate_compiled_no_data_id(self):
        sg = _make_sg("np", backend="np")
        sg.initialize()
        # data_id is None
        sg._invalidate_compiled()  # should not crash

    def test_invalidate_compiled_backend(self):
        sg = _make_sg_with_data()
        sg.compile()
        sg._invalidate_compiled_backend("np")
        assert len(sg._compiled) == 0

    def test_invalidate_compiled_backend_none(self):
        sg = _make_sg_with_data()
        sg.compile()
        count = len(sg._compiled)
        sg._invalidate_compiled_backend(None)
        assert len(sg._compiled) == count

    def test_invalidate_compiled_data_id(self):
        sg = _make_sg_with_data()
        sg.compile()
        sg._invalidate_compiled_data_id("d1")
        assert len(sg._compiled) == 0

    def test_invalidate_compiled_data_id_none(self):
        sg = _make_sg_with_data()
        sg.compile()
        count = len(sg._compiled)
        sg._invalidate_compiled_data_id(None)
        assert len(sg._compiled) == count


# ---------------------------------------------------------------------------
# _resolve_data_id / _resolve_compiled_backend
# ---------------------------------------------------------------------------
class TestResolve:
    def test_resolve_data_id_not_set(self):
        sg = StoreGate()
        with pytest.raises(RuntimeError, match="data_id is not set"):
            sg._resolve_data_id()

    def test_resolve_compiled_backend_no_active(self):
        sg = StoreGate()
        with pytest.raises(RuntimeError, match="No active backend"):
            sg._resolve_compiled_backend(None)

    def test_resolve_compiled_backend_explicit(self):
        sg = StoreGate()
        assert sg._resolve_compiled_backend("np") == "np"


# ---------------------------------------------------------------------------
# _reserved_backend_names
# ---------------------------------------------------------------------------
class TestReservedBackendNames:
    def test_contains_all(self):
        names = StoreGate._reserved_backend_names()
        assert "all" in names

    def test_contains_public_methods(self):
        names = StoreGate._reserved_backend_names()
        assert "compile" in names
        assert "initialize" in names
        assert "close" in names


# ---------------------------------------------------------------------------
# _AccessContext
# ---------------------------------------------------------------------------
class TestAccessContext:
    def test_resolve_data_id_from_ctx(self):
        sg = _make_sg_with_data()
        ctx = _AccessContext(sg, data_id="d1")
        assert ctx.resolve_data_id() == "d1"

    def test_resolve_data_id_from_gate(self):
        sg = _make_sg_with_data()
        ctx = _AccessContext(sg)
        assert ctx.resolve_data_id() == "d1"

    def test_get_compiled_report_unset(self):
        sg = _make_sg_with_data()
        sg.compile()
        ctx = _AccessContext(sg)
        report = ctx.get_compiled_report()
        assert report is not None

    def test_get_compiled_report_set(self):
        sg = _make_sg_with_data()
        ctx = _AccessContext(sg, compiled_report={"test": True})
        report = ctx.get_compiled_report()
        assert report == {"test": True}

    def test_get_compiled_report_none(self):
        sg = _make_sg_with_data()
        ctx = _AccessContext(sg, compiled_report=None)
        report = ctx.get_compiled_report()
        assert report is None

    def test_dispatch(self):
        sg = _make_sg_with_data()
        ctx = _AccessContext(sg)
        result = ctx.dispatch("get_phases")
        assert "train" in result


# ---------------------------------------------------------------------------
# _CompiledReportUnset
# ---------------------------------------------------------------------------
class TestCompiledReportUnset:
    def test_is_singleton(self):
        assert isinstance(_COMPILED_REPORT_UNSET, _CompiledReportUnset)
