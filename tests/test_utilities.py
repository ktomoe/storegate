"""Comprehensive tests for storegate/utilities.py."""

import copy
import logging
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from storegate import utilities as util


# ---------------------------------------------------------------------------
# ensure_str
# ---------------------------------------------------------------------------
class TestEnsureStr:
    def test_valid_simple(self):
        assert util.ensure_str("hello", "x") == "hello"

    def test_valid_alphanumeric(self):
        assert util.ensure_str("abc123", "x") == "abc123"

    def test_valid_underscore(self):
        assert util.ensure_str("my_var", "x") == "my_var"

    def test_valid_hyphen(self):
        assert util.ensure_str("my-var", "x") == "my-var"

    def test_valid_single_char(self):
        assert util.ensure_str("a", "x") == "a"

    def test_valid_128_chars(self):
        s = "a" * 128
        assert util.ensure_str(s, "x") == s

    def test_invalid_129_chars(self):
        s = "a" * 129
        with pytest.raises(ValueError, match="Invalid"):
            util.ensure_str(s, "x")

    def test_invalid_empty(self):
        with pytest.raises(ValueError, match="Invalid"):
            util.ensure_str("", "x")

    def test_invalid_special_chars(self):
        with pytest.raises(ValueError, match="Invalid"):
            util.ensure_str("hello world", "x")

    def test_invalid_dot(self):
        with pytest.raises(ValueError, match="Invalid"):
            util.ensure_str("hello.world", "x")

    def test_invalid_slash(self):
        with pytest.raises(ValueError, match="Invalid"):
            util.ensure_str("a/b", "x")

    def test_not_string_int(self):
        with pytest.raises(ValueError, match="Invalid"):
            util.ensure_str(123, "x")  # type: ignore

    def test_not_string_none(self):
        with pytest.raises(ValueError, match="Invalid"):
            util.ensure_str(None, "x")  # type: ignore


# ---------------------------------------------------------------------------
# ensure_positive_int
# ---------------------------------------------------------------------------
class TestEnsurePositiveInt:
    def test_valid_positive(self):
        assert util.ensure_positive_int(1, "x") == 1

    def test_valid_large(self):
        assert util.ensure_positive_int(999999, "x") == 999999

    def test_invalid_zero(self):
        with pytest.raises(ValueError, match="must be a positive integer"):
            util.ensure_positive_int(0, "x")

    def test_invalid_negative(self):
        with pytest.raises(ValueError, match="must be a positive integer"):
            util.ensure_positive_int(-1, "x")

    def test_invalid_bool_true(self):
        with pytest.raises(ValueError, match="must be a positive integer"):
            util.ensure_positive_int(True, "x")

    def test_invalid_bool_false(self):
        with pytest.raises(ValueError, match="must be a positive integer"):
            util.ensure_positive_int(False, "x")

    def test_invalid_float(self):
        with pytest.raises(ValueError, match="must be a positive integer"):
            util.ensure_positive_int(1.5, "x")  # type: ignore

    def test_invalid_string(self):
        with pytest.raises(ValueError, match="must be a positive integer"):
            util.ensure_positive_int("1", "x")  # type: ignore

    def test_invalid_none(self):
        with pytest.raises(ValueError, match="must be a positive integer"):
            util.ensure_positive_int(None, "x")  # type: ignore

    def test_numpy_int(self):
        assert util.ensure_positive_int(np.int64(5), "x") == 5


# ---------------------------------------------------------------------------
# build_module
# ---------------------------------------------------------------------------
class TestBuildModule:
    def test_str_with_modules(self):
        """String object resolved from a module namespace."""
        ns = types.SimpleNamespace()
        ns.MyClass = lambda x=1: {"x": x}
        result = util.build_module("MyClass", {"x": 42}, ns)
        assert result == {"x": 42}

    def test_str_without_modules_raises(self):
        with pytest.raises(ValueError, match="Cannot resolve"):
            util.build_module("MyClass", {}, None)

    def test_class_object(self):
        class Dummy:
            def __init__(self, val=0):
                self.val = val

        result = util.build_module(Dummy, {"val": 10}, None)
        assert isinstance(result, Dummy)
        assert result.val == 10

    def test_instance_object_no_args(self):
        original = {"key": "value"}
        result = util.build_module(original, {}, None)
        assert result == original
        # Must be a deepcopy
        assert result is not original

    def test_instance_object_with_args_warns(self, caplog):
        original = {"key": "value"}
        with caplog.at_level(logging.WARNING, logger="storegate.utilities"):
            result = util.build_module(original, {"extra": 1}, None)
        assert result == original
        assert "instance object is given but args is also provided" in caplog.text

    def test_str_with_args(self):
        ns = types.SimpleNamespace()
        ns.Adder = lambda a, b: a + b
        result = util.build_module("Adder", {"a": 3, "b": 4}, ns)
        assert result == 7
