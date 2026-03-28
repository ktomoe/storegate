"""Tests for storegate/task/__init__.py."""
import pytest

import storegate.task as task_mod


def test_eager_imports():
    assert task_mod.Task is not None
    assert task_mod.DLTask is not None
    assert task_mod.DLEnv is not None


def test_lazy_import_pytorch_task():
    assert task_mod.PytorchTask is not None


def test_lazy_import_self_supervised():
    assert task_mod.SelfSupervisedPytorchTask is not None


def test_lazy_import_unsupervised():
    assert task_mod.UnsupervisedPytorchTask is not None


def test_getattr_unknown_raises():
    with pytest.raises(AttributeError, match="no attribute"):
        task_mod.NonExistentClass


def test_all_exports():
    assert set(task_mod.__all__) == {
        "Task",
        "DLTask",
        "PytorchTask",
        "SelfSupervisedPytorchTask",
        "UnsupervisedPytorchTask",
        "DLEnv",
    }


def test_dir_contains_all():
    d = dir(task_mod)
    for name in task_mod.__all__:
        assert name in d
