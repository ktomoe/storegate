import importlib
import sys

import pytest


MODULE_NAME = 'storegate.task'


@pytest.fixture(autouse=True)
def restore_storegate_task_module():
    real_import_module = importlib.import_module
    yield
    sys.modules.pop(MODULE_NAME, None)
    real_import_module(MODULE_NAME)


def _import_task_module_with_pytorch_import_error(
    monkeypatch: pytest.MonkeyPatch,
    *,
    missing_name: str,
):
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == 'storegate.task.pytorch_task':
            raise ImportError(
                f"No module named '{missing_name}'",
                name=missing_name,
            )
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, 'import_module', fake_import_module)
    sys.modules.pop(MODULE_NAME, None)
    return real_import_module(MODULE_NAME)


def test_missing_torch_dependency_is_suppressed(monkeypatch: pytest.MonkeyPatch) -> None:
    task_module = _import_task_module_with_pytorch_import_error(
        monkeypatch,
        missing_name='torch',
    )

    assert 'PytorchTask' not in task_module.__all__
    assert not hasattr(task_module, 'PytorchTask')


def test_unrelated_import_error_is_not_suppressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ImportError, match='broken_dep'):
        _import_task_module_with_pytorch_import_error(
            monkeypatch,
            missing_name='broken_dep',
        )
