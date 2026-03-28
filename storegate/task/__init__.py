from importlib import import_module
from typing import Any

from storegate.task.task import Task
from storegate.task.dl_task import DLTask
from storegate.task.dl_env import DLEnv

__all__ = [
    'Task',
    'DLTask',
    'PytorchTask',
    'SelfSupervisedPytorchTask',
    'UnsupervisedPytorchTask',
    'DLEnv',
]

_LAZY_IMPORTS = {
    "PytorchTask",
    "SelfSupervisedPytorchTask",
    "UnsupervisedPytorchTask",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        return getattr(import_module("storegate.task.pytorch_task"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
