from importlib import import_module
from typing import Any

from storegate.task.task import Task
from storegate.task.agent_task import AgentTask
from storegate.task.dl_task import DLTask
from storegate.task.dl_env import DLEnv

__all__ = [
    'Task',
    'AgentTask',
    'DLTask',
    'DLEnv',
]


def _is_optional_pytorch_dependency_error(exc: ImportError) -> bool:
    missing_name = exc.name
    if missing_name is None:
        return False
    return any(
        missing_name == prefix or missing_name.startswith(f'{prefix}.')
        for prefix in ('torch', 'torchinfo')
    )


def _load_optional_pytorch_task() -> Any | None:
    try:
        return import_module('storegate.task.pytorch_task').PytorchTask
    except ImportError as exc:
        if _is_optional_pytorch_dependency_error(exc):
            return None
        raise


_pytorch_task = _load_optional_pytorch_task()
if _pytorch_task is not None:
    PytorchTask = _pytorch_task
    __all__.append('PytorchTask')
