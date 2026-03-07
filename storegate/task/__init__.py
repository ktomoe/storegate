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

try:
    from storegate.task.pytorch_task import PytorchTask as PytorchTask
    __all__.append('PytorchTask')
except ImportError:
    pass
