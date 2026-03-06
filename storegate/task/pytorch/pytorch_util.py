import copy
import inspect
from typing import Any

import torch

from storegate import logger


def build_module(obj: Any, obj_args: dict[str, Any], modules: Any) -> Any:
    # str object
    if isinstance(obj, str):
        if modules is None:
            raise ValueError(
                f"Cannot resolve '{obj}' from a string when no module namespace is provided. "
                f"Pass the class directly instead (e.g. model=torch.nn.{obj})."
            )
        return getattr(modules, obj)(**obj_args)

    # class object
    elif inspect.isclass(obj):
        return obj(**obj_args)

    # instance object
    else:
        if obj_args != {}:
            logger.warn('instance object is given but args is also provided')
        return copy.copy(obj)


def inputs_size(inputs: torch.Tensor | list[Any] | Any) -> int:
    if isinstance(inputs, (list, tuple)):
        inputs = inputs[0]

    if isinstance(inputs, torch.Tensor):
        result: int = inputs.size(0)
    elif getattr(inputs, 'batch_size', None) is not None:
        result = inputs.batch_size
    else:
        result = len(inputs)
    return result
