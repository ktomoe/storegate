import copy
import inspect
import torch

from storegate import logger

def build_module(obj, obj_args, modules):
    # str object
    if isinstance(obj, str):
        return getattr(modules, obj)(**obj_args)

    # class object
    elif inspect.isclass(obj):
        return obj(**obj_args)

    # instance object
    else:
        if obj_args != {}:
            logger.warn('instance object is given but args is also provided')
        return copy.copy(obj)


def inputs_size(inputs):
    if isinstance(inputs, (list, tuple)):
        inputs = inputs[0]

    if isinstance(inputs, torch.Tensor):
        result = inputs.size(0)
    elif getattr(inputs, 'batch_size', None) is not None:
        result = inputs.batch_size
    else:
        result = len(inputs)
    return result
