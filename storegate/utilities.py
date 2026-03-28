import copy
import inspect
import logging
import operator
import re
from typing import Any

logger = logging.getLogger(__name__)


_VALID_IDENTIFIER = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


def ensure_str(target: str, name: str) -> str:
    """Raise ValueError if target str contains invalid characters."""
    if not isinstance(target, str) or not _VALID_IDENTIFIER.match(target):
        raise ValueError(
            f"Invalid {name}:{target}. Must be 1-128 characters: alphanumeric, underscore, or hyphen only."
        )
    return target


def ensure_positive_int(target: int, name: str) -> int:
    """Raise ValueError if target is not a positive integer."""
    if isinstance(target, bool):
        raise ValueError(f"{name} must be a positive integer, got: {target!r}")

    try:
        value = operator.index(target)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer, got: {target!r}") from exc

    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got: {target!r}")

    return value


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
            logger.warning('instance object is given but args is also provided')
        return copy.deepcopy(obj)
