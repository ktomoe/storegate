from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from storegate.database import Database
from storegate.database import NumpyDatabase
from storegate.database import HybridDatabase
from storegate.storegate import StoreGateReadView
from storegate.storegate import StoreGate

try:
    __version__ = version("storegate")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "Database",
    "NumpyDatabase",
    "ZarrDatabase",
    "HybridDatabase",
    "StoreGateReadView",
    "StoreGate",
    "__version__",
]

_LAZY_IMPORTS = {
    "ZarrDatabase",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        return getattr(import_module("storegate.database.zarr_database"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
