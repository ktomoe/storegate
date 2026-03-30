from importlib import import_module
from typing import Any

from storegate.database.database import Database
from storegate.database.numpy_database import NumpyDatabase
from storegate.database.hybrid_database import HybridDatabase

__all__ = [
    "Database",
    "ZarrDatabase",
    "NumpyDatabase",
    "HybridDatabase",
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
