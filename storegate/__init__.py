from importlib.metadata import PackageNotFoundError, version

from storegate.database import Database
from storegate.database import NumpyDatabase
from storegate.database import ZarrDatabase
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
