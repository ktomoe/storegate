"""Module to define agent abstraction."""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any


class Agent(metaclass=ABCMeta):
    """Base class of Agent."""
    @abstractmethod
    def execute(self) -> Any:
        """Execute Agent."""

    @abstractmethod
    def finalize(self) -> None:
        """Finalize Agent."""
