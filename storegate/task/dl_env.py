from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DLEnv:
    """Data class to store compiled DL objects."""
    model:     Any = None
    optimizer: Any = None
    loss:      Any = None

    def clear(self) -> None:
        self.model = self.optimizer = self.loss = None
