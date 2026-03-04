from dataclasses import dataclass


@dataclass
class DLEnv:
    """Data class to store compiled DL objects."""
    model:     object = None
    optimizer: object = None
    loss:      object = None

    def clear(self):
        self.model = self.optimizer = self.loss = None
