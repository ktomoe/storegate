from storegate import logger


class DLEnv:
    """Data class to store compiled DL objects."""
    def __init__(self, model=None, optimizer=None, loss=None):
        """Initialize MLEnv."""
        self._model = model
        self._optimizer = optimizer
        self._loss = loss

    def clear(self):
        self._model = None
        self._optimizer = None
        self._loss = None

    @property
    def model(self):
        """Returns model."""
        return self._model

    @model.setter
    def model(self, model):
        """Set model."""
        self._model = model

    @property
    def optimizer(self):
        """Returns optimizer."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        """Set optimizer."""
        self._optimizer = optimizer

    @property
    def loss(self):
        """Returns loss."""
        return self._loss

    @loss.setter
    def loss(self, loss):
        """Set loss."""
        self._loss = loss
