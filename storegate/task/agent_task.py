from storegate import logger
from storegate.task import Task

class AgentTask(Task):
    """Agent task class for the default functions."""
    _PROTECTED_KEYS = frozenset()

    def __init__(self, storegate):
        self._storegate = storegate
        self._data_id = None

    def set_hps(self, params):
        """Set hyperparameters to this task."""
        for key, value in params.items():
            if key in self._PROTECTED_KEYS:
                raise AttributeError(f'{key} is not a valid hyperparameter.')
            if not hasattr(self, '_' + key):
                raise AttributeError(f'{key} is not defined.')

            setattr(self, '_' + key, value)

        if self._data_id is not None:
            self._storegate.set_data_id(self._data_id)

    def execute(self):
        """Execute base task.

        Users implement their algorithms.
        """

    def finalize(self):
        """Finalize base task.

        Users implement their algorithms.
        """

    @property
    def storegate(self):
        """Return storegate of task."""
        return self._storegate

    @storegate.setter
    def storegate(self, storegate):
        """Set storegate."""
        self._storegate = storegate


