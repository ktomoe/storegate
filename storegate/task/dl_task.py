from storegate import logger, const
from storegate.task.agent_task import AgentTask
from storegate.task.dl_env import DLEnv

class DLTask(AgentTask):
    """DL task class for the default functions."""
    def __init__(self,
                 device='cuda',
                 input_var_names=None,
                 output_var_names=None,
                 true_var_names=None,
                 model=None,
                 model_args=None,
                 optimizer=None,
                 optimizer_args=None,
                 loss=None,
                 loss_args=None,
                 metrics=None,
                 num_epochs=10,
                 batch_size=64,
                 preload=False,
                 **kwargs):

        super().__init__(**kwargs)

        if model_args is None:
            model_args = {}

        if optimizer_args is None:
            optimizer_args = {}

        if loss_args is None:
            loss_args = {}

        self._ml = DLEnv()

        self._input_var_names = input_var_names
        self._output_var_names = output_var_names
        self._true_var_names = true_var_names

        self._model = model
        self._model_args = model_args
        self._optimizer = optimizer
        self._optimizer_args = optimizer_args
        self._loss = loss
        self._loss_args = loss_args
        self._metrics = metrics

        self._num_epochs = num_epochs
        self._batch_size = batch_size 
        self._preload = preload


    _PROTECTED_KEYS = AgentTask._PROTECTED_KEYS

    def set_hps(self, params):
        """Set hyperparameters to this task."""
        for key, value in params.items():

            if key.startswith('model__'):
                self._model_args[key.replace('model__', '')] = value

            elif key.startswith('optimizer__'):
                self._optimizer_args[key.replace('optimizer__', '')] = value

            elif key.startswith('loss__'):
                self._loss_args[key.replace('loss__', '')] = value

            else:
                if key in self._PROTECTED_KEYS:
                    raise AttributeError(f'{key} is not a valid hyperparameter.')
                if not hasattr(self, '_' + key):
                    raise AttributeError(f'{key} is not defined.')

                setattr(self, '_' + key, value)

        if self._data_id is not None:
            self._storegate.set_data_id(self._data_id)

    def execute(self):
        """Execute a task."""

        self.compile()


        org_backend = self._storegate.get_backend()
        if self._preload:
            for phase in const.PHASES:
                for input_var_name in self._input_var_names:
                    self._storegate.copy_to_memory(input_var_name, phase=phase)
                for true_var_name in self._true_var_names:
                    self._storegate.copy_to_memory(true_var_name, phase=phase)

            self._storegate.set_backend('numpy')
            self._storegate.compile()

        rtn_fit = self.fit()
        rtn_predict = self.predict()

        self._storegate.set_backend(org_backend)
        
        return rtn_fit | rtn_predict


    def fit(self):
        """Fit model."""
        return {}


    def predict(self):
        """Predict model."""
        return {}


    def compile(self):
        """Compile model, optimizer and loss."""
        self._ml.clear()

        self.compile_var_names()
        self.compile_model()
        self.compile_loss()
        self.compile_optimizer()

        self.storegate.compile()

    def compile_var_names(self):
        """Compile variable names."""
        if isinstance(self._input_var_names, str):
            self._input_var_names = [self._input_var_names]

        if isinstance(self._output_var_names, str):
            self._output_var_names = [self._output_var_names]

        if isinstance(self._true_var_names, str):
            self._true_var_names = [self._true_var_names]

    def compile_model(self):
        """Compile model."""
        pass

    def compile_optimizer(self):
        """Compile optimizer."""
        pass

    def compile_loss(self):
        """Compile loss."""
        pass
