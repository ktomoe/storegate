import torch
import torch.utils.data as tdata


class StoreGateDataset(tdata.Dataset):
    """StoreGate dataset class."""
    def __init__(self,
                 storegate,
                 phase,
                 input_var_names=None,
                 true_var_names=None,
                 preload=False):

        self._storegate = storegate
        self._phase = phase
        self._input_var_names = input_var_names
        self._true_var_names = true_var_names
        self._preload = preload
        self._size = len(storegate[phase])

       
        self._data = None
        self._target = None

        if preload:
            self._data = self._load_tensors(input_var_names)                                                                                                                                                                                       
            self._target = self._load_tensors(true_var_names)  
        

    def __len__(self):
        return self._size

    def _load_tensors(self, var_names):
        if isinstance(var_names, str):
            arr = self._storegate.get_data(var_names, self._phase)
            return torch.as_tensor(arr)
        elif len(var_names) == 1:
            arr = self._storegate.get_data(var_names[0], self._phase)
            return torch.as_tensor(arr)
        else:
            return [self._load_tensors(var_name) for var_name in var_names]

    def __getitem__(self, index):
        if self._preload:
            data = self._index_tensor(self._data, index)
            target = self._index_tensor(self._target, index)
            return data, target

        data = self.make_inputs(self._input_var_names, index)
        target = self.make_inputs(self._true_var_names, index)

        return data, target

    def _index_tensor(self, tensors, index):
        if isinstance(tensors, list):
            return [self._index_tensor(t, index) for t in tensors]
        return tensors[index]

    def make_inputs(self, var_names, index):
        if isinstance(var_names, str):
            return torch.as_tensor(self._storegate.get_data(var_names, self._phase, index))
        elif len(var_names) == 1:
            return torch.as_tensor(self._storegate.get_data(var_names[0], self._phase, index))
        else:
            return [self.make_inputs(var_name, index) for var_name in var_names]
