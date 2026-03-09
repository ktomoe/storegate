from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.utils.data as tdata

if TYPE_CHECKING:
    from storegate.storegate import StoreGate

type _TensorBatch = torch.Tensor | list[torch.Tensor]
type _DatasetItem = _TensorBatch | None | tuple[_TensorBatch | None, _TensorBatch]


class StoreGateDataset(tdata.Dataset):  # type: ignore[type-arg]
    """StoreGate dataset class."""
    def __init__(self,
                 storegate: StoreGate,
                 phase: str,
                 input_var_names: str | list[str] | None = None,
                 true_var_names: str | list[str] | None = None,
                 preload: bool = True) -> None:
        """Initialize the dataset.

        Args:
            storegate: StoreGate instance (must have been compiled).
            phase: One of ``'train'``, ``'valid'``, or ``'test'``.
            input_var_names: Variable name(s) used as model input.
            true_var_names: Variable name(s) used as target/label.
            preload (bool): If ``True``, all data is loaded into memory as
                tensors at construction time.  Each ``__getitem__`` call then
                performs a fast tensor index — strongly recommended when the
                dataset fits in RAM, as lazy mode issues one zarr read per
                sample per epoch.  Default: ``True``.

        Note:
            The default ``preload=True`` is recommended for most use cases.
            Set ``preload=False`` only when the dataset is too large to fit
            in memory; lazy mode reads one sample at a time from the zarr
            store, which is significantly slower for random-access workloads
            typical of DataLoader with ``shuffle=True``.
        """

        self._storegate = storegate
        self._phase = phase
        self._input_var_names = input_var_names
        self._true_var_names = true_var_names
        self._preload = preload
        self._size: int = len(storegate[phase])

        self._data: _TensorBatch | None = None
        self._target: _TensorBatch | None = None

        if preload:
            self._data = self._get_tensors(input_var_names)
            self._target = self._get_tensors(true_var_names)

    def __len__(self) -> int:
        return self._size

    @staticmethod
    def _to_tensor_copy(data: object) -> torch.Tensor:
        """Materialize StoreGate data as an owned tensor without aliasing numpy buffers."""
        return torch.tensor(data)

    def _get_tensors(
        self,
        var_names: str | list[str] | None,
        index: int | None = None,
    ) -> _TensorBatch | None:
        """Load var_names from storegate as tensors.

        Args:
            var_names: str or list of str. None returns None.
            index: sample index. None returns all samples.
        """
        if var_names is None:
            return None
        if isinstance(var_names, str):
            return self._to_tensor_copy(
                self._storegate.get_data(var_names, self._phase, index)
            )
        if len(var_names) == 1:
            return self._to_tensor_copy(
                self._storegate.get_data(var_names[0], self._phase, index)
            )
        return [
            self._to_tensor_copy(self._storegate.get_data(var_name, self._phase, index))
            for var_name in var_names
        ]

    def __getitem__(
        self,
        index: int,
    ) -> _DatasetItem:
        if self._preload:
            data = self._index_tensor(self._data, index)
            target = self._index_tensor(self._target, index)
            if target is None:
                return data
            return data, target

        data = self._get_tensors(self._input_var_names, index)
        target = self._get_tensors(self._true_var_names, index)
        if target is None:
            return data
        return data, target

    def _index_tensor(
        self,
        tensors: _TensorBatch | None,
        index: int,
    ) -> _TensorBatch | None:
        if tensors is None:
            return None
        if isinstance(tensors, list):
            return [tensor[index] for tensor in tensors]
        return tensors[index]
