from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.utils.data as tdata

if TYPE_CHECKING:
    from storegate.storegate import StoreGate


class StoreGateDataset(tdata.Dataset):  # type: ignore[type-arg]
    """StoreGate dataset class."""
    def __init__(self,
                 storegate: StoreGate,
                 phase: str,
                 input_var_names: str | list[str] | None = None,
                 true_var_names: str | list[str] | None = None,
                 preload: bool = False) -> None:

        self._storegate = storegate
        self._phase = phase
        self._input_var_names = input_var_names
        self._true_var_names = true_var_names
        self._preload = preload
        self._size: int = len(storegate[phase])

        self._data: torch.Tensor | list[torch.Tensor] | None = None
        self._target: torch.Tensor | list[torch.Tensor] | None = None

        if preload:
            self._data = self._get_tensors(input_var_names)
            self._target = self._get_tensors(true_var_names)

    def __len__(self) -> int:
        return self._size

    def _get_tensors(
        self,
        var_names: str | list[str] | None,
        index: int | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Load var_names from storegate as tensors.

        Args:
            var_names: str or list of str.
            index: sample index. None returns all samples.
        """
        if isinstance(var_names, str):
            return torch.as_tensor(self._storegate.get_data(var_names, self._phase, index))
        if len(var_names) == 1:  # type: ignore[arg-type]
            return torch.as_tensor(self._storegate.get_data(var_names[0], self._phase, index))  # type: ignore[index]
        return [self._get_tensors(v, index) for v in var_names]  # type: ignore[union-attr,return-value]

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor | list[torch.Tensor], torch.Tensor | list[torch.Tensor]]:
        if self._preload:
            data = self._index_tensor(self._data, index)
            target = self._index_tensor(self._target, index)
            return data, target

        return self._get_tensors(self._input_var_names, index), \
               self._get_tensors(self._true_var_names, index)

    def _index_tensor(
        self,
        tensors: torch.Tensor | list[torch.Tensor],
        index: int,
    ) -> torch.Tensor | list[torch.Tensor]:
        if isinstance(tensors, list):
            return [self._index_tensor(t, index) for t in tensors]  # type: ignore[return-value]
        return tensors[index]
