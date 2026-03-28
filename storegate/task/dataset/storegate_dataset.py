from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.utils.data as tdata

if TYPE_CHECKING:
    from storegate.storegate import StoreGate, StoreGateReadView

class StoreGateDataset(tdata.Dataset):  # type: ignore[type-arg]
    """StoreGate dataset class."""
    def __init__(self,
                 storegate: "StoreGate | StoreGateReadView",
                 phase: str,
                 input_var_names: str | list[str] | None = None,
                 true_var_names: str | list[str] | None = None,
                 preload: bool | str = True,
                 callbacks: list | None = None,
                 device: torch.device | str | None = None) -> None:
        """Initialize the dataset."""

        from storegate.storegate import StoreGate, StoreGateReadView

        if isinstance(storegate, StoreGate):
            self._storegate = storegate.pin()
        elif isinstance(storegate, StoreGateReadView):
            self._storegate = storegate
        else:
            raise TypeError("storegate must be StoreGate or StoreGateReadView")
        self._phase = phase
        self._input_var_names: list[str] = (
            [input_var_names] if isinstance(input_var_names, str)
            else list(input_var_names) if input_var_names is not None
            else []
        )
        self._true_var_names: list[str] = (
            [true_var_names] if isinstance(true_var_names, str)
            else list(true_var_names) if true_var_names is not None
            else []
        )
        self._preload_mode = self._normalize_preload_mode(preload)
        self._device = self._resolve_preload_device(device)
        self._callbacks = callbacks
        try:
            self._size: int = len(self._storegate[phase])
        except RuntimeError as exc:
            raise RuntimeError(
                "StoreGateDataset requires a compiled StoreGate or pinned view. "
                "Call storegate.compile() before creating the dataset."
            ) from exc

        self._data: list[torch.Tensor] | None
        self._target: list[torch.Tensor] | None

        if self._preload_mode != 'none':
            self._data = self._load_vars(self._input_var_names)
            self._target = self._load_vars(self._true_var_names)
        else:
            self._data = None
            self._target = None

    @property
    def preload_mode(self) -> str:
        return self._preload_mode

    def _normalize_preload_mode(self, preload: bool | str) -> str:
        if preload is True:
            return 'cpu'
        if preload is False:
            return 'none'
        if preload in {'cpu', 'gpu'}:
            return preload
        raise ValueError("preload must be one of False, True, 'cpu', or 'gpu'.")

    def _resolve_preload_device(
        self,
        device: torch.device | str | None,
    ) -> torch.device | None:
        if self._preload_mode != 'gpu':
            return None
        if device is None:
            raise ValueError("preload='gpu' requires a device.")

        resolved_device = torch.device(device)
        if resolved_device.type == 'cpu':
            raise ValueError("preload='gpu' requires a non-CPU device.")
        return resolved_device

    def _load_vars(self, var_names: list[str]) -> list[torch.Tensor]:
        """Load all events for *var_names* as tensors."""
        tensors: list[torch.Tensor] = []
        for name in var_names:
            tensor = torch.from_numpy(
                self._storegate[self._phase][name][:]
            )
            if self._preload_mode == 'gpu':
                assert self._device is not None
                tensor = tensor.to(self._device)
            tensors.append(tensor)
        return tensors

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> tuple[torch.Tensor | list[torch.Tensor], torch.Tensor | list[torch.Tensor]]:
        if self._data is not None:
            assert self._target is not None
            data_list = [t[index] for t in self._data]
            target_list = [t[index] for t in self._target]
        else:
            data_list = [
                torch.from_numpy(
                    self._storegate[self._phase][name][index]
                )
                for name in self._input_var_names
            ]
            target_list = [
                torch.from_numpy(
                    self._storegate[self._phase][name][index]
                )
                for name in self._true_var_names
            ]

        data: torch.Tensor | list[torch.Tensor] = data_list[0] if len(data_list) == 1 else data_list
        target: torch.Tensor | list[torch.Tensor] = target_list[0] if len(target_list) == 1 else target_list

        if self._callbacks is not None:
            for callback in self._callbacks:
                data, target = callback(data, target)

        return data, target
