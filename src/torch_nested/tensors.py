from __future__ import annotations

import copy
from typing import Any, Generator

import torch

from .signals_hasattr_tensors import AccessTensorsAttr, ObjectWithTensorsAttr
from .size import Size
from .type_definitions import SIZE_TYPES


class Tensors:
    """TODO"""

    def __init__(self, data: Any) -> None:
        self.data = data

        self._size: Size | torch.Size | None = None
        self._element_size: int | None = None
        self._access_keys: list[list[Any]] = []
        self._next_index: int = 0

    def __getitem__(self, key: Any) -> torch.Tensor:
        if not self._access_keys:
            self._size, self._element_size = self._update_info()

        x = self.data

        for step in self._access_keys[key]:
            if isinstance(step, AccessTensorsAttr):
                x = x.tensors
            else:
                x = x[step]

        # so far not achieved -> not covered by tests:
        if not isinstance(x, torch.Tensor):
            raise RuntimeError(
                "There was an error retrieving a torch.Tensor from the data. "
                "Please report this to https://github.com/snimu/torch-nested "
                "as an issue and include the traceback. Thank you."
            )

        return x

    def __setitem__(self, key: int, value: Any) -> None:
        if not self._access_keys:
            self._size, self._element_size = self._update_info()

        self.data = self._setitem_recursive(
            self.data, self._access_keys[key], value, key
        )

    def _setitem_recursive(
        self, data: Any, steps: list[Any], value: Any, key: int
    ) -> Any | None:
        if hasattr(data, "tensors"):
            data.tensors = self._setitem_recursive(data.tensors, steps[1:], value, key)
            return data

        if not steps or not data:
            return value

        if len(steps) == 1:
            data[steps[0]] = value

        try:
            data[steps[0]] = self._setitem_recursive(
                data[steps[0]], steps[1:], value, key
            )
            return data
        except TypeError as e:
            what = (
                f"Tensors at index {key} "
                f"is of type {type(data[steps[0]])}, "
                f"which does not support item assignment."
            )
            raise TypeError(what) from e

    def __iter__(self) -> Generator[torch.Tensor, None, None]:
        for i in range(len(self)):
            yield self[i]

    def __next__(self) -> torch.Tensor:
        try:
            item = self[self._next_index]
        except IndexError as e:
            self._next_index = 0
            raise StopIteration() from e

        self._next_index += 1
        return item

    def __len__(self) -> int:
        if not self._access_keys:
            self._size, self._element_size = self._update_info()

        return len(self._access_keys)

    def size(self) -> Size | torch.Size | None:
        if self._size is None:
            self._size, self._element_size = self._update_info()

        return self._size

    def element_size(self) -> int:
        if self._element_size is None:
            self._size, self._element_size = self._update_info()

        return self._element_size

    def _update_info(self) -> tuple[Size | torch.Size | None, int]:
        size, element_size = self._extract_info(self.data, [])

        if not isinstance(size, torch.Size):
            return Size(size), element_size
        return size, element_size

    def _extract_info(self, data: Any, path: list[Any]) -> tuple[SIZE_TYPES, int]:
        size: SIZE_TYPES = None
        element_size: int = 0

        if isinstance(data, torch.Tensor):
            self._access_keys.append(copy.deepcopy(path))
            size, element_size = data.size(), data.element_size()

        elif hasattr(data, "tensors"):
            crnt_path = copy.deepcopy(path)
            crnt_path.append(AccessTensorsAttr())
            tensors_size, element_size = self._extract_info(data.tensors, crnt_path)

            size = ObjectWithTensorsAttr(type(data).__qualname__, tensors_size)

        if isinstance(data, dict):
            dict_size = {}

            for key, val in data.items():
                crnt_path = copy.deepcopy(path)
                crnt_path.append(key)
                dict_size[key], element_size_ = self._extract_info(val, crnt_path)
                element_size += element_size_

            size = dict_size if dict_size else None

        if isinstance(data, (list, tuple)):
            list_size = []

            for i, item in enumerate(data):
                crnt_path = copy.deepcopy(path)
                crnt_path.append(i)
                size_, element_size_ = self._extract_info(item, crnt_path)
                list_size.append(size_)
                element_size += element_size_

            if isinstance(data, tuple):
                size = tuple(list_size) if list_size else None
            size = list_size if list_size else None

        # TODO: sets, generators, collections.<...>

        return size, element_size
