from __future__ import annotations

import copy
from typing import Any, Generator, Sequence

import torch


class Tensors:
    """TODO"""

    def __init__(self, data: Any) -> None:
        self.data = data

        self._size: torch.Size | Sequence[Any] | dict[Any, Any] | None = None
        self._element_size: int | None = None
        self._access_keys: list[list[Any]] = []
        self._next_index: int = 0

    def __getitem__(self, key: Any) -> torch.Tensor:
        if not self._access_keys:
            self._size, self._element_size = self._extract_size(self.data, [])

        x = self.data

        for step in self._access_keys[key]:
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
            self._size, self._element_size = self._extract_size(self.data, [])

        self.data = self._setitem_recursive(
            self.data, self._access_keys[key], value, key
        )

    def _setitem_recursive(
        self, data: Any, steps: list[Any], value: Any, key: int
    ) -> Any | None:
        if not steps or not data:
            return data

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
            self._size, self._element_size = self._extract_size(self.data, [])

        return len(self._access_keys)

    def size(self) -> torch.Size | dict[Any, Any] | Sequence[Any] | None:
        if self._size is None:
            self._size, self._element_size = self._extract_size(self.data, [])
        return self._size

    def element_size(self) -> int:
        if self._element_size is None:
            self._size, self._element_size = self._extract_size(self.data, [])
        return self._element_size

    def _extract_size(
        self, data: Any, path: list[Any]
    ) -> tuple[torch.Size | Sequence[Any] | dict[Any, Any] | None, int]:
        if isinstance(data, torch.Tensor):
            self._access_keys.append(copy.deepcopy(path))
            return data.size(), data.element_size()
        if not hasattr(data, "__getitem__") or not data:
            return None, 0

        if isinstance(data, dict):
            dict_size = {}
            element_size = 0

            for key, val in data.items():
                crnt_path = copy.deepcopy(path)
                crnt_path.append(key)
                dict_size[key], element_size_ = self._extract_size(val, crnt_path)
                element_size += element_size_

            return dict_size, element_size

        if isinstance(data, (list, tuple)):
            list_size = []
            element_size = 0

            for i, item in enumerate(data):
                crnt_path = copy.deepcopy(path)
                crnt_path.append(i)
                size_, element_size_ = self._extract_size(item, crnt_path)
                list_size.append(size_)
                element_size += element_size_

            if isinstance(data, tuple):
                return tuple(list_size), element_size
            return list_size, element_size

        # TODO: sets, generators, collections.<...>

        return None, 0
