"""
Signals to NestedTensors and NestedSize
about the type of some entries to the data they hold.
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AccessTensorsAttr:
    """
    Signal to `NestedTensors.__getitem__` that the `data.tensors`-attribute
    should be accessed instead of `data[...]`.
    """


@dataclass
class AccessDataAttr:
    """
    Signal to `NestedTensors.__getitem__` that the `data.data`-attribute
    should be accessed instead of `data[...]`.
    """


@dataclass
class ObjectWithTensorsAttr:
    """Allow `NestedShape` to represent objects with `tensor`-attribute."""

    name: str
    _size: Any | None

    @property
    def tensors(self) -> Any | None:
        return self._size

    def __len__(self) -> int:
        if self._size is None:
            return 0
        return len(self._size)

    def __bool__(self) -> bool:
        return bool(self._size)


@dataclass
class ObjectWithDataAttr:
    """Allow `NestedShape` to represent objects with `data`-attribute."""

    name: str
    _size: Any | None

    @property
    def data(self) -> Any | None:
        return self._size

    def __len__(self) -> int:
        if self._size is None:
            return 0
        return len(self._size)

    def __bool__(self) -> bool:
        return bool(self._size)
