from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AccessTensorsAttr:
    """
    Signal to `Tensors.__getitem__` that the `data.tensors`-attribute
    should be accessed instead of `data[...]`.
    """


@dataclass
class ObjectWithTensorsAttr:
    """Allow `Shape` to represent objects with `tensor`-attribute."""

    name: str
    _size: Any | None

    @property
    def tensors(self) -> Any | None:
        return self._size
