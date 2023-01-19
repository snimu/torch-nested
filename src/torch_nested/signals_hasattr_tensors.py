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
    """
    Signal to `Shapes.__getitem__` that the `data.tensors`-attribute
    should be accessed instead of `data[...]` .
    """

    name: str
    size: Any
