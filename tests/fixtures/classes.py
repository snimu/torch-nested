from dataclasses import dataclass
from typing import Any


@dataclass
class ObjectWithTensors:
    """An object with a `tensors`-attribute."""

    tensors: Any
