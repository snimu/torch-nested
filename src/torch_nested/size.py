from __future__ import annotations

from typing import Any

from .signals_hasattr_tensors import ObjectWithTensorsAttr
from .type_definitions import SIZE_TYPES


class Size:
    """TODO"""

    def __init__(self, size: SIZE_TYPES) -> None:
        self._size = size

    def __getitem__(self, item: Any) -> Any:
        if self._size is None:
            raise AttributeError("`None` has no attribute `__getitem__`")

        if isinstance(self._size, ObjectWithTensorsAttr):
            raise AttributeError(
                "Indexing not possible. Please use the `.tensors`-property."
            )

        try:
            return self._size[item]
        except Exception as e:
            raise RuntimeError(f"Cannot access item {item}.") from e

    @property
    def tensors(self) -> Any:
        if not isinstance(self._size, ObjectWithTensorsAttr):
            raise AttributeError(
                f"The type of `Size._size` is {type(self._size)}, "
                f"which has no attribute `tensors`."
            )
        return self._size.tensors
