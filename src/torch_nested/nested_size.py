from __future__ import annotations

from typing import Any

from .str_repr import create_str_repr
from .type_definitions import SIZE_TYPES
from .type_signals import ObjectWithDataAttr, ObjectWithTensorsAttr


class NestedSize:
    """TODO"""

    def __init__(self, size: SIZE_TYPES) -> None:
        self._size = size

    def __getitem__(self, item: Any) -> Any:
        if self._size is None or isinstance(self._size, int):
            raise AttributeError("`None` has no attribute `__getitem__`")

        if isinstance(self._size, ObjectWithTensorsAttr):
            raise AttributeError(
                "Indexing not possible. Please use the `.tensors`-property."
            )

        if isinstance(self._size, ObjectWithDataAttr):
            raise AttributeError(
                "Indexing not possible. Please use the `.data`-property."
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

    def __str__(self) -> str:
        return create_str_repr(
            data=self._size, newline=False, target_type=f"{type(self).__name__}"
        )

    def __repr__(self) -> str:
        return create_str_repr(
            data=self._size, newline=True, target_type=f"{type(self).__name__}"
        )
