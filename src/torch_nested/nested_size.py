from __future__ import annotations

from typing import Any

import torch

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
        return self._create_str_repr(newline=False)

    def __repr__(self) -> str:
        return self._create_str_repr(newline=True)

    def _create_str_repr(self, newline: bool) -> str:
        if newline:
            opening = "torch_nested.NestedSize(\n  "
            content = self._create_text(self._size, spacing=" " * 2, newline=newline)
            closing = "\n)\n"
        else:
            opening = "torch_nested.NestedSize("
            content = self._create_text(self._size, spacing="", newline=newline)
            closing = ")"

        return opening + content + closing

    def _create_text(self, size: Any | None, spacing: str, newline: bool) -> str:
        # The seperator after an opening (e.g. "[" or "{")
        sep0 = "\n" if newline else ""
        # The seperator after a closing (e.g. "]" or "}")
        sep1 = "\n" if newline else " "

        if size is None:
            text = "None"
        elif isinstance(size, (torch.Size, int)):
            text = str(size)

        elif isinstance(size, dict):
            opening = "{" + f"{sep0}"
            closing = spacing + "}" if sep0 == "\n" else "}"
            spacing += " " * 2
            content = ""

            for i, (key, value) in enumerate(size.items()):
                content += spacing if sep0 == "\n" else ""
                content += str(key) + ": "
                content += self._create_text(value, spacing, newline)
                content += f"{sep0}" if i == len(size) - 1 else f",{sep1}"

            text = opening + content + closing

        elif isinstance(size, (list, tuple)):
            opening = f"({sep0}" if isinstance(size, tuple) else f"[{sep0}"
            closing = ")" if isinstance(size, tuple) else "]"
            closing = spacing + closing if sep0 == "\n" else closing
            spacing += " " * 2
            content = ""

            for i, item in enumerate(size):
                content += spacing if sep0 == "\n" else ""
                content += self._create_text(item, spacing, newline)
                content += f"{sep0}" if i == len(size) - 1 else f",{sep1}"

            text = opening + content + closing

        elif isinstance(size, ObjectWithTensorsAttr):
            opening = f"{size.name}({sep0}"
            opening += spacing + " " * 2 + "tensors: " if sep0 == "\n" else "tensors: "
            closing = spacing + ")" if sep0 == "\n" else ")"
            spacing += " " * 4

            text = (
                opening
                + self._create_text(size.tensors, spacing, newline)
                + f"{sep0}"
                + closing
            )

        elif isinstance(size, ObjectWithDataAttr):
            opening = f"{size.name}({sep0}"
            opening += spacing + " " * 2 + "data: " if sep0 == "\n" else "data: "
            closing = spacing + ")" if sep0 == "\n" else ")"
            spacing += " " * 4

            text = (
                opening
                + self._create_text(size.data, spacing, newline)
                + f"{sep0}"
                + closing
            )

        else:
            text = "None"

        return text
