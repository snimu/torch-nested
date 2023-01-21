from __future__ import annotations

from typing import Any

import torch

from .type_signals import ObjectWithDataAttr, ObjectWithTensorsAttr


def create_str_repr(data: Any, newline: bool, target_type: str) -> str:
    if newline:
        opening = f"torch_nested.{target_type}(\n  "
        content = create_text(data, spacing=" " * 2, newline=newline)
        closing = "\n)\n"
    else:
        opening = f"torch_nested.{target_type}("
        content = create_text(data, spacing="", newline=newline)
        closing = ")"

    return opening + content + closing


def create_text(data: Any | None, spacing: str, newline: bool) -> str:
    # The seperator after an opening (e.g. "[" or "{")
    sep0 = "\n" if newline else ""
    # The seperator after a closing (e.g. "]" or "}")
    sep1 = "\n" if newline else " "

    if data is None:
        text = "None"
    elif isinstance(data, (torch.Tensor, torch.Size, int)):
        text = str(data)

    elif isinstance(data, dict):
        opening = "{" + f"{sep0}"
        closing = spacing + "}" if sep0 == "\n" else "}"
        spacing += " " * 2
        content = ""

        for i, (key, value) in enumerate(data.items()):
            content += spacing if sep0 == "\n" else ""
            content += str(key) + ": "
            content += create_text(value, spacing, newline)
            content += f"{sep0}" if i == len(data) - 1 else f",{sep1}"

        text = opening + content + closing

    elif isinstance(data, (list, tuple)):
        opening = f"({sep0}" if isinstance(data, tuple) else f"[{sep0}"
        closing = ")" if isinstance(data, tuple) else "]"
        closing = spacing + closing if sep0 == "\n" else closing
        spacing += " " * 2
        content = ""

        for i, item in enumerate(data):
            content += spacing if sep0 == "\n" else ""
            content += create_text(item, spacing, newline)
            content += f"{sep0}" if i == len(data) - 1 else f",{sep1}"

        text = opening + content + closing

    elif isinstance(data, ObjectWithTensorsAttr):
        opening = f"{data.name}({sep0}"
        opening += spacing + " " * 2 + "tensors: " if sep0 == "\n" else "tensors: "
        closing = spacing + ")" if sep0 == "\n" else ")"
        spacing += " " * 4

        text = (
            opening + create_text(data.tensors, spacing, newline) + f"{sep0}" + closing
        )

    elif isinstance(data, ObjectWithDataAttr):
        opening = f"{data.name}({sep0}"
        opening += spacing + " " * 2 + "data: " if sep0 == "\n" else "data: "
        closing = spacing + ")" if sep0 == "\n" else ")"
        spacing += " " * 4

        text = opening + create_text(data.data, spacing, newline) + f"{sep0}" + closing

    else:
        text = "None"

    return text
