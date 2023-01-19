from __future__ import annotations

from pathlib import Path

import pytest
import torch

from torch_nested import Size, Tensors
from torch_nested.signals import ObjectWithTensorsAttr

from .fixtures.input_data import INPUT_DATA


def test_getitem() -> None:
    size = Tensors(INPUT_DATA).size()
    assert size[0][0] == torch.Size([3])  # type: ignore[index]


def test_getitem_exceptions() -> None:
    size = Size(None)

    with pytest.raises(AttributeError):
        _ = size[0]

    size = Size(ObjectWithTensorsAttr("obj", 1))

    with pytest.raises(AttributeError):
        _ = size[0]

    size = Size([1, 2, 3])
    with pytest.raises(RuntimeError):
        _ = size["not an int"]


def test_torch_size() -> None:
    size = Tensors(torch.ones(3)).size()
    assert isinstance(size, torch.Size)


def test_size_tensors() -> None:
    size = Size(ObjectWithTensorsAttr("obj", torch.Size([3])))
    assert size.tensors == torch.Size([3])

    size = Size([1, 2, 3])

    with pytest.raises(AttributeError):
        _ = size.tensors


def test_size_undefined_size() -> None:
    size = Size([1])
    assert str(size) == "torch_nested.Size([None])"


def test_str() -> None:
    size = Tensors(INPUT_DATA).size()

    filename = "tests/size_outputs/str.out"
    expected = Path(filename).read_text(encoding="utf-8")

    assert str(size) == expected


def test_repr() -> None:
    size = Tensors(INPUT_DATA).size()

    filename = "tests/size_outputs/repr.out"
    expected = Path(filename).read_text(encoding="utf-8")

    assert repr(size) == expected
