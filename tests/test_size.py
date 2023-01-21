from __future__ import annotations

import pytest
import torch

from torch_nested import NestedSize, NestedTensors
from torch_nested.type_signals import ObjectWithDataAttr, ObjectWithTensorsAttr

from .fixtures.input_data import INPUT_DATA


def test_getitem() -> None:
    size = NestedTensors(INPUT_DATA).size()
    assert size[0][0] == torch.Size([3])  # type: ignore[index]


def test_getitem_exceptions() -> None:
    size = NestedSize(None)
    with pytest.raises(AttributeError):
        _ = size[0]

    size = NestedSize(1)
    with pytest.raises(AttributeError):
        _ = size[0]

    size = NestedSize(ObjectWithTensorsAttr("obj", 1))
    with pytest.raises(AttributeError):
        _ = size[0]

    size = NestedSize(ObjectWithDataAttr("obj", 1))
    with pytest.raises(AttributeError):
        _ = size[0]

    size = NestedSize([1, 2, 3])
    with pytest.raises(RuntimeError):
        _ = size["not an int"]


def test_torch_size() -> None:
    size = NestedTensors(torch.ones(3)).size()
    assert isinstance(size, torch.Size)


def test_size_tensors() -> None:
    size = NestedSize(ObjectWithTensorsAttr("obj", torch.Size([3])))
    assert size.tensors == torch.Size([3])

    size = NestedSize([1, 2, 3])

    with pytest.raises(AttributeError):
        _ = size.tensors
