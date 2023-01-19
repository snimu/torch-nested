from __future__ import annotations

from torch_nested.signals import ObjectWithTensorsAttr


def test_len() -> None:
    obj = ObjectWithTensorsAttr("obj", [1, 2, 3])
    assert len(obj) == 3

    obj = ObjectWithTensorsAttr("obj", None)
    assert len(obj) == 0

    obj = ObjectWithTensorsAttr("obj", [])
    assert len(obj) == 0


def test_tensors() -> None:
    obj = ObjectWithTensorsAttr("obj", 1)
    assert obj.tensors == 1
