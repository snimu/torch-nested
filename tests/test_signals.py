from __future__ import annotations

from torch_nested.type_signals import ObjectWithDataAttr, ObjectWithTensorsAttr


class TestObjectWithTensorsAttr:
    """Tests for ObjectWithTensorsAttr."""

    @staticmethod
    def test_len() -> None:
        obj = ObjectWithTensorsAttr("obj", [1, 2, 3])
        assert len(obj) == 3

        obj = ObjectWithTensorsAttr("obj", None)
        assert len(obj) == 0

        obj = ObjectWithTensorsAttr("obj", [])
        assert len(obj) == 0

    @staticmethod
    def test_tensors() -> None:
        obj = ObjectWithTensorsAttr("obj", 1)
        assert obj.tensors == 1


class TestObjectWithDataAttr:
    """Tests for ObjectWithDataAttr."""

    @staticmethod
    def test_len() -> None:
        obj = ObjectWithDataAttr("obj", [1, 2, 3])
        assert len(obj) == 3

        obj = ObjectWithDataAttr("obj", None)
        assert len(obj) == 0

        obj = ObjectWithDataAttr("obj", [])
        assert len(obj) == 0

    @staticmethod
    def test_tensors() -> None:
        obj = ObjectWithDataAttr("obj", 1)
        assert obj.data == 1
