from __future__ import annotations

import sys
from pathlib import Path

from torch_nested import NestedSize, NestedTensors

from .fixtures.input_data import INPUT_DATA


def overwrite_output(filepath: Path, text: str) -> None:
    """Overwrite the output-file given in path with text."""
    filepath.parent.mkdir(exist_ok=True)
    filepath.touch(exist_ok=True)
    filepath.write_text(text, encoding="utf-8")


class TestStrReprNestedSize:
    """Test the __str__ and __repr__-methods of NestedSize."""

    @staticmethod
    def valid_size_output(text_type: str, dim: int | None = None) -> bool:
        size = NestedTensors(INPUT_DATA).size(dim=dim)

        filename = f"tests/size_outputs/{text_type}"
        filename += ".out" if dim is None else f"_dim{dim}.out"

        filepath = Path(filename)
        text = str(size) if text_type == "str" else repr(size)

        if "--overwrite" in sys.argv:
            overwrite_output(filepath=filepath, text=text)
            return True

        expected = filepath.read_text(encoding="utf-8")
        return text == expected

    def test_str(self) -> None:
        assert self.valid_size_output("str")
        assert self.valid_size_output("str", dim=0)
        assert self.valid_size_output("str", dim=1)
        assert self.valid_size_output("str", dim=2)
        assert self.valid_size_output("str", dim=3)

    def test_repr(self) -> None:
        assert self.valid_size_output("repr")
        assert self.valid_size_output("repr", dim=0)
        assert self.valid_size_output("repr", dim=1)
        assert self.valid_size_output("repr", dim=2)
        assert self.valid_size_output("repr", dim=3)

    def test_dim_doesnt_change_tensors_size(self) -> None:
        tensors = NestedTensors(INPUT_DATA)

        expected = Path("tests/size_outputs/repr_dim1.out").read_text(encoding="utf-8")
        assert repr(tensors.size(dim=1)) == expected

        expected = Path("tests/size_outputs/repr.out").read_text(encoding="utf-8")
        assert repr(tensors.size()) == expected

    @staticmethod
    def test_size_int() -> None:
        size = NestedSize([1])
        assert str(size) == "torch_nested.NestedSize([1])"

    @staticmethod
    def test_size_undefined_dtype() -> None:
        size = NestedSize([1.0])
        assert str(size) == "torch_nested.NestedSize([None])"


class TestStrReprNestedTensors:
    """Test the __str__ and __repr__-methods of NestedTensors."""

    @staticmethod
    def produces_valid_output(text_type: str) -> bool:
        tensors = NestedTensors(INPUT_DATA)
        text = str(tensors) if text_type == "str" else repr(tensors)
        filepath = Path(f"tests/tensors_output/{text_type}.out")

        if "--overwrite" in sys.argv:
            overwrite_output(filepath=filepath, text=text)
            return True

        expected = filepath.read_text(encoding="utf-8")
        return text == expected

    def test_str(self) -> None:
        assert self.produces_valid_output("str")

    def test_repr(self) -> None:
        assert self.produces_valid_output("repr")
