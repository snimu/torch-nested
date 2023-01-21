from __future__ import annotations

from pathlib import Path

from torch_nested import NestedSize, NestedTensors

from .fixtures.input_data import INPUT_DATA


class TestStrReprNestedSize:
    """Test the __str__ and __repr__-methods of NestedSize."""

    @staticmethod
    def valid_size_output(text_type: str, dim: int | None = None) -> bool:
        size = NestedTensors(INPUT_DATA).size(dim=dim)

        filename = f"tests/size_outputs/{text_type}"
        filename += ".out" if dim is None else f"_dim{dim}.out"

        expected = Path(filename).read_text(encoding="utf-8")
        text = str(size) if text_type == "str" else repr(size)
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
