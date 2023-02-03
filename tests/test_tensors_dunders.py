from __future__ import annotations

import copy
import sys

import pytest
import torch
from packaging import version

from torch_nested import NestedTensors

from .fixtures.input_data import INPUT_DATA


def test__getitem__() -> None:
    tensors = NestedTensors(INPUT_DATA)

    assert torch.all(tensors[0] == torch.ones(3))
    assert torch.all(tensors[1] == torch.zeros(2))
    assert torch.all(tensors[2] == torch.ones((2, 2, 2)))
    assert torch.all(tensors[3] == torch.ones(2))
    assert torch.all(tensors[4] == torch.ones(3, 3))
    assert torch.all(tensors[-1] == torch.ones(3, 3))


def test__iter__() -> None:
    tensors = NestedTensors(INPUT_DATA)

    for tensor in tensors:
        assert isinstance(tensor, torch.Tensor)


def test__next__() -> None:
    tensors = NestedTensors(INPUT_DATA)

    for _ in range(6):
        assert isinstance(next(tensors), torch.Tensor)

    with pytest.raises(StopIteration):
        next(tensors)


def test__len__() -> None:
    tensors = NestedTensors(INPUT_DATA)
    assert len(tensors) == 6


def test__setitem__() -> None:
    tensors = NestedTensors(INPUT_DATA)

    tensors[2] = torch.zeros((3, 3, 3))
    assert torch.all(tensors[2] == torch.zeros(3, 3, 3))
    assert torch.all(tensors.data[1] == torch.zeros(3, 3, 3))

    tensors[-1] = torch.zeros(2)
    assert torch.all(tensors[-1] == torch.zeros(2))
    assert torch.all(tensors.data[5].data == torch.zeros(2))

    tensors[4] = torch.zeros(2)
    assert torch.all(tensors[4] == torch.zeros(2))

    with pytest.raises(TypeError):  # tuple should not be assignable
        tensors[0] = torch.zeros(3)


def test__abs__() -> None:
    tensors = NestedTensors([torch.randn(3), torch.randn(3), -torch.ones(3)])
    tensors__abs__ = abs(tensors)

    for tensor, tensor_abs in zip(tensors, tensors__abs__):
        assert torch.all(tensor_abs == abs(tensor))


def test__and__() -> None:
    randn = torch.randn(3) * 2
    randn = randn.to(dtype=torch.int8)
    tensors = NestedTensors([copy.deepcopy(randn), copy.deepcopy(randn)])
    ones = torch.ones(3, dtype=torch.int8)
    tensors__and__ = tensors & ones

    for tensor, tensor_and in zip(tensors, tensors__and__):
        assert torch.all(tensor_and == (tensor & ones))


def test__or__() -> None:
    randn = torch.randn(3) * 2
    randn = randn.to(dtype=torch.int8)
    tensors = NestedTensors([copy.deepcopy(randn), copy.deepcopy(randn)])
    ones = torch.ones(3, dtype=torch.int8)
    tensors__or__ = tensors | ones

    for tensor, tensor_or in zip(tensors, tensors__or__):
        assert torch.all(tensor_or == (tensor | ones))


def test__invert__() -> None:
    tensors = NestedTensors(
        [torch.ones(2, dtype=torch.int8), torch.ones(2, dtype=torch.int8)]
    )
    tensors__invert__ = ~tensors

    for tensor, tensor_invert in zip(tensors, tensors__invert__):
        assert torch.all(~tensor == tensor_invert)


def test__neg__() -> None:
    tensors = NestedTensors([torch.ones(2), torch.ones(2)])
    tensors__neg__ = -tensors

    for tensor, tensor_neg in zip(tensors, tensors__neg__):
        assert torch.all(tensor_neg == -tensor)


def test__pos__() -> None:
    tensors = NestedTensors(INPUT_DATA)
    assert tensors == +tensors


class TestComparisons:
    """Tests for <, >, >=, <=."""

    TENSORS = NestedTensors([torch.randn(3), torch.randn(3)])
    RANDN = torch.randn(3)

    def test__lt__(self) -> None:
        tensors = copy.deepcopy(self.TENSORS)
        tensors__lt__ = tensors < self.RANDN

        for tensor, tensor_lt in zip(tensors, tensors__lt__):
            assert torch.all(tensor_lt == (tensor < self.RANDN))

    def test__le__(self) -> None:
        tensors = copy.deepcopy(self.TENSORS)
        tensors__le__ = tensors <= self.RANDN

        for tensor, tensor_le in zip(tensors, tensors__le__):
            assert torch.all(tensor_le == (tensor <= self.RANDN))

    def test__gt__(self) -> None:
        tensors = copy.deepcopy(self.TENSORS)
        tensors__gt__ = tensors > self.RANDN

        for tensor, tensor_gt in zip(tensors, tensors__gt__):
            assert torch.all(tensor_gt == (tensor > self.RANDN))

    def test__ge__(self) -> None:
        tensors = copy.deepcopy(self.TENSORS)
        tensors__ge__ = tensors >= self.RANDN

        for tensor, tensor_ge in zip(tensors, tensors__ge__):
            assert torch.all(tensor_ge == (tensor >= self.RANDN))


class TestMod:
    """Tests for ``__mod__`-, `__rmod__`-, and `__imod__`-methods."""

    @property
    def tensors_(self) -> NestedTensors:
        return NestedTensors(
            [torch.ones(3, dtype=torch.int8) * 10, torch.ones(3, dtype=torch.int8) * 10]
        )

    def test__mod__(self) -> None:
        tensors = self.tensors_
        tensors__mod__ = tensors % 3

        for tensor, tensor_mod in zip(tensors, tensors__mod__):
            assert torch.all((tensor % 3) == tensor_mod)

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse("1.5.1"),
        reason="Doesn't work for PyTorch 1.4",
    )
    def test__rmod__(self) -> None:
        tensors = self.tensors_
        randn = torch.randn(3) * 1000
        tensors__rmod__ = randn % tensors

        for tensor, tensor_rmod in zip(tensors, tensors__rmod__):
            assert torch.all((randn % tensor) == tensor_rmod)

    def test__imod__(self) -> None:
        tensors = self.tensors_
        tensors_control = copy.deepcopy(tensors)

        tensors %= 3

        for tensor, tensor_control in zip(tensors, tensors_control):
            assert torch.all(tensor == (tensor_control % 3))


class TestAddRaddIadd:
    """Tests for the `__add__`-, `__radd__`-, and `__iadd__`-methods."""

    @staticmethod
    def test__add__() -> None:
        tensors = NestedTensors([torch.ones(2), torch.ones(2)])
        tensors_control = tensors + 1

        for tensor, tensor_control in zip(tensors, tensors_control):
            assert torch.all(tensor + 1 == tensor_control)

        tensors_control = tensors + torch.ones(2)
        for tensor, tensor_control in zip(tensors, tensors_control):
            assert torch.all(tensor + torch.ones(2) == tensor_control)

        tensors = NestedTensors([torch.ones(2), torch.ones(3)])
        with pytest.raises(RuntimeError):
            _ = tensors + torch.ones(2)

    @staticmethod
    def test__radd__() -> None:
        tensors = NestedTensors([torch.ones(2), torch.ones(2)])
        tensors__add__ = tensors + 1
        tensors__radd__ = 1 + tensors

        for tensorl, tensorr in zip(tensors__add__, tensors__radd__):
            assert torch.all(tensorl == tensorr)

    @staticmethod
    def test__iadd__() -> None:
        tensors = NestedTensors([torch.ones(2), torch.ones(2)])
        tensors += 1

        for tensor in tensors:
            assert torch.all(tensor == torch.ones(2) + 1)


class TestSubRsubIsub:
    """Tests for the `__sub__`-, `__rsub__`-, and `__isub__`-methods."""

    @staticmethod
    def test__sub__() -> None:
        tensors = NestedTensors([torch.ones(2), torch.ones(2)])
        tensors_control = tensors - 1

        for tensor, tensor_control in zip(tensors, tensors_control):
            assert torch.all(tensor - 1 == tensor_control)

        tensors_control = tensors - torch.ones(2)
        for tensor, tensor_control in zip(tensors, tensors_control):
            assert torch.all(tensor - torch.ones(2) == tensor_control)

        tensors = NestedTensors([torch.ones(2), torch.ones(3)])
        with pytest.raises(RuntimeError):
            _ = tensors + torch.ones(2)

    @staticmethod
    def test__rsub__() -> None:
        tensors = NestedTensors([torch.ones(2), torch.ones(2)])
        tensors__radd__ = 1 - tensors

        for tensor, tensorr in zip(tensors, tensors__radd__):
            assert torch.all(1 - tensor == tensorr)

    @staticmethod
    def test__iadd__() -> None:
        tensors = NestedTensors([torch.ones(2), torch.ones(2)])
        tensors -= 1

        for tensor in tensors:
            assert torch.all(tensor == torch.ones(2) - 1)


class TestMulRmulImul:
    """Tests for the `__mul__`-, `__ramul__`-, and `__imul__`-methods."""

    @staticmethod
    def test__mul__() -> None:
        tensors = NestedTensors([torch.ones(2), torch.ones(2)])
        tensors__mul__ = tensors * 2

        for tensor, tensor_mul in zip(tensors, tensors__mul__):
            assert torch.all(tensor * 2 == tensor_mul)

        randn = torch.randn(2)
        tensors__mul__ = tensors * randn
        for tensor, tensor_mul in zip(tensors, tensors__mul__):
            assert torch.all(tensor * randn == tensor_mul)

        with pytest.raises(RuntimeError):
            _ = tensors * torch.ones(3)

    @staticmethod
    def test__rmul__() -> None:
        tensors = NestedTensors([torch.ones(2), torch.ones(2)])
        tensors__mul__ = tensors * 2
        tensors__rmul__ = 2 * tensors

        for tensorl, tensorr in zip(tensors__mul__, tensors__rmul__):
            assert torch.all(tensorl == tensorr)

    @staticmethod
    def test__imul__() -> None:
        tensors = NestedTensors([torch.ones(2), torch.ones(2)])
        tensors *= 2

        for tensor in tensors:
            assert torch.all(tensor == torch.ones(2) * 2)


class TestMatmulRmatmulImatmul:
    """Tests for the `__matmul__`-, `__rmatmul__`-, and `__imatmul__`-methods."""

    @staticmethod
    def test__matmul__() -> None:
        tensors = NestedTensors([torch.ones(2, 2), torch.ones(2, 2)])
        randn = torch.randn((2, 2))
        tensors__matmul__ = tensors @ randn

        for tensor, tensor_matmul in zip(tensors, tensors__matmul__):
            assert torch.all(tensor_matmul == (tensor @ randn))

    @staticmethod
    def test__rmatmul__() -> None:
        tensors = NestedTensors([torch.ones(2, 2), torch.ones(2, 2)])
        randn = torch.randn((2, 2))
        tensors__rmatmul__ = randn @ tensors

        for tensor, tensor_matmul in zip(tensors, tensors__rmatmul__):
            assert torch.all(tensor_matmul == (randn @ tensor))

    @staticmethod
    def test__imatmul__() -> None:
        tensors = NestedTensors([torch.ones(2, 2), torch.ones(2, 2)])
        randn = torch.randn((2, 2))
        tensors @= randn

        for tensor in tensors:
            assert torch.all(tensor == (torch.ones(2, 2) @ randn))


class TestTruedivRtruedivItruediv:
    """Tests for the `__truediv__`-, `__rtruediv__`-, and `__itruediv__`-methods."""

    @staticmethod
    def test__truediv__() -> None:
        tensors = NestedTensors([torch.ones(2), torch.ones(2)])
        tensors__div__ = tensors / 2

        for tensor, tensor_div in zip(tensors, tensors__div__):
            assert torch.all(tensor_div == tensor / 2)

    @staticmethod
    def test__rtruediv__() -> None:
        tensors = NestedTensors([torch.ones(2), torch.ones(2)])
        tensors__div__ = tensors / 0.5
        tensors__rdiv__ = 2 / tensors

        for t1, t2 in zip(tensors__div__, tensors__rdiv__):
            assert torch.all(torch.abs(t1 - t2) < 1e-6)

    @staticmethod
    def test__itruediv__() -> None:
        tensors = NestedTensors([torch.ones(2), torch.ones(2)])
        tensors /= 2

        for tensor in tensors:
            assert torch.all(tensor == torch.ones(2) / 2)


class TestFloordivs:
    """Tests for the different floordiv-methods."""

    TENSORS = NestedTensors([torch.randn(2) * 4, torch.randn(2) * 4])

    def test__floordiv__(self) -> None:
        tensors = copy.deepcopy(self.TENSORS)
        tensors__fd__ = tensors // 2

        for tensor, tensor_fd in zip(tensors, tensors__fd__):
            assert torch.all(tensor_fd == tensor // 2)

    @pytest.mark.skipif(
        (
            version.parse(torch.__version__) < version.parse("1.5.1")
            or (
                # sys.version gives the version-str and then a long text...
                version.parse(sys.version.split(" ", maxsplit=1)[0])
                < version.parse("3.8")
            )
        ),
        reason="`__rfloordiv__` doesn't work for Python3.7 or PyTorch1.4",
    )
    def test__rfloordiv__(self) -> None:
        tensors = copy.deepcopy(self.TENSORS)
        randn = torch.randn(2) * 8
        tensor__rfd__ = randn // tensors

        for tensor, tensor_rfd in zip(tensors, tensor__rfd__):
            assert torch.all(tensor_rfd == randn // tensor)

    def test__ifloordiv__(self) -> None:
        tensors = copy.deepcopy(self.TENSORS)
        tensors //= 2

        for i, tensor in enumerate(tensors):
            assert torch.all(tensor == self.TENSORS[i] // 2)


class TestShifts:
    """Tests for the different shift-methods."""

    INPUTS = [torch.ones(2).to(dtype=torch.int8), torch.ones(2).to(dtype=torch.int8)]
    TENSORS = NestedTensors(INPUTS)

    def test__lshift__rshift__(self) -> None:
        tensors = copy.deepcopy(self.TENSORS)
        tensors__lshift__ = tensors << 1
        tensors__rshift__ = tensors >> 1

        for i, tensor in enumerate(tensors):
            assert torch.all(tensors__lshift__[i] == tensor << 1)
            assert torch.all(tensors__rshift__[i] == tensor >> 1)

    def test__rlshift__rrshift__(self) -> None:
        tensors = copy.deepcopy(self.TENSORS)
        threes = (torch.ones(2) * 3).to(dtype=torch.int8)
        tensors__rlshift__ = threes << tensors
        tensors__rrshift__ = threes >> tensors

        for i, tensor in enumerate(tensors):
            assert torch.all(tensors__rlshift__[i] == threes << tensor)
            assert torch.all(tensors__rrshift__[i] == threes >> tensor)

    def test__ilshift__(self) -> None:
        tensors = copy.deepcopy(self.TENSORS)
        tensors <<= 1

        for i, tensor in enumerate(tensors):
            assert torch.all(tensor == self.TENSORS[i] << 1)

    def test__irshift__(self) -> None:
        tensors = copy.deepcopy(self.TENSORS)
        tensors >>= 1

        for i, tensor in enumerate(tensors):
            assert torch.all(tensor == self.TENSORS[i] >> 1)


class TestPow:
    """Tests for the `__pow__`-, `__rpow__`-, and `__ipow__`-methods."""

    RANDN = torch.randn(3)

    @property
    def input_data(self) -> list[torch.Tensor]:
        return [copy.deepcopy(self.RANDN), copy.deepcopy(self.RANDN)]

    def test__pow__(self) -> None:
        tensors = NestedTensors(self.input_data)
        tensors__rpow__ = tensors**2

        for tensor, tensor_rpow in zip(tensors, tensors__rpow__):
            assert torch.all(tensor_rpow == (tensor**2))

    def test__rpow__(self) -> None:
        tensors = NestedTensors(self.input_data)
        tensors__rpow__ = 2**tensors

        for tensor, tensor_rpow in zip(tensors, tensors__rpow__):
            assert torch.all(tensor_rpow == (2**tensor))

    def test__ipow__(self) -> None:
        tensors = NestedTensors(self.input_data)
        tensors_control = NestedTensors(self.input_data)
        tensors **= 2

        assert tensors != tensors_control

        for t, tc in zip(tensors, tensors_control):
            assert torch.all(t == (tc**2))
