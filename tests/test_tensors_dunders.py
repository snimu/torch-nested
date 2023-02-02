from __future__ import annotations

import copy

import pytest
import torch

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
