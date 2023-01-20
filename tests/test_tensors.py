from __future__ import annotations

import copy

import pytest
import torch

from torch_nested import NestedTensors

from .fixtures.input_data import INPUT_DATA


def test_getitem() -> None:
    tensors = NestedTensors(INPUT_DATA)

    assert torch.all(tensors[0] == torch.ones(3))
    assert torch.all(tensors[1] == torch.zeros(2))
    assert torch.all(tensors[2] == torch.ones((2, 2, 2)))
    assert torch.all(tensors[3] == torch.ones(2))
    assert torch.all(tensors[4] == torch.ones(3, 3))
    assert torch.all(tensors[-1] == torch.ones(3, 3))


def test_iter() -> None:
    tensors = NestedTensors(INPUT_DATA)

    for tensor in tensors:
        assert isinstance(tensor, torch.Tensor)


def test_next() -> None:
    tensors = NestedTensors(INPUT_DATA)

    for _ in range(6):
        assert isinstance(next(tensors), torch.Tensor)

    with pytest.raises(StopIteration):
        next(tensors)


def test_len() -> None:
    tensors = NestedTensors(INPUT_DATA)
    assert len(tensors) == 6


def test_element_size() -> None:
    element_size = (
        torch.ones(3).element_size()
        + torch.zeros(2).element_size()
        + torch.ones((2, 2, 2)).element_size()
        + torch.ones(2).element_size()
        + torch.ones((3, 3)).element_size()
        + torch.ones((3, 3)).element_size()
    )

    tensors = NestedTensors(INPUT_DATA)
    assert tensors.element_size() == element_size


def test_size() -> None:
    tensors = NestedTensors(INPUT_DATA)
    size = tensors.size()

    # The "type: ignore[index]"-comments are needed because
    #   `Tensors.size()` returns a `torch_nested.size.Size`,
    #   a `torch.Size`, or `None`, and I don't check which one
    #   it is (because I want an exception to be raised if it is
    #   the wrong one).
    assert size[0][0] == torch.Size([3])  # type: ignore[index]
    assert size[0][1] == torch.Size([2])  # type: ignore[index]
    assert size[1] == torch.Size([2, 2, 2])  # type: ignore[index]
    assert size[2]["foo"] == torch.Size([2])  # type: ignore[index]
    assert size[2]["bar"] is None  # type: ignore[index]
    assert size[2]["har"] is None  # type: ignore[index]
    assert size[3] is None  # type: ignore[index]
    assert size[4].tensors == torch.Size([3, 3])  # type: ignore[union-attr, index]
    assert size[5].data == torch.Size([3, 3])  # type: ignore[union-attr, index]


def test_setitem() -> None:
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


def test_just_tensor() -> None:
    tensors = NestedTensors(torch.ones(3))

    assert torch.all(tensors[0] == torch.ones(3))
    assert len(tensors) == 1

    tensors[0] = torch.zeros(3)
    assert torch.all(tensors[0] == torch.zeros(3))


def test_empty() -> None:
    tensors = NestedTensors(None)
    assert len(tensors) == 0


class TestAbs:
    """A class for testing the `abs`- and `abs_`-methods of `NestedTensors`."""

    @staticmethod
    def test_abs() -> None:
        input_data = [torch.randn(10) for _ in range(10)]

        tensors = NestedTensors(copy.deepcopy(input_data))
        tensors_control = NestedTensors(copy.deepcopy(input_data))
        tensors_abs = tensors.abs()

        # Check that abs() doesn't change original data:
        for tensor, tensor_control in zip(tensors, tensors_control):
            assert torch.all(tensor == tensor_control)

        # Check that torch.abs() was applied:
        for tensor, tensor_abs in zip(tensors, tensors_abs):
            assert torch.all(torch.abs(tensor) == tensor_abs)

    @staticmethod
    def test_abs_() -> None:
        input_data = [torch.randn(5), torch.randn(5), torch.randn(5)]
        tensors = NestedTensors(copy.deepcopy(input_data))
        tensors_control = NestedTensors(copy.deepcopy(input_data))

        tensors.abs_()

        for tensor, tensor_control in zip(tensors, tensors_control):
            assert torch.all(tensor == torch.abs(tensor_control))


class TestAdd:
    """Tests for the `add`- and `add_`-methods of `NestedTensors`."""

    dtypes = [torch.float32, torch.int8, torch.complex64]
    number_to_add_dict = {
        torch.float32: 1.0,
        torch.int8: 1,
        torch.complex64: complex(1.0, 0.0),
    }

    def test_add(self) -> None:
        for dtype in self.dtypes:
            tensors = NestedTensors([torch.ones(2), torch.ones(2)])
            tensors_control = NestedTensors([torch.ones(2), torch.ones(2)])
            tensors_add_tensor = tensors.add(torch.ones(2, dtype=dtype), alpha=2)

            number_to_add = self.number_to_add_dict.get(dtype)
            tensors_add_number = tensors.add(number_to_add)

            # Check that tensors hasn't changed
            for tensor, tensor_control in zip(tensors, tensors_control):
                assert torch.all(tensor == tensor_control)

            # Check that addition was successful for tensors
            for tensor, tensor_add in zip(tensors, tensors_add_tensor):
                assert torch.all(
                    tensor.add(torch.ones(2, dtype=dtype), alpha=2) == tensor_add
                )

            # Check that addition was successful for numbers
            for tensor, tensor_add in zip(tensors, tensors_add_number):
                number_to_add = self.number_to_add_dict.get(dtype)
                assert torch.all(
                    tensor.add(number_to_add) == tensor_add  # type: ignore[arg-type]
                )

    def test_add_(self) -> None:
        for dtype in self.dtypes:
            tensors = NestedTensors([torch.ones(2), torch.ones(2)])
            tensors_control = NestedTensors([torch.ones(2), torch.ones(2)])

            tensors_add_tensor = tensors.add_(torch.ones(2, dtype=dtype), alpha=2)

            number_to_add = self.number_to_add_dict.get(dtype)
            tensors_add_number = tensors.add_(number_to_add)

            # Check that tensors has changed
            for tensor, tensor_control in zip(tensors, tensors_control):
                assert torch.all(tensor != tensor_control)

            # Check that addition was successful for tensors
            for tensor, tensor_add in zip(tensors, tensors_add_tensor):
                assert torch.all(tensor == tensor_add)

            # Check that addition was successful for numbers
            for tensor, tensor_add in zip(tensors, tensors_add_number):
                assert torch.all(tensor == tensor_add)

    def test_add_wrong_shape(self) -> None:
        tensors = NestedTensors([torch.ones(2), torch.ones(2)])
        with pytest.raises(RuntimeError):
            tensors.add(torch.ones(3))
