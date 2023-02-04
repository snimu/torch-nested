from __future__ import annotations

import copy

import pytest
import torch
from packaging import version

from torch_nested import NestedTensors

from .fixtures.input_data import INPUT_DATA


def test_element_size() -> None:
    element_size = torch.ones(3).element_size()

    tensors = NestedTensors(INPUT_DATA)
    assert tensors.element_size() == element_size

    tensors = NestedTensors(
        [torch.randn(3, dtype=torch.float32), torch.randn(3, dtype=torch.float64)]
    )

    with pytest.raises(ValueError):
        _ = tensors.element_size()


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
    assert size[4].tensors[0] == torch.Size([3, 3])  # type: ignore[union-attr, index]
    assert size[5].data == torch.Size([3, 3])  # type: ignore[union-attr, index]


def test_element_sizes() -> None:
    tensors = NestedTensors(INPUT_DATA)
    element_sizes = tensors.element_sizes()

    for tensor, es in zip(tensors, element_sizes):
        assert es == tensor.element_size()


def test_sizes() -> None:
    tensors = NestedTensors(INPUT_DATA)
    sizes = tensors.sizes()

    for tensor, size in zip(tensors, sizes):
        assert size == tensor.size()


def test_just_tensor() -> None:
    tensors = NestedTensors(torch.ones(3))

    assert torch.all(tensors[0] == torch.ones(3))
    assert len(tensors) == 1

    tensors[0] = torch.zeros(3)
    assert torch.all(tensors[0] == torch.zeros(3))


def test_empty() -> None:
    tensors = NestedTensors(None)
    assert len(tensors) == 0


class TestAllAny:
    """Tests for all and any."""

    ALL_TRUE = NestedTensors(
        [torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool)]
    )
    SOME_TRUE = NestedTensors(
        [torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)]
    )
    NONE_TRUE = NestedTensors(
        [torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)]
    )

    def test_all(self) -> None:
        assert self.ALL_TRUE.all()
        assert not self.SOME_TRUE.all()
        assert not self.NONE_TRUE.all()

    def test_any(self) -> None:
        assert self.ALL_TRUE.any()
        assert self.SOME_TRUE.any()
        assert not self.NONE_TRUE.any()


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
    def test_absolute() -> None:
        input_data = [torch.randn(5) for _ in range(5)]
        tensors = NestedTensors(input_data)

        for t1, t2 in zip(tensors.absolute(), tensors.abs()):
            assert torch.all(t1 == t2)

    @staticmethod
    def test_abs_() -> None:
        input_data = [torch.randn(5), torch.randn(5), torch.randn(5)]
        tensors = NestedTensors(copy.deepcopy(input_data))
        tensors_control = NestedTensors(copy.deepcopy(input_data))

        tensors.abs_()

        for tensor, tensor_control in zip(tensors, tensors_control):
            assert torch.all(tensor == torch.abs(tensor_control))

    @staticmethod
    def test_absolute_() -> None:
        input_data = [torch.randn(5), torch.randn(5), torch.randn(5)]
        tensors = NestedTensors(copy.deepcopy(input_data))
        tensors_control = NestedTensors(copy.deepcopy(input_data))

        tensors.absolute_()
        tensors_control.abs_()

        for t, tc in zip(tensors, tensors_control):
            assert torch.all(t == tc)


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse("1.6"),
    reason="`add` and `add_` behave differently in versions < 1.6",
)
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

    @staticmethod
    def test_add_wrong_shape() -> None:
        tensors = NestedTensors([torch.ones(2), torch.ones(2)])
        with pytest.raises(RuntimeError):
            tensors.add(torch.ones(3))

    @staticmethod
    def test_immutable_raises_exception() -> None:
        input_data = (torch.ones(2), torch.ones(2))
        tensors = NestedTensors(input_data)

        with pytest.raises(RuntimeError):
            _ = tensors.add_(input_data)  # type: ignore[arg-type]

    @staticmethod
    def test_inplace_returns_self() -> None:
        # Establish baseline about assumptions
        x = torch.ones(2)
        y = x.add_(1)
        y.add_(1)
        assert torch.all(x == y)

        # Repeat this for NestedTensors
        tensors1 = NestedTensors([torch.ones(2), torch.ones(2)])
        tensors2 = tensors1.add_(1)
        tensors2.add_(1)

        for tensor1, tensor2 in zip(tensors1, tensors2):
            assert torch.all(tensor1 == tensor2)


class TestAcos:
    """Tests for `acos`, `arccos`, `acos_`, and `arccos_`."""

    @property
    def tensors(self) -> NestedTensors:
        return NestedTensors([torch.randn(5).clamp(0, 1), torch.randn(5).clamp(0, 1)])

    def test_acos(self) -> None:
        tensors = self.tensors
        tensors_acos = tensors.acos()

        for t, ta in zip(tensors, tensors_acos):
            assert torch.all(t.acos() == ta)

    def test_arccos(self) -> None:
        tensors = self.tensors
        assert (tensors.acos() == tensors.arccos()).all()

    def test_acos_(self) -> None:
        tensors = self.tensors
        tensors_control = copy.deepcopy(tensors)
        tensors.acos_()

        assert (tensors == tensors_control.acos()).all()

    def test_arccos_(self) -> None:
        tensors = self.tensors
        tensors_control = copy.deepcopy(tensors)
        tensors.arccos_()
        tensors_control.acos_()

        assert (tensors == tensors_control).all()


class TestAddbmm:
    """Tests for `addbmm` and `addbmm_`."""

    @property
    def batch(self) -> torch.Tensor:
        return torch.ones((3, 3, 3))

    @property
    def tensors(self) -> NestedTensors:
        return NestedTensors([torch.ones((3, 3)), torch.ones((3, 3))])

    def test_addbmm(self) -> None:
        tensors = self.tensors
        tensors_addbmm = tensors.addbmm(self.batch, self.batch, beta=1.0, alpha=1.0)

        for t, ta in zip(tensors, tensors_addbmm):
            assert torch.all(
                t.addbmm(self.batch, self.batch, beta=1.0, alpha=1.0) == ta
            )

    def test_addbmm_(self) -> None:
        tensors = self.tensors
        tensors_control = self.tensors
        tensors.addbmm_(self.batch, self.batch, beta=1.0, alpha=1.0)

        for t, tc in zip(tensors, tensors_control):
            assert torch.all(
                tc.addbmm(self.batch, self.batch, beta=1.0, alpha=1.0) == t
            )


class TestAddcdiv:
    """Tests for `addcdiv` and `addcdiv_`."""

    T1 = torch.randn((3, 3, 3))
    T2 = torch.randn((3, 3, 3))

    @property
    def tensors(self) -> NestedTensors:
        return NestedTensors([torch.randn((3, 3, 3)), torch.randn((3, 3, 3))])

    def test_addcdiv(self) -> None:
        tensors = self.tensors
        tensors_addcdiv = tensors.addcdiv(self.T1, self.T2, value=0.5)

        for t, ta in zip(tensors, tensors_addcdiv):
            assert torch.all(ta == t.addcdiv(self.T1, self.T2, value=0.5))

    def test_addcdiv_(self) -> None:
        tensors = self.tensors
        tensors_control = copy.deepcopy(tensors)
        tensors.addcdiv_(self.T1, self.T2, value=0.5)

        for t, tc in zip(tensors, tensors_control):
            assert torch.all(t == tc.addcdiv(self.T1, self.T2, value=0.5))


class TestTo:
    """Tests for the `to`-method."""

    @staticmethod
    def test_to_dtype() -> None:
        tensors = NestedTensors(
            [torch.ones(2, dtype=torch.float32), torch.ones(2, dtype=torch.float32)]
        )
        tensors_same = tensors.to(dtype=torch.float32)
        assert tensors_same is tensors

        tensors_moved = tensors.to(dtype=torch.int8)
        for tensor in tensors_moved:
            assert tensor.dtype is torch.int8

    @staticmethod
    def test_to_same_device() -> None:
        tensors = NestedTensors([torch.randn(2), torch.randn(2)])
        tensors_same = tensors.to("cpu")
        assert tensors is tensors_same

    @staticmethod
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Needs CUDA to work.")
    def test_to_other_device() -> None:
        tensors = NestedTensors([torch.randn(2), torch.randn(2)])
        tensors_moved = tensors.to("cuda")

        for tensor in tensors_moved:
            assert tensor.is_cuda
