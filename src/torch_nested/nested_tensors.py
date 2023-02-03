from __future__ import annotations

import copy
from typing import Any, Callable, Generator

import torch

from .nested_size import NestedSize
from .str_repr import create_str_repr
from .type_definitions import NUMBER_TYPES, SIZE_TYPES
from .type_signals import (
    AccessDataAttr,
    AccessTensorsAttr,
    ObjectWithDataAttr,
    ObjectWithTensorsAttr,
)


class NestedTensors:
    """TODO"""

    def __init__(self, data: Any) -> None:
        self.data = data

        self._access_keys: list[list[Any]] = []
        self._size = self._update_info()
        self._next_index: int = 0

    def __getitem__(self, key: Any) -> torch.Tensor:
        x = self.data
        for step in self._access_keys[key]:
            if isinstance(step, AccessTensorsAttr):
                x = x.tensors
            elif isinstance(step, AccessDataAttr):
                x = x.data
            else:
                x = x[step]

        # so far not achieved -> not covered by tests:
        if not isinstance(x, torch.Tensor):
            raise RuntimeError(
                "There was an error retrieving a torch.Tensor from the data. "
                "Please report this to https://github.com/snimu/torch-nested "
                "as an issue and include the traceback. Thank you."
            )

        return x

    def __setitem__(self, key: int, value: Any) -> None:
        self.data = self._setitem_recursive(
            self.data, self._access_keys[key], value, key
        )
        self._size = self._update_info()

    def _setitem_recursive(
        self, data: Any, steps: list[Any], value: Any, key: int
    ) -> Any | None:
        if not steps or not data:
            return value

        if hasattr(data, "tensors"):
            data.tensors = self._setitem_recursive(data.tensors, steps[1:], value, key)
            return data

        if not isinstance(data, torch.Tensor) and hasattr(data, "data"):
            data.data = self._setitem_recursive(data.data, steps[1:], value, key)
            return data

        try:
            data[steps[0]] = self._setitem_recursive(
                data[steps[0]], steps[1:], value, key
            )
            return data
        except TypeError as e:
            what = (
                f"Tensors at index {key} "
                f"is of type {type(data[steps[0]])}, "
                f"which does not support item assignment."
            )
            raise TypeError(what) from e

    def __iter__(self) -> Generator[torch.Tensor, None, None]:
        for i in range(len(self)):
            yield self[i]

    def __next__(self) -> torch.Tensor:
        try:
            item = self[self._next_index]
        except IndexError as e:
            self._next_index = 0
            raise StopIteration() from e

        self._next_index += 1
        return item

    def __len__(self) -> int:
        return len(self._access_keys)

    def __str__(self) -> str:
        return create_str_repr(
            data=self.data, newline=False, target_type=f"{type(self).__name__}"
        )

    def __repr__(self) -> str:
        return create_str_repr(
            data=self.data, newline=True, target_type=f"{type(self).__name__}"
        )

    def __abs__(self) -> NestedTensors:
        return self._math_op(None, op=lambda t, _: abs(t))

    def __neg__(self) -> NestedTensors:
        return self._math_op(None, op=lambda t, _: -t)

    def __pos__(self) -> NestedTensors:
        return self

    def __add__(self, other: Any) -> NestedTensors:
        return self._math_op(other, op=lambda t, o: t + o)

    def __radd__(self, other: Any) -> NestedTensors:
        return self.__add__(other)

    def __iadd__(self, other: Any) -> NestedTensors:
        res = self + other
        self.data = res.data
        return self

    def __sub__(self, other: Any) -> NestedTensors:
        return self._math_op(other, op=lambda t, o: t - o)

    def __rsub__(self, other: Any) -> NestedTensors:
        return self._math_op(other, op=lambda t, o: o - t)

    def __isub__(self, other: Any) -> NestedTensors:
        res = self - other
        self.data = res.data
        return self

    def __mul__(self, other: Any) -> NestedTensors:
        return self._math_op(other, op=lambda t, o: t * o)

    def __rmul__(self, other: Any) -> NestedTensors:
        return self.__mul__(other)

    def __imul__(self, other: Any) -> NestedTensors:
        res = self * other
        self.data = res.data
        return self

    def __pow__(self, other: Any) -> NestedTensors:
        return self._math_op(other, op=lambda t, o: t**o)

    def __rpow__(self, other: Any) -> NestedTensors:
        return self._math_op(other, op=lambda t, o: o**t)

    def __ipow__(self, other: Any) -> NestedTensors:
        res = self**other
        self.data = res.data
        self._update_info()
        return self

    def __matmul__(self, other: Any) -> NestedTensors:
        return self._math_op(other, op=lambda t, o: t @ o)

    def __rmatmul__(self, other: Any) -> NestedTensors:
        return self._math_op(other, op=lambda t, o: o @ t)

    def __imatmul__(self, other: Any) -> NestedTensors:
        res = self @ other
        self.data = res.data
        return self

    def __truediv__(self, other: Any) -> NestedTensors:
        return self._math_op(other, op=lambda t, o: t / o)

    def __rtruediv__(self, other: Any) -> NestedTensors:
        return self._math_op(other, op=lambda t, o: o / t)

    def __itruediv__(self, other: Any) -> NestedTensors:
        res = self / other
        self.data = res.data
        return self

    def __floordiv__(self, other: Any) -> NestedTensors:
        return self._math_op(other, op=lambda t, o: t // o)

    def __rfloordiv__(self, other: Any) -> NestedTensors:
        return self._math_op(other, op=lambda t, o: o // t)

    def __ifloordiv__(self, other: Any) -> NestedTensors:
        res = self // other
        self.data = res.data
        return self

    def __mod__(self, other: Any) -> NestedTensors:
        return self._math_op(other, op=lambda t, o: t % o)

    def __rmod__(self, other: Any) -> NestedTensors:
        return self._math_op(other, op=lambda t, o: o % t)

    def __imod__(self, other: Any) -> NestedTensors:
        res = self % other
        self.data = res.data
        return self

    def __rshift__(self, other: Any) -> NestedTensors:
        return self._math_op(other, op=lambda t, o: t >> o)

    def __rrshift__(self, other: Any) -> NestedTensors:
        return self._math_op(other, op=lambda t, o: o >> t)

    def __irshift__(self, other: Any) -> NestedTensors:
        res = self >> other
        self.data = res.data
        return self

    def __lshift__(self, other: Any) -> NestedTensors:
        return self._math_op(other, op=lambda t, o: t << o)

    def __rlshift__(self, other: Any) -> NestedTensors:
        return self._math_op(other, op=lambda t, o: o << t)

    def __ilshift__(self, other: Any) -> NestedTensors:
        res = self << other
        self.data = res.data
        return self

    def __invert__(self) -> NestedTensors:
        return self._math_op(None, op=lambda t, _: ~t)

    def __lt__(self, other: Any) -> NestedTensors:
        return self._math_op(other, op=lambda t, o: t < o)

    def __le__(self, other: Any) -> NestedTensors:
        return self._math_op(other, op=lambda t, o: t <= o)

    def __gt__(self, other: Any) -> NestedTensors:
        return self._math_op(other, op=lambda t, o: t > o)

    def __ge__(self, other: Any) -> NestedTensors:
        return self._math_op(other, op=lambda t, o: t >= o)

    def __and__(self, other: Any) -> NestedTensors:
        return self._math_op(other, op=lambda t, o: t & o)

    def __rand__(self, other: Any) -> NestedTensors:
        return self._math_op(other, op=lambda t, o: o & t)

    def __iand__(self, other: Any) -> NestedTensors:
        res = self & other
        self.data = res.data
        return self

    def __or__(self, other: Any) -> NestedTensors:
        return self._math_op(other, op=lambda t, o: t | o)

    def __ror__(self, other: Any) -> NestedTensors:
        return self._math_op(other, op=lambda t, o: o | t)

    def __ior__(self, other: Any) -> NestedTensors:
        res = self | other
        self.data = res.data
        return self

    def __xor__(self, other: Any) -> NestedTensors:
        return self._math_op(other, op=lambda t, o: t ^ o)

    def __rxor__(self, other: Any) -> NestedTensors:
        return self._math_op(other, op=lambda t, o: o ^ t)

    def __ixor__(self, other: Any) -> NestedTensors:
        res = self ^ other
        self.data = res.data
        return self

    def _math_op(self, other: Any, op: Callable[[Any, Any], Any]) -> NestedTensors:
        data = copy.deepcopy(self.data)

        for i, tensor in enumerate(self):
            try:
                data[i] = op(tensor, other)
            except Exception as e:
                # Don't use {other=} because it doesn't work with old Python
                raise RuntimeError(
                    f"Couldn't add other to self[{i}], "
                    f"where other={other} and self[i]={self[i]}."
                ) from e

        return NestedTensors(data)

    def size(self, dim: int | None = None) -> NestedSize | torch.Size | None:
        if dim is not None:
            size = self._update_info(dim=dim)  # don't update self._size!
            return size
        return self._size

    def element_size(self) -> int:
        element_size = self[0].element_size() if len(self) > 0 else 0

        for tensor in self:
            if tensor.element_size() != element_size:
                raise ValueError(
                    "Inconsistent element sizes. Please use `element_sizes` instead."
                )

        return element_size

    def sizes(self) -> list[torch.Size]:
        return [tensor.size() for tensor in self]

    def element_sizes(self) -> list[int]:
        return [tensor.element_size() for tensor in self]

    def abs(self) -> NestedTensors:
        return self._exec(torch.abs)

    def abs_(self) -> NestedTensors:
        return self._exec_inplace(torch.abs)

    def add(
        self, other: torch.Tensor | NUMBER_TYPES, *, alpha: NUMBER_TYPES = 1
    ) -> NestedTensors:
        # Ignore arg-type because torch.add has weird overloaded function-types
        return self._exec(torch.add, other, alpha=alpha)  # type: ignore[arg-type]

    def add_(
        self, other: torch.Tensor | NUMBER_TYPES, *, alpha: NUMBER_TYPES = 1
    ) -> NestedTensors:
        # Ignore arg-type because torch.add has weird overloaded function-types
        return self._exec_inplace(
            torch.add, other, alpha=alpha  # type: ignore[arg-type]
        )

    def _update_info(self, dim: int | None = None) -> NestedSize | torch.Size | None:
        size = self._extract_info(self.data, [], dim=dim)

        if not isinstance(size, torch.Size):
            return NestedSize(size)
        return size

    def _extract_info(
        self, data: Any, path: list[Any], dim: int | None = None
    ) -> SIZE_TYPES:
        size: SIZE_TYPES = None

        if isinstance(data, torch.Tensor):
            self._access_keys.append(copy.deepcopy(path))
            size = data.size()

            if dim is not None:
                size = size[dim] if 0 <= dim < len(size) else None

        elif hasattr(data, "tensors"):
            crnt_path = copy.deepcopy(path)
            crnt_path.append(AccessTensorsAttr())
            tensors_size = self._extract_info(data.tensors, crnt_path, dim)

            size = ObjectWithTensorsAttr(type(data).__qualname__, tensors_size)

        elif hasattr(data, "data"):
            crnt_path = copy.deepcopy(path)
            crnt_path.append(AccessDataAttr())
            data_size = self._extract_info(data.data, crnt_path, dim)

            size = ObjectWithDataAttr(type(data).__qualname__, data_size)

        elif isinstance(data, dict):
            dict_size = {}

            for key, val in data.items():
                crnt_path = copy.deepcopy(path)
                crnt_path.append(key)
                dict_size[key] = self._extract_info(val, crnt_path, dim)

            size = dict_size if dict_size else None

        elif isinstance(data, (list, tuple)):
            list_size = []

            for i, item in enumerate(data):
                crnt_path = copy.deepcopy(path)
                crnt_path.append(i)
                size_ = self._extract_info(item, crnt_path, dim)
                list_size.append(size_)

            if isinstance(data, tuple):
                size = tuple(list_size) if list_size else None
            else:
                size = list_size if list_size else None

        return size

    def _exec_inplace(
        self, function: Callable[[Any], torch.Tensor], *args: Any, **kwargs: Any
    ) -> NestedTensors:
        for i, tensor in enumerate(self):
            try:
                self[i] = function(tensor, *args, **kwargs)
            except Exception as e:
                raise RuntimeError(
                    f"Couldn't apply operation {function.__name__} "
                    f"to Tensor of shape {self[i].shape}."
                ) from e

        self._size = self._update_info()
        return self

    def _exec(
        self, function: Callable[[Any], torch.Tensor], *args: Any, **kwargs: Any
    ) -> NestedTensors:
        data_copy = copy.deepcopy(self.data)

        result = NestedTensors(copy.deepcopy(self.data))
        for i, tensor in enumerate(result):
            result[i] = function(tensor, *args, **kwargs)

        self.data = copy.deepcopy(data_copy)
        self._size = self._update_info()
        return result
