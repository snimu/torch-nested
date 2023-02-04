from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Union

import torch

from .type_signals import ObjectWithDataAttr, ObjectWithTensorsAttr

SIZE_TYPES = Optional[
    Union[
        torch.Size,
        Sequence[Any],
        Dict[Any, Any],
        ObjectWithTensorsAttr,
        ObjectWithDataAttr,
        int,
    ]
]

NUMBER_TYPES = Optional[Union[int, float, complex]]

TORCH_NUMBER_TYPES = Union[int, float, bool]

EXEC_CALLABLE_TYPES = Union[
    Callable[[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any], torch.Tensor],
    Callable[[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any], torch.Tensor],
    Callable[[Any, Any, Any, Any, Any, Any, Any, Any, Any], torch.Tensor],
    Callable[[Any, Any, Any, Any, Any, Any, Any, Any], torch.Tensor],
    Callable[[Any, Any, Any, Any, Any, Any, Any], torch.Tensor],
    Callable[[Any, Any, Any, Any, Any, Any], torch.Tensor],
    Callable[[Any, Any, Any, Any, Any], torch.Tensor],
    Callable[[Any, Any, Any, Any], torch.Tensor],
    Callable[[Any, Any, Any], torch.Tensor],
    Callable[[Any, Any], torch.Tensor],
    Callable[[Any], torch.Tensor],
]
