from __future__ import annotations

from typing import Any, Sequence, Union

import torch

from .signals_hasattr_tensors import ObjectWithTensorsAttr

SIZE_TYPES = Union[
    torch.Size, Sequence[Any], dict[Any, Any], ObjectWithTensorsAttr, None
]
