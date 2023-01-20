from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

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
