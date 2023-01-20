from __future__ import annotations

import torch

from .classes import ObjectWithTensors

INPUT_DATA = [
    (torch.ones(3), torch.zeros(2)),
    torch.ones((2, 2, 2)),
    {"foo": torch.ones(2), "bar": [], "har": "rar"},
    1,
    ObjectWithTensors(torch.ones((3, 3))),
]
