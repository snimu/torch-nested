# torch-nested

[![Python 3.7+](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.4+-blue.svg)](https://pypi.org/project/torch/1.4.0/)

[![PyPI](https://img.shields.io/pypi/v/torch-nested)](https://pypi.org/project/torch-nested/)

[![Tests](https://github.com/snimu/torch-nested/actions/workflows/test.yml/badge.svg)](https://github.com/snimu/torch-nested/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/snimu/torch-nested/branch/main/graph/badge.svg)](https://codecov.io/gh/snimu/torch-nested)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/snimu/torch-nested/main.svg)](https://results.pre-commit.ci/latest/github/snimu/torch-nested/main)

[![License](https://img.shields.io/pypi/l/torch-nested)](https://github.com/snimu/torch-nested/blob/main/LICENSE)

Easily manipulate `torch.Tensors` inside highly nested data-structures.

You may want to consider using [torch.nested](https://pytorch.org/docs/stable/nested.html),
but if you are working with nested `dicts`, `lists`, `tuples`, etc. of `torch.Tensors`, 
here is the package for you.

A proper documentation is coming. Until then, a basic example is shown below, and you can look at the docstrings 
or tests of this package for more information.

## Basic usage

Given a nested structure that contains `torch.Tensor`, this package makes it easy to access these `Tensors` and 
work with them: 

```python
import torch
from torch_nested import NestedTensors


INPUT_DATA = [
    (
        torch.ones(3), 
        torch.zeros(2)
    ),
    torch.ones((2, 2, 2)),
    {
        "foo": torch.ones(2), 
        "bar": [], 
        "har": "rar"
    },
    1
]

tensors = NestedTensors(INPUT_DATA)

# Original data preserved in .data-member
assert tensors.data == INPUT_DATA

# Simple accessing and setting
for i, tensor in enumerate(tensors):
    tensors[i] = tensor + i 

# Has basic dunders
assert len(tensors) == 4
assert torch.all(next(tensors) == torch.ones(3))
```

Calling `print(tensors.shape())` would yield:

```
torch_nested.Size(
  [
    (
      torch.Size([3]),
      torch.Size([2])
    ),
    torch.Size([2, 2, 2]),
    {
      foo: torch.Size([2]),
      bar: None,
      har: None
    },
    None
  ]
)

```

### Supported data-structures

The following data-structures are supported so far:

- `torch.Tensor`
- `dict`
- `list`
- `tuple`
- `None`
- Any class with a `.tensors`-attribute
- Any class with a `.data`-attribute, even if it isn't a `torch.Tensor`

For example

```python
class ObjWithTensors:
    tensors = [torch.ones(2), torch.zeros(2)]

class ObjWithData:
    data = [torch.ones(2), torch.zeros(2)]

tensors = NestedTensors([ObjWithTensors(), ObjWithData()])
```

Running `print(tensors.size())` would result in the following output:

```
NestedSize(
  [
    ObjWithTensors(
      tensors: [
        torch.Size([2]),
        torch.Size([2])
      ]
    ),
    ObjWithData(
      data: [
        torch.Size([2]),
        torch.Size([2])
      ]
    )
  ]
)
```

More data-structures will be supported in the future. Any data that is of an unsupported type 
will not have its `Tensors` readable or writable, and `NestedShape` will show `None` there.


