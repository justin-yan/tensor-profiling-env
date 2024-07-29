import inspect
import logging
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
from jaxtyping import Float
from jaxtyping._array_types import _NamedDim, _FixedDim, _NamedVariadicDim, _SymbolicDim
from torch import Tensor


class TestModule(nn.Module):
    def __init__(self, out_features: int):
        super(TestModule, self).__init__()
        self.linear = nn.Linear(32, out_features)
        self.relu = nn.ReLU()

    def forward(self, x: Float[torch.Tensor, "B 32"]) -> Float[torch.Tensor, "B O"]:
        x = self.linear(x)
        x = self.relu(x)
        return x


dtype_mapping = {
    "bool": torch.bool,
    "bool_": torch.bool,
    "uint4": torch.int32,
    "uint8": torch.int32,
    "uint16": torch.int32,
    "uint32": torch.int32,
    "uint64": torch.int32,
    "int4": torch.int32,
    "int8": torch.int32,
    "int16": torch.int32,
    "int32": torch.int32,
    "int64": torch.int32,
    "bfloat16": torch.float32,
    "float16": torch.float32,
    "float32": torch.float32,
    "float64": torch.float32,
    "complex64": torch.float32,
    "complex128": torch.float32,
}


@dataclass
class TensorSpec:
    shape: tuple[int, ...]
    dtype: torch.dtype


def input_gen(layer: nn.Module, seed: int = 0) -> dict[str, Tensor]:
    """
    For a given layer that is type annotated with jaxtyping, produce a map of mock tensors that can be used like so:

    in_tens = input_gen(layer)
    layer.forward(**in_tens)
    """
    torch.manual_seed(seed)
    random.seed(seed)
    signature = inspect.signature(layer.forward)
    tensor_specs: dict[str, TensorSpec] = {}  # parameter to shape
    # Across all of the parameters, we'll have a mix of `_NamedDim`, `_FixedDim`, `_NamedVariadicDim`, or `_SymbolicDim`
    # We want to pick concrete dimensions for everything that isn't fixed,
    #   and then we want to generate tensors for everything.
    dimension_name_map: dict[str, int] = {}
    for name, param_obj in signature.parameters.items():
        shape: list[int] = []
        for dim in param_obj.annotation.dims:
            match dim:
                case _NamedDim(nm, _, _):
                    sz = dimension_name_map.setdefault(nm, random.randint(4, 64))
                    shape.append(sz)
                case _FixedDim(sz, _):
                    shape.append(sz)
                case _NamedVariadicDim() | _SymbolicDim() | _:
                    raise NotImplementedError("Don't yet handle these dimension cases")
        dt = dtype_mapping[param_obj.annotation.dtypes[0]]
        tensor_specs[name] = TensorSpec(tuple(shape), dt)

    output = {}
    for name, spec in tensor_specs.items():
        match spec.dtype:
            case torch.float32:
                mock_ten = torch.rand(spec.shape, dtype=spec.dtype)
            case torch.int32:
                mock_ten = torch.randint(-100, 100, spec.shape, dtype=spec.dtype)
            case _:
                logging.error(spec)
                raise NotImplementedError("Don't yet handle these dtypes")
        output[name] = mock_ten

    return output


if __name__ == "__main__":
    layer = TestModule(5)
    o = input_gen(layer)
    print(o)
    print(layer(**o))
