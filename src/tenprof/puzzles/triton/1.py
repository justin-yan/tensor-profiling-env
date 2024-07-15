import torch
import triton
from torch import Tensor
import triton.language as tl
import jaxtyping
from jaxtyping import Float32, Int32


def add_spec(x: Float32[Tensor, "32"]) -> Float32[Tensor, "32"]:
    "This is the spec that you should implement. Uses typing to define sizes."
    return x + 10.


@triton.jit
def add_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    """Spec allows you to assume N0 == B0"""
    range = tl.arange(0, B0)
    x = tl.load(x_ptr + range)
    z_ptr = tl.store(z_ptr + range, x + 10)
