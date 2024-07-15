import torch
import triton
from torch import Tensor
import triton.language as tl
import jaxtyping
from jaxtyping import Float32, Int32


def add_vec_spec(x: Float32[Tensor, "32"], y: Float32[Tensor, "32"]) -> Float32[Tensor, "32 32"]:
    return x[None, :] + y[:, None]

@triton.jit
def add_vec_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    """
    This is an outer vector add.  z_j,i = x_i + y_j.  Assume N0 = B0 and N1 = B1
    """
    x_range = tl.arange(0, B0)
    x = tl.load(x_ptr + x_range)[:, None]  # No mask needed

    y_range = tl.arange(0, B1)
    y = tl.load(y_ptr + y_range)[None, :]  # No mask needed

    z_range = x_range[:, None] + N1 * y_range[None, :]
    z_out = x + y
    tl.store(z_ptr + z_range, z_out)
