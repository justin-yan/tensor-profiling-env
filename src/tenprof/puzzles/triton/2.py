import torch
import triton
from torch import Tensor
import triton.language as tl
import jaxtyping
from jaxtyping import Float32, Int32


def add2_spec(x: Float32[Tensor, "200"]) -> Float32[Tensor, "200"]:
    return x + 10.

@triton.jit
def add_mask2_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    """Now, you don't get to assume that N0 is exactly B0 nor is it a multiple"""
    pid = tl.program_id(0)  # Assume grid uses single dimension for scaling
    range = tl.arange(0, B0) + pid * B0
    x = tl.load(x_ptr + range, range < N0, 0)  # Masks are used to guard the overflow in the last block
    z_ptr = tl.store(z_ptr + range, x + 10, range < N0)  # Mask also necessary for the output tensor
