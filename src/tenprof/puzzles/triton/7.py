import torch
import triton
from torch import Tensor
import triton.language as tl
import jaxtyping
from jaxtyping import Float32, Int32



def sum_spec(x: Float32[Tensor, "4 200"]) -> Float32[Tensor, "4"]:
    return x.sum(1)

@triton.jit
def sum_kernel(x_ptr, z_ptr, N0, N1, T, B0: tl.constexpr, B1: tl.constexpr):
    """
    Sum of a batch of numbers.

    Uses one program blocks. Block size `B0` represents a range of batches of  `x` of length `N0`.
    Each element is of length `T`. Process it `B1 < T` elements at a time.

    $$z_{i} = \sum^{T}_j x_{i,j} =  \text{ for } i = 1\ldots N_0$$

    Hint: You will need a for loop for this problem. These work and look the same as in Python.
    """

    row_idx = tl.program_id(0)

    accumulator = 0
    for off in range(0, T, B1):
        i_range = tl.arange(0, B0)[:, None] + row_idx * B0
        j_range = tl.arange(off, off + B1)[None, :]
        # Torch Tensors are row-major
        load_range = i_range * T + j_range
        x_block = tl.load(x_ptr + load_range, (i_range < N0) & (j_range < T), 0)
        accumulator += tl.sum(x_block)
        tl.store(z_ptr + row_idx, accumulator)
