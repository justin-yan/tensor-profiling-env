import torch
import triton
from torch import Tensor
import triton.language as tl
import jaxtyping
from jaxtyping import Float32, Int32


def mul_relu_block_spec(x: Float32[Tensor, "100"], y: Float32[Tensor, "90"]) -> Float32[Tensor, "90 100"]:
    return torch.relu(x[None, :] * y[:, None])

@triton.jit
def mul_relu_block_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    """
    Outer multiply + relu.  z_j,i = relu(x_i * y_j).  Assume that N0 > B0 and N1 > B1 and are not necessarily multiples.

    Will result in grids where dim_0 tiles over X and dim_1 tiles over Y.
    """
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    x_range = tl.arange(0, B0)[:, None] + B0 * pid_0  # 0 - 31, 32 - 63, etc.
    x = tl.load(x_ptr + x_range, x_range < N0, 0)

    y_range = tl.arange(0, B1)[None, :] + B1 * pid_1  # 0 - 31, 32 - 63, etc.
    y = tl.load(y_ptr + y_range, y_range < N1, 0)

    # (0,0)   to  (0, 100)
    #  to            to
    # (90, 0) to (90, 100)

    # Torch output layout is *row major*, so
    # (0,0) (0,1), ... (0, 100), (1, 0)
    # so x + N0 * y is the global indexing

    z_range = x_range + N0 * y_range
    z_out = tl.where(x * y > 0, x * y, 0)
    tl.store(z_ptr + z_range, z_out, mask=(x_range < N0) & (y_range < N1))
