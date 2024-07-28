import torch
import triton
from torch import Tensor
import triton.language as tl
import jaxtyping
from jaxtyping import Float32, Int32



def mul_relu_block_back_spec(x: Float32[Tensor, "90 100"], y: Float32[Tensor, "90"],
                             dz: Float32[Tensor, "90 100"]) -> Float32[Tensor, "90 100"]:
    x = x.clone()
    y = y.clone()
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    z = torch.relu(x * y[:, None])
    z.backward(dz)
    dx = x.grad
    return dx

@triton.jit
def mul_relu_block_back_kernel(x_ptr, y_ptr, dz_ptr, dx_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    """
    I found the notation with N0/N1 in this description very confusing, and modified the test case to be
    N0 = 90, N1 = 100.

    The "forward" operation being performed here is simply element-wise multiplication of each row of x with the
        corresponding value from y, and then relu'ing.

    Given the dz's, then dz/dx = drelu/dr' * dr'/dx =
    (1 if r' > 0, 0 otherwise, where r' = x * y)

    y if x*y > 0,
    0 otherwise

    I believe
    """

    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    """
    Since everything is perfectly elementwise, we need to take matching blocks from dz and x, and then assign to dz.
    As long as we pull the correct corresponding rows from y, we can do the corresponding math quite easily.
    
    The grid being dispatched is N0/B0 and N1/B1, which means that here, we need to load B0 x B1 2-D blocks.
    """

    # Load the correct B0 x B1 blocks from dz and x
    i_range = tl.arange(0, B0)[:, None] + pid_0 * B0
    j_range = tl.arange(0, B1)[None, :] + pid_1 * B1
    # Torch Tensors are row-major
    range = i_range * N1 + j_range
    x_block = tl.load(x_ptr + range, (i_range < N0) & (j_range < N1), 0)
    dz_block = tl.load(dz_ptr + range, (i_range < N0) & (j_range < N1), 0)
    y_block = tl.load(y_ptr + i_range, i_range < N0, 0)

    # elementwise forward pass multiplication
    xy_block = x_block * y_block
    relus_block = tl.where(xy_block > 0, 1, 0)
    partials_block = relus_block * y_block
    dx_block = dz_block * partials_block

    tl.store(dx_ptr + range, dx_block, mask=(i_range < N0) & (j_range < N1))
