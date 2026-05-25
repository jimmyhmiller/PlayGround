"""
SwiGLU MLP for the Llama backbone of Chatterbox T3.

  gate = x @ gate_w        # (rows, hidden) @ (hidden, intermediate) → (rows, intermediate)
  up   = x @ up_w
  hidden = silu(gate) * up
  out  = hidden @ down_w   # (rows, intermediate) @ (intermediate, hidden) → (rows, hidden)

silu(x) = x * sigmoid(x) = x / (1 + exp(-x))

Weights are pre-transposed by the host (see oracle/dump_mlp_case.py) so we can
use straight matmul (C = A @ B) without a B^T variant.

Matmuls use linalg.matmul[target="gpu"]; the only kernel we own here is the
silu_mul fusion for `silu(gate) * up`.
"""

from std.math import exp
from std.gpu import block_idx, thread_idx
from layout import TileTensor, TensorLayout


def silu_mul_kernel[
    dtype: DType,
    GLayout: TensorLayout,
    ULayout: TensorLayout,
    OutLayout: TensorLayout,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],
    gate: TileTensor[dtype, GLayout, MutAnyOrigin],
    up: TileTensor[dtype, ULayout, MutAnyOrigin],
    n_elems: Int,
):
    """Pointwise: out[i] = silu(gate[i]) * up[i].

    Treats inputs as flat 1D buffers (silu_mul is shape-oblivious — RoPE-style
    pointwise op). Computation in fp32, cast back at write time.

    Launch: grid_dim = ceil(n_elems / BLOCK), block_dim = BLOCK.
    """
    comptime assert gate.flat_rank == 2
    comptime assert up.flat_rank == 2
    comptime assert output.flat_rank == 2

    var idx = block_idx.x * BLOCK + thread_idx.x
    if idx >= n_elems:
        return

    # gate/up/out are 2D (rows, intermediate) row-major; map flat idx back.
    var inter = Int(gate.dim[1]())
    var r = idx // inter
    var c = idx % inter

    var g = rebind[Scalar[dtype]](gate[r, c]).cast[DType.float32]()
    var u = rebind[Scalar[dtype]](up[r, c]).cast[DType.float32]()

    var sigmoid_g = 1.0 / (1.0 + exp(-g))
    var silu_g = g * sigmoid_g
    var result = silu_g * u

    output[r, c] = rebind[output.ElementType](result.cast[dtype]())
