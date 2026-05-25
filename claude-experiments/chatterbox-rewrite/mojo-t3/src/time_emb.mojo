"""
Time embedding kernels for the CFM estimator (ConditionalDecoder).

  sinusoidal_pos_emb_kernel:  t scalars (B,) → (B, dim)
      emb[b, k]            = sin(scale * t[b] * exp(-log(10000) * k / (half_dim - 1)))    for k < half_dim
      emb[b, half_dim + k] = cos(scale * t[b] * exp(-log(10000) * k / (half_dim - 1)))    for k < half_dim

  silu_kernel:  pointwise y = x * sigmoid(x)
"""

from std.math import sin, cos, exp, log
from std.gpu import block_idx, thread_idx
from layout import TileTensor, TensorLayout


def sinusoidal_pos_emb_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    OutLayout: TensorLayout,
    DIM: Int,
    HALF_DIM: Int,    # = DIM // 2
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],     # (B, DIM)
    t: TileTensor[dtype, InLayout, MutAnyOrigin],            # (B,)
    batch: Int,
    scale: Float32,
):
    """Matches matcha.decoder.SinusoidalPosEmb. Launch: grid = B, block = DIM."""
    comptime assert t.flat_rank == 1
    comptime assert output.flat_rank == 2

    var b = block_idx.x
    var k = thread_idx.x
    if k >= DIM:
        return

    var t_val = rebind[Scalar[dtype]](t[b]).cast[DType.float32]()
    var log10k = Float32(9.2103403719761836)   # log(10000)
    var decay_step = log10k / Float32(HALF_DIM - 1)

    var half_k: Int
    var is_cos: Bool
    if k < HALF_DIM:
        half_k = k
        is_cos = False
    else:
        half_k = k - HALF_DIM
        is_cos = True

    var arg = scale * t_val * exp(-decay_step * Float32(half_k))
    var v: Float32
    if is_cos:
        v = cos(arg)
    else:
        v = sin(arg)

    output[b, k] = rebind[output.ElementType](v.cast[dtype]())


def silu_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    OutLayout: TensorLayout,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],
    n_elems: Int,
):
    """Pointwise SiLU/Swish: y = x * sigmoid(x).

    Operates on a flat rank-1 view.
    """
    comptime assert inp.flat_rank == 1
    comptime assert output.flat_rank == 1

    var idx = block_idx.x * BLOCK + thread_idx.x
    if idx >= n_elems:
        return

    var x = rebind[Scalar[dtype]](inp[idx]).cast[DType.float32]()
    var sig = 1.0 / (1.0 + exp(-x))
    var y = x * sig
    output[idx] = rebind[output.ElementType](y.cast[dtype]())
