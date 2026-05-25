"""VoiceEncoder forward: 3-layer LSTM → proj Linear → ReLU → row-L2-norm.

Drop-in for chatterbox/src/chatterbox/models/voice_encoder/voice_encoder.py
forward(). Input: raw mel (B, T, 40). Output: speaker embed (B, 256), L2-normed.
"""
from std.math import sqrt
from std.gpu import block_idx, thread_idx, global_idx
from std.gpu.sync import barrier
from std.gpu.memory import AddressSpace
from layout import TileTensor, TensorLayout, row_major, stack_allocation


def extract_last_step_kernel[
    dtype: DType,
    SeqLayout: TensorLayout,
    OutLayout: TensorLayout,
    HIDDEN: Int,
    BLOCK: Int,
](
    dst: TileTensor[dtype, OutLayout, MutAnyOrigin],     # (B, 1, HIDDEN)
    seq: TileTensor[dtype, SeqLayout, MutAnyOrigin],     # (B, T, HIDDEN)
    batch: Int, time: Int,
):
    """dst[b, 0, h] = seq[b, T-1, h]. Launch: grid=B, block_dim=BLOCK."""
    comptime assert seq.flat_rank == 3
    comptime assert dst.flat_rank == 3
    var b = block_idx.x
    var h = thread_idx.x
    while h < HIDDEN:
        dst[b, 0, h] = rebind[dst.ElementType](
            rebind[Scalar[dtype]](seq[b, time - 1, h])
        )
        h += BLOCK


def relu_l2_norm_row_kernel[
    dtype: DType,
    XLayout: TensorLayout,
    OutLayout: TensorLayout,
    EMBED: Int,
    BLOCK: Int,
](
    dst: TileTensor[dtype, OutLayout, MutAnyOrigin],     # (B, EMBED)
    x: TileTensor[dtype, XLayout, MutAnyOrigin],         # (B, 1, EMBED) — Linear output
    batch: Int,
):
    """ReLU then L2-normalize each row. Launch: grid=B, block_dim=BLOCK.

    1. clamp x to >= 0
    2. norm = sqrt(sum_h (x[b, h]^2))   (over EMBED elements)
    3. dst[b, h] = x[b, h] / norm
    """
    comptime assert x.flat_rank == 3
    comptime assert dst.flat_rank == 2
    var b = block_idx.x
    var tid = thread_idx.x

    var partial = stack_allocation[DType.float32,
        address_space=AddressSpace.SHARED](row_major[BLOCK]())
    var relud = stack_allocation[DType.float32,
        address_space=AddressSpace.SHARED](row_major[EMBED]())

    # Apply ReLU and accumulate sum-of-squares.
    var sum_sq: Float32 = 0.0
    var h = tid
    while h < EMBED:
        var v = rebind[Scalar[dtype]](x[b, 0, h]).cast[DType.float32]()
        if v < 0.0:
            v = 0.0
        relud[h] = v
        sum_sq += v * v
        h += BLOCK
    partial[tid] = sum_sq
    barrier()

    # Reduce partial sums to partial[0].
    var stride = BLOCK // 2
    while stride > 0:
        if tid < stride:
            partial[tid] = rebind[Scalar[DType.float32]](partial[tid]) + rebind[Scalar[DType.float32]](partial[tid + stride])
        barrier()
        stride //= 2

    var total: Float32 = rebind[Scalar[DType.float32]](partial[0])
    var inv: Float32 = 1.0 / sqrt(total + 1e-30)

    var h2 = tid
    while h2 < EMBED:
        var v: Float32 = rebind[Scalar[DType.float32]](relud[h2]) * inv
        dst[b, h2] = rebind[dst.ElementType](v.cast[dtype]())
        h2 += BLOCK
