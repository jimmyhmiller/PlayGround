"""
Statistics kernels for CAMPPlus StatsPool and BatchNorm-no-affine.

  stats_pool_kernel       — (B, C, T) -> (B, 2C). out[b, c] = mean(x[b,c,:]),
                            out[b, C+c] = std(x[b,c,:], unbiased=True).
  bn_no_affine_kernel     — BN1d with affine=False: (x - mean) / sqrt(var + eps).
                            No weight/bias. Operates on (B, C, T).
  bn_no_affine_2d_kernel  — same but for (B, C) — used after DenseLayer.linear.
"""
from std.gpu import block_idx, thread_idx
from std.gpu.sync import barrier
from std.gpu.memory import AddressSpace
from std.math import sqrt
from layout import TileTensor, TensorLayout, row_major, stack_allocation


def stats_pool_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    OutLayout: TensorLayout,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, 2*C)
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],        # (B, C, T)
    batch: Int, channels: Int, length: Int,
):
    """For each (b, c) compute mean and unbiased std over T.

    Launch: grid = B * C, block_dim = BLOCK. Two-pass within thread block.
    """
    comptime assert inp.flat_rank == 3
    comptime assert output.flat_rank == 2

    var bid = block_idx.x
    var tid = thread_idx.x
    var c = bid % channels
    var b = bid // channels

    # Pass 1: sum.
    var smem = stack_allocation[DType.float32,
        address_space=AddressSpace.SHARED](row_major[BLOCK]())
    var s: Float32 = 0.0
    var t = tid
    while t < length:
        s += rebind[Scalar[dtype]](inp[b, c, t]).cast[DType.float32]()
        t += BLOCK
    smem[tid] = s
    barrier()
    if tid == 0:
        var total: Float32 = 0.0
        for i in range(BLOCK):
            total += rebind[Scalar[DType.float32]](smem[i])
        smem[0] = total / Float32(length)   # mean
    barrier()
    var mean_val = rebind[Scalar[DType.float32]](smem[0])

    # Pass 2: sum of squared deviations.
    var s2: Float32 = 0.0
    var t2 = tid
    while t2 < length:
        var d = rebind[Scalar[dtype]](inp[b, c, t2]).cast[DType.float32]() - mean_val
        s2 += d * d
        t2 += BLOCK
    smem[tid] = s2
    barrier()
    if tid == 0:
        var total2: Float32 = 0.0
        for i in range(BLOCK):
            total2 += rebind[Scalar[DType.float32]](smem[i])
        # Unbiased variance: divide by (N-1).
        var var_val: Float32 = total2 / Float32(length - 1)
        var std_val: Float32 = sqrt(var_val)
        output[b, c] = rebind[output.ElementType](mean_val.cast[dtype]())
        output[b, channels + c] = rebind[output.ElementType](std_val.cast[dtype]())


def bn_no_affine_2d_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    PLayout: TensorLayout,
    OutLayout: TensorLayout,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, C)
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],        # (B, C)
    running_mean: TileTensor[dtype, PLayout, MutAnyOrigin],
    running_var: TileTensor[dtype, PLayout, MutAnyOrigin],
    batch: Int, channels: Int, eps: Float32,
):
    """BN1d on (B, C) input with affine=False.
       y = (x - running_mean[c]) / sqrt(running_var[c] + eps).
    Launch: grid = ceildiv(B*C, BLOCK), block_dim = BLOCK.
    """
    comptime assert inp.flat_rank == 2
    comptime assert output.flat_rank == 2
    comptime assert running_mean.flat_rank == 1
    comptime assert running_var.flat_rank == 1
    var idx = block_idx.x * BLOCK + thread_idx.x
    if idx >= batch * channels:
        return
    var c = idx % channels
    var b = idx // channels
    var rm = rebind[Scalar[dtype]](running_mean[c]).cast[DType.float32]()
    var rv = rebind[Scalar[dtype]](running_var[c]).cast[DType.float32]()
    var inv_std: Float32 = 1.0 / sqrt(rv + eps)
    var v = rebind[Scalar[dtype]](inp[b, c]).cast[DType.float32]()
    var y = (v - rm) * inv_std
    output[b, c] = rebind[output.ElementType](y.cast[dtype]())
