"""
LayerNorm kernel: applies normalization along the last dimension.

  y[..., c] = (x[..., c] - mean(x[..., :])) / sqrt(var(x[..., :]) + eps) * weight[c] + bias[c]

For a (B, T, C) input, the mean/var are computed per (b, t) along the C dim.
"""
from std.gpu import block_idx, thread_idx
from std.gpu.sync import barrier
from std.gpu.memory import AddressSpace
from std.math import sqrt
from layout import TileTensor, TensorLayout, row_major, stack_allocation


def layernorm_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    PLayout: TensorLayout,
    OutLayout: TensorLayout,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, T, C)
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],        # (B, T, C)
    weight: TileTensor[dtype, PLayout, MutAnyOrigin],      # (C,)
    bias: TileTensor[dtype, PLayout, MutAnyOrigin],        # (C,)
    batch: Int, time: Int, channels: Int, eps: Float32,
):
    """Compute LayerNorm over channel dim (last). Launch: grid = B*T, block_dim = BLOCK.
    Threads cooperate via shared memory for mean+variance reduction.
    """
    comptime assert inp.flat_rank == 3
    comptime assert output.flat_rank == 3
    comptime assert weight.flat_rank == 1
    comptime assert bias.flat_rank == 1

    var bid = block_idx.x
    var tid = thread_idx.x
    var t = bid % time
    var b = bid // time

    var smem = stack_allocation[DType.float32,
        address_space=AddressSpace.SHARED](row_major[BLOCK]())

    # Pass 1: sum.
    var s: Float32 = 0.0
    var c = tid
    while c < channels:
        s += rebind[Scalar[dtype]](inp[b, t, c]).cast[DType.float32]()
        c += BLOCK
    smem[tid] = s
    barrier()
    if tid == 0:
        var total: Float32 = 0.0
        for i in range(BLOCK):
            total += rebind[Scalar[DType.float32]](smem[i])
        smem[0] = total / Float32(channels)
    barrier()
    var mean_val = rebind[Scalar[DType.float32]](smem[0])

    # Pass 2: sum of squared deviations.
    var s2: Float32 = 0.0
    var c2 = tid
    while c2 < channels:
        var d = rebind[Scalar[dtype]](inp[b, t, c2]).cast[DType.float32]() - mean_val
        s2 += d * d
        c2 += BLOCK
    smem[tid] = s2
    barrier()
    if tid == 0:
        var total2: Float32 = 0.0
        for i in range(BLOCK):
            total2 += rebind[Scalar[DType.float32]](smem[i])
        smem[1] = total2 / Float32(channels)   # variance (biased, as PyTorch LN uses)
    barrier()
    var var_val = rebind[Scalar[DType.float32]](smem[1])
    var inv_std: Float32 = 1.0 / sqrt(var_val + eps)

    # Apply.
    var c3 = tid
    while c3 < channels:
        var x = rebind[Scalar[dtype]](inp[b, t, c3]).cast[DType.float32]()
        var g = rebind[Scalar[dtype]](weight[c3]).cast[DType.float32]()
        var bb = rebind[Scalar[dtype]](bias[c3]).cast[DType.float32]()
        var y = (x - mean_val) * inv_std * g + bb
        output[b, t, c3] = rebind[output.ElementType](y.cast[dtype]())
        c3 += BLOCK


def transpose_btc_to_bct_kernel[
    dtype: DType, InLayout: TensorLayout, OutLayout: TensorLayout, BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, C, T)
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],        # (B, T, C)
    batch: Int, time: Int, channels: Int,
):
    """Permute (B, T, C) -> (B, C, T). Launch: grid = B*C, block_dim = BLOCK over T."""
    comptime assert inp.flat_rank == 3
    comptime assert output.flat_rank == 3
    var bid = block_idx.x
    var tid = thread_idx.x
    var c = bid % channels
    var b = bid // channels
    var t = tid
    while t < time:
        var v = rebind[Scalar[dtype]](inp[b, t, c]).cast[DType.float32]()
        output[b, c, t] = rebind[output.ElementType](v.cast[dtype]())
        t += BLOCK


def transpose_bct_to_btc_kernel[
    dtype: DType, InLayout: TensorLayout, OutLayout: TensorLayout, BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, T, C)
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],        # (B, C, T)
    batch: Int, channels: Int, time: Int,
):
    """Permute (B, C, T) -> (B, T, C). Launch: grid = B*T, block_dim = BLOCK over C."""
    comptime assert inp.flat_rank == 3
    comptime assert output.flat_rank == 3
    var bid = block_idx.x
    var tid = thread_idx.x
    var t = bid % time
    var b = bid // time
    var c = tid
    while c < channels:
        var v = rebind[Scalar[dtype]](inp[b, c, t]).cast[DType.float32]()
        output[b, t, c] = rebind[output.ElementType](v.cast[dtype]())
        c += BLOCK


def residual_add_kernel[
    dtype: DType, ALayout: TensorLayout, BLayout: TensorLayout, OutLayout: TensorLayout,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],
    a: TileTensor[dtype, ALayout, MutAnyOrigin],
    b: TileTensor[dtype, BLayout, MutAnyOrigin],
    n: Int,
):
    """Pointwise out = a + b on flat 1D buffers."""
    comptime assert a.flat_rank == 1
    comptime assert b.flat_rank == 1
    comptime assert output.flat_rank == 1
    var idx = block_idx.x * BLOCK + thread_idx.x
    if idx >= n: return
    var av = rebind[Scalar[dtype]](a[idx]).cast[DType.float32]()
    var bv = rebind[Scalar[dtype]](b[idx]).cast[DType.float32]()
    output[idx] = rebind[output.ElementType]((av + bv).cast[dtype]())


def linear_kernel[
    dtype: DType,
    XLayout: TensorLayout,
    WLayout: TensorLayout,
    BiasLayout: TensorLayout,
    OutLayout: TensorLayout,
    HAS_BIAS: Bool,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, T, OUT_FEATURES)
    x: TileTensor[dtype, XLayout, MutAnyOrigin],           # (B, T, IN_FEATURES)
    w: TileTensor[dtype, WLayout, MutAnyOrigin],           # (OUT_FEATURES, IN_FEATURES)
    bias: TileTensor[dtype, BiasLayout, MutAnyOrigin],     # (OUT_FEATURES,) — only read if HAS_BIAS
    batch: Int, time: Int, in_features: Int, out_features: Int,
):
    """Linear layer over the last dim: y[b, t, o] = sum_i x[b, t, i] * w[o, i] (+ bias[o]).

    Launch: grid = B*T, block_dim = BLOCK. Threads stride over OUT_FEATURES.
    """
    comptime assert x.flat_rank == 3
    comptime assert w.flat_rank == 2
    comptime assert output.flat_rank == 3

    var bid = block_idx.x
    var tid = thread_idx.x
    var t = bid % time
    var b = bid // time

    var o = tid
    while o < out_features:
        var acc: Float32 = 0.0
        comptime if HAS_BIAS:
            comptime assert bias.flat_rank == 1
            acc = rebind[Scalar[dtype]](bias[o]).cast[DType.float32]()
        for i in range(in_features):
            var xv = rebind[Scalar[dtype]](x[b, t, i]).cast[DType.float32]()
            var wv = rebind[Scalar[dtype]](w[o, i]).cast[DType.float32]()
            acc += xv * wv
        output[b, t, o] = rebind[output.ElementType](acc.cast[dtype]())
        o += BLOCK
