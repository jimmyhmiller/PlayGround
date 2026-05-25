"""
Multi-head attention building blocks for the UpsampleConformerEncoder.

This module provides per-piece kernels so each step can be parity-tested:
  qkv_proj_reshape_kernel    — linear projection + reshape to (B, H, T, D_k)
  scaled_matmul_qkT_kernel   — scores = q @ k^T / sqrt(d_k), shape (B, H, T1, T2)
  add_bias_uv_kernel         — q += pos_bias_u/v before the matmul
  softmax_lastdim_kernel     — softmax along the last dim with optional mask
  matmul_av_kernel           — attn @ v, shape (B, H, T1, D_k)
  reshape_back_kernel        — (B, H, T1, D_k) -> (B, T1, H * D_k)
  rel_shift_kernel           — Espnet relative position shift trick

Conformer feed-forward (already covered by linear_kernel + swish):
  swish_kernel               — x * sigmoid(x)
"""
from std.gpu import block_idx, thread_idx
from std.gpu.sync import barrier
from std.gpu.memory import AddressSpace
from std.math import sqrt, exp
from layout import TileTensor, TensorLayout, row_major, stack_allocation


def swish_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    OutLayout: TensorLayout,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],
    n: Int,
):
    """y = x * sigmoid(x). Operates over a flat 1D buffer view."""
    comptime assert inp.flat_rank == 1
    comptime assert output.flat_rank == 1
    var idx = block_idx.x * BLOCK + thread_idx.x
    if idx >= n: return
    var x = rebind[Scalar[dtype]](inp[idx]).cast[DType.float32]()
    var s: Float32 = 1.0 / (1.0 + exp(-x))
    output[idx] = rebind[output.ElementType]((x * s).cast[dtype]())


def qkv_proj_reshape_gen_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    WLayout: TensorLayout,
    BiasLayout: TensorLayout,
    OutLayout: TensorLayout,
    HAS_BIAS: Bool,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, H, T, D_k)
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],        # (B, T, D_in)
    w: TileTensor[dtype, WLayout, MutAnyOrigin],           # (D_out, D_in) where D_out = H * D_k
    bias: TileTensor[dtype, BiasLayout, MutAnyOrigin],     # (D_out,)
    batch: Int, time: Int, heads: Int, d_k: Int, d_in: Int,
):
    """Linear (B, T, D_in) -> (B, T, H*D_k) then reshape to (B, T, H, D_k)
    and transpose to (B, H, T, D_k). Variant where D_in != D_out."""
    comptime assert inp.flat_rank == 3
    comptime assert w.flat_rank == 2
    comptime assert output.flat_rank == 4

    var bid = block_idx.x
    var tid = thread_idx.x
    var t = bid % time
    var h_idx = (bid // time) % heads
    var b = bid // (time * heads)

    var ki = tid
    while ki < d_k:
        var o = h_idx * d_k + ki
        var acc: Float32 = 0.0
        comptime if HAS_BIAS:
            comptime assert bias.flat_rank == 1
            acc = rebind[Scalar[dtype]](bias[o]).cast[DType.float32]()
        for i in range(d_in):
            var xv = rebind[Scalar[dtype]](inp[b, t, i]).cast[DType.float32]()
            var wv = rebind[Scalar[dtype]](w[o, i]).cast[DType.float32]()
            acc += xv * wv
        output[b, h_idx, t, ki] = rebind[output.ElementType](acc.cast[dtype]())
        ki += BLOCK


def qkv_proj_reshape_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    WLayout: TensorLayout,
    BiasLayout: TensorLayout,
    OutLayout: TensorLayout,
    HAS_BIAS: Bool,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, H, T, D_k)
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],        # (B, T, D)
    w: TileTensor[dtype, WLayout, MutAnyOrigin],           # (D, D)  (out_features = in_features = H * D_k)
    bias: TileTensor[dtype, BiasLayout, MutAnyOrigin],     # (D,)
    batch: Int, time: Int, heads: Int, d_k: Int, d_model: Int,
):
    """Linear projection (B, T, D) -> (B, T, D) then reshape to (B, T, H, D_k) and
    transpose to (B, H, T, D_k). One kernel handles both for efficiency.

    Launch: grid = B*H*T, block_dim = BLOCK over d_k.
    """
    comptime assert inp.flat_rank == 3
    comptime assert w.flat_rank == 2
    comptime assert output.flat_rank == 4

    var bid = block_idx.x
    var tid = thread_idx.x
    # bid = b * H * T + h * T + t
    var t = bid % time
    var h_idx = (bid // time) % heads
    var b = bid // (time * heads)

    var ki = tid
    while ki < d_k:
        var o = h_idx * d_k + ki   # index into (D_out,)
        var acc: Float32 = 0.0
        comptime if HAS_BIAS:
            comptime assert bias.flat_rank == 1
            acc = rebind[Scalar[dtype]](bias[o]).cast[DType.float32]()
        for i in range(d_model):
            var xv = rebind[Scalar[dtype]](inp[b, t, i]).cast[DType.float32]()
            var wv = rebind[Scalar[dtype]](w[o, i]).cast[DType.float32]()
            acc += xv * wv
        output[b, h_idx, t, ki] = rebind[output.ElementType](acc.cast[dtype]())
        ki += BLOCK


def add_pos_bias_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    BiasLayout: TensorLayout,
    OutLayout: TensorLayout,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, H, T, D_k)
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],        # (B, H, T, D_k)
    bias: TileTensor[dtype, BiasLayout, MutAnyOrigin],     # (H, D_k)
    batch: Int, heads: Int, time: Int, d_k: Int,
):
    """Add per-head positional bias: out[b, h, t, d] = inp[b, h, t, d] + bias[h, d]."""
    comptime assert inp.flat_rank == 4
    comptime assert output.flat_rank == 4
    comptime assert bias.flat_rank == 2

    var bid = block_idx.x
    var tid = thread_idx.x
    var t = bid % time
    var h = (bid // time) % heads
    var b = bid // (time * heads)

    var d = tid
    while d < d_k:
        var v = rebind[Scalar[dtype]](inp[b, h, t, d]).cast[DType.float32]()
        var bv = rebind[Scalar[dtype]](bias[h, d]).cast[DType.float32]()
        output[b, h, t, d] = rebind[output.ElementType]((v + bv).cast[dtype]())
        d += BLOCK


def matmul_qk_scaled_kernel[
    dtype: DType,
    QLayout: TensorLayout,
    KLayout: TensorLayout,
    OutLayout: TensorLayout,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, H, T1, T2)
    q: TileTensor[dtype, QLayout, MutAnyOrigin],           # (B, H, T1, D_k)
    k: TileTensor[dtype, KLayout, MutAnyOrigin],           # (B, H, T2, D_k)
    batch: Int, heads: Int, time1: Int, time2: Int, d_k: Int, scale: Float32,
):
    """scores[b, h, t1, t2] = (sum_d q[b, h, t1, d] * k[b, h, t2, d]) * scale.

    Launch: grid = B*H*T1, block_dim = BLOCK over T2.
    """
    comptime assert q.flat_rank == 4
    comptime assert k.flat_rank == 4
    comptime assert output.flat_rank == 4

    var bid = block_idx.x
    var tid = thread_idx.x
    var t1 = bid % time1
    var h = (bid // time1) % heads
    var b = bid // (time1 * heads)

    var t2 = tid
    while t2 < time2:
        var acc: Float32 = 0.0
        for d in range(d_k):
            var qv = rebind[Scalar[dtype]](q[b, h, t1, d]).cast[DType.float32]()
            var kv = rebind[Scalar[dtype]](k[b, h, t2, d]).cast[DType.float32]()
            acc += qv * kv
        output[b, h, t1, t2] = rebind[output.ElementType]((acc * scale).cast[dtype]())
        t2 += BLOCK


def matmul_qp_kernel[
    dtype: DType,
    QLayout: TensorLayout,
    PLayout: TensorLayout,
    OutLayout: TensorLayout,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, H, T1, T_pos)
    q: TileTensor[dtype, QLayout, MutAnyOrigin],           # (B, H, T1, D_k)
    p: TileTensor[dtype, PLayout, MutAnyOrigin],           # (B, H, T_pos, D_k)
    batch: Int, heads: Int, time1: Int, t_pos: Int, d_k: Int,
):
    """matrix_bd = q @ p^T, no scale. Launch: grid = B*H*T1, block_dim = BLOCK over T_pos."""
    comptime assert q.flat_rank == 4
    comptime assert p.flat_rank == 4
    comptime assert output.flat_rank == 4
    var bid = block_idx.x
    var tid = thread_idx.x
    var t1 = bid % time1
    var h = (bid // time1) % heads
    var b = bid // (time1 * heads)
    var tp = tid
    while tp < t_pos:
        var acc: Float32 = 0.0
        for d in range(d_k):
            var qv = rebind[Scalar[dtype]](q[b, h, t1, d]).cast[DType.float32]()
            var pv = rebind[Scalar[dtype]](p[b, h, tp, d]).cast[DType.float32]()
            acc += qv * pv
        output[b, h, t1, tp] = rebind[output.ElementType](acc.cast[dtype]())
        tp += BLOCK


def rel_shift_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    OutLayout: TensorLayout,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, H, T, T)
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],        # (B, H, T, 2T-1)
    batch: Int, heads: Int, time: Int,
):
    """Espnet relative-position shift.

    Equivalent to: zero-pad column-0, reshape to (B, H, 2T, T), drop first row,
    view as (B, H, T, 2T-1), then take first T columns.

    Closed-form rewrite: out[b, h, t1, t2] = inp[b, h, t1, t2 + (T - 1 - t1)]
    where the source column index is (t2 + (T - 1 - t1)). The Espnet shift
    treats the original (B, H, T, 2T-1) matrix as a positional-aware reindex.

    Launch: grid = B*H*T, block_dim = BLOCK over T.
    """
    comptime assert inp.flat_rank == 4
    comptime assert output.flat_rank == 4
    var bid = block_idx.x
    var tid = thread_idx.x
    var t1 = bid % time
    var h = (bid // time) % heads
    var b = bid // (time * heads)
    var two_T_minus_1 = 2 * time - 1
    var t2 = tid
    while t2 < time:
        var src_col = t2 + (time - 1 - t1)
        var v: Float32 = 0.0
        if src_col >= 0 and src_col < two_T_minus_1:
            v = rebind[Scalar[dtype]](inp[b, h, t1, src_col]).cast[DType.float32]()
        output[b, h, t1, t2] = rebind[output.ElementType](v.cast[dtype]())
        t2 += BLOCK


def add_4d_kernel[
    dtype: DType,
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    OutLayout: TensorLayout,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],
    a: TileTensor[dtype, ALayout, MutAnyOrigin],
    b: TileTensor[dtype, BLayout, MutAnyOrigin],
    n: Int,
):
    """Pointwise add over flat 1D view."""
    comptime assert a.flat_rank == 1
    comptime assert b.flat_rank == 1
    comptime assert output.flat_rank == 1
    var idx = block_idx.x * BLOCK + thread_idx.x
    if idx >= n: return
    var av = rebind[Scalar[dtype]](a[idx]).cast[DType.float32]()
    var bv = rebind[Scalar[dtype]](b[idx]).cast[DType.float32]()
    output[idx] = rebind[output.ElementType]((av + bv).cast[dtype]())


def matmul_av_kernel[
    dtype: DType,
    ALayout: TensorLayout,
    VLayout: TensorLayout,
    OutLayout: TensorLayout,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, H, T1, D_k)
    a: TileTensor[dtype, ALayout, MutAnyOrigin],           # (B, H, T1, T2)
    v: TileTensor[dtype, VLayout, MutAnyOrigin],           # (B, H, T2, D_k)
    batch: Int, heads: Int, time1: Int, time2: Int, d_k: Int,
):
    """out[b, h, t1, d] = sum_t2 a[b, h, t1, t2] * v[b, h, t2, d]."""
    comptime assert a.flat_rank == 4
    comptime assert v.flat_rank == 4
    comptime assert output.flat_rank == 4
    var bid = block_idx.x
    var tid = thread_idx.x
    var t1 = bid % time1
    var h = (bid // time1) % heads
    var b = bid // (time1 * heads)
    var d = tid
    while d < d_k:
        var acc: Float32 = 0.0
        for t2 in range(time2):
            var av = rebind[Scalar[dtype]](a[b, h, t1, t2]).cast[DType.float32]()
            var vv = rebind[Scalar[dtype]](v[b, h, t2, d]).cast[DType.float32]()
            acc += av * vv
        output[b, h, t1, d] = rebind[output.ElementType](acc.cast[dtype]())
        d += BLOCK


def merge_heads_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    OutLayout: TensorLayout,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, T, H*D_k) = (B, T, D)
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],        # (B, H, T, D_k)
    batch: Int, heads: Int, time: Int, d_k: Int,
):
    """(B, H, T, D_k).transpose(1, 2).view(B, T, H*D_k). Launch grid=B*T*H, block=BLOCK over D_k."""
    comptime assert inp.flat_rank == 4
    comptime assert output.flat_rank == 3
    var bid = block_idx.x
    var tid = thread_idx.x
    var h = bid % heads
    var t = (bid // heads) % time
    var b = bid // (heads * time)
    var d = tid
    while d < d_k:
        var v = rebind[Scalar[dtype]](inp[b, h, t, d]).cast[DType.float32]()
        output[b, t, h * d_k + d] = rebind[output.ElementType](v.cast[dtype]())
        d += BLOCK


def softmax_lastdim_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    OutLayout: TensorLayout,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, H, T1, T2)
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],        # (B, H, T1, T2)
    batch: Int, heads: Int, time1: Int, time2: Int,
):
    """Numerically-stable softmax along the last dim (T2). Launch: grid=B*H*T1,
    block_dim=BLOCK over T2."""
    comptime assert inp.flat_rank == 4
    comptime assert output.flat_rank == 4

    var bid = block_idx.x
    var tid = thread_idx.x
    var t1 = bid % time1
    var h_idx = (bid // time1) % heads
    var b = bid // (time1 * heads)

    var smem = stack_allocation[DType.float32,
        address_space=AddressSpace.SHARED](row_major[BLOCK]())

    # Pass 1: max.
    var local_max: Float32 = -1.0e38
    var t2 = tid
    while t2 < time2:
        var v = rebind[Scalar[dtype]](inp[b, h_idx, t1, t2]).cast[DType.float32]()
        if v > local_max: local_max = v
        t2 += BLOCK
    smem[tid] = local_max
    barrier()
    if tid == 0:
        var m: Float32 = -1.0e38
        for i in range(BLOCK):
            var v = rebind[Scalar[DType.float32]](smem[i])
            if v > m: m = v
        smem[0] = m
    barrier()
    var max_val = rebind[Scalar[DType.float32]](smem[0])

    # Pass 2: sum of exp(x - max).
    var local_sum: Float32 = 0.0
    var t2b = tid
    while t2b < time2:
        var v = rebind[Scalar[dtype]](inp[b, h_idx, t1, t2b]).cast[DType.float32]()
        local_sum += exp(v - max_val)
        t2b += BLOCK
    smem[tid] = local_sum
    barrier()
    if tid == 0:
        var s: Float32 = 0.0
        for i in range(BLOCK):
            s += rebind[Scalar[DType.float32]](smem[i])
        smem[1] = s
    barrier()
    var sum_val = rebind[Scalar[DType.float32]](smem[1])

    # Pass 3: write.
    var t2c = tid
    while t2c < time2:
        var v = rebind[Scalar[dtype]](inp[b, h_idx, t1, t2c]).cast[DType.float32]()
        var y = exp(v - max_val) / sum_val
        output[b, h_idx, t1, t2c] = rebind[output.ElementType](y.cast[dtype]())
        t2c += BLOCK
