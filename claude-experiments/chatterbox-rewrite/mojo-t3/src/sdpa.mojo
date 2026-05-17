"""
Scaled dot-product attention kernels for Llama-style MHA.

Three separate kernels:
  qk_scaled_kernel: Q @ K^T * scale + causal_mask  →  logits   (B,H,S,S)
  softmax_kernel:   softmax over last dim, fp32 internal       (B,H,S,S)
  av_kernel:        probs @ V                                  (B,H,S,D)

We split into three because (1) each is cleanly testable independently and
(2) replacing any one with a fused/tuned version doesn't break the others.

All kernels are templated on dtype; computation is in fp32, cast in/out at the
boundaries (matches HF eager_attention_forward, where softmax is forced fp32).
"""

from std.math import exp, sqrt
from std.gpu import barrier, block_idx, thread_idx, lane_id, WARP_SIZE
from std.gpu.primitives import warp
from std.gpu.memory import AddressSpace
from layout import TileTensor, TensorLayout, row_major, stack_allocation


def qk_scaled_kernel[
    dtype: DType,
    QLayout: TensorLayout,
    KLayout: TensorLayout,
    MaskLayout: TensorLayout,
    OutLayout: TensorLayout,
    HEAD_DIM: Int,
    SEQ: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],          # (B,H,S,S)
    q: TileTensor[dtype, QLayout, MutAnyOrigin],                  # (B,H,S,D)
    k: TileTensor[dtype, KLayout, MutAnyOrigin],                  # (B,H,S,D)
    mask: TileTensor[dtype, MaskLayout, MutAnyOrigin],            # (S,S)
    n_heads: Int,
    scale: Float32,
):
    """logits[b,h,sq,sk] = dot(Q[b,h,sq,:], K[b,h,sk,:]) * scale + mask[sq,sk].

    Launch: grid = B*H*S (one block per (b,h,sq) row), block_dim = S
            (one thread per sk). Each thread reduces over D.
    """
    comptime assert q.flat_rank == 4
    comptime assert k.flat_rank == 4
    comptime assert mask.flat_rank == 2
    comptime assert output.flat_rank == 4

    var bid = block_idx.x
    var sk = thread_idx.x  # 0..S-1

    var sq = bid % SEQ
    var h = (bid // SEQ) % n_heads
    var b = bid // (SEQ * n_heads)

    # Reduce over D in fp32.
    var acc: Float32 = 0.0
    for d in range(HEAD_DIM):
        var qv = rebind[Scalar[dtype]](q[b, h, sq, d]).cast[DType.float32]()
        var kv = rebind[Scalar[dtype]](k[b, h, sk, d]).cast[DType.float32]()
        acc += qv * kv
    var scaled = acc * scale

    var m = rebind[Scalar[dtype]](mask[sq, sk]).cast[DType.float32]()
    var logit_fp32 = scaled + m

    output[b, h, sq, sk] = rebind[output.ElementType](logit_fp32.cast[dtype]())


def softmax_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    OutLayout: TensorLayout,
    SEQ: Int,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],
    n_heads: Int,
):
    """Per-row softmax over the last dim. fp32 internal, cast back to dtype.

    Launch: grid = B*H*S, block_dim = BLOCK. Threads cooperatively reduce over
    SEQ elements in the row.
    """
    comptime assert inp.flat_rank == 4
    comptime assert output.flat_rank == 4
    comptime NUM_WARPS = BLOCK // WARP_SIZE

    var bid = block_idx.x
    var tid = thread_idx.x

    var sq = bid % SEQ
    var h = (bid // SEQ) % n_heads
    var b = bid // (SEQ * n_heads)

    # Stage 1: per-thread max over strided columns.
    var local_max: Float32 = -3.4e38  # roughly -inf; safe initial sentinel
    var col = tid
    while col < SEQ:
        var v = rebind[Scalar[dtype]](inp[b, h, sq, col]).cast[DType.float32]()
        if v > local_max:
            local_max = v
        col += BLOCK

    var warp_max = warp.max(local_max)
    var max_per_warp = stack_allocation[
        DType.float32, address_space=AddressSpace.SHARED
    ](row_major[NUM_WARPS]())
    if lane_id() == 0:
        max_per_warp[tid // WARP_SIZE] = warp_max
    barrier()

    var row_max_shared = stack_allocation[
        DType.float32, address_space=AddressSpace.SHARED
    ](row_major[1]())
    if tid < WARP_SIZE:
        var v: Float32 = -3.4e38
        if tid < NUM_WARPS:
            v = rebind[Scalar[DType.float32]](max_per_warp[tid])
        var m = warp.max(v)
        if tid == 0:
            row_max_shared[0] = m
    barrier()
    var row_max = rebind[Scalar[DType.float32]](row_max_shared[0])

    # Stage 2: per-thread partial sum of exp(x - row_max).
    var local_sum: Float32 = 0.0
    col = tid
    while col < SEQ:
        var v = rebind[Scalar[dtype]](inp[b, h, sq, col]).cast[DType.float32]()
        local_sum += exp(v - row_max)
        col += BLOCK

    var warp_sum = warp.sum(local_sum)
    var sum_per_warp = stack_allocation[
        DType.float32, address_space=AddressSpace.SHARED
    ](row_major[NUM_WARPS]())
    if lane_id() == 0:
        sum_per_warp[tid // WARP_SIZE] = warp_sum
    barrier()

    var row_sum_shared = stack_allocation[
        DType.float32, address_space=AddressSpace.SHARED
    ](row_major[1]())
    if tid < WARP_SIZE:
        var v: Float32 = 0.0
        if tid < NUM_WARPS:
            v = rebind[Scalar[DType.float32]](sum_per_warp[tid])
        var s = warp.sum(v)
        if tid == 0:
            row_sum_shared[0] = s
    barrier()
    var row_sum = rebind[Scalar[DType.float32]](row_sum_shared[0])
    var inv_sum = 1.0 / row_sum

    # Stage 3: write probabilities.
    col = tid
    while col < SEQ:
        var v = rebind[Scalar[dtype]](inp[b, h, sq, col]).cast[DType.float32]()
        var p = exp(v - row_max) * inv_sum
        output[b, h, sq, col] = rebind[output.ElementType](p.cast[dtype]())
        col += BLOCK


def av_kernel[
    dtype: DType,
    PLayout: TensorLayout,
    VLayout: TensorLayout,
    OutLayout: TensorLayout,
    SEQ: Int,
    HEAD_DIM: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],          # (B,H,S,D)
    probs: TileTensor[dtype, PLayout, MutAnyOrigin],              # (B,H,S,S)
    v: TileTensor[dtype, VLayout, MutAnyOrigin],                  # (B,H,S,D)
    n_heads: Int,
):
    """out[b,h,sq,d] = sum_sk probs[b,h,sq,sk] * V[b,h,sk,d].

    Launch: grid = B*H*S (one block per (b,h,sq)), block_dim = D
            (one thread per output column d). Reduces over sk.
    """
    comptime assert probs.flat_rank == 4
    comptime assert v.flat_rank == 4
    comptime assert output.flat_rank == 4

    var bid = block_idx.x
    var d = thread_idx.x

    var sq = bid % SEQ
    var h = (bid // SEQ) % n_heads
    var b = bid // (SEQ * n_heads)

    var acc: Float32 = 0.0
    for sk in range(SEQ):
        var p = rebind[Scalar[dtype]](probs[b, h, sq, sk]).cast[DType.float32]()
        var vv = rebind[Scalar[dtype]](v[b, h, sk, d]).cast[DType.float32]()
        acc += p * vv

    output[b, h, sq, d] = rebind[output.ElementType](acc.cast[dtype]())


# ============================================================================
# Decode-step kernels (Q has seq=1, K/V come from a cache of length `cur_len`)
#
# Layouts:
#   q          (B, H, 1, D)
#   k_cache    (B, H, MAX_CTX, D)
#   v_cache    (B, H, MAX_CTX, D)
#   logits     (B, H, 1, MAX_CTX)
#   probs      (B, H, 1, MAX_CTX)
#   output     (B, H, 1, D)
#
# `cur_len` is a runtime argument <= MAX_CTX; entries [cur_len, MAX_CTX) of
# the cache are considered invalid and are masked out (qk_decode writes a
# very-negative sentinel so softmax assigns them ~0 probability).
# ============================================================================


def qk_decode_kernel[
    dtype: DType,
    QLayout: TensorLayout,
    KLayout: TensorLayout,
    OutLayout: TensorLayout,
    HEAD_DIM: Int,
    MAX_CTX: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],          # (B,H,1,MAX_CTX)
    q: TileTensor[dtype, QLayout, MutAnyOrigin],                  # (B,H,1,D)
    k: TileTensor[dtype, KLayout, MutAnyOrigin],                  # (B,H,MAX_CTX,D)
    n_heads: Int,
    cur_len: Int,
    scale: Float32,
):
    """logits[b,h,0,sk] = dot(Q[b,h,0,:], K_cache[b,h,sk,:]) * scale.

    Invalid slots (sk >= cur_len) get a very-negative sentinel so softmax
    drops them.

    Launch: grid = B*H (one block per (b,h)), block_dim = MAX_CTX (one
    thread per sk).
    """
    comptime assert q.flat_rank == 4
    comptime assert k.flat_rank == 4
    comptime assert output.flat_rank == 4

    var bid = block_idx.x
    var sk = thread_idx.x  # 0..MAX_CTX-1

    var h = bid % n_heads
    var b = bid // n_heads

    if sk >= cur_len:
        # Sentinel ~ -inf so softmax assigns probability 0.
        output[b, h, 0, sk] = rebind[output.ElementType](
            Scalar[DType.float32](-3.4e38).cast[dtype]()
        )
        return

    var acc: Float32 = 0.0
    for d in range(HEAD_DIM):
        var qv = rebind[Scalar[dtype]](q[b, h, 0, d]).cast[DType.float32]()
        var kv = rebind[Scalar[dtype]](k[b, h, sk, d]).cast[DType.float32]()
        acc += qv * kv
    var scaled = acc * scale
    output[b, h, 0, sk] = rebind[output.ElementType](scaled.cast[dtype]())


def softmax_decode_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    OutLayout: TensorLayout,
    MAX_CTX: Int,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],
    n_heads: Int,
):
    """Per-row softmax over MAX_CTX for the single decode row.

    Identical structure to softmax_kernel but with SEQ=1 implicit (sq always 0).
    Sentinels written by qk_decode_kernel make masked positions vanish in fp32
    softmax (exp(-3.4e38) underflows to 0).

    Launch: grid = B*H, block_dim = BLOCK.
    """
    comptime assert inp.flat_rank == 4
    comptime assert output.flat_rank == 4
    comptime NUM_WARPS = BLOCK // WARP_SIZE

    var bid = block_idx.x
    var tid = thread_idx.x

    var h = bid % n_heads
    var b = bid // n_heads

    # Stage 1: per-thread max.
    var local_max: Float32 = -3.4e38
    var col = tid
    while col < MAX_CTX:
        var v = rebind[Scalar[dtype]](inp[b, h, 0, col]).cast[DType.float32]()
        if v > local_max:
            local_max = v
        col += BLOCK

    var warp_max = warp.max(local_max)
    var max_per_warp = stack_allocation[
        DType.float32, address_space=AddressSpace.SHARED
    ](row_major[NUM_WARPS]())
    if lane_id() == 0:
        max_per_warp[tid // WARP_SIZE] = warp_max
    barrier()

    var row_max_shared = stack_allocation[
        DType.float32, address_space=AddressSpace.SHARED
    ](row_major[1]())
    if tid < WARP_SIZE:
        var v: Float32 = -3.4e38
        if tid < NUM_WARPS:
            v = rebind[Scalar[DType.float32]](max_per_warp[tid])
        var m = warp.max(v)
        if tid == 0:
            row_max_shared[0] = m
    barrier()
    var row_max = rebind[Scalar[DType.float32]](row_max_shared[0])

    # Stage 2: per-thread partial sum of exp(x - row_max).
    var local_sum: Float32 = 0.0
    col = tid
    while col < MAX_CTX:
        var v = rebind[Scalar[dtype]](inp[b, h, 0, col]).cast[DType.float32]()
        local_sum += exp(v - row_max)
        col += BLOCK

    var warp_sum = warp.sum(local_sum)
    var sum_per_warp = stack_allocation[
        DType.float32, address_space=AddressSpace.SHARED
    ](row_major[NUM_WARPS]())
    if lane_id() == 0:
        sum_per_warp[tid // WARP_SIZE] = warp_sum
    barrier()

    var row_sum_shared = stack_allocation[
        DType.float32, address_space=AddressSpace.SHARED
    ](row_major[1]())
    if tid < WARP_SIZE:
        var v: Float32 = 0.0
        if tid < NUM_WARPS:
            v = rebind[Scalar[DType.float32]](sum_per_warp[tid])
        var s = warp.sum(v)
        if tid == 0:
            row_sum_shared[0] = s
    barrier()
    var row_sum = rebind[Scalar[DType.float32]](row_sum_shared[0])
    var inv_sum = 1.0 / row_sum

    # Stage 3: write probabilities.
    col = tid
    while col < MAX_CTX:
        var v = rebind[Scalar[dtype]](inp[b, h, 0, col]).cast[DType.float32]()
        var p = exp(v - row_max) * inv_sum
        output[b, h, 0, col] = rebind[output.ElementType](p.cast[dtype]())
        col += BLOCK


def av_decode_kernel[
    dtype: DType,
    PLayout: TensorLayout,
    VLayout: TensorLayout,
    OutLayout: TensorLayout,
    MAX_CTX: Int,
    HEAD_DIM: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],          # (B,H,1,D)
    probs: TileTensor[dtype, PLayout, MutAnyOrigin],              # (B,H,1,MAX_CTX)
    v: TileTensor[dtype, VLayout, MutAnyOrigin],                  # (B,H,MAX_CTX,D)
    n_heads: Int,
    cur_len: Int,
):
    """out[b,h,0,d] = sum_{sk<cur_len} probs[b,h,0,sk] * V_cache[b,h,sk,d].

    Launch: grid = B*H, block_dim = HEAD_DIM.
    """
    comptime assert probs.flat_rank == 4
    comptime assert v.flat_rank == 4
    comptime assert output.flat_rank == 4

    var bid = block_idx.x
    var d = thread_idx.x

    var h = bid % n_heads
    var b = bid // n_heads

    var acc: Float32 = 0.0
    for sk in range(cur_len):
        var p = rebind[Scalar[dtype]](probs[b, h, 0, sk]).cast[DType.float32]()
        var vv = rebind[Scalar[dtype]](v[b, h, sk, d]).cast[DType.float32]()
        acc += p * vv

    output[b, h, 0, d] = rebind[output.ElementType](acc.cast[dtype]())
