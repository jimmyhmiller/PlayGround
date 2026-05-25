"""
Small "glue" pieces of flow.inference:
  embedding_lookup_kernel  — flow.input_embedding (vocab x emb_dim) lookup by int64 token id.
  flow.spk_embed_affine_layer is just `linear_kernel` (already in src/layernorm.mojo).
"""
from std.gpu import block_idx, thread_idx
from layout import TileTensor, TensorLayout, row_major


def build_t_span_kernel[
    dtype: DType, OutLayout: TensorLayout, BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (N+1,)
    n_steps: Int,
):
    """t_span = torch.linspace(0, 1, n_steps + 1)."""
    comptime assert output.flat_rank == 1
    var idx = block_idx.x * BLOCK + thread_idx.x
    if idx > n_steps: return
    var v: Float32 = Float32(idx) / Float32(n_steps)
    output[idx] = rebind[output.ElementType](v.cast[dtype]())


def build_rel_pos_emb_kernel[
    dtype: DType, OutLayout: TensorLayout, D_MODEL: Int, BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (1, 2*T-1, D)
    time: Int,
):
    """Espnet relative positional encoding (precomputed for length 2*T-1).

    pe_positive[i, 2k]   = sin(i * exp(-2k * log(10000)/D))
    pe_positive[i, 2k+1] = cos(i * exp(-2k * log(10000)/D))
    pe_negative[i, 2k]   = sin(-i * ...)
    pe_negative[i, 2k+1] = cos(-i * ...)
    Then flip pe_positive, concat [flipped_positive, pe_negative[1:]].

    Output shape (1, 2T-1, D_MODEL).
    Launch: grid = 2T-1, block_dim = BLOCK over D_MODEL.
    """
    from std.math import sin, cos, log, exp, pi
    comptime assert output.flat_rank == 3
    var bid = block_idx.x
    var tid = thread_idx.x
    var two_t_minus_1 = 2 * time - 1
    # Compute the conceptual position p:
    # First T positions correspond to flipped pe_positive [T-1, T-2, ..., 0].
    # Next T-1 positions correspond to pe_negative[1..T-1] (negative indices).
    var p: Int
    var is_negative: Int
    if bid < time:
        # bid in [0, T-1] → flipped positive → conceptual pos = (T-1-bid)
        p = (time - 1 - bid)
        is_negative = 0
    else:
        # bid in [T, 2T-2] → pe_negative[bid - T + 1] → conceptual pos = (bid - T + 1), negative
        p = (bid - time + 1)
        is_negative = 1
    var d = tid
    var emb_factor: Float32 = Float32(log(Float32(10000.0))) / Float32(D_MODEL)
    while d < D_MODEL:
        # 2k = d if d even, 2k+1 = d if d odd. Index into base freq with k = d // 2.
        var k = d // 2
        var coeff: Float32 = exp(-2.0 * Float32(k) * emb_factor)
        var pos_f: Float32 = Float32(p)
        if is_negative == 1:
            pos_f = -pos_f
        var arg = pos_f * coeff
        var val: Float32
        if d % 2 == 0:
            val = sin(arg)
        else:
            val = cos(arg)
        output[0, bid, d] = rebind[output.ElementType](val.cast[dtype]())
        d += BLOCK


def build_conds_kernel[
    dtype: DType, OutLayout: TensorLayout, PromptLayout: TensorLayout, BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],     # (B, 80, T_h)  zeros with prompt pasted at [:mel_len1]
    prompt_feat: TileTensor[dtype, PromptLayout, MutAnyOrigin],  # (B, mel_len1, 80) — BTC layout
    batch: Int, mel_c: Int, t_h: Int, mel_len1: Int,
):
    """Build the CFM cond tensor: zeros(B, 80, T_h), then output[:, :, :mel_len1] = prompt_feat.transpose.
    Equivalent to upstream:
        conds = torch.zeros([B, T_h, 80], device=...).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)  # → (B, 80, T_h)
    """
    comptime assert output.flat_rank == 3
    comptime assert prompt_feat.flat_rank == 3
    var bid = block_idx.x
    var tid = thread_idx.x
    var c = bid % mel_c
    var b = bid // mel_c
    var t = tid
    while t < t_h:
        var v: Float32 = 0.0
        if t < mel_len1:
            v = rebind[Scalar[dtype]](prompt_feat[b, t, c]).cast[DType.float32]()
        output[b, c, t] = rebind[output.ElementType](v.cast[dtype]())
        t += BLOCK


def build_mask_kernel[
    dtype: DType, OutLayout: TensorLayout, BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, 1, T)
    batch: Int, time: Int, valid_len: Int,
):
    """Build a (B, 1, T) mask: 1.0 for positions < valid_len, else 0.0.
    Replicates `(~make_pad_mask(lens)).to(dtype)`.
    """
    comptime assert output.flat_rank == 3
    var bid = block_idx.x
    var tid = thread_idx.x
    var b = bid
    var t = tid
    while t < time:
        var v: Float32 = 1.0 if t < valid_len else 0.0
        output[b, 0, t] = rebind[output.ElementType](v.cast[dtype]())
        t += BLOCK


def gaussian_noise_kernel[
    dtype: DType, OutLayout: TensorLayout, BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],
    n: Int,
    seed: UInt64,
):
    """Fill `output` with deterministic Gaussian-like noise via LCG + Box-Muller-style approx.
    Used as random initial z for CFM. Not byte-exact to torch.randn but provides valid noise.
    """
    comptime assert output.flat_rank == 1
    var idx = block_idx.x * BLOCK + thread_idx.x
    if idx >= n: return
    # Per-element deterministic state.
    var s: UInt64 = seed ^ (UInt64(idx) * UInt64(2654435761))
    # Two LCG draws → Box-Muller approx: u1+u2 has mean 0, var 2/3; scale to ~unit variance.
    s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
    var u1: Float32 = Float32(Int(s >> UInt64(33))) / Float32(2147483647.0) - 1.0
    s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
    var u2: Float32 = Float32(Int(s >> UInt64(33))) / Float32(2147483647.0) - 1.0
    var v: Float32 = (u1 + u2) * Float32(1.2247449)
    output[idx] = rebind[output.ElementType](v.cast[dtype]())


def embedding_lookup_kernel[
    dtype: DType,
    OutLayout: TensorLayout,
    TokLayout: TensorLayout,
    WLayout: TensorLayout,
    EMB_DIM: Int,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, T, EMB_DIM)
    tokens: TileTensor[DType.int64, TokLayout, MutAnyOrigin],  # (B, T) int64
    weight: TileTensor[dtype, WLayout, MutAnyOrigin],     # (vocab, EMB_DIM)
    batch: Int, time: Int,
):
    """out[b, t, :] = weight[tokens[b, t], :]. Launch grid=B*T, block_dim=BLOCK over EMB_DIM."""
    comptime assert tokens.flat_rank == 2
    comptime assert weight.flat_rank == 2
    comptime assert output.flat_rank == 3
    var bid = block_idx.x
    var tid = thread_idx.x
    var t = bid % time
    var b = bid // time
    var tok = Int(rebind[Scalar[DType.int64]](tokens[b, t]))
    var d = tid
    while d < EMB_DIM:
        var w = rebind[Scalar[dtype]](weight[tok, d]).cast[DType.float32]()
        output[b, t, d] = rebind[output.ElementType](w.cast[dtype]())
        d += BLOCK


def normalize_l2_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    OutLayout: TensorLayout,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, D)
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],        # (B, D)
    batch: Int, dim: Int, eps: Float32,
):
    """L2 normalize each row: y = x / max(||x||_2, eps). Launch grid=B, block_dim=BLOCK."""
    comptime assert inp.flat_rank == 2
    comptime assert output.flat_rank == 2
    from std.gpu.sync import barrier
    from std.gpu.memory import AddressSpace
    from std.math import sqrt
    from layout import stack_allocation
    var b = block_idx.x
    var tid = thread_idx.x

    var smem = stack_allocation[DType.float32,
        address_space=AddressSpace.SHARED](row_major[BLOCK]())

    var s: Float32 = 0.0
    var d = tid
    while d < dim:
        var v = rebind[Scalar[dtype]](inp[b, d]).cast[DType.float32]()
        s += v * v
        d += BLOCK
    smem[tid] = s
    barrier()
    if tid == 0:
        var total: Float32 = 0.0
        for i in range(BLOCK):
            total += rebind[Scalar[DType.float32]](smem[i])
        var n: Float32 = sqrt(total)
        if n < eps:
            n = eps
        smem[0] = 1.0 / n
    barrier()
    var inv_norm = rebind[Scalar[DType.float32]](smem[0])

    var d2 = tid
    while d2 < dim:
        var v = rebind[Scalar[dtype]](inp[b, d2]).cast[DType.float32]()
        output[b, d2] = rebind[output.ElementType]((v * inv_norm).cast[dtype]())
        d2 += BLOCK
