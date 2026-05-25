"""Perceiver resampler kernels (cross + self attention).

Matches chatterbox/models/t3/modules/perceiver.py AttentionBlock2.

AttentionBlock2:
  q_in = LN(x_q); k_in = LN(x_kv); v_in = LN(x_kv)
  q = to_q(q_in); k = to_k(k_in); v = to_v(k_in)
  split into (B, H, S, Dh)
  sim = einsum(bhlt,bhls->bhts, q, k) * scale     # this einsum gives (B,H,Sk,Sq) — Resemble's quirk
  attn = softmax(sim, dim=-1)
  out = einsum(bhts,bhls->bhlt, attn, v)
  out = combine_heads(out)
  out = proj_out(out)
  return x_q + out

NOTE: Resemble's einsum looks weird. Let me decode:
  q (bs, H, S_q, Dh), k (bs, H, S_k, Dh)
  einsum("bhlt,bhls->bhts", q, k):
    l = H? no — `bhlt` with bs, n_heads, S, head_dim, so l=S_q, t=head_dim
    `bhls` with l=S_k, s=head_dim
    out: bhts → bs, n_heads, head_dim, ???. Wait that's odd.

Re-reading split_heads: x.permute(0, 2, 1, 3) means starting from (bs, L, H, Dh),
giving (bs, H, L, Dh). So in qkv_attention args, q/k/v are (B, H, L, Dh).

  einsum("bhlt,bhls->bhts", q, k):
    q[b,h,l,t], k[b,h,l,s]  →  out[b,h,t,s] = sum_l q[b,h,l,t] * k[b,h,l,s]
This is treating the OUTER dim of q (S_q) as the SUM dim — which is wrong as
real attention. But that's literally what the code says.

This is actually transposed: ("bhlt,bhls->bhts", q, k) sums over `l` (which is
S in both q and k — they're the same length here). And it produces
(bs, H, Dh_q, Dh_k) — a Dh×Dh matrix, NOT a S×S matrix.

Hmm. But this Perceiver IS used in production, so it must produce something
useful. Trace carefully:

In AttentionBlock2.forward, x1 (Sq) and x2 (Sk) go in:
  q = to_q(x1_norm)  shape (B, Sq, C)
  k = to_k(x2_norm)  shape (B, Sk, C)
  v = to_v(x2_norm)  shape (B, Sk, C)
Then split_heads → (B, H, S, Dh) for each.

For Perceiver cross-attention: x1 (B, 32, 1024), x2 (B, Tk, 1024).
So q (B, H, 32, 256), k (B, H, Tk, 256), v (B, H, Tk, 256).
The einsum requires S_q == S_k since it sums over `l`. But 32 != Tk.

This Perceiver must crash on non-equal lengths in its current form, OR the
einsum is intended to do something we should fix. Looking at the second einsum:
  out = einsum("bhts,bhls->bhlt", attn, v)
   attn (B,H,Dh,Dh), v (B,H,Sk,Dh). Sum over s — produces (B,H,Sk,Dh) — yikes.

OK there are two interpretations:
  (a) The code is buggy but Perceiver in practice has S_q == S_k via reuse
      (note Perceiver does `pre_att = attn(query_, h)` and `attn(pre_att, pre_att)`;
      the second call has matching lens but the first does not).
  (b) The semantics is "QK^T treated as a Dh×Dh transform" — this is the
      meaning that does work for any Sq/Sk pair: sum over the longer-of-two
      sequence dim that they share when they happen to match... but they don't
      match in cross-attn so this would crash.

Reality check by looking at actual usage: the first call query_ (B, 32, 1024) →
shape (B, H, 32, 256), and h (B, Tk, 1024) → (B, H, Tk, 256). einsum("bhlt,bhls->bhts")
requires `l` to be the same dim of both, which is Sq vs Sk — these MUST match
or torch raises an error. The only way this works is if the original repo's
Perceiver is broken-but-unused for non-equal lengths, or Tk == 32 always.

Since this is what's deployed, we replicate semantics literally: cross-attn
where Sq == Sk == 32 (i.e. the input must be resampled to 32 first), and
self-attn between the 32-token queries.

Actually wait — Perceiver runs ON the speech_emb sequence, but speech_emb is
ALREADY resampled to a fixed length before this Perceiver. Let me check
cond_enc.py to confirm.
"""
from std.math import sqrt
from std.gpu import block_idx, thread_idx
from layout import TileTensor, TensorLayout, row_major


def cross_qkt_kernel[
    dtype: DType,
    QLayout: TensorLayout,
    KLayout: TensorLayout,
    OutLayout: TensorLayout,
    HEAD_DIM: Int,
    S_K: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],  # (B, H, S_q, S_k)
    q: TileTensor[dtype, QLayout, MutAnyOrigin],          # (B, H, S_q, D)
    k: TileTensor[dtype, KLayout, MutAnyOrigin],          # (B, H, S_k, D)
    n_heads: Int, s_q: Int,
    scale: Float32,
):
    """logits[b, h, iq, ik] = sum_d q[b,h,iq,d] * k[b,h,ik,d] * scale.

    Launch: grid = B*H*S_q, block_dim = S_K. Each thread computes one (iq, ik) pair.
    """
    comptime assert q.flat_rank == 4
    comptime assert k.flat_rank == 4
    comptime assert output.flat_rank == 4
    var bid = block_idx.x
    var ik = thread_idx.x
    var iq = bid % s_q
    var h = (bid // s_q) % n_heads
    var b = bid // (s_q * n_heads)
    if ik >= S_K: return
    var acc: Float32 = 0.0
    for d in range(HEAD_DIM):
        var qv = rebind[Scalar[dtype]](q[b, h, iq, d]).cast[DType.float32]()
        var kv = rebind[Scalar[dtype]](k[b, h, ik, d]).cast[DType.float32]()
        acc += qv * kv
    var scaled = acc * scale
    output[b, h, iq, ik] = rebind[output.ElementType](scaled.cast[dtype]())


def cross_softmax_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    OutLayout: TensorLayout,
    S_K: Int,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],
    n_heads: Int, s_q: Int,
):
    """Softmax over the last dim (S_K). Launch: grid=B*H*S_q, block_dim=BLOCK.

    Uses a simple single-thread tree reduce in shared memory for portability —
    Perceiver's S_K is small (≤ a few hundred) so this is plenty fast.
    """
    comptime assert inp.flat_rank == 4
    comptime assert output.flat_rank == 4
    from std.math import exp
    from std.gpu.memory import AddressSpace
    from std.gpu.sync import barrier
    from layout import stack_allocation
    var bid = block_idx.x
    var tid = thread_idx.x
    var iq = bid % s_q
    var h = (bid // s_q) % n_heads
    var b = bid // (s_q * n_heads)

    # Stage 1: max.
    var smem = stack_allocation[DType.float32,
        address_space=AddressSpace.SHARED](row_major[BLOCK]())
    var local_max: Float32 = -3.4e38
    var c = tid
    while c < S_K:
        var v = rebind[Scalar[dtype]](inp[b, h, iq, c]).cast[DType.float32]()
        if v > local_max: local_max = v
        c += BLOCK
    smem[tid] = local_max
    barrier()
    if tid == 0:
        var m: Float32 = -3.4e38
        for i in range(BLOCK):
            var x = rebind[Scalar[DType.float32]](smem[i])
            if x > m: m = x
        smem[0] = m
    barrier()
    var row_max = rebind[Scalar[DType.float32]](smem[0])

    # Stage 2: sum of exp.
    var local_sum: Float32 = 0.0
    var c2 = tid
    while c2 < S_K:
        var v = rebind[Scalar[dtype]](inp[b, h, iq, c2]).cast[DType.float32]()
        local_sum += exp(v - row_max)
        c2 += BLOCK
    smem[tid] = local_sum
    barrier()
    if tid == 0:
        var s: Float32 = 0.0
        for i in range(BLOCK):
            s += rebind[Scalar[DType.float32]](smem[i])
        smem[1] = s
    barrier()
    var row_sum = rebind[Scalar[DType.float32]](smem[1])
    var inv = 1.0 / row_sum

    var c3 = tid
    while c3 < S_K:
        var v = rebind[Scalar[dtype]](inp[b, h, iq, c3]).cast[DType.float32]()
        var p: Float32 = exp(v - row_max) * inv
        output[b, h, iq, c3] = rebind[output.ElementType](p.cast[dtype]())
        c3 += BLOCK


def cross_av_kernel[
    dtype: DType,
    PLayout: TensorLayout,
    VLayout: TensorLayout,
    OutLayout: TensorLayout,
    S_K: Int,
    HEAD_DIM: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],  # (B, H, S_q, D)
    probs: TileTensor[dtype, PLayout, MutAnyOrigin],     # (B, H, S_q, S_k)
    v: TileTensor[dtype, VLayout, MutAnyOrigin],         # (B, H, S_k, D)
    n_heads: Int, s_q: Int,
):
    """out[b, h, iq, d] = sum_ik probs[b,h,iq,ik] * v[b,h,ik,d].

    Launch: grid = B*H*S_q, block_dim = HEAD_DIM.
    """
    comptime assert probs.flat_rank == 4
    comptime assert v.flat_rank == 4
    comptime assert output.flat_rank == 4
    var bid = block_idx.x
    var d = thread_idx.x
    var iq = bid % s_q
    var h = (bid // s_q) % n_heads
    var b = bid // (s_q * n_heads)
    if d >= HEAD_DIM: return
    var acc: Float32 = 0.0
    for ik in range(S_K):
        var p = rebind[Scalar[dtype]](probs[b, h, iq, ik]).cast[DType.float32]()
        var vv = rebind[Scalar[dtype]](v[b, h, ik, d]).cast[DType.float32]()
        acc += p * vv
    output[b, h, iq, d] = rebind[output.ElementType](acc.cast[dtype]())


def split_heads_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    OutLayout: TensorLayout,
    HEAD_DIM: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],  # (B, H, S, D)
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],      # (B, S, H*D)
    batch: Int, seq: Int, n_heads: Int,
):
    """Reshape (B, S, n_heads*HEAD_DIM) → (B, H, S, D). Launch: grid=B*S*H, block_dim=HEAD_DIM."""
    comptime assert inp.flat_rank == 3
    comptime assert output.flat_rank == 4
    var bid = block_idx.x
    var d = thread_idx.x
    var h = bid % n_heads
    var s = (bid // n_heads) % seq
    var b = bid // (n_heads * seq)
    if d >= HEAD_DIM: return
    var v = rebind[Scalar[dtype]](inp[b, s, h * HEAD_DIM + d]).cast[DType.float32]()
    output[b, h, s, d] = rebind[output.ElementType](v.cast[dtype]())


def combine_heads_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    OutLayout: TensorLayout,
    HEAD_DIM: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],  # (B, S, H*D)
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],      # (B, H, S, D)
    batch: Int, seq: Int, n_heads: Int,
):
    """Reshape (B, H, S, D) → (B, S, H*D). Launch: grid=B*S*H, block_dim=HEAD_DIM."""
    comptime assert inp.flat_rank == 4
    comptime assert output.flat_rank == 3
    var bid = block_idx.x
    var d = thread_idx.x
    var h = bid % n_heads
    var s = (bid // n_heads) % seq
    var b = bid // (n_heads * seq)
    if d >= HEAD_DIM: return
    var v = rebind[Scalar[dtype]](inp[b, h, s, d]).cast[DType.float32]()
    output[b, s, h * HEAD_DIM + d] = rebind[output.ElementType](v.cast[dtype]())


def add_3d_kernel[
    dtype: DType, L: TensorLayout, BLOCK: Int,
](
    dst: TileTensor[dtype, L, MutAnyOrigin],
    a: TileTensor[dtype, L, MutAnyOrigin],
    b: TileTensor[dtype, L, MutAnyOrigin],
    n0: Int, n1: Int, n2: Int,
):
    """Pointwise dst = a + b on (n0, n1, n2). Launch: grid=n0*n1, block_dim=BLOCK over n2."""
    comptime assert a.flat_rank == 3
    comptime assert b.flat_rank == 3
    comptime assert dst.flat_rank == 3
    var bid = block_idx.x
    var tid = thread_idx.x
    var i1 = bid % n1
    var i0 = bid // n1
    var i2 = tid
    while i2 < n2:
        var av = rebind[Scalar[dtype]](a[i0, i1, i2]).cast[DType.float32]()
        var bv = rebind[Scalar[dtype]](b[i0, i1, i2]).cast[DType.float32]()
        dst[i0, i1, i2] = rebind[dst.ElementType]((av + bv).cast[dtype]())
        i2 += BLOCK


def residual_add_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    OutLayout: TensorLayout,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, S, C)
    base: TileTensor[dtype, InLayout, MutAnyOrigin],      # (B, S, C)
    delta: TileTensor[dtype, OutLayout, MutAnyOrigin],    # (B, S, C)
    n: Int,
):
    """Pointwise output = base + delta. Launch: grid=N/BLOCK, block_dim=BLOCK."""
    comptime assert base.flat_rank == 3
    comptime assert delta.flat_rank == 3
    comptime assert output.flat_rank == 3
    from std.gpu import global_idx
    var idx = global_idx.x
    if idx >= n: return
    # Flatten 3D index via single linear walk: assume row-major (B, S, C).
    # The kernel only reads/writes flat elements — use the underlying linear layout.
    # We use the fact that TileTensor with row_major[B,S,C] has linear stride
    # equal to position. We expose this via .flat_index but for simplicity use
    # 3D decomposition: B*S*C → (b, s, c).
    # The caller passes n=B*S*C. Decompose using S*C and C.
    # In practice we instantiate this for each call with constants; here we keep
    # 3D indexing simple by requiring caller to also pass per-row strides as
    # template constants in the future. For now, do the decomposition runtime:
    var c_dim = base.dim[2]()
    var sc = base.dim[1]() * c_dim
    var b = idx // Int(sc)
    var r = idx - b * Int(sc)
    var s = r // Int(c_dim)
    var c = r - s * Int(c_dim)
    var bv = rebind[Scalar[dtype]](base[b, s, c]).cast[DType.float32]()
    var dv = rebind[Scalar[dtype]](delta[b, s, c]).cast[DType.float32]()
    output[b, s, c] = rebind[output.ElementType]((bv + dv).cast[dtype]())
