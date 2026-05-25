"""Attention helpers built from `linalg.matmul` + `nn.softmax` + `elementwise`.

We split SDPA into three steps so we can reuse `linalg.matmul` for both qk and av:

  1. logits = (Q @ K.T) * scale + mask     # batched matmul + elementwise add
  2. probs  = softmax(logits)               # nn.softmax
  3. out    = probs @ V                     # batched matmul

For LLM decoder use we need: cross-attention (Sq ≠ Sk), self-attention (Sq = Sk).
For decode-step we need: q (B, H, 1, D), k_cache/v_cache (B, H, MAX_CTX, D)
with masking on cur_len. The `nn.kv_cache` package provides fused versions but
we start with the simple split form here.
"""
from std.math import sqrt
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList
from std.sys.info import simd_width_of
from layout import Idx, TileTensor, row_major

from nn.softmax import softmax as nn_softmax
from linalg.bmm import batched_matmul as nn_bmm


def qk_scaled_and_masked(
    mut ctx: DeviceContext,
    mut q_buf: DeviceBuffer[DType.float32],     # (B*H, Sq, D)
    mut k_buf: DeviceBuffer[DType.float32],     # (B*H, Sk, D)
    mut mask_buf: DeviceBuffer[DType.float32],  # (Sq, Sk) — bias mask (0 or -inf)
    mut out_buf: DeviceBuffer[DType.float32],   # (B*H, Sq, Sk) — logits
    bh: Int, sq: Int, sk: Int, d: Int,
    scale: Float32,
    has_mask: Bool,
) raises:
    """logits[b,h,iq,ik] = scale * sum_d Q[b,h,iq,d] * K[b,h,ik,d] + mask[iq,ik].

    Implemented as `linalg.bmm.batched_matmul` with transpose_b=True, then
    an elementwise scale+mask pass.
    """
    var dctx = DeviceContextPtr(ctx)
    var q_t = TileTensor(q_buf, row_major(Idx(bh), Idx(sq), Idx(d)))
    var k_t = TileTensor(k_buf, row_major(Idx(bh), Idx(sk), Idx(d)))
    var out_t = TileTensor(out_buf, row_major(Idx(bh), Idx(sq), Idx(sk)))
    nn_bmm[target="gpu", transpose_b=True](out_t, q_t, k_t, context=dctx)

    var out_ptr = out_buf.unsafe_ptr()
    var mask_ptr = mask_buf.unsafe_ptr()
    var total = bh * sq * sk

    @always_inline
    @parameter
    @__copy_capture(out_ptr, mask_ptr, sq, sk, scale, has_mask)
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var v = out_ptr[i] * scale
        if has_mask:
            var bh_idx = i // (sq * sk)
            var pos_in = i - bh_idx * sq * sk
            v += mask_ptr[pos_in]
        out_ptr[i] = v
    elementwise[func, simd_width=1, target="gpu"](
        IndexList[1](total), dctx,
    )


def softmax_2d(
    mut ctx: DeviceContext,
    mut logits_buf: DeviceBuffer[DType.float32],   # (B*H*Sq, Sk)
    mut probs_buf: DeviceBuffer[DType.float32],    # same shape
    rows: Int, cols: Int,
) raises:
    """Softmax over the last axis. Flatten BHSq into a single batch dim."""
    var in_ptr = logits_buf.unsafe_ptr()
    var out_ptr = probs_buf.unsafe_ptr()
    var shape = IndexList[2](rows, cols)

    @always_inline
    @parameter
    @__copy_capture(in_ptr, cols)
    def input_fn[width: Int, rank: Int](coords: IndexList[rank]) -> SIMD[DType.float32, width]:
        var c = rebind[IndexList[2]](coords)
        return in_ptr.load[width=width](c[0] * cols + c[1])

    var out_t = TileTensor(out_ptr, row_major(Idx(rows), Idx(cols)))
    var dctx = DeviceContextPtr(ctx)
    nn_softmax[
        DType.float32, simd_width_of[DType.float32](), 2,
        input_fn, target="gpu",
    ](shape, out_t, 1, dctx)


def av_matmul(
    mut ctx: DeviceContext,
    mut probs_buf: DeviceBuffer[DType.float32],  # (B*H, Sq, Sk)
    mut v_buf:     DeviceBuffer[DType.float32],  # (B*H, Sk, D)
    mut out_buf:   DeviceBuffer[DType.float32],  # (B*H, Sq, D)
    bh: Int, sq: Int, sk: Int, d: Int,
) raises:
    """out = probs @ V. `linalg.bmm.batched_matmul`."""
    var dctx = DeviceContextPtr(ctx)
    var p_t = TileTensor(probs_buf, row_major(Idx(bh), Idx(sq), Idx(sk)))
    var v_t = TileTensor(v_buf,     row_major(Idx(bh), Idx(sk), Idx(d)))
    var o_t = TileTensor(out_buf,   row_major(Idx(bh), Idx(sq), Idx(d)))
    nn_bmm[target="gpu", transpose_b=False](o_t, p_t, v_t, context=dctx)
