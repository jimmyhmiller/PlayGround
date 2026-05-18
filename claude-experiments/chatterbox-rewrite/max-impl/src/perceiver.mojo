"""Perceiver resampler (32 learnable queries cross-attending then self-attending).

Built from `MHASelfAttention` (which uses `linalg.bmm` + `nn.softmax`).
For cross-attention we use a variant where Sq ≠ Sk.
"""
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr

from modules import Linear, linear_forward, LayerNorm, layer_norm_forward, residual_add
from attention import qk_scaled_and_masked, softmax_2d, av_matmul
from transformer_blocks import reshape_bsd_to_bhsd, reshape_bhsd_to_bsd
from std.math import sqrt


@fieldwise_init
struct PerceiverBlock(Copyable, Movable):
    """One AttentionBlock2 — Perceiver's pre/self attention block.

    LayerNorm both inputs (shared norm), then to_q/k/v Linear, attention,
    proj_out Linear, +x_q residual.
    """
    var norm: LayerNorm
    var to_q: Linear
    var to_k: Linear
    var to_v: Linear
    var proj_out: Linear
    var n_heads: Int
    var head_dim: Int


def perceiver_block_forward(
    mut ctx: DeviceContext,
    mut module: PerceiverBlock,
    mut x_q_buf: DeviceBuffer[DType.float32],   # (B, Sq, D)
    mut x_kv_buf: DeviceBuffer[DType.float32],  # (B, Sk, D)
    mut out_buf: DeviceBuffer[DType.float32],   # (B, Sq, D)
    mut zero_mask_buf: DeviceBuffer[DType.float32],  # (Sq, Sk) — all zeros
    b: Int, sq: Int, sk: Int,
) raises:
    var H = module.n_heads
    var Dh = module.head_dim
    var D = H * Dh

    var qn = ctx.enqueue_create_buffer[DType.float32](b * sq * D)
    var kn = ctx.enqueue_create_buffer[DType.float32](b * sk * D)
    layer_norm_forward(ctx, module.norm, x_q_buf,  qn, b * sq)
    layer_norm_forward(ctx, module.norm, x_kv_buf, kn, b * sk)

    var q_lin = ctx.enqueue_create_buffer[DType.float32](b * sq * D)
    var k_lin = ctx.enqueue_create_buffer[DType.float32](b * sk * D)
    var v_lin = ctx.enqueue_create_buffer[DType.float32](b * sk * D)
    linear_forward(ctx, module.to_q, qn, q_lin, b * sq)
    linear_forward(ctx, module.to_k, kn, k_lin, b * sk)
    linear_forward(ctx, module.to_v, kn, v_lin, b * sk)

    var q_perm = ctx.enqueue_create_buffer[DType.float32](b * H * sq * Dh)
    var k_perm = ctx.enqueue_create_buffer[DType.float32](b * H * sk * Dh)
    var v_perm = ctx.enqueue_create_buffer[DType.float32](b * H * sk * Dh)
    reshape_bsd_to_bhsd(ctx, q_lin, q_perm, b, sq, H, Dh)
    reshape_bsd_to_bhsd(ctx, k_lin, k_perm, b, sk, H, Dh)
    reshape_bsd_to_bhsd(ctx, v_lin, v_perm, b, sk, H, Dh)

    var scale: Float32 = 1.0 / sqrt(Float32(Dh))
    var logits = ctx.enqueue_create_buffer[DType.float32](b * H * sq * sk)
    var probs  = ctx.enqueue_create_buffer[DType.float32](b * H * sq * sk)
    var attn_perm = ctx.enqueue_create_buffer[DType.float32](b * H * sq * Dh)
    qk_scaled_and_masked(ctx, q_perm, k_perm, zero_mask_buf, logits,
                          b * H, sq, sk, Dh, scale, False)
    softmax_2d(ctx, logits, probs, b * H * sq, sk)
    av_matmul(ctx, probs, v_perm, attn_perm, b * H, sq, sk, Dh)
    var attn_flat = ctx.enqueue_create_buffer[DType.float32](b * sq * D)
    reshape_bhsd_to_bsd(ctx, attn_perm, attn_flat, b, H, sq, Dh)
    var proj = ctx.enqueue_create_buffer[DType.float32](b * sq * D)
    linear_forward(ctx, module.proj_out, attn_flat, proj, b * sq)

    # out = x_q + proj.
    ctx.enqueue_copy(out_buf, x_q_buf)
    residual_add(ctx, out_buf, proj, b * sq * D)


@fieldwise_init
struct Perceiver(Copyable, Movable):
    var pre_attention_query: DeviceBuffer[DType.float32]   # (1, n_q, D) learnable
    var block: PerceiverBlock
    var n_queries: Int
    var d_model: Int


def perceiver_forward(
    mut ctx: DeviceContext,
    mut model: Perceiver,
    mut h_buf: DeviceBuffer[DType.float32],   # (B, Tk, D)
    mut out_buf: DeviceBuffer[DType.float32], # (B, n_q, D)
    mut zero_mask_q: DeviceBuffer[DType.float32],   # (n_q, Tk) zeros
    mut zero_mask_qq: DeviceBuffer[DType.float32],  # (n_q, n_q) zeros
    b: Int, tk: Int,
) raises:
    """Two-stage: cross-attn(query → h), then self-attn(pre_att → pre_att)."""
    var nq = model.n_queries
    var D = model.d_model

    # Tile pre_attention_query (1, nq, D) → (B, nq, D).
    var q_buf = ctx.enqueue_create_buffer[DType.float32](b * nq * D)
    if b == 1:
        ctx.enqueue_copy(q_buf, model.pre_attention_query)
    else:
        # Generic tiling.
        var src = model.pre_attention_query.unsafe_ptr()
        var dst = q_buf.unsafe_ptr()
        from std.algorithm.functional import elementwise, IndexList

        @always_inline
        @parameter
        @__copy_capture(src, dst, nq, D)
        def tile_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
            var i = idx[0]
            dst[i] = src[i % (nq * D)]
        elementwise[tile_func, simd_width=1, target="gpu"](
            IndexList[1](b * nq * D), DeviceContextPtr(ctx),
        )

    # Cross-attn.
    var pre_att = ctx.enqueue_create_buffer[DType.float32](b * nq * D)
    perceiver_block_forward(ctx, model.block, q_buf, h_buf, pre_att,
                              zero_mask_q, b, nq, tk)
    # Self-attn (q == kv == pre_att — clone to avoid aliasing).
    var pre_att_clone = ctx.enqueue_create_buffer[DType.float32](b * nq * D)
    ctx.enqueue_copy(pre_att_clone, pre_att)
    perceiver_block_forward(ctx, model.block, pre_att, pre_att_clone, out_buf,
                              zero_mask_qq, b, nq, nq)
