"""T3 Llama-style transformer block (RMSNorm + MHA + RMSNorm + SwiGLU).

Uses MAX abstractions throughout:
  - `nn.normalization.rms_norm` via `modules.rms_norm_forward`
  - `linalg.matmul` for Q/K/V/out projections
  - `nn.rope.apply_rope` for rotary position embedding (or our helper)
  - `linalg.bmm.batched_matmul` for SDPA matmuls
  - `nn.softmax.softmax` for attention probs
  - `elementwise[..., target="gpu"]` for elementwise ops (residual, scale, mask)
"""
from std.math import sqrt
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from modules import (
    Linear, linear_forward,
    RMSNorm, rms_norm_forward,
    residual_add,
)
from attention import qk_scaled_and_masked, softmax_2d, av_matmul
from transformer_blocks import (
    reshape_bsd_to_bhsd, reshape_bhsd_to_bsd, apply_rope_s3_style,
    LlamaMLP, llama_mlp_forward,
)


@fieldwise_init
struct T3Block(Copyable, Movable):
    """One Llama-30L transformer block."""
    var in_norm: RMSNorm
    var post_norm: RMSNorm
    var to_q: Linear
    var to_k: Linear
    var to_v: Linear
    var to_out: Linear
    var mlp: LlamaMLP
    var n_heads: Int
    var head_dim: Int


def t3_block_forward(
    mut ctx: DeviceContext,
    mut module: T3Block,
    mut x_buf: DeviceBuffer[DType.float32],   # (B, S, D) — also output (in-place residual)
    mut cos_buf: DeviceBuffer[DType.float32], # (S, HALF)
    mut sin_buf: DeviceBuffer[DType.float32],
    mut mask_buf: DeviceBuffer[DType.float32], # (S, S) bias
    b: Int, s: Int,
    has_mask: Bool,
) raises:
    """Forward pass for one T3 block.

    x_buf is modified in-place via two residual adds (pre-attn + pre-mlp).
    """
    var H = module.n_heads
    var Dh = module.head_dim
    var D = H * Dh

    # ---- attention ----
    var x_norm = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    rms_norm_forward(ctx, module.in_norm, x_buf, x_norm, b * s)

    var q_lin = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    var k_lin = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    var v_lin = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    linear_forward(ctx, module.to_q, x_norm, q_lin, b * s)
    linear_forward(ctx, module.to_k, x_norm, k_lin, b * s)
    linear_forward(ctx, module.to_v, x_norm, v_lin, b * s)

    # Reshape (B, S, H, Dh).
    var q_4d = ctx.enqueue_create_buffer[DType.float32](b * s * H * Dh)
    var k_4d = ctx.enqueue_create_buffer[DType.float32](b * s * H * Dh)
    ctx.enqueue_copy(q_4d, q_lin)
    ctx.enqueue_copy(k_4d, k_lin)
    # Apply RoPE on the (B, S, H, Dh) layout.
    var q_rope = ctx.enqueue_create_buffer[DType.float32](b * s * H * Dh)
    var k_rope = ctx.enqueue_create_buffer[DType.float32](b * s * H * Dh)
    apply_rope_s3_style(ctx, q_4d, q_rope, cos_buf, sin_buf, b, s, H, Dh)
    apply_rope_s3_style(ctx, k_4d, k_rope, cos_buf, sin_buf, b, s, H, Dh)

    # Permute to (B, H, S, Dh) for SDPA.
    var q_perm = ctx.enqueue_create_buffer[DType.float32](b * H * s * Dh)
    var k_perm = ctx.enqueue_create_buffer[DType.float32](b * H * s * Dh)
    var v_perm = ctx.enqueue_create_buffer[DType.float32](b * H * s * Dh)
    reshape_bsd_to_bhsd(ctx, q_rope, q_perm, b, s, H, Dh)
    reshape_bsd_to_bhsd(ctx, k_rope, k_perm, b, s, H, Dh)
    reshape_bsd_to_bhsd(ctx, v_lin,  v_perm, b, s, H, Dh)

    var scale: Float32 = 1.0 / sqrt(Float32(Dh))
    var logits = ctx.enqueue_create_buffer[DType.float32](b * H * s * s)
    var probs  = ctx.enqueue_create_buffer[DType.float32](b * H * s * s)
    var attn_perm = ctx.enqueue_create_buffer[DType.float32](b * H * s * Dh)
    qk_scaled_and_masked(ctx, q_perm, k_perm, mask_buf, logits,
                          b * H, s, s, Dh, scale, has_mask)
    softmax_2d(ctx, logits, probs, b * H * s, s)
    av_matmul(ctx, probs, v_perm, attn_perm, b * H, s, s, Dh)
    var attn_flat = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    reshape_bhsd_to_bsd(ctx, attn_perm, attn_flat, b, H, s, Dh)
    var attn_out = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    linear_forward(ctx, module.to_out, attn_flat, attn_out, b * s)

    # Residual: x += attn_out.
    residual_add(ctx, x_buf, attn_out, b * s * D)

    # ---- MLP ----
    var x_norm2 = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    rms_norm_forward(ctx, module.post_norm, x_buf, x_norm2, b * s)
    var mlp_out = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    llama_mlp_forward(ctx, module.mlp, x_norm2, mlp_out, b * s)
    residual_add(ctx, x_buf, mlp_out, b * s * D)
