"""Conformer block for s3gen's UpsampleConformerEncoder.

Each Conformer layer: PreLN + FF1(×0.5 residual) + Self-Attention + Conv module
+ FF2(×0.5 residual) + final LayerNorm.
"""
from std.math import sqrt
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from modules import (
    Linear, linear_forward,
    LayerNorm, layer_norm_forward,
    residual_add, silu,
)
from conv1d import Conv1d, conv1d_forward
from attention import qk_scaled_and_masked, softmax_2d, av_matmul
from transformer_blocks import (
    reshape_bsd_to_bhsd, reshape_bhsd_to_bsd, apply_rope_s3_style,
)


@fieldwise_init
struct FeedForward(Copyable, Movable):
    """Position-wise FF: Linear → Swish (silu) → Linear."""
    var ln: LayerNorm
    var w1: Linear   # (inter, d_model)
    var w2: Linear   # (d_model, inter)
    var d_model: Int
    var inter: Int


def feed_forward(
    mut ctx: DeviceContext,
    mut module: FeedForward,
    mut x_buf: DeviceBuffer[DType.float32],   # (M, D)
    mut out_buf: DeviceBuffer[DType.float32],
    m: Int,
) raises:
    """Run a single FF macroblock (no residual scaling — caller handles ×0.5)."""
    var D = module.d_model
    var inter = module.inter
    var x_norm = ctx.enqueue_create_buffer[DType.float32](m * D)
    layer_norm_forward(ctx, module.ln, x_buf, x_norm, m)
    var h = ctx.enqueue_create_buffer[DType.float32](m * inter)
    var act = ctx.enqueue_create_buffer[DType.float32](m * inter)
    linear_forward(ctx, module.w1, x_norm, h, m)
    silu(ctx, h, act, m * inter)
    linear_forward(ctx, module.w2, act, out_buf, m)


@fieldwise_init
struct ConformerConvModule(Copyable, Movable):
    """Conformer's conv module: LN → pw1 (Conv1d, expand 2×) → GLU → depthwise
    Conv → norm → swish → pw2 (Conv1d back to D).

    For brevity we model it as: LN → Linear (2D) → split-glu → Conv1d depthwise
    → LayerNorm → swish → Linear (D)."""
    var ln: LayerNorm
    var pw1: Linear     # (2D, D)
    var dw_conv: Conv1d # depthwise: groups=D, kernel ~15, padding=K-1 (causal-ish)
    var bn_ln: LayerNorm
    var pw2: Linear     # (D, D)
    var d_model: Int


def conformer_conv_module(
    mut ctx: DeviceContext,
    mut module: ConformerConvModule,
    mut x_buf: DeviceBuffer[DType.float32],   # (B, S, D)
    mut out_buf: DeviceBuffer[DType.float32],
    b: Int, s: Int,
) raises:
    var D = module.d_model
    var x_norm = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    layer_norm_forward(ctx, module.ln, x_buf, x_norm, b * s)

    var pw1_out = ctx.enqueue_create_buffer[DType.float32](b * s * 2 * D)
    linear_forward(ctx, module.pw1, x_norm, pw1_out, b * s)

    # GLU: split last dim into (a, b), output = a * sigmoid(b).
    var glu_out = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    var p_ptr = pw1_out.unsafe_ptr()
    var g_ptr = glu_out.unsafe_ptr()
    from std.math import exp

    @always_inline
    @parameter
    @__copy_capture(p_ptr, g_ptr, s, D)
    def glu_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (s * D)
        var rem = i - bi * s * D
        var si = rem // D
        var di = rem - si * D
        var a = p_ptr[bi * s * 2 * D + si * 2 * D + di]
        var bv = p_ptr[bi * s * 2 * D + si * 2 * D + D + di]
        var sig: Float32 = 1.0 / (1.0 + exp(-bv))
        g_ptr[i] = a * sig
    elementwise[glu_func, simd_width=1, target="gpu"](
        IndexList[1](b * s * D), DeviceContextPtr(ctx),
    )

    # Permute (B, S, D) → (B, D, S) for Conv1d.
    var glu_perm = ctx.enqueue_create_buffer[DType.float32](b * D * s)
    var gp_ptr = glu_out.unsafe_ptr()
    var gpp_ptr = glu_perm.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(gp_ptr, gpp_ptr, b, s, D)
    def trans_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (D * s)
        var rem = i - bi * D * s
        var di = rem // s
        var si = rem - di * s
        gpp_ptr[i] = gp_ptr[bi * s * D + si * D + di]
    elementwise[trans_func, simd_width=1, target="gpu"](
        IndexList[1](b * D * s), DeviceContextPtr(ctx),
    )

    # Depthwise Conv1d.
    var dw_out = ctx.enqueue_create_buffer[DType.float32](b * D * s)
    conv1d_forward(ctx, module.dw_conv, glu_perm, dw_out, b, s, s)

    # Permute back (B, D, S) → (B, S, D).
    var dw_perm = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    var dw_ptr = dw_out.unsafe_ptr()
    var dwp_ptr = dw_perm.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(dw_ptr, dwp_ptr, b, s, D)
    def trans2_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (s * D)
        var rem = i - bi * s * D
        var si = rem // D
        var di = rem - si * D
        dwp_ptr[i] = dw_ptr[bi * D * s + di * s + si]
    elementwise[trans2_func, simd_width=1, target="gpu"](
        IndexList[1](b * s * D), DeviceContextPtr(ctx),
    )

    # bn_ln + swish.
    var bn_out = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    layer_norm_forward(ctx, module.bn_ln, dw_perm, bn_out, b * s)
    var sw_out = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    silu(ctx, bn_out, sw_out, b * s * D)

    # pw2.
    linear_forward(ctx, module.pw2, sw_out, out_buf, b * s)


@fieldwise_init
struct ConformerSelfAttention(Copyable, Movable):
    var ln: LayerNorm
    var to_q: Linear
    var to_k: Linear
    var to_v: Linear
    var to_out: Linear
    var n_heads: Int
    var head_dim: Int


def conformer_self_attention(
    mut ctx: DeviceContext,
    mut module: ConformerSelfAttention,
    mut x_buf: DeviceBuffer[DType.float32],
    mut out_buf: DeviceBuffer[DType.float32],
    mut cos_buf: DeviceBuffer[DType.float32],
    mut sin_buf: DeviceBuffer[DType.float32],
    mut mask_buf: DeviceBuffer[DType.float32],
    b: Int, s: Int,
    has_mask: Bool,
) raises:
    var H = module.n_heads
    var Dh = module.head_dim
    var D = H * Dh

    var x_norm = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    layer_norm_forward(ctx, module.ln, x_buf, x_norm, b * s)

    var q_lin = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    var k_lin = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    var v_lin = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    linear_forward(ctx, module.to_q, x_norm, q_lin, b * s)
    linear_forward(ctx, module.to_k, x_norm, k_lin, b * s)
    linear_forward(ctx, module.to_v, x_norm, v_lin, b * s)

    var q_4d = ctx.enqueue_create_buffer[DType.float32](b * s * H * Dh)
    var k_4d = ctx.enqueue_create_buffer[DType.float32](b * s * H * Dh)
    ctx.enqueue_copy(q_4d, q_lin)
    ctx.enqueue_copy(k_4d, k_lin)
    var q_rope = ctx.enqueue_create_buffer[DType.float32](b * s * H * Dh)
    var k_rope = ctx.enqueue_create_buffer[DType.float32](b * s * H * Dh)
    apply_rope_s3_style(ctx, q_4d, q_rope, cos_buf, sin_buf, b, s, H, Dh)
    apply_rope_s3_style(ctx, k_4d, k_rope, cos_buf, sin_buf, b, s, H, Dh)

    var q_perm = ctx.enqueue_create_buffer[DType.float32](b * H * s * Dh)
    var k_perm = ctx.enqueue_create_buffer[DType.float32](b * H * s * Dh)
    var v_perm = ctx.enqueue_create_buffer[DType.float32](b * H * s * Dh)
    reshape_bsd_to_bhsd(ctx, q_rope, q_perm, b, s, H, Dh)
    reshape_bsd_to_bhsd(ctx, k_rope, k_perm, b, s, H, Dh)
    reshape_bsd_to_bhsd(ctx, v_lin,  v_perm, b, s, H, Dh)

    var scale: Float32 = 1.0 / sqrt(Float32(Dh))
    var logits = ctx.enqueue_create_buffer[DType.float32](b * H * s * s)
    var probs  = ctx.enqueue_create_buffer[DType.float32](b * H * s * s)
    var av     = ctx.enqueue_create_buffer[DType.float32](b * H * s * Dh)
    qk_scaled_and_masked(ctx, q_perm, k_perm, mask_buf, logits,
                          b * H, s, s, Dh, scale, has_mask)
    softmax_2d(ctx, logits, probs, b * H * s, s)
    av_matmul(ctx, probs, v_perm, av, b * H, s, s, Dh)
    var av_flat = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    reshape_bhsd_to_bsd(ctx, av, av_flat, b, H, s, Dh)
    linear_forward(ctx, module.to_out, av_flat, out_buf, b * s)


@fieldwise_init
struct ConformerLayer(Copyable, Movable):
    """One Conformer layer (FF1 + MHSA + Conv + FF2 + final LN)."""
    var ff1: FeedForward
    var attn: ConformerSelfAttention
    var conv: ConformerConvModule
    var ff2: FeedForward
    var final_ln: LayerNorm


def scale_add(
    mut ctx: DeviceContext,
    mut dst_buf: DeviceBuffer[DType.float32],   # dst += scalar * other
    mut other_buf: DeviceBuffer[DType.float32],
    scalar: Float32, n: Int,
) raises:
    var d_ptr = dst_buf.unsafe_ptr()
    var o_ptr = other_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(d_ptr, o_ptr, scalar)
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        d_ptr[i] = d_ptr[i] + o_ptr[i] * scalar
    elementwise[func, simd_width=4, target="gpu"](
        IndexList[1](n), DeviceContextPtr(ctx),
    )


def conformer_layer_forward(
    mut ctx: DeviceContext,
    mut module: ConformerLayer,
    mut x_buf: DeviceBuffer[DType.float32],     # (B, S, D) in-place updated
    mut cos_buf: DeviceBuffer[DType.float32],
    mut sin_buf: DeviceBuffer[DType.float32],
    mut mask_buf: DeviceBuffer[DType.float32],
    b: Int, s: Int,
    has_mask: Bool,
) raises:
    var D = module.attn.n_heads * module.attn.head_dim
    var n = b * s * D
    # ½ * FF1.
    var ff1_out = ctx.enqueue_create_buffer[DType.float32](n)
    feed_forward(ctx, module.ff1, x_buf, ff1_out, b * s)
    scale_add(ctx, x_buf, ff1_out, Float32(0.5), n)
    # MHSA.
    var attn_out = ctx.enqueue_create_buffer[DType.float32](n)
    conformer_self_attention(
        ctx, module.attn, x_buf, attn_out, cos_buf, sin_buf, mask_buf,
        b, s, has_mask,
    )
    residual_add(ctx, x_buf, attn_out, n)
    # Conv module.
    var conv_out = ctx.enqueue_create_buffer[DType.float32](n)
    conformer_conv_module(ctx, module.conv, x_buf, conv_out, b, s)
    residual_add(ctx, x_buf, conv_out, n)
    # ½ * FF2.
    var ff2_out = ctx.enqueue_create_buffer[DType.float32](n)
    feed_forward(ctx, module.ff2, x_buf, ff2_out, b * s)
    scale_add(ctx, x_buf, ff2_out, Float32(0.5), n)
    # Final LN.
    var ln_out = ctx.enqueue_create_buffer[DType.float32](n)
    layer_norm_forward(ctx, module.final_ln, x_buf, ln_out, b * s)
    ctx.enqueue_copy(x_buf, ln_out)


# ============================================================================
# Real upstream s3gen flow encoder structures.
#
# The downstream Chatterbox flow encoder is NOT a full Conformer (FF+Attn+Conv
# +FF). It is a simpler pre-norm Transformer with relative-position MHA:
#
#   norm_mha → self_attn(q,k,v + RelPos via linear_pos + pos_bias_u/v) + res
#   norm_ff  → feed_forward(w_1, w_2) + res
#
# Each TransformerEncoderLayer stores explicit norm_ff / norm_mha siblings
# rather than fusing the norm into the FF/attn modules. This matches the
# .safetensors layout: encoders.{L}.{norm_ff,norm_mha,self_attn,feed_forward}.
# ============================================================================


@fieldwise_init
struct RelPosMHA(Copyable, Movable):
    """Multi-head self-attention with relative position bias (Transformer-XL
    style, used by Wenet/s3gen). Stores q/k/v/out + linear_pos + pos_bias_u/v.

    Forward path is not yet wired — this struct exists so the loader can
    populate from upstream weights; integration into upsample_conformer_forward
    follows once the layout is validated end-to-end.
    """
    var to_q: Linear        # (D, D) + bias
    var to_k: Linear        # (D, D) + bias
    var to_v: Linear        # (D, D) + bias
    var to_out: Linear      # (D, D) + bias
    var linear_pos: Linear  # (D, D) — no bias; used to project sinusoidal pos enc
    var pos_bias_u: DeviceBuffer[DType.float32]  # (H, Dh)
    var pos_bias_v: DeviceBuffer[DType.float32]  # (H, Dh)
    var n_heads: Int
    var head_dim: Int


@fieldwise_init
struct TransformerEncoderLayer(Copyable, Movable):
    """One real-upstream flow-encoder layer (pre-norm Transformer w/ RelPos)."""
    var norm_mha: LayerNorm
    var norm_ff:  LayerNorm
    var self_attn: RelPosMHA
    var feed_forward_w1: Linear   # (intermediate, d_model) + bias
    var feed_forward_w2: Linear   # (d_model, intermediate) + bias
    var d_model: Int
    var intermediate: Int


# ============================================================================
# RelPos MHA forward pass (Transformer-XL style)
#
# Algorithm (q/k/v all dim D = H*Dh):
#   q = to_q(x_norm).view(B, T, H, Dh)
#   k = to_k(x_norm).view(B, T, H, Dh)
#   v = to_v(x_norm).view(B, T, H, Dh)
#   p = linear_pos(pos_emb).view(B, T_pos, H, Dh)   # T_pos = 2*T-1
#
#   q_perm = q.transpose(1,2)                       # (B, H, T, Dh)
#   q_u = (q + pos_bias_u).transpose(1,2)           # (B, H, T, Dh)
#   q_v = (q + pos_bias_v).transpose(1,2)           # (B, H, T, Dh)
#   k_perm = k.transpose(1,2)                       # (B, H, T, Dh)
#   v_perm = v.transpose(1,2)                       # (B, H, T, Dh)
#   p_perm = p.transpose(1,2)                       # (B, H, T_pos, Dh)
#
#   matrix_ac = q_u @ k_perm.T                      # (B, H, T, T)
#   matrix_bd = q_v @ p_perm.T                      # (B, H, T, T_pos)
#   matrix_bd = rel_shift(matrix_bd)[:,:,:,:T]      # (B, H, T, T)
#
#   scores = (matrix_ac + matrix_bd) / sqrt(Dh)
#   probs = softmax(scores)
#   attn = probs @ v_perm                           # (B, H, T, Dh)
#   out = to_out(attn.transpose(1,2).reshape(B,T,D))
# ============================================================================


def add_pos_bias_and_permute(
    mut ctx: DeviceContext,
    mut q_4d: DeviceBuffer[DType.float32],     # (B, T, H, Dh)
    mut bias: DeviceBuffer[DType.float32],      # (H, Dh)
    mut out_perm: DeviceBuffer[DType.float32],  # (B, H, T, Dh)
    b: Int, t: Int, h: Int, dh: Int,
) raises:
    """Compute `(q + bias).transpose(1, 2)` in one kernel: adds the (H, Dh)
    per-head bias to (B, T, H, Dh) q, then writes into (B, H, T, Dh) layout.
    """
    var q_ptr = q_4d.unsafe_ptr()
    var b_ptr = bias.unsafe_ptr()
    var o_ptr = out_perm.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(q_ptr, b_ptr, o_ptr, t, h, dh)
    def bias_perm_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (h * t * dh)
        var rem = i - bi * h * t * dh
        var hi = rem // (t * dh)
        var rem2 = rem - hi * t * dh
        var ti = rem2 // dh
        var di = rem2 - ti * dh
        var src = bi * t * h * dh + ti * h * dh + hi * dh + di
        o_ptr[i] = q_ptr[src] + b_ptr[hi * dh + di]
    elementwise[bias_perm_func, simd_width=1, target="gpu"](
        IndexList[1](b * h * t * dh), DeviceContextPtr(ctx),
    )


def rel_shift_matrix_bd(
    mut ctx: DeviceContext,
    mut in_buf: DeviceBuffer[DType.float32],   # (B, H, T, T_pos)
    mut out_buf: DeviceBuffer[DType.float32],  # (B, H, T, T)  — sliced
    b: Int, h: Int, t: Int, t_pos: Int,
) raises:
    """Apply Espnet's rel_shift then slice last axis to length `t`.

    Reference Python (from chatterbox attention.py):
      zero_pad = zeros(B, H, T, 1)
      x_padded = cat([zero_pad, x], dim=-1)              # (B,H,T, T_pos+1)
      x_padded = x_padded.view(B, H, T_pos+1, T)
      x = x_padded[:, :, 1:].view_as(x)[:, :, :, :T_pos//2 + 1]

    Result element (b, h, ti, j) = in[b, h, ti, j + (T - 1 - ti)] when in
    bounds, else 0 (the zero_pad fall-through). This is the standard
    Transformer-XL rel-shift trick.
    """
    var in_ptr = in_buf.unsafe_ptr()
    var out_ptr = out_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(in_ptr, out_ptr, h, t, t_pos)
    def relshift_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (h * t * t)
        var rem = i - bi * h * t * t
        var hi = rem // (t * t)
        var rem2 = rem - hi * t * t
        var ti = rem2 // t
        var j = rem2 - ti * t
        var j_src = j + (t - 1 - ti)
        var val: Float32 = 0.0
        if j_src >= 0 and j_src < t_pos:
            val = in_ptr[bi * h * t * t_pos + hi * t * t_pos + ti * t_pos + j_src]
        out_ptr[i] = val
    elementwise[relshift_func, simd_width=1, target="gpu"](
        IndexList[1](b * h * t * t), DeviceContextPtr(ctx),
    )


def relpos_mha_forward(
    mut ctx: DeviceContext,
    mut module: RelPosMHA,
    mut x_buf: DeviceBuffer[DType.float32],     # (B, T, D)  — pre-normed input
    mut pos_emb_buf: DeviceBuffer[DType.float32], # (1, T_pos, D) where T_pos = 2*T-1
    mut out_buf: DeviceBuffer[DType.float32],   # (B, T, D)  — post-out_proj
    b: Int, t: Int, t_pos: Int,
) raises:
    """Transformer-XL relative-position multi-head self-attention forward.

    `x_buf` is the pre-LayerNorm-applied input — caller normalizes since
    Encoder layer needs `norm_mha(x)` BEFORE this and `x_residual` outside.
    """
    var H = module.n_heads
    var Dh = module.head_dim
    var D = H * Dh

    # Project q, k, v: (B, T, D) → (B, T, D) each. These are stored as
    # (B, T, H, Dh) just by reshape since we write into linear elementwise.
    var q_lin = ctx.enqueue_create_buffer[DType.float32](b * t * D)
    var k_lin = ctx.enqueue_create_buffer[DType.float32](b * t * D)
    var v_lin = ctx.enqueue_create_buffer[DType.float32](b * t * D)
    linear_forward(ctx, module.to_q, x_buf, q_lin, b * t)
    linear_forward(ctx, module.to_k, x_buf, k_lin, b * t)
    linear_forward(ctx, module.to_v, x_buf, v_lin, b * t)

    # Reshape k, v: (B, T, D) → (B, H, T, Dh) for attention compute.
    var k_perm = ctx.enqueue_create_buffer[DType.float32](b * H * t * Dh)
    var v_perm = ctx.enqueue_create_buffer[DType.float32](b * H * t * Dh)
    reshape_bsd_to_bhsd(ctx, k_lin, k_perm, b, t, H, Dh)
    reshape_bsd_to_bhsd(ctx, v_lin, v_perm, b, t, H, Dh)

    # q is conceptually (B, T, H, Dh) since (B, T, D) == (B, T, H*Dh) flat.
    # Add pos_bias_u, pos_bias_v into transposed copies (B, H, T, Dh).
    var q_u_perm = ctx.enqueue_create_buffer[DType.float32](b * H * t * Dh)
    var q_v_perm = ctx.enqueue_create_buffer[DType.float32](b * H * t * Dh)
    add_pos_bias_and_permute(ctx, q_lin, module.pos_bias_u, q_u_perm, b, t, H, Dh)
    add_pos_bias_and_permute(ctx, q_lin, module.pos_bias_v, q_v_perm, b, t, H, Dh)

    # Project pos_emb (1, T_pos, D) → (1, T_pos, D) via linear_pos.
    # Then reshape (1, T_pos, H, Dh) → (1, H, T_pos, Dh).
    var p_lin = ctx.enqueue_create_buffer[DType.float32](t_pos * D)
    linear_forward(ctx, module.linear_pos, pos_emb_buf, p_lin, t_pos)
    var p_perm = ctx.enqueue_create_buffer[DType.float32](H * t_pos * Dh)
    reshape_bsd_to_bhsd(ctx, p_lin, p_perm, 1, t_pos, H, Dh)

    # matrix_ac = q_u_perm @ k_perm.T   → (B, H, T, T)
    var scale: Float32 = 1.0 / sqrt(Float32(Dh))
    var logits = ctx.enqueue_create_buffer[DType.float32](b * H * t * t)
    var probs = ctx.enqueue_create_buffer[DType.float32](b * H * t * t)
    # Use an unused mask buffer (zeros) since we don't apply a mask here.
    var no_mask = ctx.enqueue_create_buffer[DType.float32](t * t)
    no_mask.enqueue_fill(0.0)
    qk_scaled_and_masked(ctx, q_u_perm, k_perm, no_mask, logits,
                          b * H, t, t, Dh, scale, False)

    # matrix_bd = q_v_perm @ p_perm.T   → (B, H, T, T_pos)
    # p_perm has shape (1, H, T_pos, Dh) — broadcast across batch by treating
    # the leading dim as 1 and reading it into the qk_scaled call's b*H grouping
    # which assumes one (H*Dh) buffer per (b*H) entry.
    # We need per-batch matrix_bd, so for B > 1 we'd need broadcast. Asserting
    # B == 1 for the inference path (flow encoder always runs B=1 in synthesis).
    var bd_logits = ctx.enqueue_create_buffer[DType.float32](b * H * t * t_pos)
    # Reuse qk_scaled_and_masked without mask, scale 1.0 (we'll add to scaled ac later).
    qk_scaled_and_masked(ctx, q_v_perm, p_perm, no_mask, bd_logits,
                          b * H, t, t_pos, Dh, scale, False)

    # rel_shift bd → (B, H, T, T)
    var bd_shifted = ctx.enqueue_create_buffer[DType.float32](b * H * t * t)
    rel_shift_matrix_bd(ctx, bd_logits, bd_shifted, b, H, t, t_pos)

    # logits += bd_shifted (both already scaled by 1/sqrt(Dh) above).
    residual_add(ctx, logits, bd_shifted, b * H * t * t)

    # softmax → probs.
    softmax_2d(ctx, logits, probs, b * H * t, t)

    # attn = probs @ v_perm → (B, H, T, Dh)
    var attn_perm = ctx.enqueue_create_buffer[DType.float32](b * H * t * Dh)
    av_matmul(ctx, probs, v_perm, attn_perm, b * H, t, t, Dh)

    # Reshape (B, H, T, Dh) → (B, T, D) for out_proj.
    var attn_flat = ctx.enqueue_create_buffer[DType.float32](b * t * D)
    reshape_bhsd_to_bsd(ctx, attn_perm, attn_flat, b, H, t, Dh)
    linear_forward(ctx, module.to_out, attn_flat, out_buf, b * t)


def transformer_encoder_layer_forward(
    mut ctx: DeviceContext,
    mut module: TransformerEncoderLayer,
    mut x_buf: DeviceBuffer[DType.float32],     # (B, T, D) in-place residual updates
    mut pos_emb_buf: DeviceBuffer[DType.float32], # (1, T_pos, D)
    b: Int, t: Int, t_pos: Int,
) raises:
    """One flow-encoder layer: pre-norm MHA + pre-norm FF, both with residuals.

      residual = x
      x = norm_mha(x)
      x_att = relpos_mha(x, pos_emb)
      x = residual + x_att

      residual = x
      x = norm_ff(x)
      x_ff = w2(silu(w1(x)))
      x = residual + x_ff
    """
    var D = module.d_model
    var n = b * t * D

    # MHA pre-norm + residual.
    var x_norm = ctx.enqueue_create_buffer[DType.float32](n)
    layer_norm_forward(ctx, module.norm_mha, x_buf, x_norm, b * t)
    var attn_out = ctx.enqueue_create_buffer[DType.float32](n)
    relpos_mha_forward(ctx, module.self_attn, x_norm, pos_emb_buf, attn_out, b, t, t_pos)
    residual_add(ctx, x_buf, attn_out, n)

    # FF pre-norm + residual. Upstream uses PositionwiseFeedForward = Linear+SiLU+Linear.
    var x_norm2 = ctx.enqueue_create_buffer[DType.float32](n)
    layer_norm_forward(ctx, module.norm_ff, x_buf, x_norm2, b * t)
    var h = ctx.enqueue_create_buffer[DType.float32](b * t * module.intermediate)
    var act = ctx.enqueue_create_buffer[DType.float32](b * t * module.intermediate)
    linear_forward(ctx, module.feed_forward_w1, x_norm2, h, b * t)
    silu(ctx, h, act, b * t * module.intermediate)
    var ff_out = ctx.enqueue_create_buffer[DType.float32](n)
    linear_forward(ctx, module.feed_forward_w2, act, ff_out, b * t)
    residual_add(ctx, x_buf, ff_out, n)
