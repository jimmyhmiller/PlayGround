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
