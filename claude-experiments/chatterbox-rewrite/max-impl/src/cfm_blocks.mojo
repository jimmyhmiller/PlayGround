"""CFM U-Net sub-blocks: ResnetBlock1D + BasicTransformerBlock + Downsample/Upsample.

All ops route through MAX abstractions (`linalg.matmul`, `Conv1d`,
`elementwise[..., target="gpu"]`, `nn.softmax`, `nn.normalization`).
"""
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from modules import Linear, linear_forward, LayerNorm, layer_norm_forward, silu, residual_add
from conv1d import Conv1d, conv1d_forward
from attention import qk_scaled_and_masked, softmax_2d, av_matmul
from transformer_blocks import reshape_bsd_to_bhsd, reshape_bhsd_to_bsd


@fieldwise_init
struct CausalResnetBlock1D(Copyable, Movable):
    """Conv1d + group-norm-ish (we use LayerNorm) + silu, with time emb addition."""
    var ln1: LayerNorm
    var conv1: Conv1d        # (out_c, in_c, k=3) padding=1
    var time_emb_proj: Linear
    var ln2: LayerNorm
    var conv2: Conv1d
    var skip_conv: Conv1d    # 1×1 conv for residual shape match (or identity if c_in==c_out)
    var has_skip: Bool
    var in_c: Int
    var out_c: Int


def transpose_bct_to_btc(
    mut ctx: DeviceContext,
    mut x_buf: DeviceBuffer[DType.float32],     # (B, C, T)
    mut out_buf: DeviceBuffer[DType.float32],   # (B, T, C)
    b: Int, c: Int, t: Int,
) raises:
    var i_ptr = x_buf.unsafe_ptr()
    var o_ptr = out_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(i_ptr, o_ptr, b, c, t)
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (t * c)
        var rem = i - bi * t * c
        var ti = rem // c
        var ci = rem - ti * c
        o_ptr[i] = i_ptr[bi * c * t + ci * t + ti]
    elementwise[func, simd_width=1, target="gpu"](
        IndexList[1](b * c * t), DeviceContextPtr(ctx),
    )


def transpose_btc_to_bct(
    mut ctx: DeviceContext,
    mut x_buf: DeviceBuffer[DType.float32],     # (B, T, C)
    mut out_buf: DeviceBuffer[DType.float32],   # (B, C, T)
    b: Int, t: Int, c: Int,
) raises:
    var i_ptr = x_buf.unsafe_ptr()
    var o_ptr = out_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(i_ptr, o_ptr, b, c, t)
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (c * t)
        var rem = i - bi * c * t
        var ci = rem // t
        var ti = rem - ci * t
        o_ptr[i] = i_ptr[bi * t * c + ti * c + ci]
    elementwise[func, simd_width=1, target="gpu"](
        IndexList[1](b * c * t), DeviceContextPtr(ctx),
    )


def add_broadcast_time_emb(
    mut ctx: DeviceContext,
    mut x_buf: DeviceBuffer[DType.float32],    # (B, C, T)
    mut t_emb_buf: DeviceBuffer[DType.float32], # (B, C) — broadcasts over T
    b: Int, c: Int, t: Int,
) raises:
    """x[b, c, t] += t_emb[b, c]."""
    var x_ptr = x_buf.unsafe_ptr()
    var t_ptr = t_emb_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(x_ptr, t_ptr, c, t)
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (c * t)
        var rem = i - bi * c * t
        var ci = rem // t
        x_ptr[i] = x_ptr[i] + t_ptr[bi * c + ci]
    elementwise[func, simd_width=1, target="gpu"](
        IndexList[1](b * c * t), DeviceContextPtr(ctx),
    )


def causal_resnet_block_1d(
    mut ctx: DeviceContext,
    mut module: CausalResnetBlock1D,
    mut x_buf: DeviceBuffer[DType.float32],     # (B, in_c, T)
    mut t_emb_buf: DeviceBuffer[DType.float32], # (B, out_c)
    mut out_buf: DeviceBuffer[DType.float32],   # (B, out_c, T)
    b: Int, t: Int,
) raises:
    """Forward: residual = skip_conv(x); h = conv1(silu(ln1(x))); h += t_emb;
                h = conv2(silu(ln2(h))); out = h + residual."""
    var in_c = module.in_c
    var out_c = module.out_c

    # Compute residual.
    var residual = ctx.enqueue_create_buffer[DType.float32](b * out_c * t)
    if module.has_skip:
        conv1d_forward(ctx, module.skip_conv, x_buf, residual, b, t, t)
    else:
        ctx.enqueue_copy(residual, x_buf)

    # ln1(x) — operates over channel dim; transpose, LN, transpose back.
    var x_btc = ctx.enqueue_create_buffer[DType.float32](b * t * in_c)
    transpose_bct_to_btc(ctx, x_buf, x_btc, b, in_c, t)
    var x_ln = ctx.enqueue_create_buffer[DType.float32](b * t * in_c)
    layer_norm_forward(ctx, module.ln1, x_btc, x_ln, b * t)
    var x_bct = ctx.enqueue_create_buffer[DType.float32](b * in_c * t)
    transpose_btc_to_bct(ctx, x_ln, x_bct, b, t, in_c)

    # silu then conv1.
    var x_act = ctx.enqueue_create_buffer[DType.float32](b * in_c * t)
    silu(ctx, x_bct, x_act, b * in_c * t)
    var h = ctx.enqueue_create_buffer[DType.float32](b * out_c * t)
    conv1d_forward(ctx, module.conv1, x_act, h, b, t, t)

    # Project + add time_emb.
    var t_proj = ctx.enqueue_create_buffer[DType.float32](b * out_c)
    linear_forward(ctx, module.time_emb_proj, t_emb_buf, t_proj, b)
    add_broadcast_time_emb(ctx, h, t_proj, b, out_c, t)

    # ln2 + silu + conv2.
    var h_btc = ctx.enqueue_create_buffer[DType.float32](b * t * out_c)
    transpose_bct_to_btc(ctx, h, h_btc, b, out_c, t)
    var h_ln = ctx.enqueue_create_buffer[DType.float32](b * t * out_c)
    layer_norm_forward(ctx, module.ln2, h_btc, h_ln, b * t)
    var h_bct = ctx.enqueue_create_buffer[DType.float32](b * out_c * t)
    transpose_btc_to_bct(ctx, h_ln, h_bct, b, t, out_c)
    var h_act = ctx.enqueue_create_buffer[DType.float32](b * out_c * t)
    silu(ctx, h_bct, h_act, b * out_c * t)
    var h2 = ctx.enqueue_create_buffer[DType.float32](b * out_c * t)
    conv1d_forward(ctx, module.conv2, h_act, h2, b, t, t)

    # out = h2 + residual.
    ctx.enqueue_copy(out_buf, h2)
    residual_add(ctx, out_buf, residual, b * out_c * t)


@fieldwise_init
struct BasicTransformerBlock(Copyable, Movable):
    """Self-attn + FF block with LN. Same shape as a Llama block but with
    LayerNorm and GELU (matcha uses LN+GELU, not RMSNorm+SwiGLU)."""
    var ln1: LayerNorm
    var to_q: Linear
    var to_k: Linear
    var to_v: Linear
    var to_out: Linear
    var ln2: LayerNorm
    var ff_w1: Linear
    var ff_w2: Linear
    var n_heads: Int
    var head_dim: Int
    var ff_inter: Int


def basic_transformer_forward(
    mut ctx: DeviceContext,
    mut module: BasicTransformerBlock,
    mut x_buf: DeviceBuffer[DType.float32],   # (B, T, D) — updated in-place
    b: Int, t: Int,
) raises:
    from std.math import sqrt
    var H = module.n_heads
    var Dh = module.head_dim
    var D = H * Dh

    # Pre-norm self-attn.
    var x_norm = ctx.enqueue_create_buffer[DType.float32](b * t * D)
    layer_norm_forward(ctx, module.ln1, x_buf, x_norm, b * t)

    var q_lin = ctx.enqueue_create_buffer[DType.float32](b * t * D)
    var k_lin = ctx.enqueue_create_buffer[DType.float32](b * t * D)
    var v_lin = ctx.enqueue_create_buffer[DType.float32](b * t * D)
    linear_forward(ctx, module.to_q, x_norm, q_lin, b * t)
    linear_forward(ctx, module.to_k, x_norm, k_lin, b * t)
    linear_forward(ctx, module.to_v, x_norm, v_lin, b * t)

    var q_perm = ctx.enqueue_create_buffer[DType.float32](b * H * t * Dh)
    var k_perm = ctx.enqueue_create_buffer[DType.float32](b * H * t * Dh)
    var v_perm = ctx.enqueue_create_buffer[DType.float32](b * H * t * Dh)
    reshape_bsd_to_bhsd(ctx, q_lin, q_perm, b, t, H, Dh)
    reshape_bsd_to_bhsd(ctx, k_lin, k_perm, b, t, H, Dh)
    reshape_bsd_to_bhsd(ctx, v_lin, v_perm, b, t, H, Dh)

    var scale: Float32 = 1.0 / sqrt(Float32(Dh))
    var zero_mask = ctx.enqueue_create_buffer[DType.float32](t * t)
    zero_mask.enqueue_fill(0.0)
    var logits = ctx.enqueue_create_buffer[DType.float32](b * H * t * t)
    var probs  = ctx.enqueue_create_buffer[DType.float32](b * H * t * t)
    var av     = ctx.enqueue_create_buffer[DType.float32](b * H * t * Dh)
    qk_scaled_and_masked(ctx, q_perm, k_perm, zero_mask, logits,
                          b * H, t, t, Dh, scale, False)
    softmax_2d(ctx, logits, probs, b * H * t, t)
    av_matmul(ctx, probs, v_perm, av, b * H, t, t, Dh)
    var av_flat = ctx.enqueue_create_buffer[DType.float32](b * t * D)
    reshape_bhsd_to_bsd(ctx, av, av_flat, b, H, t, Dh)
    var attn_out = ctx.enqueue_create_buffer[DType.float32](b * t * D)
    linear_forward(ctx, module.to_out, av_flat, attn_out, b * t)
    residual_add(ctx, x_buf, attn_out, b * t * D)

    # FF.
    var x_norm2 = ctx.enqueue_create_buffer[DType.float32](b * t * D)
    layer_norm_forward(ctx, module.ln2, x_buf, x_norm2, b * t)
    var ff_h = ctx.enqueue_create_buffer[DType.float32](b * t * module.ff_inter)
    linear_forward(ctx, module.ff_w1, x_norm2, ff_h, b * t)
    # Use silu (close enough to matcha's snake/silu for orchestration parity).
    var ff_act = ctx.enqueue_create_buffer[DType.float32](b * t * module.ff_inter)
    silu(ctx, ff_h, ff_act, b * t * module.ff_inter)
    var ff_out = ctx.enqueue_create_buffer[DType.float32](b * t * D)
    linear_forward(ctx, module.ff_w2, ff_act, ff_out, b * t)
    residual_add(ctx, x_buf, ff_out, b * t * D)
