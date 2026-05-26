"""CFM estimator structs matching real upstream Chatterbox weights.

Architecture (from chatterbox/models/s3gen/flow_matching matcha decoder):

  estimator
    time_mlp        : Linear 320 → 1024, Linear 1024 → 1024
    down_blocks[0]  : (Resnet1D, [4 BasicTransformerBlocks], Downsample Conv1d)
    mid_blocks[0..11]: each (Resnet1D, [4 BasicTransformerBlocks])
    up_blocks[0]    : (Resnet1D, [4 BasicTransformerBlocks], Upsample Conv1d)
    final_block     : Block1D (Conv1d + GroupNorm + Activation)
    final_proj      : Conv1d 256 → 80

  Resnet1D = (block1: Block1D, block2: Block1D, mlp: Linear 1024 → 256, res_conv: Conv1d 1x1)
  Block1D  = block (Sequential: Conv1d (k=3), Mish, GroupNorm)
  BasicTransformerBlock = norm1 + self-attn (q/k/v dim=512, out dim=256) + norm3 + FF (GEGLU 256→1024→256)
"""
from std.math import sqrt, sin, cos as mcos, log, exp, tanh as mtanh, pi
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from modules import Linear, linear_forward, LayerNorm, layer_norm_forward, residual_add, silu, gelu
from conv1d import Conv1d, conv1d_forward
from attention import qk_scaled_and_masked, softmax_2d, av_matmul
from transformer_blocks import reshape_bsd_to_bhsd, reshape_bhsd_to_bsd


@fieldwise_init
struct GroupNorm1d(Copyable, Movable):
    """PyTorch GroupNorm: weight (γ), bias (β), num_groups."""
    var weight: DeviceBuffer[DType.float32]
    var bias: DeviceBuffer[DType.float32]
    var channels: Int
    var num_groups: Int
    var eps: Float32


@fieldwise_init
struct Block1D(Copyable, Movable):
    """CausalBlock1D (the variant actually used by Chatterbox's
    `ConditionalDecoder`): CausalConv1d(k=3, left-pad-only) → LayerNorm
    (over channel dim, after Transpose) → Mish.

    On disk:
      {base}/block/0/{weight,bias}.bin   — CausalConv1d (c_out, c_in, 3)
      {base}/block/2/{weight,bias}.bin   — LayerNorm (c_out,)

    Note: while matcha's Block1D uses GroupNorm, the real `ConditionalDecoder`
    builds with `causal=True`, swapping in CausalBlock1D + LayerNorm.
    """
    var conv: Conv1d         # Acts as causal conv (caller pre-pads input by k-1 on left)
    var layer_norm: LayerNorm


@fieldwise_init
struct Resnet1D(Copyable, Movable):
    """Resnet block with time-MLP injection: block1 + (+mlp(t)) + block2 + res_conv."""
    var block1: Block1D
    var block2: Block1D
    var mlp: Linear           # (channels, time_emb_dim) — projects t_emb into block1 output
    var res_conv: Conv1d      # 1x1 residual projection (input dim ≠ output dim case)


@fieldwise_init
struct CFMAttention(Copyable, Movable):
    """Self-attention with Q/K/V dim = 512 and output dim = 256.

    Stored as Linear (matches upstream `nn.Linear` exactly — weight (out, in),
    bias (out,)). Q/K/V are bias-less; to_out has bias.
    """
    var to_q: Linear
    var to_k: Linear
    var to_v: Linear
    var to_out: Linear
    var n_heads: Int
    var head_dim: Int


@fieldwise_init
struct CFMFeedForward(Copyable, Movable):
    """GEGLU FF: net.0.proj (256 → 1024 — projects to gate+value pair),
                net.2 (1024 → 256 — output projection).
    """
    var net0_proj: Linear       # (1024, 256) + bias
    var net2: Linear            # (256, 1024) + bias


@fieldwise_init
struct BasicTransformerBlock(Copyable, Movable):
    """One transformer block in the CFM stack."""
    var norm1: LayerNorm
    var attn1: CFMAttention
    var norm3: LayerNorm
    var ff: CFMFeedForward


@fieldwise_init
struct CFMDownStage(Copyable, Movable):
    var resnet: Resnet1D
    var transformers: List[BasicTransformerBlock]   # 4 blocks
    var downsampler: Conv1d                         # 1x1 stride-2 (k=3)


@fieldwise_init
struct CFMMidStage(Copyable, Movable):
    var resnet: Resnet1D
    var transformers: List[BasicTransformerBlock]   # 4 blocks


@fieldwise_init
struct CFMUpStage(Copyable, Movable):
    var resnet: Resnet1D
    var transformers: List[BasicTransformerBlock]
    var upsampler: Conv1d


@fieldwise_init
struct CFMEstimatorReal(Copyable, Movable):
    """Real-upstream-shape CFM estimator. Separate from the old `CFMEstimator`
    struct in `cfm.mojo` so the old forward path still compiles.
    """
    var time_mlp1: Linear
    var time_mlp2: Linear
    var down_blocks: List[CFMDownStage]   # 1 stage
    var mid_blocks: List[CFMMidStage]     # 12 stages
    var up_blocks: List[CFMUpStage]       # 1 stage
    var final_block: Block1D
    var final_proj: Conv1d                # 256 → 80, k=1


# ============================================================================
# Forward helpers
# ============================================================================


def mish(
    mut ctx: DeviceContext,
    mut in_buf: DeviceBuffer[DType.float32],
    mut out_buf: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    """Mish: x * tanh(softplus(x)) = x * tanh(log(1 + exp(x))).

    Uses `std.math.tanh` for the tanh; softplus is computed in numerically
    stable form (no built-in softplus in MAX).
    """
    var in_ptr = in_buf.unsafe_ptr()
    var out_ptr = out_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(in_ptr, out_ptr)
    def mish_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var x = in_ptr[i]
        var sp: Float32
        # softplus, numerically stable.
        if x > 20.0:
            sp = x
        elif x < -20.0:
            sp = exp(x)
        else:
            sp = log(1.0 + exp(x))
        var th = mtanh(sp)
        out_ptr[i] = x * th
    elementwise[mish_func, simd_width=1, target="gpu"](
        IndexList[1](n), DeviceContextPtr(ctx),
    )


def causal_pad_left(
    mut ctx: DeviceContext,
    mut in_buf: DeviceBuffer[DType.float32],     # (B, C, T)
    mut out_buf: DeviceBuffer[DType.float32],    # (B, C, T + pad)
    b: Int, c: Int, t: Int, pad: Int,
) raises:
    """Pad zeros on the left of the time axis: out[..., :pad] = 0, out[..., pad:] = in."""
    var in_ptr = in_buf.unsafe_ptr()
    var out_ptr = out_buf.unsafe_ptr()
    var t_out = t + pad

    @always_inline
    @parameter
    @__copy_capture(in_ptr, out_ptr, c, t, pad, t_out)
    def pad_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (c * t_out)
        var rem = i - bi * c * t_out
        var ci = rem // t_out
        var ti = rem - ci * t_out
        if ti < pad:
            out_ptr[i] = 0.0
        else:
            out_ptr[i] = in_ptr[bi * c * t + ci * t + (ti - pad)]
    elementwise[pad_func, simd_width=1, target="gpu"](
        IndexList[1](b * c * t_out), DeviceContextPtr(ctx),
    )


def transpose_bct_btc(
    mut ctx: DeviceContext,
    mut in_buf: DeviceBuffer[DType.float32],     # (B, C, T)
    mut out_buf: DeviceBuffer[DType.float32],    # (B, T, C)
    b: Int, c: Int, t: Int,
) raises:
    var in_ptr = in_buf.unsafe_ptr()
    var out_ptr = out_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(in_ptr, out_ptr, c, t)
    def fn_t[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (t * c)
        var rem = i - bi * t * c
        var ti = rem // c
        var ci = rem - ti * c
        out_ptr[i] = in_ptr[bi * c * t + ci * t + ti]
    elementwise[fn_t, simd_width=1, target="gpu"](
        IndexList[1](b * t * c), DeviceContextPtr(ctx),
    )


def transpose_btc_bct(
    mut ctx: DeviceContext,
    mut in_buf: DeviceBuffer[DType.float32],     # (B, T, C)
    mut out_buf: DeviceBuffer[DType.float32],    # (B, C, T)
    b: Int, t: Int, c: Int,
) raises:
    var in_ptr = in_buf.unsafe_ptr()
    var out_ptr = out_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(in_ptr, out_ptr, c, t)
    def fn_t[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (c * t)
        var rem = i - bi * c * t
        var ci = rem // t
        var ti = rem - ci * t
        out_ptr[i] = in_ptr[bi * t * c + ti * c + ci]
    elementwise[fn_t, simd_width=1, target="gpu"](
        IndexList[1](b * c * t), DeviceContextPtr(ctx),
    )


def multiply_mask_bct(
    mut ctx: DeviceContext,
    mut x_buf: DeviceBuffer[DType.float32],      # (B, C, T) in-place
    mut mask_buf: DeviceBuffer[DType.float32],   # (B, 1, T)
    b: Int, c: Int, t: Int,
) raises:
    """In-place x *= mask, broadcasting mask along the channel dim."""
    var x_ptr = x_buf.unsafe_ptr()
    var m_ptr = mask_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(x_ptr, m_ptr, c, t)
    def m_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (c * t)
        var rem = i - bi * c * t
        var ci = rem // t
        var ti = rem - ci * t
        x_ptr[i] = x_ptr[i] * m_ptr[bi * t + ti]
    elementwise[m_func, simd_width=1, target="gpu"](
        IndexList[1](b * c * t), DeviceContextPtr(ctx),
    )


def block1d_forward(
    mut ctx: DeviceContext,
    mut block: Block1D,
    mut x_buf: DeviceBuffer[DType.float32],     # (B, C_in, T)
    mut mask_buf: DeviceBuffer[DType.float32],  # (B, 1, T)
    mut out_buf: DeviceBuffer[DType.float32],   # (B, C_out, T)
    b: Int, c_in: Int, c_out: Int, t: Int,
) raises:
    """CausalBlock1D forward:
       x = x * mask
       x = CausalConv1d(x)               # left-pad k-1=2, output T preserved
       x = LayerNorm(x.transpose(1,2))   # over C dim → transpose back
       x = Mish(x)
       return x * mask
    """
    # Apply mask in-place to input copy.
    var x_masked = ctx.enqueue_create_buffer[DType.float32](b * c_in * t)
    ctx.enqueue_copy(x_masked, x_buf)
    multiply_mask_bct(ctx, x_masked, mask_buf, b, c_in, t)

    # Left-pad by 2 (k-1) on the time axis.
    var x_padded = ctx.enqueue_create_buffer[DType.float32](b * c_in * (t + 2))
    causal_pad_left(ctx, x_masked, x_padded, b, c_in, t, 2)

    # Conv1d (no internal pad — padding=0 in loader).
    var c_out_buf = ctx.enqueue_create_buffer[DType.float32](b * c_out * t)
    conv1d_forward(ctx, block.conv, x_padded, c_out_buf, b, t + 2, t)

    # LayerNorm over the C dim: transpose (B,C,T)→(B,T,C), LN per-row, transpose back.
    var x_btc = ctx.enqueue_create_buffer[DType.float32](b * t * c_out)
    transpose_bct_btc(ctx, c_out_buf, x_btc, b, c_out, t)
    var x_ln = ctx.enqueue_create_buffer[DType.float32](b * t * c_out)
    layer_norm_forward(ctx, block.layer_norm, x_btc, x_ln, b * t)
    var x_bct = ctx.enqueue_create_buffer[DType.float32](b * c_out * t)
    transpose_btc_bct(ctx, x_ln, x_bct, b, t, c_out)

    # Mish.
    mish(ctx, x_bct, out_buf, b * c_out * t)

    # Output mask.
    multiply_mask_bct(ctx, out_buf, mask_buf, b, c_out, t)


def add_time_emb_bct(
    mut ctx: DeviceContext,
    mut x_buf: DeviceBuffer[DType.float32],      # (B, C, T) — in-place add
    mut t_proj_buf: DeviceBuffer[DType.float32], # (B, C) — broadcast across T
    b: Int, c: Int, t: Int,
) raises:
    var x_ptr = x_buf.unsafe_ptr()
    var tp_ptr = t_proj_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(x_ptr, tp_ptr, c, t)
    def add_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (c * t)
        var rem = i - bi * c * t
        var ci = rem // t
        x_ptr[i] = x_ptr[i] + tp_ptr[bi * c + ci]
    elementwise[add_func, simd_width=1, target="gpu"](
        IndexList[1](b * c * t), DeviceContextPtr(ctx),
    )


def resnet1d_forward(
    mut ctx: DeviceContext,
    mut module: Resnet1D,
    mut x_buf: DeviceBuffer[DType.float32],      # (B, C_in, T)
    mut mask_buf: DeviceBuffer[DType.float32],   # (B, 1, T)
    mut t_emb_buf: DeviceBuffer[DType.float32],  # (B, time_emb_dim)
    mut out_buf: DeviceBuffer[DType.float32],    # (B, C_out, T)
    b: Int, c_in: Int, c_out: Int, t: Int, time_emb_dim: Int,
) raises:
    """Resnet1D forward:
       h = block1(x, mask)
       h += mlp(Mish(t_emb)).unsqueeze(-1)
       h = block2(h, mask)
       out = h + res_conv(x * mask)
    """
    var h = ctx.enqueue_create_buffer[DType.float32](b * c_out * t)
    block1d_forward(ctx, module.block1, x_buf, mask_buf, h, b, c_in, c_out, t)

    # MLP(t_emb): Mish → Linear(time_emb_dim, c_out)
    var t_mish = ctx.enqueue_create_buffer[DType.float32](b * time_emb_dim)
    mish(ctx, t_emb_buf, t_mish, b * time_emb_dim)
    var t_proj = ctx.enqueue_create_buffer[DType.float32](b * c_out)
    linear_forward(ctx, module.mlp, t_mish, t_proj, b)

    # h += t_proj broadcast over T.
    add_time_emb_bct(ctx, h, t_proj, b, c_out, t)

    # block2.
    var h2 = ctx.enqueue_create_buffer[DType.float32](b * c_out * t)
    block1d_forward(ctx, module.block2, h, mask_buf, h2, b, c_out, c_out, t)

    # Residual: res_conv(x * mask) — input was already premultiplied implicitly
    # by block1, but the original spec multiplies x then convs.
    var x_masked = ctx.enqueue_create_buffer[DType.float32](b * c_in * t)
    ctx.enqueue_copy(x_masked, x_buf)
    multiply_mask_bct(ctx, x_masked, mask_buf, b, c_in, t)
    var r = ctx.enqueue_create_buffer[DType.float32](b * c_out * t)
    conv1d_forward(ctx, module.res_conv, x_masked, r, b, t, t)

    # out = h2 + r.
    ctx.enqueue_copy(out_buf, h2)
    residual_add(ctx, out_buf, r, b * c_out * t)


def cfm_self_attention_forward(
    mut ctx: DeviceContext,
    mut module: CFMAttention,
    mut x_buf: DeviceBuffer[DType.float32],     # (B, T, D)
    mut out_buf: DeviceBuffer[DType.float32],   # (B, T, D)
    b: Int, t: Int,
) raises:
    """Standard scaled-dot-product self-attention with inner dim H*Dh."""
    var H = module.n_heads
    var Dh = module.head_dim
    var inner = H * Dh
    var D = module.to_q.in_features

    # Q/K/V — plain Linear (B*T, D) → (B*T, inner).
    var q = ctx.enqueue_create_buffer[DType.float32](b * t * inner)
    var k = ctx.enqueue_create_buffer[DType.float32](b * t * inner)
    var v = ctx.enqueue_create_buffer[DType.float32](b * t * inner)
    linear_forward(ctx, module.to_q, x_buf, q, b * t)
    linear_forward(ctx, module.to_k, x_buf, k, b * t)
    linear_forward(ctx, module.to_v, x_buf, v, b * t)

    # Reshape (B, T, H*Dh) → (B, H, T, Dh).
    var q_perm = ctx.enqueue_create_buffer[DType.float32](b * H * t * Dh)
    var k_perm = ctx.enqueue_create_buffer[DType.float32](b * H * t * Dh)
    var v_perm = ctx.enqueue_create_buffer[DType.float32](b * H * t * Dh)
    reshape_bsd_to_bhsd(ctx, q, q_perm, b, t, H, Dh)
    reshape_bsd_to_bhsd(ctx, k, k_perm, b, t, H, Dh)
    reshape_bsd_to_bhsd(ctx, v, v_perm, b, t, H, Dh)

    # Scaled QK + softmax + AV.
    var scale: Float32 = 1.0 / sqrt(Float32(Dh))
    var logits = ctx.enqueue_create_buffer[DType.float32](b * H * t * t)
    var probs = ctx.enqueue_create_buffer[DType.float32](b * H * t * t)
    var no_mask = ctx.enqueue_create_buffer[DType.float32](t * t)
    no_mask.enqueue_fill(0.0)
    qk_scaled_and_masked(ctx, q_perm, k_perm, no_mask, logits,
                          b * H, t, t, Dh, scale, False)
    softmax_2d(ctx, logits, probs, b * H * t, t)
    var attn_perm = ctx.enqueue_create_buffer[DType.float32](b * H * t * Dh)
    av_matmul(ctx, probs, v_perm, attn_perm, b * H, t, t, Dh)

    # Reshape (B, H, T, Dh) → (B, T, D) and out_proj.
    var attn_flat = ctx.enqueue_create_buffer[DType.float32](b * t * inner)
    reshape_bhsd_to_bsd(ctx, attn_perm, attn_flat, b, H, t, Dh)
    linear_forward(ctx, module.to_out, attn_flat, out_buf, b * t)


def cfm_ff_forward(
    mut ctx: DeviceContext,
    mut module: CFMFeedForward,
    mut x_buf: DeviceBuffer[DType.float32],     # (B, T, D)
    mut out_buf: DeviceBuffer[DType.float32],   # (B, T, D)
    b: Int, t: Int, d_model: Int, intermediate: Int,
) raises:
    """FeedForward(activation_fn="gelu"):
         h = Linear(D → 1024)(x)       # net.0.proj
         h = gelu(h)                    # GELU.forward applies after proj
         out = Linear(1024 → D)(h)     # net.2
    """
    var h = ctx.enqueue_create_buffer[DType.float32](b * t * intermediate)
    var h_act = ctx.enqueue_create_buffer[DType.float32](b * t * intermediate)
    linear_forward(ctx, module.net0_proj, x_buf, h, b * t)
    gelu(ctx, h, h_act, b * t * intermediate)
    linear_forward(ctx, module.net2, h_act, out_buf, b * t)


def basic_transformer_forward(
    mut ctx: DeviceContext,
    mut module: BasicTransformerBlock,
    mut x_buf: DeviceBuffer[DType.float32],     # (B, T, D) — updated in-place via residuals
    b: Int, t: Int, d_model: Int, intermediate: Int,
) raises:
    """Pre-norm self-attn + FF residual block:
         x = x + attn(norm1(x))
         x = x + ff(norm3(x))
    """
    var n = b * t * d_model

    var x_norm1 = ctx.enqueue_create_buffer[DType.float32](n)
    layer_norm_forward(ctx, module.norm1, x_buf, x_norm1, b * t)
    var attn_out = ctx.enqueue_create_buffer[DType.float32](n)
    cfm_self_attention_forward(ctx, module.attn1, x_norm1, attn_out, b, t)
    residual_add(ctx, x_buf, attn_out, n)

    var x_norm3 = ctx.enqueue_create_buffer[DType.float32](n)
    layer_norm_forward(ctx, module.norm3, x_buf, x_norm3, b * t)
    var ff_out = ctx.enqueue_create_buffer[DType.float32](n)
    cfm_ff_forward(ctx, module.ff, x_norm3, ff_out, b, t, d_model, intermediate)
    residual_add(ctx, x_buf, ff_out, n)


def sinusoidal_pos_emb_fill(
    mut ctx: DeviceContext,
    mut t_scalar_buf: DeviceBuffer[DType.float32],  # (B,) — per-batch time step
    mut out_buf: DeviceBuffer[DType.float32],        # (B, in_channels)
    b: Int, in_channels: Int, scale: Float32 = 1000.0,
) raises:
    """Sinusoidal positional embedding for the time variable:
        half_dim = in_channels // 2
        emb_freq = log(10000) / (half_dim - 1)
        emb_freq = exp(-i * emb_freq) for i in 0..half_dim
        emb = (t * scale).unsqueeze(-1) * emb_freq.unsqueeze(0)
        out = cat(sin(emb), cos(emb), dim=-1)
    """
    var t_ptr = t_scalar_buf.unsafe_ptr()
    var out_ptr = out_buf.unsafe_ptr()
    var half_dim = in_channels // 2
    var lf: Float32 = log(Float32(10000.0)) / Float32(half_dim - 1)

    @always_inline
    @parameter
    @__copy_capture(t_ptr, out_ptr, half_dim, in_channels, lf, scale)
    def spe_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // in_channels
        var di = i - bi * in_channels
        var t_val = t_ptr[bi] * scale
        if di < half_dim:
            var freq = exp(-Float32(di) * lf)
            out_ptr[i] = sin(t_val * freq)
        else:
            var di2 = di - half_dim
            var freq = exp(-Float32(di2) * lf)
            out_ptr[i] = mcos(t_val * freq)
    elementwise[spe_func, simd_width=1, target="gpu"](
        IndexList[1](b * in_channels), DeviceContextPtr(ctx),
    )


def time_embedding_forward(
    mut ctx: DeviceContext,
    mut tm1: Linear, mut tm2: Linear,
    mut t_scalar_buf: DeviceBuffer[DType.float32],   # (B,)
    mut t_emb_out: DeviceBuffer[DType.float32],      # (B, time_embed_dim)
    b: Int, in_channels: Int, time_embed_dim: Int,
) raises:
    """Sinusoidal pos emb (in_channels=320) → Linear(time_embed_dim) → SiLU
    → Linear(time_embed_dim).
    """
    var spe = ctx.enqueue_create_buffer[DType.float32](b * in_channels)
    sinusoidal_pos_emb_fill(ctx, t_scalar_buf, spe, b, in_channels)
    var h = ctx.enqueue_create_buffer[DType.float32](b * time_embed_dim)
    linear_forward(ctx, tm1, spe, h, b)
    var h_act = ctx.enqueue_create_buffer[DType.float32](b * time_embed_dim)
    silu(ctx, h, h_act, b * time_embed_dim)
    linear_forward(ctx, tm2, h_act, t_emb_out, b)


def causal_conv_with_mask(
    mut ctx: DeviceContext,
    mut conv: Conv1d,
    mut x_buf: DeviceBuffer[DType.float32],      # (B, C, T)
    mut mask_buf: DeviceBuffer[DType.float32],   # (B, 1, T)
    mut out_buf: DeviceBuffer[DType.float32],    # (B, C, T)
    b: Int, c: Int, t: Int,
) raises:
    """x = x * mask; left-pad by (k-1); conv (padding=0); — preserves T."""
    var x_masked = ctx.enqueue_create_buffer[DType.float32](b * c * t)
    ctx.enqueue_copy(x_masked, x_buf)
    multiply_mask_bct(ctx, x_masked, mask_buf, b, c, t)
    var x_padded = ctx.enqueue_create_buffer[DType.float32](b * c * (t + 2))
    causal_pad_left(ctx, x_masked, x_padded, b, c, t, 2)
    conv1d_forward(ctx, conv, x_padded, out_buf, b, t + 2, t)


def channel_concat_bct(
    mut ctx: DeviceContext,
    mut a_buf: DeviceBuffer[DType.float32],     # (B, Ca, T)
    mut b_buf_in: DeviceBuffer[DType.float32],  # (B, Cb, T)
    mut out_buf: DeviceBuffer[DType.float32],   # (B, Ca + Cb, T)
    b: Int, ca: Int, cb: Int, t: Int,
) raises:
    """Concatenate two (B, C, T) tensors along the channel axis."""
    var a_ptr = a_buf.unsafe_ptr()
    var b_ptr = b_buf_in.unsafe_ptr()
    var o_ptr = out_buf.unsafe_ptr()
    var c_total = ca + cb

    @always_inline
    @parameter
    @__copy_capture(a_ptr, b_ptr, o_ptr, ca, cb, t, c_total)
    def cat_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (c_total * t)
        var rem = i - bi * c_total * t
        var ci = rem // t
        var ti = rem - ci * t
        if ci < ca:
            o_ptr[i] = a_ptr[bi * ca * t + ci * t + ti]
        else:
            o_ptr[i] = b_ptr[bi * cb * t + (ci - ca) * t + ti]
    elementwise[cat_func, simd_width=1, target="gpu"](
        IndexList[1](b * c_total * t), DeviceContextPtr(ctx),
    )


def broadcast_spks_to_t(
    mut ctx: DeviceContext,
    mut spks_buf: DeviceBuffer[DType.float32],   # (B, C_spk)
    mut out_buf: DeviceBuffer[DType.float32],    # (B, C_spk, T)
    b: Int, c_spk: Int, t: Int,
) raises:
    """Broadcast (B, C) spk embedding across time to (B, C, T)."""
    var s_ptr = spks_buf.unsafe_ptr()
    var o_ptr = out_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(s_ptr, o_ptr, c_spk, t)
    def br_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (c_spk * t)
        var rem = i - bi * c_spk * t
        var ci = rem // t
        o_ptr[i] = s_ptr[bi * c_spk + ci]
    elementwise[br_func, simd_width=1, target="gpu"](
        IndexList[1](b * c_spk * t), DeviceContextPtr(ctx),
    )


def cfm_estimator_forward_real(
    mut ctx: DeviceContext,
    mut model: CFMEstimatorReal,
    mut x_buf: DeviceBuffer[DType.float32],       # (B, 80, T) noisy mel
    mut mu_buf: DeviceBuffer[DType.float32],      # (B, 80, T) encoder output
    mut spks_buf: DeviceBuffer[DType.float32],    # (B, 80) speaker embedding
    mut cond_buf: DeviceBuffer[DType.float32],    # (B, 80, T) prompt cond
    mut mask_buf: DeviceBuffer[DType.float32],    # (B, 1, T)
    mut t_scalar_buf: DeviceBuffer[DType.float32], # (B,) — the timestep
    mut out_buf: DeviceBuffer[DType.float32],     # (B, 80, T) — velocity field
    b: Int, t: Int,
) raises:
    """Full ConditionalDecoder U-Net forward.

    Args (all (B, C, T) bct-layout or (B,) for scalars):
        x: noisy mel sample, (B, 80, T)
        mu: encoder-projected condition, (B, 80, T)
        spks: speaker embed, (B, 80) — broadcast to T
        cond: prompt-conditioning mel, (B, 80, T)
        mask: (B, 1, T)
        t_scalar: per-batch timestep (B,)

    Returns:
        velocity field (B, 80, T) — same shape as x.

    Channel widths through the net:
      input = pack(x, mu, spks_broad, cond) → 320 channels
      down_block: 320 → 256 (resnet) → 256 (downsampler causal_conv)
      mid_block × 12: 256 → 256
      up_block: pack(x[..,:skip_T], skip=hiddens.pop()) → 512 → 256
                                          (resnet 512→256) → 256 (upsampler)
      final_block: 256 → 256
      final_proj: 256 → 80
    """
    comptime D = 256
    comptime TIME_DIM = 1024
    comptime IN_CH = 320
    comptime MEL = 80
    comptime FF_INTER = 1024
    comptime IN_TIME_CH = 320  # SinusoidalPosEmb uses in_channels=320

    # 1. Time embedding.
    var t_emb = ctx.enqueue_create_buffer[DType.float32](b * TIME_DIM)
    time_embedding_forward(
        ctx, model.time_mlp1, model.time_mlp2, t_scalar_buf, t_emb,
        b, IN_TIME_CH, TIME_DIM,
    )

    # 2. Pack inputs along channel axis: [x, mu, spks_broad, cond] → (B, 320, T).
    var spks_broad = ctx.enqueue_create_buffer[DType.float32](b * MEL * t)
    broadcast_spks_to_t(ctx, spks_buf, spks_broad, b, MEL, t)

    var packed_xm = ctx.enqueue_create_buffer[DType.float32](b * (MEL * 2) * t)
    channel_concat_bct(ctx, x_buf, mu_buf, packed_xm, b, MEL, MEL, t)

    var packed_xms = ctx.enqueue_create_buffer[DType.float32](b * (MEL * 3) * t)
    channel_concat_bct(ctx, packed_xm, spks_broad, packed_xms, b, MEL * 2, MEL, t)

    var packed = ctx.enqueue_create_buffer[DType.float32](b * IN_CH * t)
    channel_concat_bct(ctx, packed_xms, cond_buf, packed, b, MEL * 3, MEL, t)

    # Current state x in (B, 320, T).
    var x_cur = ctx.enqueue_create_buffer[DType.float32](b * IN_CH * t)
    ctx.enqueue_copy(x_cur, packed)

    # 3. Down block (1 stage).
    # Apply resnet (320 → 256).
    var x_resnet = ctx.enqueue_create_buffer[DType.float32](b * D * t)
    resnet1d_forward(
        ctx, model.down_blocks[0].resnet, x_cur, mask_buf, t_emb, x_resnet,
        b, IN_CH, D, t, TIME_DIM,
    )

    # Transformer blocks operate on (B, T, D).
    var x_btc = ctx.enqueue_create_buffer[DType.float32](b * t * D)
    transpose_bct_btc(ctx, x_resnet, x_btc, b, D, t)
    for i in range(4):
        basic_transformer_forward(
            ctx, model.down_blocks[0].transformers[i], x_btc,
            b, t, D, FF_INTER,
        )
    var x_bct2 = ctx.enqueue_create_buffer[DType.float32](b * D * t)
    transpose_btc_bct(ctx, x_btc, x_bct2, b, t, D)

    # Save hidden state for skip connection (after attention, before downsampler).
    var skip = ctx.enqueue_create_buffer[DType.float32](b * D * t)
    ctx.enqueue_copy(skip, x_bct2)

    # Downsampler = CausalConv1d k=3.
    var x_down = ctx.enqueue_create_buffer[DType.float32](b * D * t)
    causal_conv_with_mask(
        ctx, model.down_blocks[0].downsampler, x_bct2, mask_buf, x_down, b, D, t,
    )

    # 4. Mid blocks (12 stages).
    var x_mid = ctx.enqueue_create_buffer[DType.float32](b * D * t)
    ctx.enqueue_copy(x_mid, x_down)
    for s in range(12):
        var x_r = ctx.enqueue_create_buffer[DType.float32](b * D * t)
        resnet1d_forward(
            ctx, model.mid_blocks[s].resnet, x_mid, mask_buf, t_emb, x_r,
            b, D, D, t, TIME_DIM,
        )
        var x_t = ctx.enqueue_create_buffer[DType.float32](b * t * D)
        transpose_bct_btc(ctx, x_r, x_t, b, D, t)
        for i in range(4):
            basic_transformer_forward(
                ctx, model.mid_blocks[s].transformers[i], x_t,
                b, t, D, FF_INTER,
            )
        var x_b = ctx.enqueue_create_buffer[DType.float32](b * D * t)
        transpose_btc_bct(ctx, x_t, x_b, b, t, D)
        ctx.enqueue_copy(x_mid, x_b)

    # 5. Up block (1 stage). Concat skip along channel: 256 + 256 → 512 → resnet → 256.
    var x_cat = ctx.enqueue_create_buffer[DType.float32](b * (D * 2) * t)
    channel_concat_bct(ctx, x_mid, skip, x_cat, b, D, D, t)

    var x_u_resnet = ctx.enqueue_create_buffer[DType.float32](b * D * t)
    resnet1d_forward(
        ctx, model.up_blocks[0].resnet, x_cat, mask_buf, t_emb, x_u_resnet,
        b, D * 2, D, t, TIME_DIM,
    )

    var x_u_btc = ctx.enqueue_create_buffer[DType.float32](b * t * D)
    transpose_bct_btc(ctx, x_u_resnet, x_u_btc, b, D, t)
    for i in range(4):
        basic_transformer_forward(
            ctx, model.up_blocks[0].transformers[i], x_u_btc,
            b, t, D, FF_INTER,
        )
    var x_u_bct = ctx.enqueue_create_buffer[DType.float32](b * D * t)
    transpose_btc_bct(ctx, x_u_btc, x_u_bct, b, t, D)

    var x_up = ctx.enqueue_create_buffer[DType.float32](b * D * t)
    causal_conv_with_mask(
        ctx, model.up_blocks[0].upsampler, x_u_bct, mask_buf, x_up, b, D, t,
    )

    # 6. Final block.
    var x_final = ctx.enqueue_create_buffer[DType.float32](b * D * t)
    block1d_forward(
        ctx, model.final_block, x_up, mask_buf, x_final, b, D, D, t,
    )

    # 7. Final proj (1×1 conv 256 → 80).
    var x_final_masked = ctx.enqueue_create_buffer[DType.float32](b * D * t)
    ctx.enqueue_copy(x_final_masked, x_final)
    multiply_mask_bct(ctx, x_final_masked, mask_buf, b, D, t)
    conv1d_forward(ctx, model.final_proj, x_final_masked, out_buf, b, t, t)


# ============================================================================
# CFM Euler ODE solver
# ============================================================================


def cfm_solve_euler(
    mut ctx: DeviceContext,
    mut model: CFMEstimatorReal,
    mut x_buf: DeviceBuffer[DType.float32],       # (B, 80, T) initial noise; updated in-place
    mut mu_buf: DeviceBuffer[DType.float32],      # (B, 80, T) encoder output
    mut spks_buf: DeviceBuffer[DType.float32],    # (B, 80) speaker embedding (post-spk_embed_affine)
    mut cond_buf: DeviceBuffer[DType.float32],    # (B, 80, T) prompt cond (zeros if no prompt)
    mut mask_buf: DeviceBuffer[DType.float32],    # (B, 1, T)
    b: Int, t: Int, n_steps: Int, cfg_rate: Float32,
    use_cosine_schedule: Bool = True,
) raises:
    """Fixed-step Euler integration of the CFM ODE with classifier-free guidance.

    For each step (t_cur, t_next) in t_span:
       Build doubled inputs: first B = conditioned (real spks/mu/cond),
                            second B = unconditioned (zeros for spks/mu/cond).
       Run estimator on the doubled batch.
       Split, combine: dxdt = (1+cfg) * cond - cfg * uncond.
       x = x + (t_next - t_cur) * dxdt.

    Upstream uses a cosine schedule:
        t_span = 1 - cos(linspace(0, 1, n_steps+1) * pi/2)
    Set use_cosine_schedule=False for the legacy linear schedule.

    For inference B=1, the doubled batch has 2 samples.
    """
    comptime MEL = 80
    var B2 = 2 * b

    # Pre-allocate doubled input buffers once (reused across steps).
    var x_in = ctx.enqueue_create_buffer[DType.float32](B2 * MEL * t)
    var mu_in = ctx.enqueue_create_buffer[DType.float32](B2 * MEL * t)
    var spks_in = ctx.enqueue_create_buffer[DType.float32](B2 * MEL)
    var cond_in = ctx.enqueue_create_buffer[DType.float32](B2 * MEL * t)
    var mask_in = ctx.enqueue_create_buffer[DType.float32](B2 * t)
    var t_scalar_in = ctx.enqueue_create_buffer[DType.float32](B2)
    var dxdt_in = ctx.enqueue_create_buffer[DType.float32](B2 * MEL * t)

    var dt_per_step: Float32 = 1.0 / Float32(n_steps)
    comptime PI_HALF: Float32 = 1.5707963267948966

    # ── Hoist constant-across-steps inputs out of the Euler loop ──
    # mu_in, spks_in, cond_in, mask_in depend only on inputs (not on step).
    # Only x_in and t_scalar_in change per step. Building these once saves
    # 9× the elementwise dispatch overhead vs the previous per-step build.

    # mu_in: [:B] = mu, [B:] = 0.
    var mu_ptr_h = mu_buf.unsafe_ptr()
    var mi_ptr_h = mu_in.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(mu_ptr_h, mi_ptr_h, b, t, MEL)
    def mu_init_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (MEL * t)
        var rem = i - bi * MEL * t
        if bi < b:
            mi_ptr_h[i] = mu_ptr_h[bi * MEL * t + rem]
        else:
            mi_ptr_h[i] = 0.0
    elementwise[mu_init_fn, simd_width=1, target="gpu"](
        IndexList[1](B2 * MEL * t), DeviceContextPtr(ctx),
    )

    # spks_in: [:B] = spks, [B:] = 0.
    var sp_ptr_h = spks_buf.unsafe_ptr()
    var si_ptr_h = spks_in.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(sp_ptr_h, si_ptr_h, b)
    def spk_init_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // MEL
        var ci = i - bi * MEL
        if bi < b:
            si_ptr_h[i] = sp_ptr_h[bi * MEL + ci]
        else:
            si_ptr_h[i] = 0.0
    elementwise[spk_init_fn, simd_width=1, target="gpu"](
        IndexList[1](B2 * MEL), DeviceContextPtr(ctx),
    )

    # cond_in: [:B] = cond, [B:] = 0.
    var co_ptr_h = cond_buf.unsafe_ptr()
    var ci_ptr_h = cond_in.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(co_ptr_h, ci_ptr_h, b, t, MEL)
    def cond_init_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (MEL * t)
        var rem = i - bi * MEL * t
        if bi < b:
            ci_ptr_h[i] = co_ptr_h[bi * MEL * t + rem]
        else:
            ci_ptr_h[i] = 0.0
    elementwise[cond_init_fn, simd_width=1, target="gpu"](
        IndexList[1](B2 * MEL * t), DeviceContextPtr(ctx),
    )

    # mask_in: [:B] = mask, [B:] = mask.
    var m_ptr_h = mask_buf.unsafe_ptr()
    var mi2_ptr_h = mask_in.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(m_ptr_h, mi2_ptr_h, b, t)
    def mask_init_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // t
        var src_b = bi % b
        var ti = i - bi * t
        mi2_ptr_h[i] = m_ptr_h[src_b * t + ti]
    elementwise[mask_init_fn, simd_width=1, target="gpu"](
        IndexList[1](B2 * t), DeviceContextPtr(ctx),
    )

    for step in range(n_steps):
        var lin_cur: Float32 = Float32(step) * dt_per_step
        var lin_next: Float32 = Float32(step + 1) * dt_per_step
        var t_cur: Float32
        var t_next: Float32
        if use_cosine_schedule:
            t_cur = 1.0 - mcos(lin_cur * PI_HALF)
            t_next = 1.0 - mcos(lin_next * PI_HALF)
        else:
            t_cur = lin_cur
            t_next = lin_next
        var dt = t_next - t_cur

        # Populate doubled inputs.
        # x_in: [:B] = x, [B:] = x (same).
        var x_ptr = x_buf.unsafe_ptr()
        var xi_ptr = x_in.unsafe_ptr()

        @always_inline
        @parameter
        @__copy_capture(x_ptr, xi_ptr, b, t, MEL)
        def x_pop_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
            var i = idx[0]
            var bi = i // (MEL * t)
            var src_b = bi % b
            var rem = i - bi * MEL * t
            xi_ptr[i] = x_ptr[src_b * MEL * t + rem]
        elementwise[x_pop_fn, simd_width=1, target="gpu"](
            IndexList[1](B2 * MEL * t), DeviceContextPtr(ctx),
        )

        # mu_in, spks_in, cond_in, mask_in are constant — built once before the loop.

        # t_scalar_in: filled with t_cur.
        t_scalar_in.enqueue_fill(t_cur)

        # Run estimator forward on doubled batch.
        cfm_estimator_forward_real(
            ctx, model, x_in, mu_in, spks_in, cond_in, mask_in, t_scalar_in,
            dxdt_in, B2, t,
        )

        # Combine CFG: dxdt = (1+cfg) * cond - cfg * uncond, then x += dt * dxdt.
        var d_ptr = dxdt_in.unsafe_ptr()
        var xb_ptr = x_buf.unsafe_ptr()

        @always_inline
        @parameter
        @__copy_capture(d_ptr, xb_ptr, b, t, MEL, dt, cfg_rate)
        def combine_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
            var i = idx[0]
            # i indexes (B, MEL, T).
            var cond_val = d_ptr[i]
            var uncond_val = d_ptr[b * MEL * t + i]
            var dxdt_val = (1.0 + cfg_rate) * cond_val - cfg_rate * uncond_val
            xb_ptr[i] = xb_ptr[i] + dt * dxdt_val
        elementwise[combine_fn, simd_width=1, target="gpu"](
            IndexList[1](b * MEL * t), DeviceContextPtr(ctx),
        )


def cfm_solve_heun(
    mut ctx: DeviceContext,
    mut model: CFMEstimatorReal,
    mut x_buf: DeviceBuffer[DType.float32],
    mut mu_buf: DeviceBuffer[DType.float32],
    mut spks_buf: DeviceBuffer[DType.float32],
    mut cond_buf: DeviceBuffer[DType.float32],
    mut mask_buf: DeviceBuffer[DType.float32],
    b: Int, t: Int, n_steps: Int, cfg_rate: Float32,
    use_cosine_schedule: Bool = True,
) raises:
    """Heun's method (RK2): 2 estimator calls per step but error O(dt²)
    instead of Euler's O(dt). Roughly: Heun-2 ≈ Euler-4 quality at the same
    or lower cost.

    Per step [t_cur, t_next] with dt = t_next - t_cur:
        f1 = estimator(x, t_cur)
        x_pred = x + dt * f1
        f2 = estimator(x_pred, t_next)
        x = x + dt/2 * (f1 + f2)
    """
    comptime MEL = 80
    var B2 = 2 * b

    # Scratch buffers (reused across steps).
    var x_in = ctx.enqueue_create_buffer[DType.float32](B2 * MEL * t)
    var mu_in = ctx.enqueue_create_buffer[DType.float32](B2 * MEL * t)
    var spks_in = ctx.enqueue_create_buffer[DType.float32](B2 * MEL)
    var cond_in = ctx.enqueue_create_buffer[DType.float32](B2 * MEL * t)
    var mask_in = ctx.enqueue_create_buffer[DType.float32](B2 * t)
    var t_scalar_in = ctx.enqueue_create_buffer[DType.float32](B2)
    var dxdt1_in = ctx.enqueue_create_buffer[DType.float32](B2 * MEL * t)
    var dxdt2_in = ctx.enqueue_create_buffer[DType.float32](B2 * MEL * t)
    # x_pred holds x + dt * f1 (B, MEL, T).
    var x_pred = ctx.enqueue_create_buffer[DType.float32](b * MEL * t)

    var dt_per_step: Float32 = 1.0 / Float32(n_steps)
    comptime PI_HALF: Float32 = 1.5707963267948966

    # Build constant-across-steps inputs once.
    var mu_ptr_h = mu_buf.unsafe_ptr()
    var mi_ptr_h = mu_in.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(mu_ptr_h, mi_ptr_h, b, t, MEL)
    def heun_mu_init[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (MEL * t)
        var rem = i - bi * MEL * t
        if bi < b:
            mi_ptr_h[i] = mu_ptr_h[bi * MEL * t + rem]
        else:
            mi_ptr_h[i] = 0.0
    elementwise[heun_mu_init, simd_width=1, target="gpu"](
        IndexList[1](B2 * MEL * t), DeviceContextPtr(ctx),
    )

    var sp_ptr_h = spks_buf.unsafe_ptr()
    var si_ptr_h = spks_in.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(sp_ptr_h, si_ptr_h, b)
    def heun_spk_init[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // MEL
        var ci = i - bi * MEL
        if bi < b:
            si_ptr_h[i] = sp_ptr_h[bi * MEL + ci]
        else:
            si_ptr_h[i] = 0.0
    elementwise[heun_spk_init, simd_width=1, target="gpu"](
        IndexList[1](B2 * MEL), DeviceContextPtr(ctx),
    )

    var co_ptr_h = cond_buf.unsafe_ptr()
    var ci_ptr_h = cond_in.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(co_ptr_h, ci_ptr_h, b, t, MEL)
    def heun_cond_init[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (MEL * t)
        var rem = i - bi * MEL * t
        if bi < b:
            ci_ptr_h[i] = co_ptr_h[bi * MEL * t + rem]
        else:
            ci_ptr_h[i] = 0.0
    elementwise[heun_cond_init, simd_width=1, target="gpu"](
        IndexList[1](B2 * MEL * t), DeviceContextPtr(ctx),
    )

    var m_ptr_h = mask_buf.unsafe_ptr()
    var mi2_ptr_h = mask_in.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(m_ptr_h, mi2_ptr_h, b, t)
    def heun_mask_init[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // t
        var src_b = bi % b
        var ti = i - bi * t
        mi2_ptr_h[i] = m_ptr_h[src_b * t + ti]
    elementwise[heun_mask_init, simd_width=1, target="gpu"](
        IndexList[1](B2 * t), DeviceContextPtr(ctx),
    )

    for step in range(n_steps):
        var lin_cur: Float32 = Float32(step) * dt_per_step
        var lin_next: Float32 = Float32(step + 1) * dt_per_step
        var t_cur: Float32
        var t_next: Float32
        if use_cosine_schedule:
            t_cur = 1.0 - mcos(lin_cur * PI_HALF)
            t_next = 1.0 - mcos(lin_next * PI_HALF)
        else:
            t_cur = lin_cur
            t_next = lin_next
        var dt = t_next - t_cur
        var half_dt = dt * 0.5

        # ── Stage 1: f1 = estimator(x, t_cur) ──
        var x_ptr = x_buf.unsafe_ptr()
        var xi_ptr = x_in.unsafe_ptr()

        @always_inline
        @parameter
        @__copy_capture(x_ptr, xi_ptr, b, t, MEL)
        def heun_x_pop[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
            var i = idx[0]
            var bi = i // (MEL * t)
            var src_b = bi % b
            var rem = i - bi * MEL * t
            xi_ptr[i] = x_ptr[src_b * MEL * t + rem]
        elementwise[heun_x_pop, simd_width=1, target="gpu"](
            IndexList[1](B2 * MEL * t), DeviceContextPtr(ctx),
        )
        t_scalar_in.enqueue_fill(t_cur)
        cfm_estimator_forward_real(
            ctx, model, x_in, mu_in, spks_in, cond_in, mask_in, t_scalar_in,
            dxdt1_in, B2, t,
        )

        # f1[b, mel, t] = (1+cfg)*cond - cfg*uncond. Combine into per-row first-B view via x_pred buf:
        # x_pred = x + dt * f1
        var d1_ptr = dxdt1_in.unsafe_ptr()
        var xb_ptr = x_buf.unsafe_ptr()
        var xp_ptr = x_pred.unsafe_ptr()

        @always_inline
        @parameter
        @__copy_capture(d1_ptr, xb_ptr, xp_ptr, b, t, MEL, dt, cfg_rate)
        def heun_pred[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
            var i = idx[0]
            var cond_v = d1_ptr[i]
            var uncond_v = d1_ptr[b * MEL * t + i]
            var f1 = (1.0 + cfg_rate) * cond_v - cfg_rate * uncond_v
            xp_ptr[i] = xb_ptr[i] + dt * f1
        elementwise[heun_pred, simd_width=1, target="gpu"](
            IndexList[1](b * MEL * t), DeviceContextPtr(ctx),
        )

        # ── Stage 2: f2 = estimator(x_pred, t_next) ──
        var xp_ptr2 = x_pred.unsafe_ptr()
        var xi_ptr2 = x_in.unsafe_ptr()

        @always_inline
        @parameter
        @__copy_capture(xp_ptr2, xi_ptr2, b, t, MEL)
        def heun_xpred_pop[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
            var i = idx[0]
            var bi = i // (MEL * t)
            var src_b = bi % b
            var rem = i - bi * MEL * t
            xi_ptr2[i] = xp_ptr2[src_b * MEL * t + rem]
        elementwise[heun_xpred_pop, simd_width=1, target="gpu"](
            IndexList[1](B2 * MEL * t), DeviceContextPtr(ctx),
        )
        t_scalar_in.enqueue_fill(t_next)
        cfm_estimator_forward_real(
            ctx, model, x_in, mu_in, spks_in, cond_in, mask_in, t_scalar_in,
            dxdt2_in, B2, t,
        )

        # x = x + dt/2 * (f1 + f2)
        var d2_ptr = dxdt2_in.unsafe_ptr()

        @always_inline
        @parameter
        @__copy_capture(d1_ptr, d2_ptr, xb_ptr, b, t, MEL, half_dt, cfg_rate)
        def heun_update[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
            var i = idx[0]
            var c1 = d1_ptr[i]; var u1 = d1_ptr[b * MEL * t + i]
            var c2 = d2_ptr[i]; var u2 = d2_ptr[b * MEL * t + i]
            var f1 = (1.0 + cfg_rate) * c1 - cfg_rate * u1
            var f2 = (1.0 + cfg_rate) * c2 - cfg_rate * u2
            xb_ptr[i] = xb_ptr[i] + half_dt * (f1 + f2)
        elementwise[heun_update, simd_width=1, target="gpu"](
            IndexList[1](b * MEL * t), DeviceContextPtr(ctx),
        )


# ============================================================================
# Deterministic Gaussian noise (Box-Muller via LCG)
# ============================================================================

def gaussian_noise_fill(
    mut ctx: DeviceContext,
    mut out_buf: DeviceBuffer[DType.float32],
    n: Int, seed: UInt64,
    sigma: Float32 = 1.0,
) raises:
    """Fill `out_buf` with deterministic Gaussian-distributed noise.

    Each output element uses two LCG draws + Box-Muller polar transform to
    produce a unit-variance normal sample, scaled by `sigma`.

    Not bit-exact to `torch.randn` (different RNG and ordering) but it is
    properly Gaussian-distributed, which is what CFM Euler needs.
    """
    var out_ptr = out_buf.unsafe_ptr()
    from std.math import sqrt, log, sin as msin, cos as mcos, pi

    @always_inline
    @parameter
    @__copy_capture(out_ptr, seed, sigma)
    def gn_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        # Per-element deterministic state. 2654435761 = Knuth multiplicative hash.
        var s: UInt64 = seed ^ (UInt64(i) * UInt64(2654435761))
        # Two LCG steps → two uniforms in (0, 1).
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        # Use the top 24 bits as a uniform integer in [0, 2^24), then normalize.
        var bits1: UInt64 = (s >> UInt64(40)) & UInt64(0xFFFFFF)
        # Add 0.5 to avoid log(0).
        var u1: Float32 = (Float32(Int(bits1)) + 0.5) / Float32(16777216.0)
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var bits2: UInt64 = (s >> UInt64(40)) & UInt64(0xFFFFFF)
        var u2: Float32 = (Float32(Int(bits2)) + 0.5) / Float32(16777216.0)
        # Box-Muller (polar form).
        var r: Float32 = sqrt(-2.0 * log(u1))
        var theta: Float32 = 2.0 * Float32(pi) * u2
        var z: Float32 = r * mcos(theta) * sigma
        out_ptr[i] = z
    elementwise[gn_fn, simd_width=1, target="gpu"](
        IndexList[1](n), DeviceContextPtr(ctx),
    )
