"""UpsampleConformerEncoder (s3gen flow encoder).

Pipeline:
  token_ids → input_embedding lookup → embed.out (Linear + LN) → +pos_emb
    → pre_lookahead (transpose, pad-right-3, conv1 k=4 leaky, pad-left-2, conv2 k=3, transpose, +residual)
    → 6× TransformerEncoderLayer
    → transpose, nearest_upsample×2, pad-left-4, up_layer.conv k=5, transpose
    → up_embed.out (Linear + LN) → +up_pos_emb
    → 4× TransformerEncoderLayer
    → after_norm (LayerNorm)
    → encoder_proj (Linear 512 → 80)
"""
from std.math import sin as msin, cos as mcos, log, exp, sqrt, pi
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from modules import Linear, linear_forward, LayerNorm, layer_norm_forward, residual_add, Embedding, embedding_forward
from conv1d import Conv1d, conv1d_forward
from conformer import (
    TransformerEncoderLayer, transformer_encoder_layer_forward,
)


@fieldwise_init
struct EmbedOut(Copyable, Movable):
    """Linear(idim→odim) + LayerNorm(odim) used at both `embed.out` and `up_embed.out`."""
    var linear: Linear
    var norm: LayerNorm


@fieldwise_init
struct PreLookaheadLayer(Copyable, Movable):
    var conv1: Conv1d   # (channels, channels, K=pre_lookahead_len+1=4)
    var conv2: Conv1d   # (channels, channels, K=3)


@fieldwise_init
struct UpLayerConv(Copyable, Movable):
    """Upsample1D's Conv1d: (in, out, K=stride*2+1=5)."""
    var conv: Conv1d


@fieldwise_init
struct UpsampleConformerEncoderReal(Copyable, Movable):
    var input_embedding: Embedding
    var embed: EmbedOut
    var up_embed: EmbedOut
    var pre_lookahead: PreLookaheadLayer
    var up_layer: UpLayerConv
    var encoders: List[TransformerEncoderLayer]    # 6 pre-upsample layers
    var up_encoders: List[TransformerEncoderLayer] # 4 post-upsample layers
    var after_norm: LayerNorm
    var encoder_proj: Linear                        # (output_size=512, 80)
    var d_model: Int
    var mel_dim: Int


# ============================================================================
# Forward helpers
# ============================================================================


def fill_espnet_relpos(
    mut ctx: DeviceContext,
    mut out_buf: DeviceBuffer[DType.float32],   # (T_pos, D) where T_pos = 2T - 1
    t: Int, d_model: Int,
) raises:
    """Build EspnetRelPositionalEncoding's pos_emb on the fly for a given T.

    Layout in `out_buf` (T_pos = 2T - 1, D):
      Rows [0..T-1]  = `pe_positive_reversed`: pe_pos[T-1 - i, d] for i in [0, T)
      Rows [T..2T-2] = `pe_negative[1..T-1]`:  pe_neg[i - T + 1, d] for i in [T, 2T-1)

    Each row d-index splits even/odd channels:
       even: sin(pos * div_term[d//2])
       odd:  cos(pos * div_term[d//2])
    with div_term[d2] = exp(-(d2*2) * log(10000) / D).
    """
    var out_ptr = out_buf.unsafe_ptr()
    var t_pos = 2 * t - 1
    var lf: Float32 = log(Float32(10000.0)) / Float32(d_model)

    @always_inline
    @parameter
    @__copy_capture(out_ptr, t, t_pos, d_model, lf)
    def pe_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var row = i // d_model
        var col = i - row * d_model
        var d2 = col // 2
        var is_odd = (col - d2 * 2) == 1
        var div_term = exp(-Float32(d2 * 2) * lf)
        var pos: Float32
        # Positive-flipped part for rows < T; negative part for rows >= T.
        if row < t:
            # pe_positive flipped: original index (T-1 - row) corresponds to position (T - 1 - row).
            pos = Float32(t - 1 - row)
        else:
            # pe_negative[1..T-1]: row - T + 1 means absolute index in negative.
            var neg_idx = row - t + 1
            pos = -Float32(neg_idx)
        var arg = pos * div_term
        if is_odd:
            out_ptr[i] = mcos(arg)
        else:
            out_ptr[i] = msin(arg)
    elementwise[pe_fn, simd_width=1, target="gpu"](
        IndexList[1](t_pos * d_model), DeviceContextPtr(ctx),
    )


def scale_buf_inplace(
    mut ctx: DeviceContext,
    mut buf: DeviceBuffer[DType.float32],
    scale: Float32, n: Int,
) raises:
    var p = buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(p, scale)
    def s_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        p[i] = p[i] * scale
    elementwise[s_fn, simd_width=1, target="gpu"](
        IndexList[1](n), DeviceContextPtr(ctx),
    )


def transpose_btc_to_bct(
    mut ctx: DeviceContext,
    mut in_buf: DeviceBuffer[DType.float32],     # (B, T, C)
    mut out_buf: DeviceBuffer[DType.float32],    # (B, C, T)
    b: Int, t: Int, c: Int,
) raises:
    var in_ptr = in_buf.unsafe_ptr()
    var out_ptr = out_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(in_ptr, out_ptr, t, c)
    def trans_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (c * t)
        var rem = i - bi * c * t
        var ci = rem // t
        var ti = rem - ci * t
        out_ptr[i] = in_ptr[bi * t * c + ti * c + ci]
    elementwise[trans_fn, simd_width=1, target="gpu"](
        IndexList[1](b * c * t), DeviceContextPtr(ctx),
    )


def transpose_bct_to_btc(
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
    def trans_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (t * c)
        var rem = i - bi * t * c
        var ti = rem // c
        var ci = rem - ti * c
        out_ptr[i] = in_ptr[bi * c * t + ci * t + ti]
    elementwise[trans_fn, simd_width=1, target="gpu"](
        IndexList[1](b * t * c), DeviceContextPtr(ctx),
    )


def pad_axis_t(
    mut ctx: DeviceContext,
    mut in_buf: DeviceBuffer[DType.float32],     # (B, C, T)
    mut out_buf: DeviceBuffer[DType.float32],    # (B, C, T + pad_l + pad_r)
    b: Int, c: Int, t: Int, pad_l: Int, pad_r: Int,
) raises:
    """Pad the time axis by `pad_l` zeros on the left and `pad_r` on the right."""
    var in_ptr = in_buf.unsafe_ptr()
    var out_ptr = out_buf.unsafe_ptr()
    var t_out = t + pad_l + pad_r

    @always_inline
    @parameter
    @__copy_capture(in_ptr, out_ptr, c, t, t_out, pad_l)
    def pad_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (c * t_out)
        var rem = i - bi * c * t_out
        var ci = rem // t_out
        var ti = rem - ci * t_out
        var src_t = ti - pad_l
        if src_t < 0 or src_t >= t:
            out_ptr[i] = 0.0
        else:
            out_ptr[i] = in_ptr[bi * c * t + ci * t + src_t]
    elementwise[pad_fn, simd_width=1, target="gpu"](
        IndexList[1](b * c * t_out), DeviceContextPtr(ctx),
    )


def leaky_relu_inplace(
    mut ctx: DeviceContext,
    mut x_buf: DeviceBuffer[DType.float32],
    n: Int,
    slope: Float32 = 0.01,
) raises:
    var x_ptr = x_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(x_ptr, slope)
    def lr_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var v = x_ptr[i]
        if v < 0.0:
            x_ptr[i] = v * slope
    elementwise[lr_fn, simd_width=1, target="gpu"](
        IndexList[1](n), DeviceContextPtr(ctx),
    )


def nearest_upsample_2x_bct(
    mut ctx: DeviceContext,
    mut in_buf: DeviceBuffer[DType.float32],     # (B, C, T)
    mut out_buf: DeviceBuffer[DType.float32],    # (B, C, 2T)
    b: Int, c: Int, t: Int,
) raises:
    """Nearest-neighbor upsample by 2 along time: out[..., 2i] = out[..., 2i+1] = in[..., i]."""
    var in_ptr = in_buf.unsafe_ptr()
    var out_ptr = out_buf.unsafe_ptr()
    var t_out = 2 * t

    @always_inline
    @parameter
    @__copy_capture(in_ptr, out_ptr, c, t, t_out)
    def up_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (c * t_out)
        var rem = i - bi * c * t_out
        var ci = rem // t_out
        var ti = rem - ci * t_out
        out_ptr[i] = in_ptr[bi * c * t + ci * t + (ti // 2)]
    elementwise[up_fn, simd_width=1, target="gpu"](
        IndexList[1](b * c * t_out), DeviceContextPtr(ctx),
    )


def embed_forward(
    mut ctx: DeviceContext,
    mut module: EmbedOut,
    mut in_buf: DeviceBuffer[DType.float32],     # (B, T, idim)
    mut out_buf: DeviceBuffer[DType.float32],    # (B, T, odim)
    b: Int, t: Int, idim: Int, odim: Int,
) raises:
    """embed.out forward: Linear → LayerNorm."""
    var h = ctx.enqueue_create_buffer[DType.float32](b * t * odim)
    linear_forward(ctx, module.linear, in_buf, h, b * t)
    layer_norm_forward(ctx, module.norm, h, out_buf, b * t)


def pre_lookahead_forward(
    mut ctx: DeviceContext,
    mut module: PreLookaheadLayer,
    mut x_buf: DeviceBuffer[DType.float32],      # (B, T, D) — updated in-place via residual
    b: Int, t: Int, d: Int,
) raises:
    """PreLookaheadLayer forward:
       x_bct = transpose(x)  # (B, D, T)
       y = pad_right(x_bct, 3)     # (B, D, T+3)
       y = conv1(y)                  # K=4 → (B, D, T)
       y = leaky_relu(y)
       y = pad_left(y, 2)             # (B, D, T+2)
       y = conv2(y)                  # K=3 → (B, D, T)
       y_btc = transpose(y)         # (B, T, D)
       x = x + y_btc
    """
    var n = b * t * d

    # Transpose (B, T, D) → (B, D, T).
    var x_bct = ctx.enqueue_create_buffer[DType.float32](n)
    transpose_btc_to_bct(ctx, x_buf, x_bct, b, t, d)

    # Pad right by 3 (lookahead), conv1 K=4 stride=1 pad=0.
    var pad1 = ctx.enqueue_create_buffer[DType.float32](b * d * (t + 3))
    pad_axis_t(ctx, x_bct, pad1, b, d, t, 0, 3)
    var c1_out = ctx.enqueue_create_buffer[DType.float32](n)
    conv1d_forward(ctx, module.conv1, pad1, c1_out, b, t + 3, t)

    leaky_relu_inplace(ctx, c1_out, n, 0.01)

    # Pad left by 2, conv2 K=3.
    var pad2 = ctx.enqueue_create_buffer[DType.float32](b * d * (t + 2))
    pad_axis_t(ctx, c1_out, pad2, b, d, t, 2, 0)
    var c2_out = ctx.enqueue_create_buffer[DType.float32](n)
    conv1d_forward(ctx, module.conv2, pad2, c2_out, b, t + 2, t)

    # Transpose back (B, D, T) → (B, T, D).
    var y_btc = ctx.enqueue_create_buffer[DType.float32](n)
    transpose_bct_to_btc(ctx, c2_out, y_btc, b, d, t)

    # x += y.
    residual_add(ctx, x_buf, y_btc, n)


def up_layer_forward(
    mut ctx: DeviceContext,
    mut up: UpLayerConv,
    mut in_btc: DeviceBuffer[DType.float32],    # (B, T, D)
    mut out_btc: DeviceBuffer[DType.float32],   # (B, 2T, D)
    b: Int, t: Int, d: Int,
) raises:
    """Upsample1D forward (stride=2, K=5):
       x_bct = transpose(x)
       x_up = interpolate(x_bct, scale=2, nearest)     # (B, D, 2T)
       x_up = pad_left(x_up, 4)                          # (B, D, 2T+4)
       y = conv(x_up)                                      # K=5 → (B, D, 2T)
       y_btc = transpose(y)
    """
    var t2 = 2 * t
    var x_bct = ctx.enqueue_create_buffer[DType.float32](b * d * t)
    transpose_btc_to_bct(ctx, in_btc, x_bct, b, t, d)

    var x_up = ctx.enqueue_create_buffer[DType.float32](b * d * t2)
    nearest_upsample_2x_bct(ctx, x_bct, x_up, b, d, t)

    var x_padded = ctx.enqueue_create_buffer[DType.float32](b * d * (t2 + 4))
    pad_axis_t(ctx, x_up, x_padded, b, d, t2, 4, 0)

    var y = ctx.enqueue_create_buffer[DType.float32](b * d * t2)
    conv1d_forward(ctx, up.conv, x_padded, y, b, t2 + 4, t2)

    transpose_bct_to_btc(ctx, y, out_btc, b, d, t2)


def upsample_conformer_forward(
    mut ctx: DeviceContext,
    mut model: UpsampleConformerEncoderReal,
    mut token_ids: DeviceBuffer[DType.int64],   # (B, T_in) speech token ids
    mut mu_out: DeviceBuffer[DType.float32],    # (B, 80, 2*T_in) — encoder mel projection
    b: Int, t_in: Int,
) raises:
    """Full UpsampleConformerEncoder forward returning the (B, mel=80, 2T) mu
    spectrogram for the CFM decoder.
    """
    var d = model.d_model
    var t_up = 2 * t_in
    var t_pos = 2 * t_in - 1
    var t_pos_up = 2 * t_up - 1
    var mel = model.mel_dim

    # 1. input_embedding lookup (token_ids → (B, T, D)).
    var x_emb = ctx.enqueue_create_buffer[DType.float32](b * t_in * d)
    embedding_forward(ctx, model.input_embedding, token_ids, x_emb, b, t_in)

    # 2. embed.out (Linear + LN).
    var x_embedded = ctx.enqueue_create_buffer[DType.float32](b * t_in * d)
    embed_forward(ctx, model.embed, x_emb, x_embedded, b, t_in, d, d)

    # 3. Apply pe.forward scaling: x *= sqrt(d_model).
    var xscale: Float32 = sqrt(Float32(d))
    scale_buf_inplace(ctx, x_embedded, xscale, b * t_in * d)

    # 4. Build the RelPos sinusoidal pos_emb (1, 2T-1, D) — used by RelPos MHA.
    var pos_emb = ctx.enqueue_create_buffer[DType.float32](t_pos * d)
    fill_espnet_relpos(ctx, pos_emb, t_in, d)

    # 5. pre_lookahead_layer in-place.
    pre_lookahead_forward(ctx, model.pre_lookahead, x_embedded, b, t_in, d)

    # 6. 6 pre-upsample layers.
    for i in range(len(model.encoders)):
        transformer_encoder_layer_forward(
            ctx, model.encoders[i], x_embedded, pos_emb, b, t_in, t_pos,
        )

    # 7. up_layer (interpolate ×2 + conv).
    var x_up = ctx.enqueue_create_buffer[DType.float32](b * t_up * d)
    up_layer_forward(ctx, model.up_layer, x_embedded, x_up, b, t_in, d)

    # 8. up_embed.out + scale + new pos_emb.
    var x_up_embedded = ctx.enqueue_create_buffer[DType.float32](b * t_up * d)
    embed_forward(ctx, model.up_embed, x_up, x_up_embedded, b, t_up, d, d)
    scale_buf_inplace(ctx, x_up_embedded, xscale, b * t_up * d)
    var pos_emb_up = ctx.enqueue_create_buffer[DType.float32](t_pos_up * d)
    fill_espnet_relpos(ctx, pos_emb_up, t_up, d)

    # 9. 4 post-upsample layers.
    for i in range(len(model.up_encoders)):
        transformer_encoder_layer_forward(
            ctx, model.up_encoders[i], x_up_embedded, pos_emb_up, b, t_up, t_pos_up,
        )

    # 10. after_norm.
    var x_norm = ctx.enqueue_create_buffer[DType.float32](b * t_up * d)
    layer_norm_forward(ctx, model.after_norm, x_up_embedded, x_norm, b * t_up)

    # 11. encoder_proj (D → 80) → mu_out is (B, mel, T_up), so we need transpose.
    var mu_btc = ctx.enqueue_create_buffer[DType.float32](b * t_up * mel)
    linear_forward(ctx, model.encoder_proj, x_norm, mu_btc, b * t_up)
    transpose_btc_to_bct(ctx, mu_btc, mu_out, b, t_up, mel)
