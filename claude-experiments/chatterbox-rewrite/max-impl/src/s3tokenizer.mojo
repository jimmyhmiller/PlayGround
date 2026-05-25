"""S3TokenizerV2 forward: log_mel_16k → 2 strided Conv1d (+ GELU) → N Conformer
blocks (FSMN attention + MLP) → FSQ codebook → speech token indices.

Composed from MAX abstractions throughout:
  - `nn.gather_scatter.gather` for embedding/codebook lookups (not used here directly)
  - `linalg.matmul` for Linear projections inside blocks
  - `linalg.bmm.batched_matmul` for attention inside blocks
  - `nn.softmax`, `nn.normalization.layer_norm`, `nn.rope.apply_rope`
  - `elementwise[..., target="gpu"]` for FSQ scoring + reshape + activations
"""
from std.math import tanh
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from modules import Linear, linear_forward, gelu
from conv1d import Conv1d, conv1d_forward
from s3tokenizer_block import S3TokenizerBlock, s3tokenizer_block_forward


@fieldwise_init
struct S3Tokenizer(Copyable, Movable):
    var conv1: Conv1d        # (n_state, n_mels, 3) stride=2 padding=1
    var conv2: Conv1d        # (n_state, n_state, 3) stride=2 padding=1
    var blocks: List[S3TokenizerBlock]
    var project_down: Linear # (8, n_state) — FSQ projection
    var n_mels: Int
    var n_state: Int
    var n_heads: Int
    var head_dim: Int
    var n_layers: Int


def fsq_encode(
    mut ctx: DeviceContext,
    mut x_buf: DeviceBuffer[DType.float32],     # (B, T, 8) after project_down
    mut idx_buf: DeviceBuffer[DType.int32],      # (B, T) — codebook indices
    b: Int, t: Int,
) raises:
    """FSQ codebook: h = tanh(x) * 0.999; r = round(h) + 1; mu = sum_i r_i * 3^i."""
    var x_ptr = x_buf.unsafe_ptr()
    var i_ptr = idx_buf.unsafe_ptr()
    var dctx = DeviceContextPtr(ctx)

    @always_inline
    @parameter
    @__copy_capture(x_ptr, i_ptr)
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var off = i * 8
        var mu: Int32 = 0
        var p: Int32 = 1
        comptime SCALE: Float32 = 0.9990000128746033
        for k in range(8):
            var v = tanh(x_ptr[off + k]) * SCALE
            var r: Int32 = 0
            if v >= 0.0:
                # round-half-away-from-zero — v in (-0.999, 0.999) so half-cases don't arise.
                r = Int32(Int(v + 0.5))
            else:
                r = -Int32(Int(-v + 0.5))
            r += 1
            mu += r * p
            p *= 3
        i_ptr[i] = mu
    elementwise[func, simd_width=1, target="gpu"](
        IndexList[1](b * t), dctx,
    )


def s3tokenizer_forward(
    mut ctx: DeviceContext,
    mut model: S3Tokenizer,
    mut mel_buf: DeviceBuffer[DType.float32],   # (B, n_mels, T_mel) log-mel input
    mut tokens_out: DeviceBuffer[DType.int32],   # (B, T_token) speech token ids
    mut cos_buf: DeviceBuffer[DType.float32],   # RoPE cos/sin (max ctx)
    mut sin_buf: DeviceBuffer[DType.float32],
    mut mask_pad_buf: DeviceBuffer[DType.float32], # (B, T_token, 1)
    mut attn_mask_buf: DeviceBuffer[DType.float32], # (T_token, T_token)
    b: Int, t_mel: Int,
) raises:
    """Run AudioEncoderV2 (conv1+gelu+conv2+gelu+N blocks) + FSQ.

    Output sequence length T_token = (T_mel after 2 strided conv1ds) — typically T_mel/4.
    """
    var ns = model.n_state
    # Conv1 down: T_mel → T1.
    var t1 = (t_mel + 2 - (3 - 1) - 1) // 2 + 1
    var c1 = ctx.enqueue_create_buffer[DType.float32](b * ns * t1)
    conv1d_forward(ctx, model.conv1, mel_buf, c1, b, t_mel, t1)
    var c1_act = ctx.enqueue_create_buffer[DType.float32](b * ns * t1)
    gelu(ctx, c1, c1_act, b * ns * t1)

    # Conv2 down: T1 → T2.
    var t2 = (t1 + 2 - (3 - 1) - 1) // 2 + 1
    var c2 = ctx.enqueue_create_buffer[DType.float32](b * ns * t2)
    conv1d_forward(ctx, model.conv2, c1_act, c2, b, t1, t2)
    var c2_act = ctx.enqueue_create_buffer[DType.float32](b * ns * t2)
    gelu(ctx, c2, c2_act, b * ns * t2)

    # Transpose (B, C, T) → (B, T, C) for transformer blocks.
    var x_seq = ctx.enqueue_create_buffer[DType.float32](b * t2 * ns)
    var cp = c2_act.unsafe_ptr()
    var sp = x_seq.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(cp, sp, b, t2, ns)
    def trans_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (t2 * ns)
        var rem = i - bi * t2 * ns
        var ti = rem // ns
        var ci = rem - ti * ns
        sp[i] = cp[bi * ns * t2 + ci * t2 + ti]
    elementwise[trans_func, simd_width=1, target="gpu"](
        IndexList[1](b * t2 * ns), DeviceContextPtr(ctx),
    )

    # Run all blocks (FSMN attn + MLP) in-place.
    for i in range(model.n_layers):
        s3tokenizer_block_forward(
            ctx, model.blocks[i], x_seq, cos_buf, sin_buf,
            mask_pad_buf, attn_mask_buf, b, t2,
            has_attn_mask=False,    # we run without attention masking here
        )

    # FSQ: project_down (B, T, n_state) → (B, T, 8), then encode.
    var proj_buf = ctx.enqueue_create_buffer[DType.float32](b * t2 * 8)
    linear_forward(ctx, model.project_down, x_seq, proj_buf, b * t2)
    fsq_encode(ctx, proj_buf, tokens_out, b, t2)
