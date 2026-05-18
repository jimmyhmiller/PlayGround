"""s3gen flow: speech_tokens → encoder → CFM mel → HiFiGAN audio.

Orchestration uses only MAX abstractions via our wrapper modules.
"""
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from modules import Linear, linear_forward, Embedding, embedding_forward, LayerNorm
from conv1d import Conv1d, conv1d_forward
from conformer import ConformerLayer, conformer_layer_forward
from cfm import CFMEstimator, cfm_euler_solve
from hifigan import HiFiGAN, hifigan_forward


@fieldwise_init
struct UpsampleConformerEncoder(Copyable, Movable):
    """6 conformer layers + nearest upsample by 2 + 4 more conformer layers
       + final encoder_proj to mel."""
    var token_emb: Embedding
    var layers_pre: List[ConformerLayer]       # 6 layers
    var layers_post: List[ConformerLayer]      # 4 layers (post-upsample)
    var encoder_proj: Linear                   # (mel=80, d_enc)
    var d_enc: Int
    var mel: Int


def upsample_conformer_forward(
    mut ctx: DeviceContext,
    mut enc: UpsampleConformerEncoder,
    mut speech_tokens_buf: DeviceBuffer[DType.int64],   # (B, T_token)
    mut out_mel_buf: DeviceBuffer[DType.float32],       # (B, mel, T_h)
    mut cos_buf: DeviceBuffer[DType.float32],
    mut sin_buf: DeviceBuffer[DType.float32],
    mut mask_buf: DeviceBuffer[DType.float32],
    b: Int, t_token: Int,
) raises:
    var D = enc.d_enc
    # Embed.
    var x = ctx.enqueue_create_buffer[DType.float32](b * t_token * D)
    embedding_forward(ctx, enc.token_emb, speech_tokens_buf, x, b, t_token)
    # 6 pre-conformer layers.
    for i in range(len(enc.layers_pre)):
        conformer_layer_forward(
            ctx, enc.layers_pre[i], x, cos_buf, sin_buf, mask_buf, b, t_token, False,
        )
    # Upsample T by 2 (interleave: each timestep duplicated).
    var t_h = t_token * 2
    var x_up = ctx.enqueue_create_buffer[DType.float32](b * t_h * D)
    var xp = x.unsafe_ptr()
    var ump = x_up.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(xp, ump, t_token, D)
    def up_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (t_token * 2 * D)
        var rem = i - bi * t_token * 2 * D
        var ti = rem // D
        var di = rem - ti * D
        var src_t = ti // 2
        ump[i] = xp[bi * t_token * D + src_t * D + di]
    elementwise[up_func, simd_width=1, target="gpu"](
        IndexList[1](b * t_h * D), DeviceContextPtr(ctx),
    )
    # 4 post-conformer layers.
    for i in range(len(enc.layers_post)):
        conformer_layer_forward(
            ctx, enc.layers_post[i], x_up, cos_buf, sin_buf, mask_buf, b, t_h, False,
        )
    # Project to mel: (B, T_h, D) → (B, T_h, mel) → transpose to (B, mel, T_h).
    var mel_btc = ctx.enqueue_create_buffer[DType.float32](b * t_h * enc.mel)
    linear_forward(ctx, enc.encoder_proj, x_up, mel_btc, b * t_h)
    var mbp = mel_btc.unsafe_ptr()
    var mlp = out_mel_buf.unsafe_ptr()
    var mel_dim = enc.mel

    @always_inline
    @parameter
    @__copy_capture(mbp, mlp, t_h, mel_dim)
    def trans_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (mel_dim * t_h)
        var rem = i - bi * mel_dim * t_h
        var ci = rem // t_h
        var ti = rem - ci * t_h
        mlp[i] = mbp[bi * t_h * mel_dim + ti * mel_dim + ci]
    elementwise[trans_func, simd_width=1, target="gpu"](
        IndexList[1](b * mel_dim * t_h), DeviceContextPtr(ctx),
    )


@fieldwise_init
struct S3Gen(Copyable, Movable):
    """Full s3gen wrapper."""
    var encoder: UpsampleConformerEncoder
    var cfm_estimator: CFMEstimator
    var hifigan: HiFiGAN


def s3gen_synthesize(
    mut ctx: DeviceContext,
    mut model: S3Gen,
    mut speech_tokens_buf: DeviceBuffer[DType.int64],
    mut spks_buf: DeviceBuffer[DType.float32],
    mut prompt_cond_buf: DeviceBuffer[DType.float32],   # (B, mel, T_h) CFM prompt cond
    mut prompt_mask_buf: DeviceBuffer[DType.float32],
    mut z_init_buf: DeviceBuffer[DType.float32],        # Gaussian noise
    mut packed_buf: DeviceBuffer[DType.float32],
    mut v_buf: DeviceBuffer[DType.float32],
    mut v_cfg_buf: DeviceBuffer[DType.float32],
    mut t_emb_buf: DeviceBuffer[DType.float32],
    mut cos_buf: DeviceBuffer[DType.float32],
    mut sin_buf: DeviceBuffer[DType.float32],
    mut mask_buf: DeviceBuffer[DType.float32],
    mut encoder_mel_buf: DeviceBuffer[DType.float32],   # encoder output
    mut audio_out_buf: DeviceBuffer[DType.float32],
    b: Int, t_token: Int, mel: Int, n_steps: Int, cfg_rate: Float32,
    time_emb_dim: Int, c_concat: Int,
    pre_c_out: Int,
    stage_dims: List[Int],
    audio_len: Int,
) raises:
    """End-to-end s3gen: speech_tokens → mel → audio."""
    # 1. Encoder.
    var t_h = t_token * 2
    upsample_conformer_forward(
        ctx, model.encoder, speech_tokens_buf, encoder_mel_buf,
        cos_buf, sin_buf, mask_buf, b, t_token,
    )
    # 2. CFM Euler solve (using encoder_mel as `cond_buf` is the upstream convention).
    cfm_euler_solve(
        ctx, model.cfm_estimator, z_init_buf, spks_buf,
        encoder_mel_buf, prompt_mask_buf,
        packed_buf, v_buf, v_cfg_buf, t_emb_buf,
        b, mel, t_h, c_concat, n_steps, cfg_rate, time_emb_dim,
    )
    # 3. HiFiGAN.
    hifigan_forward(
        ctx, model.hifigan, z_init_buf, audio_out_buf,
        b, mel, t_h, pre_c_out, stage_dims, audio_len,
    )
