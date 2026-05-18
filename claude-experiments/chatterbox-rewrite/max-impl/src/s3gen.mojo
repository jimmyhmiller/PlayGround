"""s3gen flow: speech_tokens → encoder features → CFM → HiFiGAN → audio.

This module composes the pieces:
  1. UpsampleConformerEncoder: input_embedding + N conformer blocks + 4 more
     conformer blocks at upsampled rate + encoder_proj Linear.
  2. CFM ConditionalDecoder + 10-step Euler ODE solver: estimator U-Net with
     down/mid/up + transformer blocks; produces mel spectrogram.
  3. HiFiGAN vocoder: upsampling Conv1d + ResBlock stack + Conv1d post → audio.
  4. SourceModuleHnNSF: F0 predictor + harmonic + noise → STFT source signal
     concatenated with mel into HiFiGAN.

All blocks delegate to MAX primitives (`linalg.matmul`/`bmm`, `nn.softmax`,
`nn.normalization.*`, `nn.activations.*`, `nn.rope.*`,
`elementwise[..., target="gpu"]`, `nn.gather_scatter.gather`).

Detailed per-component implementations are scaffolds we tested individually
in the previous Mojo project (mojo-t3) — here we keep just the orchestrator
shape and the MAX-abstraction call sites. The fine-grained ports of each
sub-block (e.g. each CFM U-Net resnet) are added as `_block.mojo` files in
follow-up commits.
"""
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from modules import Linear, linear_forward, Embedding, embedding_forward
from conv1d import Conv1d, conv1d_forward


@fieldwise_init
struct S3Gen(Copyable, Movable):
    """s3gen wrapper. The detailed orchestration is built incrementally; this
    struct holds the top-level layer handles and is consumed by
    `s3gen_synthesize`."""
    var speech_token_emb: Embedding         # (V_speech, D_enc) input_embedding
    var encoder_proj: Linear                # (80, D_enc)
    var post_conv: Conv1d                   # CFM cond projector
    var hifigan_pre: Conv1d                 # HiFiGAN entry conv
    var hifigan_post: Conv1d                # HiFiGAN final conv
    var d_enc: Int
    var n_mels: Int


def s3gen_synthesize_stub(
    mut ctx: DeviceContext,
    mut model: S3Gen,
    mut speech_tokens_buf: DeviceBuffer[DType.int64],   # (B, T_token)
    mut spk_emb_buf: DeviceBuffer[DType.float32],       # (B, 80) CFM speaker
    mut prompt_feat_buf: DeviceBuffer[DType.float32],   # (B, T_prompt, 80) CFM cond
    mut audio_out_buf: DeviceBuffer[DType.float32],     # (B, audio_len) PCM
    b: Int, t_token: Int,
) raises:
    """End-to-end s3gen forward stub. Wires the top-level passes.

    Detailed CFM/encoder/HiFiGAN bodies are added in follow-up commits, each
    building on `modules.*` + `conv1d.*` + `attention.*` (no hand kernels).
    """
    # 1. Embed speech tokens via input_embedding.
    var x = ctx.enqueue_create_buffer[DType.float32](b * t_token * model.d_enc)
    embedding_forward(ctx, model.speech_token_emb, speech_tokens_buf, x,
                       b, t_token)

    # 2. UpsampleConformerEncoder (deferred — see encoder.mojo for blocks).
    # 3. CFM Euler solver (deferred).
    # 4. HiFiGAN vocoder (deferred).
    pass    # final audio is written via the deferred modules in their respective files.
