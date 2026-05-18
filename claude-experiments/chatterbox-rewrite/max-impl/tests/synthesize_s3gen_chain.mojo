"""End-to-end s3gen forward chain on real upstream weights.

Takes pre-generated speech token IDs (from T3) + speaker embedding +
initial noise + mask and produces audio samples via:

  speech_tokens (B, T_token) → UpsampleConformerEncoder → mu (B, 80, 2*T_token)
  z (random) + mu + spks + cond + mask → CFM Euler solve → mel (B, 80, 2*T_token)
  mel → HiFTGenerator (NSF-HiFiGAN trunk + iSTFT) → audio (B, T_audio)

Uses small T_token to keep memory in check. With T_token=4, T_mel=8,
T_audio = 8 * 256 + reflection = 2049 samples (~0.085s at 24kHz).

This is the full s3gen path. The FCM 2D head and BPE tokenizer are still
external (T3 tokens are provided, spk_emb is provided).
"""
from std.sys import has_accelerator
from std.math import sin
from std.testing import TestSuite, assert_true
from std.gpu.host import DeviceContext

from weights import (
    load_upsample_conformer_encoder, load_cfm_estimator_real,
    load_hift_generator,
)
from upsample_encoder import upsample_conformer_forward
from cfm_estimator_new import cfm_solve_euler
from hift_generator import (
    hift_decode_trunk, istft_forward, hann_window_periodic_fill,
)
from fixture import save_wav


comptime B = 1
comptime T_TOKEN = 4
comptime T_MEL = 2 * T_TOKEN   # 8
comptime MEL = 80
# HiFT: T_mel = 8 → ups [8,8,4] → 2048 → +1 reflection → 2049
comptime T_HIFT_FRAMES = T_MEL * 8 * 8 * 4 + 1
comptime N_FFT = 16
comptime HOP = 4
comptime T_AUDIO = (T_HIFT_FRAMES - 1) * HOP
comptime N_OUT = 18
comptime N_CFM_STEPS = 2
comptime CFG: Float32 = 0.7


def test_synthesize_s3gen_chain() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    print("[e2e] loading 3 s3gen models...")
    var enc = load_upsample_conformer_encoder(ctx, "weights/s3gen/flow")
    var cfm = load_cfm_estimator_real(ctx, "weights/s3gen/flow/decoder/estimator")
    var hift = load_hift_generator(ctx, "weights/s3gen/mel2wav")
    print("[e2e] all models loaded.")

    # 1. Speech tokens (B, T_TOKEN) — pretend these came from T3.
    var token_ids = ctx.enqueue_create_buffer[DType.int64](B * T_TOKEN)
    with token_ids.map_to_host() as h:
        for i in range(T_TOKEN):
            h[i] = Int64((i * 137 + 42) % 6562)

    # 2. Speaker embedding (B, MEL) — pretend from CAMPPlus → spk_embed_affine_layer.
    var spks = ctx.enqueue_create_buffer[DType.float32](B * MEL)
    with spks.map_to_host() as h:
        for c in range(MEL):
            h[c] = sin(Float32(c) * 0.11) * 0.1

    # 3. Run flow encoder: tokens → mu (B, MEL, T_MEL).
    var mu = ctx.enqueue_create_buffer[DType.float32](B * MEL * T_MEL)
    print("[e2e] step 1/3: flow encoder...")
    upsample_conformer_forward(ctx, enc, token_ids, mu, B, T_TOKEN)
    ctx.synchronize()

    # 4. CFM Euler: noise → mel.
    var x = ctx.enqueue_create_buffer[DType.float32](B * MEL * T_MEL)
    # Deterministic pseudo-noise.
    with x.map_to_host() as h:
        for c in range(MEL):
            for ti in range(T_MEL):
                h[c * T_MEL + ti] = sin(Float32(c) * 0.31 + Float32(ti) * 0.71) * 0.5
    var cond = ctx.enqueue_create_buffer[DType.float32](B * MEL * T_MEL)
    cond.enqueue_fill(0.0)  # No prompt cond.
    var mask = ctx.enqueue_create_buffer[DType.float32](B * T_MEL)
    mask.enqueue_fill(1.0)
    print("[e2e] step 2/3: CFM Euler solve (", N_CFM_STEPS, " steps)...")
    cfm_solve_euler(ctx, cfm, x, mu, spks, cond, mask, B, T_MEL, N_CFM_STEPS, CFG)
    ctx.synchronize()

    # 5. HiFT trunk + iSTFT → audio.
    var s_stft_dummy = ctx.enqueue_create_buffer[DType.float32](B * N_OUT * 1)
    var spec = ctx.enqueue_create_buffer[DType.float32](B * N_OUT * T_HIFT_FRAMES)
    print("[e2e] step 3/3: HiFT trunk (no source fusion)...")
    hift_decode_trunk(ctx, hift, x, s_stft_dummy, spec,
                      B, T_MEL, 1, T_HIFT_FRAMES, use_source=False)

    var window = ctx.enqueue_create_buffer[DType.float32](N_FFT)
    hann_window_periodic_fill(ctx, window, N_FFT)
    var audio = ctx.enqueue_create_buffer[DType.float32](B * T_AUDIO)
    print("[e2e] iSTFT (", T_HIFT_FRAMES, " frames → ", T_AUDIO, " samples)...")
    istft_forward(ctx, spec, window, audio, B, N_FFT, T_HIFT_FRAMES, T_AUDIO)
    ctx.synchronize()

    # Audio stats.
    var n_nan = 0
    var sum_abs: Float32 = 0.0
    var max_a: Float32 = 0.0
    with audio.map_to_host() as h:
        for i in range(B * T_AUDIO):
            var v = h[i]
            if v != v: n_nan += 1
            var av = v
            if av < 0.0: av = -av
            sum_abs += av
            if av > max_a: max_a = av
    print("[e2e] audio: mean-abs=", sum_abs / Float32(B * T_AUDIO),
          " max=", max_a, " nan=", n_nan, " duration ~",
          Float32(T_AUDIO) / 24000.0, "s @ 24kHz")
    assert_true(n_nan == 0, "no NaNs in audio output")

    # Save the audio to disk as a tangible artifact (won't be meaningful speech
    # without real T3 tokens + matched speaker emb, but it confirms the full
    # forward graph runs end-to-end and produces valid PCM samples).
    var samples = List[Float32]()
    with audio.map_to_host() as h:
        for i in range(T_AUDIO):
            samples.append(h[i])
    save_wav(String("max_impl_s3gen_chain.wav"), samples, 24000)
    print("[e2e] saved max_impl_s3gen_chain.wav")
    print("[e2e] FULL S3GEN CHAIN PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
