"""Test multi-partial VoiceEncoder inference vs upstream's speaker_emb dump."""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.math import sqrt
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from fixture import load_fp32
from weights import load_voice_encoder, upload_fp32
from voice_encoder import voice_encoder_inference
from mel_ve import mel_ve_forward
from mel_24k import build_hann_window, build_librosa_mel_filterbank
from resampler import resample_24k_to_16k


def test_ve_inference() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var ve = load_voice_encoder(ctx, "weights/ve")

    # Load 24kHz ref wav.
    var wav24_t = load_fp32("weights/s3gen_prompt/resample_diag/wav_24k.bin")
    var n_24 = 240000   # 10s
    var wav_24 = ctx.enqueue_create_buffer[DType.float32](n_24)
    with wav_24.map_to_host() as h:
        for i in range(n_24):
            h[i] = wav24_t.data[i]

    # Resample to 16k.
    var n_16 = 160000
    var wav_16 = ctx.enqueue_create_buffer[DType.float32](n_16)
    resample_24k_to_16k(ctx, wav_24, wav_16, n_24, n_16)
    ctx.synchronize()

    # Compute VE mel.
    var win = ctx.enqueue_create_buffer[DType.float32](400)
    build_hann_window(ctx, win, 400)
    var fb = ctx.enqueue_create_buffer[DType.float32](40 * 201)
    build_librosa_mel_filterbank(ctx, fb, 40, 400, Float64(16000.0),
                                   Float64(0.0), Float64(8000.0))
    var T_ve = (n_16 + 2 * 200 - 400) // 160 + 1
    print("[ve] T_ve=", T_ve)
    var mel_ve = ctx.enqueue_create_buffer[DType.float32](T_ve * 40)
    mel_ve_forward(ctx, wav_16, win, fb, mel_ve, n_16, T_ve)
    ctx.synchronize()

    # Multi-partial inference.
    var embed = ctx.enqueue_create_buffer[DType.float32](256)
    voice_encoder_inference(ctx, ve, mel_ve, embed, T_ve)
    ctx.synchronize()

    # Compare to upstream's dumped speaker_emb.
    var reference = load_fp32("weights/s3gen_prompt/cond_enc_diag/speaker_emb.bin")
    var dot: Float32 = 0.0
    var an: Float32 = 0.0
    var bn: Float32 = 0.0
    var max_abs: Float32 = 0.0
    with embed.map_to_host() as h:
        for i in range(256):
            var dd = h[i] - reference.data[i]
            var d = dd
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            dot += h[i] * reference.data[i]
            an += h[i] * h[i]
            bn += reference.data[i] * reference.data[i]
    var cos_sim = dot / (sqrt(an) * sqrt(bn))
    print("[ve] L2 norm Mojo=", sqrt(an), " upstream=", sqrt(bn))
    print("[ve] cos_sim =", cos_sim, " max-abs diff =", max_abs)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
