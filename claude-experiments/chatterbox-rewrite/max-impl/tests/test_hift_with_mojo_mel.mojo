"""Feed Mojo HiFT the exact mel that came from Mojo CFM with LCG noise
(also bit-exact to what upstream torch CFM produces given the same noise),
and check whether the resulting audio clips."""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.math import sqrt
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr

from fixture import load_fp32, save_wav
from weights import load_hift_generator, upload_fp32
from hift_generator import (
    hift_decode_trunk, istft_forward, hann_window_periodic_fill,
    f0_predictor_forward, f0_upsample_nearest, source_module_forward,
    build_s_stft_from_signal,
)


comptime B = 1
comptime MEL = 80
comptime N_FFT = 16
comptime HOP = 4
comptime N_OUT = 18


def test_hift_with_mojo_mel() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var fix = "weights/s3gen_prompt/lcg_diag/"

    var hift = load_hift_generator(ctx, "weights/s3gen/mel2wav")

    # Upload upstream's mel-from-Mojo-LCG (shape (1, 80, 102)).
    var mel = upload_fp32(ctx, fix + "upstream_mel_trim_from_mojo_lcg.bin")
    var T = 102

    var T_HIFT = T * 120 + 1
    var T_AUDIO = (T_HIFT - 1) * HOP
    var T_AUDIO_FULL = T * 480
    var T_S_FRAMES = T_AUDIO_FULL // HOP + 1

    print("[mojo-mel-hift] f0_predictor...")
    var f0 = ctx.enqueue_create_buffer[DType.float32](B * T)
    f0_predictor_forward(ctx, hift.f0_predictor, mel, f0, B, T)
    var f0_up = ctx.enqueue_create_buffer[DType.float32](B * T_AUDIO_FULL)
    f0_upsample_nearest(ctx, f0, f0_up, B, T, 480)

    print("[mojo-mel-hift] source module (LCG)...")
    var sine_merge = ctx.enqueue_create_buffer[DType.float32](B * 1 * T_AUDIO_FULL)
    source_module_forward(ctx, hift.m_source, f0_up, sine_merge,
                           B, T_AUDIO_FULL,
                           sampling_rate=24000, harmonic_num=8,
                           sine_amp=Float32(0.1), noise_std=Float32(0.003),
                           voiced_threshold=Float32(10.0))
    ctx.synchronize()

    # Print sine_merge stats.
    var sm_min: Float32 = 1e30
    var sm_max: Float32 = -1e30
    var sm_mean_abs: Float32 = 0.0
    var n_sm = B * 1 * T_AUDIO_FULL
    with sine_merge.map_to_host() as h:
        for i in range(n_sm):
            var v = h[i]
            if v < sm_min: sm_min = v
            if v > sm_max: sm_max = v
            var av = v
            if av < 0.0: av = -av
            sm_mean_abs += av
    sm_mean_abs /= Float32(n_sm)
    print("[mojo-mel-hift] sine_merge: min=", sm_min, " max=", sm_max, " mean-abs=", sm_mean_abs)

    # Also print f0 stats.
    var f0_min: Float32 = 1e30
    var f0_max: Float32 = -1e30
    var n_f0 = B * T
    with f0.map_to_host() as h:
        for i in range(n_f0):
            if h[i] < f0_min: f0_min = h[i]
            if h[i] > f0_max: f0_max = h[i]
    print("[mojo-mel-hift] f0: min=", f0_min, " max=", f0_max)

    print("[mojo-mel-hift] STFT...")
    var window_s = ctx.enqueue_create_buffer[DType.float32](N_FFT)
    hann_window_periodic_fill(ctx, window_s, N_FFT)
    var s_stft = ctx.enqueue_create_buffer[DType.float32](B * N_OUT * T_S_FRAMES)
    build_s_stft_from_signal(ctx, sine_merge, window_s, s_stft,
                              B, T_AUDIO_FULL, N_FFT, HOP, T_S_FRAMES)

    print("[mojo-mel-hift] HiFT trunk + iSTFT...")
    var spec = ctx.enqueue_create_buffer[DType.float32](B * N_OUT * T_HIFT)
    hift_decode_trunk(ctx, hift, mel, s_stft, spec,
                      B, T, T_S_FRAMES, T_HIFT, use_source=True)
    ctx.synchronize()

    var spec_max: Float32 = -1e30
    var spec_min: Float32 = 1e30
    var spec_mean: Float32 = 0.0
    var n_spec = B * N_OUT * T_HIFT
    with spec.map_to_host() as h:
        for i in range(n_spec):
            if h[i] > spec_max: spec_max = h[i]
            if h[i] < spec_min: spec_min = h[i]
            spec_mean += h[i]
    spec_mean /= Float32(n_spec)
    print("[mojo-mel-hift] conv_post_out stats: min=", spec_min,
          " max=", spec_max, " mean=", spec_mean)

    var window_i = ctx.enqueue_create_buffer[DType.float32](N_FFT)
    hann_window_periodic_fill(ctx, window_i, N_FFT)
    var audio = ctx.enqueue_create_buffer[DType.float32](B * T_AUDIO)
    istft_forward(ctx, spec, window_i, audio, B, N_FFT, T_HIFT, T_AUDIO)
    ctx.synchronize()

    var max_a: Float32 = 0.0
    var mean_a: Float32 = 0.0
    var n_clip = 0
    var samples = List[Float32]()
    with audio.map_to_host() as h:
        for i in range(B * T_AUDIO):
            var v = h[i]
            samples.append(v)
            var av = v
            if av < 0.0: av = -av
            if av > max_a: max_a = av
            mean_a += av
            if av > 0.95: n_clip += 1
    mean_a /= Float32(B * T_AUDIO)
    print("[mojo-mel-hift] audio max=", max_a, " mean-abs=", mean_a,
          " n(|x|>0.95)=", n_clip)
    save_wav(String("mojo_hift_from_mojo_mel.wav"), samples, 24000)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
