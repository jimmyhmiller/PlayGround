"""Run Mojo HiFT stage-by-stage on upstream's exact mel; diff each stage
against the dumped upstream intermediate. Localizes the divergence.
"""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.math import sqrt

from fixture import load_fp32
from weights import load_hift_generator, upload_fp32
from hift_generator import (
    hift_decode_trunk, istft_forward, hann_window_periodic_fill,
    f0_predictor_forward, f0_upsample_nearest, source_module_forward,
    source_module_forward_deterministic, build_s_stft_from_signal,
)


def _diff(name: String, mut mojo: DeviceBuffer[DType.float32], ref_path: String) raises:
    var reference = load_fp32(ref_path)
    var ref_n = reference.numel()
    var max_abs: Float32 = 0.0
    var sum_diff_sq: Float32 = 0.0
    var sum_ref_sq: Float32 = 0.0
    with mojo.map_to_host() as h:
        for i in range(ref_n):
            var dd = h[i] - reference.data[i]
            var d = dd
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_diff_sq += dd * dd
            sum_ref_sq += reference.data[i] * reference.data[i]
    var rel = sqrt(sum_diff_sq / sum_ref_sq) if sum_ref_sq > 0.0 else Float32(0.0)
    print("[hift parity]", name, ": max-abs=", max_abs, " rel_l2=", rel, " (n=", ref_n, ")")


def test_hift_stages() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var fix = "weights/s3gen_prompt/hift_dump/"

    var hift = load_hift_generator(ctx, "weights/s3gen/mel2wav")

    # Use the bit-exact mel as input.
    var mel = upload_fp32(ctx, "weights/s3gen_prompt/expected_mel.bin")
    var T = 102
    var B = 1
    var MEL = 80
    var HOP = 4

    var T_AUDIO_FULL = T * 480
    var T_S_FRAMES = T_AUDIO_FULL // HOP + 1

    # Stage 1: f0_predictor.
    var f0 = ctx.enqueue_create_buffer[DType.float32](B * T)
    f0_predictor_forward(ctx, hift.f0_predictor, mel, f0, B, T)
    ctx.synchronize()
    _diff("f0", f0, fix + "f0.bin")

    # Stage 2: f0_upsample.
    var f0_up = ctx.enqueue_create_buffer[DType.float32](B * T_AUDIO_FULL)
    f0_upsample_nearest(ctx, f0, f0_up, B, T, 480)
    ctx.synchronize()
    # (no upstream ref for this — skip)

    # Stage 3: source_module — deterministic variant using upstream's phase_vec + noise.
    var phase_vec = upload_fp32(ctx, fix + "sg_phase_vec.bin")    # (1, 9, 1)
    var noise_buf = upload_fp32(ctx, fix + "sg_noise.bin")        # (1, 9, T_AUDIO_FULL)
    var sine_merge = ctx.enqueue_create_buffer[DType.float32](B * 1 * T_AUDIO_FULL)
    source_module_forward_deterministic(
        ctx, hift.m_source, f0_up, phase_vec, noise_buf, sine_merge,
        B, T_AUDIO_FULL,
        sampling_rate=24000, harmonic_num=8,
        sine_amp=Float32(0.1), voiced_threshold=Float32(10.0),
    )
    ctx.synchronize()
    _diff("sine_merge (deterministic)", sine_merge, fix + "sine_merge.bin")

    # Stage 4: STFT → s_stft (18, T_S_FRAMES) using our deterministic sine_merge.
    var window_s = ctx.enqueue_create_buffer[DType.float32](16)
    hann_window_periodic_fill(ctx, window_s, 16)
    var s_stft = ctx.enqueue_create_buffer[DType.float32](B * 18 * T_S_FRAMES)
    build_s_stft_from_signal(ctx, sine_merge, window_s, s_stft,
                              B, T_AUDIO_FULL, 16, HOP, T_S_FRAMES)
    ctx.synchronize()
    _diff("s_stft", s_stft, fix + "s_stft.bin")

    # Stage 5: hift_decode_trunk → conv_post output (18, T_HIFT_FRAMES).
    var T_HIFT = T * 120 + 1
    var spec = ctx.enqueue_create_buffer[DType.float32](B * 18 * T_HIFT)
    hift_decode_trunk(ctx, hift, mel, s_stft, spec,
                      B, T, T_S_FRAMES, T_HIFT, use_source=True)
    ctx.synchronize()
    _diff("conv_post_out", spec, fix + "conv_post_out.bin")

    # Stage 6: iSTFT → audio.
    var window = ctx.enqueue_create_buffer[DType.float32](16)
    hann_window_periodic_fill(ctx, window, 16)
    var T_AUDIO = (T_HIFT - 1) * HOP
    var audio = ctx.enqueue_create_buffer[DType.float32](B * T_AUDIO)
    istft_forward(ctx, spec, window, audio, B, 16, T_HIFT, T_AUDIO)
    ctx.synchronize()
    _diff("audio", audio, fix + "audio.bin")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
