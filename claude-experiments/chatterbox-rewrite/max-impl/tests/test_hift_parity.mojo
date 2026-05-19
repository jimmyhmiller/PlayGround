"""HiFTGenerator parity test vs upstream torch oracle (zero source path).

Compares both:
  - conv_post output (pre-iSTFT) at (B, 18, T_pre)
  - final audio (post-iSTFT + clamp) at (B, T_audio)
"""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt

from fixture import load_fp32
from weights import load_hift_generator, upload_fp32
from hift_generator import (
    hift_decode_trunk, istft_forward, hann_window_periodic_fill,
)


comptime B = 1
comptime T_MEL = 4
# ups [8, 5, 3] = 120; +1 reflection on last stage.
comptime T_PRE = T_MEL * 120 + 1   # 481
comptime N_FFT = 16
comptime HOP = 4
comptime T_AUDIO = (T_PRE - 1) * HOP    # 1920
comptime N_OUT = 18
comptime MEL = 80


def stats(name: String, got: DeviceBuffer[DType.float32], expected: List[Float32], n: Int) raises:
    var max_abs: Float32 = 0.0
    var sum_diff_sq: Float32 = 0.0
    var sum_ref_sq: Float32 = 0.0
    with got.map_to_host() as h:
        for i in range(n):
            var d = h[i] - expected[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            var dd = h[i] - expected[i]
            sum_diff_sq += dd * dd
            sum_ref_sq += expected[i] * expected[i]
        for i in range(8):
            print("  [", i, "] got=", h[i], " want=", expected[i],
                  " diff=", h[i] - expected[i])
    var rel_l2 = sqrt(sum_diff_sq / sum_ref_sq) if sum_ref_sq > 0.0 else Float32(0.0)
    print("[", name, "] max-abs =", max_abs, " rel_l2 =", rel_l2)


def test_hift_parity() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var hift = load_hift_generator(ctx, "weights/s3gen/mel2wav")

    var mel = upload_fp32(ctx, "weights/hift_parity/mel.bin")
    var s_stft_dummy = ctx.enqueue_create_buffer[DType.float32](B * N_OUT * 1)

    var spec_out = ctx.enqueue_create_buffer[DType.float32](B * N_OUT * T_PRE)
    print("[hift-parity] running decode trunk (T_mel=", T_MEL, " → T_pre=", T_PRE, ")...")
    hift_decode_trunk(ctx, hift, mel, s_stft_dummy, spec_out,
                      B, T_MEL, 1, T_PRE, use_source=False)
    ctx.synchronize()

    var expected_spec = load_fp32("weights/hift_parity/expected_spec.bin")
    stats(String("spec"), spec_out, expected_spec.data, B * N_OUT * T_PRE)

    var window = ctx.enqueue_create_buffer[DType.float32](N_FFT)
    hann_window_periodic_fill(ctx, window, N_FFT)
    var audio = ctx.enqueue_create_buffer[DType.float32](B * T_AUDIO)
    istft_forward(ctx, spec_out, window, audio, B, N_FFT, T_PRE, T_AUDIO)
    ctx.synchronize()

    var expected_audio = load_fp32("weights/hift_parity/expected_audio.bin")
    stats(String("audio"), audio, expected_audio.data, B * T_AUDIO)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
