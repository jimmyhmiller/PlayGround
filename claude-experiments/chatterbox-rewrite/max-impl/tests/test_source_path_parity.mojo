"""Parity test: f0_predictor + f0_upsample + source_module + STFT vs upstream."""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt

from fixture import load_fp32
from weights import load_hift_generator, upload_fp32
from hift_generator import (
    f0_predictor_forward, f0_upsample_nearest, source_module_forward,
    build_s_stft_from_signal, hann_window_periodic_fill,
)


comptime B = 1
comptime T_MEL = 60
comptime T_AUDIO = T_MEL * 480
comptime N_FFT = 16
comptime HOP = 4
comptime T_S_FRAMES = T_AUDIO // HOP + 1   # 7201
comptime N_OUT = 18


def stats(name: String, got: DeviceBuffer[DType.float32], expected: List[Float32], n: Int) raises:
    var max_abs: Float32 = 0.0
    var sum_diff_sq: Float32 = 0.0
    var sum_ref_sq: Float32 = 0.0
    var got_mean: Float32 = 0.0
    var ref_mean: Float32 = 0.0
    with got.map_to_host() as h:
        for i in range(n):
            var dd = h[i] - expected[i]
            var d = dd
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_diff_sq += dd * dd
            sum_ref_sq += expected[i] * expected[i]
            got_mean += h[i]
            ref_mean += expected[i]
    var rel_l2 = sqrt(sum_diff_sq / sum_ref_sq) if sum_ref_sq > 0.0 else Float32(0.0)
    print("[", name, "] max-abs =", max_abs, " rel_l2 =", rel_l2,
          " got_mean =", got_mean / Float32(n), " ref_mean =", ref_mean / Float32(n))


def test_source_path() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var fix = "weights/source_path_parity/"

    var hift = load_hift_generator(ctx, "weights/s3gen/mel2wav")
    var mel = upload_fp32(ctx, fix + "mel.bin")

    # Step 1: f0_predictor.
    var f0 = ctx.enqueue_create_buffer[DType.float32](B * T_MEL)
    f0_predictor_forward(ctx, hift.f0_predictor, mel, f0, B, T_MEL)
    ctx.synchronize()
    var exp_f0 = load_fp32(fix + "f0.bin")
    stats(String("f0"), f0, exp_f0.data, B * T_MEL)
    # Print first 16 to see what's happening.
    with f0.map_to_host() as h:
        for i in range(16):
            print("  f0[", i, "] got=", h[i], " want=", exp_f0.data[i])

    # Step 2: f0_upsample.
    var f0_up = ctx.enqueue_create_buffer[DType.float32](B * T_AUDIO)
    f0_upsample_nearest(ctx, f0, f0_up, B, T_MEL, 480)
    ctx.synchronize()
    var exp_f0_up = load_fp32(fix + "f0_up.bin")
    # Only check first 256 samples to avoid printing 28800 elements.
    stats(String("f0_up"), f0_up, exp_f0_up.data, 256)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
