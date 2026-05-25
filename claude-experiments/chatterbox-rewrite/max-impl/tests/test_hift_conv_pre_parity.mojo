"""Isolate just conv_pre forward."""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt

from fixture import load_fp32
from weights import load_hift_generator, upload_fp32
from conv1d import Conv1d, conv1d_forward


comptime B = 1
comptime T_MEL = 4
comptime MEL = 80
comptime BASE = 512


def test_conv_pre_parity() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var hift = load_hift_generator(ctx, "weights/s3gen/mel2wav")

    var mel = upload_fp32(ctx, "weights/hift_parity/mel.bin")
    var expected = load_fp32("weights/hift_parity/conv_pre_out.bin")

    var out = ctx.enqueue_create_buffer[DType.float32](B * BASE * T_MEL)
    conv1d_forward(ctx, hift.conv_pre, mel, out, B, T_MEL, T_MEL)
    ctx.synchronize()

    var n = B * BASE * T_MEL
    var max_abs: Float32 = 0.0
    var sum_diff_sq: Float32 = 0.0
    var sum_ref_sq: Float32 = 0.0
    with out.map_to_host() as h:
        for i in range(n):
            var d = h[i] - expected.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            var dd = h[i] - expected.data[i]
            sum_diff_sq += dd * dd
            sum_ref_sq += expected.data[i] * expected.data[i]
        for i in range(8):
            print("  [", i, "] got=", h[i], " want=", expected.data[i],
                  " diff=", h[i] - expected.data[i])
    var rel_l2 = sqrt(sum_diff_sq / sum_ref_sq) if sum_ref_sq > 0.0 else Float32(0.0)
    print("[conv_pre] max-abs =", max_abs, " rel_l2 =", rel_l2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
