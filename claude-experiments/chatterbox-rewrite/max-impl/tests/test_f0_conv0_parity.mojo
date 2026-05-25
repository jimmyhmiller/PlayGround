"""f0_predictor.condnet[0] (first conv) parity vs upstream."""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt

from fixture import load_fp32
from weights import load_hift_generator, upload_fp32
from conv1d import conv1d_forward


comptime B = 1
comptime T_MEL = 60
comptime C_OUT = 512


def test_conv0() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var hift = load_hift_generator(ctx, "weights/s3gen/mel2wav")

    var mel = upload_fp32(ctx, "weights/source_path_parity/mel.bin")
    var expected = load_fp32("weights/source_path_parity/conv0.bin")

    var out = ctx.enqueue_create_buffer[DType.float32](B * C_OUT * T_MEL)
    conv1d_forward(ctx, hift.f0_predictor.condnet[0], mel, out, B, T_MEL, T_MEL)
    ctx.synchronize()

    var n = B * C_OUT * T_MEL
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
            print("  [", i, "] got=", h[i], " want=", expected.data[i])
    var rel_l2 = sqrt(sum_diff_sq / sum_ref_sq) if sum_ref_sq > 0.0 else Float32(0.0)
    print("[conv0] max-abs =", max_abs, " rel_l2 =", rel_l2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
