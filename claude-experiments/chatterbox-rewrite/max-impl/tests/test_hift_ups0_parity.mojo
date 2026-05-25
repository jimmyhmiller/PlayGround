"""ups[0] (ConvTranspose1d) parity test. Input is conv_pre_out after
leaky_relu. Compare against upstream ups0_out.
"""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt

from fixture import load_fp32
from weights import load_hift_generator, upload_fp32
from hift_generator import conv_transpose1d_naive, leaky_relu_inplace


comptime B = 1
comptime T_MEL = 4
comptime T_AFTER = 32   # T_MEL * 8 (ups0 stride)
comptime BASE = 512
comptime OUT_CH = 256
comptime K = 16
comptime STRIDE = 8
comptime PAD = (K - STRIDE) // 2   # = 4


def test_ups0_parity() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var hift = load_hift_generator(ctx, "weights/s3gen/mel2wav")

    var conv_pre = upload_fp32(ctx, "weights/hift_parity/conv_pre_out.bin")

    # Apply leaky_relu (slope = 0.1, matches upstream).
    var x = ctx.enqueue_create_buffer[DType.float32](B * BASE * T_MEL)
    ctx.enqueue_copy(x, conv_pre)
    leaky_relu_inplace(ctx, x, B * BASE * T_MEL, Float32(0.1))

    var out = ctx.enqueue_create_buffer[DType.float32](B * OUT_CH * T_AFTER)
    conv_transpose1d_naive(
        ctx, x, hift.ups[0].weight, hift.ups[0].bias, out,
        B, BASE, OUT_CH, K, STRIDE, PAD, T_MEL, T_AFTER,
    )
    ctx.synchronize()

    var expected = load_fp32("weights/hift_parity/ups0_out.bin")
    var n = B * OUT_CH * T_AFTER
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
    print("[ups0] max-abs =", max_abs, " rel_l2 =", rel_l2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
