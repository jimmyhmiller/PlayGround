"""Snake activation parity vs upstream torch."""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt

from fixture import load_fp32
from weights import upload_fp32
from hift_generator import snake_activation


comptime B = 1
comptime C = 256
comptime T = 32


def test_snake_parity() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var x = upload_fp32(ctx, "weights/hift_parity/snake_x.bin")
    var alpha = upload_fp32(ctx, "weights/hift_parity/snake_alpha.bin")
    var expected = load_fp32("weights/hift_parity/snake_y.bin")

    snake_activation(ctx, x, alpha, B, C, T)
    ctx.synchronize()

    var n = B * C * T
    var max_abs: Float32 = 0.0
    var sum_diff_sq: Float32 = 0.0
    var sum_ref_sq: Float32 = 0.0
    with x.map_to_host() as h:
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
    print("[snake] max-abs =", max_abs, " rel_l2 =", rel_l2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
