"""Full CAMPPlus xvector parity test vs upstream torch."""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt

from fixture import load_fp32
from weights import load_campplus, upload_fp32
from campplus import xvector_forward


comptime B = 1
comptime T_IN = 16
comptime EMB = 192


def test_campplus_parity() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var cp = load_campplus(ctx, "weights/s3gen/speaker_encoder")

    var x = upload_fp32(ctx, "weights/campplus_parity/x.bin")
    var expected = load_fp32("weights/campplus_parity/expected.bin")

    var embed = ctx.enqueue_create_buffer[DType.float32](B * EMB)
    xvector_forward(ctx, cp.xvector, x, embed, B, T_IN)
    ctx.synchronize()

    var n = B * EMB
    var max_abs: Float32 = 0.0
    var sum_diff_sq: Float32 = 0.0
    var sum_ref_sq: Float32 = 0.0
    with embed.map_to_host() as h:
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
    print("[campplus] max-abs diff =", max_abs, " rel_l2 =", rel_l2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
