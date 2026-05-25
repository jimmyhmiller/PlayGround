"""transit1 (BN+ReLU + 1x1 Conv 512→256) parity vs torch."""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt

from fixture import load_fp32
from weights import load_campplus, upload_fp32
from campplus import transit_layer_forward


comptime B = 1
comptime T = 8
comptime IN_CH = 512
comptime OUT_CH = 256


def test_transit1() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var cp = load_campplus(ctx, "weights/s3gen/speaker_encoder")

    var x = upload_fp32(ctx, "weights/campplus_parity/block1_out.bin")
    var expected = load_fp32("weights/campplus_parity/transit1_out.bin")

    var out = ctx.enqueue_create_buffer[DType.float32](B * OUT_CH * T)
    transit_layer_forward(ctx, cp.xvector.transit1, x, out, B, IN_CH, OUT_CH, T)
    ctx.synchronize()

    var n = B * OUT_CH * T
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
        for i in range(4):
            print("  [", i, "] got=", h[i], " want=", expected.data[i])
    var rel_l2 = sqrt(sum_diff_sq / sum_ref_sq) if sum_ref_sq > 0.0 else Float32(0.0)
    print("[transit1] max-abs =", max_abs, " rel_l2 =", rel_l2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
