"""Just the first TDNN layer."""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt

from fixture import load_fp32
from weights import load_campplus, upload_fp32
from campplus import tdnn_first_forward


comptime B = 1
comptime T_IN = 16
comptime T_OUT = 8
comptime C_OUT = 128


def test_tdnn() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var cp = load_campplus(ctx, "weights/s3gen/speaker_encoder")

    var x = upload_fp32(ctx, "weights/campplus_parity/x.bin")
    var expected = load_fp32("weights/campplus_parity/tdnn_out.bin")

    var out = ctx.enqueue_create_buffer[DType.float32](B * C_OUT * T_OUT)
    tdnn_first_forward(ctx, cp.xvector.tdnn, x, out, B, T_IN, T_OUT)
    ctx.synchronize()

    var n = B * C_OUT * T_OUT
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
    print("[tdnn] max-abs =", max_abs, " rel_l2 =", rel_l2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
