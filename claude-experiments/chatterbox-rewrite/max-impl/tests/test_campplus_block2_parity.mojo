"""block2 parity vs transit1_out → block2_out."""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt

from fixture import load_fp32
from weights import load_campplus, upload_fp32
from campplus import camdense_tdnn_block_forward


comptime B = 1
comptime T = 8
comptime IN_CH = 256
comptime OUT_CH = 256 + 24 * 32   # 1024


def test_block2() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var cp = load_campplus(ctx, "weights/s3gen/speaker_encoder")

    var x = upload_fp32(ctx, "weights/campplus_parity/transit1_out.bin")
    var expected = load_fp32("weights/campplus_parity/block2_out.bin")

    var out = ctx.enqueue_create_buffer[DType.float32](B * OUT_CH * T)
    # block2: kernel=3, dilation=2, pad=2
    camdense_tdnn_block_forward(
        ctx, cp.xvector.block2, x, out, B, IN_CH, T, 3, 2, 2,
    )
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
    print("[block2] max-abs =", max_abs, " rel_l2 =", rel_l2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
