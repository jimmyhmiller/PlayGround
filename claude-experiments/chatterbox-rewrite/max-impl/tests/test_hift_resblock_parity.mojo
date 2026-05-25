"""hift.resblocks[0] parity test against ups0_out → resblock0 path.

Upstream applies resblock0 to ups0 output (with source fusion if enabled).
With s=zeros, source_downs[0](s_stft) is the zero source signal mapped through
the layers. We feed ups0_out directly (skipping fusion).
"""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt

from fixture import load_fp32
from weights import load_hift_generator, upload_fp32
from hift_generator import hift_resblock_forward


comptime B = 1
comptime T = 32
comptime C = 256


def test_resblock_parity() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var hift = load_hift_generator(ctx, "weights/s3gen/mel2wav")

    var ups0 = upload_fp32(ctx, "weights/hift_parity/ups0_out.bin")
    var expected = load_fp32("weights/hift_parity/resblock0_out.bin")

    # Make a fresh copy of ups0 since resblock modifies in place.
    var x = ctx.enqueue_create_buffer[DType.float32](B * C * T)
    ctx.enqueue_copy(x, ups0)

    hift_resblock_forward(ctx, hift.resblocks[0], x, B, T)
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
    print("[resblock0] max-abs =", max_abs, " rel_l2 =", rel_l2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
