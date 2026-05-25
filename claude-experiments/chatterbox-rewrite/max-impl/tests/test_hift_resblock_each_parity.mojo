"""Compare each of resblocks[0..2] in isolation."""
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


def stats(name: String, got: DeviceBuffer[DType.float32], expected: List[Float32], n: Int) raises:
    var max_abs: Float32 = 0.0
    var sum_diff_sq: Float32 = 0.0
    var sum_ref_sq: Float32 = 0.0
    with got.map_to_host() as h:
        for i in range(n):
            var d = h[i] - expected[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            var dd = h[i] - expected[i]
            sum_diff_sq += dd * dd
            sum_ref_sq += expected[i] * expected[i]
    var rel_l2 = sqrt(sum_diff_sq / sum_ref_sq) if sum_ref_sq > 0.0 else Float32(0.0)
    print("[", name, "] max-abs =", max_abs, " rel_l2 =", rel_l2)


def test_each_resblock() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var hift = load_hift_generator(ctx, "weights/s3gen/mel2wav")
    var ups0 = upload_fp32(ctx, "weights/hift_parity/ups0_out.bin")

    for idx in range(3):
        var x = ctx.enqueue_create_buffer[DType.float32](B * C * T)
        ctx.enqueue_copy(x, ups0)
        hift_resblock_forward(ctx, hift.resblocks[idx], x, B, T)
        ctx.synchronize()
        var name = String("resblock") + String(idx)
        var exp = load_fp32("weights/hift_parity/" + name + "_out.bin")
        stats(name, x, exp.data, B * C * T)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
