"""resblocks[6] parity test against reflection_pad output."""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt

from fixture import load_fp32
from weights import load_hift_generator, upload_fp32
from hift_generator import hift_resblock_forward


comptime B = 1
comptime C = 64
comptime T = 481


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
        for i in range(4):
            print("  [", i, "] got=", h[i], " want=", expected[i])
    var rel_l2 = sqrt(sum_diff_sq / sum_ref_sq) if sum_ref_sq > 0.0 else Float32(0.0)
    print("[", name, "] max-abs =", max_abs, " rel_l2 =", rel_l2)


def test_resblock6() raises:
    var ctx = DeviceContext()
    var hift = load_hift_generator(ctx, "weights/s3gen/mel2wav")

    var x = upload_fp32(ctx, "weights/hift_parity/reflection_pad_out.bin")
    hift_resblock_forward(ctx, hift.resblocks[6], x, B, T)
    ctx.synchronize()

    var expected = load_fp32("weights/hift_parity/resblock6_out.bin")
    stats(String("resblock6"), x, expected.data, B * C * T)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
