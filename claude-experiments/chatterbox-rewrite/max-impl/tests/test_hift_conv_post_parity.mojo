"""conv_post parity: compute MRF stage 2 avg from reflection_pad_out,
then leaky_relu, then conv_post, compare to conv_post_out.
"""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from fixture import load_fp32
from weights import load_hift_generator, upload_fp32
from hift_generator import (
    leaky_relu_inplace, hift_resblock_forward,
)
from modules import residual_add
from conv1d import conv1d_forward


comptime B = 1
comptime C = 64
comptime T = 481
comptime OUT_C = 18


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


def test_conv_post() raises:
    var ctx = DeviceContext()
    var hift = load_hift_generator(ctx, "weights/s3gen/mel2wav")

    var x_in = upload_fp32(ctx, "weights/hift_parity/reflection_pad_out.bin")

    # MRF aggregate stage 2.
    var x_sum = ctx.enqueue_create_buffer[DType.float32](B * C * T)
    x_sum.enqueue_fill(0.0)
    for j in range(3):
        var xs = ctx.enqueue_create_buffer[DType.float32](B * C * T)
        ctx.enqueue_copy(xs, x_in)
        hift_resblock_forward(ctx, hift.resblocks[6 + j], xs, B, T)
        residual_add(ctx, x_sum, xs, B * C * T)
    var inv: Float32 = 1.0 / 3.0
    var p = x_sum.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(p, inv)
    def avg[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        p[i] = p[i] * inv
    elementwise[avg, simd_width=1, target="gpu"](
        IndexList[1](B * C * T), DeviceContextPtr(ctx),
    )

    leaky_relu_inplace(ctx, x_sum, B * C * T, Float32(0.01))

    var out = ctx.enqueue_create_buffer[DType.float32](B * OUT_C * T)
    conv1d_forward(ctx, hift.conv_post, x_sum, out, B, T, T)
    ctx.synchronize()

    var expected = load_fp32("weights/hift_parity/conv_post_out.bin")
    stats(String("conv_post"), out, expected.data, B * OUT_C * T)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
