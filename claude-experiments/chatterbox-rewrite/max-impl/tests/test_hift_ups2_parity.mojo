"""ups[2] parity: feed it the MRF-averaged stage 1 output and compare to ups2_out."""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from fixture import load_fp32
from weights import load_hift_generator, upload_fp32
from hift_generator import (
    leaky_relu_inplace, conv_transpose1d_naive, hift_resblock_forward,
)
from modules import residual_add


comptime B = 1
comptime T_AFTER_UPS1 = 160
comptime C_AFTER_UPS1 = 128


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


def test_ups2_chain() raises:
    """Compute x_after_stage_1 = MRF_avg(resblocks[3..5](ups1_out)), then ups[2]."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var hift = load_hift_generator(ctx, "weights/s3gen/mel2wav")

    var ups1 = upload_fp32(ctx, "weights/hift_parity/ups1_out.bin")

    var x_sum = ctx.enqueue_create_buffer[DType.float32](B * C_AFTER_UPS1 * T_AFTER_UPS1)
    x_sum.enqueue_fill(0.0)
    for j in range(3):
        var xs = ctx.enqueue_create_buffer[DType.float32](B * C_AFTER_UPS1 * T_AFTER_UPS1)
        ctx.enqueue_copy(xs, ups1)
        hift_resblock_forward(ctx, hift.resblocks[3 + j], xs, B, T_AFTER_UPS1)
        residual_add(ctx, x_sum, xs, B * C_AFTER_UPS1 * T_AFTER_UPS1)
    var inv: Float32 = 1.0 / 3.0
    var p = x_sum.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(p, inv)
    def avg[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        p[i] = p[i] * inv
    elementwise[avg, simd_width=1, target="gpu"](
        IndexList[1](B * C_AFTER_UPS1 * T_AFTER_UPS1), DeviceContextPtr(ctx),
    )

    leaky_relu_inplace(ctx, x_sum, B * C_AFTER_UPS1 * T_AFTER_UPS1, Float32(0.1))

    # ups[2]: 128 → 64, K=7, stride=3, pad=(7-3)//2=2.
    var T_AFTER_UPS2 = T_AFTER_UPS1 * 3   # 480
    var C_AFTER_UPS2 = 64
    var out = ctx.enqueue_create_buffer[DType.float32](B * C_AFTER_UPS2 * T_AFTER_UPS2)
    conv_transpose1d_naive(
        ctx, x_sum, hift.ups[2].weight, hift.ups[2].bias, out,
        B, C_AFTER_UPS1, C_AFTER_UPS2, 7, 3, 2, T_AFTER_UPS1, T_AFTER_UPS2,
    )
    ctx.synchronize()

    var expected = load_fp32("weights/hift_parity/ups2_out.bin")
    stats(String("ups2"), out, expected.data, B * C_AFTER_UPS2 * T_AFTER_UPS2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
