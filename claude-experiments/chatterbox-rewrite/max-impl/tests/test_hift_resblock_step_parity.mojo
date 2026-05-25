"""Step-by-step inside resblock[0]:
  step 1: snake(ups0_out, alpha1[0]) == resblock0_act1_0_out
  step 2: conv1[0](snake_out) == resblock0_conv1_0_out
"""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt

from fixture import load_fp32
from weights import load_hift_generator, upload_fp32
from conv1d import Conv1d, conv1d_forward
from hift_generator import snake_activation


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
        for i in range(8):
            print("  [", i, "] got=", h[i], " want=", expected[i])
    var rel_l2 = sqrt(sum_diff_sq / sum_ref_sq) if sum_ref_sq > 0.0 else Float32(0.0)
    print("[", name, "] max-abs =", max_abs, " rel_l2 =", rel_l2)


def test_resblock_steps() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var hift = load_hift_generator(ctx, "weights/s3gen/mel2wav")

    var ups0 = upload_fp32(ctx, "weights/hift_parity/ups0_out.bin")

    # Step 1: snake.
    var x = ctx.enqueue_create_buffer[DType.float32](B * C * T)
    ctx.enqueue_copy(x, ups0)
    snake_activation(ctx, x, hift.resblocks[0].activations1[0].alpha, B, C, T)
    ctx.synchronize()

    var expected_snake = load_fp32("weights/hift_parity/resblock0_act1_0_out.bin")
    stats(String("snake1[0]"), x, expected_snake.data, B * C * T)

    # Step 2: conv1[0] (dilated k=3).
    var y = ctx.enqueue_create_buffer[DType.float32](B * C * T)
    conv1d_forward(ctx, hift.resblocks[0].convs1[0], x, y, B, T, T)
    ctx.synchronize()

    var expected_conv = load_fp32("weights/hift_parity/resblock0_conv1_0_out.bin")
    stats(String("conv1[0]"), y, expected_conv.data, B * C * T)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
