"""reflection_pad((1, 0)) parity: input ups2_out (1, 64, 480) → output (1, 64, 481)."""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt

from fixture import load_fp32
from weights import upload_fp32
from hift_generator import reflection_pad1_right


comptime B = 1
comptime C = 64
comptime T_IN = 480
comptime T_OUT = 481


def test_reflection_pad() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var ups2 = upload_fp32(ctx, "weights/hift_parity/ups2_out.bin")
    var expected = load_fp32("weights/hift_parity/reflection_pad_out.bin")

    var out = ctx.enqueue_create_buffer[DType.float32](B * C * T_OUT)
    reflection_pad1_right(ctx, ups2, out, B, C, T_IN)
    ctx.synchronize()

    var n = B * C * T_OUT
    var max_abs: Float32 = 0.0
    with out.map_to_host() as h:
        for i in range(n):
            var d = h[i] - expected.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
        # Check first 4 of each: position 0, 1, 480-1, 480.
        print("[reflect] out[0] =", h[0], " expected =", expected.data[0])
        print("[reflect] out[1] =", h[1], " expected =", expected.data[1])
        print("[reflect] ups2[0] (should equal out[1]) =", h[1])
        print("[reflect] out[480] =", h[480], " expected =", expected.data[480])
    print("[reflect] max-abs =", max_abs)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
