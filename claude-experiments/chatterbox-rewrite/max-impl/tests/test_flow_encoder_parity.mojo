"""Full UpsampleConformerEncoder parity test vs upstream torch oracle."""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt

from fixture import load_fp32, load_i64
from weights import load_upsample_conformer_encoder
from upsample_encoder import upsample_conformer_forward


comptime B = 1
comptime T_IN = 4
comptime T_UP = 2 * T_IN
comptime MEL = 80


def test_flow_encoder_parity() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var enc = load_upsample_conformer_encoder(ctx, "weights/s3gen/flow")

    var tok_data = load_i64("weights/flow_enc_parity/token_ids.bin")
    var token_ids = ctx.enqueue_create_buffer[DType.int64](B * T_IN)
    with token_ids.map_to_host() as h:
        for i in range(T_IN):
            h[i] = tok_data.data[i]

    var expected = load_fp32("weights/flow_enc_parity/expected_mu.bin")
    print("[flow-enc] expected mu shape: (B, 80,", T_UP, ")")

    var mu_out = ctx.enqueue_create_buffer[DType.float32](B * MEL * T_UP)
    upsample_conformer_forward(ctx, enc, token_ids, mu_out, B, T_IN)
    ctx.synchronize()

    var n = B * MEL * T_UP
    var max_abs: Float32 = 0.0
    var sum_diff_sq: Float32 = 0.0
    var sum_ref_sq: Float32 = 0.0
    with mu_out.map_to_host() as h:
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
    print("[flow-enc] max-abs diff =", max_abs, " rel_l2 =", rel_l2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
