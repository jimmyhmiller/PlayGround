"""Isolate the time-embedding path and compare against upstream.

Tests just `SinusoidalPosEmb + time_mlp` (linear_1 → silu → linear_2) given
t_scalar=0.5 → expected shape (1, 1024).
"""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt

from fixture import load_fp32
from weights import load_cfm_estimator_real, upload_fp32
from cfm_estimator_new import time_embedding_forward


comptime B = 1
comptime IN_CH = 320
comptime TIME_DIM = 1024


def test_cfm_time_emb_parity() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var cfm = load_cfm_estimator_real(ctx, "weights/s3gen/flow/decoder/estimator")
    var t_scalar = upload_fp32(ctx, "weights/cfm_parity/t_scalar.bin")
    var expected = load_fp32("weights/cfm_parity/time_mlp_out.bin")

    var t_emb = ctx.enqueue_create_buffer[DType.float32](B * TIME_DIM)
    time_embedding_forward(
        ctx, cfm.time_mlp1, cfm.time_mlp2, t_scalar, t_emb,
        B, IN_CH, TIME_DIM,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    var sum_diff_sq: Float32 = 0.0
    var sum_ref_sq: Float32 = 0.0
    var n = B * TIME_DIM
    with t_emb.map_to_host() as h:
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
    print("[time-emb] max-abs diff =", max_abs, " rel_l2 =", rel_l2)
    assert_true(max_abs < 0.01, "time-emb should match")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
