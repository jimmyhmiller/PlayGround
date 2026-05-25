"""block3 + transit3 + out_nonlinear + stats + dense parity."""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt

from fixture import load_fp32
from weights import load_campplus, upload_fp32
from campplus import (
    camdense_tdnn_block_forward, transit_layer_forward,
    batchnorm1d_forward, relu_inplace_bct, stats_pool_forward,
    dense_layer_forward,
)


comptime B = 1
comptime T = 8


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


def test_chain() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var cp = load_campplus(ctx, "weights/s3gen/speaker_encoder")

    # block3: transit2_out (512 ch) → 1024 ch
    var x = upload_fp32(ctx, "weights/campplus_parity/transit2_out.bin")
    var out_b3 = ctx.enqueue_create_buffer[DType.float32](B * 1024 * T)
    camdense_tdnn_block_forward(
        ctx, cp.xvector.block3, x, out_b3, B, 512, T, 3, 2, 2,
    )
    var exp_b3 = load_fp32("weights/campplus_parity/block3_out.bin")
    stats(String("block3"), out_b3, exp_b3.data, B * 1024 * T)

    # transit3: 1024 → 512
    var out_t3 = ctx.enqueue_create_buffer[DType.float32](B * 512 * T)
    transit_layer_forward(ctx, cp.xvector.transit3, out_b3, out_t3, B, 1024, 512, T)
    var exp_t3 = load_fp32("weights/campplus_parity/transit3_out.bin")
    stats(String("transit3"), out_t3, exp_t3.data, B * 512 * T)

    # out_nonlinear: BN(512) + ReLU
    var out_nl = ctx.enqueue_create_buffer[DType.float32](B * 512 * T)
    batchnorm1d_forward(ctx, cp.xvector.out_nonlinear, out_t3, out_nl, B, 512, T)
    relu_inplace_bct(ctx, out_nl, B * 512 * T)
    var exp_nl = load_fp32("weights/campplus_parity/out_nonlinear_out.bin")
    stats(String("out_nonlinear"), out_nl, exp_nl.data, B * 512 * T)

    # stats_pool → (B, 1024)
    var stats_buf = ctx.enqueue_create_buffer[DType.float32](B * 1024)
    stats_pool_forward(ctx, out_nl, stats_buf, B, 512, T)
    var exp_stats = load_fp32("weights/campplus_parity/stats_out.bin")
    stats(String("stats_pool"), stats_buf, exp_stats.data, B * 1024)

    # dense
    var dense_buf = ctx.enqueue_create_buffer[DType.float32](B * 192)
    dense_layer_forward(ctx, cp.xvector.dense, stats_buf, dense_buf, B, 1024, 192)
    var exp_dense = load_fp32("weights/campplus_parity/dense_out.bin")
    stats(String("dense"), dense_buf, exp_dense.data, B * 192)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
