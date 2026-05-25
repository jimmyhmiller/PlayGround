"""Isolate the first down-block resnet (Resnet1D 320→256) parity test."""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from fixture import load_fp32
from weights import load_cfm_estimator_real, upload_fp32
from cfm_estimator_new import (
    time_embedding_forward, resnet1d_forward, channel_concat_bct,
    broadcast_spks_to_t,
)


comptime B = 1
comptime T = 16
comptime MEL = 80
comptime IN_CH = 320
comptime D = 256
comptime TIME_DIM = 1024


def test_cfm_down0_resnet_parity() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var cfm = load_cfm_estimator_real(ctx, "weights/s3gen/flow/decoder/estimator")

    var x = upload_fp32(ctx, "weights/cfm_parity/x.bin")
    var mu = upload_fp32(ctx, "weights/cfm_parity/mu.bin")
    var spks = upload_fp32(ctx, "weights/cfm_parity/spks.bin")
    var cond = upload_fp32(ctx, "weights/cfm_parity/cond.bin")
    var mask = upload_fp32(ctx, "weights/cfm_parity/mask.bin")
    var t_scalar = upload_fp32(ctx, "weights/cfm_parity/t_scalar.bin")
    var expected = load_fp32("weights/cfm_parity/down0_resnet_out.bin")

    # Build time embedding.
    var t_emb = ctx.enqueue_create_buffer[DType.float32](B * TIME_DIM)
    time_embedding_forward(
        ctx, cfm.time_mlp1, cfm.time_mlp2, t_scalar, t_emb,
        B, IN_CH, TIME_DIM,
    )

    # Build packed input (B, 320, T) = [x, mu, spks_broad, cond].
    var spks_broad = ctx.enqueue_create_buffer[DType.float32](B * MEL * T)
    broadcast_spks_to_t(ctx, spks, spks_broad, B, MEL, T)

    var p_xm = ctx.enqueue_create_buffer[DType.float32](B * (MEL * 2) * T)
    channel_concat_bct(ctx, x, mu, p_xm, B, MEL, MEL, T)

    var p_xms = ctx.enqueue_create_buffer[DType.float32](B * (MEL * 3) * T)
    channel_concat_bct(ctx, p_xm, spks_broad, p_xms, B, MEL * 2, MEL, T)

    var p = ctx.enqueue_create_buffer[DType.float32](B * IN_CH * T)
    channel_concat_bct(ctx, p_xms, cond, p, B, MEL * 3, MEL, T)

    var out = ctx.enqueue_create_buffer[DType.float32](B * D * T)
    resnet1d_forward(
        ctx, cfm.down_blocks[0].resnet, p, mask, t_emb, out,
        B, IN_CH, D, T, TIME_DIM,
    )
    ctx.synchronize()

    var n = B * D * T
    var max_abs: Float32 = 0.0
    var sum_diff_sq: Float32 = 0.0
    var sum_ref_sq: Float32 = 0.0
    with out.map_to_host() as h:
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
    print("[down0-resnet] max-abs diff =", max_abs, " rel_l2 =", rel_l2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
