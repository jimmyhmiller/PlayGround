"""Runtime smoke test for the CAMPPlus xvector backbone forward.

Loads real upstream CAMPPlus weights and runs the full xvector forward
(tdnn → block1 → transit1 → block2 → transit2 → block3 → transit3 →
out_nonlinear → stats_pool → dense) on a fake (B, 320, T) FCM-output input.
The 2D FCM head is skipped — caller assumed to provide FCM output directly.
"""
from std.sys import has_accelerator
from std.math import sin
from std.testing import TestSuite, assert_true
from std.gpu.host import DeviceContext

from weights import load_campplus
from campplus import xvector_forward


comptime B = 1
comptime T_IN = 16    # post-FCM time length (FCM downsamples by ~8 from raw mel T)
comptime FCM_OUT = 320


def test_campplus_xvector_forward_smoke() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    print("[campplus-fwd] loading from weights/s3gen/speaker_encoder/...")
    var sp = load_campplus(ctx, "weights/s3gen/speaker_encoder")

    var x = ctx.enqueue_create_buffer[DType.float32](B * FCM_OUT * T_IN)
    with x.map_to_host() as h:
        for c in range(FCM_OUT):
            for ti in range(T_IN):
                h[c * T_IN + ti] = sin(Float32(c) * 0.05 + Float32(ti) * 0.1) * 0.1

    var embed = ctx.enqueue_create_buffer[DType.float32](B * 192)
    print("[campplus-fwd] running xvector forward (T_in=", T_IN, ")...")
    xvector_forward(ctx, sp.xvector, x, embed, B, T_IN)
    ctx.synchronize()

    var n_nan = 0
    var sum_abs: Float32 = 0.0
    with embed.map_to_host() as h:
        for i in range(B * 192):
            var v = h[i]
            if v != v: n_nan += 1
            if v < 0.0: sum_abs -= v
            else:       sum_abs += v
    print("[campplus-fwd] 192-d speaker emb mean-abs=",
          sum_abs / Float32(B * 192), " nan_count=", n_nan)
    assert_true(n_nan == 0, "no NaNs in CAMPPlus output")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
