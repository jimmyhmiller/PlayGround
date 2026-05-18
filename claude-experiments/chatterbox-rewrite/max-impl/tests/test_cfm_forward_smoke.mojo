"""Runtime smoke test for the full CFM Matcha estimator forward.

Loads real upstream weights and runs the entire U-Net (1 down + 12 mid + 1 up
+ final block + final_proj) with small T=16 fake inputs. Validates that:
  - all transformer blocks chain without crashing
  - output shape is (B, 80, T)
  - no NaNs
This is a structural smoke — full torch parity needs an oracle dump.
"""
from std.sys import has_accelerator
from std.math import sin
from std.testing import TestSuite, assert_true
from std.gpu.host import DeviceContext

from weights import load_cfm_estimator_real
from cfm_estimator_new import cfm_estimator_forward_real


comptime B = 1
comptime T = 16
comptime MEL = 80


def test_cfm_forward_smoke() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    print("[cfm-fwd] loading CFM estimator from weights/s3gen/flow/decoder/estimator/...")
    var cfm = load_cfm_estimator_real(ctx, "weights/s3gen/flow/decoder/estimator")

    # Fake inputs.
    var x = ctx.enqueue_create_buffer[DType.float32](B * MEL * T)
    var mu = ctx.enqueue_create_buffer[DType.float32](B * MEL * T)
    var spks = ctx.enqueue_create_buffer[DType.float32](B * MEL)
    var cond = ctx.enqueue_create_buffer[DType.float32](B * MEL * T)
    var mask = ctx.enqueue_create_buffer[DType.float32](B * T)
    var t_scalar = ctx.enqueue_create_buffer[DType.float32](B)
    var out = ctx.enqueue_create_buffer[DType.float32](B * MEL * T)

    with x.map_to_host() as h:
        for c in range(MEL):
            for ti in range(T):
                h[c * T + ti] = sin(Float32(c) * 0.05 + Float32(ti) * 0.1) * 0.1
    with mu.map_to_host() as h:
        for c in range(MEL):
            for ti in range(T):
                h[c * T + ti] = sin(Float32(c) * 0.07 + Float32(ti) * 0.13) * 0.1
    with spks.map_to_host() as h:
        for c in range(MEL):
            h[c] = sin(Float32(c) * 0.11) * 0.1
    with cond.map_to_host() as h:
        for c in range(MEL):
            for ti in range(T):
                h[c * T + ti] = 0.0
    with mask.map_to_host() as h:
        for ti in range(T):
            h[ti] = 1.0
    with t_scalar.map_to_host() as h:
        h[0] = 0.5

    print("[cfm-fwd] running estimator forward (B=", B, " T=", T, ")...")
    cfm_estimator_forward_real(
        ctx, cfm, x, mu, spks, cond, mask, t_scalar, out, B, T,
    )
    ctx.synchronize()

    var n_nan = 0
    var sum_abs: Float32 = 0.0
    with out.map_to_host() as h:
        for i in range(B * MEL * T):
            var v = h[i]
            if v != v: n_nan += 1
            if v < 0.0: sum_abs -= v
            else:       sum_abs += v
    print("[cfm-fwd] output mean-abs=", sum_abs / Float32(B * MEL * T),
          " nan_count=", n_nan)
    assert_true(n_nan == 0, "no NaNs in CFM output")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
