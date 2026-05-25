"""Smoke test for the CFM Euler ODE solver.

Runs n_steps=2 of CFG-guided Euler integration with the real CFM estimator
on small (B=1, T=8) random inputs. Validates the entire chain — doubled-batch
input prep + estimator forward + CFG combine + Euler step — works end-to-end.
"""
from std.sys import has_accelerator
from std.math import sin
from std.testing import TestSuite, assert_true
from std.gpu.host import DeviceContext

from weights import load_cfm_estimator_real
from cfm_estimator_new import cfm_solve_euler


comptime B = 1
comptime T = 8
comptime MEL = 80
comptime N_STEPS = 2     # keep tiny so total work is manageable
comptime CFG: Float32 = 0.7


def test_cfm_solver_smoke() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    print("[cfm-solver] loading CFM estimator...")
    var cfm = load_cfm_estimator_real(ctx, "weights/s3gen/flow/decoder/estimator")

    var x = ctx.enqueue_create_buffer[DType.float32](B * MEL * T)
    var mu = ctx.enqueue_create_buffer[DType.float32](B * MEL * T)
    var spks = ctx.enqueue_create_buffer[DType.float32](B * MEL)
    var cond = ctx.enqueue_create_buffer[DType.float32](B * MEL * T)
    var mask = ctx.enqueue_create_buffer[DType.float32](B * T)

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

    print("[cfm-solver] running", N_STEPS, "Euler steps with CFG=", CFG, "...")
    cfm_solve_euler(ctx, cfm, x, mu, spks, cond, mask, B, T, N_STEPS, CFG)
    ctx.synchronize()

    var n_nan = 0
    var sum_abs: Float32 = 0.0
    with x.map_to_host() as h:
        for i in range(B * MEL * T):
            var v = h[i]
            if v != v: n_nan += 1
            if v < 0.0: sum_abs -= v
            else:       sum_abs += v
    print("[cfm-solver] final mel mean-abs=",
          sum_abs / Float32(B * MEL * T), " nan_count=", n_nan)
    assert_true(n_nan == 0, "no NaNs in solver output")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
