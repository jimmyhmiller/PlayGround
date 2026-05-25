"""CFM estimator parity test against upstream torch oracle.

Loads the deterministic inputs + expected output dumped by
`scripts/dump_cfm_oracle.py` and asserts the Mojo `cfm_estimator_forward_real`
matches within tolerance.
"""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true
from std.gpu.host import DeviceContext, DeviceBuffer

from fixture import load_fp32
from weights import load_cfm_estimator_real, upload_fp32
from cfm_estimator_new import cfm_estimator_forward_real


comptime B = 1
comptime T = 16
comptime MEL = 80


def test_cfm_parity() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var fix = "weights/cfm_parity/"

    print("[cfm-parity] loading CFM estimator...")
    var cfm = load_cfm_estimator_real(ctx, "weights/s3gen/flow/decoder/estimator")

    # Upload oracle inputs.
    var x = upload_fp32(ctx, fix + "x.bin")
    var mu = upload_fp32(ctx, fix + "mu.bin")
    var spks = upload_fp32(ctx, fix + "spks.bin")
    var cond = upload_fp32(ctx, fix + "cond.bin")
    var mask = upload_fp32(ctx, fix + "mask.bin")
    var t_scalar = upload_fp32(ctx, fix + "t_scalar.bin")

    var expected = load_fp32(fix + "expected.bin")

    # Run Mojo forward.
    var out = ctx.enqueue_create_buffer[DType.float32](B * MEL * T)
    print("[cfm-parity] running Mojo forward...")
    cfm_estimator_forward_real(
        ctx, cfm, x, mu, spks, cond, mask, t_scalar, out, B, T,
    )
    ctx.synchronize()

    # Compare.
    var n = B * MEL * T
    var max_abs: Float32 = 0.0
    var sum_abs: Float32 = 0.0
    var n_nan = 0
    var sum_diff_sq: Float32 = 0.0
    var sum_ref_sq: Float32 = 0.0

    with out.map_to_host() as h:
        for i in range(n):
            var v = h[i]
            var r = expected.data[i]
            if v != v: n_nan += 1
            var d = v - r
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_abs += d
            var dd = v - r
            sum_diff_sq += dd * dd
            sum_ref_sq += r * r

    var mean_abs = sum_abs / Float32(n)
    var rmse = Float32(0.0)
    from std.math import sqrt
    rmse = sqrt(sum_diff_sq / Float32(n))
    var rel_l2 = sqrt(sum_diff_sq / sum_ref_sq) if sum_ref_sq > 0.0 else Float32(0.0)
    print("[cfm-parity] max-abs diff =", max_abs)
    print("[cfm-parity] mean-abs diff =", mean_abs)
    print("[cfm-parity] RMSE =", rmse)
    print("[cfm-parity] relative L2 =", rel_l2)

    # Print first few values for inspection.
    with out.map_to_host() as h:
        for i in range(8):
            print("  [", i, "] got=", h[i], " want=", expected.data[i],
                  " diff=", h[i] - expected.data[i])

    assert_true(n_nan == 0, "no NaNs")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
