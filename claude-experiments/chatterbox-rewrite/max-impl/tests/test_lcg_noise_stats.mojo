"""Sanity check: gaussian_noise_fill should produce ~N(0, 1) samples."""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.math import sqrt
from std.gpu.host import DeviceContext

from cfm_estimator_new import gaussian_noise_fill


def test_lcg_stats() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var N = 80 * 602
    var buf = ctx.enqueue_create_buffer[DType.float32](N)
    gaussian_noise_fill(ctx, buf, N, UInt64(0xC0FFEE), Float32(1.0))
    ctx.synchronize()

    var mean: Float64 = 0.0
    var sum_sq: Float64 = 0.0
    var min_v: Float32 = 1.0e30
    var max_v: Float32 = -1.0e30
    var n_big = 0   # |x| > 4
    with buf.map_to_host() as h:
        for i in range(N):
            var v = h[i]
            mean += Float64(v)
            sum_sq += Float64(v) * Float64(v)
            if v < min_v: min_v = v
            if v > max_v: max_v = v
            var av = v
            if av < 0.0: av = -av
            if av > 4.0: n_big += 1
    mean /= Float64(N)
    var var_v = sum_sq / Float64(N) - mean * mean
    var std_v = sqrt(var_v)
    print("[lcg] n=", N, " mean=", Float32(mean), " std=", Float32(std_v), " min=", min_v, " max=", max_v, " n(|x|>4)=", n_big)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
