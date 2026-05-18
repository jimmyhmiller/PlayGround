"""Smoke test for Module wrappers (Linear, LayerNorm, softmax_lastdim)."""
from std.math import exp, sqrt, erf
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from modules import (
    Linear, linear_forward,
    LayerNorm, layer_norm_forward,
    RMSNorm, rms_norm_forward,
    softmax_lastdim,
    silu, gelu,
)


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def test_linear_smoke() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var M = 3
    var IN = 4
    var OUT = 5
    var n_w = OUT * IN
    var n_b = OUT
    var n_x = M * IN
    var n_y = M * OUT

    # Build a simple Linear: weight[i, j] = (i*IN + j) * 0.01, bias[i] = i*0.1.
    var w_data = List[Float32]()
    for i in range(n_w):
        w_data.append(Float32(i) * 0.01)
    var b_data = List[Float32]()
    for i in range(n_b):
        b_data.append(Float32(i) * 0.1)
    var x_data = List[Float32]()
    for i in range(n_x):
        x_data.append(Float32(i + 1) * 0.05)

    var w_buf = ctx.enqueue_create_buffer[DType.float32](n_w)
    var b_buf = ctx.enqueue_create_buffer[DType.float32](n_b)
    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var y_buf = ctx.enqueue_create_buffer[DType.float32](n_y)
    upload(w_buf, w_data, n_w)
    upload(b_buf, b_data, n_b)
    upload(x_buf, x_data, n_x)

    var lin = Linear(w_buf, b_buf, IN, OUT, True)
    linear_forward(ctx, lin, x_buf, y_buf, M)
    ctx.synchronize()

    # CPU reference.
    var expected = List[Float32]()
    for i in range(n_y): expected.append(Float32(0.0))
    for m in range(M):
        for o in range(OUT):
            var acc: Float32 = 0.0
            for k in range(IN):
                acc += x_data[m * IN + k] * w_data[o * IN + k]
            expected[m * OUT + o] = acc + b_data[o]

    var max_abs: Float32 = 0.0
    with y_buf.map_to_host() as h:
        for i in range(n_y):
            var d = h[i] - expected[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            assert_almost_equal(h[i], expected[i], atol=1.0e-5)
    print("Linear smoke PASS — max abs:", max_abs)


def test_layer_norm_smoke() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var B = 2
    var F = 8
    var n = B * F

    var x_data = List[Float32]()
    for i in range(n):
        x_data.append(Float32(i) * 0.1 - 0.3)
    var g_data = List[Float32]()
    for i in range(F):
        g_data.append(1.0 + Float32(i) * 0.01)
    var b_data = List[Float32]()
    for i in range(F):
        b_data.append(Float32(i) * 0.05)

    var x_buf = ctx.enqueue_create_buffer[DType.float32](n)
    var y_buf = ctx.enqueue_create_buffer[DType.float32](n)
    var g_buf = ctx.enqueue_create_buffer[DType.float32](F)
    var bb_buf = ctx.enqueue_create_buffer[DType.float32](F)
    upload(x_buf, x_data, n)
    upload(g_buf, g_data, F)
    upload(bb_buf, b_data, F)

    var ln = LayerNorm(g_buf, bb_buf, F, Float32(1.0e-5))
    layer_norm_forward(ctx, ln, x_buf, y_buf, B)
    ctx.synchronize()

    # CPU reference.
    var expected = List[Float32]()
    for i in range(n): expected.append(Float32(0.0))
    for bi in range(B):
        var off = bi * F
        var mean: Float32 = 0.0
        for j in range(F): mean += x_data[off + j]
        mean /= Float32(F)
        var var_v: Float32 = 0.0
        for j in range(F):
            var d = x_data[off + j] - mean
            var_v += d * d
        var_v /= Float32(F)
        var inv = 1.0 / sqrt(var_v + 1.0e-5)
        for j in range(F):
            expected[off + j] = (x_data[off + j] - mean) * inv * g_data[j] + b_data[j]

    var max_abs: Float32 = 0.0
    with y_buf.map_to_host() as h:
        for i in range(n):
            var d = h[i] - expected[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            assert_almost_equal(h[i], expected[i], atol=1.0e-5)
    print("LayerNorm smoke PASS — max abs:", max_abs)


def test_rms_norm_smoke() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var B = 2
    var F = 8
    var n = B * F

    var x_data = List[Float32]()
    for i in range(n):
        x_data.append(Float32(i) * 0.1 - 0.3)
    var g_data = List[Float32]()
    for i in range(F):
        g_data.append(1.0 + Float32(i) * 0.01)

    var x_buf = ctx.enqueue_create_buffer[DType.float32](n)
    var y_buf = ctx.enqueue_create_buffer[DType.float32](n)
    var g_buf = ctx.enqueue_create_buffer[DType.float32](F)
    upload(x_buf, x_data, n)
    upload(g_buf, g_data, F)

    var rn = RMSNorm(g_buf, F, Float32(1.0e-5))
    rms_norm_forward(ctx, rn, x_buf, y_buf, B)
    ctx.synchronize()

    # CPU reference: x / sqrt(mean(x^2) + eps) * gamma.
    var expected = List[Float32]()
    for i in range(n): expected.append(Float32(0.0))
    for bi in range(B):
        var off = bi * F
        var sq: Float32 = 0.0
        for j in range(F): sq += x_data[off + j] * x_data[off + j]
        sq /= Float32(F)
        var inv = 1.0 / sqrt(sq + 1.0e-5)
        for j in range(F):
            expected[off + j] = x_data[off + j] * inv * g_data[j]

    var max_abs: Float32 = 0.0
    with y_buf.map_to_host() as h:
        for i in range(n):
            var d = h[i] - expected[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            assert_almost_equal(h[i], expected[i], atol=1.0e-5)
    print("RMSNorm smoke PASS — max abs:", max_abs)


def test_silu_gelu_smoke() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var n = 16

    var x_data = List[Float32]()
    for i in range(n):
        x_data.append(Float32(i) * 0.2 - 1.5)

    var x_buf = ctx.enqueue_create_buffer[DType.float32](n)
    var y_silu_buf = ctx.enqueue_create_buffer[DType.float32](n)
    var y_gelu_buf = ctx.enqueue_create_buffer[DType.float32](n)
    upload(x_buf, x_data, n)

    silu(ctx, x_buf, y_silu_buf, n)
    gelu(ctx, x_buf, y_gelu_buf, n)
    ctx.synchronize()

    # CPU silu = x * sigmoid(x); gelu = 0.5*x*(1+erf(x/sqrt2))
    var max_silu: Float32 = 0.0
    var max_gelu: Float32 = 0.0
    with y_silu_buf.map_to_host() as hs:
        with y_gelu_buf.map_to_host() as hg:
            for i in range(n):
                var x = x_data[i]
                var sig: Float32 = 1.0 / (1.0 + exp(-x))
                var exp_silu = x * sig
                var exp_gelu = 0.5 * x * (1.0 + erf(x * 0.7071067811865476))
                var ds = hs[i] - exp_silu
                if ds < 0.0: ds = -ds
                if ds > max_silu: max_silu = ds
                var dg = hg[i] - exp_gelu
                if dg < 0.0: dg = -dg
                if dg > max_gelu: max_gelu = dg
                assert_almost_equal(hs[i], exp_silu, atol=1.0e-6)
                assert_almost_equal(hg[i], exp_gelu, atol=1.0e-6)
    print("SiLU max abs:", max_silu, "  GELU max abs:", max_gelu)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
