"""
conv1d / transposed_conv1d / leaky_relu parity tests vs PyTorch.
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import Tensor, TensorI64, load_fp32, load_i64
from conv import conv1d_kernel, transposed_conv1d_kernel, leaky_relu_kernel


comptime POINTWISE_BLOCK = 256


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def test_conv1d_fp32() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/conv/"
    var x = load_fp32(fix + "conv1d_x.bin")
    var w = load_fp32(fix + "conv1d_w.bin")
    var bias = load_fp32(fix + "conv1d_bias.bin")
    var exp = load_fp32(fix + "conv1d_expected.bin")
    var meta = load_i64(fix + "conv1d_meta.bin")

    var B = Int(meta.data[0])
    var C_in = Int(meta.data[1])
    var C_out = Int(meta.data[2])
    var L_in = Int(meta.data[3])
    var L_out = Int(meta.data[4])
    comptime K = 7
    var stride = Int(meta.data[6])
    var padding = Int(meta.data[7])
    var dilation = Int(meta.data[8])

    assert_equal(B, 1)
    assert_equal(C_in, 4)
    assert_equal(C_out, 6)
    assert_equal(L_in, 16)
    assert_equal(L_out, 16)

    var n_x = B * C_in * L_in
    var n_w = C_out * C_in * K
    var n_out = B * C_out * L_out

    var ctx = DeviceContext()
    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var w_buf = ctx.enqueue_create_buffer[DType.float32](n_w)
    var bias_buf = ctx.enqueue_create_buffer[DType.float32](C_out)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    upload(x_buf, x.data, n_x)
    upload(w_buf, w.data, n_w)
    upload(bias_buf, bias.data, C_out)

    comptime x_layout = row_major[1, 4, 16]()
    comptime w_layout = row_major[6, 4, K]()
    comptime bias_layout = row_major[6]()
    comptime out_layout = row_major[1, 6, 16]()

    var x_t = TileTensor(x_buf, x_layout)
    var w_t = TileTensor(w_buf, w_layout)
    var bias_t = TileTensor(bias_buf, bias_layout)
    var out_t = TileTensor(out_buf, out_layout)

    comptime kernel = conv1d_kernel[
        DType.float32, type_of(x_layout), type_of(w_layout),
        type_of(bias_layout), type_of(out_layout), K, True,
    ]
    ctx.enqueue_function[kernel, kernel](
        out_t, x_t, w_t, bias_t,
        B, C_in, C_out, L_in, L_out, stride, padding, dilation,
        grid_dim=B * C_out * L_out, block_dim=1,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n_out):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-5)
    print("conv1d fp32 — max abs:", max_abs)


def test_conv1d_dilated_fp32() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/conv/"
    var x = load_fp32(fix + "conv1d_x.bin")          # same x as conv1d
    var w = load_fp32(fix + "conv1d_dil_w.bin")
    var bias = load_fp32(fix + "conv1d_dil_bias.bin")
    var exp = load_fp32(fix + "conv1d_dil_expected.bin")
    var meta = load_i64(fix + "conv1d_dil_meta.bin")

    var B = Int(meta.data[0])
    var C_in = Int(meta.data[1])
    var C_out = Int(meta.data[2])
    var L_in = Int(meta.data[3])
    var L_out = Int(meta.data[4])
    comptime K = 3
    var stride = Int(meta.data[6])
    var padding = Int(meta.data[7])
    var dilation = Int(meta.data[8])

    var n_x = B * C_in * L_in
    var n_w = C_out * C_in * K
    var n_out = B * C_out * L_out

    var ctx = DeviceContext()
    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var w_buf = ctx.enqueue_create_buffer[DType.float32](n_w)
    var bias_buf = ctx.enqueue_create_buffer[DType.float32](C_out)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    upload(x_buf, x.data, n_x)
    upload(w_buf, w.data, n_w)
    upload(bias_buf, bias.data, C_out)

    comptime x_layout = row_major[1, 4, 16]()
    comptime w_layout = row_major[6, 4, K]()
    comptime bias_layout = row_major[6]()
    comptime out_layout = row_major[1, 6, 16]()

    var x_t = TileTensor(x_buf, x_layout)
    var w_t = TileTensor(w_buf, w_layout)
    var bias_t = TileTensor(bias_buf, bias_layout)
    var out_t = TileTensor(out_buf, out_layout)

    comptime kernel = conv1d_kernel[
        DType.float32, type_of(x_layout), type_of(w_layout),
        type_of(bias_layout), type_of(out_layout), K, True,
    ]
    ctx.enqueue_function[kernel, kernel](
        out_t, x_t, w_t, bias_t,
        B, C_in, C_out, L_in, L_out, stride, padding, dilation,
        grid_dim=B * C_out * L_out, block_dim=1,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n_out):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-5)
    print("conv1d (dilated) fp32 — max abs:", max_abs)


def test_transposed_conv1d_fp32() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/conv/"
    var x = load_fp32(fix + "tconv1d_x.bin")
    var w = load_fp32(fix + "tconv1d_w.bin")
    var bias = load_fp32(fix + "tconv1d_bias.bin")
    var exp = load_fp32(fix + "tconv1d_expected.bin")
    var meta = load_i64(fix + "tconv1d_meta.bin")

    var B = Int(meta.data[0])
    var C_in = Int(meta.data[1])
    var C_out = Int(meta.data[2])
    var L_in = Int(meta.data[3])
    var L_out = Int(meta.data[4])
    comptime K = 8
    var stride = Int(meta.data[6])
    var padding = Int(meta.data[7])
    var dilation = Int(meta.data[8])

    var n_x = B * C_in * L_in
    var n_w = C_in * C_out * K
    var n_out = B * C_out * L_out

    var ctx = DeviceContext()
    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var w_buf = ctx.enqueue_create_buffer[DType.float32](n_w)
    var bias_buf = ctx.enqueue_create_buffer[DType.float32](C_out)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    upload(x_buf, x.data, n_x)
    upload(w_buf, w.data, n_w)
    upload(bias_buf, bias.data, C_out)

    comptime x_layout = row_major[1, 4, 8]()
    comptime w_layout = row_major[4, 6, K]()
    comptime bias_layout = row_major[6]()
    comptime out_layout = row_major[1, 6, 32]()

    var x_t = TileTensor(x_buf, x_layout)
    var w_t = TileTensor(w_buf, w_layout)
    var bias_t = TileTensor(bias_buf, bias_layout)
    var out_t = TileTensor(out_buf, out_layout)

    comptime kernel = transposed_conv1d_kernel[
        DType.float32, type_of(x_layout), type_of(w_layout),
        type_of(bias_layout), type_of(out_layout), K, True,
    ]
    ctx.enqueue_function[kernel, kernel](
        out_t, x_t, w_t, bias_t,
        B, C_in, C_out, L_in, L_out, stride, padding, dilation,
        grid_dim=B * C_out * L_out, block_dim=1,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n_out):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-5)
    print("transposed_conv1d fp32 — max abs:", max_abs)


def test_leaky_relu_fp32() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/conv/"
    var x = load_fp32(fix + "leaky_x.bin")
    var exp = load_fp32(fix + "leaky_expected.bin")

    var n = 1 * 6 * 16

    var ctx = DeviceContext()
    var x_buf = ctx.enqueue_create_buffer[DType.float32](n)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n)
    upload(x_buf, x.data, n)

    comptime flat_layout = row_major[1 * 6 * 16]()
    var x_t = TileTensor(x_buf, flat_layout)
    var out_t = TileTensor(out_buf, flat_layout)

    comptime kernel = leaky_relu_kernel[
        DType.float32, type_of(flat_layout), type_of(flat_layout), POINTWISE_BLOCK,
    ]
    ctx.enqueue_function[kernel, kernel](
        out_t, x_t, n, Float32(0.1),
        grid_dim=ceildiv(n, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            assert_almost_equal(h[i], exp.data[i], atol=0.0)
    print("leaky_relu fp32 — max abs:", max_abs)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
