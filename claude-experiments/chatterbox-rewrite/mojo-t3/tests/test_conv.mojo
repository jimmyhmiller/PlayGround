"""
conv1d / transposed_conv1d / leaky_relu parity tests vs PyTorch.
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import Tensor, TensorI64, load_fp32, load_i64
from conv import conv1d_kernel, transposed_conv1d_kernel, leaky_relu_kernel, snake_kernel
from util_kernels import add_kernel


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


def test_snake_fp32() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/conv/"
    var x = load_fp32(fix + "snake_x.bin")
    var alpha = load_fp32(fix + "snake_alpha.bin")
    var exp = load_fp32(fix + "snake_expected.bin")
    var meta = load_i64(fix + "snake_meta.bin")

    var B = Int(meta.data[0])
    var C = Int(meta.data[1])
    var L = Int(meta.data[2])
    assert_equal(B, 1)
    assert_equal(C, 8)
    assert_equal(L, 16)

    var n = B * C * L

    var ctx = DeviceContext()
    var x_buf = ctx.enqueue_create_buffer[DType.float32](n)
    var a_buf = ctx.enqueue_create_buffer[DType.float32](C)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n)
    upload(x_buf, x.data, n)
    upload(a_buf, alpha.data, C)

    comptime x_layout = row_major[1, 8, 16]()
    comptime a_layout = row_major[8]()

    var x_t = TileTensor(x_buf, x_layout)
    var a_t = TileTensor(a_buf, a_layout)
    var out_t = TileTensor(out_buf, x_layout)

    comptime kernel = snake_kernel[
        DType.float32, type_of(x_layout), type_of(a_layout), type_of(x_layout),
    ]
    ctx.enqueue_function[kernel, kernel](
        out_t, x_t, a_t, B, C, L,
        grid_dim=B * C, block_dim=L,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-5)
    print("snake fp32 — max abs:", max_abs)


def test_resblock_fp32() raises:
    """Single-ResBlock parity: chain of (snake → conv1 → snake → conv2 → +residual)
    for each of dilations [1, 3, 5]. Uses the conv1d kernel for all conv calls
    and the snake kernel for activations.
    """
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/conv/"
    var x_in = load_fp32(fix + "resblock_x.bin")
    var exp = load_fp32(fix + "resblock_expected.bin")
    var gmeta = load_i64(fix + "resblock_global_meta.bin")

    var B = Int(gmeta.data[0])
    var C = Int(gmeta.data[1])
    var L = Int(gmeta.data[2])
    comptime K = 3
    var n_dils = Int(gmeta.data[4])

    assert_equal(B, 1)
    assert_equal(C, 8)
    assert_equal(L, 32)
    assert_equal(n_dils, 3)

    var n = B * C * L
    var n_w = C * C * K

    var ctx = DeviceContext()
    # Two ping-pong buffers for the residual stream + a scratch for xt stages.
    var x_buf = ctx.enqueue_create_buffer[DType.float32](n)
    var x_next_buf = ctx.enqueue_create_buffer[DType.float32](n)
    var xt_buf = ctx.enqueue_create_buffer[DType.float32](n)
    var xt2_buf = ctx.enqueue_create_buffer[DType.float32](n)
    var w_buf = ctx.enqueue_create_buffer[DType.float32](n_w)
    var bias_buf = ctx.enqueue_create_buffer[DType.float32](C)
    var alpha_buf = ctx.enqueue_create_buffer[DType.float32](C)
    upload(x_buf, x_in.data, n)

    comptime x_layout = row_major[1, 8, 32]()
    comptime w_layout = row_major[8, 8, K]()
    comptime bias_layout = row_major[8]()
    comptime alpha_layout = row_major[8]()
    comptime flat_layout = row_major[1, 1 * 8 * 32]()

    var x_t = TileTensor(x_buf, x_layout)
    var x_next_t = TileTensor(x_next_buf, x_layout)
    var xt_t = TileTensor(xt_buf, x_layout)
    var xt2_t = TileTensor(xt2_buf, x_layout)
    var x_flat = TileTensor(x_buf, flat_layout)
    var x_next_flat = TileTensor(x_next_buf, flat_layout)
    var xt2_flat = TileTensor(xt2_buf, flat_layout)
    var w_t = TileTensor(w_buf, w_layout)
    var bias_t = TileTensor(bias_buf, bias_layout)
    var alpha_t = TileTensor(alpha_buf, alpha_layout)

    comptime conv_k = conv1d_kernel[
        DType.float32, type_of(x_layout), type_of(w_layout),
        type_of(bias_layout), type_of(x_layout), K, True,
    ]
    comptime snake_k = snake_kernel[
        DType.float32, type_of(x_layout), type_of(alpha_layout), type_of(x_layout),
    ]
    comptime add_k = add_kernel[
        DType.float32, type_of(flat_layout), type_of(flat_layout),
        type_of(flat_layout), 256,
    ]

    for i in range(n_dils):
        var meta = load_i64(fix + "resblock_meta_" + String(i) + ".bin")
        var dil = Int(meta.data[0])
        var pad1 = Int(meta.data[1])
        var pad2 = Int(meta.data[2])
        var w1 = load_fp32(fix + "resblock_w1_" + String(i) + ".bin")
        var b1 = load_fp32(fix + "resblock_b1_" + String(i) + ".bin")
        var w2 = load_fp32(fix + "resblock_w2_" + String(i) + ".bin")
        var b2 = load_fp32(fix + "resblock_b2_" + String(i) + ".bin")
        var a1 = load_fp32(fix + "resblock_a1_" + String(i) + ".bin")
        var a2 = load_fp32(fix + "resblock_a2_" + String(i) + ".bin")

        # xt = snake(x, a1)
        upload(alpha_buf, a1.data, C)
        ctx.enqueue_function[snake_k, snake_k](
            xt_t, x_t, alpha_t, B, C, L,
            grid_dim=B * C, block_dim=L,
        )
        # xt = conv1d(xt, w1, b1, dilation=dil, padding=pad1)
        upload(w_buf, w1.data, n_w)
        upload(bias_buf, b1.data, C)
        ctx.enqueue_function[conv_k, conv_k](
            xt2_t, xt_t, w_t, bias_t,
            B, C, C, L, L, 1, pad1, dil,
            grid_dim=B * C * L, block_dim=1,
        )
        # xt = snake(xt, a2)
        upload(alpha_buf, a2.data, C)
        ctx.enqueue_function[snake_k, snake_k](
            xt_t, xt2_t, alpha_t, B, C, L,
            grid_dim=B * C, block_dim=L,
        )
        # xt = conv1d(xt, w2, b2, dilation=1, padding=pad2)
        upload(w_buf, w2.data, n_w)
        upload(bias_buf, b2.data, C)
        ctx.enqueue_function[conv_k, conv_k](
            xt2_t, xt_t, w_t, bias_t,
            B, C, C, L, L, 1, pad2, 1,
            grid_dim=B * C * L, block_dim=1,
        )
        # x_next = x + xt2  (residual; can't alias x_buf with itself)
        ctx.enqueue_function[add_k, add_k](
            x_next_flat, x_flat, xt2_flat, n,
            grid_dim=(n + 255) // 256, block_dim=256,
        )
        # Copy x_next back to x for the next iteration.
        ctx.synchronize()
        with x_next_buf.map_to_host() as src:
            with x_buf.map_to_host() as dst:
                for j in range(n):
                    dst[j] = src[j]

    ctx.synchronize()
    var max_abs: Float32 = 0.0
    with x_buf.map_to_host() as h:
        for i in range(n):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            assert_almost_equal(h[i], exp.data[i], atol=5.0e-5)
    print("ResBlock fp32 — max abs:", max_abs)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
