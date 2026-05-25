"""
Parity test for the first conv2d + BN2d + ReLU of the CAMPPlus FCM head.

Validates: conv2d_kernel, batchnorm2d_kernel, relu_kernel against real
CAMPPlus weights.
"""
from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import Tensor, load_fp32
from conv import conv2d_kernel, batchnorm2d_kernel, relu_kernel


comptime BATCH = 1
comptime IN_C = 1
comptime OUT_C = 32
comptime H = 80
comptime W = 998
comptime KH = 3
comptime KW = 3
comptime PAD_H = 1
comptime PAD_W = 1
comptime STRIDE_H = 1
comptime STRIDE_W = 1
comptime POINTWISE_BLOCK = 256
comptime EPS: Float32 = 1.0e-5


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def test_fcm_head_first_layer() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/campplus/"
    var x_in = load_fp32(fix + "fcm_input_4d.bin")
    var w_conv = load_fp32(fix + "weights/head__conv1__weight.bin")
    var bn_weight = load_fp32(fix + "weights/head__bn1__weight.bin")
    var bn_bias = load_fp32(fix + "weights/head__bn1__bias.bin")
    var bn_mean = load_fp32(fix + "weights/head__bn1__running_mean.bin")
    var bn_var = load_fp32(fix + "weights/head__bn1__running_var.bin")
    var exp = load_fp32(fix + "fcm_relu1_out.bin")

    assert_equal(x_in.shape[2], H)
    assert_equal(x_in.shape[3], W)
    assert_equal(w_conv.shape[0], OUT_C)

    var n_x = BATCH * IN_C * H * W
    var n_w = OUT_C * IN_C * KH * KW
    var n_out = BATCH * OUT_C * H * W

    var ctx = DeviceContext()
    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var w_buf = ctx.enqueue_create_buffer[DType.float32](n_w)
    var conv_out_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    var bn_w_buf = ctx.enqueue_create_buffer[DType.float32](OUT_C)
    var bn_b_buf = ctx.enqueue_create_buffer[DType.float32](OUT_C)
    var bn_m_buf = ctx.enqueue_create_buffer[DType.float32](OUT_C)
    var bn_v_buf = ctx.enqueue_create_buffer[DType.float32](OUT_C)
    var bn_out_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_out)

    upload(x_buf, x_in.data, n_x)
    upload(w_buf, w_conv.data, n_w)
    upload(bn_w_buf, bn_weight.data, OUT_C)
    upload(bn_b_buf, bn_bias.data, OUT_C)
    upload(bn_m_buf, bn_mean.data, OUT_C)
    upload(bn_v_buf, bn_var.data, OUT_C)

    comptime x_layout = row_major[BATCH, IN_C, H, W]()
    comptime w_layout = row_major[OUT_C, IN_C, KH, KW]()
    comptime out_layout = row_major[BATCH, OUT_C, H, W]()
    comptime bn_p_layout = row_major[OUT_C]()
    comptime flat_layout = row_major[BATCH * OUT_C * H * W]()
    # conv2d kernel requires a bias arg; pass a dummy buffer (won't be read when HAS_BIAS=False).
    var dummy_b_buf = ctx.enqueue_create_buffer[DType.float32](OUT_C)
    var dummy_b_t = TileTensor(dummy_b_buf, bn_p_layout)

    var x_t = TileTensor(x_buf, x_layout)
    var w_t = TileTensor(w_buf, w_layout)
    var conv_out_t = TileTensor(conv_out_buf, out_layout)
    var bn_w_t = TileTensor(bn_w_buf, bn_p_layout)
    var bn_b_t = TileTensor(bn_b_buf, bn_p_layout)
    var bn_m_t = TileTensor(bn_m_buf, bn_p_layout)
    var bn_v_t = TileTensor(bn_v_buf, bn_p_layout)
    var bn_out_t = TileTensor(bn_out_buf, out_layout)
    var bn_out_flat = TileTensor(bn_out_buf, flat_layout)
    var out_flat = TileTensor(out_buf, flat_layout)

    # conv2d (no bias).
    comptime conv_k = conv2d_kernel[
        DType.float32, type_of(x_layout), type_of(w_layout),
        type_of(bn_p_layout), type_of(out_layout),
        KH, KW, False, 256,
    ]
    ctx.enqueue_function[conv_k, conv_k](
        conv_out_t, x_t, w_t, dummy_b_t,
        BATCH, IN_C, OUT_C, H, W, H, W,
        STRIDE_H, STRIDE_W, PAD_H, PAD_W,
        grid_dim=BATCH * OUT_C * H, block_dim=256,
    )
    # batchnorm2d.
    comptime bn_k = batchnorm2d_kernel[
        DType.float32, type_of(out_layout), type_of(bn_p_layout),
        type_of(out_layout), 256,
    ]
    ctx.enqueue_function[bn_k, bn_k](
        bn_out_t, conv_out_t, bn_w_t, bn_b_t, bn_m_t, bn_v_t,
        BATCH, OUT_C, H, W, EPS,
        grid_dim=BATCH * OUT_C * H, block_dim=256,
    )
    # relu.
    comptime relu_k = relu_kernel[
        DType.float32, type_of(flat_layout), type_of(flat_layout), POINTWISE_BLOCK,
    ]
    ctx.enqueue_function[relu_k, relu_k](
        out_flat, bn_out_flat, n_out,
        grid_dim=ceildiv(n_out, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    var sum_abs: Float64 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n_out):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_abs += Float64(d)
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-4)
    print("CAMPPlus FCM first layer (conv2d+BN2d+ReLU) — max abs:", max_abs,
          " mean abs:", sum_abs / Float64(n_out))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
