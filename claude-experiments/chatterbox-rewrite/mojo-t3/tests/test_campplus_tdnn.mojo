"""
Parity test for CAMPPlus xvector.tdnn (the first TDNN layer):
  Conv1d(320, 128, k=5, stride=2, padding=2, dilation=1, bias=False)
  BatchNorm1d(128)
  ReLU

Input:  fcm_out.bin       (1, 320, 998)
Target: tdnn_out.bin      (1, 128, 499)
"""
from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import load_fp32
from conv import conv1d_kernel_fast, batchnorm1d_kernel, relu_kernel


comptime B = 1
comptime C_IN = 320
comptime C_OUT = 128
comptime L_IN = 998
comptime L_OUT = 499
comptime K = 5
comptime STRIDE = 2
comptime PADDING = 2
comptime DILATION = 1
comptime BLOCK = 256
comptime EPS: Float32 = 1.0e-5


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def test_xvector_tdnn() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/campplus/"
    var ctx = DeviceContext()

    var x_in = load_fp32(fix + "fcm_out.bin")
    var w = load_fp32(fix + "weights/xvector__tdnn__linear__weight.bin")
    var bn_w = load_fp32(fix + "weights/xvector__tdnn__nonlinear__batchnorm__weight.bin")
    var bn_b = load_fp32(fix + "weights/xvector__tdnn__nonlinear__batchnorm__bias.bin")
    var bn_m = load_fp32(fix + "weights/xvector__tdnn__nonlinear__batchnorm__running_mean.bin")
    var bn_v = load_fp32(fix + "weights/xvector__tdnn__nonlinear__batchnorm__running_var.bin")
    var exp = load_fp32(fix + "tdnn_out.bin")

    var n_in = B * C_IN * L_IN
    var n_out = B * C_OUT * L_OUT
    var n_w = C_OUT * C_IN * K

    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_in)
    var w_buf = ctx.enqueue_create_buffer[DType.float32](n_w)
    var bias_buf = ctx.enqueue_create_buffer[DType.float32](C_OUT)
    var bn_w_buf = ctx.enqueue_create_buffer[DType.float32](C_OUT)
    var bn_b_buf = ctx.enqueue_create_buffer[DType.float32](C_OUT)
    var bn_m_buf = ctx.enqueue_create_buffer[DType.float32](C_OUT)
    var bn_v_buf = ctx.enqueue_create_buffer[DType.float32](C_OUT)
    var conv_out = ctx.enqueue_create_buffer[DType.float32](n_out)
    var bn_out = ctx.enqueue_create_buffer[DType.float32](n_out)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_out)

    upload(x_buf, x_in.data, n_in)
    upload(w_buf, w.data, n_w)
    upload(bn_w_buf, bn_w.data, C_OUT)
    upload(bn_b_buf, bn_b.data, C_OUT)
    upload(bn_m_buf, bn_m.data, C_OUT)
    upload(bn_v_buf, bn_v.data, C_OUT)

    comptime in_layout = row_major[B, C_IN, L_IN]()
    comptime w_layout = row_major[C_OUT, C_IN, K]()
    comptime out_layout = row_major[B, C_OUT, L_OUT]()
    comptime p_layout = row_major[C_OUT]()
    comptime flat_layout = row_major[B * C_OUT * L_OUT]()

    var x_t = TileTensor(x_buf, in_layout)
    var w_t = TileTensor(w_buf, w_layout)
    var bias_t = TileTensor(bias_buf, p_layout)
    var conv_t = TileTensor(conv_out, out_layout)
    var bn_w_t = TileTensor(bn_w_buf, p_layout)
    var bn_b_t = TileTensor(bn_b_buf, p_layout)
    var bn_m_t = TileTensor(bn_m_buf, p_layout)
    var bn_v_t = TileTensor(bn_v_buf, p_layout)
    var bn_t = TileTensor(bn_out, out_layout)
    var bn_flat = TileTensor(bn_out, flat_layout)
    var out_flat = TileTensor(out_buf, flat_layout)

    comptime conv_k = conv1d_kernel_fast[
        DType.float32, type_of(in_layout), type_of(w_layout),
        type_of(p_layout), type_of(out_layout),
        K, False, BLOCK,
    ]
    ctx.enqueue_function[conv_k, conv_k](
        conv_t, x_t, w_t, bias_t,
        B, C_IN, C_OUT, L_IN, L_OUT, STRIDE, PADDING, DILATION,
        grid_dim=B * C_OUT, block_dim=BLOCK,
    )
    comptime bn_k = batchnorm1d_kernel[
        DType.float32, type_of(out_layout), type_of(p_layout),
        type_of(out_layout), BLOCK,
    ]
    ctx.enqueue_function[bn_k, bn_k](
        bn_t, conv_t, bn_w_t, bn_b_t, bn_m_t, bn_v_t,
        B, C_OUT, L_OUT, EPS,
        grid_dim=B * C_OUT, block_dim=BLOCK,
    )
    comptime relu_k = relu_kernel[
        DType.float32, type_of(flat_layout), type_of(flat_layout), BLOCK,
    ]
    ctx.enqueue_function[relu_k, relu_k](
        out_flat, bn_flat, n_out,
        grid_dim=ceildiv(n_out, BLOCK), block_dim=BLOCK,
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
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-3)
    print("CAMPPlus xvector.tdnn — max abs:", max_abs,
          " mean:", sum_abs / Float64(n_out))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
