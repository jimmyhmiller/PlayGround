"""
Parity test for the first BasicResBlock of CAMPPlus (head.layer1[0]).

Structure:
  out = ReLU(BN2d(Conv2d(x))           # 3x3, stride=(2,1), pad=1
        → BN2d(Conv2d(prev))            # 3x3, stride=1, pad=1
        + shortcut(x))
  shortcut: Conv2d(1x1, stride=(2,1)) + BN2d

Input  (1, 32, 80, 998)
Output (1, 32, 40, 998)
"""
from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import Tensor, load_fp32
from conv import conv2d_kernel, batchnorm2d_kernel, relu_kernel
from util_kernels import add_kernel


comptime BATCH = 1
comptime C = 32
comptime H_IN = 80
comptime W = 998
comptime H_OUT = 40   # stride_h=2
comptime KH = 3
comptime KW = 3
comptime POINTWISE_BLOCK = 256
comptime EPS: Float32 = 1.0e-5


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def test_basic_resblock_stride2() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/campplus/"
    var x_in = load_fp32(fix + "fcm_relu1_out.bin")           # (1, 32, 80, 998)
    var w_c1 = load_fp32(fix + "weights/head__layer1__0__conv1__weight.bin")  # (32, 32, 3, 3)
    var w_c2 = load_fp32(fix + "weights/head__layer1__0__conv2__weight.bin")
    var w_sc = load_fp32(fix + "weights/head__layer1__0__shortcut__0__weight.bin")  # (32, 32, 1, 1)
    var bn1_w = load_fp32(fix + "weights/head__layer1__0__bn1__weight.bin")
    var bn1_b = load_fp32(fix + "weights/head__layer1__0__bn1__bias.bin")
    var bn1_m = load_fp32(fix + "weights/head__layer1__0__bn1__running_mean.bin")
    var bn1_v = load_fp32(fix + "weights/head__layer1__0__bn1__running_var.bin")
    var bn2_w = load_fp32(fix + "weights/head__layer1__0__bn2__weight.bin")
    var bn2_b = load_fp32(fix + "weights/head__layer1__0__bn2__bias.bin")
    var bn2_m = load_fp32(fix + "weights/head__layer1__0__bn2__running_mean.bin")
    var bn2_v = load_fp32(fix + "weights/head__layer1__0__bn2__running_var.bin")
    var sc_w = load_fp32(fix + "weights/head__layer1__0__shortcut__1__weight.bin")
    var sc_b = load_fp32(fix + "weights/head__layer1__0__shortcut__1__bias.bin")
    var sc_m = load_fp32(fix + "weights/head__layer1__0__shortcut__1__running_mean.bin")
    var sc_v = load_fp32(fix + "weights/head__layer1__0__shortcut__1__running_var.bin")
    var exp = load_fp32(fix + "fcm_layer1_block0_out.bin")    # (1, 32, 40, 998)

    var n_in = BATCH * C * H_IN * W
    var n_out = BATCH * C * H_OUT * W
    var n_w_3x3 = C * C * KH * KW
    var n_w_1x1 = C * C

    var ctx = DeviceContext()

    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_in)
    var w_c1_buf = ctx.enqueue_create_buffer[DType.float32](n_w_3x3)
    var w_c2_buf = ctx.enqueue_create_buffer[DType.float32](n_w_3x3)
    var w_sc_buf = ctx.enqueue_create_buffer[DType.float32](n_w_1x1)
    var bn1_w_buf = ctx.enqueue_create_buffer[DType.float32](C)
    var bn1_b_buf = ctx.enqueue_create_buffer[DType.float32](C)
    var bn1_m_buf = ctx.enqueue_create_buffer[DType.float32](C)
    var bn1_v_buf = ctx.enqueue_create_buffer[DType.float32](C)
    var bn2_w_buf = ctx.enqueue_create_buffer[DType.float32](C)
    var bn2_b_buf = ctx.enqueue_create_buffer[DType.float32](C)
    var bn2_m_buf = ctx.enqueue_create_buffer[DType.float32](C)
    var bn2_v_buf = ctx.enqueue_create_buffer[DType.float32](C)
    var sc_w_buf = ctx.enqueue_create_buffer[DType.float32](C)
    var sc_b_buf = ctx.enqueue_create_buffer[DType.float32](C)
    var sc_m_buf = ctx.enqueue_create_buffer[DType.float32](C)
    var sc_v_buf = ctx.enqueue_create_buffer[DType.float32](C)
    var dummy_bias = ctx.enqueue_create_buffer[DType.float32](C)

    var conv1_out_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    var bn1_out_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    var conv2_out_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    var bn2_out_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    var sc_conv_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    var sc_bn_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    var sum_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_out)

    upload(x_buf, x_in.data, n_in)
    upload(w_c1_buf, w_c1.data, n_w_3x3)
    upload(w_c2_buf, w_c2.data, n_w_3x3)
    upload(w_sc_buf, w_sc.data, n_w_1x1)
    upload(bn1_w_buf, bn1_w.data, C)
    upload(bn1_b_buf, bn1_b.data, C)
    upload(bn1_m_buf, bn1_m.data, C)
    upload(bn1_v_buf, bn1_v.data, C)
    upload(bn2_w_buf, bn2_w.data, C)
    upload(bn2_b_buf, bn2_b.data, C)
    upload(bn2_m_buf, bn2_m.data, C)
    upload(bn2_v_buf, bn2_v.data, C)
    upload(sc_w_buf, sc_w.data, C)
    upload(sc_b_buf, sc_b.data, C)
    upload(sc_m_buf, sc_m.data, C)
    upload(sc_v_buf, sc_v.data, C)

    comptime in_layout = row_major[BATCH, C, H_IN, W]()
    comptime out_layout = row_major[BATCH, C, H_OUT, W]()
    comptime w_3x3_layout = row_major[C, C, KH, KW]()
    comptime w_1x1_layout = row_major[C, C, 1, 1]()
    comptime p_layout = row_major[C]()
    comptime flat_layout = row_major[1, BATCH * C * H_OUT * W]()

    var x_t = TileTensor(x_buf, in_layout)
    var w_c1_t = TileTensor(w_c1_buf, w_3x3_layout)
    var w_c2_t = TileTensor(w_c2_buf, w_3x3_layout)
    var w_sc_t = TileTensor(w_sc_buf, w_1x1_layout)
    var bn1_w_t = TileTensor(bn1_w_buf, p_layout)
    var bn1_b_t = TileTensor(bn1_b_buf, p_layout)
    var bn1_m_t = TileTensor(bn1_m_buf, p_layout)
    var bn1_v_t = TileTensor(bn1_v_buf, p_layout)
    var bn2_w_t = TileTensor(bn2_w_buf, p_layout)
    var bn2_b_t = TileTensor(bn2_b_buf, p_layout)
    var bn2_m_t = TileTensor(bn2_m_buf, p_layout)
    var bn2_v_t = TileTensor(bn2_v_buf, p_layout)
    var sc_w_t = TileTensor(sc_w_buf, p_layout)
    var sc_b_t = TileTensor(sc_b_buf, p_layout)
    var sc_m_t = TileTensor(sc_m_buf, p_layout)
    var sc_v_t = TileTensor(sc_v_buf, p_layout)
    var dummy_t = TileTensor(dummy_bias, p_layout)

    var conv1_out_t = TileTensor(conv1_out_buf, out_layout)
    var bn1_out_t = TileTensor(bn1_out_buf, out_layout)
    var conv2_out_t = TileTensor(conv2_out_buf, out_layout)
    var bn2_out_t = TileTensor(bn2_out_buf, out_layout)
    var sc_conv_t = TileTensor(sc_conv_buf, out_layout)
    var sc_bn_t = TileTensor(sc_bn_buf, out_layout)
    var sum_t = TileTensor(sum_buf, out_layout)

    var bn2_out_flat = TileTensor(bn2_out_buf, flat_layout)
    var sc_bn_flat = TileTensor(sc_bn_buf, flat_layout)
    var sum_flat = TileTensor(sum_buf, flat_layout)
    var out_flat_1d = row_major[BATCH * C * H_OUT * W]()
    var sum_flat_1d_view = TileTensor(sum_buf, out_flat_1d)
    var out_flat_1d_view = TileTensor(out_buf, out_flat_1d)

    # Conv2d 3x3 stride=(2,1) pad=1, no bias.
    comptime conv3x3_k = conv2d_kernel[
        DType.float32, type_of(in_layout), type_of(w_3x3_layout),
        type_of(p_layout), type_of(out_layout),
        KH, KW, False, 256,
    ]
    ctx.enqueue_function[conv3x3_k, conv3x3_k](
        conv1_out_t, x_t, w_c1_t, dummy_t,
        BATCH, C, C, H_IN, W, H_OUT, W, 2, 1, 1, 1,
        grid_dim=BATCH * C * H_OUT, block_dim=256,
    )
    # BN2d.
    comptime bn_k = batchnorm2d_kernel[
        DType.float32, type_of(out_layout), type_of(p_layout),
        type_of(out_layout), 256,
    ]
    ctx.enqueue_function[bn_k, bn_k](
        bn1_out_t, conv1_out_t, bn1_w_t, bn1_b_t, bn1_m_t, bn1_v_t,
        BATCH, C, H_OUT, W, EPS,
        grid_dim=BATCH * C * H_OUT, block_dim=256,
    )
    # ReLU (apply to bn1_out in-place via flat).
    var bn1_out_flat = TileTensor(bn1_out_buf, out_flat_1d)
    comptime relu_k = relu_kernel[
        DType.float32, type_of(out_flat_1d), type_of(out_flat_1d), POINTWISE_BLOCK,
    ]
    # Need a non-aliased dst — write into conv2_out as scratch then overwrite.
    # Simpler: write the relu-of-bn1 result into a fresh buffer via the relu kernel.
    var relu1_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    var relu1_flat = TileTensor(relu1_buf, out_flat_1d)
    ctx.enqueue_function[relu_k, relu_k](
        relu1_flat, bn1_out_flat, n_out,
        grid_dim=ceildiv(n_out, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )
    # conv2 (stride 1) on relu1.
    var relu1_t = TileTensor(relu1_buf, out_layout)
    # conv2d_kernel parameterized for 3x3 with H_IN=H_OUT (stride=1).
    comptime conv3x3_same_k = conv2d_kernel[
        DType.float32, type_of(out_layout), type_of(w_3x3_layout),
        type_of(p_layout), type_of(out_layout),
        KH, KW, False, 256,
    ]
    ctx.enqueue_function[conv3x3_same_k, conv3x3_same_k](
        conv2_out_t, relu1_t, w_c2_t, dummy_t,
        BATCH, C, C, H_OUT, W, H_OUT, W, 1, 1, 1, 1,
        grid_dim=BATCH * C * H_OUT, block_dim=256,
    )
    # BN2d on conv2.
    ctx.enqueue_function[bn_k, bn_k](
        bn2_out_t, conv2_out_t, bn2_w_t, bn2_b_t, bn2_m_t, bn2_v_t,
        BATCH, C, H_OUT, W, EPS,
        grid_dim=BATCH * C * H_OUT, block_dim=256,
    )
    # Shortcut: Conv2d 1x1 stride=(2,1) on x.
    comptime conv1x1_k = conv2d_kernel[
        DType.float32, type_of(in_layout), type_of(w_1x1_layout),
        type_of(p_layout), type_of(out_layout),
        1, 1, False, 256,
    ]
    ctx.enqueue_function[conv1x1_k, conv1x1_k](
        sc_conv_t, x_t, w_sc_t, dummy_t,
        BATCH, C, C, H_IN, W, H_OUT, W, 2, 1, 0, 0,
        grid_dim=BATCH * C * H_OUT, block_dim=256,
    )
    # BN2d on shortcut.
    ctx.enqueue_function[bn_k, bn_k](
        sc_bn_t, sc_conv_t, sc_w_t, sc_b_t, sc_m_t, sc_v_t,
        BATCH, C, H_OUT, W, EPS,
        grid_dim=BATCH * C * H_OUT, block_dim=256,
    )
    # sum = bn2_out + sc_bn (using 2D add kernel via flat layout).
    comptime add_k = add_kernel[
        DType.float32, type_of(flat_layout), type_of(flat_layout),
        type_of(flat_layout), POINTWISE_BLOCK,
    ]
    ctx.enqueue_function[add_k, add_k](
        sum_flat, bn2_out_flat, sc_bn_flat, n_out,
        grid_dim=ceildiv(n_out, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )
    # out = relu(sum)
    ctx.enqueue_function[relu_k, relu_k](
        out_flat_1d_view, sum_flat_1d_view, n_out,
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
    print("CAMPPlus layer1[0] BasicResBlock (real weights, stride=2) — max abs:", max_abs,
          " mean:", sum_abs / Float64(n_out))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
