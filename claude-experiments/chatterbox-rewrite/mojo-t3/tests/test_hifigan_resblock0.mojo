"""
HiFiGAN parity test: a single real ResBlock at upsample stage 0.

resblocks.0 (i=0, j=0) is the first of three resblocks at the i=0 stage:
  channels = 256, K = 3, dilations = [1, 3, 5].

Validates our ResBlock kernel composition (snake + conv1d chain + residual)
at real HiFiGAN scale and against real weights.

Input:  stage_up0_pre_resblocks.bin  (1, 256, 256)
Output: stage_up0_resblock0_out.bin  (1, 256, 256)
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import Tensor, load_fp32
from conv import conv1d_kernel, snake_kernel
from util_kernels import add_kernel


comptime BATCH = 1
comptime C = 256
comptime T = 256
comptime K = 3
comptime N_DILS = 3
# Padding for K=3: dilation 1 -> 1, dilation 3 -> 3, dilation 5 -> 5.
# convs2 always uses dilation=1, padding=1.


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def test_resblock0_fp32() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/hifigan/"
    var x_in = load_fp32(fix + "stage_up0_pre_resblocks.bin")
    var exp = load_fp32(fix + "stage_up0_resblock0_out.bin")

    assert_equal(x_in.shape[1], C)
    assert_equal(x_in.shape[2], T)

    var n = BATCH * C * T
    var n_w = C * C * K

    var ctx = DeviceContext()
    var x_buf = ctx.enqueue_create_buffer[DType.float32](n)
    var x_next_buf = ctx.enqueue_create_buffer[DType.float32](n)
    var xt_buf = ctx.enqueue_create_buffer[DType.float32](n)
    var xt2_buf = ctx.enqueue_create_buffer[DType.float32](n)
    var w_buf = ctx.enqueue_create_buffer[DType.float32](n_w)
    var bias_buf = ctx.enqueue_create_buffer[DType.float32](C)
    var alpha_buf = ctx.enqueue_create_buffer[DType.float32](C)
    upload(x_buf, x_in.data, n)

    comptime x_layout = row_major[BATCH, C, T]()
    comptime w_layout = row_major[C, C, K]()
    comptime bias_layout = row_major[C]()
    comptime alpha_layout = row_major[C]()
    comptime flat_layout = row_major[1, BATCH * C * T]()

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
    comptime SNAKE_BLOCK = 256
    comptime snake_k = snake_kernel[
        DType.float32, type_of(x_layout), type_of(alpha_layout), type_of(x_layout),
        SNAKE_BLOCK,
    ]
    comptime add_k = add_kernel[
        DType.float32, type_of(flat_layout), type_of(flat_layout),
        type_of(flat_layout), 256,
    ]

    var dilations = List[Int]()
    dilations.append(1)
    dilations.append(3)
    dilations.append(5)

    for j in range(N_DILS):
        var dil = dilations[j]
        var pad1 = ((K - 1) * dil) // 2  # for convs1
        var pad2 = ((K - 1) * 1) // 2    # for convs2 (always dil=1)

        # resblocks.0.convs1.{j}.{weight,bias}, .convs2.{j}.{weight,bias},
        # .activations1.{j}.alpha, .activations2.{j}.alpha
        var w1 = load_fp32(fix + "weights/resblocks__0__convs1__" + String(j) + "__weight.bin")
        var b1 = load_fp32(fix + "weights/resblocks__0__convs1__" + String(j) + "__bias.bin")
        var w2 = load_fp32(fix + "weights/resblocks__0__convs2__" + String(j) + "__weight.bin")
        var b2 = load_fp32(fix + "weights/resblocks__0__convs2__" + String(j) + "__bias.bin")
        var a1 = load_fp32(fix + "weights/resblocks__0__activations1__" + String(j) + "__alpha.bin")
        var a2 = load_fp32(fix + "weights/resblocks__0__activations2__" + String(j) + "__alpha.bin")

        # xt = snake(x, a1)
        upload(alpha_buf, a1.data, C)
        ctx.enqueue_function[snake_k, snake_k](
            xt_t, x_t, alpha_t, BATCH, C, T,
            grid_dim=BATCH * C, block_dim=SNAKE_BLOCK,
        )
        # xt = conv1d(xt, w1, b1, dilation=dil, padding=pad1)
        upload(w_buf, w1.data, n_w)
        upload(bias_buf, b1.data, C)
        ctx.enqueue_function[conv_k, conv_k](
            xt2_t, xt_t, w_t, bias_t,
            BATCH, C, C, T, T, 1, pad1, dil,
            grid_dim=BATCH * C * T, block_dim=1,
        )
        # xt = snake(xt, a2)
        upload(alpha_buf, a2.data, C)
        ctx.enqueue_function[snake_k, snake_k](
            xt_t, xt2_t, alpha_t, BATCH, C, T,
            grid_dim=BATCH * C, block_dim=SNAKE_BLOCK,
        )
        # xt2 = conv1d(xt, w2, b2, dilation=1, padding=pad2)
        upload(w_buf, w2.data, n_w)
        upload(bias_buf, b2.data, C)
        ctx.enqueue_function[conv_k, conv_k](
            xt2_t, xt_t, w_t, bias_t,
            BATCH, C, C, T, T, 1, pad2, 1,
            grid_dim=BATCH * C * T, block_dim=1,
        )
        # x_next = x + xt2 (residual)
        ctx.enqueue_function[add_k, add_k](
            x_next_flat, x_flat, xt2_flat, n,
            grid_dim=ceildiv(n, 256), block_dim=256,
        )
        ctx.synchronize()
        # Copy x_next → x for the next dilation iteration.
        with x_next_buf.map_to_host() as src:
            with x_buf.map_to_host() as dst:
                for k in range(n):
                    dst[k] = src[k]

    var max_abs: Float32 = 0.0
    var sum_abs: Float64 = 0.0
    with x_buf.map_to_host() as h:
        for i in range(n):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_abs += Float64(d)
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-3)
    print("HiFiGAN resblocks.0 fp32 (real weights, C=256, T=256) — max abs:", max_abs,
          " mean abs:", sum_abs / Float64(n))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
