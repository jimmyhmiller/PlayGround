"""
Parity test for the encoder.pre_lookahead_layer (after the embed).

Input:  enc_embed_xs.bin      (1, 376, 512)
Target: enc_pre_lookahead.bin (1, 376, 512)

Sequence:
  outputs = inputs.transpose(1, 2)                    # (B, C, T)
  outputs = F.pad(outputs, (0, 3), value=0.0)         # right-pad T by 3
  outputs = F.leaky_relu(conv1(outputs))              # k=4 (3+1), pad=0; collapses pad
  outputs = F.pad(outputs, (2, 0), value=0.0)         # left-pad T by 2
  outputs = conv2(outputs)                            # k=3, pad=0; collapses pad
  outputs = outputs.transpose(1, 2)                   # (B, T, C)
  outputs = outputs + inputs                          # residual

We use conv1d_kernel_fast with padding=0 (for conv1: right-pad via bounds) and
padding=2 (for conv2: left-pad).
"""
from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import load_fp32
from conv import conv1d_kernel_fast, leaky_relu_kernel
from layernorm import (
    transpose_btc_to_bct_kernel, transpose_bct_to_btc_kernel, residual_add_kernel,
)


comptime B = 1
comptime T = 376
comptime C = 512
comptime K1 = 4   # pre_lookahead_len + 1
comptime K2 = 3
comptime BLOCK = 256


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def test_pre_lookahead() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/s3gen/"
    var ctx = DeviceContext()

    var x_in = load_fp32(fix + "enc_embed_xs.bin")    # (1, 376, 512)
    var w_c1 = load_fp32(fix + "weights/flow__encoder__pre_lookahead_layer__conv1__weight.bin")  # (C, C, 4)
    var b_c1 = load_fp32(fix + "weights/flow__encoder__pre_lookahead_layer__conv1__bias.bin")
    var w_c2 = load_fp32(fix + "weights/flow__encoder__pre_lookahead_layer__conv2__weight.bin")  # (C, C, 3)
    var b_c2 = load_fp32(fix + "weights/flow__encoder__pre_lookahead_layer__conv2__bias.bin")
    var exp = load_fp32(fix + "enc_pre_lookahead.bin")

    var n_x = B * T * C
    var n_w1 = C * C * K1
    var n_w2 = C * C * K2

    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var w_c1_buf = ctx.enqueue_create_buffer[DType.float32](n_w1)
    var b_c1_buf = ctx.enqueue_create_buffer[DType.float32](C)
    var w_c2_buf = ctx.enqueue_create_buffer[DType.float32](n_w2)
    var b_c2_buf = ctx.enqueue_create_buffer[DType.float32](C)
    var bct_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var conv1_out = ctx.enqueue_create_buffer[DType.float32](n_x)
    var relu_out = ctx.enqueue_create_buffer[DType.float32](n_x)
    var conv2_out = ctx.enqueue_create_buffer[DType.float32](n_x)
    var btc_back = ctx.enqueue_create_buffer[DType.float32](n_x)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_x)

    upload(x_buf, x_in.data, n_x)
    upload(w_c1_buf, w_c1.data, n_w1)
    upload(b_c1_buf, b_c1.data, C)
    upload(w_c2_buf, w_c2.data, n_w2)
    upload(b_c2_buf, b_c2.data, C)

    comptime btc_layout = row_major[B, T, C]()
    comptime bct_layout = row_major[B, C, T]()
    comptime w1_layout = row_major[C, C, K1]()
    comptime w2_layout = row_major[C, C, K2]()
    comptime p_layout = row_major[C]()
    comptime flat = row_major[B * T * C]()

    var x_t = TileTensor(x_buf, btc_layout)
    var bct_t = TileTensor(bct_buf, bct_layout)
    var w_c1_t = TileTensor(w_c1_buf, w1_layout)
    var b_c1_t = TileTensor(b_c1_buf, p_layout)
    var w_c2_t = TileTensor(w_c2_buf, w2_layout)
    var b_c2_t = TileTensor(b_c2_buf, p_layout)
    var conv1_out_t = TileTensor(conv1_out, bct_layout)
    var conv1_out_flat = TileTensor(conv1_out, flat)
    var relu_out_flat = TileTensor(relu_out, flat)
    var relu_out_t = TileTensor(relu_out, bct_layout)
    var conv2_out_t = TileTensor(conv2_out, bct_layout)
    var btc_back_t = TileTensor(btc_back, btc_layout)
    var x_flat = TileTensor(x_buf, flat)
    var btc_back_flat = TileTensor(btc_back, flat)
    var out_flat = TileTensor(out_buf, flat)

    # Transpose (B, T, C) -> (B, C, T).
    comptime tp1_k = transpose_btc_to_bct_kernel[
        DType.float32, type_of(btc_layout), type_of(bct_layout), BLOCK,
    ]
    ctx.enqueue_function[tp1_k, tp1_k](
        bct_t, x_t, B, T, C,
        grid_dim=B * C, block_dim=BLOCK,
    )

    # conv1: k=4, padding=0; right-pad of 3 emulated by bounds check (l_out = T).
    comptime conv1_k = conv1d_kernel_fast[
        DType.float32, type_of(bct_layout), type_of(w1_layout),
        type_of(p_layout), type_of(bct_layout),
        K1, True, BLOCK,
    ]
    ctx.enqueue_function[conv1_k, conv1_k](
        conv1_out_t, bct_t, w_c1_t, b_c1_t,
        B, C, C, T, T, 1, 0, 1,
        grid_dim=B * C, block_dim=BLOCK,
    )

    # leaky_relu (default negative_slope=0.01 per PyTorch).
    comptime relu_k = leaky_relu_kernel[
        DType.float32, type_of(flat), type_of(flat), BLOCK,
    ]
    ctx.enqueue_function[relu_k, relu_k](
        relu_out_flat, conv1_out_flat, n_x, Float32(0.01),
        grid_dim=ceildiv(n_x, BLOCK), block_dim=BLOCK,
    )

    # conv2: k=3, padding=2 (left only); right side fits exactly.
    comptime conv2_k = conv1d_kernel_fast[
        DType.float32, type_of(bct_layout), type_of(w2_layout),
        type_of(p_layout), type_of(bct_layout),
        K2, True, BLOCK,
    ]
    ctx.enqueue_function[conv2_k, conv2_k](
        conv2_out_t, relu_out_t, w_c2_t, b_c2_t,
        B, C, C, T, T, 1, 2, 1,
        grid_dim=B * C, block_dim=BLOCK,
    )

    # Transpose back (B, C, T) -> (B, T, C).
    comptime tp2_k = transpose_bct_to_btc_kernel[
        DType.float32, type_of(bct_layout), type_of(btc_layout), BLOCK,
    ]
    ctx.enqueue_function[tp2_k, tp2_k](
        btc_back_t, conv2_out_t, B, C, T,
        grid_dim=B * T, block_dim=BLOCK,
    )

    # Residual.
    comptime add_k = residual_add_kernel[
        DType.float32, type_of(flat), type_of(flat), type_of(flat), BLOCK,
    ]
    ctx.enqueue_function[add_k, add_k](
        out_flat, btc_back_flat, x_flat, n_x,
        grid_dim=ceildiv(n_x, BLOCK), block_dim=BLOCK,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    var sum_abs: Float64 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n_x):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_abs += Float64(d)
            if i < 8:
                print("plk[", i, "]: mojo=", h[i], "  torch=", exp.data[i], "  diff=", d)
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-3)
    print("encoder.pre_lookahead — max abs:", max_abs,
          " mean:", sum_abs / Float64(n_x))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
