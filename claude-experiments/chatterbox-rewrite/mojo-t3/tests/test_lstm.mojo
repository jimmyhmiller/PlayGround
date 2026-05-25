"""Parity test for the Mojo LSTM kernel vs torch nn.LSTM (single layer)."""
from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import load_fp32
from lstm import lstm_layer_first_kernel


comptime B = 1
comptime T = 5
comptime INPUT = 8
comptime HIDDEN = 16
comptime BLOCK = 32


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def test_lstm() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/lstm/"
    var ctx = DeviceContext()

    var x = load_fp32(fix + "x.bin")
    var w_ih = load_fp32(fix + "weight_ih_l0.bin")
    var w_hh = load_fp32(fix + "weight_hh_l0.bin")
    var b_ih = load_fp32(fix + "bias_ih_l0.bin")
    var b_hh = load_fp32(fix + "bias_hh_l0.bin")
    var exp = load_fp32(fix + "out.bin")

    var n_x = B * T * INPUT
    var n_out = B * T * HIDDEN
    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var w_ih_buf = ctx.enqueue_create_buffer[DType.float32](4 * HIDDEN * INPUT)
    var w_hh_buf = ctx.enqueue_create_buffer[DType.float32](4 * HIDDEN * HIDDEN)
    var b_ih_buf = ctx.enqueue_create_buffer[DType.float32](4 * HIDDEN)
    var b_hh_buf = ctx.enqueue_create_buffer[DType.float32](4 * HIDDEN)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    upload(x_buf, x.data, n_x)
    upload(w_ih_buf, w_ih.data, 4 * HIDDEN * INPUT)
    upload(w_hh_buf, w_hh.data, 4 * HIDDEN * HIDDEN)
    upload(b_ih_buf, b_ih.data, 4 * HIDDEN)
    upload(b_hh_buf, b_hh.data, 4 * HIDDEN)

    comptime x_layout = row_major[B, T, INPUT]()
    comptime h_layout = row_major[B, T, HIDDEN]()
    comptime w_ih_layout = row_major[4 * HIDDEN, INPUT]()
    comptime w_hh_layout = row_major[4 * HIDDEN, HIDDEN]()
    comptime b_layout = row_major[4 * HIDDEN]()

    var x_t = TileTensor(x_buf, x_layout)
    var w_ih_t = TileTensor(w_ih_buf, w_ih_layout)
    var w_hh_t = TileTensor(w_hh_buf, w_hh_layout)
    var b_ih_t = TileTensor(b_ih_buf, b_layout)
    var b_hh_t = TileTensor(b_hh_buf, b_layout)
    var out_t = TileTensor(out_buf, h_layout)

    comptime k = lstm_layer_first_kernel[
        DType.float32, type_of(x_layout), type_of(h_layout),
        type_of(w_ih_layout), type_of(w_hh_layout), type_of(b_layout),
        HIDDEN, INPUT, BLOCK,
    ]
    ctx.enqueue_function[k, k](
        out_t, x_t, w_ih_t, w_hh_t, b_ih_t, b_hh_t,
        B, T,
        grid_dim=B, block_dim=BLOCK,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n_out):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            if i < 8:
                print("lstm[", i, "]: mojo=", h[i], "  torch=", exp.data[i], "  diff=", d)
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-5)
    print("LSTM (B=1, T=5, I=8, H=16) — max abs:", max_abs)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
