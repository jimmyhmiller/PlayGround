"""LSTM parity test (single layer) — reuses mojo-t3 fixture.

B=1, T=5, INPUT=8, HIDDEN=16. Checks pure-MAX LSTM (linalg.matmul + elementwise)
against torch nn.LSTM reference output.
"""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer

from fixture import load_fp32
from lstm import LSTMLayer, lstm_layer_forward


comptime B = 1
comptime T = 5
comptime INPUT = 8
comptime HIDDEN = 16


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def test_lstm_single_layer() raises:
    comptime assert has_accelerator(), "Requires GPU"
    # Reuse fixtures from the mojo-t3 project — same torch oracle.
    var fix = "../mojo-t3/tests/fixtures/lstm/"
    var ctx = DeviceContext()

    var x_t = load_fp32(fix + "x.bin")
    var exp_t = load_fp32(fix + "out.bin")
    var w_ih_t = load_fp32(fix + "weight_ih_l0.bin")
    var w_hh_t = load_fp32(fix + "weight_hh_l0.bin")
    var b_ih_t = load_fp32(fix + "bias_ih_l0.bin")
    var b_hh_t = load_fp32(fix + "bias_hh_l0.bin")

    var x_buf = ctx.enqueue_create_buffer[DType.float32](B * T * INPUT)
    upload(x_buf, x_t.data, B * T * INPUT)
    var w_ih_buf = ctx.enqueue_create_buffer[DType.float32](4 * HIDDEN * INPUT)
    upload(w_ih_buf, w_ih_t.data, 4 * HIDDEN * INPUT)
    var w_hh_buf = ctx.enqueue_create_buffer[DType.float32](4 * HIDDEN * HIDDEN)
    upload(w_hh_buf, w_hh_t.data, 4 * HIDDEN * HIDDEN)
    var b_ih_buf = ctx.enqueue_create_buffer[DType.float32](4 * HIDDEN)
    upload(b_ih_buf, b_ih_t.data, 4 * HIDDEN)
    var b_hh_buf = ctx.enqueue_create_buffer[DType.float32](4 * HIDDEN)
    upload(b_hh_buf, b_hh_t.data, 4 * HIDDEN)

    # Outputs + state.
    var h_seq_buf = ctx.enqueue_create_buffer[DType.float32](B * T * HIDDEN)
    var h_state_buf = ctx.enqueue_create_buffer[DType.float32](B * HIDDEN)
    var c_state_buf = ctx.enqueue_create_buffer[DType.float32](B * HIDDEN)
    h_state_buf.enqueue_fill(0.0)
    c_state_buf.enqueue_fill(0.0)
    # Scratch.
    var pre_xw_buf = ctx.enqueue_create_buffer[DType.float32](B * 4 * HIDDEN)
    var pre_hw_buf = ctx.enqueue_create_buffer[DType.float32](B * 4 * HIDDEN)

    var layer = LSTMLayer(w_ih_buf, w_hh_buf, b_ih_buf, b_hh_buf, INPUT, HIDDEN)
    lstm_layer_forward(
        ctx, layer, x_buf, h_seq_buf, h_state_buf, c_state_buf,
        pre_xw_buf, pre_hw_buf, B, T,
    )
    ctx.synchronize()

    var n_out = B * T * HIDDEN
    var max_abs: Float32 = 0.0
    with h_seq_buf.map_to_host() as h:
        for i in range(n_out):
            var d = h[i] - exp_t.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            if i < 8:
                print("lstm[", i, "]: max= ", h[i], " torch= ", exp_t.data[i], " diff= ", d)
            assert_almost_equal(h[i], exp_t.data[i], atol=2.0e-5)
    print("LSTM single-layer (B=1, T=5, IN=8, H=16) — max abs:", max_abs)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
