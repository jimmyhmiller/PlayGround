"""3-layer LSTM (40 → 256 → 256 → 256) — mirrors VoiceEncoder structure."""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import load_fp32
from lstm import lstm_layer_first_kernel


comptime B = 1
comptime T = 10
comptime INPUT = 40
comptime HIDDEN = 256
comptime BLOCK = 128


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def upload_w(mut ctx: DeviceContext, fix: String, name: String) raises -> DeviceBuffer[DType.float32]:
    var t = load_fp32(fix + name)
    var n = len(t.data)
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    upload(buf, t.data, n)
    return buf^


def test_lstm3() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/lstm3/"
    var ctx = DeviceContext()

    var x = load_fp32(fix + "x.bin")
    var exp = load_fp32(fix + "out.bin")

    var x_buf = ctx.enqueue_create_buffer[DType.float32](B * T * INPUT)
    var h0_buf = ctx.enqueue_create_buffer[DType.float32](B * T * HIDDEN)
    var h1_buf = ctx.enqueue_create_buffer[DType.float32](B * T * HIDDEN)
    var h2_buf = ctx.enqueue_create_buffer[DType.float32](B * T * HIDDEN)
    upload(x_buf, x.data, B * T * INPUT)

    var w_ih_0 = upload_w(ctx, fix, "weight_ih_l0.bin")
    var w_hh_0 = upload_w(ctx, fix, "weight_hh_l0.bin")
    var b_ih_0 = upload_w(ctx, fix, "bias_ih_l0.bin")
    var b_hh_0 = upload_w(ctx, fix, "bias_hh_l0.bin")
    var w_ih_1 = upload_w(ctx, fix, "weight_ih_l1.bin")
    var w_hh_1 = upload_w(ctx, fix, "weight_hh_l1.bin")
    var b_ih_1 = upload_w(ctx, fix, "bias_ih_l1.bin")
    var b_hh_1 = upload_w(ctx, fix, "bias_hh_l1.bin")
    var w_ih_2 = upload_w(ctx, fix, "weight_ih_l2.bin")
    var w_hh_2 = upload_w(ctx, fix, "weight_hh_l2.bin")
    var b_ih_2 = upload_w(ctx, fix, "bias_ih_l2.bin")
    var b_hh_2 = upload_w(ctx, fix, "bias_hh_l2.bin")

    comptime x0_layout = row_major[B, T, INPUT]()
    comptime h_layout = row_major[B, T, HIDDEN]()
    comptime w_ih_0_layout = row_major[4 * HIDDEN, INPUT]()
    comptime w_ih_hh_layout = row_major[4 * HIDDEN, HIDDEN]()
    comptime b_layout = row_major[4 * HIDDEN]()

    var x_t = TileTensor(x_buf, x0_layout)
    var h0_t = TileTensor(h0_buf, h_layout)
    var h1_t = TileTensor(h1_buf, h_layout)
    var h2_t = TileTensor(h2_buf, h_layout)
    var w_ih_0_t = TileTensor(w_ih_0, w_ih_0_layout)
    var w_hh_0_t = TileTensor(w_hh_0, w_ih_hh_layout)
    var b_ih_0_t = TileTensor(b_ih_0, b_layout)
    var b_hh_0_t = TileTensor(b_hh_0, b_layout)
    var w_ih_1_t = TileTensor(w_ih_1, w_ih_hh_layout)
    var w_hh_1_t = TileTensor(w_hh_1, w_ih_hh_layout)
    var b_ih_1_t = TileTensor(b_ih_1, b_layout)
    var b_hh_1_t = TileTensor(b_hh_1, b_layout)
    var w_ih_2_t = TileTensor(w_ih_2, w_ih_hh_layout)
    var w_hh_2_t = TileTensor(w_hh_2, w_ih_hh_layout)
    var b_ih_2_t = TileTensor(b_ih_2, b_layout)
    var b_hh_2_t = TileTensor(b_hh_2, b_layout)

    # Layer 0: input=40, hidden=256.
    comptime k0 = lstm_layer_first_kernel[
        DType.float32, type_of(x0_layout), type_of(h_layout),
        type_of(w_ih_0_layout), type_of(w_ih_hh_layout), type_of(b_layout),
        HIDDEN, INPUT, BLOCK,
    ]
    ctx.enqueue_function[k0, k0](
        h0_t, x_t, w_ih_0_t, w_hh_0_t, b_ih_0_t, b_hh_0_t,
        B, T, grid_dim=B, block_dim=BLOCK,
    )
    # Layer 1: input=256, hidden=256.
    comptime k1 = lstm_layer_first_kernel[
        DType.float32, type_of(h_layout), type_of(h_layout),
        type_of(w_ih_hh_layout), type_of(w_ih_hh_layout), type_of(b_layout),
        HIDDEN, HIDDEN, BLOCK,
    ]
    ctx.enqueue_function[k1, k1](
        h1_t, h0_t, w_ih_1_t, w_hh_1_t, b_ih_1_t, b_hh_1_t,
        B, T, grid_dim=B, block_dim=BLOCK,
    )
    # Layer 2.
    ctx.enqueue_function[k1, k1](
        h2_t, h1_t, w_ih_2_t, w_hh_2_t, b_ih_2_t, b_hh_2_t,
        B, T, grid_dim=B, block_dim=BLOCK,
    )
    ctx.synchronize()

    var n_out = B * T * HIDDEN
    var max_abs: Float32 = 0.0
    with h2_buf.map_to_host() as h:
        for i in range(n_out):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            if i < 8:
                print("lstm3[", i, "]: mojo=", h[i], "  torch=", exp.data[i], "  diff=", d)
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-5)
    print("3-layer LSTM (40 → 256 × 3) — max abs:", max_abs)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
