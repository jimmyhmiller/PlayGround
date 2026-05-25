"""Conv1d parity test using existing torch fixture."""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer

from fixture import load_fp32
from conv1d import Conv1d, conv1d_forward


comptime B = 1
comptime C_IN = 4
comptime C_OUT = 6
comptime L = 16
comptime L_OUT = 16
comptime K = 7
comptime STRIDE = 1
comptime PAD = 3
comptime DIL = 1


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def test_conv1d() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "../mojo-t3/tests/fixtures/conv/"
    var ctx = DeviceContext()

    var x_t = load_fp32(fix + "conv1d_x.bin")
    var w_t = load_fp32(fix + "conv1d_w.bin")
    var b_t = load_fp32(fix + "conv1d_bias.bin")
    var exp_t = load_fp32(fix + "conv1d_expected.bin")

    var n_x = B * C_IN * L
    var n_w = C_OUT * C_IN * K
    var n_b = C_OUT
    var n_y = B * C_OUT * L_OUT

    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var w_buf = ctx.enqueue_create_buffer[DType.float32](n_w)
    var b_buf = ctx.enqueue_create_buffer[DType.float32](n_b)
    var y_buf = ctx.enqueue_create_buffer[DType.float32](n_y)
    upload(x_buf, x_t.data, n_x)
    upload(w_buf, w_t.data, n_w)
    upload(b_buf, b_t.data, n_b)

    var conv = Conv1d(w_buf, b_buf, C_IN, C_OUT, K, STRIDE, PAD, DIL, 1, True)
    conv1d_forward(ctx, conv, x_buf, y_buf, B, L, L_OUT)
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    with y_buf.map_to_host() as h:
        for i in range(n_y):
            var d = h[i] - exp_t.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            if i < 8:
                print("c[", i, "]: max=", h[i], " torch=", exp_t.data[i], " diff=", d)
            assert_almost_equal(h[i], exp_t.data[i], atol=1.0e-5)
    print("Conv1d (B=1, C_in=4, C_out=6, L=16, K=7) — max abs:", max_abs)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
