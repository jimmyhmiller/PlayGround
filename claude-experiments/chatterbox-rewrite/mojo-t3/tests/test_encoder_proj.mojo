"""
Parity test for the encoder_proj linear projection.

Input:  encoder_h.bin       (1, 752, 512) — encoder output
Target: encoder_proj_h.bin  (1, 752, 80)

This is `flow.encoder_proj`: nn.Linear(512, 80, bias=True).
"""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import load_fp32
from layernorm import linear_kernel


comptime B = 1
comptime T = 752
comptime D_IN = 512
comptime D_OUT = 80
comptime BLOCK = 256


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def test_encoder_proj() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/s3gen/"
    var ctx = DeviceContext()

    var x_in = load_fp32(fix + "encoder_h.bin")
    var w = load_fp32(fix + "weights/flow__encoder_proj__weight.bin")
    var b = load_fp32(fix + "weights/flow__encoder_proj__bias.bin")
    var exp = load_fp32(fix + "encoder_proj_h.bin")

    var n_in = B * T * D_IN
    var n_w = D_OUT * D_IN
    var n_out = B * T * D_OUT

    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_in)
    var w_buf = ctx.enqueue_create_buffer[DType.float32](n_w)
    var b_buf = ctx.enqueue_create_buffer[DType.float32](D_OUT)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    upload(x_buf, x_in.data, n_in)
    upload(w_buf, w.data, n_w)
    upload(b_buf, b.data, D_OUT)

    comptime in_layout = row_major[B, T, D_IN]()
    comptime w_layout = row_major[D_OUT, D_IN]()
    comptime out_layout = row_major[B, T, D_OUT]()
    comptime p_layout = row_major[D_OUT]()
    var x_t = TileTensor(x_buf, in_layout)
    var w_t = TileTensor(w_buf, w_layout)
    var b_t = TileTensor(b_buf, p_layout)
    var out_t = TileTensor(out_buf, out_layout)

    comptime lin_k = linear_kernel[
        DType.float32, type_of(in_layout), type_of(w_layout),
        type_of(p_layout), type_of(out_layout),
        True, BLOCK,
    ]
    ctx.enqueue_function[lin_k, lin_k](
        out_t, x_t, w_t, b_t, B, T, D_IN, D_OUT,
        grid_dim=B * T, block_dim=BLOCK,
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
            if i < 8:
                print("proj[", i, "]: mojo=", h[i], "  torch=", exp.data[i], "  diff=", d)
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-4)
    print("encoder_proj — max abs:", max_abs, " mean:", sum_abs / Float64(n_out))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
