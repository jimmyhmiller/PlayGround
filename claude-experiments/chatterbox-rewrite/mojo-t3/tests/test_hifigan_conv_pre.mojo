"""
HiFiGAN parity test: stage 1 — conv_pre(mel).

Validates our conv1d kernel against real HiFTGenerator weights at real shape:
  mel  (1, 80, 32)
  w    (512, 80, 7)  — conv_pre.weight
  bias (512,)        — conv_pre.bias
  out  (1, 512, 32)  — matches upstream's `stage_after_conv_pre`

This is the first real-weight test of the HiFiGAN port; subsequent stages
build on this.
"""

from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import Tensor, load_fp32
from conv import conv1d_kernel


comptime BATCH = 1
comptime C_IN = 80
comptime C_OUT = 512
comptime T_MEL = 32
comptime K = 7
comptime STRIDE = 1
comptime PADDING = 3
comptime DILATION = 1


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def test_conv_pre_fp32() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/hifigan/"
    var mel = load_fp32(fix + "mel.bin")
    var w = load_fp32(fix + "weights/conv_pre__weight.bin")
    var bias = load_fp32(fix + "weights/conv_pre__bias.bin")
    var exp = load_fp32(fix + "stage_after_conv_pre.bin")

    assert_equal(mel.shape[0], BATCH)
    assert_equal(mel.shape[1], C_IN)
    assert_equal(mel.shape[2], T_MEL)
    assert_equal(w.shape[0], C_OUT)
    assert_equal(w.shape[1], C_IN)
    assert_equal(w.shape[2], K)
    assert_equal(exp.shape[1], C_OUT)
    assert_equal(exp.shape[2], T_MEL)

    var n_x = BATCH * C_IN * T_MEL
    var n_w = C_OUT * C_IN * K
    var n_out = BATCH * C_OUT * T_MEL

    var ctx = DeviceContext()
    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var w_buf = ctx.enqueue_create_buffer[DType.float32](n_w)
    var bias_buf = ctx.enqueue_create_buffer[DType.float32](C_OUT)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    upload(x_buf, mel.data, n_x)
    upload(w_buf, w.data, n_w)
    upload(bias_buf, bias.data, C_OUT)

    comptime x_layout = row_major[BATCH, C_IN, T_MEL]()
    comptime w_layout = row_major[C_OUT, C_IN, K]()
    comptime bias_layout = row_major[C_OUT]()
    comptime out_layout = row_major[BATCH, C_OUT, T_MEL]()

    var x_t = TileTensor(x_buf, x_layout)
    var w_t = TileTensor(w_buf, w_layout)
    var bias_t = TileTensor(bias_buf, bias_layout)
    var out_t = TileTensor(out_buf, out_layout)

    comptime kernel = conv1d_kernel[
        DType.float32, type_of(x_layout), type_of(w_layout),
        type_of(bias_layout), type_of(out_layout), K, True,
    ]
    ctx.enqueue_function[kernel, kernel](
        out_t, x_t, w_t, bias_t,
        BATCH, C_IN, C_OUT, T_MEL, T_MEL, STRIDE, PADDING, DILATION,
        grid_dim=BATCH * C_OUT * T_MEL, block_dim=1,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n_out):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            # Real-weight conv1d: tolerance covers fp32 reduction-order
            # differences vs PyTorch (still small).
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-4)
    print("conv_pre fp32 (real T3 weights) — max abs:", max_abs)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
