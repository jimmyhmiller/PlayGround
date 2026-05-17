"""
HiFiGAN parity tests against real Chatterbox HiFTGenerator weights:
  1. conv_pre(mel)                            — conv1d at (1,80,32) -> (1,512,32)
  2. leaky_relu(.)                             — checks our LR kernel on real data
  3. ups[0] transposed_conv1d                  — first upsample, 32 -> 256 samples
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import Tensor, load_fp32
from conv import conv1d_kernel, transposed_conv1d_kernel, leaky_relu_kernel


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


comptime POINTWISE_BLOCK = 256

# First upsample stage: ups[0] = ConvTranspose1d(512 -> 256, K=16, stride=8, padding=4).
comptime UPS0_C_IN = 512
comptime UPS0_C_OUT = 256
comptime UPS0_K = 16
comptime UPS0_STRIDE = 8
comptime UPS0_PADDING = 4
comptime UPS0_T_IN = 32     # = T_MEL
comptime UPS0_T_OUT = 256   # (32 - 1) * 8 - 8 + 16 = 256


def test_leaky_relu_after_conv_pre_fp32() raises:
    """Verify leaky_relu(conv_pre(mel)) matches stage_up0_after_lrelu."""
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/hifigan/"
    var x_in = load_fp32(fix + "stage_after_conv_pre.bin")
    var exp = load_fp32(fix + "stage_up0_after_lrelu.bin")

    var n = BATCH * C_OUT * T_MEL

    var ctx = DeviceContext()
    var x_buf = ctx.enqueue_create_buffer[DType.float32](n)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n)
    upload(x_buf, x_in.data, n)

    comptime flat_layout = row_major[BATCH * C_OUT * T_MEL]()
    var x_t = TileTensor(x_buf, flat_layout)
    var out_t = TileTensor(out_buf, flat_layout)

    comptime kernel = leaky_relu_kernel[
        DType.float32, type_of(flat_layout), type_of(flat_layout), POINTWISE_BLOCK,
    ]
    ctx.enqueue_function[kernel, kernel](
        out_t, x_t, n, Float32(0.1),
        grid_dim=ceildiv(n, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            assert_almost_equal(h[i], exp.data[i], atol=0.0)
    print("leaky_relu fp32 (after conv_pre) — max abs:", max_abs)


def test_ups0_fp32() raises:
    """Verify ups[0] transposed_conv1d matches stage_up0_after_transposed_conv.

    Uses the upstream-dumped lrelu output as input (so this test isolates
    the transposed conv).
    """
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/hifigan/"
    var x_in = load_fp32(fix + "stage_up0_after_lrelu.bin")
    var w = load_fp32(fix + "weights/ups__0__weight.bin")
    var bias = load_fp32(fix + "weights/ups__0__bias.bin")
    var exp = load_fp32(fix + "stage_up0_after_transposed_conv.bin")

    assert_equal(w.shape[0], UPS0_C_IN)
    assert_equal(w.shape[1], UPS0_C_OUT)
    assert_equal(w.shape[2], UPS0_K)
    assert_equal(bias.shape[0], UPS0_C_OUT)
    assert_equal(exp.shape[1], UPS0_C_OUT)
    assert_equal(exp.shape[2], UPS0_T_OUT)

    var n_x = BATCH * UPS0_C_IN * UPS0_T_IN
    var n_w = UPS0_C_IN * UPS0_C_OUT * UPS0_K
    var n_out = BATCH * UPS0_C_OUT * UPS0_T_OUT

    var ctx = DeviceContext()
    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var w_buf = ctx.enqueue_create_buffer[DType.float32](n_w)
    var bias_buf = ctx.enqueue_create_buffer[DType.float32](UPS0_C_OUT)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    upload(x_buf, x_in.data, n_x)
    upload(w_buf, w.data, n_w)
    upload(bias_buf, bias.data, UPS0_C_OUT)

    comptime x_layout = row_major[BATCH, UPS0_C_IN, UPS0_T_IN]()
    comptime w_layout = row_major[UPS0_C_IN, UPS0_C_OUT, UPS0_K]()
    comptime bias_layout = row_major[UPS0_C_OUT]()
    comptime out_layout = row_major[BATCH, UPS0_C_OUT, UPS0_T_OUT]()

    var x_t = TileTensor(x_buf, x_layout)
    var w_t = TileTensor(w_buf, w_layout)
    var bias_t = TileTensor(bias_buf, bias_layout)
    var out_t = TileTensor(out_buf, out_layout)

    comptime kernel = transposed_conv1d_kernel[
        DType.float32, type_of(x_layout), type_of(w_layout),
        type_of(bias_layout), type_of(out_layout), UPS0_K, True,
    ]
    ctx.enqueue_function[kernel, kernel](
        out_t, x_t, w_t, bias_t,
        BATCH, UPS0_C_IN, UPS0_C_OUT, UPS0_T_IN, UPS0_T_OUT,
        UPS0_STRIDE, UPS0_PADDING, 1,
        grid_dim=BATCH * UPS0_C_OUT * UPS0_T_OUT, block_dim=1,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n_out):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-4)
    print("ups[0] transposed_conv1d fp32 (real weights) — max abs:", max_abs)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
