"""Compare transposed_conv1d_kernel_fast vs slow at production scale."""
from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import load_fp32
from conv import transposed_conv1d_kernel, transposed_conv1d_kernel_fast


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def test_ups0_real() raises:
    comptime assert has_accelerator(), "Requires GPU"
    comptime BATCH = 1
    comptime IN_C = 512
    comptime IN_T = 262
    comptime OUT_C = 256
    comptime OUT_T = 2096
    comptime K = 16
    comptime STRIDE = 8
    comptime PAD = 4
    comptime BLOCK = 256

    var w_fix = "tests/fixtures/hifigan/weights/"
    # Use stage_after_conv_pre as input — but it's at T_MEL=32. We need
    # an input at T=262. Build synthetic input.
    var n_x = BATCH * IN_C * IN_T
    var n_w = IN_C * OUT_C * K
    var n_out = BATCH * OUT_C * OUT_T

    var w = load_fp32(w_fix + "ups__0__weight.bin")
    var b = load_fp32(w_fix + "ups__0__bias.bin")

    var ctx = DeviceContext()
    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var w_buf = ctx.enqueue_create_buffer[DType.float32](n_w)
    var b_buf = ctx.enqueue_create_buffer[DType.float32](OUT_C)
    var slow_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    var fast_buf = ctx.enqueue_create_buffer[DType.float32](n_out)

    # Synthetic random-ish input (no need for real data — we're testing
    # whether two kernels agree).
    with x_buf.map_to_host() as h:
        for i in range(n_x): h[i] = Float32(i % 100) * 0.01 - 0.5
    upload(w_buf, w.data, n_w)
    upload(b_buf, b.data, OUT_C)

    comptime x_layout = row_major[BATCH, IN_C, IN_T]()
    comptime w_layout = row_major[IN_C, OUT_C, K]()
    comptime b_layout = row_major[OUT_C]()
    comptime out_layout = row_major[BATCH, OUT_C, OUT_T]()

    var x_t = TileTensor(x_buf, x_layout)
    var w_t = TileTensor(w_buf, w_layout)
    var b_t = TileTensor(b_buf, b_layout)
    var slow_t = TileTensor(slow_buf, out_layout)
    var fast_t = TileTensor(fast_buf, out_layout)

    comptime slow_k = transposed_conv1d_kernel[
        DType.float32, type_of(x_layout), type_of(w_layout),
        type_of(b_layout), type_of(out_layout), K, True,
    ]
    ctx.enqueue_function[slow_k, slow_k](
        slow_t, x_t, w_t, b_t,
        BATCH, IN_C, OUT_C, IN_T, OUT_T, STRIDE, PAD, 1,
        grid_dim=BATCH * OUT_C * OUT_T, block_dim=1,
    )
    comptime fast_k = transposed_conv1d_kernel_fast[
        DType.float32, type_of(x_layout), type_of(w_layout),
        type_of(b_layout), type_of(out_layout), K, True, BLOCK,
    ]
    ctx.enqueue_function[fast_k, fast_k](
        fast_t, x_t, w_t, b_t,
        BATCH, IN_C, OUT_C, IN_T, OUT_T, STRIDE, PAD, 1,
        grid_dim=BATCH * OUT_C, block_dim=BLOCK,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    with slow_buf.map_to_host() as s:
        with fast_buf.map_to_host() as f:
            for i in range(n_out):
                var d = s[i] - f[i]
                if d < 0.0: d = -d
                if d > max_abs: max_abs = d
    print("ups[0] fast vs slow on production-scale weights — max abs:", max_abs)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
