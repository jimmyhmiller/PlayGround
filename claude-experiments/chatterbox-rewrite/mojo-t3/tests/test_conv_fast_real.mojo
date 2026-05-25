"""Compare fast vs slow conv1d at real-HiFiGAN production scale."""
from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import load_fp32
from conv import conv1d_kernel, conv1d_kernel_fast


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def test_conv_pre_real() raises:
    comptime assert has_accelerator(), "Requires GPU"
    comptime BATCH = 1
    comptime MEL_C = 80
    comptime MEL_T = 262
    comptime PRE_C = 512
    comptime K = 7

    var fix = "tests/fixtures/real/"
    var w_fix = "tests/fixtures/hifigan/weights/"
    var mel = load_fp32(fix + "real_mel.bin")
    var w = load_fp32(w_fix + "conv_pre__weight.bin")
    var b = load_fp32(w_fix + "conv_pre__bias.bin")

    var n_x = BATCH * MEL_C * MEL_T
    var n_w = PRE_C * MEL_C * K
    var n_out = BATCH * PRE_C * MEL_T

    var ctx = DeviceContext()
    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var w_buf = ctx.enqueue_create_buffer[DType.float32](n_w)
    var b_buf = ctx.enqueue_create_buffer[DType.float32](PRE_C)
    var slow_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    var fast_buf = ctx.enqueue_create_buffer[DType.float32](n_out)

    upload(x_buf, mel.data, n_x)
    upload(w_buf, w.data, n_w)
    upload(b_buf, b.data, PRE_C)

    comptime x_layout = row_major[BATCH, MEL_C, MEL_T]()
    comptime w_layout = row_major[PRE_C, MEL_C, K]()
    comptime b_layout = row_major[PRE_C]()
    comptime out_layout = row_major[BATCH, PRE_C, MEL_T]()

    var x_t = TileTensor(x_buf, x_layout)
    var w_t = TileTensor(w_buf, w_layout)
    var b_t = TileTensor(b_buf, b_layout)
    var slow_t = TileTensor(slow_buf, out_layout)
    var fast_t = TileTensor(fast_buf, out_layout)

    comptime slow_k = conv1d_kernel[
        DType.float32, type_of(x_layout), type_of(w_layout),
        type_of(b_layout), type_of(out_layout), K, True,
    ]
    ctx.enqueue_function[slow_k, slow_k](
        slow_t, x_t, w_t, b_t,
        BATCH, MEL_C, PRE_C, MEL_T, MEL_T, 1, 3, 1,
        grid_dim=BATCH * PRE_C * MEL_T, block_dim=1,
    )
    comptime BLOCK = 256
    comptime fast_k = conv1d_kernel_fast[
        DType.float32, type_of(x_layout), type_of(w_layout),
        type_of(b_layout), type_of(out_layout), K, True, BLOCK,
    ]
    ctx.enqueue_function[fast_k, fast_k](
        fast_t, x_t, w_t, b_t,
        BATCH, MEL_C, PRE_C, MEL_T, MEL_T, 1, 3, 1,
        grid_dim=BATCH * PRE_C, block_dim=BLOCK,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    var n_diff: Int = 0
    with slow_buf.map_to_host() as s:
        with fast_buf.map_to_host() as f:
            for i in range(n_out):
                var d = s[i] - f[i]
                if d < 0.0: d = -d
                if d > max_abs: max_abs = d
                if d > 1.0e-4:
                    n_diff += 1
    print("conv_pre fast vs slow on real data — max abs:", max_abs,
          " mismatches (>1e-4):", n_diff, " of ", n_out)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
