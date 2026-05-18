"""Sanity test: conv1d_kernel_fast must match conv1d_kernel on the same input."""
from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import load_fp32, load_i64
from conv import conv1d_kernel, conv1d_kernel_fast


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def test_conv1d_fast_matches_slow() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/conv/"
    var x = load_fp32(fix + "conv1d_x.bin")
    var w = load_fp32(fix + "conv1d_w.bin")
    var bias = load_fp32(fix + "conv1d_bias.bin")
    var meta = load_i64(fix + "conv1d_meta.bin")

    var B = Int(meta.data[0])
    var C_in = Int(meta.data[1])
    var C_out = Int(meta.data[2])
    var L_in = Int(meta.data[3])
    var L_out = Int(meta.data[4])
    comptime K = 7
    var stride = Int(meta.data[6])
    var padding = Int(meta.data[7])
    var dilation = Int(meta.data[8])
    comptime BLOCK = 256

    var n_x = B * C_in * L_in
    var n_w = C_out * C_in * K
    var n_out = B * C_out * L_out

    var ctx = DeviceContext()
    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var w_buf = ctx.enqueue_create_buffer[DType.float32](n_w)
    var bias_buf = ctx.enqueue_create_buffer[DType.float32](C_out)
    var slow_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    var fast_buf = ctx.enqueue_create_buffer[DType.float32](n_out)

    upload(x_buf, x.data, n_x)
    upload(w_buf, w.data, n_w)
    upload(bias_buf, bias.data, C_out)

    comptime x_layout = row_major[1, 4, 16]()
    comptime w_layout = row_major[6, 4, K]()
    comptime b_layout = row_major[6]()
    comptime out_layout = row_major[1, 6, 16]()

    var x_t = TileTensor(x_buf, x_layout)
    var w_t = TileTensor(w_buf, w_layout)
    var b_t = TileTensor(bias_buf, b_layout)
    var slow_t = TileTensor(slow_buf, out_layout)
    var fast_t = TileTensor(fast_buf, out_layout)

    # slow
    comptime slow_k = conv1d_kernel[
        DType.float32, type_of(x_layout), type_of(w_layout),
        type_of(b_layout), type_of(out_layout), K, True,
    ]
    ctx.enqueue_function[slow_k, slow_k](
        slow_t, x_t, w_t, b_t,
        B, C_in, C_out, L_in, L_out, stride, padding, dilation,
        grid_dim=B * C_out * L_out, block_dim=1,
    )
    # fast
    comptime fast_k = conv1d_kernel_fast[
        DType.float32, type_of(x_layout), type_of(w_layout),
        type_of(b_layout), type_of(out_layout), K, True, BLOCK,
    ]
    ctx.enqueue_function[fast_k, fast_k](
        fast_t, x_t, w_t, b_t,
        B, C_in, C_out, L_in, L_out, stride, padding, dilation,
        grid_dim=B * C_out, block_dim=BLOCK,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    with slow_buf.map_to_host() as s:
        with fast_buf.map_to_host() as f:
            for i in range(n_out):
                var d = s[i] - f[i]
                if d < 0.0: d = -d
                if d > max_abs: max_abs = d
                assert_almost_equal(f[i], s[i], atol=1.0e-6)
    print("conv1d_fast vs slow — max abs:", max_abs)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
