"""
Parity test for seg_pool_kernel against ctx_seg fixture from upstream.
ctx_seg shape: (1, 128, 499)
"""
from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import load_fp32
from cam_kernels import seg_pool_kernel


comptime B = 1
comptime C = 128
comptime T = 499
comptime SEG_LEN = 100
comptime BLOCK = 256


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def test_seg_pool() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/campplus/"
    var ctx = DeviceContext()

    var nl2 = load_fp32(fix + "tdnnd1_nl2.bin")     # (1, 128, 499)
    var exp = load_fp32(fix + "tdnnd1_cam_ctx_seg.bin")  # (1, 128, 499)
    var n = B * C * T

    var x_buf = ctx.enqueue_create_buffer[DType.float32](n)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n)
    upload(x_buf, nl2.data, n)

    comptime in_layout = row_major[B, C, T]()
    var x_t = TileTensor(x_buf, in_layout)
    var out_t = TileTensor(out_buf, in_layout)

    comptime sp_k = seg_pool_kernel[
        DType.float32, type_of(in_layout), type_of(in_layout),
        SEG_LEN, BLOCK,
    ]
    ctx.enqueue_function[sp_k, sp_k](
        out_t, x_t, B, C, T,
        grid_dim=B * C, block_dim=BLOCK,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    var sum_abs: Float64 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_abs += Float64(d)
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-5)
    print("seg_pool — max abs:", max_abs, " mean:", sum_abs / Float64(n))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
