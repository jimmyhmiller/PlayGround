"""Parity test for FSQ codebook (project_down + tanh + round + powers-of-3)."""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import load_fp32, load_i32
from fsq_codebook import fsq_encode_kernel


comptime B = 2
comptime T = 16
comptime D = 1280
comptime BLOCK = 1   # only thread 0 of each block does work


def upload_f32(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def test_fsq() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/fsq/"
    var ctx = DeviceContext()

    var x_t = load_fp32(fix + "x.bin")
    var w_t = load_fp32(fix + "project_down_w.bin")
    var b_t = load_fp32(fix + "project_down_b.bin")
    var exp = load_i32(fix + "idx.bin")

    var x_buf = ctx.enqueue_create_buffer[DType.float32](B * T * D)
    var w_buf = ctx.enqueue_create_buffer[DType.float32](8 * D)
    var b_buf = ctx.enqueue_create_buffer[DType.float32](8)
    var out_buf = ctx.enqueue_create_buffer[DType.int32](B * T)

    upload_f32(x_buf, x_t.data, B * T * D)
    upload_f32(w_buf, w_t.data, 8 * D)
    upload_f32(b_buf, b_t.data, 8)

    comptime x_layout = row_major[B, T, D]()
    comptime w_layout = row_major[8, D]()
    comptime b_layout = row_major[8]()
    comptime out_layout = row_major[B, T]()

    var x_tt = TileTensor(x_buf, x_layout)
    var w_tt = TileTensor(w_buf, w_layout)
    var b_tt = TileTensor(b_buf, b_layout)
    var out_tt = TileTensor(out_buf, out_layout)

    comptime k = fsq_encode_kernel[
        DType.float32, type_of(x_layout), type_of(w_layout),
        type_of(b_layout), type_of(out_layout), D, BLOCK,
    ]
    ctx.enqueue_function[k, k](
        out_tt, x_tt, w_tt, b_tt, B, T,
        grid_dim=B * T, block_dim=BLOCK,
    )
    ctx.synchronize()

    var n_out = B * T
    var num_mismatch = 0
    with out_buf.map_to_host() as h:
        for i in range(n_out):
            var got = Int(h[i])
            var want = Int(exp.data[i])
            if got != want:
                num_mismatch += 1
                if num_mismatch < 5:
                    print("FSQ mismatch[", i, "]: mojo=", got, " torch=", want)
            assert_equal(got, want)
    print("FSQ encode (B=2, T=16, D=1280) — all", n_out, "indices match")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
