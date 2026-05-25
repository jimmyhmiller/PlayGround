"""
Parity test for the first CausalBlock1D in down_blocks[0][0].block1.
This isolates one of the building blocks.

Input:  estimator_x_full.bin           (2, 320, 752)  — packed [x, mu, spks, cond]
        cfm_step_00_input_mask.bin     (2, 1, 752)
Target: estimator_resnet0_block1.bin   (2, 256, 752)
"""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer

from fixture import load_fp32
from cfm_decoder import causal_block_1d


comptime B = 2
comptime IN_C = 320
comptime OUT_C = 256
comptime T = 752


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


def test_block1() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/s3gen/"
    var ctx = DeviceContext()

    var x_in = load_fp32(fix + "estimator_x_full.bin")
    var mask = load_fp32(fix + "cfm_step_00_input_mask.bin")
    var exp = load_fp32(fix + "estimator_resnet0_block1.bin")

    var prefix = "weights/flow__decoder__estimator__down_blocks__0__0__block1__block__"
    var cw = upload_w(ctx, fix, prefix + "0__weight.bin")
    var cb = upload_w(ctx, fix, prefix + "0__bias.bin")
    var lw = upload_w(ctx, fix, prefix + "2__weight.bin")
    var lb = upload_w(ctx, fix, prefix + "2__bias.bin")

    var n_x = B * IN_C * T
    var n_mask = B * 1 * T
    var n_out = B * OUT_C * T

    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var mask_buf = ctx.enqueue_create_buffer[DType.float32](n_mask)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    upload(x_buf, x_in.data, n_x)
    upload(mask_buf, mask.data, n_mask)

    causal_block_1d[B, IN_C, OUT_C, T, 3](
        ctx, x_buf, mask_buf, out_buf, cw, cb, lw, lb,
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
                print("b1[", i, "]: mojo=", h[i], "  torch=", exp.data[i], "  diff=", d)
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-3)
    print("estimator down_blocks[0][0] block1 — max abs:", max_abs,
          " mean:", sum_abs / Float64(n_out))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
