"""
Parity test for the first BasicTransformerBlock in down_blocks[0][1][0].

Input:  estimator_resnet0_out.bin → transposed to (2, 752, 256)
Target: estimator_tblock0_out.bin   (2, 752, 256)
"""
from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu import block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major

from fixture import load_fp32
from layernorm import transpose_bct_to_btc_kernel
from cfm_decoder import basic_transformer_block, BasicTransformerWeights


comptime B = 2
comptime T = 752
comptime D = 256
comptime H = 8
comptime D_K = 64
comptime FF_INNER = 1024


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


def test_tblock0() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/s3gen/"
    var ctx = DeviceContext()

    # Resnet0 output is (B, C=256, T=752). Transformer takes (B, T, C).
    var x_in = load_fp32(fix + "estimator_resnet0_out.bin")   # (2, 256, 752)
    var exp = load_fp32(fix + "estimator_tblock0_out.bin")    # (2, 752, 256)

    var prefix = "weights/flow__decoder__estimator__down_blocks__0__1__0__"
    var w = BasicTransformerWeights(
        upload_w(ctx, fix, prefix + "norm1__weight.bin"),
        upload_w(ctx, fix, prefix + "norm1__bias.bin"),
        upload_w(ctx, fix, prefix + "attn1__to_q__weight.bin"),
        upload_w(ctx, fix, prefix + "attn1__to_k__weight.bin"),
        upload_w(ctx, fix, prefix + "attn1__to_v__weight.bin"),
        upload_w(ctx, fix, prefix + "attn1__to_out__0__weight.bin"),
        upload_w(ctx, fix, prefix + "attn1__to_out__0__bias.bin"),
        upload_w(ctx, fix, prefix + "norm3__weight.bin"),
        upload_w(ctx, fix, prefix + "norm3__bias.bin"),
        upload_w(ctx, fix, prefix + "ff__net__0__proj__weight.bin"),
        upload_w(ctx, fix, prefix + "ff__net__0__proj__bias.bin"),
        upload_w(ctx, fix, prefix + "ff__net__2__weight.bin"),
        upload_w(ctx, fix, prefix + "ff__net__2__bias.bin"),
    )

    var n_x = B * T * D
    var x_bct_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var x_btc_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    upload(x_bct_buf, x_in.data, n_x)

    comptime bct_layout = row_major[B, D, T]()
    comptime btc_layout = row_major[B, T, D]()
    var x_bct_t = TileTensor(x_bct_buf, bct_layout)
    var x_btc_t = TileTensor(x_btc_buf, btc_layout)
    comptime tp_k = transpose_bct_to_btc_kernel[
        DType.float32, type_of(bct_layout), type_of(btc_layout), 256,
    ]
    ctx.enqueue_function[tp_k, tp_k](
        x_btc_t, x_bct_t, B, D, T,
        grid_dim=B * T, block_dim=256,
    )

    basic_transformer_block[B, T, D, H, D_K, FF_INNER](
        ctx, x_btc_buf, out_buf, w,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    var sum_abs: Float64 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n_x):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_abs += Float64(d)
            if i < 8:
                print("tb0[", i, "]: mojo=", h[i], "  torch=", exp.data[i], "  diff=", d)
            assert_almost_equal(h[i], exp.data[i], atol=2.0e-3)
    print("estimator down_blocks[0][1][0] BasicTransformerBlock — max abs:", max_abs,
          " mean:", sum_abs / Float64(n_x))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
