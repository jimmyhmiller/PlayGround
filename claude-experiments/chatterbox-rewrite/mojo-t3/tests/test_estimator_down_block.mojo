"""
Parity test for the FULL down_block 0: resnet + 4 transformer blocks + downsample.

Input:  estimator_x_full.bin           (2, 320, 752)  — packed [x, mu, spks, cond]
        cfm_step_00_input_mask.bin     (2, 1, 752)
        estimator_time_mlp_out_real.bin (2, 1024)
Target: estimator_down_post_ds.bin     (2, 256, 752)  — after CausalConv1d downsample.
"""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer

from fixture import load_fp32
from cfm_decoder import (
    causal_resnet_block_1d, CausalResnetWeights,
    basic_transformer_block, BasicTransformerWeights,
    causal_conv1d_with_mask,
    transpose_with_mask_bct_btc, transpose_with_mask_btc_bct,
)


comptime B = 2
comptime IN_C = 320
comptime D = 256
comptime T = 752
comptime TIME_EMB_DIM = 1024
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


def load_resnet(mut ctx: DeviceContext, fix: String, prefix: String) raises -> CausalResnetWeights:
    return CausalResnetWeights(
        upload_w(ctx, fix, prefix + "block1__block__0__weight.bin"),
        upload_w(ctx, fix, prefix + "block1__block__0__bias.bin"),
        upload_w(ctx, fix, prefix + "block1__block__2__weight.bin"),
        upload_w(ctx, fix, prefix + "block1__block__2__bias.bin"),
        upload_w(ctx, fix, prefix + "block2__block__0__weight.bin"),
        upload_w(ctx, fix, prefix + "block2__block__0__bias.bin"),
        upload_w(ctx, fix, prefix + "block2__block__2__weight.bin"),
        upload_w(ctx, fix, prefix + "block2__block__2__bias.bin"),
        upload_w(ctx, fix, prefix + "mlp__1__weight.bin"),
        upload_w(ctx, fix, prefix + "mlp__1__bias.bin"),
        upload_w(ctx, fix, prefix + "res_conv__weight.bin"),
        upload_w(ctx, fix, prefix + "res_conv__bias.bin"),
    )


def load_tblock(mut ctx: DeviceContext, fix: String, prefix: String) raises -> BasicTransformerWeights:
    return BasicTransformerWeights(
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


def test_down_block_full() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/s3gen/"
    var ctx = DeviceContext()

    var x_in = load_fp32(fix + "estimator_x_full.bin")
    var mask = load_fp32(fix + "cfm_step_00_input_mask.bin")
    var t_emb = load_fp32(fix + "estimator_time_mlp_out_real.bin")
    var exp = load_fp32(fix + "estimator_down_post_ds.bin")

    var resnet_w = load_resnet(ctx, fix,
        "weights/flow__decoder__estimator__down_blocks__0__0__")
    var tb0_w = load_tblock(ctx, fix,
        "weights/flow__decoder__estimator__down_blocks__0__1__0__")
    var tb1_w = load_tblock(ctx, fix,
        "weights/flow__decoder__estimator__down_blocks__0__1__1__")
    var tb2_w = load_tblock(ctx, fix,
        "weights/flow__decoder__estimator__down_blocks__0__1__2__")
    var tb3_w = load_tblock(ctx, fix,
        "weights/flow__decoder__estimator__down_blocks__0__1__3__")
    # downsample: ds is the third element of down_blocks[0]. Weight key: down_blocks__0__2__weight.bin
    var ds_w = upload_w(ctx, fix, "weights/flow__decoder__estimator__down_blocks__0__2__weight.bin")
    var ds_b = upload_w(ctx, fix, "weights/flow__decoder__estimator__down_blocks__0__2__bias.bin")

    var n_x = B * IN_C * T
    var n_mask = B * 1 * T
    var n_te = B * TIME_EMB_DIM
    var n_d = B * D * T

    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var mask_buf = ctx.enqueue_create_buffer[DType.float32](n_mask)
    var t_emb_buf = ctx.enqueue_create_buffer[DType.float32](n_te)
    var resnet_out = ctx.enqueue_create_buffer[DType.float32](n_d)
    var btc_buf = ctx.enqueue_create_buffer[DType.float32](n_d)
    var tb0_out = ctx.enqueue_create_buffer[DType.float32](n_d)
    var tb1_out = ctx.enqueue_create_buffer[DType.float32](n_d)
    var tb2_out = ctx.enqueue_create_buffer[DType.float32](n_d)
    var tb3_out = ctx.enqueue_create_buffer[DType.float32](n_d)
    var bct_back = ctx.enqueue_create_buffer[DType.float32](n_d)
    var ds_out = ctx.enqueue_create_buffer[DType.float32](n_d)
    upload(x_buf, x_in.data, n_x)
    upload(mask_buf, mask.data, n_mask)
    upload(t_emb_buf, t_emb.data, n_te)

    # 1. Resnet.
    causal_resnet_block_1d[B, IN_C, D, T, TIME_EMB_DIM](
        ctx, x_buf, mask_buf, t_emb_buf, resnet_out, resnet_w,
    )
    # 2. Transpose to (B, T, D) for transformers.
    transpose_with_mask_bct_btc[B, D, T](ctx, resnet_out, btc_buf)
    # 3. 4 transformer blocks.
    basic_transformer_block[B, T, D, H, D_K, FF_INNER](ctx, btc_buf, tb0_out, tb0_w)
    basic_transformer_block[B, T, D, H, D_K, FF_INNER](ctx, tb0_out, tb1_out, tb1_w)
    basic_transformer_block[B, T, D, H, D_K, FF_INNER](ctx, tb1_out, tb2_out, tb2_w)
    basic_transformer_block[B, T, D, H, D_K, FF_INNER](ctx, tb2_out, tb3_out, tb3_w)
    # 4. Transpose back (B, T, D) -> (B, D, T).
    transpose_with_mask_btc_bct[B, T, D](ctx, tb3_out, bct_back)
    # 5. Downsample = CausalConv1d k=3 with mask multiply.
    causal_conv1d_with_mask[B, D, D, T, 3](
        ctx, bct_back, mask_buf, ds_out, ds_w, ds_b,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    var sum_abs: Float64 = 0.0
    with ds_out.map_to_host() as h:
        for i in range(n_d):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_abs += Float64(d)
            if i < 8:
                print("dn[", i, "]: mojo=", h[i], "  torch=", exp.data[i], "  diff=", d)
            assert_almost_equal(h[i], exp.data[i], atol=2.0e-2)
    print("estimator down_block 0 (resnet + 4 tblocks + ds) — max abs:", max_abs,
          " mean:", sum_abs / Float64(n_d))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
