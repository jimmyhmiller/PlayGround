"""
Verifies the `estimator_forward` helper produces the same output as the inline
pipeline in test_estimator_full.mojo.

Input:  estimator_x_full.bin  (2, 320, 752)
        cfm_step_00_input_mask.bin
        estimator_time_mlp_out_real.bin
Target: estimator_final_full.bin (2, 80, 752)
"""
from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer

from fixture import load_fp32
from cfm_decoder import (
    estimator_forward,
    CausalResnetWeights, BasicTransformerWeights,
)


comptime B = 2
comptime D = 256
comptime T = 752
comptime TIME_EMB_DIM = 1024
comptime H = 8
comptime D_K = 64
comptime FF_INNER = 1024
comptime D_OUT_MEL = 80


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


def test_estimator_forward_helper() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/s3gen/"
    var ctx = DeviceContext()

    var x_full_data = load_fp32(fix + "estimator_x_full.bin")
    var mask_data = load_fp32(fix + "cfm_step_00_input_mask.bin")
    var t_emb_data = load_fp32(fix + "estimator_time_mlp_out_real.bin")
    var exp = load_fp32(fix + "estimator_final_full.bin")

    var n_xfull = B * 320 * T
    var n_mask = B * 1 * T
    var n_te = B * TIME_EMB_DIM
    var n_out = B * D_OUT_MEL * T

    var x_full = ctx.enqueue_create_buffer[DType.float32](n_xfull)
    var mask_buf = ctx.enqueue_create_buffer[DType.float32](n_mask)
    var t_emb_buf = ctx.enqueue_create_buffer[DType.float32](n_te)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    upload(x_full, x_full_data.data, n_xfull)
    upload(mask_buf, mask_data.data, n_mask)
    upload(t_emb_buf, t_emb_data.data, n_te)

    # ---- Weights.
    var dn_rn = load_resnet(ctx, fix, "weights/flow__decoder__estimator__down_blocks__0__0__")
    var dn_tb0 = load_tblock(ctx, fix, "weights/flow__decoder__estimator__down_blocks__0__1__0__")
    var dn_tb1 = load_tblock(ctx, fix, "weights/flow__decoder__estimator__down_blocks__0__1__1__")
    var dn_tb2 = load_tblock(ctx, fix, "weights/flow__decoder__estimator__down_blocks__0__1__2__")
    var dn_tb3 = load_tblock(ctx, fix, "weights/flow__decoder__estimator__down_blocks__0__1__3__")
    var dn_ds_w = upload_w(ctx, fix, "weights/flow__decoder__estimator__down_blocks__0__2__weight.bin")
    var dn_ds_b = upload_w(ctx, fix, "weights/flow__decoder__estimator__down_blocks__0__2__bias.bin")

    var mid_rns = List[CausalResnetWeights]()
    var mid_tb0s = List[BasicTransformerWeights]()
    var mid_tb1s = List[BasicTransformerWeights]()
    var mid_tb2s = List[BasicTransformerWeights]()
    var mid_tb3s = List[BasicTransformerWeights]()
    for i in range(12):
        var p = "weights/flow__decoder__estimator__mid_blocks__" + String(i) + "__"
        mid_rns.append(load_resnet(ctx, fix, p + "0__"))
        mid_tb0s.append(load_tblock(ctx, fix, p + "1__0__"))
        mid_tb1s.append(load_tblock(ctx, fix, p + "1__1__"))
        mid_tb2s.append(load_tblock(ctx, fix, p + "1__2__"))
        mid_tb3s.append(load_tblock(ctx, fix, p + "1__3__"))

    var up_rn = load_resnet(ctx, fix, "weights/flow__decoder__estimator__up_blocks__0__0__")
    var up_tb0 = load_tblock(ctx, fix, "weights/flow__decoder__estimator__up_blocks__0__1__0__")
    var up_tb1 = load_tblock(ctx, fix, "weights/flow__decoder__estimator__up_blocks__0__1__1__")
    var up_tb2 = load_tblock(ctx, fix, "weights/flow__decoder__estimator__up_blocks__0__1__2__")
    var up_tb3 = load_tblock(ctx, fix, "weights/flow__decoder__estimator__up_blocks__0__1__3__")
    var up_us_w = upload_w(ctx, fix, "weights/flow__decoder__estimator__up_blocks__0__2__weight.bin")
    var up_us_b = upload_w(ctx, fix, "weights/flow__decoder__estimator__up_blocks__0__2__bias.bin")
    var fb_cw = upload_w(ctx, fix, "weights/flow__decoder__estimator__final_block__block__0__weight.bin")
    var fb_cb = upload_w(ctx, fix, "weights/flow__decoder__estimator__final_block__block__0__bias.bin")
    var fb_lw = upload_w(ctx, fix, "weights/flow__decoder__estimator__final_block__block__2__weight.bin")
    var fb_lb = upload_w(ctx, fix, "weights/flow__decoder__estimator__final_block__block__2__bias.bin")
    var fp_w = upload_w(ctx, fix, "weights/flow__decoder__estimator__final_proj__weight.bin")
    var fp_b = upload_w(ctx, fix, "weights/flow__decoder__estimator__final_proj__bias.bin")

    estimator_forward[B, T, D, H, D_K, FF_INNER, TIME_EMB_DIM, D_OUT_MEL](
        ctx, x_full, mask_buf, t_emb_buf, out_buf,
        dn_rn, dn_tb0, dn_tb1, dn_tb2, dn_tb3, dn_ds_w, dn_ds_b,
        mid_rns, mid_tb0s, mid_tb1s, mid_tb2s, mid_tb3s,
        up_rn, up_tb0, up_tb1, up_tb2, up_tb3, up_us_w, up_us_b,
        fb_cw, fb_cb, fb_lw, fb_lb, fp_w, fp_b,
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
                print("ef[", i, "]: mojo=", h[i], "  torch=", exp.data[i], "  diff=", d)
            assert_almost_equal(h[i], exp.data[i], atol=2.0e-1)
    print("estimator_forward helper — max abs:", max_abs,
          " mean:", sum_abs / Float64(n_out))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
