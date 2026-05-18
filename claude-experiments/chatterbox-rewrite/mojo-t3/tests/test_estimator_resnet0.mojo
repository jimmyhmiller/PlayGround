"""
Parity test for the first CausalResnetBlock1D of the estimator (down_blocks[0][0]).

Input:  estimator_x_full.bin           (2, 320, 752)  — packed [x, mu, spks, cond]
        cfm_step_00_input_mask.bin     (2, 1, 752)
        estimator_time_mlp_out_real.bin (2, 1024)     — time_mlp output
Target: estimator_resnet0_out.bin       (2, 256, 752)
"""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer

from fixture import load_fp32
from cfm_decoder import causal_resnet_block_1d, CausalResnetWeights


comptime B = 2
comptime IN_C = 320
comptime OUT_C = 256
comptime T = 752
comptime TIME_EMB_DIM = 1024


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


def test_resnet0() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/s3gen/"
    var ctx = DeviceContext()

    var x_in = load_fp32(fix + "estimator_x_full.bin")
    var mask = load_fp32(fix + "cfm_step_00_input_mask.bin")
    var t_emb = load_fp32(fix + "estimator_time_mlp_out_real.bin")
    var exp = load_fp32(fix + "estimator_resnet0_out.bin")

    var prefix = "weights/flow__decoder__estimator__down_blocks__0__0__"
    var b1_cw = upload_w(ctx, fix, prefix + "block1__block__0__weight.bin")
    var b1_cb = upload_w(ctx, fix, prefix + "block1__block__0__bias.bin")
    var b1_lw = upload_w(ctx, fix, prefix + "block1__block__2__weight.bin")
    var b1_lb = upload_w(ctx, fix, prefix + "block1__block__2__bias.bin")
    var b2_cw = upload_w(ctx, fix, prefix + "block2__block__0__weight.bin")
    var b2_cb = upload_w(ctx, fix, prefix + "block2__block__0__bias.bin")
    var b2_lw = upload_w(ctx, fix, prefix + "block2__block__2__weight.bin")
    var b2_lb = upload_w(ctx, fix, prefix + "block2__block__2__bias.bin")
    var mlp_w = upload_w(ctx, fix, prefix + "mlp__1__weight.bin")
    var mlp_b = upload_w(ctx, fix, prefix + "mlp__1__bias.bin")
    var res_w = upload_w(ctx, fix, prefix + "res_conv__weight.bin")
    var res_b = upload_w(ctx, fix, prefix + "res_conv__bias.bin")

    var w = CausalResnetWeights(
        b1_cw, b1_cb, b1_lw, b1_lb,
        b2_cw, b2_cb, b2_lw, b2_lb,
        mlp_w, mlp_b, res_w, res_b,
    )

    var n_x = B * IN_C * T
    var n_mask = B * 1 * T
    var n_te = B * TIME_EMB_DIM
    var n_out = B * OUT_C * T

    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var mask_buf = ctx.enqueue_create_buffer[DType.float32](n_mask)
    var t_emb_buf = ctx.enqueue_create_buffer[DType.float32](n_te)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    upload(x_buf, x_in.data, n_x)
    upload(mask_buf, mask.data, n_mask)
    upload(t_emb_buf, t_emb.data, n_te)

    causal_resnet_block_1d[B, IN_C, OUT_C, T, TIME_EMB_DIM](
        ctx, x_buf, mask_buf, t_emb_buf, out_buf, w,
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
                print("r0[", i, "]: mojo=", h[i], "  torch=", exp.data[i], "  diff=", d)
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-3)
    print("estimator down_blocks[0][0] resnet — max abs:", max_abs,
          " mean:", sum_abs / Float64(n_out))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
