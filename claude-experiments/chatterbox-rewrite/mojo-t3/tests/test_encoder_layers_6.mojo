"""
Parity test for the 6 chained ConformerEncoderLayers (pre-upsample).

Input:  enc_pre_lookahead.bin   (1, 376, 512)
        enc_embed_pos.bin       (1, 751, 512)
Target: enc_layer_5_out.bin     (1, 376, 512)
"""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer

from fixture import load_fp32
from encoder import conformer_layer, ConformerLayerWeights


comptime B = 1
comptime T = 376
comptime T_POS = 751
comptime D = 512
comptime H = 8
comptime D_K = 64
comptime FF_DIM = 2048


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


def load_layer(mut ctx: DeviceContext, fix: String, layer_id: Int) raises -> ConformerLayerWeights:
    var pref = "weights/flow__encoder__encoders__" + String(layer_id) + "__"
    return ConformerLayerWeights(
        upload_w(ctx, fix, pref + "norm_mha__weight.bin"),
        upload_w(ctx, fix, pref + "norm_mha__bias.bin"),
        upload_w(ctx, fix, pref + "norm_ff__weight.bin"),
        upload_w(ctx, fix, pref + "norm_ff__bias.bin"),
        upload_w(ctx, fix, pref + "self_attn__linear_q__weight.bin"),
        upload_w(ctx, fix, pref + "self_attn__linear_q__bias.bin"),
        upload_w(ctx, fix, pref + "self_attn__linear_k__weight.bin"),
        upload_w(ctx, fix, pref + "self_attn__linear_k__bias.bin"),
        upload_w(ctx, fix, pref + "self_attn__linear_v__weight.bin"),
        upload_w(ctx, fix, pref + "self_attn__linear_v__bias.bin"),
        upload_w(ctx, fix, pref + "self_attn__linear_pos__weight.bin"),
        upload_w(ctx, fix, pref + "self_attn__pos_bias_u.bin"),
        upload_w(ctx, fix, pref + "self_attn__pos_bias_v.bin"),
        upload_w(ctx, fix, pref + "self_attn__linear_out__weight.bin"),
        upload_w(ctx, fix, pref + "self_attn__linear_out__bias.bin"),
        upload_w(ctx, fix, pref + "feed_forward__w_1__weight.bin"),
        upload_w(ctx, fix, pref + "feed_forward__w_1__bias.bin"),
        upload_w(ctx, fix, pref + "feed_forward__w_2__weight.bin"),
        upload_w(ctx, fix, pref + "feed_forward__w_2__bias.bin"),
    )


def test_six_encoders() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/s3gen/"
    var ctx = DeviceContext()

    var x_in = load_fp32(fix + "enc_pre_lookahead.bin")
    var pos = load_fp32(fix + "enc_embed_pos.bin")
    var exp = load_fp32(fix + "enc_layer_5_out.bin")

    var n_x = B * T * D
    var n_pos = B * T_POS * D

    var x0 = ctx.enqueue_create_buffer[DType.float32](n_x)
    var x1 = ctx.enqueue_create_buffer[DType.float32](n_x)
    var x2 = ctx.enqueue_create_buffer[DType.float32](n_x)
    var x3 = ctx.enqueue_create_buffer[DType.float32](n_x)
    var x4 = ctx.enqueue_create_buffer[DType.float32](n_x)
    var x5 = ctx.enqueue_create_buffer[DType.float32](n_x)
    var x6 = ctx.enqueue_create_buffer[DType.float32](n_x)
    var pos_buf = ctx.enqueue_create_buffer[DType.float32](n_pos)
    upload(x0, x_in.data, n_x)
    upload(pos_buf, pos.data, n_pos)

    var w0 = load_layer(ctx, fix, 0)
    var w1 = load_layer(ctx, fix, 1)
    var w2 = load_layer(ctx, fix, 2)
    var w3 = load_layer(ctx, fix, 3)
    var w4 = load_layer(ctx, fix, 4)
    var w5 = load_layer(ctx, fix, 5)

    conformer_layer[B, T, T_POS, D, H, D_K, FF_DIM](ctx, x0, pos_buf, x1, w0)
    conformer_layer[B, T, T_POS, D, H, D_K, FF_DIM](ctx, x1, pos_buf, x2, w1)
    conformer_layer[B, T, T_POS, D, H, D_K, FF_DIM](ctx, x2, pos_buf, x3, w2)
    conformer_layer[B, T, T_POS, D, H, D_K, FF_DIM](ctx, x3, pos_buf, x4, w3)
    conformer_layer[B, T, T_POS, D, H, D_K, FF_DIM](ctx, x4, pos_buf, x5, w4)
    conformer_layer[B, T, T_POS, D, H, D_K, FF_DIM](ctx, x5, pos_buf, x6, w5)
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    var sum_abs: Float64 = 0.0
    with x6.map_to_host() as h:
        for i in range(n_x):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_abs += Float64(d)
            if i < 8:
                print("L5[", i, "]: mojo=", h[i], "  torch=", exp.data[i], "  diff=", d)
            assert_almost_equal(h[i], exp.data[i], atol=2.0e-2)
    print("encoder layers 0..5 chained — max abs:", max_abs,
          " mean:", sum_abs / Float64(n_x))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
