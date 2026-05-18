"""
Parity test for the FULL UpsampleConformerEncoder.

Input:  flow_token_emb.bin   (1, 376, 512)   — encoder input (post token embedding)
        enc_embed_pos.bin    (1, 751, 512)
        enc_up_embed_pos.bin (1, 1503, 512)   — pos for up_encoders
Target: encoder_h.bin        (1, 752, 512)   — final encoder output

Pipeline:
  embed (Linear + LayerNorm + xscale)
  pre_lookahead
  6 × ConformerEncoderLayer
  up_layer: transpose + nearest-upsample×2 + asym left-pad-4 + conv1d k=5 + transpose
  up_embed (Linear + LayerNorm + xscale)
  4 × ConformerEncoderLayer
  after_norm (LayerNorm)
"""
from std.math import ceildiv, sqrt
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu import block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major

from fixture import load_fp32
from conv import conv1d_kernel_fast, leaky_relu_kernel
from layernorm import (
    layernorm_kernel, linear_kernel,
    transpose_btc_to_bct_kernel, transpose_bct_to_btc_kernel, residual_add_kernel,
)
from encoder import conformer_layer, ConformerLayerWeights, nearest_upsample_1d_kernel


comptime B = 1
comptime T_IN = 376
comptime T_POS = 751
comptime T_OUT = 752
comptime T_OUT_POS = 1503
comptime T_UPSAMPLE = 752     # 376 * 2
comptime D = 512
comptime H = 8
comptime D_K = 64
comptime FF_DIM = 2048
comptime EPS: Float32 = 1.0e-5
comptime XSCALE: Float32 = 22.627417   # sqrt(512)
comptime BLOCK = 256


def scale_1d_kernel[
    dtype: DType, InLayout: TensorLayout, OutLayout: TensorLayout, BLOCK_: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],
    n: Int, scale: Float32,
):
    comptime assert inp.flat_rank == 1
    comptime assert output.flat_rank == 1
    var idx = block_idx.x * BLOCK_ + thread_idx.x
    if idx >= n: return
    var v = rebind[Scalar[dtype]](inp[idx]).cast[DType.float32]()
    output[idx] = rebind[output.ElementType]((v * scale).cast[dtype]())


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


def load_layer(mut ctx: DeviceContext, fix: String, prefix: String, layer_id: Int) raises -> ConformerLayerWeights:
    var pref = "weights/flow__encoder__" + prefix + "__" + String(layer_id) + "__"
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


def test_full_encoder() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/s3gen/"
    var ctx = DeviceContext()

    var x_in = load_fp32(fix + "flow_token_emb.bin")
    var pos = load_fp32(fix + "enc_embed_pos.bin")
    var pos_up = load_fp32(fix + "enc_up_embed_pos.bin")
    var exp = load_fp32(fix + "encoder_h.bin")

    # ---- Embed weights.
    var w_emb_lin = upload_w(ctx, fix, "weights/flow__encoder__embed__out__0__weight.bin")
    var b_emb_lin = upload_w(ctx, fix, "weights/flow__encoder__embed__out__0__bias.bin")
    var w_emb_ln = upload_w(ctx, fix, "weights/flow__encoder__embed__out__1__weight.bin")
    var b_emb_ln = upload_w(ctx, fix, "weights/flow__encoder__embed__out__1__bias.bin")

    # ---- pre_lookahead weights.
    var w_pl_c1 = upload_w(ctx, fix, "weights/flow__encoder__pre_lookahead_layer__conv1__weight.bin")
    var b_pl_c1 = upload_w(ctx, fix, "weights/flow__encoder__pre_lookahead_layer__conv1__bias.bin")
    var w_pl_c2 = upload_w(ctx, fix, "weights/flow__encoder__pre_lookahead_layer__conv2__weight.bin")
    var b_pl_c2 = upload_w(ctx, fix, "weights/flow__encoder__pre_lookahead_layer__conv2__bias.bin")

    # ---- up_layer weights.
    var w_up = upload_w(ctx, fix, "weights/flow__encoder__up_layer__conv__weight.bin")
    var b_up = upload_w(ctx, fix, "weights/flow__encoder__up_layer__conv__bias.bin")

    # ---- up_embed weights.
    var w_ue_lin = upload_w(ctx, fix, "weights/flow__encoder__up_embed__out__0__weight.bin")
    var b_ue_lin = upload_w(ctx, fix, "weights/flow__encoder__up_embed__out__0__bias.bin")
    var w_ue_ln = upload_w(ctx, fix, "weights/flow__encoder__up_embed__out__1__weight.bin")
    var b_ue_ln = upload_w(ctx, fix, "weights/flow__encoder__up_embed__out__1__bias.bin")

    # ---- after_norm.
    var an_w = upload_w(ctx, fix, "weights/flow__encoder__after_norm__weight.bin")
    var an_b = upload_w(ctx, fix, "weights/flow__encoder__after_norm__bias.bin")

    var n_in = B * T_IN * D
    var n_out_t = B * T_OUT * D
    var n_pos = B * T_POS * D
    var n_pos_up = B * T_OUT_POS * D

    # ---- Buffers for embed → pre_lookahead → 6 encoder layers.
    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_in)
    var lin1_out = ctx.enqueue_create_buffer[DType.float32](n_in)
    var ln1_out = ctx.enqueue_create_buffer[DType.float32](n_in)
    var emb_out = ctx.enqueue_create_buffer[DType.float32](n_in)
    var pos_buf = ctx.enqueue_create_buffer[DType.float32](n_pos)
    var pos_up_buf = ctx.enqueue_create_buffer[DType.float32](n_pos_up)
    upload(x_buf, x_in.data, n_in)
    upload(pos_buf, pos.data, n_pos)
    upload(pos_up_buf, pos_up.data, n_pos_up)

    comptime btd_in = row_major[B, T_IN, D]()
    comptime btd_out = row_major[B, T_OUT, D]()
    comptime btd_pos = row_major[B, T_POS, D]()
    comptime btd_pos_up = row_major[B, T_OUT_POS, D]()
    comptime bct_in = row_major[B, D, T_IN]()
    comptime bct_up = row_major[B, D, T_UPSAMPLE]()
    comptime bct_out = row_major[B, D, T_OUT]()
    comptime w_layout = row_major[D, D]()
    comptime p_layout = row_major[D]()
    comptime w_pl_c1_layout = row_major[D, D, 4]()
    comptime w_pl_c2_layout = row_major[D, D, 3]()
    comptime w_up_layout = row_major[D, D, 5]()
    comptime flat_in = row_major[B * T_IN * D]()
    comptime flat_out = row_major[B * T_OUT * D]()

    # ---- embed: Linear + LayerNorm + xscale.
    var x_t = TileTensor(x_buf, btd_in)
    var w_emb_lin_t = TileTensor(w_emb_lin, w_layout)
    var b_emb_lin_t = TileTensor(b_emb_lin, p_layout)
    var w_emb_ln_t = TileTensor(w_emb_ln, p_layout)
    var b_emb_ln_t = TileTensor(b_emb_ln, p_layout)
    var lin1_out_t = TileTensor(lin1_out, btd_in)
    var ln1_out_t = TileTensor(ln1_out, btd_in)
    var ln1_out_flat = TileTensor(ln1_out, flat_in)
    var emb_out_t = TileTensor(emb_out, btd_in)
    var emb_out_flat = TileTensor(emb_out, flat_in)

    comptime lin_k = linear_kernel[
        DType.float32, type_of(btd_in), type_of(w_layout),
        type_of(p_layout), type_of(btd_in),
        True, BLOCK,
    ]
    ctx.enqueue_function[lin_k, lin_k](
        lin1_out_t, x_t, w_emb_lin_t, b_emb_lin_t, B, T_IN, D, D,
        grid_dim=B * T_IN, block_dim=BLOCK,
    )
    comptime ln_k = layernorm_kernel[
        DType.float32, type_of(btd_in), type_of(p_layout), type_of(btd_in), BLOCK,
    ]
    ctx.enqueue_function[ln_k, ln_k](
        ln1_out_t, lin1_out_t, w_emb_ln_t, b_emb_ln_t,
        B, T_IN, D, EPS, grid_dim=B * T_IN, block_dim=BLOCK,
    )
    comptime sc_k = scale_1d_kernel[
        DType.float32, type_of(flat_in), type_of(flat_in), BLOCK,
    ]
    ctx.enqueue_function[sc_k, sc_k](
        emb_out_flat, ln1_out_flat, n_in, XSCALE,
        grid_dim=ceildiv(n_in, BLOCK), block_dim=BLOCK,
    )

    # ---- pre_lookahead: transpose + conv1 + leaky_relu + conv2 + transpose + residual.
    var bct_buf = ctx.enqueue_create_buffer[DType.float32](n_in)
    var conv1_out = ctx.enqueue_create_buffer[DType.float32](n_in)
    var relu_out = ctx.enqueue_create_buffer[DType.float32](n_in)
    var conv2_out = ctx.enqueue_create_buffer[DType.float32](n_in)
    var btc_back = ctx.enqueue_create_buffer[DType.float32](n_in)
    var pl_out = ctx.enqueue_create_buffer[DType.float32](n_in)
    var bct_t = TileTensor(bct_buf, bct_in)
    var w_pl_c1_t = TileTensor(w_pl_c1, w_pl_c1_layout)
    var b_pl_c1_t = TileTensor(b_pl_c1, p_layout)
    var w_pl_c2_t = TileTensor(w_pl_c2, w_pl_c2_layout)
    var b_pl_c2_t = TileTensor(b_pl_c2, p_layout)
    var conv1_out_t = TileTensor(conv1_out, bct_in)
    var conv1_out_flat = TileTensor(conv1_out, flat_in)
    var relu_out_t = TileTensor(relu_out, bct_in)
    var relu_out_flat = TileTensor(relu_out, flat_in)
    var conv2_out_t = TileTensor(conv2_out, bct_in)
    var btc_back_t = TileTensor(btc_back, btd_in)
    var btc_back_flat = TileTensor(btc_back, flat_in)
    var pl_out_flat = TileTensor(pl_out, flat_in)

    comptime tp1_k = transpose_btc_to_bct_kernel[
        DType.float32, type_of(btd_in), type_of(bct_in), BLOCK,
    ]
    ctx.enqueue_function[tp1_k, tp1_k](
        bct_t, emb_out_t, B, T_IN, D, grid_dim=B * D, block_dim=BLOCK,
    )
    comptime conv1_k = conv1d_kernel_fast[
        DType.float32, type_of(bct_in), type_of(w_pl_c1_layout),
        type_of(p_layout), type_of(bct_in),
        4, True, BLOCK,
    ]
    ctx.enqueue_function[conv1_k, conv1_k](
        conv1_out_t, bct_t, w_pl_c1_t, b_pl_c1_t,
        B, D, D, T_IN, T_IN, 1, 0, 1,
        grid_dim=B * D, block_dim=BLOCK,
    )
    comptime relu_k = leaky_relu_kernel[
        DType.float32, type_of(flat_in), type_of(flat_in), BLOCK,
    ]
    ctx.enqueue_function[relu_k, relu_k](
        relu_out_flat, conv1_out_flat, n_in, Float32(0.01),
        grid_dim=ceildiv(n_in, BLOCK), block_dim=BLOCK,
    )
    comptime conv2_k = conv1d_kernel_fast[
        DType.float32, type_of(bct_in), type_of(w_pl_c2_layout),
        type_of(p_layout), type_of(bct_in),
        3, True, BLOCK,
    ]
    ctx.enqueue_function[conv2_k, conv2_k](
        conv2_out_t, relu_out_t, w_pl_c2_t, b_pl_c2_t,
        B, D, D, T_IN, T_IN, 1, 2, 1,
        grid_dim=B * D, block_dim=BLOCK,
    )
    comptime tp2_k = transpose_bct_to_btc_kernel[
        DType.float32, type_of(bct_in), type_of(btd_in), BLOCK,
    ]
    ctx.enqueue_function[tp2_k, tp2_k](
        btc_back_t, conv2_out_t, B, D, T_IN, grid_dim=B * T_IN, block_dim=BLOCK,
    )
    comptime add_k = residual_add_kernel[
        DType.float32, type_of(flat_in), type_of(flat_in), type_of(flat_in), BLOCK,
    ]
    ctx.enqueue_function[add_k, add_k](
        pl_out_flat, btc_back_flat, emb_out_flat, n_in,
        grid_dim=ceildiv(n_in, BLOCK), block_dim=BLOCK,
    )

    # ---- 6 conformer layers.
    var x_l1 = ctx.enqueue_create_buffer[DType.float32](n_in)
    var x_l2 = ctx.enqueue_create_buffer[DType.float32](n_in)
    var x_l3 = ctx.enqueue_create_buffer[DType.float32](n_in)
    var x_l4 = ctx.enqueue_create_buffer[DType.float32](n_in)
    var x_l5 = ctx.enqueue_create_buffer[DType.float32](n_in)
    var x_l6 = ctx.enqueue_create_buffer[DType.float32](n_in)
    var w0 = load_layer(ctx, fix, "encoders", 0)
    var w1 = load_layer(ctx, fix, "encoders", 1)
    var w2 = load_layer(ctx, fix, "encoders", 2)
    var w3 = load_layer(ctx, fix, "encoders", 3)
    var w4 = load_layer(ctx, fix, "encoders", 4)
    var w5 = load_layer(ctx, fix, "encoders", 5)
    conformer_layer[B, T_IN, T_POS, D, H, D_K, FF_DIM](ctx, pl_out, pos_buf, x_l1, w0)
    conformer_layer[B, T_IN, T_POS, D, H, D_K, FF_DIM](ctx, x_l1, pos_buf, x_l2, w1)
    conformer_layer[B, T_IN, T_POS, D, H, D_K, FF_DIM](ctx, x_l2, pos_buf, x_l3, w2)
    conformer_layer[B, T_IN, T_POS, D, H, D_K, FF_DIM](ctx, x_l3, pos_buf, x_l4, w3)
    conformer_layer[B, T_IN, T_POS, D, H, D_K, FF_DIM](ctx, x_l4, pos_buf, x_l5, w4)
    conformer_layer[B, T_IN, T_POS, D, H, D_K, FF_DIM](ctx, x_l5, pos_buf, x_l6, w5)

    # ---- up_layer: transpose to (B, D, T_in), nearest-upsample to (B, D, 2T_in),
    # then conv1d k=5, padding=4 (asym left-pad emulation), then transpose back.
    var bct_pre_up = ctx.enqueue_create_buffer[DType.float32](n_in)
    var up_buf = ctx.enqueue_create_buffer[DType.float32](B * D * T_UPSAMPLE)
    var up_conv_out = ctx.enqueue_create_buffer[DType.float32](B * D * T_UPSAMPLE)
    var up_bct_to_btc = ctx.enqueue_create_buffer[DType.float32](n_out_t)

    var bct_pre_up_t = TileTensor(bct_pre_up, bct_in)
    var x_l6_t = TileTensor(x_l6, btd_in)
    ctx.enqueue_function[tp1_k, tp1_k](
        bct_pre_up_t, x_l6_t, B, T_IN, D, grid_dim=B * D, block_dim=BLOCK,
    )
    var up_buf_t = TileTensor(up_buf, bct_up)
    comptime up_k = nearest_upsample_1d_kernel[
        DType.float32, type_of(bct_in), type_of(bct_up), BLOCK,
    ]
    ctx.enqueue_function[up_k, up_k](
        up_buf_t, bct_pre_up_t, B, D, T_IN, 2,
        grid_dim=B * D, block_dim=BLOCK,
    )
    var up_conv_out_t = TileTensor(up_conv_out, bct_out)
    var w_up_t = TileTensor(w_up, w_up_layout)
    var b_up_t = TileTensor(b_up, p_layout)
    comptime up_conv_k = conv1d_kernel_fast[
        DType.float32, type_of(bct_up), type_of(w_up_layout),
        type_of(p_layout), type_of(bct_out),
        5, True, BLOCK,
    ]
    ctx.enqueue_function[up_conv_k, up_conv_k](
        up_conv_out_t, up_buf_t, w_up_t, b_up_t,
        B, D, D, T_UPSAMPLE, T_OUT, 1, 4, 1,
        grid_dim=B * D, block_dim=BLOCK,
    )
    comptime tp_up = transpose_bct_to_btc_kernel[
        DType.float32, type_of(bct_out), type_of(btd_out), BLOCK,
    ]
    var up_bct_to_btc_t = TileTensor(up_bct_to_btc, btd_out)
    ctx.enqueue_function[tp_up, tp_up](
        up_bct_to_btc_t, up_conv_out_t, B, D, T_OUT, grid_dim=B * T_OUT, block_dim=BLOCK,
    )

    # ---- up_embed (Linear + LayerNorm + xscale).
    var up_lin_out = ctx.enqueue_create_buffer[DType.float32](n_out_t)
    var up_ln_out = ctx.enqueue_create_buffer[DType.float32](n_out_t)
    var up_emb_out = ctx.enqueue_create_buffer[DType.float32](n_out_t)
    var w_ue_lin_t = TileTensor(w_ue_lin, w_layout)
    var b_ue_lin_t = TileTensor(b_ue_lin, p_layout)
    var w_ue_ln_t = TileTensor(w_ue_ln, p_layout)
    var b_ue_ln_t = TileTensor(b_ue_ln, p_layout)
    var up_lin_out_t = TileTensor(up_lin_out, btd_out)
    var up_ln_out_t = TileTensor(up_ln_out, btd_out)
    var up_ln_out_flat = TileTensor(up_ln_out, flat_out)
    var up_emb_out_flat = TileTensor(up_emb_out, flat_out)
    comptime lin_out_k = linear_kernel[
        DType.float32, type_of(btd_out), type_of(w_layout),
        type_of(p_layout), type_of(btd_out),
        True, BLOCK,
    ]
    ctx.enqueue_function[lin_out_k, lin_out_k](
        up_lin_out_t, up_bct_to_btc_t, w_ue_lin_t, b_ue_lin_t, B, T_OUT, D, D,
        grid_dim=B * T_OUT, block_dim=BLOCK,
    )
    comptime ln_out_k = layernorm_kernel[
        DType.float32, type_of(btd_out), type_of(p_layout), type_of(btd_out), BLOCK,
    ]
    ctx.enqueue_function[ln_out_k, ln_out_k](
        up_ln_out_t, up_lin_out_t, w_ue_ln_t, b_ue_ln_t,
        B, T_OUT, D, EPS, grid_dim=B * T_OUT, block_dim=BLOCK,
    )
    comptime sc_out_k = scale_1d_kernel[
        DType.float32, type_of(flat_out), type_of(flat_out), BLOCK,
    ]
    ctx.enqueue_function[sc_out_k, sc_out_k](
        up_emb_out_flat, up_ln_out_flat, n_out_t, XSCALE,
        grid_dim=ceildiv(n_out_t, BLOCK), block_dim=BLOCK,
    )

    # ---- 4 up_encoders.
    var x_u1 = ctx.enqueue_create_buffer[DType.float32](n_out_t)
    var x_u2 = ctx.enqueue_create_buffer[DType.float32](n_out_t)
    var x_u3 = ctx.enqueue_create_buffer[DType.float32](n_out_t)
    var x_u4 = ctx.enqueue_create_buffer[DType.float32](n_out_t)
    var u0 = load_layer(ctx, fix, "up_encoders", 0)
    var u1 = load_layer(ctx, fix, "up_encoders", 1)
    var u2 = load_layer(ctx, fix, "up_encoders", 2)
    var u3 = load_layer(ctx, fix, "up_encoders", 3)
    conformer_layer[B, T_OUT, T_OUT_POS, D, H, D_K, FF_DIM](ctx, up_emb_out, pos_up_buf, x_u1, u0)
    conformer_layer[B, T_OUT, T_OUT_POS, D, H, D_K, FF_DIM](ctx, x_u1, pos_up_buf, x_u2, u1)
    conformer_layer[B, T_OUT, T_OUT_POS, D, H, D_K, FF_DIM](ctx, x_u2, pos_up_buf, x_u3, u2)
    conformer_layer[B, T_OUT, T_OUT_POS, D, H, D_K, FF_DIM](ctx, x_u3, pos_up_buf, x_u4, u3)

    # ---- after_norm (LayerNorm over channels).
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_out_t)
    var an_w_t = TileTensor(an_w, p_layout)
    var an_b_t = TileTensor(an_b, p_layout)
    var x_u4_t = TileTensor(x_u4, btd_out)
    var out_t = TileTensor(out_buf, btd_out)
    ctx.enqueue_function[ln_out_k, ln_out_k](
        out_t, x_u4_t, an_w_t, an_b_t,
        B, T_OUT, D, EPS, grid_dim=B * T_OUT, block_dim=BLOCK,
    )
    ctx.synchronize()

    var n_exp = B * T_OUT * D
    var max_abs: Float32 = 0.0
    var sum_abs: Float64 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n_exp):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_abs += Float64(d)
            if i < 8:
                print("enc[", i, "]: mojo=", h[i], "  torch=", exp.data[i], "  diff=", d)
            assert_almost_equal(h[i], exp.data[i], atol=5.0e-2)
    print("FULL encoder — max abs:", max_abs, " mean:", sum_abs / Float64(n_exp))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
