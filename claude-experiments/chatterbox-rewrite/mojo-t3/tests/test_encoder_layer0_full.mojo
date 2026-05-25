"""
Parity test for the FULL ConformerEncoderLayer 0.

Input:  enc_pre_lookahead.bin  (1, 376, 512)    — output of pre_lookahead
        enc_embed_pos.bin      (1, 751, 512)    — relative pos
Target: enc_layer_0_out.bin    (1, 376, 512)    — output of encoders[0]

Sequence (normalize_before=True, no macaron, no cnn_module):
  x_in = input
  residual = x_in
  x = norm_mha(x_in)            (LayerNorm)
  x_attn = self_attn(x, x, x, pos_emb)
  x = residual + dropout(x_attn) = residual + x_attn (eval)
  residual = x
  x = norm_ff(x)
  x_ff = feed_forward(x)
  out = residual + dropout(x_ff) = residual + x_ff
"""
from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu import block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major

from fixture import load_fp32
from conv import relu_kernel
from layernorm import (
    layernorm_kernel, linear_kernel, residual_add_kernel,
)
from attention import (
    qkv_proj_reshape_kernel, add_pos_bias_kernel, matmul_qk_scaled_kernel,
    matmul_qp_kernel, rel_shift_kernel, add_4d_kernel, softmax_lastdim_kernel,
    matmul_av_kernel, merge_heads_kernel, swish_kernel,
)


comptime B = 1
comptime T = 376
comptime T_POS = 751
comptime D = 512
comptime H = 8
comptime D_K = 64
comptime FF_DIM = 2048
comptime BLOCK = 64
comptime BLOCK_T = 256
comptime EPS: Float32 = 1.0e-5
comptime SCALE: Float32 = 0.125    # 1/sqrt(64)


def linear_pos_no_bias_kernel[
    dtype: DType, InLayout: TensorLayout, WLayout: TensorLayout, OutLayout: TensorLayout,
    BLOCK_: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],
    w: TileTensor[dtype, WLayout, MutAnyOrigin],
    batch: Int, t_pos: Int, heads: Int, d_k: Int, d_model: Int,
):
    comptime assert inp.flat_rank == 3
    comptime assert w.flat_rank == 2
    comptime assert output.flat_rank == 4
    var bid = block_idx.x
    var tid = thread_idx.x
    var tp = bid % t_pos
    var h_idx = (bid // t_pos) % heads
    var b = bid // (t_pos * heads)
    var ki = tid
    while ki < d_k:
        var o = h_idx * d_k + ki
        var acc: Float32 = 0.0
        for i in range(d_model):
            var xv = rebind[Scalar[dtype]](inp[b, tp, i]).cast[DType.float32]()
            var wv = rebind[Scalar[dtype]](w[o, i]).cast[DType.float32]()
            acc += xv * wv
        output[b, h_idx, tp, ki] = rebind[output.ElementType](acc.cast[dtype]())
        ki += BLOCK_


def scale_4d_kernel[
    dtype: DType, InLayout: TensorLayout, OutLayout: TensorLayout, BLOCK_: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],
    n: Int, s: Float32,
):
    comptime assert inp.flat_rank == 1
    comptime assert output.flat_rank == 1
    var idx = block_idx.x * BLOCK_ + thread_idx.x
    if idx >= n: return
    var v = rebind[Scalar[dtype]](inp[idx]).cast[DType.float32]()
    output[idx] = rebind[output.ElementType]((v * s).cast[dtype]())


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


def test_layer0_full() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/s3gen/"
    var ctx = DeviceContext()

    # ---- Inputs.
    var x_in = load_fp32(fix + "enc_pre_lookahead.bin")
    var pos = load_fp32(fix + "enc_embed_pos.bin")
    var exp = load_fp32(fix + "enc_layer_0_out.bin")

    # Layer 0 norms.
    var nm_w = upload_w(ctx, fix, "weights/flow__encoder__encoders__0__norm_mha__weight.bin")
    var nm_b = upload_w(ctx, fix, "weights/flow__encoder__encoders__0__norm_mha__bias.bin")
    var nf_w = upload_w(ctx, fix, "weights/flow__encoder__encoders__0__norm_ff__weight.bin")
    var nf_b = upload_w(ctx, fix, "weights/flow__encoder__encoders__0__norm_ff__bias.bin")

    # Layer 0 attention weights.
    var w_q = upload_w(ctx, fix, "weights/flow__encoder__encoders__0__self_attn__linear_q__weight.bin")
    var b_q = upload_w(ctx, fix, "weights/flow__encoder__encoders__0__self_attn__linear_q__bias.bin")
    var w_k = upload_w(ctx, fix, "weights/flow__encoder__encoders__0__self_attn__linear_k__weight.bin")
    var b_k = upload_w(ctx, fix, "weights/flow__encoder__encoders__0__self_attn__linear_k__bias.bin")
    var w_v = upload_w(ctx, fix, "weights/flow__encoder__encoders__0__self_attn__linear_v__weight.bin")
    var b_v = upload_w(ctx, fix, "weights/flow__encoder__encoders__0__self_attn__linear_v__bias.bin")
    var w_pos = upload_w(ctx, fix, "weights/flow__encoder__encoders__0__self_attn__linear_pos__weight.bin")
    var pb_u = upload_w(ctx, fix, "weights/flow__encoder__encoders__0__self_attn__pos_bias_u.bin")
    var pb_v = upload_w(ctx, fix, "weights/flow__encoder__encoders__0__self_attn__pos_bias_v.bin")
    var w_out = upload_w(ctx, fix, "weights/flow__encoder__encoders__0__self_attn__linear_out__weight.bin")
    var b_out = upload_w(ctx, fix, "weights/flow__encoder__encoders__0__self_attn__linear_out__bias.bin")

    # Layer 0 feed-forward.
    var w_f1 = upload_w(ctx, fix, "weights/flow__encoder__encoders__0__feed_forward__w_1__weight.bin")
    var b_f1 = upload_w(ctx, fix, "weights/flow__encoder__encoders__0__feed_forward__w_1__bias.bin")
    var w_f2 = upload_w(ctx, fix, "weights/flow__encoder__encoders__0__feed_forward__w_2__weight.bin")
    var b_f2 = upload_w(ctx, fix, "weights/flow__encoder__encoders__0__feed_forward__w_2__bias.bin")

    var n_x = B * T * D
    var n_pos = B * T_POS * D
    var n_qkv = B * H * T * D_K
    var n_p = B * H * T_POS * D_K
    var n_scores = B * H * T * T
    var n_bd_pre = B * H * T * T_POS
    var n_ff = B * T * FF_DIM

    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var pos_buf = ctx.enqueue_create_buffer[DType.float32](n_pos)

    # MHA path buffers.
    var nm_out = ctx.enqueue_create_buffer[DType.float32](n_x)
    var q_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv)
    var k_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv)
    var v_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv)
    var q_u_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv)
    var q_v_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv)
    var p_buf = ctx.enqueue_create_buffer[DType.float32](n_p)
    var ac_buf = ctx.enqueue_create_buffer[DType.float32](n_scores)
    var bd_pre_buf = ctx.enqueue_create_buffer[DType.float32](n_bd_pre)
    var bd_buf = ctx.enqueue_create_buffer[DType.float32](n_scores)
    var scores_buf = ctx.enqueue_create_buffer[DType.float32](n_scores)
    var scaled_buf = ctx.enqueue_create_buffer[DType.float32](n_scores)
    var attn_buf = ctx.enqueue_create_buffer[DType.float32](n_scores)
    var ctx_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv)
    var merged_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var attn_out_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var post_mha = ctx.enqueue_create_buffer[DType.float32](n_x)

    # FF path buffers.
    var nf_out = ctx.enqueue_create_buffer[DType.float32](n_x)
    var w1_out = ctx.enqueue_create_buffer[DType.float32](n_ff)
    var act_out = ctx.enqueue_create_buffer[DType.float32](n_ff)
    var w2_out = ctx.enqueue_create_buffer[DType.float32](n_x)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_x)

    upload(x_buf, x_in.data, n_x)
    upload(pos_buf, pos.data, n_pos)

    comptime btd_layout = row_major[B, T, D]()
    comptime btd_pos_layout = row_major[B, T_POS, D]()
    comptime w_layout = row_major[D, D]()
    comptime w_ff1_layout = row_major[FF_DIM, D]()
    comptime w_ff2_layout = row_major[D, FF_DIM]()
    comptime p_layout = row_major[D]()
    comptime p_ff1_layout = row_major[FF_DIM]()
    comptime qkv_layout = row_major[B, H, T, D_K]()
    comptime pq_layout = row_major[B, H, T_POS, D_K]()
    comptime pb_layout = row_major[H, D_K]()
    comptime scores_layout = row_major[B, H, T, T]()
    comptime bd_pre_layout = row_major[B, H, T, T_POS]()
    comptime btff_layout = row_major[B, T, FF_DIM]()
    comptime flat = row_major[B * T * D]()
    comptime flat_scores = row_major[B * H * T * T]()
    comptime flat_ff = row_major[B * T * FF_DIM]()

    var x_t = TileTensor(x_buf, btd_layout)
    var pos_t = TileTensor(pos_buf, btd_pos_layout)
    var nm_w_t = TileTensor(nm_w, p_layout)
    var nm_b_t = TileTensor(nm_b, p_layout)
    var nf_w_t = TileTensor(nf_w, p_layout)
    var nf_b_t = TileTensor(nf_b, p_layout)
    var nm_out_t = TileTensor(nm_out, btd_layout)
    var nf_out_t = TileTensor(nf_out, btd_layout)
    var q_t = TileTensor(q_buf, qkv_layout)
    var k_t = TileTensor(k_buf, qkv_layout)
    var v_t = TileTensor(v_buf, qkv_layout)
    var q_u_t = TileTensor(q_u_buf, qkv_layout)
    var q_v_t = TileTensor(q_v_buf, qkv_layout)
    var p_t = TileTensor(p_buf, pq_layout)
    var ac_t = TileTensor(ac_buf, scores_layout)
    var bd_pre_t = TileTensor(bd_pre_buf, bd_pre_layout)
    var bd_t = TileTensor(bd_buf, scores_layout)
    var ac_flat = TileTensor(ac_buf, flat_scores)
    var bd_flat = TileTensor(bd_buf, flat_scores)
    var scores_flat = TileTensor(scores_buf, flat_scores)
    var scaled_t = TileTensor(scaled_buf, scores_layout)
    var scaled_flat = TileTensor(scaled_buf, flat_scores)
    var attn_t = TileTensor(attn_buf, scores_layout)
    var ctx_t = TileTensor(ctx_buf, qkv_layout)
    var merged_t = TileTensor(merged_buf, btd_layout)
    var attn_out_t = TileTensor(attn_out_buf, btd_layout)
    var x_flat = TileTensor(x_buf, flat)
    var attn_out_flat = TileTensor(attn_out_buf, flat)
    var post_mha_t = TileTensor(post_mha, btd_layout)
    var post_mha_flat = TileTensor(post_mha, flat)
    var w_q_t = TileTensor(w_q, w_layout)
    var b_q_t = TileTensor(b_q, p_layout)
    var w_k_t = TileTensor(w_k, w_layout)
    var b_k_t = TileTensor(b_k, p_layout)
    var w_v_t = TileTensor(w_v, w_layout)
    var b_v_t = TileTensor(b_v, p_layout)
    var w_pos_t = TileTensor(w_pos, w_layout)
    var pb_u_t = TileTensor(pb_u, pb_layout)
    var pb_v_t = TileTensor(pb_v, pb_layout)
    var w_out_t = TileTensor(w_out, w_layout)
    var b_out_t = TileTensor(b_out, p_layout)
    var w_f1_t = TileTensor(w_f1, w_ff1_layout)
    var b_f1_t = TileTensor(b_f1, p_ff1_layout)
    var w1_out_t = TileTensor(w1_out, btff_layout)
    var w1_out_flat = TileTensor(w1_out, flat_ff)
    var act_out_t = TileTensor(act_out, btff_layout)
    var act_out_flat = TileTensor(act_out, flat_ff)
    var w_f2_t = TileTensor(w_f2, w_ff2_layout)
    var b_f2_t = TileTensor(b_f2, p_layout)
    var w2_out_t = TileTensor(w2_out, btd_layout)
    var w2_out_flat = TileTensor(w2_out, flat)
    var out_flat = TileTensor(out_buf, flat)

    # ---- norm_mha.
    comptime ln_k = layernorm_kernel[
        DType.float32, type_of(btd_layout), type_of(p_layout),
        type_of(btd_layout), BLOCK_T,
    ]
    ctx.enqueue_function[ln_k, ln_k](
        nm_out_t, x_t, nm_w_t, nm_b_t,
        B, T, D, EPS, grid_dim=B * T, block_dim=BLOCK_T,
    )

    # ---- self_attn. nm_out is the input. q, k, v projections.
    comptime qkv_k = qkv_proj_reshape_kernel[
        DType.float32, type_of(btd_layout), type_of(w_layout),
        type_of(p_layout), type_of(qkv_layout),
        True, BLOCK,
    ]
    ctx.enqueue_function[qkv_k, qkv_k](
        q_t, nm_out_t, w_q_t, b_q_t,
        B, T, H, D_K, D, grid_dim=B * H * T, block_dim=BLOCK,
    )
    ctx.enqueue_function[qkv_k, qkv_k](
        k_t, nm_out_t, w_k_t, b_k_t,
        B, T, H, D_K, D, grid_dim=B * H * T, block_dim=BLOCK,
    )
    ctx.enqueue_function[qkv_k, qkv_k](
        v_t, nm_out_t, w_v_t, b_v_t,
        B, T, H, D_K, D, grid_dim=B * H * T, block_dim=BLOCK,
    )

    # ---- linear_pos.
    comptime pos_k = linear_pos_no_bias_kernel[
        DType.float32, type_of(btd_pos_layout), type_of(w_layout),
        type_of(pq_layout), BLOCK,
    ]
    ctx.enqueue_function[pos_k, pos_k](
        p_t, pos_t, w_pos_t, B, T_POS, H, D_K, D,
        grid_dim=B * H * T_POS, block_dim=BLOCK,
    )

    # ---- q_u, q_v.
    comptime bias_k = add_pos_bias_kernel[
        DType.float32, type_of(qkv_layout), type_of(pb_layout),
        type_of(qkv_layout), BLOCK,
    ]
    ctx.enqueue_function[bias_k, bias_k](
        q_u_t, q_t, pb_u_t, B, H, T, D_K, grid_dim=B * H * T, block_dim=BLOCK,
    )
    ctx.enqueue_function[bias_k, bias_k](
        q_v_t, q_t, pb_v_t, B, H, T, D_K, grid_dim=B * H * T, block_dim=BLOCK,
    )

    # ---- matrix_ac, matrix_bd, rel_shift.
    comptime mm_qk_k = matmul_qk_scaled_kernel[
        DType.float32, type_of(qkv_layout), type_of(qkv_layout),
        type_of(scores_layout), BLOCK_T,
    ]
    ctx.enqueue_function[mm_qk_k, mm_qk_k](
        ac_t, q_u_t, k_t, B, H, T, T, D_K, Float32(1.0),
        grid_dim=B * H * T, block_dim=BLOCK_T,
    )
    comptime mm_qp_k = matmul_qp_kernel[
        DType.float32, type_of(qkv_layout), type_of(pq_layout),
        type_of(bd_pre_layout), BLOCK_T,
    ]
    ctx.enqueue_function[mm_qp_k, mm_qp_k](
        bd_pre_t, q_v_t, p_t, B, H, T, T_POS, D_K,
        grid_dim=B * H * T, block_dim=BLOCK_T,
    )
    comptime shift_k = rel_shift_kernel[
        DType.float32, type_of(bd_pre_layout), type_of(scores_layout), BLOCK_T,
    ]
    ctx.enqueue_function[shift_k, shift_k](
        bd_t, bd_pre_t, B, H, T, grid_dim=B * H * T, block_dim=BLOCK_T,
    )

    # ---- scores = (ac + bd) * scale.
    comptime add_k = add_4d_kernel[
        DType.float32, type_of(flat_scores), type_of(flat_scores),
        type_of(flat_scores), 256,
    ]
    ctx.enqueue_function[add_k, add_k](
        scores_flat, ac_flat, bd_flat, B * H * T * T,
        grid_dim=ceildiv(B * H * T * T, 256), block_dim=256,
    )
    comptime sc_k = scale_4d_kernel[
        DType.float32, type_of(flat_scores), type_of(flat_scores), 256,
    ]
    ctx.enqueue_function[sc_k, sc_k](
        scaled_flat, scores_flat, B * H * T * T, SCALE,
        grid_dim=ceildiv(B * H * T * T, 256), block_dim=256,
    )

    # ---- softmax + matmul_av + merge + linear_out.
    comptime sm_k = softmax_lastdim_kernel[
        DType.float32, type_of(scores_layout), type_of(scores_layout), BLOCK_T,
    ]
    ctx.enqueue_function[sm_k, sm_k](
        attn_t, scaled_t, B, H, T, T,
        grid_dim=B * H * T, block_dim=BLOCK_T,
    )
    comptime av_k = matmul_av_kernel[
        DType.float32, type_of(scores_layout), type_of(qkv_layout),
        type_of(qkv_layout), BLOCK,
    ]
    ctx.enqueue_function[av_k, av_k](
        ctx_t, attn_t, v_t, B, H, T, T, D_K,
        grid_dim=B * H * T, block_dim=BLOCK,
    )
    comptime mh_k = merge_heads_kernel[
        DType.float32, type_of(qkv_layout), type_of(btd_layout), BLOCK,
    ]
    ctx.enqueue_function[mh_k, mh_k](
        merged_t, ctx_t, B, H, T, D_K,
        grid_dim=B * H * T, block_dim=BLOCK,
    )
    comptime lin_attn_k = linear_kernel[
        DType.float32, type_of(btd_layout), type_of(w_layout),
        type_of(p_layout), type_of(btd_layout),
        True, BLOCK_T,
    ]
    ctx.enqueue_function[lin_attn_k, lin_attn_k](
        attn_out_t, merged_t, w_out_t, b_out_t, B, T, D, D,
        grid_dim=B * T, block_dim=BLOCK_T,
    )

    # ---- Residual: post_mha = x + attn_out.
    comptime add_x = residual_add_kernel[
        DType.float32, type_of(flat), type_of(flat), type_of(flat), 256,
    ]
    ctx.enqueue_function[add_x, add_x](
        post_mha_flat, x_flat, attn_out_flat, n_x,
        grid_dim=ceildiv(n_x, 256), block_dim=256,
    )

    # ---- norm_ff.
    ctx.enqueue_function[ln_k, ln_k](
        nf_out_t, post_mha_t, nf_w_t, nf_b_t,
        B, T, D, EPS, grid_dim=B * T, block_dim=BLOCK_T,
    )

    # ---- feed_forward: w_1 (D -> FF_DIM), swish, w_2 (FF_DIM -> D).
    comptime lin_ff1 = linear_kernel[
        DType.float32, type_of(btd_layout), type_of(w_ff1_layout),
        type_of(p_ff1_layout), type_of(btff_layout),
        True, BLOCK_T,
    ]
    ctx.enqueue_function[lin_ff1, lin_ff1](
        w1_out_t, nf_out_t, w_f1_t, b_f1_t, B, T, D, FF_DIM,
        grid_dim=B * T, block_dim=BLOCK_T,
    )
    comptime swish_k = swish_kernel[
        DType.float32, type_of(flat_ff), type_of(flat_ff), 256,
    ]
    ctx.enqueue_function[swish_k, swish_k](
        act_out_flat, w1_out_flat, n_ff,
        grid_dim=ceildiv(n_ff, 256), block_dim=256,
    )
    comptime lin_ff2 = linear_kernel[
        DType.float32, type_of(btff_layout), type_of(w_ff2_layout),
        type_of(p_layout), type_of(btd_layout),
        True, BLOCK_T,
    ]
    ctx.enqueue_function[lin_ff2, lin_ff2](
        w2_out_t, act_out_t, w_f2_t, b_f2_t, B, T, FF_DIM, D,
        grid_dim=B * T, block_dim=BLOCK_T,
    )

    # ---- Final residual: out = post_mha + w2_out.
    ctx.enqueue_function[add_x, add_x](
        out_flat, post_mha_flat, w2_out_flat, n_x,
        grid_dim=ceildiv(n_x, 256), block_dim=256,
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
                print("L0[", i, "]: mojo=", h[i], "  torch=", exp.data[i], "  diff=", d)
            assert_almost_equal(h[i], exp.data[i], atol=5.0e-3)
    print("encoder layer0 FULL — max abs:", max_abs,
          " mean:", sum_abs / Float64(n_x))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
