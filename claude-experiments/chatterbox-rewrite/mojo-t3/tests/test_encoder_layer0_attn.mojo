"""
Parity test for the FULL RelPosition Multi-Head Attention of encoder layer 0.

Input:  enc_layer_0_norm_mha_out.bin   (1, 376, 512)   — post-norm input
        enc_embed_pos.bin              (1, 751, 512)   — relative pos encoding
Target: enc_layer_0_attn_out.bin       (1, 376, 512)   — output of self_attn(layer 0)

Operations:
  q, k, v = linear_q/k/v(x).view(B,T,H,D_k).transpose(1,2)     # all (B,H,T,D_k)
  q_btHd = q.transpose(1, 2)                                    # (B, T, H, D_k)
  p = linear_pos(pos_emb).view(B, -1, H, D_k).transpose(1, 2)   # (B, H, 2T-1, D_k)
  q_u = (q_btHd + pos_bias_u).transpose(1, 2)                   # (B, H, T, D_k)
  q_v = (q_btHd + pos_bias_v).transpose(1, 2)                   # (B, H, T, D_k)
  matrix_ac = q_u @ k^T                                          # (B, H, T, T)
  matrix_bd = rel_shift(q_v @ p^T)                              # (B, H, T, T)
  scores = (ac + bd) / sqrt(d_k)
  attn   = softmax(scores + mask)   (mask is all-ones so no-op)
  ctx    = attn @ v                                              # (B, H, T, D_k)
  merged = ctx.transpose(1, 2).view(B, T, H*D_k)                 # (B, T, D)
  out    = linear_out(merged)                                    # (B, T, D)
"""
from std.math import ceildiv, sqrt
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu import block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major

from fixture import load_fp32
from layernorm import linear_kernel
from attention import (
    qkv_proj_reshape_kernel, add_pos_bias_kernel, matmul_qk_scaled_kernel,
    matmul_qp_kernel, rel_shift_kernel, add_4d_kernel, softmax_lastdim_kernel,
    matmul_av_kernel, merge_heads_kernel,
)


comptime B = 1
comptime T = 376
comptime T_POS = 751   # 2 * T - 1
comptime D = 512
comptime H = 8
comptime D_K = 64
comptime BLOCK = 64
comptime BLOCK_T = 256


def linear_pos_no_bias_kernel[
    dtype: DType, InLayout: TensorLayout, WLayout: TensorLayout, OutLayout: TensorLayout,
    BLOCK_: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, H, T_pos, D_k)
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],        # (B, T_pos, D)
    w: TileTensor[dtype, WLayout, MutAnyOrigin],           # (D, D)  no bias
    batch: Int, t_pos: Int, heads: Int, d_k: Int, d_model: Int,
):
    """linear_pos(pos_emb).view(B, T_pos, H, D_k).transpose(1, 2). Launch B*H*T_pos."""
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


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def test_layer0_attn() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/s3gen/"
    var ctx = DeviceContext()

    var x_in = load_fp32(fix + "enc_layer_0_norm_mha_out.bin")
    var pos = load_fp32(fix + "enc_embed_pos.bin")
    var w_q = load_fp32(fix + "weights/flow__encoder__encoders__0__self_attn__linear_q__weight.bin")
    var b_q = load_fp32(fix + "weights/flow__encoder__encoders__0__self_attn__linear_q__bias.bin")
    var w_k = load_fp32(fix + "weights/flow__encoder__encoders__0__self_attn__linear_k__weight.bin")
    var b_k = load_fp32(fix + "weights/flow__encoder__encoders__0__self_attn__linear_k__bias.bin")
    var w_v = load_fp32(fix + "weights/flow__encoder__encoders__0__self_attn__linear_v__weight.bin")
    var b_v = load_fp32(fix + "weights/flow__encoder__encoders__0__self_attn__linear_v__bias.bin")
    var w_pos = load_fp32(fix + "weights/flow__encoder__encoders__0__self_attn__linear_pos__weight.bin")
    var pb_u = load_fp32(fix + "weights/flow__encoder__encoders__0__self_attn__pos_bias_u.bin")
    var pb_v = load_fp32(fix + "weights/flow__encoder__encoders__0__self_attn__pos_bias_v.bin")
    var w_out = load_fp32(fix + "weights/flow__encoder__encoders__0__self_attn__linear_out__weight.bin")
    var b_out = load_fp32(fix + "weights/flow__encoder__encoders__0__self_attn__linear_out__bias.bin")
    var exp = load_fp32(fix + "enc_layer_0_attn_out.bin")

    var n_x = B * T * D
    var n_pos = B * T_POS * D
    var n_w = D * D
    var n_qkv = B * H * T * D_K
    var n_p = B * H * T_POS * D_K
    var n_scores = B * H * T * T
    var n_bd_pre = B * H * T * T_POS

    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var pos_buf = ctx.enqueue_create_buffer[DType.float32](n_pos)
    var w_q_buf = ctx.enqueue_create_buffer[DType.float32](n_w)
    var b_q_buf = ctx.enqueue_create_buffer[DType.float32](D)
    var w_k_buf = ctx.enqueue_create_buffer[DType.float32](n_w)
    var b_k_buf = ctx.enqueue_create_buffer[DType.float32](D)
    var w_v_buf = ctx.enqueue_create_buffer[DType.float32](n_w)
    var b_v_buf = ctx.enqueue_create_buffer[DType.float32](D)
    var w_pos_buf = ctx.enqueue_create_buffer[DType.float32](n_w)
    var pb_u_buf = ctx.enqueue_create_buffer[DType.float32](H * D_K)
    var pb_v_buf = ctx.enqueue_create_buffer[DType.float32](H * D_K)
    var w_out_buf = ctx.enqueue_create_buffer[DType.float32](n_w)
    var b_out_buf = ctx.enqueue_create_buffer[DType.float32](D)

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
    var attn_buf = ctx.enqueue_create_buffer[DType.float32](n_scores)
    var ctx_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv)
    var merged_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_x)

    upload(x_buf, x_in.data, n_x)
    upload(pos_buf, pos.data, n_pos)
    upload(w_q_buf, w_q.data, n_w)
    upload(b_q_buf, b_q.data, D)
    upload(w_k_buf, w_k.data, n_w)
    upload(b_k_buf, b_k.data, D)
    upload(w_v_buf, w_v.data, n_w)
    upload(b_v_buf, b_v.data, D)
    upload(w_pos_buf, w_pos.data, n_w)
    upload(pb_u_buf, pb_u.data, H * D_K)
    upload(pb_v_buf, pb_v.data, H * D_K)
    upload(w_out_buf, w_out.data, n_w)
    upload(b_out_buf, b_out.data, D)

    comptime btd_layout = row_major[B, T, D]()
    comptime btd_pos_layout = row_major[B, T_POS, D]()
    comptime w_layout = row_major[D, D]()
    comptime p_layout = row_major[D]()
    comptime qkv_layout = row_major[B, H, T, D_K]()
    comptime pq_layout = row_major[B, H, T_POS, D_K]()
    comptime pb_layout = row_major[H, D_K]()
    comptime scores_layout = row_major[B, H, T, T]()
    comptime bd_pre_layout = row_major[B, H, T, T_POS]()
    comptime flat_qkv = row_major[B * H * T * D_K]()
    comptime flat_scores = row_major[B * H * T * T]()

    var x_t = TileTensor(x_buf, btd_layout)
    var pos_t = TileTensor(pos_buf, btd_pos_layout)
    var w_q_t = TileTensor(w_q_buf, w_layout)
    var b_q_t = TileTensor(b_q_buf, p_layout)
    var w_k_t = TileTensor(w_k_buf, w_layout)
    var b_k_t = TileTensor(b_k_buf, p_layout)
    var w_v_t = TileTensor(w_v_buf, w_layout)
    var b_v_t = TileTensor(b_v_buf, p_layout)
    var w_pos_t = TileTensor(w_pos_buf, w_layout)
    var pb_u_t = TileTensor(pb_u_buf, pb_layout)
    var pb_v_t = TileTensor(pb_v_buf, pb_layout)
    var w_out_t = TileTensor(w_out_buf, w_layout)
    var b_out_t = TileTensor(b_out_buf, p_layout)

    var q_t = TileTensor(q_buf, qkv_layout)
    var k_t = TileTensor(k_buf, qkv_layout)
    var v_t = TileTensor(v_buf, qkv_layout)
    var q_u_t = TileTensor(q_u_buf, qkv_layout)
    var q_v_t = TileTensor(q_v_buf, qkv_layout)
    var p_t = TileTensor(p_buf, pq_layout)
    var ac_t = TileTensor(ac_buf, scores_layout)
    var bd_pre_t = TileTensor(bd_pre_buf, bd_pre_layout)
    var bd_t = TileTensor(bd_buf, scores_layout)
    var scores_t = TileTensor(scores_buf, scores_layout)
    var attn_t = TileTensor(attn_buf, scores_layout)
    var ctx_t = TileTensor(ctx_buf, qkv_layout)
    var merged_t = TileTensor(merged_buf, btd_layout)
    var out_t = TileTensor(out_buf, btd_layout)
    var ac_flat = TileTensor(ac_buf, flat_scores)
    var bd_flat = TileTensor(bd_buf, flat_scores)
    var scores_flat = TileTensor(scores_buf, flat_scores)

    # ---- 1. q, k, v projections.
    comptime qkv_k = qkv_proj_reshape_kernel[
        DType.float32, type_of(btd_layout), type_of(w_layout),
        type_of(p_layout), type_of(qkv_layout),
        True, BLOCK,
    ]
    ctx.enqueue_function[qkv_k, qkv_k](
        q_t, x_t, w_q_t, b_q_t,
        B, T, H, D_K, D, grid_dim=B * H * T, block_dim=BLOCK,
    )
    ctx.enqueue_function[qkv_k, qkv_k](
        k_t, x_t, w_k_t, b_k_t,
        B, T, H, D_K, D, grid_dim=B * H * T, block_dim=BLOCK,
    )
    ctx.enqueue_function[qkv_k, qkv_k](
        v_t, x_t, w_v_t, b_v_t,
        B, T, H, D_K, D, grid_dim=B * H * T, block_dim=BLOCK,
    )

    # ---- 2. linear_pos(pos_emb).view(...).transpose(1, 2) → p (B, H, T_pos, D_k).
    comptime pos_k = linear_pos_no_bias_kernel[
        DType.float32, type_of(btd_pos_layout), type_of(w_layout),
        type_of(pq_layout), BLOCK,
    ]
    ctx.enqueue_function[pos_k, pos_k](
        p_t, pos_t, w_pos_t,
        B, T_POS, H, D_K, D, grid_dim=B * H * T_POS, block_dim=BLOCK,
    )

    # ---- 3. q_u = q + pos_bias_u; q_v = q + pos_bias_v.
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

    # ---- 4. matrix_ac = q_u @ k^T  /sqrt(d_k) is applied later as scale=1 here.
    # We apply scale at this step (matmul_qk_scaled multiplies by `scale`).
    # But the upstream does (ac + bd) / sqrt(d_k). So we should NOT scale here.
    comptime mm_qk_k = matmul_qk_scaled_kernel[
        DType.float32, type_of(qkv_layout), type_of(qkv_layout),
        type_of(scores_layout), BLOCK_T,
    ]
    ctx.enqueue_function[mm_qk_k, mm_qk_k](
        ac_t, q_u_t, k_t, B, H, T, T, D_K, Float32(1.0),
        grid_dim=B * H * T, block_dim=BLOCK_T,
    )

    # ---- 5. matrix_bd = rel_shift(q_v @ p^T).
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

    # ---- 6. scores = (ac + bd) / sqrt(d_k).
    # First add ac + bd → scores_buf, then scale.
    comptime add_k = add_4d_kernel[
        DType.float32, type_of(flat_scores), type_of(flat_scores),
        type_of(flat_scores), 256,
    ]
    ctx.enqueue_function[add_k, add_k](
        scores_flat, ac_flat, bd_flat, B * H * T * T,
        grid_dim=ceildiv(B * H * T * T, 256), block_dim=256,
    )
    # Scale in-place: write back scaled version. Reuse a kernel? Just multiply
    # via softmax — softmax is invariant to additive constants but NOT to
    # multiplicative scaling. So we must apply the scale.
    # Hack: do a tiny scale kernel inline.
    comptime SCALE: Float32 = 0.125    # 1.0 / sqrt(64) = 1/8 = 0.125
    var scaled_buf = ctx.enqueue_create_buffer[DType.float32](n_scores)
    var scaled_flat = TileTensor(scaled_buf, flat_scores)
    # Use add_4d_kernel with b = 0 trick? Simpler: write a single-arg scale kernel here.

    @parameter
    def scale_4d_kernel[
        dtype: DType,
        InLayout: TensorLayout, OutLayout: TensorLayout,
        BLOCK_: Int,
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

    comptime sc_k = scale_4d_kernel[
        DType.float32, type_of(flat_scores), type_of(flat_scores), 256,
    ]
    ctx.enqueue_function[sc_k, sc_k](
        scaled_flat, scores_flat, B * H * T * T, SCALE,
        grid_dim=ceildiv(B * H * T * T, 256), block_dim=256,
    )
    var scaled_t = TileTensor(scaled_buf, scores_layout)

    # ---- 7. softmax along last dim.
    comptime sm_k = softmax_lastdim_kernel[
        DType.float32, type_of(scores_layout), type_of(scores_layout), BLOCK_T,
    ]
    ctx.enqueue_function[sm_k, sm_k](
        attn_t, scaled_t, B, H, T, T,
        grid_dim=B * H * T, block_dim=BLOCK_T,
    )

    # ---- 8. ctx = attn @ v.
    comptime av_k = matmul_av_kernel[
        DType.float32, type_of(scores_layout), type_of(qkv_layout),
        type_of(qkv_layout), BLOCK,
    ]
    ctx.enqueue_function[av_k, av_k](
        ctx_t, attn_t, v_t, B, H, T, T, D_K,
        grid_dim=B * H * T, block_dim=BLOCK,
    )

    # ---- 9. merge heads (B, H, T, D_k) → (B, T, H*D_k).
    comptime mh_k = merge_heads_kernel[
        DType.float32, type_of(qkv_layout), type_of(btd_layout), BLOCK,
    ]
    ctx.enqueue_function[mh_k, mh_k](
        merged_t, ctx_t, B, H, T, D_K,
        grid_dim=B * H * T, block_dim=BLOCK,
    )

    # ---- 10. linear_out.
    comptime lin_k = linear_kernel[
        DType.float32, type_of(btd_layout), type_of(w_layout),
        type_of(p_layout), type_of(btd_layout),
        True, BLOCK_T,
    ]
    ctx.enqueue_function[lin_k, lin_k](
        out_t, merged_t, w_out_t, b_out_t, B, T, D, D,
        grid_dim=B * T, block_dim=BLOCK_T,
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
                print("attn[", i, "]: mojo=", h[i], "  torch=", exp.data[i], "  diff=", d)
            assert_almost_equal(h[i], exp.data[i], atol=2.0e-3)
    print("encoder layer0 self_attn (full RelPos MHA) — max abs:", max_abs,
          " mean:", sum_abs / Float64(n_x))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
