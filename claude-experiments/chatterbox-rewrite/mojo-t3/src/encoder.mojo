"""
UpsampleConformerEncoder host-side orchestrator.

Provides:
  ConformerLayerWeights        — bundle of weights for one ConformerEncoderLayer.
  conformer_layer              — host-side helper that runs one full layer:
       norm_mha + RelPos MHA + residual + norm_ff + Feed-Forward(D→FF_DIM→D) + residual.
  linear_pos_no_bias_kernel    — Linear with no bias that reshapes to (B, H, T_pos, D_k).
  scale_4d_kernel              — pointwise scale.
"""
from std.math import ceildiv
from std.gpu import block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major

from layernorm import (
    layernorm_kernel, linear_kernel, residual_add_kernel,
)
from attention import (
    qkv_proj_reshape_kernel, add_pos_bias_kernel, matmul_qk_scaled_kernel,
    matmul_qp_kernel, rel_shift_kernel, add_4d_kernel, softmax_lastdim_kernel,
    matmul_av_kernel, merge_heads_kernel, swish_kernel,
)


comptime EPS_LN: Float32 = 1.0e-5


def linear_pos_no_bias_kernel[
    dtype: DType, InLayout: TensorLayout, WLayout: TensorLayout, OutLayout: TensorLayout,
    BLOCK_: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, H, T_pos, D_k)
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],        # (B, T_pos, D)
    w: TileTensor[dtype, WLayout, MutAnyOrigin],           # (D, D) no bias
    batch: Int, t_pos: Int, heads: Int, d_k: Int, d_model: Int,
):
    """linear_pos(pos_emb).view(B, T_pos, H, D_k).transpose(1, 2)."""
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


def nearest_upsample_1d_kernel[
    dtype: DType, InLayout: TensorLayout, OutLayout: TensorLayout, BLOCK_: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, C, T_in * stride)
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],        # (B, C, T_in)
    batch: Int, channels: Int, t_in: Int, stride: Int,
):
    """Nearest-neighbor upsampling along the time axis: each input sample is
    repeated `stride` times. Launch: grid=B*C, block_dim=BLOCK_ over output T."""
    comptime assert inp.flat_rank == 3
    comptime assert output.flat_rank == 3
    var bid = block_idx.x
    var tid = thread_idx.x
    var c = bid % channels
    var b = bid // channels
    var t_out = t_in * stride
    var t = tid
    while t < t_out:
        var src = t // stride
        var v = rebind[Scalar[dtype]](inp[b, c, src]).cast[DType.float32]()
        output[b, c, t] = rebind[output.ElementType](v.cast[dtype]())
        t += BLOCK_


@fieldwise_init
struct ConformerLayerWeights(Copyable, Movable):
    """All weights for one ConformerEncoderLayer (no macaron, no cnn_module)."""
    var norm_mha_w: DeviceBuffer[DType.float32]
    var norm_mha_b: DeviceBuffer[DType.float32]
    var norm_ff_w: DeviceBuffer[DType.float32]
    var norm_ff_b: DeviceBuffer[DType.float32]
    var w_q: DeviceBuffer[DType.float32]
    var b_q: DeviceBuffer[DType.float32]
    var w_k: DeviceBuffer[DType.float32]
    var b_k: DeviceBuffer[DType.float32]
    var w_v: DeviceBuffer[DType.float32]
    var b_v: DeviceBuffer[DType.float32]
    var w_pos: DeviceBuffer[DType.float32]
    var pb_u: DeviceBuffer[DType.float32]
    var pb_v: DeviceBuffer[DType.float32]
    var w_out: DeviceBuffer[DType.float32]
    var b_out: DeviceBuffer[DType.float32]
    var w_f1: DeviceBuffer[DType.float32]
    var b_f1: DeviceBuffer[DType.float32]
    var w_f2: DeviceBuffer[DType.float32]
    var b_f2: DeviceBuffer[DType.float32]


def conformer_layer[
    B: Int, T: Int, T_POS: Int, D: Int, H: Int, D_K: Int, FF_DIM: Int,
](
    mut ctx: DeviceContext,
    mut x_buf: DeviceBuffer[DType.float32],         # (B, T, D)
    mut pos_buf: DeviceBuffer[DType.float32],       # (B, T_POS, D)
    mut out_buf: DeviceBuffer[DType.float32],       # (B, T, D)
    mut w: ConformerLayerWeights,
) raises:
    """One ConformerEncoderLayer forward (normalize_before=True, no macaron, no CNN).

       residual = x; x = norm_mha(x); x_attn = self_attn(x); x = residual + x_attn
       residual = x; x = norm_ff(x);  x_ff   = feed_forward(x); out = residual + x_ff
    """
    var n_x = B * T * D
    var n_qkv = B * H * T * D_K
    var n_p = B * H * T_POS * D_K
    var n_scores = B * H * T * T
    var n_bd_pre = B * H * T * T_POS
    var n_ff = B * T * FF_DIM
    var SCALE = Float32(1.0) / Float32(D_K) ** Float32(0.5)

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
    var av_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv)
    var merged_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var attn_out_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var post_mha = ctx.enqueue_create_buffer[DType.float32](n_x)
    var nf_out = ctx.enqueue_create_buffer[DType.float32](n_x)
    var w1_out = ctx.enqueue_create_buffer[DType.float32](n_ff)
    var act_out = ctx.enqueue_create_buffer[DType.float32](n_ff)
    var w2_out = ctx.enqueue_create_buffer[DType.float32](n_x)

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
    var nm_w_t = TileTensor(w.norm_mha_w, p_layout)
    var nm_b_t = TileTensor(w.norm_mha_b, p_layout)
    var nf_w_t = TileTensor(w.norm_ff_w, p_layout)
    var nf_b_t = TileTensor(w.norm_ff_b, p_layout)
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
    var av_t = TileTensor(av_buf, qkv_layout)
    var merged_t = TileTensor(merged_buf, btd_layout)
    var attn_out_t = TileTensor(attn_out_buf, btd_layout)
    var x_flat = TileTensor(x_buf, flat)
    var attn_out_flat = TileTensor(attn_out_buf, flat)
    var post_mha_t = TileTensor(post_mha, btd_layout)
    var post_mha_flat = TileTensor(post_mha, flat)
    var w_q_t = TileTensor(w.w_q, w_layout)
    var b_q_t = TileTensor(w.b_q, p_layout)
    var w_k_t = TileTensor(w.w_k, w_layout)
    var b_k_t = TileTensor(w.b_k, p_layout)
    var w_v_t = TileTensor(w.w_v, w_layout)
    var b_v_t = TileTensor(w.b_v, p_layout)
    var w_pos_t = TileTensor(w.w_pos, w_layout)
    var pb_u_t = TileTensor(w.pb_u, pb_layout)
    var pb_v_t = TileTensor(w.pb_v, pb_layout)
    var w_out_t = TileTensor(w.w_out, w_layout)
    var b_out_t = TileTensor(w.b_out, p_layout)
    var w_f1_t = TileTensor(w.w_f1, w_ff1_layout)
    var b_f1_t = TileTensor(w.b_f1, p_ff1_layout)
    var w1_out_t = TileTensor(w1_out, btff_layout)
    var w1_out_flat = TileTensor(w1_out, flat_ff)
    var act_out_t = TileTensor(act_out, btff_layout)
    var act_out_flat = TileTensor(act_out, flat_ff)
    var w_f2_t = TileTensor(w.w_f2, w_ff2_layout)
    var b_f2_t = TileTensor(w.b_f2, p_layout)
    var w2_out_t = TileTensor(w2_out, btd_layout)
    var w2_out_flat = TileTensor(w2_out, flat)
    var out_t = TileTensor(out_buf, btd_layout)
    var out_flat = TileTensor(out_buf, flat)

    # ---- norm_mha.
    comptime ln_k = layernorm_kernel[
        DType.float32, type_of(btd_layout), type_of(p_layout),
        type_of(btd_layout), 256,
    ]
    ctx.enqueue_function[ln_k, ln_k](
        nm_out_t, x_t, nm_w_t, nm_b_t, B, T, D, EPS_LN,
        grid_dim=B * T, block_dim=256,
    )

    # ---- self_attn.
    comptime qkv_k = qkv_proj_reshape_kernel[
        DType.float32, type_of(btd_layout), type_of(w_layout),
        type_of(p_layout), type_of(qkv_layout),
        True, 64,
    ]
    ctx.enqueue_function[qkv_k, qkv_k](
        q_t, nm_out_t, w_q_t, b_q_t,
        B, T, H, D_K, D, grid_dim=B * H * T, block_dim=64,
    )
    ctx.enqueue_function[qkv_k, qkv_k](
        k_t, nm_out_t, w_k_t, b_k_t,
        B, T, H, D_K, D, grid_dim=B * H * T, block_dim=64,
    )
    ctx.enqueue_function[qkv_k, qkv_k](
        v_t, nm_out_t, w_v_t, b_v_t,
        B, T, H, D_K, D, grid_dim=B * H * T, block_dim=64,
    )
    comptime pos_k = linear_pos_no_bias_kernel[
        DType.float32, type_of(btd_pos_layout), type_of(w_layout),
        type_of(pq_layout), 64,
    ]
    ctx.enqueue_function[pos_k, pos_k](
        p_t, pos_t, w_pos_t, B, T_POS, H, D_K, D,
        grid_dim=B * H * T_POS, block_dim=64,
    )
    comptime bias_k = add_pos_bias_kernel[
        DType.float32, type_of(qkv_layout), type_of(pb_layout),
        type_of(qkv_layout), 64,
    ]
    ctx.enqueue_function[bias_k, bias_k](
        q_u_t, q_t, pb_u_t, B, H, T, D_K, grid_dim=B * H * T, block_dim=64,
    )
    ctx.enqueue_function[bias_k, bias_k](
        q_v_t, q_t, pb_v_t, B, H, T, D_K, grid_dim=B * H * T, block_dim=64,
    )
    comptime mm_qk_k = matmul_qk_scaled_kernel[
        DType.float32, type_of(qkv_layout), type_of(qkv_layout),
        type_of(scores_layout), 256,
    ]
    ctx.enqueue_function[mm_qk_k, mm_qk_k](
        ac_t, q_u_t, k_t, B, H, T, T, D_K, Float32(1.0),
        grid_dim=B * H * T, block_dim=256,
    )
    comptime mm_qp_k = matmul_qp_kernel[
        DType.float32, type_of(qkv_layout), type_of(pq_layout),
        type_of(bd_pre_layout), 256,
    ]
    ctx.enqueue_function[mm_qp_k, mm_qp_k](
        bd_pre_t, q_v_t, p_t, B, H, T, T_POS, D_K,
        grid_dim=B * H * T, block_dim=256,
    )
    comptime shift_k = rel_shift_kernel[
        DType.float32, type_of(bd_pre_layout), type_of(scores_layout), 256,
    ]
    ctx.enqueue_function[shift_k, shift_k](
        bd_t, bd_pre_t, B, H, T, grid_dim=B * H * T, block_dim=256,
    )
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
    comptime sm_k = softmax_lastdim_kernel[
        DType.float32, type_of(scores_layout), type_of(scores_layout), 256,
    ]
    ctx.enqueue_function[sm_k, sm_k](
        attn_t, scaled_t, B, H, T, T,
        grid_dim=B * H * T, block_dim=256,
    )
    comptime av_k = matmul_av_kernel[
        DType.float32, type_of(scores_layout), type_of(qkv_layout),
        type_of(qkv_layout), 64,
    ]
    ctx.enqueue_function[av_k, av_k](
        av_t, attn_t, v_t, B, H, T, T, D_K,
        grid_dim=B * H * T, block_dim=64,
    )
    comptime mh_k = merge_heads_kernel[
        DType.float32, type_of(qkv_layout), type_of(btd_layout), 64,
    ]
    ctx.enqueue_function[mh_k, mh_k](
        merged_t, av_t, B, H, T, D_K,
        grid_dim=B * H * T, block_dim=64,
    )
    comptime lin_attn_k = linear_kernel[
        DType.float32, type_of(btd_layout), type_of(w_layout),
        type_of(p_layout), type_of(btd_layout),
        True, 256,
    ]
    ctx.enqueue_function[lin_attn_k, lin_attn_k](
        attn_out_t, merged_t, w_out_t, b_out_t, B, T, D, D,
        grid_dim=B * T, block_dim=256,
    )

    # ---- post-MHA residual.
    comptime add_x = residual_add_kernel[
        DType.float32, type_of(flat), type_of(flat), type_of(flat), 256,
    ]
    ctx.enqueue_function[add_x, add_x](
        post_mha_flat, x_flat, attn_out_flat, n_x,
        grid_dim=ceildiv(n_x, 256), block_dim=256,
    )

    # ---- norm_ff.
    ctx.enqueue_function[ln_k, ln_k](
        nf_out_t, post_mha_t, nf_w_t, nf_b_t, B, T, D, EPS_LN,
        grid_dim=B * T, block_dim=256,
    )

    # ---- FF: D -> FF_DIM (Linear + Swish) -> D (Linear).
    comptime lin_ff1 = linear_kernel[
        DType.float32, type_of(btd_layout), type_of(w_ff1_layout),
        type_of(p_ff1_layout), type_of(btff_layout),
        True, 256,
    ]
    ctx.enqueue_function[lin_ff1, lin_ff1](
        w1_out_t, nf_out_t, w_f1_t, b_f1_t, B, T, D, FF_DIM,
        grid_dim=B * T, block_dim=256,
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
        True, 256,
    ]
    ctx.enqueue_function[lin_ff2, lin_ff2](
        w2_out_t, act_out_t, w_f2_t, b_f2_t, B, T, FF_DIM, D,
        grid_dim=B * T, block_dim=256,
    )

    # ---- final residual.
    ctx.enqueue_function[add_x, add_x](
        out_flat, post_mha_flat, w2_out_flat, n_x,
        grid_dim=ceildiv(n_x, 256), block_dim=256,
    )
