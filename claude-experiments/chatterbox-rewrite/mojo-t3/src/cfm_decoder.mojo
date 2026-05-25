"""
CFM ConditionalDecoder (estimator) host-side orchestration.

Provides:
  causal_block_1d           — CausalConv1d(k=3) + LayerNorm(C) + Mish
  causal_resnet_block_1d    — full CausalResnetBlock1D: block1 + (mlp+time_emb) + block2 + res_conv
"""
from std.math import ceildiv
from std.gpu import block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major

from conv import conv1d_kernel_fast, relu_kernel
from layernorm import (
    layernorm_kernel, linear_kernel, residual_add_kernel,
    transpose_btc_to_bct_kernel, transpose_bct_to_btc_kernel,
)
from attention import (
    swish_kernel, add_4d_kernel,
    qkv_proj_reshape_gen_kernel, matmul_qk_scaled_kernel,
    softmax_lastdim_kernel, matmul_av_kernel, merge_heads_kernel,
)
from decoder_kernels import (
    mish_kernel, gelu_kernel,
    multiply_mask_3d_kernel, add_3d_time_emb_kernel,
)


comptime EPS_LN: Float32 = 1.0e-5


def causal_block_1d[
    B: Int, IN_C: Int, OUT_C: Int, T: Int, K: Int,
](
    mut ctx: DeviceContext,
    mut x_buf: DeviceBuffer[DType.float32],         # (B, IN_C, T)
    mut mask_buf: DeviceBuffer[DType.float32],      # (B, 1, T)
    mut out_buf: DeviceBuffer[DType.float32],       # (B, OUT_C, T)
    mut conv_w: DeviceBuffer[DType.float32],        # (OUT_C, IN_C, K)
    mut conv_b: DeviceBuffer[DType.float32],        # (OUT_C,)
    mut ln_w: DeviceBuffer[DType.float32],          # (OUT_C,)
    mut ln_b: DeviceBuffer[DType.float32],          # (OUT_C,)
) raises:
    """CausalBlock1D forward: (x*mask) -> CausalConv1d(K) -> Transpose -> LN -> Transpose -> Mish -> *mask."""
    var n_in = B * IN_C * T
    var n_out = B * OUT_C * T

    var masked_in = ctx.enqueue_create_buffer[DType.float32](n_in)
    var conv_out = ctx.enqueue_create_buffer[DType.float32](n_out)
    var btc_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    var ln_btc = ctx.enqueue_create_buffer[DType.float32](n_out)
    var bct_back = ctx.enqueue_create_buffer[DType.float32](n_out)
    var mish_out = ctx.enqueue_create_buffer[DType.float32](n_out)

    comptime in_layout = row_major[B, IN_C, T]()
    comptime out_layout = row_major[B, OUT_C, T]()
    comptime mask_layout = row_major[B, 1, T]()
    comptime btc_out_layout = row_major[B, T, OUT_C]()
    comptime w_layout = row_major[OUT_C, IN_C, K]()
    comptime p_layout = row_major[OUT_C]()
    comptime flat_out = row_major[B * OUT_C * T]()

    var x_t = TileTensor(x_buf, in_layout)
    var mask_t = TileTensor(mask_buf, mask_layout)
    var masked_in_t = TileTensor(masked_in, in_layout)
    var conv_w_t = TileTensor(conv_w, w_layout)
    var conv_b_t = TileTensor(conv_b, p_layout)
    var conv_out_t = TileTensor(conv_out, out_layout)
    var btc_buf_t = TileTensor(btc_buf, btc_out_layout)
    var ln_w_t = TileTensor(ln_w, p_layout)
    var ln_b_t = TileTensor(ln_b, p_layout)
    var ln_btc_t = TileTensor(ln_btc, btc_out_layout)
    var bct_back_t = TileTensor(bct_back, out_layout)
    var bct_back_flat = TileTensor(bct_back, flat_out)
    var mish_out_flat = TileTensor(mish_out, flat_out)
    var mish_out_t = TileTensor(mish_out, out_layout)
    var out_t = TileTensor(out_buf, out_layout)

    # 1. x * mask (input dims = IN_C).
    comptime mul_in_k = multiply_mask_3d_kernel[
        DType.float32, type_of(in_layout), type_of(mask_layout),
        type_of(in_layout), 256,
    ]
    ctx.enqueue_function[mul_in_k, mul_in_k](
        masked_in_t, x_t, mask_t, B, IN_C, T,
        grid_dim=B * IN_C, block_dim=256,
    )

    # 2. CausalConv1d k=K (left-pad K-1, no right pad).
    comptime conv_k = conv1d_kernel_fast[
        DType.float32, type_of(in_layout), type_of(w_layout),
        type_of(p_layout), type_of(out_layout),
        K, True, 256,
    ]
    ctx.enqueue_function[conv_k, conv_k](
        conv_out_t, masked_in_t, conv_w_t, conv_b_t,
        B, IN_C, OUT_C, T, T, 1, K - 1, 1,
        grid_dim=B * OUT_C, block_dim=256,
    )

    # 3. Transpose to (B, T, C) for LayerNorm.
    comptime tp1_k = transpose_bct_to_btc_kernel[
        DType.float32, type_of(out_layout), type_of(btc_out_layout), 256,
    ]
    ctx.enqueue_function[tp1_k, tp1_k](
        btc_buf_t, conv_out_t, B, OUT_C, T,
        grid_dim=B * T, block_dim=256,
    )

    # 4. LayerNorm over channels.
    comptime ln_k = layernorm_kernel[
        DType.float32, type_of(btc_out_layout), type_of(p_layout),
        type_of(btc_out_layout), 256,
    ]
    ctx.enqueue_function[ln_k, ln_k](
        ln_btc_t, btc_buf_t, ln_w_t, ln_b_t,
        B, T, OUT_C, EPS_LN,
        grid_dim=B * T, block_dim=256,
    )

    # 5. Transpose back to (B, C, T).
    comptime tp2_k = transpose_btc_to_bct_kernel[
        DType.float32, type_of(btc_out_layout), type_of(out_layout), 256,
    ]
    ctx.enqueue_function[tp2_k, tp2_k](
        bct_back_t, ln_btc_t, B, T, OUT_C,
        grid_dim=B * OUT_C, block_dim=256,
    )

    # 6. Mish.
    comptime mish_k = mish_kernel[
        DType.float32, type_of(flat_out), type_of(flat_out), 256,
    ]
    ctx.enqueue_function[mish_k, mish_k](
        mish_out_flat, bct_back_flat, n_out,
        grid_dim=ceildiv(n_out, 256), block_dim=256,
    )

    # 7. * mask (output dims = OUT_C).
    comptime mul_out_k = multiply_mask_3d_kernel[
        DType.float32, type_of(out_layout), type_of(mask_layout),
        type_of(out_layout), 256,
    ]
    ctx.enqueue_function[mul_out_k, mul_out_k](
        out_t, mish_out_t, mask_t, B, OUT_C, T,
        grid_dim=B * OUT_C, block_dim=256,
    )


@fieldwise_init
struct BasicTransformerWeights(Copyable, Movable):
    """Weights for one BasicTransformerBlock (gelu FF, self-attn only)."""
    var norm1_w: DeviceBuffer[DType.float32]
    var norm1_b: DeviceBuffer[DType.float32]
    var to_q: DeviceBuffer[DType.float32]      # (D_inner, D), no bias
    var to_k: DeviceBuffer[DType.float32]
    var to_v: DeviceBuffer[DType.float32]
    var to_out_w: DeviceBuffer[DType.float32]  # (D, D_inner)
    var to_out_b: DeviceBuffer[DType.float32]
    var norm3_w: DeviceBuffer[DType.float32]
    var norm3_b: DeviceBuffer[DType.float32]
    var ff_proj_w: DeviceBuffer[DType.float32] # (FF_INNER, D)
    var ff_proj_b: DeviceBuffer[DType.float32]
    var ff_out_w: DeviceBuffer[DType.float32]  # (D, FF_INNER)
    var ff_out_b: DeviceBuffer[DType.float32]


def basic_transformer_block[
    B: Int, T: Int, D: Int, H: Int, D_K: Int, FF_INNER: Int,
](
    mut ctx: DeviceContext,
    mut x_buf: DeviceBuffer[DType.float32],         # (B, T, D)
    mut out_buf: DeviceBuffer[DType.float32],       # (B, T, D)
    mut w: BasicTransformerWeights,
) raises:
    """BasicTransformerBlock forward (no AdaLN, no cross-attn, gelu FF, mask is all-zero bias).

    Note: D_INNER = H * D_K (attention internal dim, may differ from D).
    """
    var D_INNER = H * D_K
    var n_x = B * T * D
    var n_qkv = B * H * T * D_K
    var n_scores = B * H * T * T
    var n_ff = B * T * FF_INNER
    var SCALE = Float32(1.0) / Float32(D_K) ** Float32(0.5)

    var n1_out = ctx.enqueue_create_buffer[DType.float32](n_x)
    var q_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv)
    var k_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv)
    var v_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv)
    var scores_buf = ctx.enqueue_create_buffer[DType.float32](n_scores)
    var attn_buf = ctx.enqueue_create_buffer[DType.float32](n_scores)
    var av_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv)
    var merged_buf = ctx.enqueue_create_buffer[DType.float32](B * T * D_INNER)
    var attn_out = ctx.enqueue_create_buffer[DType.float32](n_x)
    var post_mha = ctx.enqueue_create_buffer[DType.float32](n_x)
    var n3_out = ctx.enqueue_create_buffer[DType.float32](n_x)
    var ff_h = ctx.enqueue_create_buffer[DType.float32](n_ff)
    var ff_act = ctx.enqueue_create_buffer[DType.float32](n_ff)
    var ff_out = ctx.enqueue_create_buffer[DType.float32](n_x)

    comptime btd_layout = row_major[B, T, D]()
    comptime btd_inner_layout = row_major[B, T, D * H * D_K // D]()  # = (B, T, D_INNER)
    comptime btd_ff_layout = row_major[B, T, FF_INNER]()
    comptime wd_layout = row_major[D, D]()
    comptime w_qkv_layout = row_major[H * D_K, D]()    # D_INNER × D
    comptime w_out_layout = row_major[D, H * D_K]()    # D × D_INNER
    comptime w_ff_proj_layout = row_major[FF_INNER, D]()
    comptime w_ff_out_layout = row_major[D, FF_INNER]()
    comptime p_layout = row_major[D]()
    comptime p_inner_layout = row_major[H * D_K]()
    comptime p_ff_layout = row_major[FF_INNER]()
    comptime qkv_layout = row_major[B, H, T, D_K]()
    comptime scores_layout = row_major[B, H, T, T]()
    comptime flat_x = row_major[B * T * D]()
    comptime flat_ff = row_major[B * T * FF_INNER]()

    var x_t = TileTensor(x_buf, btd_layout)
    var n1_w_t = TileTensor(w.norm1_w, p_layout)
    var n1_b_t = TileTensor(w.norm1_b, p_layout)
    var n1_out_t = TileTensor(n1_out, btd_layout)
    var to_q_t = TileTensor(w.to_q, w_qkv_layout)
    var to_k_t = TileTensor(w.to_k, w_qkv_layout)
    var to_v_t = TileTensor(w.to_v, w_qkv_layout)
    var dummy_inner = ctx.enqueue_create_buffer[DType.float32](H * D_K)
    var dummy_inner_t = TileTensor(dummy_inner, p_inner_layout)
    var q_t = TileTensor(q_buf, qkv_layout)
    var k_t = TileTensor(k_buf, qkv_layout)
    var v_t = TileTensor(v_buf, qkv_layout)
    var scores_t = TileTensor(scores_buf, scores_layout)
    var attn_t = TileTensor(attn_buf, scores_layout)
    var av_t = TileTensor(av_buf, qkv_layout)
    var merged_t = TileTensor(merged_buf, btd_inner_layout)
    var to_out_w_t = TileTensor(w.to_out_w, w_out_layout)
    var to_out_b_t = TileTensor(w.to_out_b, p_layout)
    var attn_out_t = TileTensor(attn_out, btd_layout)
    var x_flat = TileTensor(x_buf, flat_x)
    var attn_out_flat = TileTensor(attn_out, flat_x)
    var post_mha_t = TileTensor(post_mha, btd_layout)
    var post_mha_flat = TileTensor(post_mha, flat_x)
    var n3_w_t = TileTensor(w.norm3_w, p_layout)
    var n3_b_t = TileTensor(w.norm3_b, p_layout)
    var n3_out_t = TileTensor(n3_out, btd_layout)
    var ff_proj_w_t = TileTensor(w.ff_proj_w, w_ff_proj_layout)
    var ff_proj_b_t = TileTensor(w.ff_proj_b, p_ff_layout)
    var ff_h_t = TileTensor(ff_h, btd_ff_layout)
    var ff_h_flat = TileTensor(ff_h, flat_ff)
    var ff_act_t = TileTensor(ff_act, btd_ff_layout)
    var ff_act_flat = TileTensor(ff_act, flat_ff)
    var ff_out_w_t = TileTensor(w.ff_out_w, w_ff_out_layout)
    var ff_out_b_t = TileTensor(w.ff_out_b, p_layout)
    var ff_out_t = TileTensor(ff_out, btd_layout)
    var ff_out_flat = TileTensor(ff_out, flat_x)
    var out_t = TileTensor(out_buf, btd_layout)
    var out_flat = TileTensor(out_buf, flat_x)

    # 1. norm1.
    comptime ln_k = layernorm_kernel[
        DType.float32, type_of(btd_layout), type_of(p_layout),
        type_of(btd_layout), 256,
    ]
    ctx.enqueue_function[ln_k, ln_k](
        n1_out_t, x_t, n1_w_t, n1_b_t, B, T, D, EPS_LN,
        grid_dim=B * T, block_dim=256,
    )

    # 2. q/k/v.
    comptime qkv_k = qkv_proj_reshape_gen_kernel[
        DType.float32, type_of(btd_layout), type_of(w_qkv_layout),
        type_of(p_inner_layout), type_of(qkv_layout),
        False, 64,
    ]
    ctx.enqueue_function[qkv_k, qkv_k](
        q_t, n1_out_t, to_q_t, dummy_inner_t, B, T, H, D_K, D,
        grid_dim=B * H * T, block_dim=64,
    )
    ctx.enqueue_function[qkv_k, qkv_k](
        k_t, n1_out_t, to_k_t, dummy_inner_t, B, T, H, D_K, D,
        grid_dim=B * H * T, block_dim=64,
    )
    ctx.enqueue_function[qkv_k, qkv_k](
        v_t, n1_out_t, to_v_t, dummy_inner_t, B, T, H, D_K, D,
        grid_dim=B * H * T, block_dim=64,
    )

    # 3. scores = q @ k^T * scale.
    comptime mm_qk = matmul_qk_scaled_kernel[
        DType.float32, type_of(qkv_layout), type_of(qkv_layout),
        type_of(scores_layout), 256,
    ]
    ctx.enqueue_function[mm_qk, mm_qk](
        scores_t, q_t, k_t, B, H, T, T, D_K, SCALE,
        grid_dim=B * H * T, block_dim=256,
    )

    # 4. softmax.
    comptime sm_k = softmax_lastdim_kernel[
        DType.float32, type_of(scores_layout), type_of(scores_layout), 256,
    ]
    ctx.enqueue_function[sm_k, sm_k](
        attn_t, scores_t, B, H, T, T, grid_dim=B * H * T, block_dim=256,
    )

    # 5. attn @ v.
    comptime av_k = matmul_av_kernel[
        DType.float32, type_of(scores_layout), type_of(qkv_layout),
        type_of(qkv_layout), 64,
    ]
    ctx.enqueue_function[av_k, av_k](
        av_t, attn_t, v_t, B, H, T, T, D_K,
        grid_dim=B * H * T, block_dim=64,
    )

    # 6. merge heads (B, H, T, D_k) -> (B, T, H*D_k).
    comptime mh_k = merge_heads_kernel[
        DType.float32, type_of(qkv_layout), type_of(btd_inner_layout), 64,
    ]
    ctx.enqueue_function[mh_k, mh_k](
        merged_t, av_t, B, H, T, D_K,
        grid_dim=B * H * T, block_dim=64,
    )

    # 7. to_out linear (D_INNER -> D, with bias).
    comptime lin_out_k = linear_kernel[
        DType.float32, type_of(btd_inner_layout), type_of(w_out_layout),
        type_of(p_layout), type_of(btd_layout),
        True, 256,
    ]
    ctx.enqueue_function[lin_out_k, lin_out_k](
        attn_out_t, merged_t, to_out_w_t, to_out_b_t,
        B, T, H * D_K, D,
        grid_dim=B * T, block_dim=256,
    )

    # 8. residual.
    comptime add_k = residual_add_kernel[
        DType.float32, type_of(flat_x), type_of(flat_x), type_of(flat_x), 256,
    ]
    ctx.enqueue_function[add_k, add_k](
        post_mha_flat, attn_out_flat, x_flat, n_x,
        grid_dim=ceildiv(n_x, 256), block_dim=256,
    )

    # 9. norm3.
    ctx.enqueue_function[ln_k, ln_k](
        n3_out_t, post_mha_t, n3_w_t, n3_b_t, B, T, D, EPS_LN,
        grid_dim=B * T, block_dim=256,
    )

    # 10. ff.net.0 = GELU's Linear (D -> FF_INNER, with bias).
    comptime lin_ff_proj_k = linear_kernel[
        DType.float32, type_of(btd_layout), type_of(w_ff_proj_layout),
        type_of(p_ff_layout), type_of(btd_ff_layout),
        True, 256,
    ]
    ctx.enqueue_function[lin_ff_proj_k, lin_ff_proj_k](
        ff_h_t, n3_out_t, ff_proj_w_t, ff_proj_b_t,
        B, T, D, FF_INNER,
        grid_dim=B * T, block_dim=256,
    )
    # GELU (exact, no tanh approximation).
    comptime gelu_k = gelu_kernel[
        DType.float32, type_of(flat_ff), type_of(flat_ff), 256,
    ]
    ctx.enqueue_function[gelu_k, gelu_k](
        ff_act_flat, ff_h_flat, n_ff,
        grid_dim=ceildiv(n_ff, 256), block_dim=256,
    )
    # ff.net.2 (FF_INNER -> D).
    comptime lin_ff_out_k = linear_kernel[
        DType.float32, type_of(btd_ff_layout), type_of(w_ff_out_layout),
        type_of(p_layout), type_of(btd_layout),
        True, 256,
    ]
    ctx.enqueue_function[lin_ff_out_k, lin_ff_out_k](
        ff_out_t, ff_act_t, ff_out_w_t, ff_out_b_t,
        B, T, FF_INNER, D,
        grid_dim=B * T, block_dim=256,
    )

    # 11. final residual.
    ctx.enqueue_function[add_k, add_k](
        out_flat, ff_out_flat, post_mha_flat, n_x,
        grid_dim=ceildiv(n_x, 256), block_dim=256,
    )


@fieldwise_init
struct CausalResnetWeights(Copyable, Movable):
    """Weights for one CausalResnetBlock1D."""
    var b1_conv_w: DeviceBuffer[DType.float32]
    var b1_conv_b: DeviceBuffer[DType.float32]
    var b1_ln_w: DeviceBuffer[DType.float32]
    var b1_ln_b: DeviceBuffer[DType.float32]
    var b2_conv_w: DeviceBuffer[DType.float32]
    var b2_conv_b: DeviceBuffer[DType.float32]
    var b2_ln_w: DeviceBuffer[DType.float32]
    var b2_ln_b: DeviceBuffer[DType.float32]
    var mlp_w: DeviceBuffer[DType.float32]
    var mlp_b: DeviceBuffer[DType.float32]
    var res_w: DeviceBuffer[DType.float32]
    var res_b: DeviceBuffer[DType.float32]


def causal_resnet_block_1d[
    B: Int, IN_C: Int, OUT_C: Int, T: Int, TIME_EMB_DIM: Int,
](
    mut ctx: DeviceContext,
    mut x_buf: DeviceBuffer[DType.float32],         # (B, IN_C, T)
    mut mask_buf: DeviceBuffer[DType.float32],      # (B, 1, T)
    mut t_emb_buf: DeviceBuffer[DType.float32],     # (B, TIME_EMB_DIM)
    mut out_buf: DeviceBuffer[DType.float32],       # (B, OUT_C, T)
    mut w: CausalResnetWeights,
) raises:
    """CausalResnetBlock1D forward.

    h = block1(x, mask)
    mlp = mish(t_emb) @ mlp_w + mlp_b → (B, OUT_C)
    h += mlp.unsqueeze(-1)
    h = block2(h, mask)
    res = res_conv(x * mask)
    out = h + res
    """
    var n_in = B * IN_C * T
    var n_out = B * OUT_C * T
    var n_te = B * TIME_EMB_DIM

    var h_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    var t_mish = ctx.enqueue_create_buffer[DType.float32](n_te)
    var t_mlp_out = ctx.enqueue_create_buffer[DType.float32](B * OUT_C)
    var h_plus_t = ctx.enqueue_create_buffer[DType.float32](n_out)
    var h2_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    var x_masked = ctx.enqueue_create_buffer[DType.float32](n_in)
    var res_out = ctx.enqueue_create_buffer[DType.float32](n_out)

    # ---- block1.
    causal_block_1d[B, IN_C, OUT_C, T, 3](
        ctx, x_buf, mask_buf, h_buf, w.b1_conv_w, w.b1_conv_b, w.b1_ln_w, w.b1_ln_b,
    )

    # ---- mlp(t_emb) → (B, OUT_C). mlp = Mish + Linear(TIME_EMB_DIM, OUT_C).
    comptime te_layout = row_major[B, TIME_EMB_DIM]()
    comptime te_btd_layout = row_major[B, 1, TIME_EMB_DIM]()
    comptime mlp_out_layout = row_major[B, OUT_C]()
    comptime mlp_btd_layout = row_major[B, 1, OUT_C]()
    comptime te_flat = row_major[B * TIME_EMB_DIM]()
    comptime mlp_w_layout = row_major[OUT_C, TIME_EMB_DIM]()
    comptime mlp_p_layout = row_major[OUT_C]()
    comptime in_layout = row_major[B, IN_C, T]()
    comptime out_layout = row_major[B, OUT_C, T]()
    comptime mask_layout = row_major[B, 1, T]()
    comptime flat_out = row_major[B * OUT_C * T]()
    comptime w_res_layout = row_major[OUT_C, IN_C, 1]()

    var t_emb_flat = TileTensor(t_emb_buf, te_flat)
    var t_mish_flat = TileTensor(t_mish, te_flat)
    var t_mish_btd = TileTensor(t_mish, te_btd_layout)
    var mlp_w_t = TileTensor(w.mlp_w, mlp_w_layout)
    var mlp_b_t = TileTensor(w.mlp_b, mlp_p_layout)
    var t_mlp_out_btd = TileTensor(t_mlp_out, mlp_btd_layout)
    var t_mlp_out_2d = TileTensor(t_mlp_out, mlp_out_layout)

    comptime mish_te_k = mish_kernel[
        DType.float32, type_of(te_flat), type_of(te_flat), 256,
    ]
    ctx.enqueue_function[mish_te_k, mish_te_k](
        t_mish_flat, t_emb_flat, n_te,
        grid_dim=ceildiv(n_te, 256), block_dim=256,
    )
    comptime lin_te_k = linear_kernel[
        DType.float32, type_of(te_btd_layout), type_of(mlp_w_layout),
        type_of(mlp_p_layout), type_of(mlp_btd_layout),
        True, 256,
    ]
    ctx.enqueue_function[lin_te_k, lin_te_k](
        t_mlp_out_btd, t_mish_btd, mlp_w_t, mlp_b_t,
        B, 1, TIME_EMB_DIM, OUT_C,
        grid_dim=B, block_dim=256,
    )

    # ---- h += mlp(t_emb).unsqueeze(-1) — broadcast over T.
    var h_t = TileTensor(h_buf, out_layout)
    var h_plus_t_t = TileTensor(h_plus_t, out_layout)
    comptime add_te_k = add_3d_time_emb_kernel[
        DType.float32, type_of(out_layout), type_of(mlp_out_layout),
        type_of(out_layout), 256,
    ]
    ctx.enqueue_function[add_te_k, add_te_k](
        h_plus_t_t, h_t, t_mlp_out_2d, B, OUT_C, T,
        grid_dim=B * OUT_C, block_dim=256,
    )

    # ---- block2.
    causal_block_1d[B, OUT_C, OUT_C, T, 3](
        ctx, h_plus_t, mask_buf, h2_buf, w.b2_conv_w, w.b2_conv_b, w.b2_ln_w, w.b2_ln_b,
    )

    # ---- res_conv(x * mask).
    var x_t = TileTensor(x_buf, in_layout)
    var mask_t = TileTensor(mask_buf, mask_layout)
    var x_masked_t = TileTensor(x_masked, in_layout)
    comptime mul_k = multiply_mask_3d_kernel[
        DType.float32, type_of(in_layout), type_of(mask_layout),
        type_of(in_layout), 256,
    ]
    ctx.enqueue_function[mul_k, mul_k](
        x_masked_t, x_t, mask_t, B, IN_C, T,
        grid_dim=B * IN_C, block_dim=256,
    )
    var w_res_t = TileTensor(w.res_w, w_res_layout)
    var b_res_t = TileTensor(w.res_b, mlp_p_layout)
    var res_out_t = TileTensor(res_out, out_layout)
    comptime conv_res_k = conv1d_kernel_fast[
        DType.float32, type_of(in_layout), type_of(w_res_layout),
        type_of(mlp_p_layout), type_of(out_layout),
        1, True, 256,
    ]
    ctx.enqueue_function[conv_res_k, conv_res_k](
        res_out_t, x_masked_t, w_res_t, b_res_t,
        B, IN_C, OUT_C, T, T, 1, 0, 1,
        grid_dim=B * OUT_C, block_dim=256,
    )

    # ---- out = h2 + res.
    var h2_flat = TileTensor(h2_buf, flat_out)
    var res_out_flat = TileTensor(res_out, flat_out)
    var out_flat = TileTensor(out_buf, flat_out)
    comptime add_k = residual_add_kernel[
        DType.float32, type_of(flat_out), type_of(flat_out),
        type_of(flat_out), 256,
    ]
    ctx.enqueue_function[add_k, add_k](
        out_flat, h2_flat, res_out_flat, n_out,
        grid_dim=ceildiv(n_out, 256), block_dim=256,
    )


def causal_conv1d_with_mask[
    B: Int, IN_C: Int, OUT_C: Int, T: Int, K: Int,
](
    mut ctx: DeviceContext,
    mut x_buf: DeviceBuffer[DType.float32],         # (B, IN_C, T)
    mut mask_buf: DeviceBuffer[DType.float32],      # (B, 1, T)
    mut out_buf: DeviceBuffer[DType.float32],       # (B, OUT_C, T)
    mut conv_w: DeviceBuffer[DType.float32],        # (OUT_C, IN_C, K)
    mut conv_b: DeviceBuffer[DType.float32],        # (OUT_C,)
) raises:
    """(x * mask) -> CausalConv1d(K). Pre-mask is part of the upstream call pattern."""
    var n_in = B * IN_C * T
    var n_out = B * OUT_C * T

    var masked_in = ctx.enqueue_create_buffer[DType.float32](n_in)

    comptime in_layout = row_major[B, IN_C, T]()
    comptime out_layout = row_major[B, OUT_C, T]()
    comptime mask_layout = row_major[B, 1, T]()
    comptime w_layout = row_major[OUT_C, IN_C, K]()
    comptime p_layout = row_major[OUT_C]()

    var x_t = TileTensor(x_buf, in_layout)
    var mask_t = TileTensor(mask_buf, mask_layout)
    var masked_in_t = TileTensor(masked_in, in_layout)
    var conv_w_t = TileTensor(conv_w, w_layout)
    var conv_b_t = TileTensor(conv_b, p_layout)
    var out_t = TileTensor(out_buf, out_layout)

    comptime mul_k = multiply_mask_3d_kernel[
        DType.float32, type_of(in_layout), type_of(mask_layout),
        type_of(in_layout), 256,
    ]
    ctx.enqueue_function[mul_k, mul_k](
        masked_in_t, x_t, mask_t, B, IN_C, T,
        grid_dim=B * IN_C, block_dim=256,
    )
    comptime conv_k = conv1d_kernel_fast[
        DType.float32, type_of(in_layout), type_of(w_layout),
        type_of(p_layout), type_of(out_layout),
        K, True, 256,
    ]
    ctx.enqueue_function[conv_k, conv_k](
        out_t, masked_in_t, conv_w_t, conv_b_t,
        B, IN_C, OUT_C, T, T, 1, K - 1, 1,
        grid_dim=B * OUT_C, block_dim=256,
    )


def transpose_with_mask_bct_btc[
    B: Int, C: Int, T: Int,
](
    mut ctx: DeviceContext,
    mut bct_in: DeviceBuffer[DType.float32],
    mut btc_out: DeviceBuffer[DType.float32],
) raises:
    """Transpose (B, C, T) -> (B, T, C). Convenience wrapper."""
    comptime bct_layout = row_major[B, C, T]()
    comptime btc_layout = row_major[B, T, C]()
    var bct_t = TileTensor(bct_in, bct_layout)
    var btc_t = TileTensor(btc_out, btc_layout)
    comptime tp_k = transpose_bct_to_btc_kernel[
        DType.float32, type_of(bct_layout), type_of(btc_layout), 256,
    ]
    ctx.enqueue_function[tp_k, tp_k](
        btc_t, bct_t, B, C, T,
        grid_dim=B * T, block_dim=256,
    )


def transpose_with_mask_btc_bct[
    B: Int, T: Int, C: Int,
](
    mut ctx: DeviceContext,
    mut btc_in: DeviceBuffer[DType.float32],
    mut bct_out: DeviceBuffer[DType.float32],
) raises:
    """Transpose (B, T, C) -> (B, C, T)."""
    comptime btc_layout = row_major[B, T, C]()
    comptime bct_layout = row_major[B, C, T]()
    var btc_t = TileTensor(btc_in, btc_layout)
    var bct_t = TileTensor(bct_out, bct_layout)
    comptime tp_k = transpose_btc_to_bct_kernel[
        DType.float32, type_of(btc_layout), type_of(bct_layout), 256,
    ]
    ctx.enqueue_function[tp_k, tp_k](
        bct_t, btc_t, B, T, C,
        grid_dim=B * C, block_dim=256,
    )


@fieldwise_init
struct EstimatorWeights(Copyable, Movable):
    """All weights for the full ConditionalDecoder estimator.
    Time MLP: time_mlp.linear_1, time_mlp.linear_2.
    Down: 1 stage with 1 resnet + 4 transformers + downsample CausalConv1d.
    Mid: 12 stages, each with 1 resnet + 4 transformers.
    Up: 1 stage with 1 resnet (input 2*D from skip cat) + 4 transformers + upsample CausalConv1d.
    Final: CausalBlock1D + final_proj Conv1d 1x1.
    """
    # time_mlp
    var tm_w1: DeviceBuffer[DType.float32]
    var tm_b1: DeviceBuffer[DType.float32]
    var tm_w2: DeviceBuffer[DType.float32]
    var tm_b2: DeviceBuffer[DType.float32]
    # down_block 0: resnet, 4 tblocks, downsample.
    var dn_rn: CausalResnetWeights
    var dn_tb0: BasicTransformerWeights
    var dn_tb1: BasicTransformerWeights
    var dn_tb2: BasicTransformerWeights
    var dn_tb3: BasicTransformerWeights
    var dn_ds_w: DeviceBuffer[DType.float32]
    var dn_ds_b: DeviceBuffer[DType.float32]
    # final_block (CausalBlock1D: conv1d k=3 + LN).
    var fb_cw: DeviceBuffer[DType.float32]
    var fb_cb: DeviceBuffer[DType.float32]
    var fb_lw: DeviceBuffer[DType.float32]
    var fb_lb: DeviceBuffer[DType.float32]
    # final_proj (Conv1d 1x1, 256 -> 80).
    var fp_w: DeviceBuffer[DType.float32]
    var fp_b: DeviceBuffer[DType.float32]
    # up_block 0: resnet, 4 tblocks, upsample.
    var up_rn: CausalResnetWeights
    var up_tb0: BasicTransformerWeights
    var up_tb1: BasicTransformerWeights
    var up_tb2: BasicTransformerWeights
    var up_tb3: BasicTransformerWeights
    var up_us_w: DeviceBuffer[DType.float32]
    var up_us_b: DeviceBuffer[DType.float32]


def estimator_forward[
    B: Int, T: Int, D: Int, H: Int, D_K: Int, FF_INNER: Int, TIME_EMB_DIM: Int,
    D_OUT_MEL: Int,
](
    mut ctx: DeviceContext,
    mut x_full: DeviceBuffer[DType.float32],          # (B, 320, T) — already packed [x, mu, spks_expand, cond]
    mut mask_buf: DeviceBuffer[DType.float32],        # (B, 1, T)
    mut t_emb_buf: DeviceBuffer[DType.float32],       # (B, TIME_EMB_DIM)
    mut out_buf: DeviceBuffer[DType.float32],         # (B, D_OUT_MEL, T)
    # Weights: down_block.
    mut dn_rn: CausalResnetWeights,
    mut dn_tb0: BasicTransformerWeights,
    mut dn_tb1: BasicTransformerWeights,
    mut dn_tb2: BasicTransformerWeights,
    mut dn_tb3: BasicTransformerWeights,
    mut dn_ds_w: DeviceBuffer[DType.float32],
    mut dn_ds_b: DeviceBuffer[DType.float32],
    # Mid blocks (12).
    mut mid_rns: List[CausalResnetWeights],     # 12 resnets
    mut mid_tb0s: List[BasicTransformerWeights],  # 12 tb0s
    mut mid_tb1s: List[BasicTransformerWeights],
    mut mid_tb2s: List[BasicTransformerWeights],
    mut mid_tb3s: List[BasicTransformerWeights],
    # Up block.
    mut up_rn: CausalResnetWeights,
    mut up_tb0: BasicTransformerWeights,
    mut up_tb1: BasicTransformerWeights,
    mut up_tb2: BasicTransformerWeights,
    mut up_tb3: BasicTransformerWeights,
    mut up_us_w: DeviceBuffer[DType.float32],
    mut up_us_b: DeviceBuffer[DType.float32],
    # Final block + proj.
    mut fb_cw: DeviceBuffer[DType.float32],
    mut fb_cb: DeviceBuffer[DType.float32],
    mut fb_lw: DeviceBuffer[DType.float32],
    mut fb_lb: DeviceBuffer[DType.float32],
    mut fp_w: DeviceBuffer[DType.float32],
    mut fp_b: DeviceBuffer[DType.float32],
) raises:
    """Full ConditionalDecoder estimator forward.

    Note: D=256 (internal), D_OUT_MEL=80, FF_INNER=1024, H=8, D_K=64,
    TIME_EMB_DIM=1024, input has 320 channels (packed).
    """
    var n_d = B * D * T
    var n_out = B * D_OUT_MEL * T

    # ---- 1. down_block 0.
    var pre_ds = ctx.enqueue_create_buffer[DType.float32](n_d)
    var post_ds = ctx.enqueue_create_buffer[DType.float32](n_d)
    var resnet_out = ctx.enqueue_create_buffer[DType.float32](n_d)
    var btc_buf = ctx.enqueue_create_buffer[DType.float32](n_d)
    var t0 = ctx.enqueue_create_buffer[DType.float32](n_d)
    var t1 = ctx.enqueue_create_buffer[DType.float32](n_d)
    var t2 = ctx.enqueue_create_buffer[DType.float32](n_d)
    var t3 = ctx.enqueue_create_buffer[DType.float32](n_d)

    causal_resnet_block_1d[B, 320, D, T, TIME_EMB_DIM](
        ctx, x_full, mask_buf, t_emb_buf, resnet_out, dn_rn,
    )
    transpose_with_mask_bct_btc[B, D, T](ctx, resnet_out, btc_buf)
    basic_transformer_block[B, T, D, H, D_K, FF_INNER](ctx, btc_buf, t0, dn_tb0)
    basic_transformer_block[B, T, D, H, D_K, FF_INNER](ctx, t0, t1, dn_tb1)
    basic_transformer_block[B, T, D, H, D_K, FF_INNER](ctx, t1, t2, dn_tb2)
    basic_transformer_block[B, T, D, H, D_K, FF_INNER](ctx, t2, t3, dn_tb3)
    transpose_with_mask_btc_bct[B, T, D](ctx, t3, pre_ds)
    causal_conv1d_with_mask[B, D, D, T, 3](
        ctx, pre_ds, mask_buf, post_ds, dn_ds_w, dn_ds_b,
    )

    # ---- 2. mid_blocks (12).
    var mid_a = ctx.enqueue_create_buffer[DType.float32](n_d)
    var mid_b = ctx.enqueue_create_buffer[DType.float32](n_d)
    var mid_btc = ctx.enqueue_create_buffer[DType.float32](n_d)
    var mid_t0 = ctx.enqueue_create_buffer[DType.float32](n_d)
    var mid_t1 = ctx.enqueue_create_buffer[DType.float32](n_d)
    var mid_t2 = ctx.enqueue_create_buffer[DType.float32](n_d)
    var mid_t3 = ctx.enqueue_create_buffer[DType.float32](n_d)
    var mid_rs = ctx.enqueue_create_buffer[DType.float32](n_d)

    # The 12 mid blocks rotate between (post_ds → mid_a → mid_b → mid_a → ...).
    # Iteration 0: src = post_ds, dst = mid_a
    # Iteration i for i in 1..11: src = (mid_a if i%2==1 else mid_b), dst = (mid_b if i%2==1 else mid_a)
    # Final output is in mid_b if iterations 12 → src/dst swap pattern: even count = mid_a → mid_b.
    var src = post_ds
    var dst_swap = 1   # 1=use mid_a as dst, 0=use mid_b
    for i in range(12):
        causal_resnet_block_1d[B, D, D, T, TIME_EMB_DIM](
            ctx, src, mask_buf, t_emb_buf, mid_rs, mid_rns[i],
        )
        transpose_with_mask_bct_btc[B, D, T](ctx, mid_rs, mid_btc)
        basic_transformer_block[B, T, D, H, D_K, FF_INNER](ctx, mid_btc, mid_t0, mid_tb0s[i])
        basic_transformer_block[B, T, D, H, D_K, FF_INNER](ctx, mid_t0, mid_t1, mid_tb1s[i])
        basic_transformer_block[B, T, D, H, D_K, FF_INNER](ctx, mid_t1, mid_t2, mid_tb2s[i])
        basic_transformer_block[B, T, D, H, D_K, FF_INNER](ctx, mid_t2, mid_t3, mid_tb3s[i])
        if dst_swap == 1:
            transpose_with_mask_btc_bct[B, T, D](ctx, mid_t3, mid_a)
            src = mid_a
            dst_swap = 0
        else:
            transpose_with_mask_btc_bct[B, T, D](ctx, mid_t3, mid_b)
            src = mid_b
            dst_swap = 1
    var mid_final = src   # last assigned src is the final mid output

    # ---- 3. up_block: cat([mid_final, pre_ds], dim=1) → resnet (2*D → D) → 4 tblocks → causal_conv.
    var n_cat = B * (2 * D) * T
    var up_cat = ctx.enqueue_create_buffer[DType.float32](n_cat)
    var up_pre_us = ctx.enqueue_create_buffer[DType.float32](n_d)
    var up_post_us = ctx.enqueue_create_buffer[DType.float32](n_d)
    var up_rs = ctx.enqueue_create_buffer[DType.float32](n_d)
    var up_btc = ctx.enqueue_create_buffer[DType.float32](n_d)
    var up_t0 = ctx.enqueue_create_buffer[DType.float32](n_d)
    var up_t1 = ctx.enqueue_create_buffer[DType.float32](n_d)
    var up_t2 = ctx.enqueue_create_buffer[DType.float32](n_d)
    var up_t3 = ctx.enqueue_create_buffer[DType.float32](n_d)

    # Channel concat (B, D, T) ++ (B, D, T) → (B, 2*D, T).
    comptime mid_layout = row_major[B, D, T]()
    comptime cat_layout = row_major[B, 2 * D, T]()
    var mid_final_t = TileTensor(mid_final, mid_layout)
    var pre_ds_t = TileTensor(pre_ds, mid_layout)
    var up_cat_t = TileTensor(up_cat, cat_layout)

    @parameter
    def channel_concat_kernel[
        dtype: DType, OutL: TensorLayout, AL: TensorLayout, BL: TensorLayout,
        CA: Int, CB: Int, BLOCK_: Int,
    ](
        output: TileTensor[dtype, OutL, MutAnyOrigin],
        a: TileTensor[dtype, AL, MutAnyOrigin],
        b: TileTensor[dtype, BL, MutAnyOrigin],
        batch: Int, time: Int,
    ):
        comptime assert a.flat_rank == 3
        comptime assert b.flat_rank == 3
        comptime assert output.flat_rank == 3
        var bid = block_idx.x
        var tid = thread_idx.x
        var c = bid % (CA + CB)
        var bb = bid // (CA + CB)
        var tt = tid
        while tt < time:
            var v: Float32 = 0.0
            if c < CA:
                v = rebind[Scalar[dtype]](a[bb, c, tt]).cast[DType.float32]()
            else:
                v = rebind[Scalar[dtype]](b[bb, c - CA, tt]).cast[DType.float32]()
            output[bb, c, tt] = rebind[output.ElementType](v.cast[dtype]())
            tt += BLOCK_

    comptime cat_k = channel_concat_kernel[
        DType.float32, type_of(cat_layout), type_of(mid_layout), type_of(mid_layout),
        D, D, 256,
    ]
    ctx.enqueue_function[cat_k, cat_k](
        up_cat_t, mid_final_t, pre_ds_t, B, T,
        grid_dim=B * (2 * D), block_dim=256,
    )

    causal_resnet_block_1d[B, 2 * D, D, T, TIME_EMB_DIM](
        ctx, up_cat, mask_buf, t_emb_buf, up_rs, up_rn,
    )
    transpose_with_mask_bct_btc[B, D, T](ctx, up_rs, up_btc)
    basic_transformer_block[B, T, D, H, D_K, FF_INNER](ctx, up_btc, up_t0, up_tb0)
    basic_transformer_block[B, T, D, H, D_K, FF_INNER](ctx, up_t0, up_t1, up_tb1)
    basic_transformer_block[B, T, D, H, D_K, FF_INNER](ctx, up_t1, up_t2, up_tb2)
    basic_transformer_block[B, T, D, H, D_K, FF_INNER](ctx, up_t2, up_t3, up_tb3)
    transpose_with_mask_btc_bct[B, T, D](ctx, up_t3, up_pre_us)
    causal_conv1d_with_mask[B, D, D, T, 3](
        ctx, up_pre_us, mask_buf, up_post_us, up_us_w, up_us_b,
    )

    # ---- 4. final_block (CausalBlock1D).
    var fb_out = ctx.enqueue_create_buffer[DType.float32](n_d)
    causal_block_1d[B, D, D, T, 3](
        ctx, up_post_us, mask_buf, fb_out, fb_cw, fb_cb, fb_lw, fb_lb,
    )

    # ---- 5. final_proj (Conv1d 1x1, 256 → D_OUT_MEL) + * mask.
    var fp_out = ctx.enqueue_create_buffer[DType.float32](n_out)
    causal_conv1d_with_mask[B, D, D_OUT_MEL, T, 1](
        ctx, fb_out, mask_buf, fp_out, fp_w, fp_b,
    )
    # Apply trailing mask.
    comptime final_layout = row_major[B, D_OUT_MEL, T]()
    comptime mask_layout = row_major[B, 1, T]()
    var fp_out_t = TileTensor(fp_out, final_layout)
    var out_t = TileTensor(out_buf, final_layout)
    var mask_t = TileTensor(mask_buf, mask_layout)
    comptime mul_final_k = multiply_mask_3d_kernel[
        DType.float32, type_of(final_layout), type_of(mask_layout),
        type_of(final_layout), 256,
    ]
    ctx.enqueue_function[mul_final_k, mul_final_k](
        out_t, fp_out_t, mask_t, B, D_OUT_MEL, T,
        grid_dim=B * D_OUT_MEL, block_dim=256,
    )
