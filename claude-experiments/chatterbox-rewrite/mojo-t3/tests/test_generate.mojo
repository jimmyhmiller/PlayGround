"""
End-to-end multi-step argmax decode parity test vs HF.

Pipeline (all on GPU, no host roundtrips during the decode loop):
  1. Embed initial_ids → x (1, T_PREFILL, 1024) via embed_lookup + add_pos_emb.
  2. 30-layer prefill, populating per-layer K/V cache slots [0..T_PREFILL).
  3. final_norm → speech_head matmul → argmax over last row → first gen token.
  4. For N_STEPS - 1 more steps:
       a. embed new token + pos_emb at cur_pos → (1, 1, 1024)
       b. 30-layer decode step, appending K/V to slot cur_pos.
       c. final_norm → speech_head → argmax → next token.
  Compare generated id sequence to expected_ids from the oracle (must match
  exactly — argmax is deterministic given identical math).
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from layout import TileTensor, row_major
from linalg.matmul import matmul

from fixture import Tensor, TensorI64, load_fp32, load_i64
from rmsnorm import rmsnorm_kernel
from rope import rope_kernel
from sdpa import (
    qk_scaled_kernel, softmax_kernel, av_kernel,
    qk_decode_kernel, softmax_decode_kernel, av_decode_kernel,
)
from mlp import silu_mul_kernel
from util_kernels import (
    add_kernel, bshd_to_bhsd_kernel, bhsd_to_bshd_kernel,
    cache_append_kernel, cache_copy_prefix_kernel,
)
from heads import embed_lookup_kernel, add_pos_emb_kernel, argmax_kernel


# T3 Llama config.
comptime BATCH = 1
comptime T_PREFILL = 15
comptime N_STEPS = 8
comptime MAX_CTX = T_PREFILL + N_STEPS         # 23, runtime cap on cache length
comptime N_HEADS = 16
comptime HEAD_DIM = 64
comptime HIDDEN = N_HEADS * HEAD_DIM           # 1024
comptime INTERMEDIATE = 4096
comptime N_LAYERS = 30
comptime V_SPEECH = 8194
comptime P_SPEECH = 4100
comptime SCALE: Float32 = 0.125
comptime EPS: Float32 = 1.0e-5
comptime ROPE_HALF = HEAD_DIM // 2

comptime RMS_BLOCK = 256
comptime POINTWISE_BLOCK = 256
comptime SOFTMAX_BLOCK = 32
comptime ARGMAX_BLOCK = 256

# Two SEQ regimes: prefill (T_PREFILL) and decode (1).
comptime PRE_ROWS = BATCH * T_PREFILL          # 15
comptime DEC_ROWS = BATCH * 1                  # 1

# Layouts for prefill.
comptime pre_x_2d_layout = row_major[PRE_ROWS, HIDDEN]()
comptime pre_x_3d_layout = row_major[BATCH, T_PREFILL, HIDDEN]()
comptime pre_bshd_layout = row_major[BATCH, T_PREFILL, N_HEADS, HEAD_DIM]()
comptime pre_bhsd_layout = row_major[BATCH, N_HEADS, T_PREFILL, HEAD_DIM]()
comptime pre_cs_layout = row_major[BATCH, T_PREFILL, HEAD_DIM]()
comptime pre_mask_layout = row_major[T_PREFILL, T_PREFILL]()
comptime pre_ss_layout = row_major[BATCH, N_HEADS, T_PREFILL, T_PREFILL]()
comptime pre_intermediate_layout = row_major[PRE_ROWS, INTERMEDIATE]()
comptime pre_ids_layout = row_major[BATCH, T_PREFILL]()

# Layouts for decode.
comptime dec_x_2d_layout = row_major[DEC_ROWS, HIDDEN]()
comptime dec_x_3d_layout = row_major[BATCH, 1, HIDDEN]()
comptime dec_bshd_layout = row_major[BATCH, 1, N_HEADS, HEAD_DIM]()
comptime dec_bhsd_layout = row_major[BATCH, N_HEADS, 1, HEAD_DIM]()
comptime dec_cs_layout = row_major[BATCH, 1, HEAD_DIM]()
comptime dec_probs_layout = row_major[BATCH, N_HEADS, 1, MAX_CTX]()
comptime dec_intermediate_layout = row_major[DEC_ROWS, INTERMEDIATE]()
comptime dec_ids_layout = row_major[BATCH, 1]()

# Shared layouts.
comptime kv_cache_layout = row_major[BATCH, N_HEADS, MAX_CTX, HEAD_DIM]()
comptime w_hh_layout = row_major[HIDDEN, HIDDEN]()
comptime w_in_inter_layout = row_major[HIDDEN, INTERMEDIATE]()
comptime w_inter_in_layout = row_major[INTERMEDIATE, HIDDEN]()
comptime norm_w_layout = row_major[HIDDEN]()
comptime speech_emb_layout = row_major[V_SPEECH, HIDDEN]()
comptime speech_pos_layout = row_major[P_SPEECH, HIDDEN]()
comptime speech_head_layout = row_major[V_SPEECH, HIDDEN]()
comptime pre_logits_layout = row_major[BATCH, T_PREFILL, V_SPEECH]()
comptime dec_logits_layout = row_major[BATCH, 1, V_SPEECH]()
comptime dec_logits_2d_layout = row_major[BATCH * 1, V_SPEECH]()


# ---- Kernel bindings ----
comptime rms_pre = rmsnorm_kernel[
    DType.float32, type_of(pre_x_2d_layout), type_of(norm_w_layout),
    type_of(pre_x_2d_layout), RMS_BLOCK,
]
comptime rms_dec = rmsnorm_kernel[
    DType.float32, type_of(dec_x_2d_layout), type_of(norm_w_layout),
    type_of(dec_x_2d_layout), RMS_BLOCK,
]
comptime bshd_to_bhsd_pre = bshd_to_bhsd_kernel[
    DType.float32, type_of(pre_bshd_layout), type_of(pre_bhsd_layout),
    BATCH, T_PREFILL, N_HEADS, HEAD_DIM,
]
comptime bshd_to_bhsd_dec = bshd_to_bhsd_kernel[
    DType.float32, type_of(dec_bshd_layout), type_of(dec_bhsd_layout),
    BATCH, 1, N_HEADS, HEAD_DIM,
]
comptime bhsd_to_bshd_pre = bhsd_to_bshd_kernel[
    DType.float32, type_of(pre_bhsd_layout), type_of(pre_bshd_layout),
    BATCH, T_PREFILL, N_HEADS, HEAD_DIM,
]
comptime bhsd_to_bshd_dec = bhsd_to_bshd_kernel[
    DType.float32, type_of(dec_bhsd_layout), type_of(dec_bshd_layout),
    BATCH, 1, N_HEADS, HEAD_DIM,
]
comptime rope_pre = rope_kernel[
    DType.float32, type_of(pre_bhsd_layout), type_of(pre_cs_layout),
    type_of(pre_bhsd_layout), HEAD_DIM, ROPE_HALF,
]
comptime rope_dec = rope_kernel[
    DType.float32, type_of(dec_bhsd_layout), type_of(dec_cs_layout),
    type_of(dec_bhsd_layout), HEAD_DIM, ROPE_HALF,
]
comptime qk_pre = qk_scaled_kernel[
    DType.float32, type_of(pre_bhsd_layout), type_of(pre_bhsd_layout),
    type_of(pre_mask_layout), type_of(pre_ss_layout), HEAD_DIM, T_PREFILL,
]
comptime sm_pre = softmax_kernel[
    DType.float32, type_of(pre_ss_layout), type_of(pre_ss_layout), T_PREFILL, SOFTMAX_BLOCK,
]
comptime av_pre = av_kernel[
    DType.float32, type_of(pre_ss_layout), type_of(pre_bhsd_layout),
    type_of(pre_bhsd_layout), T_PREFILL, HEAD_DIM,
]
comptime qk_dec = qk_decode_kernel[
    DType.float32, type_of(dec_bhsd_layout), type_of(kv_cache_layout),
    type_of(dec_probs_layout), HEAD_DIM, MAX_CTX,
]
comptime sm_dec = softmax_decode_kernel[
    DType.float32, type_of(dec_probs_layout), type_of(dec_probs_layout),
    MAX_CTX, SOFTMAX_BLOCK,
]
comptime av_dec = av_decode_kernel[
    DType.float32, type_of(dec_probs_layout), type_of(kv_cache_layout),
    type_of(dec_bhsd_layout), MAX_CTX, HEAD_DIM,
]
comptime add_pre = add_kernel[
    DType.float32, type_of(pre_x_2d_layout), type_of(pre_x_2d_layout),
    type_of(pre_x_2d_layout), POINTWISE_BLOCK,
]
comptime add_dec = add_kernel[
    DType.float32, type_of(dec_x_2d_layout), type_of(dec_x_2d_layout),
    type_of(dec_x_2d_layout), POINTWISE_BLOCK,
]
comptime silu_pre = silu_mul_kernel[
    DType.float32, type_of(pre_intermediate_layout), type_of(pre_intermediate_layout),
    type_of(pre_intermediate_layout), POINTWISE_BLOCK,
]
comptime silu_dec = silu_mul_kernel[
    DType.float32, type_of(dec_intermediate_layout), type_of(dec_intermediate_layout),
    type_of(dec_intermediate_layout), POINTWISE_BLOCK,
]
comptime cache_app = cache_append_kernel[
    DType.float32, type_of(dec_bhsd_layout), type_of(kv_cache_layout),
    BATCH, N_HEADS, HEAD_DIM,
]
comptime cache_copy = cache_copy_prefix_kernel[
    DType.float32, type_of(pre_bhsd_layout), type_of(kv_cache_layout),
    BATCH, N_HEADS, HEAD_DIM, T_PREFILL,
]
comptime emb_pre = embed_lookup_kernel[
    DType.float32, type_of(pre_ids_layout), type_of(speech_emb_layout),
    type_of(pre_x_3d_layout), HIDDEN,
]
comptime emb_dec = embed_lookup_kernel[
    DType.float32, type_of(dec_ids_layout), type_of(speech_emb_layout),
    type_of(dec_x_3d_layout), HIDDEN,
]
comptime pos_pre = add_pos_emb_kernel[
    DType.float32, type_of(pre_x_3d_layout), type_of(speech_pos_layout),
    type_of(pre_x_3d_layout), HIDDEN,
]
comptime pos_dec = add_pos_emb_kernel[
    DType.float32, type_of(dec_x_3d_layout), type_of(speech_pos_layout),
    type_of(dec_x_3d_layout), HIDDEN,
]
comptime argmax_dec = argmax_kernel[
    DType.float32, type_of(dec_logits_layout), type_of(dec_ids_layout),
    V_SPEECH, ARGMAX_BLOCK,
]


def upload_fp32(ctx: DeviceContext, buf: DeviceBuffer[DType.float32], data: List[Float32]) raises:
    var n = len(data)
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def test_generate_fp32() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix_gen = "tests/fixtures/generate/"
    var initial_ids = load_i64(fix_gen + "initial_ids_fp32.bin")
    var expected_ids = load_i64(fix_gen + "expected_ids_fp32.bin")
    var speech_emb = load_fp32(fix_gen + "speech_emb_fp32.bin")
    var speech_pos = load_fp32(fix_gen + "speech_pos_emb_fp32.bin")
    var speech_head = load_fp32(fix_gen + "speech_head_fp32.bin")

    var cos_full = load_fp32(fix_gen + "cos_full_fp32.bin")
    var sin_full = load_fp32(fix_gen + "sin_full_fp32.bin")
    var mask_prefill = load_fp32(fix_gen + "mask_prefill_fp32.bin")
    var final_norm = load_fp32(fix_gen + "final_norm_fp32.bin")

    assert_equal(initial_ids.shape[0], T_PREFILL)
    assert_equal(expected_ids.shape[0], N_STEPS)
    assert_equal(speech_emb.shape[0], V_SPEECH)
    assert_equal(speech_emb.shape[1], HIDDEN)
    assert_equal(speech_pos.shape[0], P_SPEECH)
    assert_equal(speech_head.shape[0], V_SPEECH)
    assert_equal(cos_full.shape[1], MAX_CTX)

    # ---- Sizes ----
    var n_pre_x = PRE_ROWS * HIDDEN
    var n_dec_x = DEC_ROWS * HIDDEN
    var n_dec_cs = BATCH * 1 * HEAD_DIM
    var n_pre_mask = T_PREFILL * T_PREFILL
    var n_pre_qkv = BATCH * N_HEADS * T_PREFILL * HEAD_DIM
    var n_dec_qkv = BATCH * N_HEADS * 1 * HEAD_DIM
    var n_pre_ss = BATCH * N_HEADS * T_PREFILL * T_PREFILL
    var n_kv_cache = BATCH * N_HEADS * MAX_CTX * HEAD_DIM
    var n_dec_probs = BATCH * N_HEADS * 1 * MAX_CTX
    var n_pre_inter = PRE_ROWS * INTERMEDIATE
    var n_dec_inter = DEC_ROWS * INTERMEDIATE
    var n_w_hh = HIDDEN * HIDDEN
    var n_w_in_inter = HIDDEN * INTERMEDIATE
    var n_w_inter_in = INTERMEDIATE * HIDDEN
    var n_speech_emb = V_SPEECH * HIDDEN
    var n_speech_pos = P_SPEECH * HIDDEN
    var n_dec_logits = BATCH * 1 * V_SPEECH

    var ctx = DeviceContext()
    var dctx = DeviceContextPtr(ctx)

    # ---- Persistent buffers ----
    var emb_buf = ctx.enqueue_create_buffer[DType.float32](n_speech_emb)
    var pos_buf = ctx.enqueue_create_buffer[DType.float32](n_speech_pos)
    var head_buf = ctx.enqueue_create_buffer[DType.float32](V_SPEECH * HIDDEN)
    var final_norm_buf = ctx.enqueue_create_buffer[DType.float32](HIDDEN)
    var cos_full_buf = ctx.enqueue_create_buffer[DType.float32](MAX_CTX * HEAD_DIM)
    var sin_full_buf = ctx.enqueue_create_buffer[DType.float32](MAX_CTX * HEAD_DIM)
    var mask_pre_buf = ctx.enqueue_create_buffer[DType.float32](n_pre_mask)
    var cos_dec_buf = ctx.enqueue_create_buffer[DType.float32](n_dec_cs)
    var sin_dec_buf = ctx.enqueue_create_buffer[DType.float32](n_dec_cs)
    var pre_ids_buf = ctx.enqueue_create_buffer[DType.int64](BATCH * T_PREFILL)
    var dec_id_buf = ctx.enqueue_create_buffer[DType.int64](BATCH * 1)
    var argmax_out_buf = ctx.enqueue_create_buffer[DType.int64](BATCH * 1)

    var in_norm_buf = ctx.enqueue_create_buffer[DType.float32](HIDDEN)
    var post_norm_buf = ctx.enqueue_create_buffer[DType.float32](HIDDEN)
    var qw_buf = ctx.enqueue_create_buffer[DType.float32](n_w_hh)
    var kw_buf = ctx.enqueue_create_buffer[DType.float32](n_w_hh)
    var vw_buf = ctx.enqueue_create_buffer[DType.float32](n_w_hh)
    var ow_buf = ctx.enqueue_create_buffer[DType.float32](n_w_hh)
    var gw_buf = ctx.enqueue_create_buffer[DType.float32](n_w_in_inter)
    var uw_buf = ctx.enqueue_create_buffer[DType.float32](n_w_in_inter)
    var dw_buf = ctx.enqueue_create_buffer[DType.float32](n_w_inter_in)

    var pre_x_buf = ctx.enqueue_create_buffer[DType.float32](n_pre_x)
    var pre_emb_buf = ctx.enqueue_create_buffer[DType.float32](n_pre_x)
    var pre_norm_buf = ctx.enqueue_create_buffer[DType.float32](n_pre_x)
    var pre_q_flat_buf = ctx.enqueue_create_buffer[DType.float32](n_pre_x)
    var pre_k_flat_buf = ctx.enqueue_create_buffer[DType.float32](n_pre_x)
    var pre_v_flat_buf = ctx.enqueue_create_buffer[DType.float32](n_pre_x)
    var pre_q_perm_buf = ctx.enqueue_create_buffer[DType.float32](n_pre_qkv)
    var pre_k_perm_buf = ctx.enqueue_create_buffer[DType.float32](n_pre_qkv)
    var pre_v_perm_buf = ctx.enqueue_create_buffer[DType.float32](n_pre_qkv)
    var pre_q_rot_buf = ctx.enqueue_create_buffer[DType.float32](n_pre_qkv)
    var pre_k_rot_buf = ctx.enqueue_create_buffer[DType.float32](n_pre_qkv)
    var pre_logits_buf = ctx.enqueue_create_buffer[DType.float32](n_pre_ss)
    var pre_probs_buf = ctx.enqueue_create_buffer[DType.float32](n_pre_ss)
    var pre_attn_perm_buf = ctx.enqueue_create_buffer[DType.float32](n_pre_qkv)
    var pre_attn_flat_buf = ctx.enqueue_create_buffer[DType.float32](n_pre_x)
    var pre_attn_proj_buf = ctx.enqueue_create_buffer[DType.float32](n_pre_x)
    var pre_post_attn_buf = ctx.enqueue_create_buffer[DType.float32](n_pre_x)
    var pre_post_norm_out_buf = ctx.enqueue_create_buffer[DType.float32](n_pre_x)
    var pre_gate_buf = ctx.enqueue_create_buffer[DType.float32](n_pre_inter)
    var pre_up_buf = ctx.enqueue_create_buffer[DType.float32](n_pre_inter)
    var pre_hidden_act_buf = ctx.enqueue_create_buffer[DType.float32](n_pre_inter)
    var pre_mlp_out_buf = ctx.enqueue_create_buffer[DType.float32](n_pre_x)
    var pre_final_out_buf = ctx.enqueue_create_buffer[DType.float32](n_pre_x)

    var dec_x_buf = ctx.enqueue_create_buffer[DType.float32](n_dec_x)
    var dec_emb_buf = ctx.enqueue_create_buffer[DType.float32](n_dec_x)
    var dec_norm_buf = ctx.enqueue_create_buffer[DType.float32](n_dec_x)
    var dec_q_flat_buf = ctx.enqueue_create_buffer[DType.float32](n_dec_qkv)
    var dec_k_flat_buf = ctx.enqueue_create_buffer[DType.float32](n_dec_qkv)
    var dec_v_flat_buf = ctx.enqueue_create_buffer[DType.float32](n_dec_qkv)
    var dec_q_perm_buf = ctx.enqueue_create_buffer[DType.float32](n_dec_qkv)
    var dec_k_perm_buf = ctx.enqueue_create_buffer[DType.float32](n_dec_qkv)
    var dec_v_perm_buf = ctx.enqueue_create_buffer[DType.float32](n_dec_qkv)
    var dec_q_rot_buf = ctx.enqueue_create_buffer[DType.float32](n_dec_qkv)
    var dec_k_rot_buf = ctx.enqueue_create_buffer[DType.float32](n_dec_qkv)
    var dec_logits_attn_buf = ctx.enqueue_create_buffer[DType.float32](n_dec_probs)
    var dec_probs_buf = ctx.enqueue_create_buffer[DType.float32](n_dec_probs)
    var dec_attn_perm_buf = ctx.enqueue_create_buffer[DType.float32](n_dec_qkv)
    var dec_attn_flat_buf = ctx.enqueue_create_buffer[DType.float32](n_dec_qkv)
    var dec_attn_proj_buf = ctx.enqueue_create_buffer[DType.float32](n_dec_x)
    var dec_post_attn_buf = ctx.enqueue_create_buffer[DType.float32](n_dec_x)
    var dec_post_norm_out_buf = ctx.enqueue_create_buffer[DType.float32](n_dec_x)
    var dec_gate_buf = ctx.enqueue_create_buffer[DType.float32](n_dec_inter)
    var dec_up_buf = ctx.enqueue_create_buffer[DType.float32](n_dec_inter)
    var dec_hidden_act_buf = ctx.enqueue_create_buffer[DType.float32](n_dec_inter)
    var dec_mlp_out_buf = ctx.enqueue_create_buffer[DType.float32](n_dec_x)
    var dec_final_out_buf = ctx.enqueue_create_buffer[DType.float32](n_dec_x)
    var dec_logits_buf = ctx.enqueue_create_buffer[DType.float32](n_dec_logits)

    var k_caches = List[DeviceBuffer[DType.float32]]()
    var v_caches = List[DeviceBuffer[DType.float32]]()
    for _ in range(N_LAYERS):
        k_caches.append(ctx.enqueue_create_buffer[DType.float32](n_kv_cache))
        v_caches.append(ctx.enqueue_create_buffer[DType.float32](n_kv_cache))

    upload_fp32(ctx, emb_buf, speech_emb.data)
    upload_fp32(ctx, pos_buf, speech_pos.data)
    upload_fp32(ctx, head_buf, speech_head.data)
    upload_fp32(ctx, final_norm_buf, final_norm.data)
    upload_fp32(ctx, cos_full_buf, cos_full.data)
    upload_fp32(ctx, sin_full_buf, sin_full.data)
    upload_fp32(ctx, mask_pre_buf, mask_prefill.data)
    with pre_ids_buf.map_to_host() as h:
        for i in range(T_PREFILL): h[i] = initial_ids.data[i]

    # Pre-build a transposed speech_head for the LM matmul (D, V).
    var head_t_layout = row_major[HIDDEN, V_SPEECH]()
    var head_t_buf = ctx.enqueue_create_buffer[DType.float32](V_SPEECH * HIDDEN)
    with head_t_buf.map_to_host() as h:
        for v in range(V_SPEECH):
            for d in range(HIDDEN):
                h[d * V_SPEECH + v] = speech_head.data[v * HIDDEN + d]
    var head_t_view = TileTensor(head_t_buf, head_t_layout)

    # ---- TileTensor views ----
    var emb_t = TileTensor(emb_buf, speech_emb_layout)
    var pos_t = TileTensor(pos_buf, speech_pos_layout)
    var final_norm_t = TileTensor(final_norm_buf, norm_w_layout)
    var mask_pre_t = TileTensor(mask_pre_buf, pre_mask_layout)
    var cos_dec_t = TileTensor(cos_dec_buf, dec_cs_layout)
    var sin_dec_t = TileTensor(sin_dec_buf, dec_cs_layout)

    var pre_ids_t = TileTensor(pre_ids_buf, pre_ids_layout)
    var dec_id_t = TileTensor(dec_id_buf, dec_ids_layout)
    var argmax_out_t = TileTensor(argmax_out_buf, dec_ids_layout)

    var in_norm_t = TileTensor(in_norm_buf, norm_w_layout)
    var post_norm_t = TileTensor(post_norm_buf, norm_w_layout)
    var qw_t = TileTensor(qw_buf, w_hh_layout)
    var kw_t = TileTensor(kw_buf, w_hh_layout)
    var vw_t = TileTensor(vw_buf, w_hh_layout)
    var ow_t = TileTensor(ow_buf, w_hh_layout)
    var gw_t = TileTensor(gw_buf, w_in_inter_layout)
    var uw_t = TileTensor(uw_buf, w_in_inter_layout)
    var dw_t = TileTensor(dw_buf, w_inter_in_layout)

    var pre_x_3d = TileTensor(pre_x_buf, pre_x_3d_layout)
    var pre_x_2d = TileTensor(pre_x_buf, pre_x_2d_layout)
    var pre_emb_3d = TileTensor(pre_emb_buf, pre_x_3d_layout)
    var pre_norm_2d = TileTensor(pre_norm_buf, pre_x_2d_layout)
    var pre_q_flat_2d = TileTensor(pre_q_flat_buf, pre_x_2d_layout)
    var pre_k_flat_2d = TileTensor(pre_k_flat_buf, pre_x_2d_layout)
    var pre_v_flat_2d = TileTensor(pre_v_flat_buf, pre_x_2d_layout)
    var pre_q_flat_4d = TileTensor(pre_q_flat_buf, pre_bshd_layout)
    var pre_k_flat_4d = TileTensor(pre_k_flat_buf, pre_bshd_layout)
    var pre_v_flat_4d = TileTensor(pre_v_flat_buf, pre_bshd_layout)
    var pre_q_perm = TileTensor(pre_q_perm_buf, pre_bhsd_layout)
    var pre_k_perm = TileTensor(pre_k_perm_buf, pre_bhsd_layout)
    var pre_v_perm = TileTensor(pre_v_perm_buf, pre_bhsd_layout)
    var pre_q_rot = TileTensor(pre_q_rot_buf, pre_bhsd_layout)
    var pre_k_rot = TileTensor(pre_k_rot_buf, pre_bhsd_layout)
    var pre_logits_t = TileTensor(pre_logits_buf, pre_ss_layout)
    var pre_probs_t = TileTensor(pre_probs_buf, pre_ss_layout)
    var pre_attn_perm = TileTensor(pre_attn_perm_buf, pre_bhsd_layout)
    var pre_attn_flat_4d = TileTensor(pre_attn_flat_buf, pre_bshd_layout)
    var pre_attn_flat_2d = TileTensor(pre_attn_flat_buf, pre_x_2d_layout)
    var pre_attn_proj_2d = TileTensor(pre_attn_proj_buf, pre_x_2d_layout)
    var pre_post_attn_2d = TileTensor(pre_post_attn_buf, pre_x_2d_layout)
    var pre_post_norm_out_2d = TileTensor(pre_post_norm_out_buf, pre_x_2d_layout)
    var pre_gate_t = TileTensor(pre_gate_buf, pre_intermediate_layout)
    var pre_up_t = TileTensor(pre_up_buf, pre_intermediate_layout)
    var pre_hidden_act_t = TileTensor(pre_hidden_act_buf, pre_intermediate_layout)
    var pre_mlp_out_t = TileTensor(pre_mlp_out_buf, pre_x_2d_layout)
    var pre_final_out_2d = TileTensor(pre_final_out_buf, pre_x_2d_layout)
    var cos_pre_t = TileTensor(cos_full_buf, pre_cs_layout)
    var sin_pre_t = TileTensor(sin_full_buf, pre_cs_layout)

    var dec_x_3d = TileTensor(dec_x_buf, dec_x_3d_layout)
    var dec_x_2d = TileTensor(dec_x_buf, dec_x_2d_layout)
    var dec_emb_3d = TileTensor(dec_emb_buf, dec_x_3d_layout)
    var dec_norm_2d = TileTensor(dec_norm_buf, dec_x_2d_layout)
    var dec_q_flat_2d = TileTensor(dec_q_flat_buf, dec_x_2d_layout)
    var dec_k_flat_2d = TileTensor(dec_k_flat_buf, dec_x_2d_layout)
    var dec_v_flat_2d = TileTensor(dec_v_flat_buf, dec_x_2d_layout)
    var dec_q_flat_4d = TileTensor(dec_q_flat_buf, dec_bshd_layout)
    var dec_k_flat_4d = TileTensor(dec_k_flat_buf, dec_bshd_layout)
    var dec_v_flat_4d = TileTensor(dec_v_flat_buf, dec_bshd_layout)
    var dec_q_perm = TileTensor(dec_q_perm_buf, dec_bhsd_layout)
    var dec_k_perm = TileTensor(dec_k_perm_buf, dec_bhsd_layout)
    var dec_v_perm = TileTensor(dec_v_perm_buf, dec_bhsd_layout)
    var dec_q_rot = TileTensor(dec_q_rot_buf, dec_bhsd_layout)
    var dec_k_rot = TileTensor(dec_k_rot_buf, dec_bhsd_layout)
    var dec_logits_attn_t = TileTensor(dec_logits_attn_buf, dec_probs_layout)
    var dec_probs_t = TileTensor(dec_probs_buf, dec_probs_layout)
    var dec_attn_perm = TileTensor(dec_attn_perm_buf, dec_bhsd_layout)
    var dec_attn_flat_4d = TileTensor(dec_attn_flat_buf, dec_bshd_layout)
    var dec_attn_flat_2d = TileTensor(dec_attn_flat_buf, dec_x_2d_layout)
    var dec_attn_proj_2d = TileTensor(dec_attn_proj_buf, dec_x_2d_layout)
    var dec_post_attn_2d = TileTensor(dec_post_attn_buf, dec_x_2d_layout)
    var dec_post_norm_out_2d = TileTensor(dec_post_norm_out_buf, dec_x_2d_layout)
    var dec_gate_t = TileTensor(dec_gate_buf, dec_intermediate_layout)
    var dec_up_t = TileTensor(dec_up_buf, dec_intermediate_layout)
    var dec_hidden_act_t = TileTensor(dec_hidden_act_buf, dec_intermediate_layout)
    var dec_mlp_out_t = TileTensor(dec_mlp_out_buf, dec_x_2d_layout)
    var dec_final_out_2d = TileTensor(dec_final_out_buf, dec_x_2d_layout)
    var dec_logits_t = TileTensor(dec_logits_buf, dec_logits_layout)
    var dec_logits_2d = TileTensor(dec_logits_buf, dec_logits_2d_layout)

    var n_pre_silu = PRE_ROWS * INTERMEDIATE
    var n_dec_silu = DEC_ROWS * INTERMEDIATE
    var n_pre_perm_threads = BATCH * N_HEADS * T_PREFILL
    var n_dec_perm_threads = BATCH * N_HEADS * 1

    # STEP A: Embed initial ids + position embedding.
    ctx.enqueue_function[emb_pre, emb_pre](
        pre_emb_3d, pre_ids_t, emb_t, BATCH, T_PREFILL,
        grid_dim=BATCH * T_PREFILL, block_dim=HIDDEN,
    )
    ctx.enqueue_function[pos_pre, pos_pre](
        pre_x_3d, pre_emb_3d, pos_t, BATCH, T_PREFILL, 0,
        grid_dim=BATCH * T_PREFILL, block_dim=HIDDEN,
    )

    # STEP B: 30-layer prefill with K/V cache capture.
    for L in range(N_LAYERS):
        var layer_dir = "tests/fixtures/forward/layer" + String(L) + "/"
        var in_norm = load_fp32(layer_dir + "in_norm_fp32.bin")
        var post_norm = load_fp32(layer_dir + "post_norm_fp32.bin")
        var qw = load_fp32(layer_dir + "qw_fp32.bin")
        var kw = load_fp32(layer_dir + "kw_fp32.bin")
        var vw = load_fp32(layer_dir + "vw_fp32.bin")
        var ow = load_fp32(layer_dir + "ow_fp32.bin")
        var gate_w = load_fp32(layer_dir + "gate_w_fp32.bin")
        var up_w = load_fp32(layer_dir + "up_w_fp32.bin")
        var down_w = load_fp32(layer_dir + "down_w_fp32.bin")
        upload_fp32(ctx, in_norm_buf, in_norm.data)
        upload_fp32(ctx, post_norm_buf, post_norm.data)
        upload_fp32(ctx, qw_buf, qw.data)
        upload_fp32(ctx, kw_buf, kw.data)
        upload_fp32(ctx, vw_buf, vw.data)
        upload_fp32(ctx, ow_buf, ow.data)
        upload_fp32(ctx, gw_buf, gate_w.data)
        upload_fp32(ctx, uw_buf, up_w.data)
        upload_fp32(ctx, dw_buf, down_w.data)

        var k_cache_t = TileTensor(k_caches[L], kv_cache_layout)
        var v_cache_t = TileTensor(v_caches[L], kv_cache_layout)

        ctx.enqueue_function[rms_pre, rms_pre](
            pre_norm_2d, pre_x_2d, in_norm_t, EPS,
            grid_dim=PRE_ROWS, block_dim=RMS_BLOCK,
        )
        matmul[target="gpu"](pre_q_flat_2d, pre_norm_2d, qw_t, dctx)
        matmul[target="gpu"](pre_k_flat_2d, pre_norm_2d, kw_t, dctx)
        matmul[target="gpu"](pre_v_flat_2d, pre_norm_2d, vw_t, dctx)
        ctx.enqueue_function[bshd_to_bhsd_pre, bshd_to_bhsd_pre](
            pre_q_perm, pre_q_flat_4d, grid_dim=n_pre_perm_threads, block_dim=HEAD_DIM,
        )
        ctx.enqueue_function[bshd_to_bhsd_pre, bshd_to_bhsd_pre](
            pre_k_perm, pre_k_flat_4d, grid_dim=n_pre_perm_threads, block_dim=HEAD_DIM,
        )
        ctx.enqueue_function[bshd_to_bhsd_pre, bshd_to_bhsd_pre](
            pre_v_perm, pre_v_flat_4d, grid_dim=n_pre_perm_threads, block_dim=HEAD_DIM,
        )
        ctx.enqueue_function[rope_pre, rope_pre](
            pre_q_rot, pre_q_perm, cos_pre_t, sin_pre_t, N_HEADS, T_PREFILL,
            grid_dim=BATCH * N_HEADS * T_PREFILL, block_dim=HEAD_DIM,
        )
        ctx.enqueue_function[rope_pre, rope_pre](
            pre_k_rot, pre_k_perm, cos_pre_t, sin_pre_t, N_HEADS, T_PREFILL,
            grid_dim=BATCH * N_HEADS * T_PREFILL, block_dim=HEAD_DIM,
        )
        # Snapshot k_rot / v_perm into K/V caches slots [0..T_PREFILL).
        ctx.enqueue_function[cache_copy, cache_copy](
            k_cache_t, pre_k_rot,
            grid_dim=BATCH * N_HEADS * T_PREFILL, block_dim=HEAD_DIM,
        )
        ctx.enqueue_function[cache_copy, cache_copy](
            v_cache_t, pre_v_perm,
            grid_dim=BATCH * N_HEADS * T_PREFILL, block_dim=HEAD_DIM,
        )
        ctx.enqueue_function[qk_pre, qk_pre](
            pre_logits_t, pre_q_rot, pre_k_rot, mask_pre_t, N_HEADS, SCALE,
            grid_dim=BATCH * N_HEADS * T_PREFILL, block_dim=T_PREFILL,
        )
        ctx.enqueue_function[sm_pre, sm_pre](
            pre_probs_t, pre_logits_t, N_HEADS,
            grid_dim=BATCH * N_HEADS * T_PREFILL, block_dim=SOFTMAX_BLOCK,
        )
        ctx.enqueue_function[av_pre, av_pre](
            pre_attn_perm, pre_probs_t, pre_v_perm, N_HEADS,
            grid_dim=BATCH * N_HEADS * T_PREFILL, block_dim=HEAD_DIM,
        )
        ctx.enqueue_function[bhsd_to_bshd_pre, bhsd_to_bshd_pre](
            pre_attn_flat_4d, pre_attn_perm,
            grid_dim=n_pre_perm_threads, block_dim=HEAD_DIM,
        )
        matmul[target="gpu"](pre_attn_proj_2d, pre_attn_flat_2d, ow_t, dctx)
        ctx.enqueue_function[add_pre, add_pre](
            pre_post_attn_2d, pre_x_2d, pre_attn_proj_2d, n_pre_x,
            grid_dim=ceildiv(n_pre_x, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
        )
        ctx.enqueue_function[rms_pre, rms_pre](
            pre_post_norm_out_2d, pre_post_attn_2d, post_norm_t, EPS,
            grid_dim=PRE_ROWS, block_dim=RMS_BLOCK,
        )
        matmul[target="gpu"](pre_gate_t, pre_post_norm_out_2d, gw_t, dctx)
        matmul[target="gpu"](pre_up_t, pre_post_norm_out_2d, uw_t, dctx)
        ctx.enqueue_function[silu_pre, silu_pre](
            pre_hidden_act_t, pre_gate_t, pre_up_t, n_pre_silu,
            grid_dim=ceildiv(n_pre_silu, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
        )
        matmul[target="gpu"](pre_mlp_out_t, pre_hidden_act_t, dw_t, dctx)
        ctx.enqueue_function[add_pre, add_pre](
            pre_x_2d, pre_post_attn_2d, pre_mlp_out_t, n_pre_x,
            grid_dim=ceildiv(n_pre_x, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
        )

    ctx.enqueue_function[rms_pre, rms_pre](
        pre_final_out_2d, pre_x_2d, final_norm_t, EPS,
        grid_dim=PRE_ROWS, block_dim=RMS_BLOCK,
    )

    # STEP C: Argmax over last prefill row.
    ctx.synchronize()
    var generated = List[Int]()
    with pre_final_out_buf.map_to_host() as src:
        with dec_x_buf.map_to_host() as dst:
            for d in range(HIDDEN):
                dst[d] = src[(T_PREFILL - 1) * HIDDEN + d]

    matmul[target="gpu"](dec_logits_2d, dec_x_2d, head_t_view, dctx)
    ctx.enqueue_function[argmax_dec, argmax_dec](
        argmax_out_t, dec_logits_t, 1, 1,
        grid_dim=1, block_dim=ARGMAX_BLOCK,
    )
    ctx.synchronize()
    with argmax_out_buf.map_to_host() as ao:
        generated.append(Int(ao[0]))
    print("step 0 (prefill argmax) →", generated[0], " expected", Int(expected_ids.data[0]))

    # STEP D: N_STEPS - 1 autoregressive decode steps.
    for step in range(N_STEPS - 1):
        var cur_pos = T_PREFILL + step
        var next_id = generated[step]

        with cos_full_buf.map_to_host() as src:
            with cos_dec_buf.map_to_host() as dst:
                for d in range(HEAD_DIM):
                    dst[d] = src[cur_pos * HEAD_DIM + d]
        with sin_full_buf.map_to_host() as src:
            with sin_dec_buf.map_to_host() as dst:
                for d in range(HEAD_DIM):
                    dst[d] = src[cur_pos * HEAD_DIM + d]
        with dec_id_buf.map_to_host() as h:
            h[0] = Int64(next_id)

        ctx.enqueue_function[emb_dec, emb_dec](
            dec_emb_3d, dec_id_t, emb_t, BATCH, 1,
            grid_dim=BATCH * 1, block_dim=HIDDEN,
        )
        ctx.enqueue_function[pos_dec, pos_dec](
            dec_x_3d, dec_emb_3d, pos_t, BATCH, 1, cur_pos,
            grid_dim=BATCH * 1, block_dim=HIDDEN,
        )

        for L in range(N_LAYERS):
            var layer_dir = "tests/fixtures/forward/layer" + String(L) + "/"
            var in_norm = load_fp32(layer_dir + "in_norm_fp32.bin")
            var post_norm = load_fp32(layer_dir + "post_norm_fp32.bin")
            var qw = load_fp32(layer_dir + "qw_fp32.bin")
            var kw = load_fp32(layer_dir + "kw_fp32.bin")
            var vw = load_fp32(layer_dir + "vw_fp32.bin")
            var ow = load_fp32(layer_dir + "ow_fp32.bin")
            var gate_w = load_fp32(layer_dir + "gate_w_fp32.bin")
            var up_w = load_fp32(layer_dir + "up_w_fp32.bin")
            var down_w = load_fp32(layer_dir + "down_w_fp32.bin")
            upload_fp32(ctx, in_norm_buf, in_norm.data)
            upload_fp32(ctx, post_norm_buf, post_norm.data)
            upload_fp32(ctx, qw_buf, qw.data)
            upload_fp32(ctx, kw_buf, kw.data)
            upload_fp32(ctx, vw_buf, vw.data)
            upload_fp32(ctx, ow_buf, ow.data)
            upload_fp32(ctx, gw_buf, gate_w.data)
            upload_fp32(ctx, uw_buf, up_w.data)
            upload_fp32(ctx, dw_buf, down_w.data)

            var k_cache_t = TileTensor(k_caches[L], kv_cache_layout)
            var v_cache_t = TileTensor(v_caches[L], kv_cache_layout)

            ctx.enqueue_function[rms_dec, rms_dec](
                dec_norm_2d, dec_x_2d, in_norm_t, EPS,
                grid_dim=DEC_ROWS, block_dim=RMS_BLOCK,
            )
            matmul[target="gpu"](dec_q_flat_2d, dec_norm_2d, qw_t, dctx)
            matmul[target="gpu"](dec_k_flat_2d, dec_norm_2d, kw_t, dctx)
            matmul[target="gpu"](dec_v_flat_2d, dec_norm_2d, vw_t, dctx)
            ctx.enqueue_function[bshd_to_bhsd_dec, bshd_to_bhsd_dec](
                dec_q_perm, dec_q_flat_4d, grid_dim=n_dec_perm_threads, block_dim=HEAD_DIM,
            )
            ctx.enqueue_function[bshd_to_bhsd_dec, bshd_to_bhsd_dec](
                dec_k_perm, dec_k_flat_4d, grid_dim=n_dec_perm_threads, block_dim=HEAD_DIM,
            )
            ctx.enqueue_function[bshd_to_bhsd_dec, bshd_to_bhsd_dec](
                dec_v_perm, dec_v_flat_4d, grid_dim=n_dec_perm_threads, block_dim=HEAD_DIM,
            )
            ctx.enqueue_function[rope_dec, rope_dec](
                dec_q_rot, dec_q_perm, cos_dec_t, sin_dec_t, N_HEADS, 1,
                grid_dim=BATCH * N_HEADS * 1, block_dim=HEAD_DIM,
            )
            ctx.enqueue_function[rope_dec, rope_dec](
                dec_k_rot, dec_k_perm, cos_dec_t, sin_dec_t, N_HEADS, 1,
                grid_dim=BATCH * N_HEADS * 1, block_dim=HEAD_DIM,
            )
            ctx.enqueue_function[cache_app, cache_app](
                k_cache_t, dec_k_rot, cur_pos,
                grid_dim=BATCH * N_HEADS, block_dim=HEAD_DIM,
            )
            ctx.enqueue_function[cache_app, cache_app](
                v_cache_t, dec_v_perm, cur_pos,
                grid_dim=BATCH * N_HEADS, block_dim=HEAD_DIM,
            )
            ctx.enqueue_function[qk_dec, qk_dec](
                dec_logits_attn_t, dec_q_rot, k_cache_t, N_HEADS, cur_pos + 1, SCALE,
                grid_dim=BATCH * N_HEADS, block_dim=MAX_CTX,
            )
            ctx.enqueue_function[sm_dec, sm_dec](
                dec_probs_t, dec_logits_attn_t, N_HEADS,
                grid_dim=BATCH * N_HEADS, block_dim=SOFTMAX_BLOCK,
            )
            ctx.enqueue_function[av_dec, av_dec](
                dec_attn_perm, dec_probs_t, v_cache_t, N_HEADS, cur_pos + 1,
                grid_dim=BATCH * N_HEADS, block_dim=HEAD_DIM,
            )
            ctx.enqueue_function[bhsd_to_bshd_dec, bhsd_to_bshd_dec](
                dec_attn_flat_4d, dec_attn_perm,
                grid_dim=n_dec_perm_threads, block_dim=HEAD_DIM,
            )
            matmul[target="gpu"](dec_attn_proj_2d, dec_attn_flat_2d, ow_t, dctx)
            ctx.enqueue_function[add_dec, add_dec](
                dec_post_attn_2d, dec_x_2d, dec_attn_proj_2d, n_dec_x,
                grid_dim=ceildiv(n_dec_x, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
            )
            ctx.enqueue_function[rms_dec, rms_dec](
                dec_post_norm_out_2d, dec_post_attn_2d, post_norm_t, EPS,
                grid_dim=DEC_ROWS, block_dim=RMS_BLOCK,
            )
            matmul[target="gpu"](dec_gate_t, dec_post_norm_out_2d, gw_t, dctx)
            matmul[target="gpu"](dec_up_t, dec_post_norm_out_2d, uw_t, dctx)
            ctx.enqueue_function[silu_dec, silu_dec](
                dec_hidden_act_t, dec_gate_t, dec_up_t, n_dec_silu,
                grid_dim=ceildiv(n_dec_silu, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
            )
            matmul[target="gpu"](dec_mlp_out_t, dec_hidden_act_t, dw_t, dctx)
            ctx.enqueue_function[add_dec, add_dec](
                dec_x_2d, dec_post_attn_2d, dec_mlp_out_t, n_dec_x,
                grid_dim=ceildiv(n_dec_x, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
            )

        ctx.enqueue_function[rms_dec, rms_dec](
            dec_final_out_2d, dec_x_2d, final_norm_t, EPS,
            grid_dim=DEC_ROWS, block_dim=RMS_BLOCK,
        )
        matmul[target="gpu"](dec_logits_2d, dec_final_out_2d, head_t_view, dctx)
        ctx.enqueue_function[argmax_dec, argmax_dec](
            argmax_out_t, dec_logits_t, 1, 1,
            grid_dim=1, block_dim=ARGMAX_BLOCK,
        )
        ctx.synchronize()
        with argmax_out_buf.map_to_host() as ao:
            generated.append(Int(ao[0]))
        print("step", step + 1, "→", generated[step + 1], " expected", Int(expected_ids.data[step + 1]))

    for i in range(N_STEPS):
        assert_equal(generated[i], Int(expected_ids.data[i]))
    print("generate fp32 — all", N_STEPS, "tokens match")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
