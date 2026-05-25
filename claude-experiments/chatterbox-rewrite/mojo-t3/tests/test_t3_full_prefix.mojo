"""Full T3 input-prefix integration test.

Chains every validated primitive needed to produce the T3 backbone input:
  - T3CondEnc: speaker_emb → spkr_enc + cond_tokens → speech_emb_table +
               perceiver, emotion_adv → emotion_fc, then concat → cond_emb
  - Text path: text_tokens → text_emb_table + text_pos_emb
  - Speech start: start_id → speech_emb_table + speech_pos_emb[0]
  - Final concat3: cond_emb | text_emb | speech_emb

Output shape (1, 34 + T_text + 1, D=1024). PASS bound: 1e-4 (lots of FP add chain).
"""
from std.math import sqrt
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import load_fp32
from layernorm import layernorm_kernel, linear_kernel
from heads import embed_lookup_kernel, add_pos_emb_kernel
from perceiver import (
    cross_qkt_kernel, cross_softmax_kernel, cross_av_kernel,
    split_heads_kernel, combine_heads_kernel, add_3d_kernel,
)
from concat import concat3_t_kernel


comptime B = 1
comptime SPEAKER_EMB = 256
comptime D = 1024
comptime VOCAB_TEXT = 704
comptime VOCAB_SPEECH = 8194
comptime MAX_TEXT = 2050
comptime MAX_SPEECH = 4098
comptime COND_PROMPT_LEN = 150
comptime T_TEXT = 17
comptime T_COND = 34
comptime T_SPEECH = 1
comptime T_TOTAL = T_COND + T_TEXT + T_SPEECH    # 52
comptime SQ = 32
comptime H = 4
comptime DH = D // H
comptime BLOCK = 128


def upload_f32(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def upload_i64_from_f32(buf: DeviceBuffer[DType.int64], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = Int64(Int(data[i]))


def upload_w(mut ctx: DeviceContext, fix: String, name: String) raises -> DeviceBuffer[DType.float32]:
    var t = load_fp32(fix + name)
    var n = len(t.data)
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    upload_f32(buf, t.data, n)
    return buf^


def attention_block_2[SQ_K: Int, SK_K: Int](
    mut ctx: DeviceContext,
    mut x_q_buf: DeviceBuffer[DType.float32],
    mut x_kv_buf: DeviceBuffer[DType.float32],
    mut norm_w_buf: DeviceBuffer[DType.float32],
    mut norm_b_buf: DeviceBuffer[DType.float32],
    mut to_q_w_buf: DeviceBuffer[DType.float32],
    mut to_q_b_buf: DeviceBuffer[DType.float32],
    mut to_k_w_buf: DeviceBuffer[DType.float32],
    mut to_k_b_buf: DeviceBuffer[DType.float32],
    mut to_v_w_buf: DeviceBuffer[DType.float32],
    mut to_v_b_buf: DeviceBuffer[DType.float32],
    mut proj_w_buf: DeviceBuffer[DType.float32],
    mut proj_b_buf: DeviceBuffer[DType.float32],
    mut out_buf: DeviceBuffer[DType.float32],
) raises:
    comptime xq_layout = row_major[B, SQ_K, D]()
    comptime xkv_layout = row_major[B, SK_K, D]()
    comptime ln_w_layout = row_major[D]()
    comptime q_4d = row_major[B, H, SQ_K, DH]()
    comptime k_4d = row_major[B, H, SK_K, DH]()
    comptime logits_layout = row_major[B, H, SQ_K, SK_K]()
    comptime ln_p_layout = row_major[D, D]()
    comptime ln_pb_layout = row_major[D]()

    var qn = ctx.enqueue_create_buffer[DType.float32](B * SQ_K * D)
    var kn = ctx.enqueue_create_buffer[DType.float32](B * SK_K * D)
    var ql = ctx.enqueue_create_buffer[DType.float32](B * SQ_K * D)
    var kl = ctx.enqueue_create_buffer[DType.float32](B * SK_K * D)
    var vl = ctx.enqueue_create_buffer[DType.float32](B * SK_K * D)
    var qh = ctx.enqueue_create_buffer[DType.float32](B * H * SQ_K * DH)
    var kh = ctx.enqueue_create_buffer[DType.float32](B * H * SK_K * DH)
    var vh = ctx.enqueue_create_buffer[DType.float32](B * H * SK_K * DH)
    var lg = ctx.enqueue_create_buffer[DType.float32](B * H * SQ_K * SK_K)
    var pb = ctx.enqueue_create_buffer[DType.float32](B * H * SQ_K * SK_K)
    var avb = ctx.enqueue_create_buffer[DType.float32](B * H * SQ_K * DH)
    var comb = ctx.enqueue_create_buffer[DType.float32](B * SQ_K * D)
    var proj = ctx.enqueue_create_buffer[DType.float32](B * SQ_K * D)

    var xq_t = TileTensor(x_q_buf, xq_layout)
    var xkv_t = TileTensor(x_kv_buf, xkv_layout)
    var qn_t = TileTensor(qn, xq_layout)
    var kn_t = TileTensor(kn, xkv_layout)
    var nw_t = TileTensor(norm_w_buf, ln_w_layout)
    var nb_t = TileTensor(norm_b_buf, ln_w_layout)
    comptime kln_q = layernorm_kernel[
        DType.float32, type_of(xq_layout), type_of(ln_w_layout), type_of(xq_layout), BLOCK,
    ]
    ctx.enqueue_function[kln_q, kln_q](qn_t, xq_t, nw_t, nb_t, B, SQ_K, D, Float32(1e-5), grid_dim=B*SQ_K, block_dim=BLOCK)
    comptime kln_k = layernorm_kernel[
        DType.float32, type_of(xkv_layout), type_of(ln_w_layout), type_of(xkv_layout), BLOCK,
    ]
    ctx.enqueue_function[kln_k, kln_k](kn_t, xkv_t, nw_t, nb_t, B, SK_K, D, Float32(1e-5), grid_dim=B*SK_K, block_dim=BLOCK)

    var ql_t = TileTensor(ql, xq_layout)
    var kl_t = TileTensor(kl, xkv_layout)
    var vl_t = TileTensor(vl, xkv_layout)
    var qw_t = TileTensor(to_q_w_buf, ln_p_layout); var qb_t = TileTensor(to_q_b_buf, ln_pb_layout)
    var kw_t = TileTensor(to_k_w_buf, ln_p_layout); var kb_t = TileTensor(to_k_b_buf, ln_pb_layout)
    var vw_t = TileTensor(to_v_w_buf, ln_p_layout); var vb_t = TileTensor(to_v_b_buf, ln_pb_layout)
    comptime klq = linear_kernel[
        DType.float32, type_of(xq_layout), type_of(ln_p_layout), type_of(ln_pb_layout),
        type_of(xq_layout), True, BLOCK,
    ]
    ctx.enqueue_function[klq, klq](ql_t, qn_t, qw_t, qb_t, B, SQ_K, D, D, grid_dim=B*SQ_K, block_dim=BLOCK)
    comptime klk = linear_kernel[
        DType.float32, type_of(xkv_layout), type_of(ln_p_layout), type_of(ln_pb_layout),
        type_of(xkv_layout), True, BLOCK,
    ]
    ctx.enqueue_function[klk, klk](kl_t, kn_t, kw_t, kb_t, B, SK_K, D, D, grid_dim=B*SK_K, block_dim=BLOCK)
    ctx.enqueue_function[klk, klk](vl_t, kn_t, vw_t, vb_t, B, SK_K, D, D, grid_dim=B*SK_K, block_dim=BLOCK)

    var qh_t = TileTensor(qh, q_4d)
    var kh_t = TileTensor(kh, k_4d)
    var vh_t = TileTensor(vh, k_4d)
    comptime ksp_q = split_heads_kernel[DType.float32, type_of(xq_layout), type_of(q_4d), DH]
    ctx.enqueue_function[ksp_q, ksp_q](qh_t, ql_t, B, SQ_K, H, grid_dim=B*SQ_K*H, block_dim=DH)
    comptime ksp_k = split_heads_kernel[DType.float32, type_of(xkv_layout), type_of(k_4d), DH]
    ctx.enqueue_function[ksp_k, ksp_k](kh_t, kl_t, B, SK_K, H, grid_dim=B*SK_K*H, block_dim=DH)
    ctx.enqueue_function[ksp_k, ksp_k](vh_t, vl_t, B, SK_K, H, grid_dim=B*SK_K*H, block_dim=DH)

    var lg_t = TileTensor(lg, logits_layout)
    var pb_t = TileTensor(pb, logits_layout)
    comptime scale: Float32 = 1.0 / sqrt(Float32(DH))
    comptime kqkt = cross_qkt_kernel[
        DType.float32, type_of(q_4d), type_of(k_4d), type_of(logits_layout), DH, SK_K,
    ]
    ctx.enqueue_function[kqkt, kqkt](lg_t, qh_t, kh_t, H, SQ_K, scale, grid_dim=B*H*SQ_K, block_dim=SK_K)
    comptime ksm = cross_softmax_kernel[
        DType.float32, type_of(logits_layout), type_of(logits_layout), SK_K, BLOCK,
    ]
    ctx.enqueue_function[ksm, ksm](pb_t, lg_t, H, SQ_K, grid_dim=B*H*SQ_K, block_dim=BLOCK)
    var av_t = TileTensor(avb, q_4d)
    comptime kav = cross_av_kernel[
        DType.float32, type_of(logits_layout), type_of(k_4d), type_of(q_4d), SK_K, DH,
    ]
    ctx.enqueue_function[kav, kav](av_t, pb_t, vh_t, H, SQ_K, grid_dim=B*H*SQ_K, block_dim=DH)
    var comb_t = TileTensor(comb, xq_layout)
    comptime kch = combine_heads_kernel[DType.float32, type_of(q_4d), type_of(xq_layout), DH]
    ctx.enqueue_function[kch, kch](comb_t, av_t, B, SQ_K, H, grid_dim=B*SQ_K*H, block_dim=DH)
    var proj_t = TileTensor(proj, xq_layout)
    var pw_t = TileTensor(proj_w_buf, ln_p_layout); var pbi_t = TileTensor(proj_b_buf, ln_pb_layout)
    comptime klp = linear_kernel[
        DType.float32, type_of(xq_layout), type_of(ln_p_layout), type_of(ln_pb_layout),
        type_of(xq_layout), True, BLOCK,
    ]
    ctx.enqueue_function[klp, klp](proj_t, comb_t, pw_t, pbi_t, B, SQ_K, D, D, grid_dim=B*SQ_K, block_dim=BLOCK)
    var out_t = TileTensor(out_buf, xq_layout)
    comptime kadd = add_3d_kernel[DType.float32, type_of(xq_layout), BLOCK]
    ctx.enqueue_function[kadd, kadd](out_t, xq_t, proj_t, B, SQ_K, D, grid_dim=B*SQ_K, block_dim=BLOCK)


def test_t3_full_prefix() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/t3_full_prefix/"
    var ctx = DeviceContext()

    var speaker_emb_t = load_fp32(fix + "speaker_emb.bin")
    var cond_tokens_t = load_fp32(fix + "cond_speech_tokens.bin")
    var emotion_t = load_fp32(fix + "emotion_adv.bin")
    var text_tokens_t = load_fp32(fix + "text_tokens.bin")
    var start_speech_t = load_fp32(fix + "start_speech_tokens.bin")
    var exp = load_fp32(fix + "embeds.bin")

    # Upload inputs.
    var speaker_buf = ctx.enqueue_create_buffer[DType.float32](B * SPEAKER_EMB)
    upload_f32(speaker_buf, speaker_emb_t.data, B * SPEAKER_EMB)
    var cond_tokens_buf = ctx.enqueue_create_buffer[DType.int64](B * COND_PROMPT_LEN)
    upload_i64_from_f32(cond_tokens_buf, cond_tokens_t.data, B * COND_PROMPT_LEN)
    var emo_buf = ctx.enqueue_create_buffer[DType.float32](B * 1 * 1)
    upload_f32(emo_buf, emotion_t.data, B * 1 * 1)
    var text_tokens_buf = ctx.enqueue_create_buffer[DType.int64](B * T_TEXT)
    upload_i64_from_f32(text_tokens_buf, text_tokens_t.data, B * T_TEXT)
    var start_buf = ctx.enqueue_create_buffer[DType.int64](B * 1)
    upload_i64_from_f32(start_buf, start_speech_t.data, B * 1)

    # Weights.
    var spkr_enc_w = upload_w(ctx, fix, "spkr_enc_w.bin")
    var spkr_enc_b = upload_w(ctx, fix, "spkr_enc_b.bin")
    var emo_fc_w  = upload_w(ctx, fix, "emotion_fc_w.bin")
    var speech_emb_w = upload_w(ctx, fix, "speech_emb_w.bin")
    var text_emb_w = upload_w(ctx, fix, "text_emb_w.bin")
    var text_pos_w = upload_w(ctx, fix, "text_pos_w.bin")
    var speech_pos_w = upload_w(ctx, fix, "speech_pos_w.bin")
    var norm_w = upload_w(ctx, fix, "attn_norm_w.bin")
    var norm_b = upload_w(ctx, fix, "attn_norm_b.bin")
    var to_q_w = upload_w(ctx, fix, "to_q_w.bin"); var to_q_b = upload_w(ctx, fix, "to_q_b.bin")
    var to_k_w = upload_w(ctx, fix, "to_k_w.bin"); var to_k_b = upload_w(ctx, fix, "to_k_b.bin")
    var to_v_w = upload_w(ctx, fix, "to_v_w.bin"); var to_v_b = upload_w(ctx, fix, "to_v_b.bin")
    var proj_w = upload_w(ctx, fix, "proj_out_w.bin"); var proj_b = upload_w(ctx, fix, "proj_out_b.bin")

    # ---- T3CondEnc → cond_emb (B, 34, D) ----
    # spkr_enc.
    var cond_spkr = ctx.enqueue_create_buffer[DType.float32](B * 1 * D)
    comptime spkr_in_layout = row_major[B, 1, SPEAKER_EMB]()
    comptime spkr_out_layout = row_major[B, 1, D]()
    comptime spkr_w_layout = row_major[D, SPEAKER_EMB]()
    comptime ln_w_layout = row_major[D]()
    var spkr_in_buf = ctx.enqueue_create_buffer[DType.float32](B * 1 * SPEAKER_EMB)
    ctx.enqueue_copy(spkr_in_buf, speaker_buf)
    var spkr_in_t = TileTensor(spkr_in_buf, spkr_in_layout)
    var spkr_w_t = TileTensor(spkr_enc_w, spkr_w_layout)
    var spkr_b_t = TileTensor(spkr_enc_b, ln_w_layout)
    var cond_spkr_t = TileTensor(cond_spkr, spkr_out_layout)
    comptime klin_spkr = linear_kernel[
        DType.float32, type_of(spkr_in_layout), type_of(spkr_w_layout),
        type_of(ln_w_layout), type_of(spkr_out_layout), True, BLOCK,
    ]
    ctx.enqueue_function[klin_spkr, klin_spkr](
        cond_spkr_t, spkr_in_t, spkr_w_t, spkr_b_t, B, 1, SPEAKER_EMB, D,
        grid_dim=B * 1, block_dim=BLOCK,
    )

    # cond_speech_emb = speech_emb_table[cond_speech_tokens].
    var cond_speech_emb_buf = ctx.enqueue_create_buffer[DType.float32](B * COND_PROMPT_LEN * D)
    comptime cs_tokens_layout = row_major[B, COND_PROMPT_LEN]()
    comptime sp_table_layout = row_major[VOCAB_SPEECH, D]()
    comptime cs_emb_layout = row_major[B, COND_PROMPT_LEN, D]()
    var cs_tokens_t = TileTensor(cond_tokens_buf, cs_tokens_layout)
    var sp_table_t = TileTensor(speech_emb_w, sp_table_layout)
    var cs_emb_t = TileTensor(cond_speech_emb_buf, cs_emb_layout)
    comptime kemb_cs = embed_lookup_kernel[
        DType.float32, type_of(cs_tokens_layout), type_of(sp_table_layout),
        type_of(cs_emb_layout), D,
    ]
    ctx.enqueue_function[kemb_cs, kemb_cs](
        cs_emb_t, cs_tokens_t, sp_table_t, B, COND_PROMPT_LEN,
        grid_dim=B * COND_PROMPT_LEN, block_dim=D,
    )

    # Perceiver.
    var pq_buf = ctx.enqueue_create_buffer[DType.float32](B * SQ * D)
    upload_f32(pq_buf, load_fp32(fix + "pre_attention_query.bin").data, B * SQ * D)
    var pre_att = ctx.enqueue_create_buffer[DType.float32](B * SQ * D)
    attention_block_2[SQ, COND_PROMPT_LEN](
        ctx, pq_buf, cond_speech_emb_buf,
        norm_w, norm_b, to_q_w, to_q_b, to_k_w, to_k_b, to_v_w, to_v_b,
        proj_w, proj_b, pre_att,
    )
    var pre_att_clone = ctx.enqueue_create_buffer[DType.float32](B * SQ * D)
    ctx.enqueue_copy(pre_att_clone, pre_att)
    var perceiver_out = ctx.enqueue_create_buffer[DType.float32](B * SQ * D)
    attention_block_2[SQ, SQ](
        ctx, pre_att, pre_att_clone,
        norm_w, norm_b, to_q_w, to_q_b, to_k_w, to_k_b, to_v_w, to_v_b,
        proj_w, proj_b, perceiver_out,
    )

    # emotion_fc.
    var cond_emo = ctx.enqueue_create_buffer[DType.float32](B * 1 * D)
    comptime emo_in_layout = row_major[B, 1, 1]()
    comptime emo_w_layout = row_major[D, 1]()
    comptime emo_out_layout = row_major[B, 1, D]()
    var emo_b_buf = ctx.enqueue_create_buffer[DType.float32](D); emo_b_buf.enqueue_fill(0.0)
    var emo_t = TileTensor(emo_buf, emo_in_layout)
    var emo_w_t = TileTensor(emo_fc_w, emo_w_layout)
    var emo_b_t = TileTensor(emo_b_buf, ln_w_layout)
    var cond_emo_t = TileTensor(cond_emo, emo_out_layout)
    comptime klin_emo = linear_kernel[
        DType.float32, type_of(emo_in_layout), type_of(emo_w_layout),
        type_of(ln_w_layout), type_of(emo_out_layout), True, BLOCK,
    ]
    ctx.enqueue_function[klin_emo, klin_emo](
        cond_emo_t, emo_t, emo_w_t, emo_b_t, B, 1, 1, D,
        grid_dim=B * 1, block_dim=BLOCK,
    )

    # cond_emb = concat(spkr, perceiver_out, emo) → (B, 34, D).
    var cond_emb_buf = ctx.enqueue_create_buffer[DType.float32](B * T_COND * D)
    comptime cond_out_layout = row_major[B, T_COND, D]()
    comptime ce_a_layout = row_major[B, 1, D]()
    comptime ce_b_layout = row_major[B, SQ, D]()
    comptime ce_c_layout = row_major[B, 1, D]()
    var ce_a_t = TileTensor(cond_spkr, ce_a_layout)
    var ce_b_t = TileTensor(perceiver_out, ce_b_layout)
    var ce_c_t = TileTensor(cond_emo, ce_c_layout)
    var cond_emb_t = TileTensor(cond_emb_buf, cond_out_layout)
    comptime kcat_cond = concat3_t_kernel[
        DType.float32, type_of(ce_a_layout), type_of(ce_b_layout), type_of(ce_c_layout),
        type_of(cond_out_layout), 1, SQ, 1, BLOCK,
    ]
    ctx.enqueue_function[kcat_cond, kcat_cond](
        cond_emb_t, ce_a_t, ce_b_t, ce_c_t,
        B, D, grid_dim=B * T_COND, block_dim=BLOCK,
    )

    # ---- Text path: text_emb + text_pos_emb ----
    var text_emb_only = ctx.enqueue_create_buffer[DType.float32](B * T_TEXT * D)
    var text_emb_with_pos = ctx.enqueue_create_buffer[DType.float32](B * T_TEXT * D)
    comptime text_tokens_layout = row_major[B, T_TEXT]()
    comptime text_table_layout = row_major[VOCAB_TEXT, D]()
    comptime text_pos_layout = row_major[MAX_TEXT, D]()
    comptime text_emb_layout = row_major[B, T_TEXT, D]()
    var text_tokens_t_tt = TileTensor(text_tokens_buf, text_tokens_layout)
    var text_emb_w_tt = TileTensor(text_emb_w, text_table_layout)
    var text_pos_w_tt = TileTensor(text_pos_w, text_pos_layout)
    var text_emb_only_tt = TileTensor(text_emb_only, text_emb_layout)
    var text_emb_with_pos_tt = TileTensor(text_emb_with_pos, text_emb_layout)
    comptime kemb_text = embed_lookup_kernel[
        DType.float32, type_of(text_tokens_layout), type_of(text_table_layout),
        type_of(text_emb_layout), D,
    ]
    ctx.enqueue_function[kemb_text, kemb_text](
        text_emb_only_tt, text_tokens_t_tt, text_emb_w_tt, B, T_TEXT,
        grid_dim=B * T_TEXT, block_dim=D,
    )
    comptime kpos_text = add_pos_emb_kernel[
        DType.float32, type_of(text_emb_layout), type_of(text_pos_layout),
        type_of(text_emb_layout), D,
    ]
    ctx.enqueue_function[kpos_text, kpos_text](
        text_emb_with_pos_tt, text_emb_only_tt, text_pos_w_tt, B, T_TEXT, 0,
        grid_dim=B * T_TEXT, block_dim=D,
    )

    # ---- Start-speech path: speech_emb[start_id] + speech_pos[0] ----
    var sp_emb_only = ctx.enqueue_create_buffer[DType.float32](B * 1 * D)
    var sp_emb_with_pos = ctx.enqueue_create_buffer[DType.float32](B * 1 * D)
    comptime sp_tokens_layout = row_major[B, 1]()
    comptime sp_pos_layout = row_major[MAX_SPEECH, D]()
    comptime sp_emb_layout = row_major[B, 1, D]()
    var sp_tokens_t = TileTensor(start_buf, sp_tokens_layout)
    var sp_pos_t = TileTensor(speech_pos_w, sp_pos_layout)
    var sp_emb_only_t = TileTensor(sp_emb_only, sp_emb_layout)
    var sp_emb_with_pos_t = TileTensor(sp_emb_with_pos, sp_emb_layout)
    comptime kemb_sp = embed_lookup_kernel[
        DType.float32, type_of(sp_tokens_layout), type_of(sp_table_layout),
        type_of(sp_emb_layout), D,
    ]
    ctx.enqueue_function[kemb_sp, kemb_sp](
        sp_emb_only_t, sp_tokens_t, sp_table_t, B, 1,
        grid_dim=B * 1, block_dim=D,
    )
    comptime kpos_sp = add_pos_emb_kernel[
        DType.float32, type_of(sp_emb_layout), type_of(sp_pos_layout),
        type_of(sp_emb_layout), D,
    ]
    ctx.enqueue_function[kpos_sp, kpos_sp](
        sp_emb_with_pos_t, sp_emb_only_t, sp_pos_t, B, 1, 0,
        grid_dim=B * 1, block_dim=D,
    )

    # ---- Final concat: cond_emb | text_emb | speech_emb → (B, 52, D) ----
    var embeds_buf = ctx.enqueue_create_buffer[DType.float32](B * T_TOTAL * D)
    comptime out_total_layout = row_major[B, T_TOTAL, D]()
    var final_a_t = TileTensor(cond_emb_buf, cond_out_layout)
    var final_b_t = TileTensor(text_emb_with_pos, text_emb_layout)
    var final_c_t = TileTensor(sp_emb_with_pos, sp_emb_layout)
    var embeds_t = TileTensor(embeds_buf, out_total_layout)
    comptime kcat_final = concat3_t_kernel[
        DType.float32, type_of(cond_out_layout), type_of(text_emb_layout),
        type_of(sp_emb_layout), type_of(out_total_layout), T_COND, T_TEXT, T_SPEECH, BLOCK,
    ]
    ctx.enqueue_function[kcat_final, kcat_final](
        embeds_t, final_a_t, final_b_t, final_c_t,
        B, D, grid_dim=B * T_TOTAL, block_dim=BLOCK,
    )
    ctx.synchronize()

    var n_out = B * T_TOTAL * D
    var max_abs: Float32 = 0.0
    with embeds_buf.map_to_host() as h:
        for i in range(n_out):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            if i < 8:
                print("emb[", i, "]: mojo=", h[i], "  torch=", exp.data[i], "  diff=", d)
            if i >= T_COND * D and i < T_COND * D + 8:
                print("text-start[", i - T_COND * D, "]: mojo=", h[i], "  torch=", exp.data[i], "  diff=", d)
    print("T3 full prefix — max abs:", max_abs)
    assert_almost_equal(max_abs, 0.0, atol=1.0e-4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
