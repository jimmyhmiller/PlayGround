"""T3CondEnc parity test: chains Linear + Embedding + Perceiver + Linear + Concat
to produce the cond_emb prefix that T3 generation consumes."""
from std.math import sqrt
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import load_fp32
from layernorm import layernorm_kernel, linear_kernel
from heads import embed_lookup_kernel
from perceiver import (
    cross_qkt_kernel, cross_softmax_kernel, cross_av_kernel,
    split_heads_kernel, combine_heads_kernel, add_3d_kernel,
)
from concat import concat3_t_kernel


comptime B = 1
comptime SPEAKER_EMB = 256
comptime N_CHANNELS = 1024
comptime COND_PROMPT_LEN = 150
comptime SPEECH_VOCAB = 8194
comptime SQ = 32   # perceiver queries
comptime D = 1024
comptime H = 4
comptime DH = D // H
comptime BLOCK = 128


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def upload_i64(buf: DeviceBuffer[DType.int64], data: List[Float32], n: Int) raises:
    """Convert float-stored ids to int64 (we saved them via .float() in dump)."""
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = Int64(Int(data[i]))


def upload_w(mut ctx: DeviceContext, fix: String, name: String) raises -> DeviceBuffer[DType.float32]:
    var t = load_fp32(fix + name)
    var n = len(t.data)
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    upload(buf, t.data, n)
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
    """Inline AttentionBlock2 (same as in test_perceiver.mojo)."""
    comptime xq_layout = row_major[B, SQ_K, D]()
    comptime xkv_layout = row_major[B, SK_K, D]()
    comptime ln_w_layout = row_major[D]()
    comptime q_4d = row_major[B, H, SQ_K, DH]()
    comptime k_4d = row_major[B, H, SK_K, DH]()
    comptime logits_layout = row_major[B, H, SQ_K, SK_K]()
    comptime ln_p_layout = row_major[D, D]()
    comptime ln_pb_layout = row_major[D]()

    var qn_buf = ctx.enqueue_create_buffer[DType.float32](B * SQ_K * D)
    var kn_buf = ctx.enqueue_create_buffer[DType.float32](B * SK_K * D)
    var q_lin_buf = ctx.enqueue_create_buffer[DType.float32](B * SQ_K * D)
    var k_lin_buf = ctx.enqueue_create_buffer[DType.float32](B * SK_K * D)
    var v_lin_buf = ctx.enqueue_create_buffer[DType.float32](B * SK_K * D)
    var q_h_buf = ctx.enqueue_create_buffer[DType.float32](B * H * SQ_K * DH)
    var k_h_buf = ctx.enqueue_create_buffer[DType.float32](B * H * SK_K * DH)
    var v_h_buf = ctx.enqueue_create_buffer[DType.float32](B * H * SK_K * DH)
    var logits_buf = ctx.enqueue_create_buffer[DType.float32](B * H * SQ_K * SK_K)
    var probs_buf = ctx.enqueue_create_buffer[DType.float32](B * H * SQ_K * SK_K)
    var av_buf = ctx.enqueue_create_buffer[DType.float32](B * H * SQ_K * DH)
    var comb_buf = ctx.enqueue_create_buffer[DType.float32](B * SQ_K * D)
    var proj_buf = ctx.enqueue_create_buffer[DType.float32](B * SQ_K * D)

    var x_q_t = TileTensor(x_q_buf, xq_layout)
    var x_kv_t = TileTensor(x_kv_buf, xkv_layout)
    var qn_t = TileTensor(qn_buf, xq_layout)
    var kn_t = TileTensor(kn_buf, xkv_layout)
    var norm_w_t = TileTensor(norm_w_buf, ln_w_layout)
    var norm_b_t = TileTensor(norm_b_buf, ln_w_layout)

    comptime kln_q = layernorm_kernel[
        DType.float32, type_of(xq_layout), type_of(ln_w_layout), type_of(xq_layout), BLOCK,
    ]
    ctx.enqueue_function[kln_q, kln_q](
        qn_t, x_q_t, norm_w_t, norm_b_t, B, SQ_K, D, Float32(1.0e-5),
        grid_dim=B * SQ_K, block_dim=BLOCK,
    )
    comptime kln_k = layernorm_kernel[
        DType.float32, type_of(xkv_layout), type_of(ln_w_layout), type_of(xkv_layout), BLOCK,
    ]
    ctx.enqueue_function[kln_k, kln_k](
        kn_t, x_kv_t, norm_w_t, norm_b_t, B, SK_K, D, Float32(1.0e-5),
        grid_dim=B * SK_K, block_dim=BLOCK,
    )

    var q_lin_t = TileTensor(q_lin_buf, xq_layout)
    var k_lin_t = TileTensor(k_lin_buf, xkv_layout)
    var v_lin_t = TileTensor(v_lin_buf, xkv_layout)
    var to_q_w_t = TileTensor(to_q_w_buf, ln_p_layout)
    var to_q_b_t = TileTensor(to_q_b_buf, ln_pb_layout)
    var to_k_w_t = TileTensor(to_k_w_buf, ln_p_layout)
    var to_k_b_t = TileTensor(to_k_b_buf, ln_pb_layout)
    var to_v_w_t = TileTensor(to_v_w_buf, ln_p_layout)
    var to_v_b_t = TileTensor(to_v_b_buf, ln_pb_layout)

    comptime klq = linear_kernel[
        DType.float32, type_of(xq_layout), type_of(ln_p_layout),
        type_of(ln_pb_layout), type_of(xq_layout), True, BLOCK,
    ]
    ctx.enqueue_function[klq, klq](q_lin_t, qn_t, to_q_w_t, to_q_b_t, B, SQ_K, D, D, grid_dim=B*SQ_K, block_dim=BLOCK)
    comptime klk = linear_kernel[
        DType.float32, type_of(xkv_layout), type_of(ln_p_layout),
        type_of(ln_pb_layout), type_of(xkv_layout), True, BLOCK,
    ]
    ctx.enqueue_function[klk, klk](k_lin_t, kn_t, to_k_w_t, to_k_b_t, B, SK_K, D, D, grid_dim=B*SK_K, block_dim=BLOCK)
    ctx.enqueue_function[klk, klk](v_lin_t, kn_t, to_v_w_t, to_v_b_t, B, SK_K, D, D, grid_dim=B*SK_K, block_dim=BLOCK)

    var q_h_t = TileTensor(q_h_buf, q_4d)
    var k_h_t = TileTensor(k_h_buf, k_4d)
    var v_h_t = TileTensor(v_h_buf, k_4d)
    comptime ksp_q = split_heads_kernel[
        DType.float32, type_of(xq_layout), type_of(q_4d), DH,
    ]
    ctx.enqueue_function[ksp_q, ksp_q](q_h_t, q_lin_t, B, SQ_K, H, grid_dim=B*SQ_K*H, block_dim=DH)
    comptime ksp_k = split_heads_kernel[
        DType.float32, type_of(xkv_layout), type_of(k_4d), DH,
    ]
    ctx.enqueue_function[ksp_k, ksp_k](k_h_t, k_lin_t, B, SK_K, H, grid_dim=B*SK_K*H, block_dim=DH)
    ctx.enqueue_function[ksp_k, ksp_k](v_h_t, v_lin_t, B, SK_K, H, grid_dim=B*SK_K*H, block_dim=DH)

    var logits_t = TileTensor(logits_buf, logits_layout)
    var probs_t = TileTensor(probs_buf, logits_layout)
    comptime scale: Float32 = 1.0 / sqrt(Float32(DH))
    comptime kqkt = cross_qkt_kernel[
        DType.float32, type_of(q_4d), type_of(k_4d),
        type_of(logits_layout), DH, SK_K,
    ]
    ctx.enqueue_function[kqkt, kqkt](logits_t, q_h_t, k_h_t, H, SQ_K, scale, grid_dim=B*H*SQ_K, block_dim=SK_K)
    comptime ksm = cross_softmax_kernel[
        DType.float32, type_of(logits_layout), type_of(logits_layout), SK_K, BLOCK,
    ]
    ctx.enqueue_function[ksm, ksm](probs_t, logits_t, H, SQ_K, grid_dim=B*H*SQ_K, block_dim=BLOCK)
    var av_t = TileTensor(av_buf, q_4d)
    comptime kav = cross_av_kernel[
        DType.float32, type_of(logits_layout), type_of(k_4d),
        type_of(q_4d), SK_K, DH,
    ]
    ctx.enqueue_function[kav, kav](av_t, probs_t, v_h_t, H, SQ_K, grid_dim=B*H*SQ_K, block_dim=DH)
    var comb_t = TileTensor(comb_buf, xq_layout)
    comptime kch = combine_heads_kernel[
        DType.float32, type_of(q_4d), type_of(xq_layout), DH,
    ]
    ctx.enqueue_function[kch, kch](comb_t, av_t, B, SQ_K, H, grid_dim=B*SQ_K*H, block_dim=DH)
    var proj_t = TileTensor(proj_buf, xq_layout)
    var proj_w_t = TileTensor(proj_w_buf, ln_p_layout)
    var proj_b_t = TileTensor(proj_b_buf, ln_pb_layout)
    comptime klp = linear_kernel[
        DType.float32, type_of(xq_layout), type_of(ln_p_layout),
        type_of(ln_pb_layout), type_of(xq_layout), True, BLOCK,
    ]
    ctx.enqueue_function[klp, klp](proj_t, comb_t, proj_w_t, proj_b_t, B, SQ_K, D, D, grid_dim=B*SQ_K, block_dim=BLOCK)
    var out_t = TileTensor(out_buf, xq_layout)
    comptime kadd = add_3d_kernel[DType.float32, type_of(xq_layout), BLOCK]
    ctx.enqueue_function[kadd, kadd](out_t, x_q_t, proj_t, B, SQ_K, D, grid_dim=B*SQ_K, block_dim=BLOCK)


def test_t3_cond_enc() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/t3_cond_enc/"
    var ctx = DeviceContext()

    var speaker_emb_t = load_fp32(fix + "speaker_emb.bin")
    var cond_tokens_t = load_fp32(fix + "cond_speech_tokens.bin")
    var emotion_t = load_fp32(fix + "emotion_adv.bin")
    var exp = load_fp32(fix + "cond_emb.bin")

    # Upload inputs.
    var speaker_buf = ctx.enqueue_create_buffer[DType.float32](B * SPEAKER_EMB)
    upload(speaker_buf, speaker_emb_t.data, B * SPEAKER_EMB)
    var tokens_buf = ctx.enqueue_create_buffer[DType.int64](B * COND_PROMPT_LEN)
    upload_i64(tokens_buf, cond_tokens_t.data, B * COND_PROMPT_LEN)
    var emo_buf = ctx.enqueue_create_buffer[DType.float32](B * 1 * 1)
    upload(emo_buf, emotion_t.data, B * 1 * 1)

    # Weights.
    var spkr_enc_w = upload_w(ctx, fix, "spkr_enc_w.bin")
    var spkr_enc_b = upload_w(ctx, fix, "spkr_enc_b.bin")
    var emo_fc_w  = upload_w(ctx, fix, "emotion_fc_w.bin")
    var speech_emb_w = upload_w(ctx, fix, "speech_emb_w.bin")
    var pq = upload_w(ctx, fix, "pre_attention_query.bin")
    var norm_w = upload_w(ctx, fix, "attn_norm_w.bin")
    var norm_b = upload_w(ctx, fix, "attn_norm_b.bin")
    var to_q_w = upload_w(ctx, fix, "to_q_w.bin"); var to_q_b = upload_w(ctx, fix, "to_q_b.bin")
    var to_k_w = upload_w(ctx, fix, "to_k_w.bin"); var to_k_b = upload_w(ctx, fix, "to_k_b.bin")
    var to_v_w = upload_w(ctx, fix, "to_v_w.bin"); var to_v_b = upload_w(ctx, fix, "to_v_b.bin")
    var proj_w = upload_w(ctx, fix, "proj_out_w.bin"); var proj_b = upload_w(ctx, fix, "proj_out_b.bin")

    # Step 1: cond_spkr = spkr_enc(speaker_emb)[:, None] → (B, 1, D).
    var cond_spkr = ctx.enqueue_create_buffer[DType.float32](B * 1 * D)
    comptime spkr_in_layout = row_major[B, 1, SPEAKER_EMB]()
    comptime spkr_out_layout = row_major[B, 1, D]()
    comptime spkr_w_layout = row_major[D, SPEAKER_EMB]()
    comptime spkr_b_layout = row_major[D]()
    # Reshape speaker_emb (B, 256) → (B, 1, 256).
    var spkr_in_buf = ctx.enqueue_create_buffer[DType.float32](B * 1 * SPEAKER_EMB)
    ctx.enqueue_copy(spkr_in_buf, speaker_buf)
    var spkr_in_t = TileTensor(spkr_in_buf, spkr_in_layout)
    var spkr_w_t = TileTensor(spkr_enc_w, spkr_w_layout)
    var spkr_b_t = TileTensor(spkr_enc_b, spkr_b_layout)
    var cond_spkr_t = TileTensor(cond_spkr, spkr_out_layout)
    comptime klin_spkr = linear_kernel[
        DType.float32, type_of(spkr_in_layout), type_of(spkr_w_layout),
        type_of(spkr_b_layout), type_of(spkr_out_layout), True, BLOCK,
    ]
    ctx.enqueue_function[klin_spkr, klin_spkr](
        cond_spkr_t, spkr_in_t, spkr_w_t, spkr_b_t, B, 1, SPEAKER_EMB, D,
        grid_dim=B * 1, block_dim=BLOCK,
    )

    # Step 2: cond_speech_emb = speech_emb_table[cond_speech_tokens] → (B, 150, D).
    var speech_emb_buf = ctx.enqueue_create_buffer[DType.float32](B * COND_PROMPT_LEN * D)
    comptime emb_table_layout = row_major[SPEECH_VOCAB, D]()
    comptime tokens_layout = row_major[B, COND_PROMPT_LEN]()
    comptime emb_out_layout = row_major[B, COND_PROMPT_LEN, D]()
    var speech_emb_w_t = TileTensor(speech_emb_w, emb_table_layout)
    var tokens_tt = TileTensor(tokens_buf, tokens_layout)
    var speech_emb_tt = TileTensor(speech_emb_buf, emb_out_layout)
    comptime kemb = embed_lookup_kernel[
        DType.float32, type_of(tokens_layout), type_of(emb_table_layout),
        type_of(emb_out_layout), D,
    ]
    ctx.enqueue_function[kemb, kemb](
        speech_emb_tt, tokens_tt, speech_emb_w_t, B, COND_PROMPT_LEN,
        grid_dim=B * COND_PROMPT_LEN, block_dim=D,
    )

    # Step 3: Perceiver on cond_speech_emb. Cross-attn (32q × 150k) then self-attn.
    var pq_buf = ctx.enqueue_create_buffer[DType.float32](B * SQ * D)
    upload(pq_buf, load_fp32(fix + "pre_attention_query.bin").data, B * SQ * D)
    var pre_att_buf = ctx.enqueue_create_buffer[DType.float32](B * SQ * D)
    attention_block_2[SQ, COND_PROMPT_LEN](
        ctx, pq_buf, speech_emb_buf,
        norm_w, norm_b, to_q_w, to_q_b, to_k_w, to_k_b, to_v_w, to_v_b,
        proj_w, proj_b, pre_att_buf,
    )
    var pre_att_clone = ctx.enqueue_create_buffer[DType.float32](B * SQ * D)
    ctx.enqueue_copy(pre_att_clone, pre_att_buf)
    var perceiver_out = ctx.enqueue_create_buffer[DType.float32](B * SQ * D)
    attention_block_2[SQ, SQ](
        ctx, pre_att_buf, pre_att_clone,
        norm_w, norm_b, to_q_w, to_q_b, to_k_w, to_k_b, to_v_w, to_v_b,
        proj_w, proj_b, perceiver_out,
    )

    # Step 4: emotion_fc(emotion_adv.view(-1, 1, 1)) → (1, 1, D).
    var cond_emo = ctx.enqueue_create_buffer[DType.float32](B * 1 * D)
    comptime emo_in_layout = row_major[B, 1, 1]()
    comptime emo_w_layout = row_major[D, 1]()
    comptime emo_b_layout = row_major[D]()
    comptime emo_out_layout = row_major[B, 1, D]()
    # zero bias for emotion (no bias in upstream).
    var emo_b_buf = ctx.enqueue_create_buffer[DType.float32](D)
    emo_b_buf.enqueue_fill(0.0)
    var emo_t = TileTensor(emo_buf, emo_in_layout)
    var emo_w_t = TileTensor(emo_fc_w, emo_w_layout)
    var emo_b_t = TileTensor(emo_b_buf, emo_b_layout)
    var cond_emo_t = TileTensor(cond_emo, emo_out_layout)
    comptime klin_emo = linear_kernel[
        DType.float32, type_of(emo_in_layout), type_of(emo_w_layout),
        type_of(emo_b_layout), type_of(emo_out_layout), True, BLOCK,
    ]
    ctx.enqueue_function[klin_emo, klin_emo](
        cond_emo_t, emo_t, emo_w_t, emo_b_t, B, 1, 1, D,
        grid_dim=B * 1, block_dim=BLOCK,
    )

    # Step 5: Concat speaker (T_a=1) | perceiver_out (T_b=32) | emotion (T_c=1) → (B, 34, D).
    var cond_emb_out = ctx.enqueue_create_buffer[DType.float32](B * (1 + SQ + 1) * D)
    comptime out_layout = row_major[B, 1 + SQ + 1, D]()
    comptime a_layout = row_major[B, 1, D]()
    comptime b_layout = row_major[B, SQ, D]()
    comptime c_layout = row_major[B, 1, D]()
    var spkr_in_concat_t = TileTensor(cond_spkr, a_layout)
    var perc_in_concat_t = TileTensor(perceiver_out, b_layout)
    var emo_in_concat_t = TileTensor(cond_emo, c_layout)
    var cond_emb_out_t = TileTensor(cond_emb_out, out_layout)
    comptime kcat = concat3_t_kernel[
        DType.float32, type_of(a_layout), type_of(b_layout), type_of(c_layout),
        type_of(out_layout), 1, SQ, 1, BLOCK,
    ]
    ctx.enqueue_function[kcat, kcat](
        cond_emb_out_t, spkr_in_concat_t, perc_in_concat_t, emo_in_concat_t,
        B, D, grid_dim=B * (1 + SQ + 1), block_dim=BLOCK,
    )
    ctx.synchronize()

    var n_out = B * (1 + SQ + 1) * D
    var max_abs: Float32 = 0.0
    with cond_emb_out.map_to_host() as h:
        for i in range(n_out):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            if i < 8:
                print("ce[", i, "]: mojo=", h[i], "  torch=", exp.data[i], "  diff=", d)
    print("T3CondEnc — max abs:", max_abs)
    assert_almost_equal(max_abs, 0.0, atol=1.0e-3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
