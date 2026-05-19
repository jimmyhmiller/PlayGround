"""Pure-Mojo end-to-end TTS: text → audio.

# Voice conditioning is precomputed once per voice (the equivalent of a
# "voice profile"). The files in `weights/s3gen_prompt/` and
# `weights/t3_text_parity/cond_emb.bin` are the cached output of the voice
# encoder pipeline (wav → prompt_token, prompt_feat, embedding, cond_emb).
# They do NOT depend on the text being synthesized — the same files work for
# any input text spoken with that voice.
#
# Text → audio runs entirely in Mojo:
#   1. Mojo BPE tokenize text → text_ids.
#   2. Build text_emb = t3.text_emb[ids] + t3.text_pos_emb[pos] in Mojo.
#   3. Build bos_emb = t3.speech_emb[BOS] + t3.speech_pos_emb[0] in Mojo.
#   4. Build RoPE cos/sin tables in Mojo.
#   5. T3 prefix = concat(cond_emb, text_emb, bos_emb) — cond_emb is the
#      cached voice profile.
#   6. Mojo T3 generate with temperature/top-p sampling → speech_tokens.
#   7. Mojo flow encoder on concat(prompt_token, speech_tokens) → mu.
#   8. Build cond from prompt_feat (voice profile mel slot).
#   9. Mojo gaussian_noise_fill → x; cfm_solve_euler (cosine, CFG) → mel.
#  10. Trim prompt prefix; Mojo HiFT → audio.
#
# The voice profile is built once by `scripts/dump_*_oracle.py` (calls torch
# voice encoder). Future work: port voice encoder (FCM + CAMPPlus + s3tokenizer
# + mel extractor) entirely to Mojo so the user can supply a raw WAV at
# inference time.
"""
from std.sys import has_accelerator
from std.math import sin, cos as mcos, pi, sqrt
from std.testing import TestSuite, assert_true
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from fixture import load_fp32, load_i64, save_wav
from weights import (
    load_t3, load_t3_cond_enc, load_upsample_conformer_encoder, load_cfm_estimator_real,
    load_hift_generator, upload_fp32,
)
from cond_enc import t3_cond_enc_forward
from t3_generate import t3_generate_cfg, t3_generate_cfg_sample
from bpe_tokenizer import load_tokenizer
from text_embed import text_to_input_ids, build_text_emb, build_bos_emb, build_rope_tables
from upsample_encoder import upsample_conformer_forward
from cfm_estimator_new import cfm_solve_euler, gaussian_noise_fill
from hift_generator import (
    hift_decode_trunk, istft_forward, hann_window_periodic_fill,
    f0_predictor_forward, f0_upsample_nearest, source_module_forward,
    build_s_stft_from_signal,
)


comptime B = 1
comptime MEL = 80
comptime T_PROMPT_TOKEN = 250
comptime T_PROMPT_MEL = 500
comptime N_FFT = 16
comptime HOP = 4
comptime N_OUT = 18
comptime N_CFM_STEPS = 10
comptime CFG: Float32 = 0.7
comptime T3_CFG: Float32 = 0.5
comptime D = 1024
comptime MAX_CTX = 600
comptime MAX_NEW = 200
comptime EOS = 6562
comptime T_COND = 34
comptime T_BOS = 1


def test_end_to_end() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var voice_dir = "weights/s3gen_prompt/"        # voice cloning inputs
    var text_dir = "weights/t3_text_parity/"       # text conditioning inputs

    print("[e2e] loading T3 + flow encoder + CFM + HiFT...")
    var t3 = load_t3(ctx, "weights/t3")
    var enc = load_upsample_conformer_encoder(ctx, "weights/s3gen/flow")
    var cfm = load_cfm_estimator_real(ctx, "weights/s3gen/flow/decoder/estimator")
    var hift = load_hift_generator(ctx, "weights/s3gen/mel2wav")

    # ─────────────────────────────────────────────────────────────────────
    # Step 1: Build T3 CFG-doubled prefix from voice/text conditioning.
    # ─────────────────────────────────────────────────────────────────────
    # Build cond_emb in Mojo from speaker_emb + cond_prompt_tokens + emotion.
    # These voice inputs come from the cached voice profile (computed once per ref voice).
    var voice_dir2 = "weights/s3gen_prompt/cond_enc_diag/"
    var cond_enc_mod = load_t3_cond_enc(ctx, "weights/t3", t3.speech_emb, t3.speech_pos_emb)
    var spk_emb = upload_fp32(ctx, voice_dir2 + "speaker_emb.bin")
    var emotion = upload_fp32(ctx, voice_dir2 + "emotion_adv.bin")
    var ctok = load_i64(voice_dir2 + "cond_prompt_speech_tokens.bin")
    var CL = 150
    var SQ = 32
    var ctok_buf = ctx.enqueue_create_buffer[DType.int64](B * CL)
    with ctok_buf.map_to_host() as h:
        for i in range(B * CL):
            h[i] = ctok.data[i]
    var mask_q = ctx.enqueue_create_buffer[DType.float32](SQ * SQ)
    mask_q.enqueue_fill(0.0)
    var mask_qq = ctx.enqueue_create_buffer[DType.float32](SQ * CL)
    mask_qq.enqueue_fill(0.0)
    var cond_emb = ctx.enqueue_create_buffer[DType.float32](B * 34 * D)
    print("[e2e] building cond_emb via T3CondEnc...")
    t3_cond_enc_forward(ctx, cond_enc_mod, spk_emb, ctok_buf, emotion, cond_emb,
                         mask_q, mask_qq, B)
    ctx.synchronize()

    # Build text_emb + bos_emb in Mojo from raw text.
    var text = "the quick brown fox"
    var tok_dir = "../mojo-t3/tests/fixtures/tokenizer/"
    var tok = load_tokenizer(tok_dir + "vocab.txt", tok_dir + "merges.txt")
    var ids = text_to_input_ids(text, tok)
    var T_TEXT = len(ids)
    var T_PREFIX = T_COND + T_TEXT + T_BOS
    print("[e2e] tokenized '", text, "' →", T_TEXT, "ids; T_PREFIX=", T_PREFIX)

    var text_emb = ctx.enqueue_create_buffer[DType.float32](B * T_TEXT * D)
    build_text_emb(ctx, t3, ids, text_emb)
    var bos_emb = ctx.enqueue_create_buffer[DType.float32](B * 1 * D)
    build_bos_emb(ctx, t3, bos_emb)

    var B2 = 2 * B
    var prefix = ctx.enqueue_create_buffer[DType.float32](B2 * T_PREFIX * D)
    var ce = cond_emb.unsafe_ptr()
    var te = text_emb.unsafe_ptr()
    var be = bos_emb.unsafe_ptr()
    var px = prefix.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(ce, te, be, px, B2, T_TEXT, T_PREFIX)
    def cat_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (T_PREFIX * D)
        var rem = i - bi * T_PREFIX * D
        var ti = rem // D
        var di = rem - ti * D
        if ti < T_COND:
            px[i] = ce[(bi % B) * T_COND * D + ti * D + di]
        elif ti < T_COND + T_TEXT:
            var src_t = ti - T_COND
            if bi < B:
                px[i] = te[(bi % B) * T_TEXT * D + src_t * D + di]
            else:
                px[i] = 0.0
        else:
            var src_t = ti - T_COND - T_TEXT
            px[i] = be[(bi % B) * T_BOS * D + src_t * D + di]
    elementwise[cat_func, simd_width=1, target="gpu"](
        IndexList[1](B2 * T_PREFIX * D), DeviceContextPtr(ctx),
    )

    var cos_full = ctx.enqueue_create_buffer[DType.float32](MAX_CTX * 64)
    var sin_full = ctx.enqueue_create_buffer[DType.float32](MAX_CTX * 64)
    build_rope_tables(ctx, MAX_CTX, 64, cos_full, sin_full)
    var t3_mask = ctx.enqueue_create_buffer[DType.float32](T_PREFIX * T_PREFIX)
    with t3_mask.map_to_host() as h:
        for r in range(T_PREFIX):
            for c in range(T_PREFIX):
                if c > r:
                    h[r * T_PREFIX + c] = -1.0e30
                else:
                    h[r * T_PREFIX + c] = 0.0
    # speech_pos_emb table is loaded as part of T3.
    var speech_pos = t3.speech_pos_emb.table

    print("[e2e] Mojo T3 generate w/ top-p sampling (max_new=", MAX_NEW, " CFG=", T3_CFG, ")...")
    var generated = t3_generate_cfg_sample(
        ctx, t3, prefix, cos_full, sin_full, t3_mask, speech_pos,
        B, T_PREFIX, MAX_CTX, MAX_NEW,
        speech_pos_offset=1, eos_token=EOS, cfg_weight=T3_CFG,
        temperature=Float32(0.8), top_p=Float32(0.95), rep_penalty=Float32(1.2),
        rng_seed=UInt64(0xDEADBEEF),
    )
    ctx.synchronize()
    print("[e2e] T3 generated", len(generated), "speech tokens")
    print("[e2e] first 30 tokens:")
    for i in range(min(30, len(generated))):
        print("  [", i, "] =", Int(generated[i]))

    var speech_tokens = List[Int64]()
    for i in range(len(generated)):
        var tok = Int(generated[i])
        if tok == EOS: break
        if tok < 6561: speech_tokens.append(generated[i])
    if len(speech_tokens) == 0:
        speech_tokens.append(Int64(0))
    var T_GEN_TOKEN = len(speech_tokens)
    var T_TOTAL_TOKEN = T_PROMPT_TOKEN + T_GEN_TOKEN
    var T_TOTAL_MEL = 2 * T_TOTAL_TOKEN
    var T_OUT_MEL = T_TOTAL_MEL - T_PROMPT_MEL
    print("[e2e] T_GEN_TOKEN=", T_GEN_TOKEN, " T_TOTAL_TOKEN=", T_TOTAL_TOKEN,
          " T_TOTAL_MEL=", T_TOTAL_MEL, " T_OUT_MEL=", T_OUT_MEL)

    # ─────────────────────────────────────────────────────────────────────
    # Step 2: Build tok_buf = cat(prompt_token, speech_tokens).
    # ─────────────────────────────────────────────────────────────────────
    var prompt_tok = load_i64(voice_dir + "prompt_token.bin")
    var tok_buf = ctx.enqueue_create_buffer[DType.int64](B * T_TOTAL_TOKEN)
    with tok_buf.map_to_host() as h:
        for i in range(T_PROMPT_TOKEN):
            h[i] = prompt_tok.data[i]
        for i in range(T_GEN_TOKEN):
            h[T_PROMPT_TOKEN + i] = speech_tokens[i]

    # ─────────────────────────────────────────────────────────────────────
    # Step 3: Flow encoder → mu.
    # ─────────────────────────────────────────────────────────────────────
    var mu = ctx.enqueue_create_buffer[DType.float32](B * MEL * T_TOTAL_MEL)
    print("[e2e] flow encoder (T_token=", T_TOTAL_TOKEN, " → T_mel=", T_TOTAL_MEL, ")...")
    upsample_conformer_forward(ctx, enc, tok_buf, mu, B, T_TOTAL_TOKEN)
    ctx.synchronize()

    # ─────────────────────────────────────────────────────────────────────
    # Step 4: cond = zeros, slot prompt_feat (B, T_PROMPT_MEL, 80) into [:, :, :T_PROMPT_MEL].
    # ─────────────────────────────────────────────────────────────────────
    var prompt_feat = upload_fp32(ctx, voice_dir + "prompt_feat.bin")
    var cond = ctx.enqueue_create_buffer[DType.float32](B * MEL * T_TOTAL_MEL)
    cond.enqueue_fill(0.0)
    var pf_ptr = prompt_feat.unsafe_ptr()
    var cond_ptr = cond.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(pf_ptr, cond_ptr, T_TOTAL_MEL)
    def cond_fill[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (T_PROMPT_MEL * MEL)
        var rem = i - bi * T_PROMPT_MEL * MEL
        var ti = rem // MEL
        var ci = rem - ti * MEL
        cond_ptr[bi * MEL * T_TOTAL_MEL + ci * T_TOTAL_MEL + ti] = pf_ptr[i]
    elementwise[cond_fill, simd_width=1, target="gpu"](
        IndexList[1](B * T_PROMPT_MEL * MEL), DeviceContextPtr(ctx),
    )

    # ─────────────────────────────────────────────────────────────────────
    # Step 5: spks + Mojo Gaussian noise.
    # ─────────────────────────────────────────────────────────────────────
    var spks = upload_fp32(ctx, voice_dir + "embedding_normed_affine.bin")
    var x = ctx.enqueue_create_buffer[DType.float32](B * MEL * T_TOTAL_MEL)
    gaussian_noise_fill(ctx, x, B * MEL * T_TOTAL_MEL, UInt64(0xC0FFEE), Float32(1.0))

    var cfm_mask = ctx.enqueue_create_buffer[DType.float32](B * T_TOTAL_MEL)
    cfm_mask.enqueue_fill(1.0)

    print("[e2e] CFM Euler (", N_CFM_STEPS, " steps, CFG=", CFG, ", cosine schedule)...")
    cfm_solve_euler(ctx, cfm, x, mu, spks, cond, cfm_mask, B, T_TOTAL_MEL, N_CFM_STEPS, CFG)
    ctx.synchronize()

    # ─────────────────────────────────────────────────────────────────────
    # Step 6: Trim prompt prefix.
    # ─────────────────────────────────────────────────────────────────────
    var mel_out = ctx.enqueue_create_buffer[DType.float32](B * MEL * T_OUT_MEL)
    var x_ptr = x.unsafe_ptr()
    var mo_ptr = mel_out.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(x_ptr, mo_ptr, T_TOTAL_MEL, T_OUT_MEL)
    def trim[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (MEL * T_OUT_MEL)
        var rem = i - bi * MEL * T_OUT_MEL
        var ci = rem // T_OUT_MEL
        var ti = rem - ci * T_OUT_MEL
        mo_ptr[i] = x_ptr[bi * MEL * T_TOTAL_MEL + ci * T_TOTAL_MEL + (T_PROMPT_MEL + ti)]
    elementwise[trim, simd_width=1, target="gpu"](
        IndexList[1](B * MEL * T_OUT_MEL), DeviceContextPtr(ctx),
    )
    ctx.synchronize()

    var mel_max: Float32 = -1e30
    var mel_min: Float32 = 1e30
    with mel_out.map_to_host() as h:
        for i in range(B * MEL * T_OUT_MEL):
            if h[i] > mel_max: mel_max = h[i]
            if h[i] < mel_min: mel_min = h[i]
    print("[e2e] mel stats: min=", mel_min, " max=", mel_max)

    # ─────────────────────────────────────────────────────────────────────
    # Step 7: HiFT — pure Mojo with LCG source module.
    # ─────────────────────────────────────────────────────────────────────
    var T_HIFT = T_OUT_MEL * 120 + 1
    var T_AUDIO = (T_HIFT - 1) * HOP
    var T_AUDIO_FULL = T_OUT_MEL * 480
    var T_S_FRAMES = T_AUDIO_FULL // HOP + 1

    var f0 = ctx.enqueue_create_buffer[DType.float32](B * T_OUT_MEL)
    f0_predictor_forward(ctx, hift.f0_predictor, mel_out, f0, B, T_OUT_MEL)
    var f0_up = ctx.enqueue_create_buffer[DType.float32](B * T_AUDIO_FULL)
    f0_upsample_nearest(ctx, f0, f0_up, B, T_OUT_MEL, 480)

    print("[e2e] source module (Mojo LCG RNG)...")
    var sine_merge = ctx.enqueue_create_buffer[DType.float32](B * 1 * T_AUDIO_FULL)
    source_module_forward(ctx, hift.m_source, f0_up, sine_merge,
                           B, T_AUDIO_FULL,
                           sampling_rate=24000, harmonic_num=8,
                           sine_amp=Float32(0.1), noise_std=Float32(0.003),
                           voiced_threshold=Float32(10.0))

    var window_s = ctx.enqueue_create_buffer[DType.float32](N_FFT)
    hann_window_periodic_fill(ctx, window_s, N_FFT)
    var s_stft = ctx.enqueue_create_buffer[DType.float32](B * N_OUT * T_S_FRAMES)
    build_s_stft_from_signal(ctx, sine_merge, window_s, s_stft,
                              B, T_AUDIO_FULL, N_FFT, HOP, T_S_FRAMES)

    var spec = ctx.enqueue_create_buffer[DType.float32](B * N_OUT * T_HIFT)
    print("[e2e] HiFT trunk (T_mel=", T_OUT_MEL, " → T_frames=", T_HIFT, ")...")
    hift_decode_trunk(ctx, hift, mel_out, s_stft, spec,
                      B, T_OUT_MEL, T_S_FRAMES, T_HIFT, use_source=True)

    var window_i = ctx.enqueue_create_buffer[DType.float32](N_FFT)
    hann_window_periodic_fill(ctx, window_i, N_FFT)
    var audio = ctx.enqueue_create_buffer[DType.float32](B * T_AUDIO)
    print("[e2e] iSTFT...")
    istft_forward(ctx, spec, window_i, audio, B, N_FFT, T_HIFT, T_AUDIO)
    ctx.synchronize()

    var n_nan = 0
    var max_a: Float32 = 0.0
    with audio.map_to_host() as h:
        for i in range(B * T_AUDIO):
            var v = h[i]
            if v != v: n_nan += 1
            var av = v
            if av < 0.0: av = -av
            if av > max_a: max_a = av
    print("[e2e] audio max=", max_a, " nan=", n_nan, " duration ~", Float32(T_AUDIO) / 24000.0, "s")
    assert_true(n_nan == 0, "no NaNs")

    var samples = List[Float32]()
    with audio.map_to_host() as h:
        for i in range(T_AUDIO):
            samples.append(h[i])
    save_wav(String("max_impl_end_to_end.wav"), samples, 24000)
    print("[e2e] saved max_impl_end_to_end.wav")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
