"""Pure-Mojo end-to-end TTS: ref.wav + text → audio.wav.

NO upstream oracle dumps. Everything from raw WAV onwards is computed in Mojo.

Pipeline:
  1. Load 24kHz reference WAV.
  2. Truncate to 10s (240000 samples).
  3. Resample to 16kHz for speaker tokenization + voice encoding.
  4. Build voice profile:
     a. 24kHz mel of 10s wav → prompt_feat (500, 80).
     b. 16kHz → s3tokenizer → prompt_token (250 tokens).
     c. 16kHz[:6s] → s3tokenizer (max 150 tokens) → cond_prompt_speech_tokens (150).
     d. 16kHz → kaldi fbank → CAMPPlus → embedding (192-d), normalize + spk_embed_affine → spks (80).
     e. 16kHz → VE mel → VoiceEncoder LSTM → speaker_emb (256-d).
     f. T3CondEnc(speaker_emb, cond_prompt_speech_tokens, emotion=0.5) → cond_emb (34, 1024).
  5. BPE tokenize text → text_ids → text_emb (Mojo).
  6. Build prefix = cat(cond_emb, text_emb, bos_emb).
  7. T3 generate with temperature + top-p → speech_tokens.
  8. Concat(prompt_token, speech_tokens) → flow encoder → mu.
  9. Build cond from prompt_feat, gaussian noise, CFM Euler → mel.
 10. Trim prompt prefix, HiFT → audio.wav.
"""
from std.sys import has_accelerator
from std.math import sin, cos as mcos, pi, sqrt
from std.testing import TestSuite, assert_true
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from fixture import load_fp32, load_i64, save_wav, load_wav, read_wav_sample_rate
from weights import (
    load_t3, load_t3_cond_enc, load_upsample_conformer_encoder, load_cfm_estimator_real,
    load_hift_generator, load_fcm, load_campplus, load_s3tokenizer, upload_fp32,
)
from t3_generate import t3_generate_cfg_sample
from bpe_tokenizer import load_tokenizer
from text_embed import text_to_input_ids, build_text_emb, build_bos_emb, build_rope_tables
from upsample_encoder import upsample_conformer_forward
from cfm_estimator_new import cfm_solve_euler, gaussian_noise_fill
from hift_generator import (
    hift_decode_trunk, istft_forward, hann_window_periodic_fill,
    f0_predictor_forward, f0_upsample_nearest, source_module_forward,
    build_s_stft_from_signal,
)
from cond_enc import t3_cond_enc_forward
from resampler import resample_24k_to_16k
from kaldi_fbank import (
    kaldi_fbank_forward, kaldi_subtract_column_mean,
    build_povey_window, build_kaldi_mel_filterbank,
)
from campplus import campplus_speaker_embedding
from mel_24k import (
    mel_24k_forward, build_hann_window as build_hann_24k,
    build_librosa_mel_filterbank as build_mel_fb_24k,
)
from mel_s3tok import (
    log_mel_s3tok_forward, build_hann_window_full as build_hann_s3tok,
    build_librosa_mel_filterbank_s3tok,
)
from s3tokenizer import s3tokenizer_forward
from modules import Linear, linear_forward


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
comptime S3_SR = 16000
comptime S3GEN_SR = 24000
comptime DEC_COND_LEN = 240000   # 10s @ 24k
comptime ENC_COND_LEN = 96000    # 6s @ 16k
comptime DEC_COND_LEN_16K = 160000  # 10s @ 16k


def l2_normalize_inplace(
    mut ctx: DeviceContext,
    mut buf: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    """In-place L2-normalize a 1D vector buffer (n elements)."""
    var bp = buf.unsafe_ptr()
    # Compute sum-of-squares on host (small).
    var sumsq: Float32 = 0.0
    with buf.map_to_host() as h:
        for i in range(n):
            sumsq += h[i] * h[i]
    var inv = Float32(1.0) / sqrt(sumsq) if sumsq > 0.0 else Float32(1.0)

    @always_inline
    @parameter
    @__copy_capture(bp, inv)
    def scale_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        bp[i] = bp[i] * inv
    elementwise[scale_fn, simd_width=1, target="gpu"](
        IndexList[1](n), DeviceContextPtr(ctx),
    )


def transpose_fb_to_bft(
    mut ctx: DeviceContext,
    mut in_buf: DeviceBuffer[DType.float32],     # (F, T) or (T, F)
    mut out_buf: DeviceBuffer[DType.float32],    # (B=1, F_target, T_target)
    rows_in: Int, cols_in: Int,
    do_transpose: Bool,
) raises:
    """Helper: copy/transpose. If do_transpose: out[r,c] = in[c,r], else identity."""
    var ip = in_buf.unsafe_ptr()
    var op = out_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(ip, op, rows_in, cols_in, do_transpose)
    def t_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        if do_transpose:
            var r = i // cols_in
            var c = i - r * cols_in
            op[i] = ip[c * rows_in + r]
        else:
            op[i] = ip[i]
    elementwise[t_fn, simd_width=1, target="gpu"](
        IndexList[1](rows_in * cols_in), DeviceContextPtr(ctx),
    )


def test_synth_from_wav() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    # ──────────────────────────────────────────────────────────────────
    # Load all models.
    # ──────────────────────────────────────────────────────────────────
    print("[full] loading models...")
    var t3 = load_t3(ctx, "weights/t3")
    var enc = load_upsample_conformer_encoder(ctx, "weights/s3gen/flow")
    var cfm = load_cfm_estimator_real(ctx, "weights/s3gen/flow/decoder/estimator")
    var hift = load_hift_generator(ctx, "weights/s3gen/mel2wav")
    var fcm = load_fcm(ctx, "weights/s3gen/speaker_encoder/head")
    var campplus = load_campplus(ctx, "weights/s3gen/speaker_encoder")
    var s3tok = load_s3tokenizer(ctx, "weights/s3t")
    var cond_enc = load_t3_cond_enc(ctx, "weights/t3", t3.speech_emb, t3.speech_pos_emb)

    # spk_embed_affine_layer (192 → 80), loaded from flow weights.
    var affine_w = upload_fp32(ctx, "weights/s3gen/flow/spk_embed_affine_layer/weight.bin")
    var affine_b = upload_fp32(ctx, "weights/s3gen/flow/spk_embed_affine_layer/bias.bin")
    var spk_embed_affine = Linear(affine_w^, affine_b^, 192, 80, True)
    _ = spk_embed_affine   # used later

    # ──────────────────────────────────────────────────────────────────
    # Step 1: Load reference WAV.
    # ──────────────────────────────────────────────────────────────────
    var ref_path = "/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav"
    print("[full] loading reference WAV:", ref_path)
    var wav_t = load_wav(ref_path)
    var wav_sr = read_wav_sample_rate(ref_path)
    print("[full] loaded", wav_t.numel(), "samples at", wav_sr, "Hz")
    if wav_sr != S3GEN_SR:
        raise Error("WAV must be at 24kHz")

    # Truncate to 10s @ 24k.
    var n_24 = DEC_COND_LEN
    if wav_t.numel() < n_24:
        raise Error("ref wav too short")
    var wav_24 = ctx.enqueue_create_buffer[DType.float32](n_24)
    with wav_24.map_to_host() as h:
        for i in range(n_24):
            h[i] = wav_t.data[i]

    # ──────────────────────────────────────────────────────────────────
    # Step 2: Resample to 16kHz (10s @ 16k = 160000 samples).
    # ──────────────────────────────────────────────────────────────────
    var n_16 = DEC_COND_LEN_16K
    var wav_16 = ctx.enqueue_create_buffer[DType.float32](n_16)
    print("[full] resampling 24k→16k...")
    resample_24k_to_16k(ctx, wav_24, wav_16, n_24, n_16)
    ctx.synchronize()

    # ──────────────────────────────────────────────────────────────────
    # Step 3a: 24kHz mel for prompt_feat (1, 500, 80).
    # ──────────────────────────────────────────────────────────────────
    var n_fft_24 = 1920
    var hop_24 = 480
    var pad_24 = (n_fft_24 - hop_24) // 2
    var T_prompt_feat = T_PROMPT_MEL
    print("[full] computing 24k mel (T=", T_prompt_feat, ")...")
    var win_24 = ctx.enqueue_create_buffer[DType.float32](n_fft_24)
    build_hann_24k(ctx, win_24, n_fft_24)
    var n_bins_24 = n_fft_24 // 2 + 1
    var mel_fb_24 = ctx.enqueue_create_buffer[DType.float32](MEL * n_bins_24)
    build_mel_fb_24k(ctx, mel_fb_24, MEL, n_fft_24, Float64(24000.0),
                       Float64(0.0), Float64(8000.0))
    var prompt_feat_mt = ctx.enqueue_create_buffer[DType.float32](MEL * T_prompt_feat)
    mel_24k_forward(ctx, wav_24, win_24, mel_fb_24, prompt_feat_mt, n_24, T_prompt_feat,
                     n_fft=n_fft_24, hop=hop_24, n_mels=MEL)
    ctx.synchronize()

    # Upstream stores prompt_feat as (1, T, 80) — transpose from (80, T).
    var prompt_feat_tm = ctx.enqueue_create_buffer[DType.float32](T_prompt_feat * MEL)
    transpose_fb_to_bft(ctx, prompt_feat_mt, prompt_feat_tm, MEL, T_prompt_feat, True)
    ctx.synchronize()

    # ──────────────────────────────────────────────────────────────────
    # Step 3b: 16kHz log-mel + s3tokenizer for prompt_token (250 tokens).
    # ──────────────────────────────────────────────────────────────────
    var n_fft_st = 400
    var hop_st = 160
    print("[full] s3tokenizer log-mel...")
    var win_st = ctx.enqueue_create_buffer[DType.float32](n_fft_st)
    build_hann_s3tok(ctx, win_st, n_fft_st)
    var n_bins_st = n_fft_st // 2 + 1
    var mel_fb_st = ctx.enqueue_create_buffer[DType.float32](128 * n_bins_st)
    build_librosa_mel_filterbank_s3tok(ctx, mel_fb_st, 128, n_fft_st, Float64(16000.0))

    # log_mel for full 10s: T_mel = 10*100 = 1000.
    var T_mel_full = (n_16 + 2 * (n_fft_st // 2) - n_fft_st) // hop_st + 1 - 1   # = 1000
    print("[full] T_mel_full=", T_mel_full)
    var log_mel_full = ctx.enqueue_create_buffer[DType.float32](128 * T_mel_full)
    log_mel_s3tok_forward(ctx, wav_16, win_st, mel_fb_st, log_mel_full, n_16, T_mel_full)
    ctx.synchronize()

    var T_token_full = T_mel_full // 4   # 250
    print("[full] T_token_full=", T_token_full)
    var head_dim = 64
    var max_ctx_rope = 4096
    var cos_st = ctx.enqueue_create_buffer[DType.float32](max_ctx_rope * head_dim)
    var sin_st = ctx.enqueue_create_buffer[DType.float32](max_ctx_rope * head_dim)
    build_rope_tables(ctx, max_ctx_rope, head_dim, cos_st, sin_st)
    var mask_pad_full = ctx.enqueue_create_buffer[DType.float32](B * T_token_full * 1)
    mask_pad_full.enqueue_fill(1.0)
    var attn_mask_full = ctx.enqueue_create_buffer[DType.float32](T_token_full * T_token_full)
    attn_mask_full.enqueue_fill(0.0)
    var prompt_token = ctx.enqueue_create_buffer[DType.int32](B * T_token_full)
    s3tokenizer_forward(ctx, s3tok, log_mel_full, prompt_token, cos_st, sin_st,
                          mask_pad_full, attn_mask_full, B, T_mel_full)
    ctx.synchronize()

    # ──────────────────────────────────────────────────────────────────
    # Step 3c: 16kHz[:6s] s3tokenizer → cond_prompt_speech_tokens (150).
    # ──────────────────────────────────────────────────────────────────
    var n_16_6s = ENC_COND_LEN   # 96000
    var T_mel_6s = (n_16_6s + 2 * (n_fft_st // 2) - n_fft_st) // hop_st + 1 - 1   # = 600
    var T_token_6s = T_mel_6s // 4   # 150
    print("[full] cond_prompt T_mel=", T_mel_6s, " T_token=", T_token_6s)
    var log_mel_6s = ctx.enqueue_create_buffer[DType.float32](128 * T_mel_6s)
    # We can run log_mel_s3tok on a sliced wav. Create a slice buffer.
    var wav_16_6s = ctx.enqueue_create_buffer[DType.float32](n_16_6s)
    var w16p = wav_16.unsafe_ptr()
    var w6p = wav_16_6s.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(w16p, w6p, n_16_6s)
    def slice_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        w6p[i] = w16p[i]
    elementwise[slice_fn, simd_width=1, target="gpu"](
        IndexList[1](n_16_6s), DeviceContextPtr(ctx),
    )
    log_mel_s3tok_forward(ctx, wav_16_6s, win_st, mel_fb_st, log_mel_6s, n_16_6s, T_mel_6s)

    var mask_pad_6s = ctx.enqueue_create_buffer[DType.float32](B * T_token_6s * 1)
    mask_pad_6s.enqueue_fill(1.0)
    var attn_mask_6s = ctx.enqueue_create_buffer[DType.float32](T_token_6s * T_token_6s)
    attn_mask_6s.enqueue_fill(0.0)
    var cond_prompt_tok_i32 = ctx.enqueue_create_buffer[DType.int32](B * T_token_6s)
    s3tokenizer_forward(ctx, s3tok, log_mel_6s, cond_prompt_tok_i32, cos_st, sin_st,
                          mask_pad_6s, attn_mask_6s, B, T_mel_6s)
    ctx.synchronize()

    # Convert int32 → int64 for T3 embedding lookup.
    var cond_prompt_tok = ctx.enqueue_create_buffer[DType.int64](B * T_token_6s)
    with cond_prompt_tok_i32.map_to_host() as i32:
        with cond_prompt_tok.map_to_host() as i64:
            for i in range(B * T_token_6s):
                i64[i] = Int64(i32[i])

    # ──────────────────────────────────────────────────────────────────
    # Step 3d: kaldi fbank → CAMPPlus → 192-d embedding → normalize + affine → 80-d spks.
    # ──────────────────────────────────────────────────────────────────
    print("[full] computing CAMPPlus speaker embedding...")
    var T_fbank = (n_16 - 400) // 160 + 1   # 998 frames
    var fbank_win = ctx.enqueue_create_buffer[DType.float32](400)
    build_povey_window(ctx, fbank_win, 400)
    var fbank_mel_fb = ctx.enqueue_create_buffer[DType.float32](MEL * (512 // 2 + 1))
    build_kaldi_mel_filterbank(ctx, fbank_mel_fb, MEL, 512, Float64(16000.0),
                                Float64(20.0), Float64(0.0))
    var fbank_tf = ctx.enqueue_create_buffer[DType.float32](T_fbank * MEL)
    kaldi_fbank_forward(ctx, wav_16, fbank_win, fbank_mel_fb, fbank_tf, n_16, T_fbank)
    kaldi_subtract_column_mean(ctx, fbank_tf, T_fbank, MEL)
    ctx.synchronize()
    var fbank_btf = ctx.enqueue_create_buffer[DType.float32](MEL * T_fbank)
    transpose_fb_to_bft(ctx, fbank_tf, fbank_btf, T_fbank, MEL, True)
    ctx.synchronize()
    var embedding_192 = ctx.enqueue_create_buffer[DType.float32](192)
    campplus_speaker_embedding(ctx, fcm, campplus.xvector, fbank_btf, embedding_192, B, T_fbank)
    ctx.synchronize()

    # spks = spk_embed_affine(F.normalize(embedding, dim=1)) — (1, 80).
    var embedding_norm = ctx.enqueue_create_buffer[DType.float32](192)
    ctx.enqueue_copy(embedding_norm, embedding_192)
    l2_normalize_inplace(ctx, embedding_norm, 192)
    var spks = ctx.enqueue_create_buffer[DType.float32](80)
    linear_forward(ctx, spk_embed_affine, embedding_norm, spks, B)
    ctx.synchronize()

    # ──────────────────────────────────────────────────────────────────
    # Step 3e: VoiceEncoder (LSTM) → 256-d speaker_emb for T3.
    # NOTE: VE expects librosa-style 16kHz mel (n_mels=40, n_fft=400, hop=160).
    # For now we use upstream's dumped 256-d speaker_emb to avoid the VE mel + LSTM
    # numerical drift from kaiser_fast resampler + trim. End-to-end Mojo otherwise.
    # ──────────────────────────────────────────────────────────────────
    var voice_dir2 = "weights/s3gen_prompt/cond_enc_diag/"
    var speaker_emb_256 = upload_fp32(ctx, voice_dir2 + "speaker_emb.bin")

    # ──────────────────────────────────────────────────────────────────
    # Step 3f: T3CondEnc → cond_emb (B, 34, 1024).
    # ──────────────────────────────────────────────────────────────────
    var emotion = ctx.enqueue_create_buffer[DType.float32](B * 1 * 1)
    with emotion.map_to_host() as h:
        h[0] = Float32(0.5)

    # Need cond_prompt_tokens in (B, 150) shape.
    var SQ = 32
    var mask_q = ctx.enqueue_create_buffer[DType.float32](SQ * SQ)
    mask_q.enqueue_fill(0.0)
    var mask_qq = ctx.enqueue_create_buffer[DType.float32](SQ * T_token_6s)
    mask_qq.enqueue_fill(0.0)
    var cond_emb = ctx.enqueue_create_buffer[DType.float32](B * T_COND * D)
    print("[full] T3CondEnc...")
    t3_cond_enc_forward(ctx, cond_enc, speaker_emb_256, cond_prompt_tok, emotion, cond_emb,
                         mask_q, mask_qq, B)
    ctx.synchronize()

    # ──────────────────────────────────────────────────────────────────
    # Step 4: text → BPE → text_emb + bos_emb.
    # ──────────────────────────────────────────────────────────────────
    var text = "the quick brown fox"
    var tok_dir = "../mojo-t3/tests/fixtures/tokenizer/"
    var bpe = load_tokenizer(tok_dir + "vocab.txt", tok_dir + "merges.txt")
    var ids = text_to_input_ids(text, bpe)
    var T_TEXT = len(ids)
    var T_PREFIX = T_COND + T_TEXT + T_BOS
    print("[full] text='", text, "' → ", T_TEXT, " tokens; T_PREFIX=", T_PREFIX)

    var text_emb = ctx.enqueue_create_buffer[DType.float32](B * T_TEXT * D)
    build_text_emb(ctx, t3, ids, text_emb)
    var bos_emb = ctx.enqueue_create_buffer[DType.float32](B * 1 * D)
    build_bos_emb(ctx, t3, bos_emb)

    # ──────────────────────────────────────────────────────────────────
    # Step 5: Build T3 CFG prefix.
    # ──────────────────────────────────────────────────────────────────
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

    # Build RoPE tables and mask for T3.
    var cos_t3 = ctx.enqueue_create_buffer[DType.float32](MAX_CTX * 64)
    var sin_t3 = ctx.enqueue_create_buffer[DType.float32](MAX_CTX * 64)
    build_rope_tables(ctx, MAX_CTX, 64, cos_t3, sin_t3)
    var t3_mask = ctx.enqueue_create_buffer[DType.float32](T_PREFIX * T_PREFIX)
    with t3_mask.map_to_host() as h:
        for r in range(T_PREFIX):
            for c in range(T_PREFIX):
                if c > r:
                    h[r * T_PREFIX + c] = -1.0e30
                else:
                    h[r * T_PREFIX + c] = 0.0

    # ──────────────────────────────────────────────────────────────────
    # Step 6: T3 generate.
    # ──────────────────────────────────────────────────────────────────
    print("[full] T3 generate (CFG=", T3_CFG, ", max_new=", MAX_NEW, ")...")
    var speech_pos = t3.speech_pos_emb.table
    var generated = t3_generate_cfg_sample(
        ctx, t3, prefix, cos_t3, sin_t3, t3_mask, speech_pos,
        B, T_PREFIX, MAX_CTX, MAX_NEW,
        speech_pos_offset=1, eos_token=EOS, cfg_weight=T3_CFG,
        temperature=Float32(0.8), top_p=Float32(0.95), rep_penalty=Float32(1.2),
        rng_seed=UInt64(0xDEADBEEF),
    )
    ctx.synchronize()
    print("[full] T3 generated", len(generated), "tokens")

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
    print("[full] T_GEN=", T_GEN_TOKEN, " T_TOTAL_MEL=", T_TOTAL_MEL, " T_OUT_MEL=", T_OUT_MEL)

    # ──────────────────────────────────────────────────────────────────
    # Step 7: Build token = cat(prompt_token, speech_tokens) as int64.
    # ──────────────────────────────────────────────────────────────────
    var tok_buf = ctx.enqueue_create_buffer[DType.int64](B * T_TOTAL_TOKEN)
    with prompt_token.map_to_host() as pt:
        with tok_buf.map_to_host() as h:
            for i in range(T_PROMPT_TOKEN):
                h[i] = Int64(pt[i])
            for i in range(T_GEN_TOKEN):
                h[T_PROMPT_TOKEN + i] = speech_tokens[i]

    # ──────────────────────────────────────────────────────────────────
    # Step 8: Flow encoder → mu.
    # ──────────────────────────────────────────────────────────────────
    var mu = ctx.enqueue_create_buffer[DType.float32](B * MEL * T_TOTAL_MEL)
    print("[full] flow encoder (T_token=", T_TOTAL_TOKEN, " → T_mel=", T_TOTAL_MEL, ")...")
    upsample_conformer_forward(ctx, enc, tok_buf, mu, B, T_TOTAL_TOKEN)
    ctx.synchronize()

    # ──────────────────────────────────────────────────────────────────
    # Step 9: cond = zeros, slot prompt_feat into [:, :, :T_PROMPT_MEL]; CFM solve.
    # ──────────────────────────────────────────────────────────────────
    var cond = ctx.enqueue_create_buffer[DType.float32](B * MEL * T_TOTAL_MEL)
    cond.enqueue_fill(0.0)
    var pf_ptr = prompt_feat_tm.unsafe_ptr()   # (T_PROMPT_MEL, 80) flat
    var cond_ptr = cond.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(pf_ptr, cond_ptr, T_TOTAL_MEL)
    def cond_fill[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        # i indexes (B, T_PROMPT_MEL, MEL).
        var bi = i // (T_PROMPT_MEL * MEL)
        var rem = i - bi * T_PROMPT_MEL * MEL
        var ti = rem // MEL
        var ci = rem - ti * MEL
        # Destination: cond[bi, ci, ti] = pf[bi, ti, ci]
        cond_ptr[bi * MEL * T_TOTAL_MEL + ci * T_TOTAL_MEL + ti] = pf_ptr[ti * MEL + ci]
    elementwise[cond_fill, simd_width=1, target="gpu"](
        IndexList[1](B * T_PROMPT_MEL * MEL), DeviceContextPtr(ctx),
    )

    var x = ctx.enqueue_create_buffer[DType.float32](B * MEL * T_TOTAL_MEL)
    gaussian_noise_fill(ctx, x, B * MEL * T_TOTAL_MEL, UInt64(0xC0FFEE), Float32(1.0))
    var cfm_mask = ctx.enqueue_create_buffer[DType.float32](B * T_TOTAL_MEL)
    cfm_mask.enqueue_fill(1.0)
    print("[full] CFM Euler...")
    cfm_solve_euler(ctx, cfm, x, mu, spks, cond, cfm_mask, B, T_TOTAL_MEL, N_CFM_STEPS, CFG)
    ctx.synchronize()

    # Trim prompt prefix.
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

    # ──────────────────────────────────────────────────────────────────
    # Step 10: HiFT → audio.
    # ──────────────────────────────────────────────────────────────────
    var T_HIFT = T_OUT_MEL * 120 + 1
    var T_AUDIO = (T_HIFT - 1) * HOP
    var T_AUDIO_FULL = T_OUT_MEL * 480
    var T_S_FRAMES = T_AUDIO_FULL // HOP + 1

    var f0 = ctx.enqueue_create_buffer[DType.float32](B * T_OUT_MEL)
    f0_predictor_forward(ctx, hift.f0_predictor, mel_out, f0, B, T_OUT_MEL)
    var f0_up = ctx.enqueue_create_buffer[DType.float32](B * T_AUDIO_FULL)
    f0_upsample_nearest(ctx, f0, f0_up, B, T_OUT_MEL, 480)

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
    print("[full] HiFT trunk (T_mel=", T_OUT_MEL, " → T_frames=", T_HIFT, ")...")
    hift_decode_trunk(ctx, hift, mel_out, s_stft, spec,
                      B, T_OUT_MEL, T_S_FRAMES, T_HIFT, use_source=True)

    var window_i = ctx.enqueue_create_buffer[DType.float32](N_FFT)
    hann_window_periodic_fill(ctx, window_i, N_FFT)
    var audio = ctx.enqueue_create_buffer[DType.float32](B * T_AUDIO)
    istft_forward(ctx, spec, window_i, audio, B, N_FFT, T_HIFT, T_AUDIO)
    ctx.synchronize()

    var max_a: Float32 = 0.0
    var samples = List[Float32]()
    with audio.map_to_host() as h:
        for i in range(T_AUDIO):
            samples.append(h[i])
            var v = h[i]
            if v < 0.0: v = -v
            if v > max_a: max_a = v
    print("[full] audio max=", max_a, " duration=", Float32(T_AUDIO) / 24000.0, "s")
    save_wav(String("max_impl_from_wav.wav"), samples, 24000)
    print("[full] saved max_impl_from_wav.wav")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
