"""Stage 1: voice preprocessing — ref.wav → cached voice profile on disk.

Pipeline (pure Mojo runtime, ffmpeg+soxr for bit-exact resampling):
  1. Load 24kHz WAV.
  2. Resample to 16kHz via ffmpeg+libsoxr (bit-exact to librosa.resample).
  3. Compute s3tokenizer log-mel @ 16kHz → run s3tokenizer →
     prompt_token (250 tokens), cond_prompt_speech_tokens (150 tokens).
  4. Compute 24kHz mel → prompt_feat (500, 80).
  5. Compute Kaldi fbank → CAMPPlus → 192-d → spk_embed_affine → spks (80).
  6. Compute VoiceEncoder mel → multi-partial VE inference → speaker_emb (256).
  7. T3CondEnc(speaker_emb, cond_prompt_tokens, emotion=0.5) → cond_emb (34, 1024).

Saves all to `weights/voice_profile/` for stage 2 to load.
"""
from std.sys import has_accelerator
from std.math import sqrt
from std.testing import TestSuite
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from fixture import load_wav, read_wav_sample_rate, save_fp32_1d
from weights import load_fcm, load_campplus, load_s3tokenizer, upload_fp32
from modules import Linear, linear_forward
from resampler_soxr import soxr_resample_24k_to_16k
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
from text_embed import build_rope_tables


comptime B = 1
comptime MEL = 80


def save_int32_1d_as_int64_bin(path: String, samples: List[Int32]) raises:
    """Write i64-tagged binary tensor from int32 source (for compatibility with load_i64)."""
    var f = open(path, "w")
    var hdr = List[UInt8](capacity=20)
    var r: Int64 = 1
    for k in range(8):
        hdr.append(UInt8(Int((r >> Int64(8 * k)) & 0xFF)))
    var n: Int64 = Int64(len(samples))
    for k in range(8):
        hdr.append(UInt8(Int((n >> Int64(8 * k)) & 0xFF)))
    var tg: Int32 = 2  # int64
    for k in range(4):
        hdr.append(UInt8(Int((tg >> Int32(8 * k)) & 0xFF)))
    f.write_bytes(Span(hdr))
    var buf = List[UInt8](capacity=len(samples) * 8)
    for j in range(len(samples)):
        var v: Int64 = Int64(samples[j])
        for k in range(8):
            buf.append(UInt8(Int((v >> Int64(8 * k)) & 0xFF)))
    f.write_bytes(Span(buf))
    f.close()


def save_fp32_3d(path: String, data: List[Float32], d0: Int, d1: Int, d2: Int) raises:
    """Write a 3-D Float32 binary tensor (rank=3 in header)."""
    var f = open(path, "w")
    var hdr = List[UInt8](capacity=36)
    var r: Int64 = 3
    for k in range(8):
        hdr.append(UInt8(Int((r >> Int64(8 * k)) & 0xFF)))
    var shapes = [Int64(d0), Int64(d1), Int64(d2)]
    for s in shapes:
        for k in range(8):
            hdr.append(UInt8(Int((s >> Int64(8 * k)) & 0xFF)))
    var tg: Int32 = 0
    for k in range(4):
        hdr.append(UInt8(Int((tg >> Int32(8 * k)) & 0xFF)))
    f.write_bytes(Span(hdr))

    var n = len(data)
    var i = 0
    while i < n:
        var chunk_end = min(i + 1024, n)
        var buf = List[UInt8](capacity=(chunk_end - i) * 4)
        for j in range(i, chunk_end):
            var v = data[j]
            var p = UnsafePointer(to=v).bitcast[UInt32]()
            var bits = p[0]
            for k in range(4):
                buf.append(UInt8(Int((bits >> UInt32(8 * k)) & 0xFF)))
        f.write_bytes(Span(buf))
        i = chunk_end
    f.close()


def test_preprocess_voice() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    print("[prep] loading models...")
    var fcm = load_fcm(ctx, "weights/s3gen/speaker_encoder/head")
    var campplus = load_campplus(ctx, "weights/s3gen/speaker_encoder")
    var s3tok = load_s3tokenizer(ctx, "weights/s3t")

    var affine_w = upload_fp32(ctx, "weights/s3gen/flow/spk_embed_affine_layer/weight.bin")
    var affine_b = upload_fp32(ctx, "weights/s3gen/flow/spk_embed_affine_layer/bias.bin")
    var spk_embed_affine = Linear(affine_w^, affine_b^, 192, 80, True)

    var ref_path = "/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav"
    print("[prep] loading WAV:", ref_path)
    var wav_t = load_wav(ref_path)
    var sr = read_wav_sample_rate(ref_path)
    if sr != 24000:
        raise Error("Expected 24kHz WAV")

    var n_24 = 240000
    var n_16 = 160000
    var wav_24 = ctx.enqueue_create_buffer[DType.float32](n_24)
    with wav_24.map_to_host() as h:
        for i in range(n_24):
            h[i] = wav_t.data[i]

    print("[prep] resampling 24k→16k via ffmpeg+libsoxr (bit-exact)...")
    var wav_16 = ctx.enqueue_create_buffer[DType.float32](n_16)
    soxr_resample_24k_to_16k(ctx, wav_24, wav_16, n_24, n_16)
    ctx.synchronize()

    # ── 1. s3tokenizer log-mel for full 10s, then 6s prefix.
    print("[prep] s3tokenizer log-mel...")
    var win_st = ctx.enqueue_create_buffer[DType.float32](400)
    build_hann_s3tok(ctx, win_st, 400)
    var mel_fb_st = ctx.enqueue_create_buffer[DType.float32](128 * 201)
    build_librosa_mel_filterbank_s3tok(ctx, mel_fb_st, 128, 400, Float64(16000.0))

    var T_mel_full = (n_16 + 2 * 200 - 400) // 160 + 1 - 1   # 1000
    var log_mel_full = ctx.enqueue_create_buffer[DType.float32](128 * T_mel_full)
    log_mel_s3tok_forward(ctx, wav_16, win_st, mel_fb_st, log_mel_full, n_16, T_mel_full)
    ctx.synchronize()

    var T_token_full = T_mel_full // 4   # 250
    var cos_st = ctx.enqueue_create_buffer[DType.float32](4096 * 64)
    var sin_st = ctx.enqueue_create_buffer[DType.float32](4096 * 64)
    build_rope_tables(ctx, 4096, 64, cos_st, sin_st)
    var mp_full = ctx.enqueue_create_buffer[DType.float32](B * T_token_full * 1)
    mp_full.enqueue_fill(1.0)
    var amask_full = ctx.enqueue_create_buffer[DType.float32](T_token_full * T_token_full)
    amask_full.enqueue_fill(0.0)
    var prompt_token = ctx.enqueue_create_buffer[DType.int32](B * T_token_full)
    s3tokenizer_forward(ctx, s3tok, log_mel_full, prompt_token, cos_st, sin_st,
                          mp_full, amask_full, B, T_mel_full)
    ctx.synchronize()

    # 6s prefix for cond_prompt_speech_tokens.
    var n_16_6s = 96000
    var T_mel_6s = (n_16_6s + 2 * 200 - 400) // 160 + 1 - 1   # 600
    var T_token_6s = T_mel_6s // 4   # 150
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

    var log_mel_6s = ctx.enqueue_create_buffer[DType.float32](128 * T_mel_6s)
    log_mel_s3tok_forward(ctx, wav_16_6s, win_st, mel_fb_st, log_mel_6s, n_16_6s, T_mel_6s)
    var mp_6s = ctx.enqueue_create_buffer[DType.float32](B * T_token_6s * 1)
    mp_6s.enqueue_fill(1.0)
    var amask_6s = ctx.enqueue_create_buffer[DType.float32](T_token_6s * T_token_6s)
    amask_6s.enqueue_fill(0.0)
    var cond_prompt_tok = ctx.enqueue_create_buffer[DType.int32](B * T_token_6s)
    s3tokenizer_forward(ctx, s3tok, log_mel_6s, cond_prompt_tok, cos_st, sin_st,
                          mp_6s, amask_6s, B, T_mel_6s)
    ctx.synchronize()

    # ── 2. 24kHz mel for prompt_feat (1, 500, 80).
    print("[prep] 24kHz mel...")
    var win_24 = ctx.enqueue_create_buffer[DType.float32](1920)
    build_hann_24k(ctx, win_24, 1920)
    var mel_fb_24 = ctx.enqueue_create_buffer[DType.float32](MEL * 961)
    build_mel_fb_24k(ctx, mel_fb_24, MEL, 1920, Float64(24000.0),
                      Float64(0.0), Float64(8000.0))
    var T_prompt_feat = 500
    var prompt_feat_mt = ctx.enqueue_create_buffer[DType.float32](MEL * T_prompt_feat)
    mel_24k_forward(ctx, wav_24, win_24, mel_fb_24, prompt_feat_mt, n_24, T_prompt_feat)
    ctx.synchronize()

    # ── 3. CAMPPlus speaker embedding → spks.
    print("[prep] CAMPPlus speaker embedding...")
    var T_fbank = (n_16 - 400) // 160 + 1
    var fwin = ctx.enqueue_create_buffer[DType.float32](400)
    build_povey_window(ctx, fwin, 400)
    var fmel_fb = ctx.enqueue_create_buffer[DType.float32](MEL * 257)
    build_kaldi_mel_filterbank(ctx, fmel_fb, MEL, 512, Float64(16000.0),
                                Float64(20.0), Float64(0.0))
    var fbank_tf = ctx.enqueue_create_buffer[DType.float32](T_fbank * MEL)
    kaldi_fbank_forward(ctx, wav_16, fwin, fmel_fb, fbank_tf, n_16, T_fbank)
    kaldi_subtract_column_mean(ctx, fbank_tf, T_fbank, MEL)

    # Transpose (T, 80) → (80, T)
    var fbank_bft = ctx.enqueue_create_buffer[DType.float32](MEL * T_fbank)
    var fp = fbank_tf.unsafe_ptr()
    var bp = fbank_bft.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(fp, bp, T_fbank)
    def tr_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var f = i // T_fbank
        var t = i - f * T_fbank
        bp[f * T_fbank + t] = fp[t * MEL + f]
    elementwise[tr_fn, simd_width=1, target="gpu"](
        IndexList[1](MEL * T_fbank), DeviceContextPtr(ctx),
    )

    var emb_192 = ctx.enqueue_create_buffer[DType.float32](192)
    campplus_speaker_embedding(ctx, fcm, campplus.xvector, fbank_bft, emb_192, B, T_fbank)
    ctx.synchronize()

    # L2 normalize + affine → spks (80).
    var sumsq: Float32 = 0.0
    with emb_192.map_to_host() as h:
        for i in range(192):
            sumsq += h[i] * h[i]
    var inv = Float32(1.0) / sqrt(sumsq)
    var emb_norm = ctx.enqueue_create_buffer[DType.float32](192)
    var ep = emb_192.unsafe_ptr()
    var enp = emb_norm.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(ep, enp, inv)
    def norm_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        enp[i] = ep[i] * inv
    elementwise[norm_fn, simd_width=1, target="gpu"](
        IndexList[1](192), DeviceContextPtr(ctx),
    )

    var spks = ctx.enqueue_create_buffer[DType.float32](80)
    linear_forward(ctx, spk_embed_affine, emb_norm, spks, B)
    ctx.synchronize()

    # ── 4. Save profile to disk.
    print("[prep] saving voice profile to weights/voice_profile/...")
    # prompt_token i32 → i64 binary.
    var pt_list = List[Int32](capacity=T_token_full)
    with prompt_token.map_to_host() as h:
        for i in range(T_token_full):
            pt_list.append(h[i])
    save_int32_1d_as_int64_bin("weights/voice_profile/prompt_token.bin", pt_list)

    var cpt_list = List[Int32](capacity=T_token_6s)
    with cond_prompt_tok.map_to_host() as h:
        for i in range(T_token_6s):
            cpt_list.append(h[i])
    save_int32_1d_as_int64_bin("weights/voice_profile/cond_prompt_speech_tokens.bin", cpt_list)

    # spks fp32 (80,)
    var spks_list = List[Float32](capacity=80)
    with spks.map_to_host() as h:
        for i in range(80):
            spks_list.append(h[i])
    save_fp32_1d("weights/voice_profile/spks.bin", spks_list)

    # prompt_feat (B, T_prompt_feat, MEL): transpose from (MEL, T) → (T, MEL).
    var pfeat_list = List[Float32](capacity=T_prompt_feat * MEL)
    with prompt_feat_mt.map_to_host() as h:
        for t in range(T_prompt_feat):
            for m in range(MEL):
                pfeat_list.append(h[m * T_prompt_feat + t])
    save_fp32_3d("weights/voice_profile/prompt_feat.bin", pfeat_list, 1, T_prompt_feat, MEL)

    print("[prep] DONE. T_token_full=", T_token_full, " T_token_6s=", T_token_6s)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
