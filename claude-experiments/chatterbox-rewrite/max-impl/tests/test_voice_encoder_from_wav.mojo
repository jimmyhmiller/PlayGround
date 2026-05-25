"""End-to-end: 24kHz wav → Mojo resample → kaldi fbank → CAMPPlus speaker embedding.

Tests the full voice-side pipeline that turns a reference WAV into the 192-d
speaker embedding used by the CFM.
"""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.math import sqrt
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from fixture import load_fp32
from weights import load_fcm, load_campplus, upload_fp32
from resampler import resample_24k_to_16k
from kaldi_fbank import (
    kaldi_fbank_forward, kaldi_subtract_column_mean,
    build_povey_window, build_kaldi_mel_filterbank,
)
from campplus import campplus_speaker_embedding


def _diff(name: String, mut mojo: DeviceBuffer[DType.float32], ref_path: String) raises:
    var reference = load_fp32(ref_path)
    var ref_n = reference.numel()
    var max_abs: Float32 = 0.0
    var sum_diff_sq: Float32 = 0.0
    var sum_ref_sq: Float32 = 0.0
    var dot: Float32 = 0.0
    var sum_a_sq: Float32 = 0.0
    var sum_b_sq: Float32 = 0.0
    with mojo.map_to_host() as h:
        for i in range(ref_n):
            var dd = h[i] - reference.data[i]
            var d = dd
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_diff_sq += dd * dd
            sum_ref_sq += reference.data[i] * reference.data[i]
            dot += h[i] * reference.data[i]
            sum_a_sq += h[i] * h[i]
            sum_b_sq += reference.data[i] * reference.data[i]
    var rel = sqrt(sum_diff_sq / sum_ref_sq) if sum_ref_sq > 0.0 else Float32(0.0)
    var cos_sim = dot / (sqrt(sum_a_sq) * sqrt(sum_b_sq)) if sum_a_sq > 0.0 and sum_b_sq > 0.0 else Float32(0.0)
    print("[ve-wav]", name, ": max-abs=", max_abs, " rel_l2=", rel, " cos_sim=", cos_sim, " (n=", ref_n, ")")


def test_voice_encoder_from_wav() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var fix = "weights/s3gen_prompt/resample_diag/"

    print("[ve-wav] loading FCM + CAMPPlus...")
    var fcm = load_fcm(ctx, "weights/s3gen/speaker_encoder/head")
    var campplus = load_campplus(ctx, "weights/s3gen/speaker_encoder")

    # Step 1: load 24kHz wav, resample to 16kHz
    var wav_24 = upload_fp32(ctx, fix + "wav_24k.bin")
    var n_24 = 288000
    var n_16 = n_24 * 2 // 3   # 192000
    var wav_16 = ctx.enqueue_create_buffer[DType.float32](n_16)
    resample_24k_to_16k(ctx, wav_24, wav_16, n_24, n_16)
    ctx.synchronize()

    # Step 2: kaldi fbank
    var T_frames = (n_16 - 400) // 160 + 1
    print("[ve-wav] T_frames=", T_frames)

    var window = ctx.enqueue_create_buffer[DType.float32](400)
    build_povey_window(ctx, window, 400)
    var n_fft_bins = 512 // 2 + 1
    var mel_fb = ctx.enqueue_create_buffer[DType.float32](80 * n_fft_bins)
    build_kaldi_mel_filterbank(ctx, mel_fb, 80, 512, Float64(16000.0), Float64(20.0), Float64(0.0))

    var fbank_tf = ctx.enqueue_create_buffer[DType.float32](T_frames * 80)
    kaldi_fbank_forward(ctx, wav_16, window, mel_fb, fbank_tf, n_16, T_frames)
    kaldi_subtract_column_mean(ctx, fbank_tf, T_frames, 80)
    ctx.synchronize()

    # Step 3: transpose (T, 80) → (1, 80, T)
    var fbank_btf = ctx.enqueue_create_buffer[DType.float32](80 * T_frames)
    var fp = fbank_tf.unsafe_ptr()
    var bp = fbank_btf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(fp, bp, T_frames)
    def tr_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var f = i // T_frames
        var t = i - f * T_frames
        bp[f * T_frames + t] = fp[t * 80 + f]
    elementwise[tr_fn, simd_width=1, target="gpu"](
        IndexList[1](80 * T_frames), DeviceContextPtr(ctx),
    )

    # Step 4: CAMPPlus speaker encoder.
    var emb = ctx.enqueue_create_buffer[DType.float32](192)
    campplus_speaker_embedding(ctx, fcm, campplus.xvector, fbank_btf, emb, 1, T_frames)
    ctx.synchronize()

    _diff("speaker_emb (Mojo wav→resample→fbank→spk_enc)", emb, fix + "speaker_emb_from_wav.bin")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
