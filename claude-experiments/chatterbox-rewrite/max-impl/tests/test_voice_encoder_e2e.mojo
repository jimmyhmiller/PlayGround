"""End-to-end voice encoder test: raw 16kHz waveform → 192-d speaker embedding.
Pipeline: kaldi_fbank → subtract column mean → FCM → xvector backbone."""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.math import sqrt
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from fixture import load_fp32
from weights import load_fcm, load_campplus, upload_fp32
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
    with mojo.map_to_host() as h:
        for i in range(ref_n):
            var dd = h[i] - reference.data[i]
            var d = dd
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_diff_sq += dd * dd
            sum_ref_sq += reference.data[i] * reference.data[i]
    var rel = sqrt(sum_diff_sq / sum_ref_sq) if sum_ref_sq > 0.0 else Float32(0.0)
    print("[ve-e2e]", name, ": max-abs=", max_abs, " rel_l2=", rel, " (n=", ref_n, ")")


def test_voice_encoder_e2e() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var fix = "weights/s3gen_prompt/kaldi_diag/"

    print("[ve-e2e] loading FCM + CAMPPlus...")
    var fcm = load_fcm(ctx, "weights/s3gen/speaker_encoder/head")
    var campplus = load_campplus(ctx, "weights/s3gen/speaker_encoder")

    var n_samples = 16240
    var T = (n_samples - 400) // 160 + 1   # 100

    # Load raw waveform.
    var wav = upload_fp32(ctx, fix + "wav.bin")

    # Build povey window + kaldi mel filterbank.
    var window = ctx.enqueue_create_buffer[DType.float32](400)
    build_povey_window(ctx, window, 400)
    var n_fft_bins = 512 // 2 + 1
    var mel_fb = ctx.enqueue_create_buffer[DType.float32](80 * n_fft_bins)
    build_kaldi_mel_filterbank(ctx, mel_fb, 80, 512, Float64(16000.0), Float64(20.0), Float64(0.0))

    # Kaldi fbank → (T, 80).
    var fbank_tf = ctx.enqueue_create_buffer[DType.float32](T * 80)
    kaldi_fbank_forward(ctx, wav, window, mel_fb, fbank_tf, n_samples, T)
    kaldi_subtract_column_mean(ctx, fbank_tf, T, 80)
    ctx.synchronize()

    # Transpose (T, 80) → (B=1, 80, T) for FCM input.
    var fbank_btf = ctx.enqueue_create_buffer[DType.float32](80 * T)
    var fp = fbank_tf.unsafe_ptr()
    var bp = fbank_btf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(fp, bp, T)
    def tr_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var f = i // T   # feature index in (80, T)
        var t = i - f * T
        bp[f * T + t] = fp[t * 80 + f]
    elementwise[tr_fn, simd_width=1, target="gpu"](
        IndexList[1](80 * T), DeviceContextPtr(ctx),
    )

    var emb = ctx.enqueue_create_buffer[DType.float32](192)
    print("[ve-e2e] running campplus speaker embedding...")
    campplus_speaker_embedding(ctx, fcm, campplus.xvector, fbank_btf, emb, 1, T)
    ctx.synchronize()

    _diff("speaker_emb_e2e", emb, fix + "speaker_emb_e2e.bin")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
