"""Test the full wav → CAMPPlus → spks (post-affine 80-d) path against upstream."""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.math import sqrt
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from fixture import load_fp32
from weights import load_fcm, load_campplus, upload_fp32
from modules import Linear, linear_forward
from resampler import resample_24k_to_16k
from kaldi_fbank import (
    kaldi_fbank_forward, kaldi_subtract_column_mean,
    build_povey_window, build_kaldi_mel_filterbank,
)
from campplus import campplus_speaker_embedding


def test_spks_from_wav() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    print("[spks] loading models...")
    var fcm = load_fcm(ctx, "weights/s3gen/speaker_encoder/head")
    var campplus = load_campplus(ctx, "weights/s3gen/speaker_encoder")

    var affine_w = upload_fp32(ctx, "weights/s3gen/flow/spk_embed_affine_layer/weight.bin")
    var affine_b = upload_fp32(ctx, "weights/s3gen/flow/spk_embed_affine_layer/bias.bin")
    var spk_embed_affine = Linear(affine_w^, affine_b^, 192, 80, True)

    var n_24 = 240000
    var n_16 = 160000
    var wav24_t = load_fp32("weights/s3gen_prompt/resample_diag/wav_24k.bin")
    var wav_24 = ctx.enqueue_create_buffer[DType.float32](n_24)
    with wav_24.map_to_host() as h:
        for i in range(n_24):
            h[i] = wav24_t.data[i]
    var wav_16 = ctx.enqueue_create_buffer[DType.float32](n_16)
    resample_24k_to_16k(ctx, wav_24, wav_16, n_24, n_16)
    ctx.synchronize()

    # Kaldi fbank.
    var T_fbank = (n_16 - 400) // 160 + 1
    print("[spks] T_fbank=", T_fbank)
    var fbank_win = ctx.enqueue_create_buffer[DType.float32](400)
    build_povey_window(ctx, fbank_win, 400)
    var fbank_mel_fb = ctx.enqueue_create_buffer[DType.float32](80 * (512 // 2 + 1))
    build_kaldi_mel_filterbank(ctx, fbank_mel_fb, 80, 512, Float64(16000.0),
                                Float64(20.0), Float64(0.0))
    var fbank_tf = ctx.enqueue_create_buffer[DType.float32](T_fbank * 80)
    kaldi_fbank_forward(ctx, wav_16, fbank_win, fbank_mel_fb, fbank_tf, n_16, T_fbank)
    kaldi_subtract_column_mean(ctx, fbank_tf, T_fbank, 80)
    ctx.synchronize()

    # Transpose (T, 80) → (1, 80, T).
    var fbank_btf = ctx.enqueue_create_buffer[DType.float32](80 * T_fbank)
    var fp = fbank_tf.unsafe_ptr()
    var bp = fbank_btf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(fp, bp, T_fbank)
    def tr_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var f = i // T_fbank
        var t = i - f * T_fbank
        bp[f * T_fbank + t] = fp[t * 80 + f]
    elementwise[tr_fn, simd_width=1, target="gpu"](
        IndexList[1](80 * T_fbank), DeviceContextPtr(ctx),
    )

    var emb_192 = ctx.enqueue_create_buffer[DType.float32](192)
    campplus_speaker_embedding(ctx, fcm, campplus.xvector, fbank_btf, emb_192, 1, T_fbank)
    ctx.synchronize()

    # F.normalize + spk_embed_affine.
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
    linear_forward(ctx, spk_embed_affine, emb_norm, spks, 1)
    ctx.synchronize()

    # Compare to upstream.
    var reference = load_fp32("weights/s3gen_prompt/embedding_normed_affine.bin")
    var max_abs: Float32 = 0.0
    var sum_diff_sq: Float32 = 0.0
    var sum_ref_sq: Float32 = 0.0
    var dot: Float32 = 0.0
    var an: Float32 = 0.0
    var bn: Float32 = 0.0
    with spks.map_to_host() as h:
        for i in range(80):
            var dd = h[i] - reference.data[i]
            var d = dd
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_diff_sq += dd * dd
            sum_ref_sq += reference.data[i] * reference.data[i]
            dot += h[i] * reference.data[i]
            an += h[i] * h[i]
            bn += reference.data[i] * reference.data[i]
    var rel = sqrt(sum_diff_sq / sum_ref_sq) if sum_ref_sq > 0.0 else Float32(0.0)
    var cos_sim = dot / (sqrt(an) * sqrt(bn))
    print("[spks] max-abs=", max_abs, " rel_l2=", rel, " cos_sim=", cos_sim)
    print("[spks] Mojo norm=", sqrt(an), " upstream norm=", sqrt(bn))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
