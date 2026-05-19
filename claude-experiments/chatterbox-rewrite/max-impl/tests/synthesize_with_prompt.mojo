"""End-to-end synthesis using upstream-dumped prompt prefix.

This test isolates the s3gen chain (flow encoder + CFM + HiFT) by feeding
it the EXACT prompt_token, prompt_feat, embedding, and speech_tokens that
upstream Chatterbox computes for the same text + ref voice. The goal: prove
that with the correct prompt-side inputs, our pure-Mojo s3gen chain produces
audio that sounds like upstream's.

Inputs (all dumped from upstream via scripts/dump_s3gen_prompt_oracle.py):
  weights/s3gen_prompt/prompt_token.bin             (1, 250)  int64
  weights/s3gen_prompt/prompt_feat.bin              (1, 500, 80)
  weights/s3gen_prompt/embedding_normed_affine.bin  (1, 80)   spks for CFM
  weights/s3gen_prompt/speech_tokens.bin            (39,)     int64
  weights/s3gen_prompt/expected_mel.bin             (1, 80, 78) (for parity)
"""
from std.sys import has_accelerator
from std.math import sin, cos as mcos, pi
from std.testing import TestSuite, assert_true
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from fixture import load_fp32, load_i64, save_wav
from weights import (
    load_upsample_conformer_encoder, load_cfm_estimator_real,
    load_hift_generator, upload_fp32,
)
from upsample_encoder import upsample_conformer_forward
from cfm_estimator_new import cfm_solve_euler
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


def fill_t_span_cosine(
    mut ctx: DeviceContext,
    mut buf: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    """Fill buf[0..n] with `1 - cos(linspace(0,1,n) * pi/2)`."""
    var p = buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(p, n)
    def cosine_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var t: Float32 = Float32(i) / Float32(n - 1)
        p[i] = 1.0 - mcos(t * Float32(pi) * 0.5)
    elementwise[cosine_fn, simd_width=1, target="gpu"](
        IndexList[1](n), DeviceContextPtr(ctx),
    )


def test_synth_with_prompt() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var fix = "weights/s3gen_prompt/"

    print("[synth] loading models...")
    var enc = load_upsample_conformer_encoder(ctx, "weights/s3gen/flow")
    var cfm = load_cfm_estimator_real(ctx, "weights/s3gen/flow/decoder/estimator")
    var hift = load_hift_generator(ctx, "weights/s3gen/mel2wav")

    # ─────────────────────────────────────────────────────────────────────
    # 1. Build token = cat(prompt_token, speech_tokens) — sizes derived from dump.
    # ─────────────────────────────────────────────────────────────────────
    var prompt_tok = load_i64(fix + "prompt_token.bin")
    var speech_tok = load_i64(fix + "speech_tokens.bin")
    var T_GEN_TOKEN = speech_tok.numel()
    var T_TOTAL_TOKEN = T_PROMPT_TOKEN + T_GEN_TOKEN
    var T_TOTAL_MEL = 2 * T_TOTAL_TOKEN
    var T_OUT_MEL = T_TOTAL_MEL - T_PROMPT_MEL
    var tok_buf = ctx.enqueue_create_buffer[DType.int64](B * T_TOTAL_TOKEN)
    with tok_buf.map_to_host() as h:
        for i in range(T_PROMPT_TOKEN):
            h[i] = prompt_tok.data[i]
        for i in range(T_GEN_TOKEN):
            h[T_PROMPT_TOKEN + i] = speech_tok.data[i]
    print("[synth] T_token=", T_TOTAL_TOKEN, " (prompt=", T_PROMPT_TOKEN, " + gen=", T_GEN_TOKEN, ")")

    # ─────────────────────────────────────────────────────────────────────
    # 2. Flow encoder → mu (B, MEL, T_TOTAL_MEL).
    # ─────────────────────────────────────────────────────────────────────
    print("[synth] flow encoder (T_token=", T_TOTAL_TOKEN, " → T_mel=", T_TOTAL_MEL, ")...")
    var mu = ctx.enqueue_create_buffer[DType.float32](B * MEL * T_TOTAL_MEL)
    upsample_conformer_forward(ctx, enc, tok_buf, mu, B, T_TOTAL_TOKEN)
    ctx.synchronize()

    # ─────────────────────────────────────────────────────────────────────
    # 3. Build cond (B, MEL, T_TOTAL_MEL) — zeros, then slot prompt_feat (B, T_PROMPT_MEL, MEL) into [:, :, :T_PROMPT_MEL].
    # ─────────────────────────────────────────────────────────────────────
    var prompt_feat = upload_fp32(ctx, fix + "prompt_feat.bin")
    var cond = ctx.enqueue_create_buffer[DType.float32](B * MEL * T_TOTAL_MEL)
    cond.enqueue_fill(0.0)
    # prompt_feat shape (1, 500, 80). cond is (B, 80, 578).
    # cond[b, c, t] = prompt_feat[b, t, c] for t in [0, 500), c in [0, 80).
    var pf_ptr = prompt_feat.unsafe_ptr()
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
        # Destination: cond[bi, ci, ti]
        cond_ptr[bi * MEL * T_TOTAL_MEL + ci * T_TOTAL_MEL + ti] = pf_ptr[i]
    elementwise[cond_fill, simd_width=1, target="gpu"](
        IndexList[1](B * T_PROMPT_MEL * MEL), DeviceContextPtr(ctx),
    )

    # ─────────────────────────────────────────────────────────────────────
    # 4. Load real spks (post-affine) and the exact noise z upstream used.
    # ─────────────────────────────────────────────────────────────────────
    var spks = upload_fp32(ctx, fix + "embedding_normed_affine.bin")
    var x = upload_fp32(ctx, fix + "cfm_noise_z.bin")

    var mask = ctx.enqueue_create_buffer[DType.float32](B * T_TOTAL_MEL)
    mask.enqueue_fill(1.0)

    # ─────────────────────────────────────────────────────────────────────
    # 5. CFM Euler solve. NOTE: upstream uses cosine t_span; our cfm_solve_euler
    # uses linear. We pre-compute the cosine t_span and adapt our solver to use it.
    # For now, call the existing linear solver. Real fix is to pass t_span to solver.
    # ─────────────────────────────────────────────────────────────────────
    print("[synth] CFM Euler (", N_CFM_STEPS, " steps, CFG=", CFG, ")...")
    cfm_solve_euler(ctx, cfm, x, mu, spks, cond, mask, B, T_TOTAL_MEL, N_CFM_STEPS, CFG)
    ctx.synchronize()

    # x now holds (B, 80, T_TOTAL_MEL=578). Trim prompt prefix → (B, 80, T_OUT_MEL=78).
    print("[synth] trimming prompt prefix → T_out_mel=", T_OUT_MEL, "...")
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

    # Compare against upstream mel.
    var expected_mel = load_fp32(fix + "expected_mel.bin")
    var max_abs: Float32 = 0.0
    var sum_diff_sq: Float32 = 0.0
    var sum_ref_sq: Float32 = 0.0
    with mel_out.map_to_host() as h:
        for i in range(B * MEL * T_OUT_MEL):
            var dd = h[i] - expected_mel.data[i]
            var d = dd
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_diff_sq += dd * dd
            sum_ref_sq += expected_mel.data[i] * expected_mel.data[i]
    from std.math import sqrt
    var rel_l2 = sqrt(sum_diff_sq / sum_ref_sq) if sum_ref_sq > 0.0 else Float32(0.0)
    print("[synth] mel parity vs upstream: max-abs=", max_abs, " rel_l2=", rel_l2)

    # ─────────────────────────────────────────────────────────────────────
    # 6. HiFT: mel → f0 → source → STFT → trunk → iSTFT → audio.
    # ─────────────────────────────────────────────────────────────────────
    var T_HIFT = T_OUT_MEL * 120 + 1
    var T_AUDIO = (T_HIFT - 1) * HOP
    var T_AUDIO_FULL = T_OUT_MEL * 480
    var T_S_FRAMES = T_AUDIO_FULL // HOP + 1

    print("[synth] f0_predictor...")
    var f0 = ctx.enqueue_create_buffer[DType.float32](B * T_OUT_MEL)
    f0_predictor_forward(ctx, hift.f0_predictor, mel_out, f0, B, T_OUT_MEL)
    var f0_up = ctx.enqueue_create_buffer[DType.float32](B * T_AUDIO_FULL)
    f0_upsample_nearest(ctx, f0, f0_up, B, T_OUT_MEL, 480)

    print("[synth] source module (using upstream's sine_merge to bypass RNG divergence)...")
    var sine_merge = upload_fp32(ctx, fix + "hift_dump/sine_merge.bin")

    print("[synth] STFT...")
    var window_s = ctx.enqueue_create_buffer[DType.float32](N_FFT)
    hann_window_periodic_fill(ctx, window_s, N_FFT)
    var s_stft = ctx.enqueue_create_buffer[DType.float32](B * N_OUT * T_S_FRAMES)
    build_s_stft_from_signal(ctx, sine_merge, window_s, s_stft,
                              B, T_AUDIO_FULL, N_FFT, HOP, T_S_FRAMES)

    print("[synth] HiFT trunk + iSTFT (T_mel=", T_OUT_MEL, " → T_frames=", T_HIFT, ")...")
    var spec = ctx.enqueue_create_buffer[DType.float32](B * N_OUT * T_HIFT)
    hift_decode_trunk(ctx, hift, mel_out, s_stft, spec,
                      B, T_OUT_MEL, T_S_FRAMES, T_HIFT, use_source=True)

    var window_i = ctx.enqueue_create_buffer[DType.float32](N_FFT)
    hann_window_periodic_fill(ctx, window_i, N_FFT)
    var audio = ctx.enqueue_create_buffer[DType.float32](B * T_AUDIO)
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
    print("[synth] audio max=", max_a, " nan=", n_nan,
          " duration ~", Float32(T_AUDIO) / 24000.0, "s")
    assert_true(n_nan == 0, "no NaNs")

    var samples = List[Float32]()
    with audio.map_to_host() as h:
        for i in range(T_AUDIO):
            samples.append(h[i])
    save_wav(String("max_impl_with_prompt.wav"), samples, 24000)
    print("[synth] saved max_impl_with_prompt.wav")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
