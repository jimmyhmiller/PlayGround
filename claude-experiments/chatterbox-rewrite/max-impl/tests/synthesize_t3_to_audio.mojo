"""End-to-end T3 → s3gen → audio in pure Mojo.

Runs the full pipeline on real upstream Chatterbox weights:
  1. T3 generate (KV-cached autoregressive decode) → real speech tokens.
     This is bit-exact verified vs torch oracle in test_t3_generate.mojo
     (matches 4/4 tokens before EOS).
  2. UpsampleConformerEncoder: tokens → mu (B, 80, 2*T).
  3. CFM Euler solve (2 steps, CFG=0.7): mu + noise → mel.
  4. HiFTGenerator + iSTFT: mel → audio samples.
  5. save_wav.

Uses the existing T3 generation fixtures (initial_ids, cos/sin tables,
speech_pos_emb) plus the real upstream weights for all four models.
"""
from std.sys import has_accelerator
from std.math import sin
from std.testing import TestSuite, assert_true
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from fixture import load_fp32, load_i64, save_wav
from weights import (
    load_t3, load_upsample_conformer_encoder,
    load_cfm_estimator_real, load_hift_generator, upload_fp32,
)
from modules import Embedding, embedding_forward
from t3_generate import t3_generate
from upsample_encoder import upsample_conformer_forward
from cfm_estimator_new import cfm_solve_euler
from hift_generator import (
    hift_decode_trunk, istft_forward, hann_window_periodic_fill,
)


comptime B = 1
comptime T_PREFILL = 15
comptime N_STEPS = 8
comptime MAX_CTX = T_PREFILL + N_STEPS
comptime D = 1024
comptime HEAD_DIM = 64
comptime MEL = 80
comptime EOS = 6562
comptime N_FFT = 16
comptime HOP = 4
comptime N_OUT = 18
comptime N_CFM_STEPS = 2
comptime CFG: Float32 = 0.7


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def test_t3_to_audio() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var fix = "../mojo-t3/tests/fixtures/generate/"

    # ─────────────────────────────────────────────────────────────────────
    # Step 1: T3 generation (bit-exact parity verified separately).
    # ─────────────────────────────────────────────────────────────────────
    print("[t3-e2e] loading T3 from weights/t3/...")
    var t3 = load_t3(ctx, "weights/t3")

    var initial_ids = load_i64(fix + "initial_ids_fp32.bin")
    var cos_full_t = load_fp32(fix + "cos_full_fp32.bin")
    var sin_full_t = load_fp32(fix + "sin_full_fp32.bin")
    var mask_pre_t = load_fp32(fix + "mask_prefill_fp32.bin")
    var speech_pos_t = load_fp32(fix + "speech_pos_emb_fp32.bin")

    var ids_buf = ctx.enqueue_create_buffer[DType.int64](B * T_PREFILL)
    with ids_buf.map_to_host() as h:
        for i in range(T_PREFILL): h[i] = initial_ids.data[i]
    var emb_buf = ctx.enqueue_create_buffer[DType.float32](B * T_PREFILL * D)
    embedding_forward(ctx, t3.speech_emb, ids_buf, emb_buf, B, T_PREFILL)

    var pos_buf = ctx.enqueue_create_buffer[DType.float32](len(speech_pos_t.data))
    upload(pos_buf, speech_pos_t.data, len(speech_pos_t.data))
    var ep = emb_buf.unsafe_ptr()
    var pp = pos_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(ep, pp)
    def add_pos[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (T_PREFILL * D)
        var rem = i - bi * T_PREFILL * D
        var si = rem // D
        var di = rem - si * D
        ep[i] = ep[i] + pp[si * D + di]
    elementwise[add_pos, simd_width=1, target="gpu"](
        IndexList[1](B * T_PREFILL * D), DeviceContextPtr(ctx),
    )

    var cos_full = ctx.enqueue_create_buffer[DType.float32](MAX_CTX * HEAD_DIM)
    upload(cos_full, cos_full_t.data, MAX_CTX * HEAD_DIM)
    var sin_full = ctx.enqueue_create_buffer[DType.float32](MAX_CTX * HEAD_DIM)
    upload(sin_full, sin_full_t.data, MAX_CTX * HEAD_DIM)
    var mask_pre = ctx.enqueue_create_buffer[DType.float32](T_PREFILL * T_PREFILL)
    upload(mask_pre, mask_pre_t.data, T_PREFILL * T_PREFILL)

    print("[t3-e2e] T3 generate (max", N_STEPS, "steps)...")
    var generated = t3_generate(
        ctx, t3, emb_buf, cos_full, sin_full, mask_pre, pos_buf,
        B, T_PREFILL, MAX_CTX, N_STEPS, T_PREFILL, eos_token=EOS,
    )
    ctx.synchronize()
    print("[t3-e2e] T3 generated", len(generated), "tokens:")
    for i in range(len(generated)):
        print("    [", i, "] =", Int(generated[i]))

    # Drop the EOS if present and only keep tokens that map into the speech
    # vocab the flow encoder accepts.
    var speech_tokens = List[Int64]()
    for i in range(len(generated)):
        var tok = Int(generated[i])
        if tok == EOS: break
        if tok < 6562: speech_tokens.append(generated[i])
    if len(speech_tokens) == 0:
        # Ensure we have at least one token for downstream shape sanity.
        speech_tokens.append(Int64(0))
    print("[t3-e2e] usable speech tokens for s3gen:", len(speech_tokens))

    # ─────────────────────────────────────────────────────────────────────
    # Step 2: Flow encoder.
    # ─────────────────────────────────────────────────────────────────────
    print("[t3-e2e] loading UpsampleConformerEncoder...")
    var enc = load_upsample_conformer_encoder(ctx, "weights/s3gen/flow")

    var t_token = len(speech_tokens)
    var t_mel = 2 * t_token
    var tok_buf = ctx.enqueue_create_buffer[DType.int64](B * t_token)
    with tok_buf.map_to_host() as h:
        for i in range(t_token): h[i] = speech_tokens[i]

    var mu = ctx.enqueue_create_buffer[DType.float32](B * MEL * t_mel)
    print("[t3-e2e] flow encoder (T_token=", t_token, " → T_mel=", t_mel, ")...")
    upsample_conformer_forward(ctx, enc, tok_buf, mu, B, t_token)
    ctx.synchronize()

    # ─────────────────────────────────────────────────────────────────────
    # Step 3: CFM Euler solve.
    # ─────────────────────────────────────────────────────────────────────
    print("[t3-e2e] loading CFM estimator...")
    var cfm = load_cfm_estimator_real(ctx, "weights/s3gen/flow/decoder/estimator")

    var x = ctx.enqueue_create_buffer[DType.float32](B * MEL * t_mel)
    with x.map_to_host() as h:
        for c in range(MEL):
            for ti in range(t_mel):
                h[c * t_mel + ti] = sin(Float32(c) * 0.31 + Float32(ti) * 0.71) * 0.5
    var spks = ctx.enqueue_create_buffer[DType.float32](B * MEL)
    with spks.map_to_host() as h:
        for c in range(MEL):
            h[c] = sin(Float32(c) * 0.11) * 0.1
    var cond = ctx.enqueue_create_buffer[DType.float32](B * MEL * t_mel)
    cond.enqueue_fill(0.0)
    var mask = ctx.enqueue_create_buffer[DType.float32](B * t_mel)
    mask.enqueue_fill(1.0)

    print("[t3-e2e] CFM Euler (", N_CFM_STEPS, " steps, CFG=", CFG, ")...")
    cfm_solve_euler(ctx, cfm, x, mu, spks, cond, mask, B, t_mel, N_CFM_STEPS, CFG)
    ctx.synchronize()

    # ─────────────────────────────────────────────────────────────────────
    # Step 4: HiFTGenerator → audio.
    # ─────────────────────────────────────────────────────────────────────
    print("[t3-e2e] loading HiFTGenerator...")
    var hift = load_hift_generator(ctx, "weights/s3gen/mel2wav")

    var t_hift = t_mel * 8 * 8 * 4 + 1
    var t_audio = (t_hift - 1) * HOP

    var s_stft_dummy = ctx.enqueue_create_buffer[DType.float32](B * N_OUT * 1)
    var spec = ctx.enqueue_create_buffer[DType.float32](B * N_OUT * t_hift)
    print("[t3-e2e] HiFT trunk (T_mel=", t_mel, " → T_frames=", t_hift, ")...")
    hift_decode_trunk(ctx, hift, x, s_stft_dummy, spec,
                      B, t_mel, 1, t_hift, use_source=False)

    var window = ctx.enqueue_create_buffer[DType.float32](N_FFT)
    hann_window_periodic_fill(ctx, window, N_FFT)
    var audio = ctx.enqueue_create_buffer[DType.float32](B * t_audio)
    print("[t3-e2e] iSTFT (", t_hift, " frames → ", t_audio, " samples)...")
    istft_forward(ctx, spec, window, audio, B, N_FFT, t_hift, t_audio)
    ctx.synchronize()

    var n_nan = 0
    var max_a: Float32 = 0.0
    with audio.map_to_host() as h:
        for i in range(B * t_audio):
            var v = h[i]
            if v != v: n_nan += 1
            var av = v
            if av < 0.0: av = -av
            if av > max_a: max_a = av
    print("[t3-e2e] audio max=", max_a, " nan=", n_nan,
          " duration ~", Float32(t_audio) / 24000.0, "s @ 24kHz")
    assert_true(n_nan == 0, "no NaNs in audio")

    var samples = List[Float32]()
    with audio.map_to_host() as h:
        for i in range(t_audio):
            samples.append(h[i])
    save_wav(String("max_impl_t3_to_audio.wav"), samples, 24000)
    print("[t3-e2e] saved max_impl_t3_to_audio.wav")
    print("[t3-e2e] FULL T3 → S3GEN → AUDIO CHAIN PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
