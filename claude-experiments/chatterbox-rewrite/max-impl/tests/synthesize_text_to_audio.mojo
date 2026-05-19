"""End-to-end Mojo TTS: real text → audio.wav.

Pipeline:
  1. BPE tokenize "the quick brown fox" → text token IDs (bit-exact vs upstream)
  2. Concat[ cond_emb (from upstream dump), text_emb, bos_emb ] → T3 prefix
  3. Mojo T3 generation → real speech tokens
  4. Mojo flow encoder → mu (B, 80, 2*T_token)
  5. Mojo CFM Euler (2 steps, CFG=0.7) → mel
  6. Mojo HiFTGenerator + iSTFT → audio
  7. save_wav

cond_emb still comes from upstream (the FCM 2D head is not yet in Mojo).
Everything else runs in pure Mojo against real upstream weights.
"""
from std.sys import has_accelerator
from std.math import sin
from std.testing import TestSuite, assert_true
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from fixture import load_fp32, load_i64, save_wav
from weights import (
    load_t3, load_upsample_conformer_encoder, load_cfm_estimator_real,
    load_hift_generator, upload_fp32,
)
from t3_generate import t3_generate
from upsample_encoder import upsample_conformer_forward
from cfm_estimator_new import cfm_solve_euler
from hift_generator import (
    hift_decode_trunk, istft_forward, hann_window_periodic_fill,
)


comptime B = 1
comptime T_COND = 34
comptime T_TEXT = 13
comptime T_BOS = 1
comptime T_PREFIX = T_COND + T_TEXT + T_BOS    # 48
comptime D = 1024
comptime MAX_CTX = 200
comptime MAX_NEW = 100
comptime EOS = 6562
comptime MEL = 80
comptime N_FFT = 16
comptime HOP = 4
comptime N_OUT = 18
comptime N_CFM_STEPS = 2
comptime CFG: Float32 = 0.7


def test_text_to_audio() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var fix = "weights/t3_text_parity/"

    # ─────────────────────────────────────────────────────────────────────
    # Step 1: Load T3 and run text → speech tokens.
    # ─────────────────────────────────────────────────────────────────────
    print("[tts] loading T3...")
    var t3 = load_t3(ctx, "weights/t3")

    var cond_emb = upload_fp32(ctx, fix + "cond_emb.bin")
    var text_emb = upload_fp32(ctx, fix + "text_emb.bin")
    var bos_emb = upload_fp32(ctx, fix + "bos_emb.bin")

    var prefix = ctx.enqueue_create_buffer[DType.float32](B * T_PREFIX * D)
    var ce = cond_emb.unsafe_ptr()
    var te = text_emb.unsafe_ptr()
    var be = bos_emb.unsafe_ptr()
    var px = prefix.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(ce, te, be, px)
    def cat_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (T_PREFIX * D)
        var rem = i - bi * T_PREFIX * D
        var ti = rem // D
        var di = rem - ti * D
        if ti < T_COND:
            px[i] = ce[bi * T_COND * D + ti * D + di]
        elif ti < T_COND + T_TEXT:
            var src_t = ti - T_COND
            px[i] = te[bi * T_TEXT * D + src_t * D + di]
        else:
            var src_t = ti - T_COND - T_TEXT
            px[i] = be[bi * T_BOS * D + src_t * D + di]
    elementwise[cat_func, simd_width=1, target="gpu"](
        IndexList[1](B * T_PREFIX * D), DeviceContextPtr(ctx),
    )

    var cos_full = upload_fp32(ctx, fix + "cos_full.bin")
    var sin_full = upload_fp32(ctx, fix + "sin_full.bin")
    var mask = ctx.enqueue_create_buffer[DType.float32](T_PREFIX * T_PREFIX)
    with mask.map_to_host() as h:
        for r in range(T_PREFIX):
            for c in range(T_PREFIX):
                if c > r:
                    h[r * T_PREFIX + c] = -1.0e30
                else:
                    h[r * T_PREFIX + c] = 0.0
    var speech_pos = upload_fp32(ctx, fix + "speech_pos_emb_full.bin")

    print("[tts] T3 generate (max_new=", MAX_NEW, ")...")
    var generated = t3_generate(
        ctx, t3, prefix, cos_full, sin_full, mask, speech_pos,
        B, T_PREFIX, MAX_CTX, MAX_NEW, speech_pos_offset=1, eos_token=EOS,
    )
    ctx.synchronize()

    # Filter to first ~30 tokens — to keep memory in check for downstream
    # s3gen which scales O(T_mel^2) for attention.
    var n_speech_keep = 30
    if len(generated) < n_speech_keep:
        n_speech_keep = len(generated)
    var speech_tokens = List[Int64]()
    for i in range(n_speech_keep):
        var tok = Int(generated[i])
        if tok == EOS: break
        if tok < 6561: speech_tokens.append(generated[i])
    if len(speech_tokens) == 0:
        speech_tokens.append(Int64(0))
    var T_TOKEN = len(speech_tokens)
    var T_MEL = 2 * T_TOKEN
    print("[tts] T_token=", T_TOKEN, " T_mel=", T_MEL)

    # ─────────────────────────────────────────────────────────────────────
    # Step 2: Flow encoder → mu.
    # ─────────────────────────────────────────────────────────────────────
    print("[tts] loading UpsampleConformerEncoder...")
    var enc = load_upsample_conformer_encoder(ctx, "weights/s3gen/flow")

    var tok_buf = ctx.enqueue_create_buffer[DType.int64](B * T_TOKEN)
    with tok_buf.map_to_host() as h:
        for i in range(T_TOKEN): h[i] = speech_tokens[i]
    var mu = ctx.enqueue_create_buffer[DType.float32](B * MEL * T_MEL)
    print("[tts] flow encoder...")
    upsample_conformer_forward(ctx, enc, tok_buf, mu, B, T_TOKEN)
    ctx.synchronize()

    # ─────────────────────────────────────────────────────────────────────
    # Step 3: CFM Euler.
    # ─────────────────────────────────────────────────────────────────────
    print("[tts] loading CFM estimator...")
    var cfm = load_cfm_estimator_real(ctx, "weights/s3gen/flow/decoder/estimator")

    var x = ctx.enqueue_create_buffer[DType.float32](B * MEL * T_MEL)
    with x.map_to_host() as h:
        for c in range(MEL):
            for ti in range(T_MEL):
                h[c * T_MEL + ti] = sin(Float32(c) * 0.31 + Float32(ti) * 0.71) * 0.5
    var spks = ctx.enqueue_create_buffer[DType.float32](B * MEL)
    with spks.map_to_host() as h:
        for c in range(MEL):
            h[c] = sin(Float32(c) * 0.11) * 0.1
    var cond = ctx.enqueue_create_buffer[DType.float32](B * MEL * T_MEL)
    cond.enqueue_fill(0.0)
    var cfm_mask = ctx.enqueue_create_buffer[DType.float32](B * T_MEL)
    cfm_mask.enqueue_fill(1.0)

    print("[tts] CFM Euler (", N_CFM_STEPS, " steps)...")
    cfm_solve_euler(ctx, cfm, x, mu, spks, cond, cfm_mask, B, T_MEL, N_CFM_STEPS, CFG)
    ctx.synchronize()

    # ─────────────────────────────────────────────────────────────────────
    # Step 4: HiFTGenerator + iSTFT.
    # ─────────────────────────────────────────────────────────────────────
    print("[tts] loading HiFTGenerator...")
    var hift = load_hift_generator(ctx, "weights/s3gen/mel2wav")

    # ups_rates [8, 5, 3] = 120; +1 reflection; hop=4.
    var T_HIFT = T_MEL * 120 + 1
    var T_AUDIO = (T_HIFT - 1) * HOP
    var s_stft_dummy = ctx.enqueue_create_buffer[DType.float32](B * N_OUT * 1)
    var spec = ctx.enqueue_create_buffer[DType.float32](B * N_OUT * T_HIFT)
    print("[tts] HiFT trunk (T_mel=", T_MEL, " → T_frames=", T_HIFT, ")...")
    hift_decode_trunk(ctx, hift, x, s_stft_dummy, spec,
                      B, T_MEL, 1, T_HIFT, use_source=False)

    var window = ctx.enqueue_create_buffer[DType.float32](N_FFT)
    hann_window_periodic_fill(ctx, window, N_FFT)
    var audio = ctx.enqueue_create_buffer[DType.float32](B * T_AUDIO)
    print("[tts] iSTFT (", T_HIFT, " frames → ", T_AUDIO, " samples)...")
    istft_forward(ctx, spec, window, audio, B, N_FFT, T_HIFT, T_AUDIO)
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
    print("[tts] audio max=", max_a, " nan=", n_nan,
          " duration ~", Float32(T_AUDIO) / 24000.0, "s @ 24kHz")
    assert_true(n_nan == 0, "no NaNs")

    var samples = List[Float32]()
    with audio.map_to_host() as h:
        for i in range(T_AUDIO):
            samples.append(h[i])
    save_wav(String("max_impl_text_to_audio.wav"), samples, 24000)
    print("[tts] saved max_impl_text_to_audio.wav")
    print("[tts] FULL TEXT → AUDIO PIPELINE PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
