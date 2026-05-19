"""End-to-end Mojo: 16kHz wav → log_mel → s3tokenizer tokens."""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.gpu.host import DeviceContext, DeviceBuffer

from fixture import load_fp32, load_i64
from weights import load_s3tokenizer, upload_fp32
from mel_s3tok import (
    log_mel_s3tok_forward, build_hann_window_full, build_librosa_mel_filterbank_s3tok,
)
from s3tokenizer import s3tokenizer_forward
from text_embed import build_rope_tables


def test_s3tokenizer_from_wav() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var fix = "weights/s3gen_prompt/s3tok_diag/"

    print("[s3tok-wav] loading s3tokenizer...")
    var s3t = load_s3tokenizer(ctx, "weights/s3t")

    # Step 1: load 16kHz wav.
    var wav = upload_fp32(ctx, fix + "wav_16k.bin")
    var n_samples = 192000
    var T_mel = 1200

    # Step 2: log-mel.
    var window = ctx.enqueue_create_buffer[DType.float32](400)
    build_hann_window_full(ctx, window, 400)
    var mel_fb = ctx.enqueue_create_buffer[DType.float32](128 * 201)
    build_librosa_mel_filterbank_s3tok(ctx, mel_fb, 128, 400, Float64(16000.0))

    var log_mel = ctx.enqueue_create_buffer[DType.float32](128 * T_mel)
    log_mel_s3tok_forward(ctx, wav, window, mel_fb, log_mel, n_samples, T_mel)
    ctx.synchronize()

    # Step 3: s3tokenizer.
    var T_token = 300
    var head_dim = 64
    var max_ctx = 4096
    var cos_buf = ctx.enqueue_create_buffer[DType.float32](max_ctx * head_dim)
    var sin_buf = ctx.enqueue_create_buffer[DType.float32](max_ctx * head_dim)
    build_rope_tables(ctx, max_ctx, head_dim, cos_buf, sin_buf)

    var B = 1
    var mask_pad = ctx.enqueue_create_buffer[DType.float32](B * T_token * 1)
    mask_pad.enqueue_fill(1.0)
    var attn_mask = ctx.enqueue_create_buffer[DType.float32](T_token * T_token)
    attn_mask.enqueue_fill(0.0)

    var tokens = ctx.enqueue_create_buffer[DType.int32](B * T_token)
    s3tokenizer_forward(ctx, s3t, log_mel, tokens, cos_buf, sin_buf,
                          mask_pad, attn_mask, B, T_mel)
    ctx.synchronize()

    # Compare against upstream tokens.
    var ref_tokens = load_i64(fix + "tokens.bin")
    var n_match = 0
    with tokens.map_to_host() as h:
        for i in range(20):
            print("  [", i, "] mojo=", h[i], " ref=", ref_tokens.data[i])
        for i in range(T_token):
            if Int64(h[i]) == ref_tokens.data[i]:
                n_match += 1
    print("[s3tok-wav] token agreement:", n_match, "/", T_token, " (",
          Float32(n_match) / Float32(T_token) * 100.0, "%)")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
