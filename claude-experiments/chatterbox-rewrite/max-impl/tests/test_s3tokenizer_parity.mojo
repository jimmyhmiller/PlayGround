"""S3Tokenizer parity test: feed upstream's exact log_mel, compare tokens."""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.gpu.host import DeviceContext, DeviceBuffer

from fixture import load_fp32, load_i64
from weights import load_s3tokenizer, upload_fp32
from s3tokenizer import s3tokenizer_forward


def test_s3tokenizer_parity() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var fix = "weights/s3gen_prompt/s3tok_diag/"

    print("[s3tok] loading...")
    var s3t = load_s3tokenizer(ctx, "weights/s3t")

    var log_mel = upload_fp32(ctx, fix + "log_mel_16k.bin")
    var T_mel = 1200
    var B = 1
    var n_mels = 128

    # T_mel → conv1 stride=2 → 600 → conv2 stride=2 → 300 tokens.
    var T_token = 300

    # RoPE buffers (use existing build_rope_tables).
    var head_dim = 64
    var max_ctx = 4096
    var cos_buf = ctx.enqueue_create_buffer[DType.float32](max_ctx * head_dim)
    var sin_buf = ctx.enqueue_create_buffer[DType.float32](max_ctx * head_dim)
    from text_embed import build_rope_tables
    build_rope_tables(ctx, max_ctx, head_dim, cos_buf, sin_buf)

    var mask_pad = ctx.enqueue_create_buffer[DType.float32](B * T_token * 1)
    mask_pad.enqueue_fill(1.0)
    var attn_mask = ctx.enqueue_create_buffer[DType.float32](T_token * T_token)
    attn_mask.enqueue_fill(0.0)

    var tokens = ctx.enqueue_create_buffer[DType.int32](B * T_token)
    print("[s3tok] running forward...")
    s3tokenizer_forward(ctx, s3t, log_mel, tokens, cos_buf, sin_buf,
                          mask_pad, attn_mask, B, T_mel)
    ctx.synchronize()

    # Compare against upstream tokens.
    var ref_tokens = load_i64(fix + "tokens.bin")
    print("[s3tok] ref_tokens shape:", len(ref_tokens.data))
    print("[s3tok] first 20 mojo tokens:")
    var n_match = 0
    var total = min(len(ref_tokens.data), T_token)
    with tokens.map_to_host() as h:
        for i in range(min(20, total)):
            print("  [", i, "] mojo=", h[i], " ref=", ref_tokens.data[i])
        for i in range(total):
            if Int64(h[i]) == ref_tokens.data[i]:
                n_match += 1
    print("[s3tok] token agreement:", n_match, "/", total, " (", Float32(n_match) / Float32(total) * 100.0, "%)")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
