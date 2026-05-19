"""T3CondEnc parity test: speaker_emb + cond_prompt_tokens + emotion → cond_emb."""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.math import sqrt
from std.gpu.host import DeviceContext, DeviceBuffer

from fixture import load_fp32, load_i64
from weights import load_t3, load_t3_cond_enc, upload_fp32
from cond_enc import t3_cond_enc_forward


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
    print("[cond-enc]", name, ": max-abs=", max_abs, " rel_l2=", rel, " (n=", ref_n, ")")


def test_cond_enc_parity() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var fix = "weights/s3gen_prompt/cond_enc_diag/"

    print("[cond-enc] loading T3 + T3CondEnc...")
    var t3 = load_t3(ctx, "weights/t3")
    var cond_enc = load_t3_cond_enc(ctx, "weights/t3", t3.speech_emb, t3.speech_pos_emb)

    # Load inputs from oracle.
    var spk_emb = upload_fp32(ctx, fix + "speaker_emb.bin")        # (1, 256)
    var ctok = load_i64(fix + "cond_prompt_speech_tokens.bin")     # (1, 150)
    var emotion = upload_fp32(ctx, fix + "emotion_adv.bin")        # (1, 1, 1)

    var B = 1
    var CL = 150
    var D = 1024
    var SQ = 32   # n_queries

    # Build tokens buffer.
    var ctok_buf = ctx.enqueue_create_buffer[DType.int64](B * CL)
    with ctok_buf.map_to_host() as h:
        for i in range(B * CL):
            h[i] = ctok.data[i]

    # Build dummy mask buffers for perceiver (zeros).
    var mask_q = ctx.enqueue_create_buffer[DType.float32](SQ * SQ)
    mask_q.enqueue_fill(0.0)
    var mask_qq = ctx.enqueue_create_buffer[DType.float32](SQ * CL)
    mask_qq.enqueue_fill(0.0)

    var cond_emb = ctx.enqueue_create_buffer[DType.float32](B * 34 * D)
    print("[cond-enc] running t3_cond_enc_forward...")
    t3_cond_enc_forward(ctx, cond_enc, spk_emb, ctok_buf, emotion, cond_emb,
                         mask_q, mask_qq, B)
    ctx.synchronize()

    _diff("cond_emb", cond_emb, fix + "cond_emb.bin")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
