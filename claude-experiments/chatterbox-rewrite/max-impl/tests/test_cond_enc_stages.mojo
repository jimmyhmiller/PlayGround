"""Run each stage of T3CondEnc separately and compare against upstream dumps."""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.math import sqrt
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from fixture import load_fp32, load_i64
from weights import load_t3, load_t3_cond_enc, upload_fp32
from modules import linear_forward, embedding_forward
from perceiver import perceiver_forward


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
    print("[ce-stages]", name, ": max-abs=", max_abs, " rel_l2=", rel, " (n=", ref_n, ")")


def test_cond_enc_stages() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var fix = "weights/s3gen_prompt/cond_enc_diag/"

    var t3 = load_t3(ctx, "weights/t3")
    var cond_enc = load_t3_cond_enc(ctx, "weights/t3", t3.speech_emb, t3.speech_pos_emb)

    var spk_emb = upload_fp32(ctx, fix + "speaker_emb.bin")
    var ctok = load_i64(fix + "cond_prompt_speech_tokens.bin")
    var emotion = upload_fp32(ctx, fix + "emotion_adv.bin")

    var B = 1
    var CL = 150
    var D = 1024
    var SQ = 32

    # Stage 1: spkr_enc(speaker_emb) -> (1, 1024)
    var spkr_proj = ctx.enqueue_create_buffer[DType.float32](B * D)
    linear_forward(ctx, cond_enc.spkr_enc, spk_emb, spkr_proj, B)
    ctx.synchronize()
    _diff("spkr_proj", spkr_proj, fix + "spkr_proj.bin")

    # Stage 2: speech_emb(cond_tokens) -> (1, 150, 1024)
    var ctok_buf = ctx.enqueue_create_buffer[DType.int64](B * CL)
    with ctok_buf.map_to_host() as h:
        for i in range(B * CL):
            h[i] = ctok.data[i]
    var cs_emb = ctx.enqueue_create_buffer[DType.float32](B * CL * D)
    embedding_forward(ctx, cond_enc.speech_emb, ctok_buf, cs_emb, B, CL)
    ctx.synchronize()
    _diff("cs_emb_only", cs_emb, fix + "cs_emb_only.bin")

    # Stage 3: + speech_pos_emb
    var ce_ptr = cs_emb.unsafe_ptr()
    var pp_ptr = cond_enc.speech_pos_emb.table.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(ce_ptr, pp_ptr, B, CL, D)
    def add_pos_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (CL * D)
        var rem = i - bi * CL * D
        var ti = rem // D
        var di = rem - ti * D
        ce_ptr[i] = ce_ptr[i] + pp_ptr[ti * D + di]
    elementwise[add_pos_fn, simd_width=1, target="gpu"](
        IndexList[1](B * CL * D), DeviceContextPtr(ctx),
    )
    ctx.synchronize()
    _diff("cs_emb_with_pos", cs_emb, fix + "cs_emb_with_pos.bin")

    # Stage 4: perceiver
    var mask_q = ctx.enqueue_create_buffer[DType.float32](SQ * SQ)
    mask_q.enqueue_fill(0.0)
    var mask_qq = ctx.enqueue_create_buffer[DType.float32](SQ * CL)
    mask_qq.enqueue_fill(0.0)
    var perc_out = ctx.enqueue_create_buffer[DType.float32](B * SQ * D)
    perceiver_forward(ctx, cond_enc.perceiver, cs_emb, perc_out, mask_q, mask_qq, B, CL)
    ctx.synchronize()
    _diff("perceiver_out", perc_out, fix + "perceiver_out.bin")

    # Stage 5: emotion proj
    var emo_proj = ctx.enqueue_create_buffer[DType.float32](B * D)
    linear_forward(ctx, cond_enc.emotion_fc, emotion, emo_proj, B)
    ctx.synchronize()
    _diff("emo_proj", emo_proj, fix + "emo_proj.bin")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
