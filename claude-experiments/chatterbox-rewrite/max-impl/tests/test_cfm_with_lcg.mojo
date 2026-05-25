"""Feed Mojo's CFM the SAME LCG noise that upstream's CFM consumed
successfully, and compare the resulting mel. Isolates the Mojo CFM solver bug.
"""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.math import sqrt
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from fixture import load_fp32, load_i64
from weights import (
    load_upsample_conformer_encoder, load_cfm_estimator_real, upload_fp32,
)
from upsample_encoder import upsample_conformer_forward
from cfm_estimator_new import cfm_solve_euler


comptime B = 1
comptime MEL = 80
comptime T_PROMPT_TOKEN = 250
comptime T_PROMPT_MEL = 500
comptime N_CFM_STEPS = 10
comptime CFG: Float32 = 0.7


def test_cfm_with_lcg() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var voice_dir = "weights/s3gen_prompt/"

    print("[lcg-cfm] loading models...")
    var enc = load_upsample_conformer_encoder(ctx, "weights/s3gen/flow")
    var cfm = load_cfm_estimator_real(ctx, "weights/s3gen/flow/decoder/estimator")

    var prompt_tok = load_i64(voice_dir + "prompt_token.bin")
    var speech_tok = load_i64(voice_dir + "speech_tokens.bin")
    var T_GEN_TOKEN = speech_tok.numel()
    var T_TOTAL_TOKEN = T_PROMPT_TOKEN + T_GEN_TOKEN
    var T_TOTAL_MEL = 2 * T_TOTAL_TOKEN
    print("[lcg-cfm] T_TOTAL_TOKEN=", T_TOTAL_TOKEN, " T_TOTAL_MEL=", T_TOTAL_MEL)

    var tok_buf = ctx.enqueue_create_buffer[DType.int64](B * T_TOTAL_TOKEN)
    with tok_buf.map_to_host() as h:
        for i in range(T_PROMPT_TOKEN):
            h[i] = prompt_tok.data[i]
        for i in range(T_GEN_TOKEN):
            h[T_PROMPT_TOKEN + i] = speech_tok.data[i]

    var mu = ctx.enqueue_create_buffer[DType.float32](B * MEL * T_TOTAL_MEL)
    upsample_conformer_forward(ctx, enc, tok_buf, mu, B, T_TOTAL_TOKEN)
    ctx.synchronize()

    var prompt_feat = upload_fp32(ctx, voice_dir + "prompt_feat.bin")
    var cond = ctx.enqueue_create_buffer[DType.float32](B * MEL * T_TOTAL_MEL)
    cond.enqueue_fill(0.0)
    var pf_ptr = prompt_feat.unsafe_ptr()
    var cond_ptr = cond.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(pf_ptr, cond_ptr, T_TOTAL_MEL)
    def cond_fill[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (T_PROMPT_MEL * MEL)
        var rem = i - bi * T_PROMPT_MEL * MEL
        var ti = rem // MEL
        var ci = rem - ti * MEL
        cond_ptr[bi * MEL * T_TOTAL_MEL + ci * T_TOTAL_MEL + ti] = pf_ptr[i]
    elementwise[cond_fill, simd_width=1, target="gpu"](
        IndexList[1](B * T_PROMPT_MEL * MEL), DeviceContextPtr(ctx),
    )

    var spks = upload_fp32(ctx, voice_dir + "embedding_normed_affine.bin")
    var x = upload_fp32(ctx, voice_dir + "lcg_diag/lcg_noise.bin")
    var mask = ctx.enqueue_create_buffer[DType.float32](B * T_TOTAL_MEL)
    mask.enqueue_fill(1.0)

    print("[lcg-cfm] running Mojo CFM with LCG noise...")
    cfm_solve_euler(ctx, cfm, x, mu, spks, cond, mask, B, T_TOTAL_MEL, N_CFM_STEPS, CFG)
    ctx.synchronize()

    # Compare against upstream's CFM result with the SAME noise.
    var expected_full = load_fp32(voice_dir + "lcg_diag/upstream_cfm_mel_full.bin")
    var n = B * MEL * T_TOTAL_MEL
    var max_abs: Float32 = 0.0
    var sum_diff_sq: Float32 = 0.0
    var sum_ref_sq: Float32 = 0.0
    var mojo_min: Float32 = 1e30
    var mojo_max: Float32 = -1e30
    var ref_min: Float32 = 1e30
    var ref_max: Float32 = -1e30
    with x.map_to_host() as h:
        for i in range(n):
            var dd = h[i] - expected_full.data[i]
            var d = dd
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_diff_sq += dd * dd
            sum_ref_sq += expected_full.data[i] * expected_full.data[i]
            if h[i] < mojo_min: mojo_min = h[i]
            if h[i] > mojo_max: mojo_max = h[i]
            if expected_full.data[i] < ref_min: ref_min = expected_full.data[i]
            if expected_full.data[i] > ref_max: ref_max = expected_full.data[i]
    var rel = sqrt(sum_diff_sq / sum_ref_sq) if sum_ref_sq > 0.0 else Float32(0.0)
    print("[lcg-cfm] Mojo CFM mel range: [", mojo_min, ",", mojo_max, "]")
    print("[lcg-cfm] Upstream CFM mel range: [", ref_min, ",", ref_max, "]")
    print("[lcg-cfm] parity (same LCG noise): max-abs=", max_abs, " rel_l2=", rel)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
