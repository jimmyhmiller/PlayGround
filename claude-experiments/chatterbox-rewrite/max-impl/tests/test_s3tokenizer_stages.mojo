"""Compare each stage of Mojo s3tokenizer against upstream."""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.math import sqrt
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from fixture import load_fp32
from weights import load_s3tokenizer, upload_fp32
from conv1d import conv1d_forward
from modules import gelu, layer_norm_forward, linear_forward
from s3tokenizer_block import s3tokenizer_block_forward
from transformer_blocks import (
    reshape_bsd_to_bhsd, reshape_bhsd_to_bsd, apply_rope_s3_style,
)
from attention import qk_scaled_and_masked, softmax_2d, av_matmul
from text_embed import build_rope_tables


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
    print("[s3tok-stages]", name, ": max-abs=", max_abs, " rel_l2=", rel, " (n=", ref_n, ")")


def test_s3tokenizer_stages() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var fix = "weights/s3gen_prompt/s3tok_diag/"

    var s3t = load_s3tokenizer(ctx, "weights/s3t")

    var log_mel = upload_fp32(ctx, fix + "log_mel_16k.bin")
    var B = 1
    var T_mel = 1200
    var N_MELS = 128
    var N_STATE = 1280

    # Stage 1: conv1 (mel → state, stride=2, kernel=3, pad=1)
    var T1 = (T_mel + 2 - 2 - 1) // 2 + 1   # 600
    print("[s3tok-stages] T1=", T1)
    var c1 = ctx.enqueue_create_buffer[DType.float32](B * N_STATE * T1)
    conv1d_forward(ctx, s3t.conv1, log_mel, c1, B, T_mel, T1)
    ctx.synchronize()
    _diff("conv1 (raw)", c1, fix + "conv1.bin")

    # Apply gelu (upstream applies gelu(conv1(x*mask)) — mask is all ones for us).
    var c1_act = ctx.enqueue_create_buffer[DType.float32](B * N_STATE * T1)
    gelu(ctx, c1, c1_act, B * N_STATE * T1)
    ctx.synchronize()

    # Stage 2: conv2.
    var T2 = (T1 + 2 - 2 - 1) // 2 + 1   # 300
    print("[s3tok-stages] T2=", T2)
    var c2 = ctx.enqueue_create_buffer[DType.float32](B * N_STATE * T2)
    conv1d_forward(ctx, s3t.conv2, c1_act, c2, B, T1, T2)
    ctx.synchronize()
    _diff("conv2 (raw)", c2, fix + "conv2.bin")

    # Apply gelu and transpose (B, C, T) → (B, T, C) for blocks.
    var c2_act = ctx.enqueue_create_buffer[DType.float32](B * N_STATE * T2)
    gelu(ctx, c2, c2_act, B * N_STATE * T2)
    ctx.synchronize()

    var x_seq = ctx.enqueue_create_buffer[DType.float32](B * T2 * N_STATE)
    var cp = c2_act.unsafe_ptr()
    var sp = x_seq.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(cp, sp, B, T2, N_STATE)
    def tr_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (T2 * N_STATE)
        var rem = i - bi * T2 * N_STATE
        var ti = rem // N_STATE
        var ci = rem - ti * N_STATE
        sp[i] = cp[bi * N_STATE * T2 + ci * T2 + ti]
    elementwise[tr_func, simd_width=1, target="gpu"](
        IndexList[1](B * T2 * N_STATE), DeviceContextPtr(ctx),
    )
    ctx.synchronize()

    # RoPE.
    var head_dim = 64
    var max_ctx = 4096
    var cos_buf = ctx.enqueue_create_buffer[DType.float32](max_ctx * head_dim)
    var sin_buf = ctx.enqueue_create_buffer[DType.float32](max_ctx * head_dim)
    build_rope_tables(ctx, max_ctx, head_dim, cos_buf, sin_buf)
    ctx.synchronize()
    _diff("rope_cos (first 2048)", cos_buf, fix + "rope_cos.bin")
    _diff("rope_sin (first 2048)", sin_buf, fix + "rope_sin.bin")

    var mask_pad = ctx.enqueue_create_buffer[DType.float32](B * T2 * 1)
    mask_pad.enqueue_fill(1.0)
    var attn_mask = ctx.enqueue_create_buffer[DType.float32](T2 * T2)
    attn_mask.enqueue_fill(0.0)

    # Stage-by-stage inside block 0.
    # 1. attn_ln(x_seq).
    var b0 = s3t.blocks[0].copy()
    var ln_x = ctx.enqueue_create_buffer[DType.float32](B * T2 * N_STATE)
    layer_norm_forward(ctx, b0.attn_ln, x_seq, ln_x, B * T2)
    ctx.synchronize()
    _diff("b0_attn_ln", ln_x, fix + "b0_attn_ln.bin")

    # 2. q/k/v projections.
    var q_lin = ctx.enqueue_create_buffer[DType.float32](B * T2 * N_STATE)
    var k_lin = ctx.enqueue_create_buffer[DType.float32](B * T2 * N_STATE)
    var v_lin = ctx.enqueue_create_buffer[DType.float32](B * T2 * N_STATE)
    linear_forward(ctx, b0.attn.to_q, ln_x, q_lin, B * T2)
    linear_forward(ctx, b0.attn.to_k, ln_x, k_lin, B * T2)
    linear_forward(ctx, b0.attn.to_v, ln_x, v_lin, B * T2)
    ctx.synchronize()
    _diff("b0_q", q_lin, fix + "b0_q.bin")
    _diff("b0_k", k_lin, fix + "b0_k.bin")
    _diff("b0_v", v_lin, fix + "b0_v.bin")

    # Manual attention forward replicating the s3tokenizer block.
    var H = 20
    var Dh = 64
    var D = H * Dh

    # RoPE on Q, K (reshape to BSHD first).
    var q_rope = ctx.enqueue_create_buffer[DType.float32](B * T2 * D)
    var k_rope = ctx.enqueue_create_buffer[DType.float32](B * T2 * D)
    apply_rope_s3_style(ctx, q_lin, q_rope, cos_buf, sin_buf, B, T2, H, Dh)
    apply_rope_s3_style(ctx, k_lin, k_rope, cos_buf, sin_buf, B, T2, H, Dh)

    # Permute (B, S, H, Dh) → (B, H, S, Dh).
    var q_perm = ctx.enqueue_create_buffer[DType.float32](B * H * T2 * Dh)
    var k_perm = ctx.enqueue_create_buffer[DType.float32](B * H * T2 * Dh)
    var v_perm = ctx.enqueue_create_buffer[DType.float32](B * H * T2 * Dh)
    reshape_bsd_to_bhsd(ctx, q_rope, q_perm, B, T2, H, Dh)
    reshape_bsd_to_bhsd(ctx, k_rope, k_perm, B, T2, H, Dh)
    reshape_bsd_to_bhsd(ctx, v_lin, v_perm, B, T2, H, Dh)

    # Scale Q and K each by Dh^-0.25.
    from std.math import sqrt as msqrt
    var scale: Float32 = 1.0 / msqrt(msqrt(Float32(Dh)))
    var qp = q_perm.unsafe_ptr()
    var kp = k_perm.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(qp, kp, scale)
    def scale_qk[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        qp[i] *= scale
        kp[i] *= scale
    elementwise[scale_qk, simd_width=1, target="gpu"](
        IndexList[1](B * H * T2 * Dh), DeviceContextPtr(ctx),
    )

    var logits = ctx.enqueue_create_buffer[DType.float32](B * H * T2 * T2)
    var probs = ctx.enqueue_create_buffer[DType.float32](B * H * T2 * T2)
    var attn_perm = ctx.enqueue_create_buffer[DType.float32](B * H * T2 * Dh)
    qk_scaled_and_masked(ctx, q_perm, k_perm, attn_mask, logits,
                          B * H, T2, T2, Dh, Float32(1.0), False)
    softmax_2d(ctx, logits, probs, B * H * T2, T2)
    av_matmul(ctx, probs, v_perm, attn_perm, B * H, T2, T2, Dh)

    var attn_flat = ctx.enqueue_create_buffer[DType.float32](B * T2 * D)
    reshape_bhsd_to_bsd(ctx, attn_perm, attn_flat, B, H, T2, Dh)

    var attn_lin = ctx.enqueue_create_buffer[DType.float32](B * T2 * D)
    linear_forward(ctx, b0.attn.to_out, attn_flat, attn_lin, B * T2)
    ctx.synchronize()
    _diff("b0_attn_out (pre fsm_mem)", attn_lin, fix + "b0_attn_out.bin")

    # Manual fsm_memory computation matching upstream exactly:
    # v_masked = v * mask_pad (mask is all 1s here so v_masked = v)
    # x = v_masked.transpose(1,2)   # (B, D, T)
    # x = pad(x, (15, 15), 0)         # zero-pad both sides
    # x = fsmn_block(x)                 # depthwise conv K=31, stride=1, pad=0
    # x = x.transpose(1,2)              # (B, T, D)
    # x += v_masked
    # return x * mask_pad
    # NOTE: But v here is the FSMN input — and upstream does forward_fsmn(v) where
    # v is the per-head view (B, T, H, Dh), then v.view(b, t, -1) flattens to (B,T,D).
    # That's the same memory layout as Mojo's v_lin. So v_masked input equals v_lin.

    # Permute Mojo's v_lin to (B, D, T) for the conv.
    var v_bdt = ctx.enqueue_create_buffer[DType.float32](B * D * T2)
    var vlp = v_lin.unsafe_ptr()
    var vbp = v_bdt.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(vlp, vbp, B, T2, D)
    def tr_v[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (D * T2)
        var rem = i - bi * D * T2
        var di = rem // T2
        var ti = rem - di * T2
        vbp[i] = vlp[bi * T2 * D + ti * D + di]
    elementwise[tr_v, simd_width=1, target="gpu"](
        IndexList[1](B * D * T2), DeviceContextPtr(ctx),
    )

    # Apply fsmn_block conv (depthwise, K=31, pad=15).
    var fsmn_conv_out = ctx.enqueue_create_buffer[DType.float32](B * D * T2)
    conv1d_forward(ctx, b0.attn.fsmn_conv, v_bdt, fsmn_conv_out, B, T2, T2)
    ctx.synchronize()
    _diff("b0_fsmn_block (conv only, (B,D,T))", fsmn_conv_out, fix + "b0_fsmn_block.bin")

    # Run full block 0 (x_seq updated in place).
    s3tokenizer_block_forward(
        ctx, s3t.blocks[0], x_seq, cos_buf, sin_buf,
        mask_pad, attn_mask, B, T2,
        has_attn_mask=False,
    )
    ctx.synchronize()
    _diff("block0 (full)", x_seq, fix + "block0.bin")

    # NOTE: this test runs the full s3tokenizer_block_forward(blocks[0]) on the
    # CONV-OUT input. To diagnose Mojo bug, we re-do the block manually here
    # and compare each intermediate against upstream's captures.
    # x_after_attn = original_x_seq (saved before block call) + attn_part.
    # But x_seq was modified by the block. Recompute x_after_attn from conv2.
    # ... skipped for brevity — we already know attn out matches.

    # Run remaining blocks.
    for i in range(1, 6):
        s3tokenizer_block_forward(
            ctx, s3t.blocks[i], x_seq, cos_buf, sin_buf,
            mask_pad, attn_mask, B, T2,
            has_attn_mask=False,
        )
    ctx.synchronize()
    _diff("block_last (after all 6)", x_seq, fix + "block_last.bin")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
