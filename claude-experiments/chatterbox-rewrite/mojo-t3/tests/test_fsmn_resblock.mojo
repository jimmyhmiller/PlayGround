"""Parity test for a single FSMN ResidualAttentionBlock (LN + FSMN + LN + MLP)."""
from std.math import sqrt
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import load_fp32
from layernorm import layernorm_kernel, linear_kernel
from fsmn_attention import (
    fsmn_depthwise_conv_kernel, fsmn_memory_kernel,
    rope_s3tokenizer_kernel, scale_4d_kernel,
    multiply_mask_3d_kernel,
    permute_bshd_to_bhsd_kernel, permute_bhsd_to_bsd_kernel,
)
from perceiver import cross_qkt_kernel, cross_softmax_kernel, cross_av_kernel, add_3d_kernel
from decoder_kernels import gelu_kernel


comptime B = 1
comptime S = 24
comptime D = 128
comptime H = 4
comptime DH = D // H
comptime HALF = DH // 2
comptime KSIZE = 31
comptime LEFT = 15
comptime RIGHT = 15
comptime BLOCK = 128
comptime MLP_INNER = D * 4


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def upload_w(mut ctx: DeviceContext, fix: String, name: String) raises -> DeviceBuffer[DType.float32]:
    var t = load_fp32(fix + name)
    var n = len(t.data)
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    upload(buf, t.data, n)
    return buf^


def fsmn_attn_forward(
    mut ctx: DeviceContext,
    mut x_buf: DeviceBuffer[DType.float32],          # (B, S, D) input
    mut out_buf: DeviceBuffer[DType.float32],        # (B, S, D) attn output (with FSMN added)
    mut mp_buf: DeviceBuffer[DType.float32],         # (B, S, 1) mask
    mut cos_buf: DeviceBuffer[DType.float32],
    mut sin_buf: DeviceBuffer[DType.float32],
    mut q_w: DeviceBuffer[DType.float32], mut q_b: DeviceBuffer[DType.float32],
    mut k_w: DeviceBuffer[DType.float32], mut k_b: DeviceBuffer[DType.float32],   # k_b zeros
    mut v_w: DeviceBuffer[DType.float32], mut v_b: DeviceBuffer[DType.float32],
    mut out_w: DeviceBuffer[DType.float32], mut out_b: DeviceBuffer[DType.float32],
    mut fsmn_w: DeviceBuffer[DType.float32],
) raises:
    """Full FSMN attention block forward. Writes to out_buf."""
    # Intermediates.
    var q_lin = ctx.enqueue_create_buffer[DType.float32](B * S * D)
    var k_lin = ctx.enqueue_create_buffer[DType.float32](B * S * D)
    var v_lin = ctx.enqueue_create_buffer[DType.float32](B * S * D)
    var q_4d = ctx.enqueue_create_buffer[DType.float32](B * S * H * DH)
    var k_4d = ctx.enqueue_create_buffer[DType.float32](B * S * H * DH)
    var v_4d = ctx.enqueue_create_buffer[DType.float32](B * S * H * DH)
    var q_rope = ctx.enqueue_create_buffer[DType.float32](B * S * H * DH)
    var k_rope = ctx.enqueue_create_buffer[DType.float32](B * S * H * DH)
    var v_masked = ctx.enqueue_create_buffer[DType.float32](B * S * D)
    var fsmn_conv = ctx.enqueue_create_buffer[DType.float32](B * S * D)
    var fsm_mem = ctx.enqueue_create_buffer[DType.float32](B * S * D)
    var q_bhsd = ctx.enqueue_create_buffer[DType.float32](B * H * S * DH)
    var k_bhsd = ctx.enqueue_create_buffer[DType.float32](B * H * S * DH)
    var v_bhsd = ctx.enqueue_create_buffer[DType.float32](B * H * S * DH)
    var logits = ctx.enqueue_create_buffer[DType.float32](B * H * S * S)
    var probs = ctx.enqueue_create_buffer[DType.float32](B * H * S * S)
    var av = ctx.enqueue_create_buffer[DType.float32](B * H * S * DH)
    var comb = ctx.enqueue_create_buffer[DType.float32](B * S * D)
    var attn_lin_out = ctx.enqueue_create_buffer[DType.float32](B * S * D)

    # Layouts.
    comptime x_layout = row_major[B, S, D]()
    comptime cs_layout = row_major[S, HALF]()
    comptime mp_layout = row_major[B, S, 1]()
    comptime lin_w_layout = row_major[D, D]()
    comptime lin_b_layout = row_major[D]()
    comptime x4_layout = row_major[B, S, H, DH]()
    comptime bhsd_layout = row_major[B, H, S, DH]()
    comptime logits_layout = row_major[B, H, S, S]()
    comptime fsmn_w_layout = row_major[D, 1, KSIZE]()

    var x_tt = TileTensor(x_buf, x_layout)
    var cos_tt = TileTensor(cos_buf, cs_layout)
    var sin_tt = TileTensor(sin_buf, cs_layout)
    var mp_tt = TileTensor(mp_buf, mp_layout)
    var q_w_tt = TileTensor(q_w, lin_w_layout); var q_b_tt = TileTensor(q_b, lin_b_layout)
    var k_w_tt = TileTensor(k_w, lin_w_layout); var k_b_tt = TileTensor(k_b, lin_b_layout)
    var v_w_tt = TileTensor(v_w, lin_w_layout); var v_b_tt = TileTensor(v_b, lin_b_layout)
    var out_w_tt = TileTensor(out_w, lin_w_layout); var out_b_tt = TileTensor(out_b, lin_b_layout)
    var fsmn_w_tt = TileTensor(fsmn_w, fsmn_w_layout)
    var q_lin_tt = TileTensor(q_lin, x_layout)
    var k_lin_tt = TileTensor(k_lin, x_layout)
    var v_lin_tt = TileTensor(v_lin, x_layout)
    var q_4d_tt = TileTensor(q_4d, x4_layout)
    var k_4d_tt = TileTensor(k_4d, x4_layout)
    var v_4d_tt = TileTensor(v_4d, x4_layout)
    var q_rope_tt = TileTensor(q_rope, x4_layout)
    var k_rope_tt = TileTensor(k_rope, x4_layout)
    var v_masked_tt = TileTensor(v_masked, x_layout)
    var fsmn_conv_tt = TileTensor(fsmn_conv, x_layout)
    var fsm_mem_tt = TileTensor(fsm_mem, x_layout)
    var q_bhsd_tt = TileTensor(q_bhsd, bhsd_layout)
    var k_bhsd_tt = TileTensor(k_bhsd, bhsd_layout)
    var v_bhsd_tt = TileTensor(v_bhsd, bhsd_layout)
    var logits_tt = TileTensor(logits, logits_layout)
    var probs_tt = TileTensor(probs, logits_layout)
    var av_tt = TileTensor(av, bhsd_layout)
    var comb_tt = TileTensor(comb, x_layout)
    var attn_lin_out_tt = TileTensor(attn_lin_out, x_layout)
    var out_tt = TileTensor(out_buf, x_layout)

    # Linears Q/K/V.
    comptime klin = linear_kernel[
        DType.float32, type_of(x_layout), type_of(lin_w_layout),
        type_of(lin_b_layout), type_of(x_layout),
        True, BLOCK,
    ]
    ctx.enqueue_function[klin, klin](q_lin_tt, x_tt, q_w_tt, q_b_tt, B, S, D, D, grid_dim=B*S, block_dim=BLOCK)
    ctx.enqueue_function[klin, klin](k_lin_tt, x_tt, k_w_tt, k_b_tt, B, S, D, D, grid_dim=B*S, block_dim=BLOCK)
    ctx.enqueue_function[klin, klin](v_lin_tt, x_tt, v_w_tt, v_b_tt, B, S, D, D, grid_dim=B*S, block_dim=BLOCK)

    # Reshape (B, S, D) → (B, S, H, Dh) via copy.
    ctx.enqueue_copy(q_4d, q_lin)
    ctx.enqueue_copy(k_4d, k_lin)
    ctx.enqueue_copy(v_4d, v_lin)

    # RoPE on Q and K.
    comptime krope = rope_s3tokenizer_kernel[
        DType.float32, type_of(x4_layout), type_of(cs_layout),
        type_of(x4_layout), DH, HALF,
    ]
    ctx.enqueue_function[krope, krope](q_rope_tt, q_4d_tt, cos_tt, sin_tt, S, H, grid_dim=B*S*H, block_dim=DH)
    ctx.enqueue_function[krope, krope](k_rope_tt, k_4d_tt, cos_tt, sin_tt, S, H, grid_dim=B*S*H, block_dim=DH)

    # FSMN memory branch.
    comptime kmm = multiply_mask_3d_kernel[
        DType.float32, type_of(x_layout), type_of(mp_layout),
        type_of(x_layout), BLOCK,
    ]
    ctx.enqueue_function[kmm, kmm](v_masked_tt, v_lin_tt, mp_tt, B, S, D, grid_dim=B*S, block_dim=BLOCK)
    comptime kfconv = fsmn_depthwise_conv_kernel[
        DType.float32, type_of(x_layout), type_of(fsmn_w_layout),
        type_of(x_layout), KSIZE, LEFT, RIGHT, BLOCK,
    ]
    ctx.enqueue_function[kfconv, kfconv](fsmn_conv_tt, v_masked_tt, fsmn_w_tt, B, S, D, grid_dim=B*S, block_dim=BLOCK)
    comptime kfm = fsmn_memory_kernel[
        DType.float32, type_of(x_layout), type_of(x_layout),
        type_of(mp_layout), type_of(x_layout), BLOCK,
    ]
    ctx.enqueue_function[kfm, kfm](fsm_mem_tt, fsmn_conv_tt, v_masked_tt, mp_tt, B, S, D, grid_dim=B*S, block_dim=BLOCK)

    # Permute (B, S, H, Dh) → (B, H, S, Dh).
    comptime kpb = permute_bshd_to_bhsd_kernel[
        DType.float32, type_of(x4_layout), type_of(bhsd_layout), H, DH,
    ]
    ctx.enqueue_function[kpb, kpb](q_bhsd_tt, q_rope_tt, B, S, grid_dim=B*S*H, block_dim=DH)
    ctx.enqueue_function[kpb, kpb](k_bhsd_tt, k_rope_tt, B, S, grid_dim=B*S*H, block_dim=DH)
    ctx.enqueue_function[kpb, kpb](v_bhsd_tt, v_4d_tt,   B, S, grid_dim=B*S*H, block_dim=DH)

    # Scale Q and K.
    comptime scale_amt: Float32 = 1.0 / sqrt(sqrt(Float32(DH)))
    comptime ksc = scale_4d_kernel[DType.float32, type_of(bhsd_layout), BLOCK]
    ctx.enqueue_function[ksc, ksc](q_bhsd_tt, B, H, S, DH, scale_amt, grid_dim=B*H*S, block_dim=BLOCK)
    ctx.enqueue_function[ksc, ksc](k_bhsd_tt, B, H, S, DH, scale_amt, grid_dim=B*H*S, block_dim=BLOCK)

    # Attention.
    comptime kqkt = cross_qkt_kernel[
        DType.float32, type_of(bhsd_layout), type_of(bhsd_layout),
        type_of(logits_layout), DH, S,
    ]
    ctx.enqueue_function[kqkt, kqkt](logits_tt, q_bhsd_tt, k_bhsd_tt, H, S, Float32(1.0), grid_dim=B*H*S, block_dim=S)
    comptime ksm = cross_softmax_kernel[
        DType.float32, type_of(logits_layout), type_of(logits_layout), S, BLOCK,
    ]
    ctx.enqueue_function[ksm, ksm](probs_tt, logits_tt, H, S, grid_dim=B*H*S, block_dim=BLOCK)
    comptime kav = cross_av_kernel[
        DType.float32, type_of(logits_layout), type_of(bhsd_layout),
        type_of(bhsd_layout), S, DH,
    ]
    ctx.enqueue_function[kav, kav](av_tt, probs_tt, v_bhsd_tt, H, S, grid_dim=B*H*S, block_dim=DH)

    # Combine heads (B,H,S,Dh) → (B,S,D).
    comptime kcomb = permute_bhsd_to_bsd_kernel[
        DType.float32, type_of(bhsd_layout), type_of(x_layout), H, DH,
    ]
    ctx.enqueue_function[kcomb, kcomb](comb_tt, av_tt, B, S, grid_dim=B*S*H, block_dim=DH)

    # out Linear.
    ctx.enqueue_function[klin, klin](attn_lin_out_tt, comb_tt, out_w_tt, out_b_tt, B, S, D, D, grid_dim=B*S, block_dim=BLOCK)

    # Final = attn_lin_out + fsm_mem.
    comptime kadd = add_3d_kernel[DType.float32, type_of(x_layout), BLOCK]
    ctx.enqueue_function[kadd, kadd](out_tt, attn_lin_out_tt, fsm_mem_tt, B, S, D, grid_dim=B*S, block_dim=BLOCK)


def test_fsmn_resblock() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/fsmn_resblock/"
    var ctx = DeviceContext()

    var x_t = load_fp32(fix + "x.bin")
    var exp = load_fp32(fix + "out.bin")
    var mp_t = load_fp32(fix + "mask_pad.bin")
    var cos_t = load_fp32(fix + "cos.bin")
    var sin_t = load_fp32(fix + "sin.bin")

    var x_buf = ctx.enqueue_create_buffer[DType.float32](B * S * D)
    upload(x_buf, x_t.data, B * S * D)
    var mp_buf = ctx.enqueue_create_buffer[DType.float32](B * S * 1)
    upload(mp_buf, mp_t.data, B * S * 1)
    var cos_buf = ctx.enqueue_create_buffer[DType.float32](S * HALF)
    upload(cos_buf, cos_t.data, S * HALF)
    var sin_buf = ctx.enqueue_create_buffer[DType.float32](S * HALF)
    upload(sin_buf, sin_t.data, S * HALF)

    var attn_ln_w = upload_w(ctx, fix, "attn_ln_w.bin")
    var attn_ln_b = upload_w(ctx, fix, "attn_ln_b.bin")
    var mlp_ln_w  = upload_w(ctx, fix, "mlp_ln_w.bin")
    var mlp_ln_b  = upload_w(ctx, fix, "mlp_ln_b.bin")
    var q_w = upload_w(ctx, fix, "q_w.bin"); var q_b = upload_w(ctx, fix, "q_b.bin")
    var k_w = upload_w(ctx, fix, "k_w.bin")
    var k_b = ctx.enqueue_create_buffer[DType.float32](D); k_b.enqueue_fill(0.0)
    var v_w = upload_w(ctx, fix, "v_w.bin"); var v_b = upload_w(ctx, fix, "v_b.bin")
    var out_w = upload_w(ctx, fix, "out_w.bin"); var out_b = upload_w(ctx, fix, "out_b.bin")
    var fsmn_w = upload_w(ctx, fix, "fsmn_w.bin")
    var mlp_fc1_w = upload_w(ctx, fix, "mlp_fc1_w.bin"); var mlp_fc1_b = upload_w(ctx, fix, "mlp_fc1_b.bin")
    var mlp_fc2_w = upload_w(ctx, fix, "mlp_fc2_w.bin"); var mlp_fc2_b = upload_w(ctx, fix, "mlp_fc2_b.bin")

    # Layouts.
    comptime x_layout = row_major[B, S, D]()
    comptime ln_w_layout = row_major[D]()
    comptime lin_w_layout = row_major[D, D]()
    comptime mlp_w_layout = row_major[MLP_INNER, D]()
    comptime mlp_w2_layout = row_major[D, MLP_INNER]()
    comptime mlp_b_layout = row_major[MLP_INNER]()
    comptime mlp_x_layout = row_major[B, S, MLP_INNER]()

    # Stage 1: attn_ln(x) → ln_x
    var ln_x = ctx.enqueue_create_buffer[DType.float32](B * S * D)
    var x_tt = TileTensor(x_buf, x_layout)
    var ln_x_tt = TileTensor(ln_x, x_layout)
    var attn_ln_w_tt = TileTensor(attn_ln_w, ln_w_layout)
    var attn_ln_b_tt = TileTensor(attn_ln_b, ln_w_layout)
    comptime kln = layernorm_kernel[
        DType.float32, type_of(x_layout), type_of(ln_w_layout), type_of(x_layout), BLOCK,
    ]
    ctx.enqueue_function[kln, kln](
        ln_x_tt, x_tt, attn_ln_w_tt, attn_ln_b_tt,
        B, S, D, Float32(1.0e-5),
        grid_dim=B*S, block_dim=BLOCK,
    )

    # Stage 2: FSMN attention forward.
    var attn_out = ctx.enqueue_create_buffer[DType.float32](B * S * D)
    fsmn_attn_forward(ctx, ln_x, attn_out, mp_buf, cos_buf, sin_buf,
                      q_w, q_b, k_w, k_b, v_w, v_b, out_w, out_b, fsmn_w)

    # Stage 3: x = x + attn_out (residual).
    var post_attn = ctx.enqueue_create_buffer[DType.float32](B * S * D)
    var attn_out_tt = TileTensor(attn_out, x_layout)
    var post_attn_tt = TileTensor(post_attn, x_layout)
    comptime kadd = add_3d_kernel[DType.float32, type_of(x_layout), BLOCK]
    ctx.enqueue_function[kadd, kadd](
        post_attn_tt, x_tt, attn_out_tt, B, S, D,
        grid_dim=B*S, block_dim=BLOCK,
    )

    # Stage 4: mlp_ln(post_attn).
    var ln_post = ctx.enqueue_create_buffer[DType.float32](B * S * D)
    var ln_post_tt = TileTensor(ln_post, x_layout)
    var mlp_ln_w_tt = TileTensor(mlp_ln_w, ln_w_layout)
    var mlp_ln_b_tt = TileTensor(mlp_ln_b, ln_w_layout)
    ctx.enqueue_function[kln, kln](
        ln_post_tt, post_attn_tt, mlp_ln_w_tt, mlp_ln_b_tt,
        B, S, D, Float32(1.0e-5),
        grid_dim=B*S, block_dim=BLOCK,
    )

    # Stage 5: MLP fc1 → GELU → fc2.
    var mlp_inner_buf = ctx.enqueue_create_buffer[DType.float32](B * S * MLP_INNER)
    var mlp_inner_tt = TileTensor(mlp_inner_buf, mlp_x_layout)
    var mlp_fc1_w_tt = TileTensor(mlp_fc1_w, mlp_w_layout)
    var mlp_fc1_b_tt = TileTensor(mlp_fc1_b, mlp_b_layout)
    comptime klin_fc1 = linear_kernel[
        DType.float32, type_of(x_layout), type_of(mlp_w_layout),
        type_of(mlp_b_layout), type_of(mlp_x_layout),
        True, BLOCK,
    ]
    ctx.enqueue_function[klin_fc1, klin_fc1](
        mlp_inner_tt, ln_post_tt, mlp_fc1_w_tt, mlp_fc1_b_tt,
        B, S, D, MLP_INNER,
        grid_dim=B*S, block_dim=BLOCK,
    )
    # GELU into a separate buffer (avoid aliasing).
    comptime mlp_flat_layout = row_major[B * S * MLP_INNER]()
    var mlp_inner_flat_tt = TileTensor(mlp_inner_buf, mlp_flat_layout)
    var mlp_gelu_buf = ctx.enqueue_create_buffer[DType.float32](B * S * MLP_INNER)
    var mlp_gelu_flat_tt = TileTensor(mlp_gelu_buf, mlp_flat_layout)
    comptime kgelu = gelu_kernel[
        DType.float32, type_of(mlp_flat_layout), type_of(mlp_flat_layout), BLOCK,
    ]
    ctx.enqueue_function[kgelu, kgelu](
        mlp_gelu_flat_tt, mlp_inner_flat_tt, B * S * MLP_INNER,
        grid_dim=(B * S * MLP_INNER + BLOCK - 1) // BLOCK, block_dim=BLOCK,
    )
    var mlp_gelu_tt = TileTensor(mlp_gelu_buf, mlp_x_layout)
    var mlp_out_buf = ctx.enqueue_create_buffer[DType.float32](B * S * D)
    var mlp_out_tt = TileTensor(mlp_out_buf, x_layout)
    var mlp_fc2_w_tt = TileTensor(mlp_fc2_w, mlp_w2_layout)
    var mlp_fc2_b_tt = TileTensor(mlp_fc2_b, mlp_b_layout)   # bias is (D,) not MLP_INNER — wrong!

    # Stage 6: x = post_attn + mlp_out (residual).
    var final_buf = ctx.enqueue_create_buffer[DType.float32](B * S * D)
    var final_tt = TileTensor(final_buf, x_layout)

    comptime ln_b_d_layout = row_major[D]()
    var mlp_fc2_b_tt_correct = TileTensor(mlp_fc2_b, ln_b_d_layout)
    comptime klin_fc2 = linear_kernel[
        DType.float32, type_of(mlp_x_layout), type_of(mlp_w2_layout),
        type_of(ln_b_d_layout), type_of(x_layout),
        True, BLOCK,
    ]
    ctx.enqueue_function[klin_fc2, klin_fc2](
        mlp_out_tt, mlp_gelu_tt, mlp_fc2_w_tt, mlp_fc2_b_tt_correct,
        B, S, MLP_INNER, D,
        grid_dim=B*S, block_dim=BLOCK,
    )

    ctx.enqueue_function[kadd, kadd](
        final_tt, post_attn_tt, mlp_out_tt, B, S, D,
        grid_dim=B*S, block_dim=BLOCK,
    )
    ctx.synchronize()

    var n_out = B * S * D
    var max_abs: Float32 = 0.0
    var max_rel: Float32 = 0.0
    with final_buf.map_to_host() as h:
        for i in range(n_out):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            var r0 = exp.data[i]
            if r0 < 0.0: r0 = -r0
            if r0 > 1.0e-3:
                var r = d / r0
                if r > max_rel: max_rel = r
            if i < 8:
                print("rb[", i, "]: mojo=", h[i], "  torch=", exp.data[i], "  diff=", d)
    print("FSMN ResBlock — max abs:", max_abs, "  max_rel:", max_rel)
    assert_almost_equal(max_abs, 0.0, atol=2.0e-4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
