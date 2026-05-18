"""Parity test for FSMN multi-head attention (small B=1, S=24, D=128, H=4)."""
from std.math import sqrt
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import load_fp32
from layernorm import linear_kernel
from fsmn_attention import (
    fsmn_depthwise_conv_kernel, fsmn_memory_kernel,
    rope_s3tokenizer_kernel, scale_4d_kernel,
    multiply_mask_3d_kernel,
    permute_bshd_to_bhsd_kernel, permute_bhsd_to_bsd_kernel,
)
from perceiver import cross_qkt_kernel, cross_softmax_kernel, cross_av_kernel
from perceiver import split_heads_kernel, combine_heads_kernel, add_3d_kernel


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


def test_fsmn_attention() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/fsmn_attn/"
    var ctx = DeviceContext()

    var x_t = load_fp32(fix + "x.bin")
    var exp = load_fp32(fix + "out.bin")
    var cos_t = load_fp32(fix + "cos.bin")
    var sin_t = load_fp32(fix + "sin.bin")
    var mask_pad_t = load_fp32(fix + "mask_pad.bin")
    var q_w = upload_w(ctx, fix, "q_w.bin")
    var q_b = upload_w(ctx, fix, "q_b.bin")
    var k_w = upload_w(ctx, fix, "k_w.bin")
    var v_w = upload_w(ctx, fix, "v_w.bin")
    var v_b = upload_w(ctx, fix, "v_b.bin")
    var out_w = upload_w(ctx, fix, "out_w.bin")
    var out_b = upload_w(ctx, fix, "out_b.bin")
    var fsmn_w = upload_w(ctx, fix, "fsmn_w.bin")

    # Buffers.
    var x_buf = ctx.enqueue_create_buffer[DType.float32](B * S * D)
    upload(x_buf, x_t.data, B * S * D)
    var cos_buf = ctx.enqueue_create_buffer[DType.float32](S * HALF)
    upload(cos_buf, cos_t.data, S * HALF)
    var sin_buf = ctx.enqueue_create_buffer[DType.float32](S * HALF)
    upload(sin_buf, sin_t.data, S * HALF)
    var mp_buf = ctx.enqueue_create_buffer[DType.float32](B * S * 1)
    upload(mp_buf, mask_pad_t.data, B * S * 1)
    # k bias = zeros (key has bias=False).
    var k_b_buf = ctx.enqueue_create_buffer[DType.float32](D)
    k_b_buf.enqueue_fill(0.0)

    # Intermediates.
    var q_lin_buf = ctx.enqueue_create_buffer[DType.float32](B * S * D)
    var k_lin_buf = ctx.enqueue_create_buffer[DType.float32](B * S * D)
    var v_lin_buf = ctx.enqueue_create_buffer[DType.float32](B * S * D)
    var q_4d_buf = ctx.enqueue_create_buffer[DType.float32](B * S * H * DH)   # (B,S,H,Dh)
    var k_4d_buf = ctx.enqueue_create_buffer[DType.float32](B * S * H * DH)
    var v_4d_buf = ctx.enqueue_create_buffer[DType.float32](B * S * H * DH)
    var q_rope_buf = ctx.enqueue_create_buffer[DType.float32](B * S * H * DH)
    var k_rope_buf = ctx.enqueue_create_buffer[DType.float32](B * S * H * DH)
    var v_masked_buf = ctx.enqueue_create_buffer[DType.float32](B * S * D)
    var fsmn_conv_buf = ctx.enqueue_create_buffer[DType.float32](B * S * D)
    var fsm_mem_buf = ctx.enqueue_create_buffer[DType.float32](B * S * D)
    # For attention itself we want Q/K/V in (B, H, S, Dh) layout (used by cross_* kernels).
    var q_bhsd_buf = ctx.enqueue_create_buffer[DType.float32](B * H * S * DH)
    var k_bhsd_buf = ctx.enqueue_create_buffer[DType.float32](B * H * S * DH)
    var v_bhsd_buf = ctx.enqueue_create_buffer[DType.float32](B * H * S * DH)
    var logits_buf = ctx.enqueue_create_buffer[DType.float32](B * H * S * S)
    var probs_buf = ctx.enqueue_create_buffer[DType.float32](B * H * S * S)
    var av_buf = ctx.enqueue_create_buffer[DType.float32](B * H * S * DH)
    var comb_buf = ctx.enqueue_create_buffer[DType.float32](B * S * D)
    var attn_out_buf = ctx.enqueue_create_buffer[DType.float32](B * S * D)
    var final_out_buf = ctx.enqueue_create_buffer[DType.float32](B * S * D)

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

    # TileTensors.
    var x_tt = TileTensor(x_buf, x_layout)
    var cos_tt = TileTensor(cos_buf, cs_layout)
    var sin_tt = TileTensor(sin_buf, cs_layout)
    var mp_tt = TileTensor(mp_buf, mp_layout)
    var q_w_tt = TileTensor(q_w, lin_w_layout)
    var q_b_tt = TileTensor(q_b, lin_b_layout)
    var k_w_tt = TileTensor(k_w, lin_w_layout)
    var k_b_tt = TileTensor(k_b_buf, lin_b_layout)
    var v_w_tt = TileTensor(v_w, lin_w_layout)
    var v_b_tt = TileTensor(v_b, lin_b_layout)
    var out_w_tt = TileTensor(out_w, lin_w_layout)
    var out_b_tt = TileTensor(out_b, lin_b_layout)
    var fsmn_w_tt = TileTensor(fsmn_w, fsmn_w_layout)
    var q_lin_tt = TileTensor(q_lin_buf, x_layout)
    var k_lin_tt = TileTensor(k_lin_buf, x_layout)
    var v_lin_tt = TileTensor(v_lin_buf, x_layout)
    var q_4d_tt = TileTensor(q_4d_buf, x4_layout)
    var k_4d_tt = TileTensor(k_4d_buf, x4_layout)
    var v_4d_tt = TileTensor(v_4d_buf, x4_layout)
    var q_rope_tt = TileTensor(q_rope_buf, x4_layout)
    var k_rope_tt = TileTensor(k_rope_buf, x4_layout)
    var v_masked_tt = TileTensor(v_masked_buf, x_layout)
    var fsmn_conv_tt = TileTensor(fsmn_conv_buf, x_layout)
    var fsm_mem_tt = TileTensor(fsm_mem_buf, x_layout)
    var q_bhsd_tt = TileTensor(q_bhsd_buf, bhsd_layout)
    var k_bhsd_tt = TileTensor(k_bhsd_buf, bhsd_layout)
    var v_bhsd_tt = TileTensor(v_bhsd_buf, bhsd_layout)
    var logits_tt = TileTensor(logits_buf, logits_layout)
    var probs_tt = TileTensor(probs_buf, logits_layout)
    var av_tt = TileTensor(av_buf, bhsd_layout)
    var comb_tt = TileTensor(comb_buf, x_layout)
    var attn_out_tt = TileTensor(attn_out_buf, x_layout)
    var final_out_tt = TileTensor(final_out_buf, x_layout)

    # ---- Linear Q/K/V ----
    comptime klin = linear_kernel[
        DType.float32, type_of(x_layout), type_of(lin_w_layout),
        type_of(lin_b_layout), type_of(x_layout),
        True, BLOCK,
    ]
    ctx.enqueue_function[klin, klin](
        q_lin_tt, x_tt, q_w_tt, q_b_tt, B, S, D, D,
        grid_dim=B * S, block_dim=BLOCK,
    )
    ctx.enqueue_function[klin, klin](
        k_lin_tt, x_tt, k_w_tt, k_b_tt, B, S, D, D,
        grid_dim=B * S, block_dim=BLOCK,
    )
    ctx.enqueue_function[klin, klin](
        v_lin_tt, x_tt, v_w_tt, v_b_tt, B, S, D, D,
        grid_dim=B * S, block_dim=BLOCK,
    )

    # ---- Reshape (B, S, D) → (B, S, H, Dh) ----
    # We can do this with a trivial copy kernel; since the memory layout is
    # row-major, (B, S, D) and (B, S, H, Dh) are bit-identical when D = H*Dh
    # and rows are contiguous. Just rebind via a separate buffer copy.
    ctx.enqueue_copy(q_4d_buf, q_lin_buf)
    ctx.enqueue_copy(k_4d_buf, k_lin_buf)
    ctx.enqueue_copy(v_4d_buf, v_lin_buf)

    # ---- RoPE on Q and K ----
    comptime krope = rope_s3tokenizer_kernel[
        DType.float32, type_of(x4_layout), type_of(cs_layout),
        type_of(x4_layout), DH, HALF,
    ]
    ctx.enqueue_function[krope, krope](
        q_rope_tt, q_4d_tt, cos_tt, sin_tt, S, H,
        grid_dim=B * S * H, block_dim=DH,
    )
    ctx.enqueue_function[krope, krope](
        k_rope_tt, k_4d_tt, cos_tt, sin_tt, S, H,
        grid_dim=B * S * H, block_dim=DH,
    )

    # ---- FSMN memory branch ----
    # v_masked = v_lin * mask_pad
    comptime kmul = fsmn_memory_kernel[
        DType.float32, type_of(x_layout), type_of(x_layout),
        type_of(mp_layout), type_of(x_layout), BLOCK,
    ]
    # v_masked = v_lin * mask_pad.
    comptime kmm = multiply_mask_3d_kernel[
        DType.float32, type_of(x_layout), type_of(mp_layout),
        type_of(x_layout), BLOCK,
    ]
    ctx.enqueue_function[kmm, kmm](
        v_masked_tt, v_lin_tt, mp_tt, B, S, D,
        grid_dim=B * S, block_dim=BLOCK,
    )

    # Depthwise conv1d over (B, S, D) with center-padded 31-tap kernel.
    comptime kfconv = fsmn_depthwise_conv_kernel[
        DType.float32, type_of(x_layout), type_of(fsmn_w_layout),
        type_of(x_layout), KSIZE, LEFT, RIGHT, BLOCK,
    ]
    ctx.enqueue_function[kfconv, kfconv](
        fsmn_conv_tt, v_masked_tt, fsmn_w_tt, B, S, D,
        grid_dim=B * S, block_dim=BLOCK,
    )

    # fsm_mem = (fsmn_conv + v_masked) * mp.
    ctx.enqueue_function[kmul, kmul](
        fsm_mem_tt, fsmn_conv_tt, v_masked_tt, mp_tt, B, S, D,
        grid_dim=B * S, block_dim=BLOCK,
    )

    # ---- Permute (B, S, H, Dh) → (B, H, S, Dh) for Q, K, V ----
    comptime kpb = permute_bshd_to_bhsd_kernel[
        DType.float32, type_of(x4_layout), type_of(bhsd_layout),
        H, DH,
    ]
    ctx.enqueue_function[kpb, kpb](
        q_bhsd_tt, q_rope_tt, B, S,
        grid_dim=B * S * H, block_dim=DH,
    )
    ctx.enqueue_function[kpb, kpb](
        k_bhsd_tt, k_rope_tt, B, S,
        grid_dim=B * S * H, block_dim=DH,
    )
    ctx.enqueue_function[kpb, kpb](
        v_bhsd_tt, v_4d_tt, B, S,
        grid_dim=B * S * H, block_dim=DH,
    )

    # ---- Scale Q and K by Dh^-0.25 ----
    comptime scale_amt: Float32 = 1.0 / sqrt(sqrt(Float32(DH)))   # Dh^-0.25
    comptime ksc = scale_4d_kernel[
        DType.float32, type_of(bhsd_layout), BLOCK,
    ]
    ctx.enqueue_function[ksc, ksc](
        q_bhsd_tt, B, H, S, DH, scale_amt,
        grid_dim=B * H * S, block_dim=BLOCK,
    )
    ctx.enqueue_function[ksc, ksc](
        k_bhsd_tt, B, H, S, DH, scale_amt,
        grid_dim=B * H * S, block_dim=BLOCK,
    )

    # ---- QK^T with scale=1 (we already scaled) ----
    comptime kqkt = cross_qkt_kernel[
        DType.float32, type_of(bhsd_layout), type_of(bhsd_layout),
        type_of(logits_layout), DH, S,
    ]
    ctx.enqueue_function[kqkt, kqkt](
        logits_tt, q_bhsd_tt, k_bhsd_tt, H, S, Float32(1.0),
        grid_dim=B * H * S, block_dim=S,
    )
    # Softmax (no mask bias).
    comptime ksm = cross_softmax_kernel[
        DType.float32, type_of(logits_layout), type_of(logits_layout),
        S, BLOCK,
    ]
    ctx.enqueue_function[ksm, ksm](
        probs_tt, logits_tt, H, S,
        grid_dim=B * H * S, block_dim=BLOCK,
    )
    # P @ V.
    comptime kav = cross_av_kernel[
        DType.float32, type_of(logits_layout), type_of(bhsd_layout),
        type_of(bhsd_layout), S, DH,
    ]
    ctx.enqueue_function[kav, kav](
        av_tt, probs_tt, v_bhsd_tt, H, S,
        grid_dim=B * H * S, block_dim=DH,
    )

    # Combine heads (B, H, S, Dh) → (B, S, D).
    comptime kcomb = permute_bhsd_to_bsd_kernel[
        DType.float32, type_of(bhsd_layout), type_of(x_layout),
        H, DH,
    ]
    ctx.enqueue_function[kcomb, kcomb](
        comb_tt, av_tt, B, S,
        grid_dim=B * S * H, block_dim=DH,
    )

    # ---- out Linear ----
    ctx.enqueue_function[klin, klin](
        attn_out_tt, comb_tt, out_w_tt, out_b_tt, B, S, D, D,
        grid_dim=B * S, block_dim=BLOCK,
    )

    # ---- final = attn_out + fsm_mem ----
    comptime kadd = add_3d_kernel[
        DType.float32, type_of(x_layout), BLOCK,
    ]
    ctx.enqueue_function[kadd, kadd](
        final_out_tt, attn_out_tt, fsm_mem_tt, B, S, D,
        grid_dim=B * S, block_dim=BLOCK,
    )
    ctx.synchronize()


    var n_out = B * S * D
    var max_abs: Float32 = 0.0
    var max_rel: Float32 = 0.0
    with final_out_buf.map_to_host() as h:
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
                print("fsmn[", i, "]: mojo=", h[i], "  torch=", exp.data[i], "  diff=", d)
    print("FSMN attention — max abs:", max_abs, "  max_rel:", max_rel)
    assert_almost_equal(max_abs, 0.0, atol=2.0e-4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
