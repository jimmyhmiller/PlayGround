"""Parity test for Perceiver resampler (32 queries × 1024, 4-head)."""
from std.math import sqrt
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import load_fp32
from layernorm import layernorm_kernel, linear_kernel
from perceiver import (
    cross_qkt_kernel, cross_softmax_kernel, cross_av_kernel,
    split_heads_kernel, combine_heads_kernel, add_3d_kernel,
)


comptime B = 1
comptime SQ = 32
comptime TK = 150
comptime D = 1024
comptime H = 4
comptime DH = D // H
comptime BLOCK = 128


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as hh:
        for i in range(n):
            hh[i] = data[i]


def upload_w(mut ctx: DeviceContext, fix: String, name: String) raises -> DeviceBuffer[DType.float32]:
    var t = load_fp32(fix + name)
    var n = len(t.data)
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    upload(buf, t.data, n)
    return buf^


def attention_block_2[
    SQ_K: Int, SK_K: Int,
](
    mut ctx: DeviceContext,
    mut x_q_buf: DeviceBuffer[DType.float32],   # (B, SQ_K, D)
    mut x_kv_buf: DeviceBuffer[DType.float32],  # (B, SK_K, D)
    mut norm_w_buf: DeviceBuffer[DType.float32],
    mut norm_b_buf: DeviceBuffer[DType.float32],
    mut to_q_w_buf: DeviceBuffer[DType.float32],
    mut to_q_b_buf: DeviceBuffer[DType.float32],
    mut to_k_w_buf: DeviceBuffer[DType.float32],
    mut to_k_b_buf: DeviceBuffer[DType.float32],
    mut to_v_w_buf: DeviceBuffer[DType.float32],
    mut to_v_b_buf: DeviceBuffer[DType.float32],
    mut proj_w_buf: DeviceBuffer[DType.float32],
    mut proj_b_buf: DeviceBuffer[DType.float32],
    mut out_buf: DeviceBuffer[DType.float32],  # (B, SQ_K, D) — output
) raises:
    """Run AttentionBlock2: LN both inputs → Q/K/V Linear → MH-SDPA → proj_out → +x_q residual."""
    # Layouts for this call.
    comptime xq_layout = row_major[B, SQ_K, D]()
    comptime xkv_layout = row_major[B, SK_K, D]()
    comptime ln_w_layout = row_major[D]()
    comptime q_4d = row_major[B, H, SQ_K, DH]()
    comptime k_4d = row_major[B, H, SK_K, DH]()
    comptime logits_layout = row_major[B, H, SQ_K, SK_K]()
    comptime ln_p_layout = row_major[D, D]()
    comptime ln_pb_layout = row_major[D]()

    # Buffers we need for intermediates.
    var qn_buf = ctx.enqueue_create_buffer[DType.float32](B * SQ_K * D)
    var kn_buf = ctx.enqueue_create_buffer[DType.float32](B * SK_K * D)
    var q_lin_buf = ctx.enqueue_create_buffer[DType.float32](B * SQ_K * D)
    var k_lin_buf = ctx.enqueue_create_buffer[DType.float32](B * SK_K * D)
    var v_lin_buf = ctx.enqueue_create_buffer[DType.float32](B * SK_K * D)
    var q_h_buf = ctx.enqueue_create_buffer[DType.float32](B * H * SQ_K * DH)
    var k_h_buf = ctx.enqueue_create_buffer[DType.float32](B * H * SK_K * DH)
    var v_h_buf = ctx.enqueue_create_buffer[DType.float32](B * H * SK_K * DH)
    var logits_buf = ctx.enqueue_create_buffer[DType.float32](B * H * SQ_K * SK_K)
    var probs_buf = ctx.enqueue_create_buffer[DType.float32](B * H * SQ_K * SK_K)
    var av_buf = ctx.enqueue_create_buffer[DType.float32](B * H * SQ_K * DH)
    var comb_buf = ctx.enqueue_create_buffer[DType.float32](B * SQ_K * D)
    var proj_buf = ctx.enqueue_create_buffer[DType.float32](B * SQ_K * D)

    var x_q_t = TileTensor(x_q_buf, xq_layout)
    var x_kv_t = TileTensor(x_kv_buf, xkv_layout)
    var qn_t = TileTensor(qn_buf, xq_layout)
    var kn_t = TileTensor(kn_buf, xkv_layout)
    var norm_w_t = TileTensor(norm_w_buf, ln_w_layout)
    var norm_b_t = TileTensor(norm_b_buf, ln_w_layout)

    # LayerNorm both inputs.
    comptime kln_q = layernorm_kernel[
        DType.float32, type_of(xq_layout), type_of(ln_w_layout), type_of(xq_layout), BLOCK,
    ]
    ctx.enqueue_function[kln_q, kln_q](
        qn_t, x_q_t, norm_w_t, norm_b_t,
        B, SQ_K, D, Float32(1.0e-5),
        grid_dim=B * SQ_K, block_dim=BLOCK,
    )
    comptime kln_k = layernorm_kernel[
        DType.float32, type_of(xkv_layout), type_of(ln_w_layout), type_of(xkv_layout), BLOCK,
    ]
    ctx.enqueue_function[kln_k, kln_k](
        kn_t, x_kv_t, norm_w_t, norm_b_t,
        B, SK_K, D, Float32(1.0e-5),
        grid_dim=B * SK_K, block_dim=BLOCK,
    )

    # to_q, to_k, to_v Linear projections.
    var q_lin_t = TileTensor(q_lin_buf, xq_layout)
    var k_lin_t = TileTensor(k_lin_buf, xkv_layout)
    var v_lin_t = TileTensor(v_lin_buf, xkv_layout)
    var to_q_w_t = TileTensor(to_q_w_buf, ln_p_layout)
    var to_q_b_t = TileTensor(to_q_b_buf, ln_pb_layout)
    var to_k_w_t = TileTensor(to_k_w_buf, ln_p_layout)
    var to_k_b_t = TileTensor(to_k_b_buf, ln_pb_layout)
    var to_v_w_t = TileTensor(to_v_w_buf, ln_p_layout)
    var to_v_b_t = TileTensor(to_v_b_buf, ln_pb_layout)

    comptime klq = linear_kernel[
        DType.float32, type_of(xq_layout), type_of(ln_p_layout),
        type_of(ln_pb_layout), type_of(xq_layout),
        True, BLOCK,
    ]
    ctx.enqueue_function[klq, klq](
        q_lin_t, qn_t, to_q_w_t, to_q_b_t,
        B, SQ_K, D, D,
        grid_dim=B * SQ_K, block_dim=BLOCK,
    )
    comptime klk = linear_kernel[
        DType.float32, type_of(xkv_layout), type_of(ln_p_layout),
        type_of(ln_pb_layout), type_of(xkv_layout),
        True, BLOCK,
    ]
    ctx.enqueue_function[klk, klk](
        k_lin_t, kn_t, to_k_w_t, to_k_b_t,
        B, SK_K, D, D,
        grid_dim=B * SK_K, block_dim=BLOCK,
    )
    ctx.enqueue_function[klk, klk](
        v_lin_t, kn_t, to_v_w_t, to_v_b_t,
        B, SK_K, D, D,
        grid_dim=B * SK_K, block_dim=BLOCK,
    )

    # Split heads: (B, S, D) → (B, H, S, DH).
    var q_h_t = TileTensor(q_h_buf, q_4d)
    var k_h_t = TileTensor(k_h_buf, k_4d)
    var v_h_t = TileTensor(v_h_buf, k_4d)

    comptime ksp_q = split_heads_kernel[
        DType.float32, type_of(xq_layout), type_of(q_4d), DH,
    ]
    ctx.enqueue_function[ksp_q, ksp_q](
        q_h_t, q_lin_t, B, SQ_K, H,
        grid_dim=B * SQ_K * H, block_dim=DH,
    )
    comptime ksp_k = split_heads_kernel[
        DType.float32, type_of(xkv_layout), type_of(k_4d), DH,
    ]
    ctx.enqueue_function[ksp_k, ksp_k](
        k_h_t, k_lin_t, B, SK_K, H,
        grid_dim=B * SK_K * H, block_dim=DH,
    )
    ctx.enqueue_function[ksp_k, ksp_k](
        v_h_t, v_lin_t, B, SK_K, H,
        grid_dim=B * SK_K * H, block_dim=DH,
    )

    # Cross QK^T with scaling.
    var logits_t = TileTensor(logits_buf, logits_layout)
    var probs_t = TileTensor(probs_buf, logits_layout)
    comptime scale: Float32 = 1.0 / sqrt(Float32(DH))
    comptime kqkt = cross_qkt_kernel[
        DType.float32, type_of(q_4d), type_of(k_4d),
        type_of(logits_layout), DH, SK_K,
    ]
    ctx.enqueue_function[kqkt, kqkt](
        logits_t, q_h_t, k_h_t, H, SQ_K, scale,
        grid_dim=B * H * SQ_K, block_dim=SK_K,
    )
    # Softmax.
    comptime ksm = cross_softmax_kernel[
        DType.float32, type_of(logits_layout), type_of(logits_layout),
        SK_K, BLOCK,
    ]
    ctx.enqueue_function[ksm, ksm](
        probs_t, logits_t, H, SQ_K,
        grid_dim=B * H * SQ_K, block_dim=BLOCK,
    )
    # P @ V.
    var av_t = TileTensor(av_buf, q_4d)
    comptime kav = cross_av_kernel[
        DType.float32, type_of(logits_layout), type_of(k_4d),
        type_of(q_4d), SK_K, DH,
    ]
    ctx.enqueue_function[kav, kav](
        av_t, probs_t, v_h_t, H, SQ_K,
        grid_dim=B * H * SQ_K, block_dim=DH,
    )
    # Combine heads.
    var comb_t = TileTensor(comb_buf, xq_layout)
    comptime kch = combine_heads_kernel[
        DType.float32, type_of(q_4d), type_of(xq_layout), DH,
    ]
    ctx.enqueue_function[kch, kch](
        comb_t, av_t, B, SQ_K, H,
        grid_dim=B * SQ_K * H, block_dim=DH,
    )
    # proj_out Linear.
    var proj_t = TileTensor(proj_buf, xq_layout)
    var proj_w_t = TileTensor(proj_w_buf, ln_p_layout)
    var proj_b_t = TileTensor(proj_b_buf, ln_pb_layout)
    comptime klp = linear_kernel[
        DType.float32, type_of(xq_layout), type_of(ln_p_layout),
        type_of(ln_pb_layout), type_of(xq_layout),
        True, BLOCK,
    ]
    ctx.enqueue_function[klp, klp](
        proj_t, comb_t, proj_w_t, proj_b_t,
        B, SQ_K, D, D,
        grid_dim=B * SQ_K, block_dim=BLOCK,
    )
    # Residual: out = x_q + proj.
    var out_t = TileTensor(out_buf, xq_layout)
    comptime kadd = add_3d_kernel[DType.float32, type_of(xq_layout), BLOCK]
    ctx.enqueue_function[kadd, kadd](
        out_t, x_q_t, proj_t, B, SQ_K, D,
        grid_dim=B * SQ_K, block_dim=BLOCK,
    )


def test_perceiver() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/perceiver/"
    var ctx = DeviceContext()

    var h_t = load_fp32(fix + "h.bin")
    var pq = load_fp32(fix + "pre_attention_query.bin")
    var exp = load_fp32(fix + "out.bin")
    var norm_w = load_fp32(fix + "attn_norm_w.bin")
    var norm_b = load_fp32(fix + "attn_norm_b.bin")
    var to_q_w = load_fp32(fix + "to_q_w.bin")
    var to_q_b = load_fp32(fix + "to_q_b.bin")
    var to_k_w = load_fp32(fix + "to_k_w.bin")
    var to_k_b = load_fp32(fix + "to_k_b.bin")
    var to_v_w = load_fp32(fix + "to_v_w.bin")
    var to_v_b = load_fp32(fix + "to_v_b.bin")
    var proj_w = load_fp32(fix + "proj_out_w.bin")
    var proj_b = load_fp32(fix + "proj_out_b.bin")

    var h_buf = ctx.enqueue_create_buffer[DType.float32](B * TK * D)
    var q_buf = ctx.enqueue_create_buffer[DType.float32](B * SQ * D)
    var pre_buf = ctx.enqueue_create_buffer[DType.float32](B * SQ * D)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](B * SQ * D)
    upload(h_buf, h_t.data, B * TK * D)
    # Broadcast pq (1, 32, D) → q_buf (B, 32, D). For B=1 this is just copy.
    upload(q_buf, pq.data, B * SQ * D)

    var norm_w_buf = ctx.enqueue_create_buffer[DType.float32](D)
    upload(norm_w_buf, norm_w.data, D)
    var norm_b_buf = ctx.enqueue_create_buffer[DType.float32](D)
    upload(norm_b_buf, norm_b.data, D)
    var to_q_w_buf = ctx.enqueue_create_buffer[DType.float32](D * D)
    upload(to_q_w_buf, to_q_w.data, D * D)
    var to_q_b_buf = ctx.enqueue_create_buffer[DType.float32](D)
    upload(to_q_b_buf, to_q_b.data, D)
    var to_k_w_buf = ctx.enqueue_create_buffer[DType.float32](D * D)
    upload(to_k_w_buf, to_k_w.data, D * D)
    var to_k_b_buf = ctx.enqueue_create_buffer[DType.float32](D)
    upload(to_k_b_buf, to_k_b.data, D)
    var to_v_w_buf = ctx.enqueue_create_buffer[DType.float32](D * D)
    upload(to_v_w_buf, to_v_w.data, D * D)
    var to_v_b_buf = ctx.enqueue_create_buffer[DType.float32](D)
    upload(to_v_b_buf, to_v_b.data, D)
    var proj_w_buf = ctx.enqueue_create_buffer[DType.float32](D * D)
    upload(proj_w_buf, proj_w.data, D * D)
    var proj_b_buf = ctx.enqueue_create_buffer[DType.float32](D)
    upload(proj_b_buf, proj_b.data, D)

    # Stage 1: cross-attention.  q (B, 32, D), kv = h (B, 150, D).
    attention_block_2[SQ, TK](
        ctx, q_buf, h_buf,
        norm_w_buf, norm_b_buf,
        to_q_w_buf, to_q_b_buf,
        to_k_w_buf, to_k_b_buf,
        to_v_w_buf, to_v_b_buf,
        proj_w_buf, proj_b_buf,
        pre_buf,
    )
    # Stage 2: self-attention.  q == kv == pre_buf — clone to avoid aliasing.
    var pre_clone = ctx.enqueue_create_buffer[DType.float32](B * SQ * D)
    ctx.enqueue_copy(pre_clone, pre_buf)
    attention_block_2[SQ, SQ](
        ctx, pre_buf, pre_clone,
        norm_w_buf, norm_b_buf,
        to_q_w_buf, to_q_b_buf,
        to_k_w_buf, to_k_b_buf,
        to_v_w_buf, to_v_b_buf,
        proj_w_buf, proj_b_buf,
        out_buf,
    )
    ctx.synchronize()

    var n_out = B * SQ * D
    var max_abs: Float32 = 0.0
    var max_rel: Float32 = 0.0
    with out_buf.map_to_host() as hh:
        for i in range(n_out):
            var d = hh[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            var r0 = exp.data[i]
            if r0 < 0.0: r0 = -r0
            if r0 > 1.0e-3:
                var r = d / r0
                if r > max_rel: max_rel = r
            if i < 8:
                print("perc[", i, "]: mojo=", hh[i], "  torch=", exp.data[i], "  diff=", d)
    print("Perceiver — max abs:", max_abs, "  max_rel:", max_rel)
    assert_almost_equal(max_abs, 0.0, atol=2.0e-3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
