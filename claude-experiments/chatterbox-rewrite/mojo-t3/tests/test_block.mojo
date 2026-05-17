"""
Single transformer block parity test vs HF LlamaDecoderLayer with real T3
layer-0 weights.

Pipeline:
  norm_x = rmsnorm(x, in_norm_w)               # (B, S, H_total)
  q_flat = norm_x @ q_w                         # (B, S, H_total)
  k_flat = norm_x @ k_w
  v_flat = norm_x @ v_w
  q,k,v  = reshape (B, S, H, D) → permute → (B, H, S, D)
  q,k    = rope(q, k, cos, sin)
  attn   = sdpa(q, k, v, mask)                  # (B, H, S, D)
  attn   = permute → (B, S, H, D) → flatten → (B, S, H_total)
  attn   = attn @ o_w                            # (B, S, H_total)
  x      = x + attn                              # residual
  norm_x = rmsnorm(x, post_norm_w)
  mlp_out = mlp_pipeline(norm_x)                 # (B, S, H_total)
  out    = x + mlp_out                           # residual
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.gpu.host import DeviceContext
from std.runtime.asyncrt import DeviceContextPtr
from layout import TileTensor, row_major
from linalg.matmul import matmul

from fixture import Tensor, TensorBF16, load_fp32, load_bf16
from rmsnorm import rmsnorm_kernel
from rope import rope_kernel
from sdpa import qk_scaled_kernel, softmax_kernel, av_kernel
from mlp import silu_mul_kernel
from util_kernels import add_kernel, bshd_to_bhsd_kernel, bhsd_to_bshd_kernel


# T3 Llama config (layer 0).
comptime BATCH = 1
comptime SEQ = 16
comptime N_HEADS = 16
comptime HEAD_DIM = 64
comptime HIDDEN = N_HEADS * HEAD_DIM        # 1024
comptime INTERMEDIATE = 4096
comptime SCALE: Float32 = 0.125              # 1/sqrt(64)
comptime EPS: Float32 = 1.0e-5
comptime ROPE_HALF = HEAD_DIM // 2

# Block sizes for kernels.
comptime RMS_BLOCK = 256
comptime ROWS = BATCH * SEQ                  # flatten (B, S) → rows
comptime POINTWISE_BLOCK = 256
comptime SOFTMAX_BLOCK = 32

# Layouts.
comptime x_2d_layout = row_major[ROWS, HIDDEN]()
comptime x_3d_layout = row_major[BATCH, SEQ, HIDDEN]()           # for matmul-out / RMSNorm
comptime w_hh_layout = row_major[HIDDEN, HIDDEN]()                # q/k/v/o weights
comptime w_in_inter_layout = row_major[HIDDEN, INTERMEDIATE]()
comptime w_inter_in_layout = row_major[INTERMEDIATE, HIDDEN]()
comptime intermediate_layout = row_major[ROWS, INTERMEDIATE]()
comptime bshd_layout = row_major[BATCH, SEQ, N_HEADS, HEAD_DIM]() # post q_proj reshape
comptime bhsd_layout = row_major[BATCH, N_HEADS, SEQ, HEAD_DIM]() # post-permute, SDPA input
comptime cs_layout = row_major[BATCH, SEQ, HEAD_DIM]()
comptime mask_layout = row_major[SEQ, SEQ]()
comptime ss_layout = row_major[BATCH, N_HEADS, SEQ, SEQ]()
comptime norm_w_layout = row_major[HIDDEN]()


def test_block_fp32() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/block/"
    var x = load_fp32(fix + "x_fp32.bin")
    var cos = load_fp32(fix + "cos_fp32.bin")
    var sin = load_fp32(fix + "sin_fp32.bin")
    var mask = load_fp32(fix + "mask_fp32.bin")
    var in_norm = load_fp32(fix + "in_norm_fp32.bin")
    var post_norm = load_fp32(fix + "post_norm_fp32.bin")
    var qw = load_fp32(fix + "qw_fp32.bin")
    var kw = load_fp32(fix + "kw_fp32.bin")
    var vw = load_fp32(fix + "vw_fp32.bin")
    var ow = load_fp32(fix + "ow_fp32.bin")
    var gate_w = load_fp32(fix + "gate_w_fp32.bin")
    var up_w = load_fp32(fix + "up_w_fp32.bin")
    var down_w = load_fp32(fix + "down_w_fp32.bin")
    var exp = load_fp32(fix + "expected_fp32.bin")

    assert_equal(x.shape[0] * x.shape[1], ROWS)
    assert_equal(x.shape[2], HIDDEN)

    var n_x = ROWS * HIDDEN
    var n_cs = BATCH * SEQ * HEAD_DIM
    var n_mask = SEQ * SEQ
    var n_qkv_flat = ROWS * HIDDEN              # same as n_x
    var n_qkv_perm = BATCH * N_HEADS * SEQ * HEAD_DIM  # same elements, different layout
    var n_ss = BATCH * N_HEADS * SEQ * SEQ
    var n_inter = ROWS * INTERMEDIATE

    var ctx = DeviceContext()
    var dctx = DeviceContextPtr(ctx)

    # --- Allocate device buffers ---
    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var cos_buf = ctx.enqueue_create_buffer[DType.float32](n_cs)
    var sin_buf = ctx.enqueue_create_buffer[DType.float32](n_cs)
    var mask_buf = ctx.enqueue_create_buffer[DType.float32](n_mask)
    var in_norm_buf = ctx.enqueue_create_buffer[DType.float32](HIDDEN)
    var post_norm_buf = ctx.enqueue_create_buffer[DType.float32](HIDDEN)
    var qw_buf = ctx.enqueue_create_buffer[DType.float32](HIDDEN * HIDDEN)
    var kw_buf = ctx.enqueue_create_buffer[DType.float32](HIDDEN * HIDDEN)
    var vw_buf = ctx.enqueue_create_buffer[DType.float32](HIDDEN * HIDDEN)
    var ow_buf = ctx.enqueue_create_buffer[DType.float32](HIDDEN * HIDDEN)
    var gw_buf = ctx.enqueue_create_buffer[DType.float32](HIDDEN * INTERMEDIATE)
    var uw_buf = ctx.enqueue_create_buffer[DType.float32](HIDDEN * INTERMEDIATE)
    var dw_buf = ctx.enqueue_create_buffer[DType.float32](INTERMEDIATE * HIDDEN)

    # Working buffers.
    var norm_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var q_flat_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv_flat)
    var k_flat_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv_flat)
    var v_flat_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv_flat)
    var q_perm_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv_perm)
    var k_perm_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv_perm)
    var v_perm_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv_perm)
    var q_rot_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv_perm)
    var k_rot_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv_perm)
    var logits_buf = ctx.enqueue_create_buffer[DType.float32](n_ss)
    var probs_buf = ctx.enqueue_create_buffer[DType.float32](n_ss)
    var attn_perm_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv_perm)
    var attn_flat_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv_flat)
    var attn_proj_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var post_attn_buf = ctx.enqueue_create_buffer[DType.float32](n_x)  # residual 1 result
    var post_norm_out_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var gate_buf = ctx.enqueue_create_buffer[DType.float32](n_inter)
    var up_buf = ctx.enqueue_create_buffer[DType.float32](n_inter)
    var hidden_act_buf = ctx.enqueue_create_buffer[DType.float32](n_inter)
    var mlp_out_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_x)

    # --- Stage all inputs to device ---
    with x_buf.map_to_host() as h:
        for i in range(n_x): h[i] = x.data[i]
    with cos_buf.map_to_host() as h:
        for i in range(n_cs): h[i] = cos.data[i]
    with sin_buf.map_to_host() as h:
        for i in range(n_cs): h[i] = sin.data[i]
    with mask_buf.map_to_host() as h:
        for i in range(n_mask): h[i] = mask.data[i]
    with in_norm_buf.map_to_host() as h:
        for i in range(HIDDEN): h[i] = in_norm.data[i]
    with post_norm_buf.map_to_host() as h:
        for i in range(HIDDEN): h[i] = post_norm.data[i]
    with qw_buf.map_to_host() as h:
        for i in range(HIDDEN * HIDDEN): h[i] = qw.data[i]
    with kw_buf.map_to_host() as h:
        for i in range(HIDDEN * HIDDEN): h[i] = kw.data[i]
    with vw_buf.map_to_host() as h:
        for i in range(HIDDEN * HIDDEN): h[i] = vw.data[i]
    with ow_buf.map_to_host() as h:
        for i in range(HIDDEN * HIDDEN): h[i] = ow.data[i]
    with gw_buf.map_to_host() as h:
        for i in range(HIDDEN * INTERMEDIATE): h[i] = gate_w.data[i]
    with uw_buf.map_to_host() as h:
        for i in range(HIDDEN * INTERMEDIATE): h[i] = up_w.data[i]
    with dw_buf.map_to_host() as h:
        for i in range(INTERMEDIATE * HIDDEN): h[i] = down_w.data[i]

    # --- Build TileTensor views ---
    # The same buffer can be viewed under different shapes (no copy).
    var x_2d = TileTensor(x_buf, x_2d_layout)
    var norm_2d = TileTensor(norm_buf, x_2d_layout)
    var in_norm_t = TileTensor(in_norm_buf, norm_w_layout)
    var post_norm_t = TileTensor(post_norm_buf, norm_w_layout)

    var qw_t = TileTensor(qw_buf, w_hh_layout)
    var kw_t = TileTensor(kw_buf, w_hh_layout)
    var vw_t = TileTensor(vw_buf, w_hh_layout)
    var ow_t = TileTensor(ow_buf, w_hh_layout)
    var gw_t = TileTensor(gw_buf, w_in_inter_layout)
    var uw_t = TileTensor(uw_buf, w_in_inter_layout)
    var dw_t = TileTensor(dw_buf, w_inter_in_layout)

    var q_flat_2d = TileTensor(q_flat_buf, x_2d_layout)
    var k_flat_2d = TileTensor(k_flat_buf, x_2d_layout)
    var v_flat_2d = TileTensor(v_flat_buf, x_2d_layout)
    var q_flat_4d = TileTensor(q_flat_buf, bshd_layout)
    var k_flat_4d = TileTensor(k_flat_buf, bshd_layout)
    var v_flat_4d = TileTensor(v_flat_buf, bshd_layout)
    var q_perm = TileTensor(q_perm_buf, bhsd_layout)
    var k_perm = TileTensor(k_perm_buf, bhsd_layout)
    var v_perm = TileTensor(v_perm_buf, bhsd_layout)
    var q_rot = TileTensor(q_rot_buf, bhsd_layout)
    var k_rot = TileTensor(k_rot_buf, bhsd_layout)

    var cos_t = TileTensor(cos_buf, cs_layout)
    var sin_t = TileTensor(sin_buf, cs_layout)
    var mask_t = TileTensor(mask_buf, mask_layout)
    var logits_t = TileTensor(logits_buf, ss_layout)
    var probs_t = TileTensor(probs_buf, ss_layout)

    var attn_perm = TileTensor(attn_perm_buf, bhsd_layout)
    var attn_flat_4d = TileTensor(attn_flat_buf, bshd_layout)
    var attn_flat_2d = TileTensor(attn_flat_buf, x_2d_layout)
    var attn_proj_2d = TileTensor(attn_proj_buf, x_2d_layout)

    var post_attn_2d = TileTensor(post_attn_buf, x_2d_layout)
    var post_norm_out_2d = TileTensor(post_norm_out_buf, x_2d_layout)

    var gate_t = TileTensor(gate_buf, intermediate_layout)
    var up_t = TileTensor(up_buf, intermediate_layout)
    var hidden_act_t = TileTensor(hidden_act_buf, intermediate_layout)
    var mlp_out_t = TileTensor(mlp_out_buf, x_2d_layout)
    var out_t = TileTensor(out_buf, x_2d_layout)

    # --- Bind kernels ---
    # RMSNorm kernel expects 2D (rows, hidden); reuse the same buffer as a 2D view.
    comptime rms_k = rmsnorm_kernel[
        DType.float32, type_of(x_2d_layout), type_of(norm_w_layout),
        type_of(x_2d_layout), RMS_BLOCK,
    ]
    comptime bshd_to_bhsd = bshd_to_bhsd_kernel[
        DType.float32, type_of(bshd_layout), type_of(bhsd_layout),
        BATCH, SEQ, N_HEADS, HEAD_DIM,
    ]
    comptime bhsd_to_bshd = bhsd_to_bshd_kernel[
        DType.float32, type_of(bhsd_layout), type_of(bshd_layout),
        BATCH, SEQ, N_HEADS, HEAD_DIM,
    ]
    comptime rope_k = rope_kernel[
        DType.float32, type_of(bhsd_layout), type_of(cs_layout),
        type_of(bhsd_layout), HEAD_DIM, ROPE_HALF,
    ]
    comptime qk_k = qk_scaled_kernel[
        DType.float32, type_of(bhsd_layout), type_of(bhsd_layout),
        type_of(mask_layout), type_of(ss_layout), HEAD_DIM, SEQ,
    ]
    comptime sm_k = softmax_kernel[
        DType.float32, type_of(ss_layout), type_of(ss_layout), SEQ, SOFTMAX_BLOCK,
    ]
    comptime av_k = av_kernel[
        DType.float32, type_of(ss_layout), type_of(bhsd_layout),
        type_of(bhsd_layout), SEQ, HEAD_DIM,
    ]
    comptime add_k = add_kernel[
        DType.float32, type_of(x_2d_layout), type_of(x_2d_layout),
        type_of(x_2d_layout), POINTWISE_BLOCK,
    ]
    comptime silu_k = silu_mul_kernel[
        DType.float32, type_of(intermediate_layout), type_of(intermediate_layout),
        type_of(intermediate_layout), POINTWISE_BLOCK,
    ]

    # --- Pipeline ---

    # 1. norm_x = rmsnorm(x, in_norm_w)
    ctx.enqueue_function[rms_k, rms_k](
        norm_2d, x_2d, in_norm_t, EPS,
        grid_dim=ROWS, block_dim=RMS_BLOCK,
    )

    # 2. q_flat = norm_x @ qw,   k_flat, v_flat similarly
    matmul[target="gpu"](q_flat_2d, norm_2d, qw_t, dctx)
    matmul[target="gpu"](k_flat_2d, norm_2d, kw_t, dctx)
    matmul[target="gpu"](v_flat_2d, norm_2d, vw_t, dctx)

    # 3. permute (B,S,H,D) → (B,H,S,D)
    var n_perm_threads = BATCH * N_HEADS * SEQ
    ctx.enqueue_function[bshd_to_bhsd, bshd_to_bhsd](
        q_perm, q_flat_4d,
        grid_dim=n_perm_threads, block_dim=HEAD_DIM,
    )
    ctx.enqueue_function[bshd_to_bhsd, bshd_to_bhsd](
        k_perm, k_flat_4d,
        grid_dim=n_perm_threads, block_dim=HEAD_DIM,
    )
    ctx.enqueue_function[bshd_to_bhsd, bshd_to_bhsd](
        v_perm, v_flat_4d,
        grid_dim=n_perm_threads, block_dim=HEAD_DIM,
    )

    # 4. RoPE on q and k
    ctx.enqueue_function[rope_k, rope_k](
        q_rot, q_perm, cos_t, sin_t, N_HEADS, SEQ,
        grid_dim=BATCH * N_HEADS * SEQ, block_dim=HEAD_DIM,
    )
    ctx.enqueue_function[rope_k, rope_k](
        k_rot, k_perm, cos_t, sin_t, N_HEADS, SEQ,
        grid_dim=BATCH * N_HEADS * SEQ, block_dim=HEAD_DIM,
    )

    # 5. SDPA: qk → softmax → av
    ctx.enqueue_function[qk_k, qk_k](
        logits_t, q_rot, k_rot, mask_t, N_HEADS, SCALE,
        grid_dim=BATCH * N_HEADS * SEQ, block_dim=SEQ,
    )
    ctx.enqueue_function[sm_k, sm_k](
        probs_t, logits_t, N_HEADS,
        grid_dim=BATCH * N_HEADS * SEQ, block_dim=SOFTMAX_BLOCK,
    )
    ctx.enqueue_function[av_k, av_k](
        attn_perm, probs_t, v_perm, N_HEADS,
        grid_dim=BATCH * N_HEADS * SEQ, block_dim=HEAD_DIM,
    )

    # 6. Permute back (B,H,S,D) → (B,S,H,D)
    ctx.enqueue_function[bhsd_to_bshd, bhsd_to_bshd](
        attn_flat_4d, attn_perm,
        grid_dim=n_perm_threads, block_dim=HEAD_DIM,
    )

    # 7. attn_proj = attn_flat @ ow
    matmul[target="gpu"](attn_proj_2d, attn_flat_2d, ow_t, dctx)

    # 8. post_attn = x + attn_proj (residual)
    ctx.enqueue_function[add_k, add_k](
        post_attn_2d, x_2d, attn_proj_2d, n_x,
        grid_dim=ceildiv(n_x, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )

    # 9. post_norm = rmsnorm(post_attn, post_norm_w)
    ctx.enqueue_function[rms_k, rms_k](
        post_norm_out_2d, post_attn_2d, post_norm_t, EPS,
        grid_dim=ROWS, block_dim=RMS_BLOCK,
    )

    # 10. MLP: gate = norm @ gw, up = norm @ uw, hidden = silu(gate)*up, out = hidden @ dw
    matmul[target="gpu"](gate_t, post_norm_out_2d, gw_t, dctx)
    matmul[target="gpu"](up_t, post_norm_out_2d, uw_t, dctx)
    var n_silu = ROWS * INTERMEDIATE
    ctx.enqueue_function[silu_k, silu_k](
        hidden_act_t, gate_t, up_t, n_silu,
        grid_dim=ceildiv(n_silu, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )
    matmul[target="gpu"](mlp_out_t, hidden_act_t, dw_t, dctx)

    # 11. out = post_attn + mlp_out (residual 2)
    ctx.enqueue_function[add_k, add_k](
        out_t, post_attn_2d, mlp_out_t, n_x,
        grid_dim=ceildiv(n_x, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )

    ctx.synchronize()

    with out_buf.map_to_host() as h:
        for i in range(n_x):
            # Block composes RMSNorm + 4 attn matmuls + RoPE + SDPA + 3 MLP matmuls
            # + 2 adds. Errors compound; budget is conservative.
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-3)


def test_block_bf16() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/block/"
    var x = load_bf16(fix + "x_bf16.bin")
    var cos = load_bf16(fix + "cos_bf16.bin")
    var sin = load_bf16(fix + "sin_bf16.bin")
    var mask = load_bf16(fix + "mask_bf16.bin")
    var in_norm = load_bf16(fix + "in_norm_bf16.bin")
    var post_norm = load_bf16(fix + "post_norm_bf16.bin")
    var qw = load_bf16(fix + "qw_bf16.bin")
    var kw = load_bf16(fix + "kw_bf16.bin")
    var vw = load_bf16(fix + "vw_bf16.bin")
    var ow = load_bf16(fix + "ow_bf16.bin")
    var gate_w = load_bf16(fix + "gate_w_bf16.bin")
    var up_w = load_bf16(fix + "up_w_bf16.bin")
    var down_w = load_bf16(fix + "down_w_bf16.bin")
    var exp = load_bf16(fix + "expected_bf16.bin")

    var n_x = ROWS * HIDDEN
    var n_cs = BATCH * SEQ * HEAD_DIM
    var n_mask = SEQ * SEQ
    var n_qkv_flat = ROWS * HIDDEN
    var n_qkv_perm = BATCH * N_HEADS * SEQ * HEAD_DIM
    var n_ss = BATCH * N_HEADS * SEQ * SEQ
    var n_inter = ROWS * INTERMEDIATE

    var ctx = DeviceContext()
    var dctx = DeviceContextPtr(ctx)

    var x_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_x)
    var cos_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_cs)
    var sin_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_cs)
    var mask_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_mask)
    var in_norm_buf = ctx.enqueue_create_buffer[DType.bfloat16](HIDDEN)
    var post_norm_buf = ctx.enqueue_create_buffer[DType.bfloat16](HIDDEN)
    var qw_buf = ctx.enqueue_create_buffer[DType.bfloat16](HIDDEN * HIDDEN)
    var kw_buf = ctx.enqueue_create_buffer[DType.bfloat16](HIDDEN * HIDDEN)
    var vw_buf = ctx.enqueue_create_buffer[DType.bfloat16](HIDDEN * HIDDEN)
    var ow_buf = ctx.enqueue_create_buffer[DType.bfloat16](HIDDEN * HIDDEN)
    var gw_buf = ctx.enqueue_create_buffer[DType.bfloat16](HIDDEN * INTERMEDIATE)
    var uw_buf = ctx.enqueue_create_buffer[DType.bfloat16](HIDDEN * INTERMEDIATE)
    var dw_buf = ctx.enqueue_create_buffer[DType.bfloat16](INTERMEDIATE * HIDDEN)

    var norm_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_x)
    var q_flat_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv_flat)
    var k_flat_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv_flat)
    var v_flat_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv_flat)
    var q_perm_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv_perm)
    var k_perm_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv_perm)
    var v_perm_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv_perm)
    var q_rot_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv_perm)
    var k_rot_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv_perm)
    var logits_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_ss)
    var probs_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_ss)
    var attn_perm_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv_perm)
    var attn_flat_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv_flat)
    var attn_proj_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_x)
    var post_attn_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_x)
    var post_norm_out_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_x)
    var gate_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_inter)
    var up_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_inter)
    var hidden_act_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_inter)
    var mlp_out_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_x)
    var out_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_x)

    with x_buf.map_to_host() as h:
        for i in range(n_x): h[i] = x.data[i]
    with cos_buf.map_to_host() as h:
        for i in range(n_cs): h[i] = cos.data[i]
    with sin_buf.map_to_host() as h:
        for i in range(n_cs): h[i] = sin.data[i]
    with mask_buf.map_to_host() as h:
        for i in range(n_mask): h[i] = mask.data[i]
    with in_norm_buf.map_to_host() as h:
        for i in range(HIDDEN): h[i] = in_norm.data[i]
    with post_norm_buf.map_to_host() as h:
        for i in range(HIDDEN): h[i] = post_norm.data[i]
    with qw_buf.map_to_host() as h:
        for i in range(HIDDEN * HIDDEN): h[i] = qw.data[i]
    with kw_buf.map_to_host() as h:
        for i in range(HIDDEN * HIDDEN): h[i] = kw.data[i]
    with vw_buf.map_to_host() as h:
        for i in range(HIDDEN * HIDDEN): h[i] = vw.data[i]
    with ow_buf.map_to_host() as h:
        for i in range(HIDDEN * HIDDEN): h[i] = ow.data[i]
    with gw_buf.map_to_host() as h:
        for i in range(HIDDEN * INTERMEDIATE): h[i] = gate_w.data[i]
    with uw_buf.map_to_host() as h:
        for i in range(HIDDEN * INTERMEDIATE): h[i] = up_w.data[i]
    with dw_buf.map_to_host() as h:
        for i in range(INTERMEDIATE * HIDDEN): h[i] = down_w.data[i]

    var x_2d = TileTensor(x_buf, x_2d_layout)
    var norm_2d = TileTensor(norm_buf, x_2d_layout)
    var in_norm_t = TileTensor(in_norm_buf, norm_w_layout)
    var post_norm_t = TileTensor(post_norm_buf, norm_w_layout)
    var qw_t = TileTensor(qw_buf, w_hh_layout)
    var kw_t = TileTensor(kw_buf, w_hh_layout)
    var vw_t = TileTensor(vw_buf, w_hh_layout)
    var ow_t = TileTensor(ow_buf, w_hh_layout)
    var gw_t = TileTensor(gw_buf, w_in_inter_layout)
    var uw_t = TileTensor(uw_buf, w_in_inter_layout)
    var dw_t = TileTensor(dw_buf, w_inter_in_layout)
    var q_flat_2d = TileTensor(q_flat_buf, x_2d_layout)
    var k_flat_2d = TileTensor(k_flat_buf, x_2d_layout)
    var v_flat_2d = TileTensor(v_flat_buf, x_2d_layout)
    var q_flat_4d = TileTensor(q_flat_buf, bshd_layout)
    var k_flat_4d = TileTensor(k_flat_buf, bshd_layout)
    var v_flat_4d = TileTensor(v_flat_buf, bshd_layout)
    var q_perm = TileTensor(q_perm_buf, bhsd_layout)
    var k_perm = TileTensor(k_perm_buf, bhsd_layout)
    var v_perm = TileTensor(v_perm_buf, bhsd_layout)
    var q_rot = TileTensor(q_rot_buf, bhsd_layout)
    var k_rot = TileTensor(k_rot_buf, bhsd_layout)
    var cos_t = TileTensor(cos_buf, cs_layout)
    var sin_t = TileTensor(sin_buf, cs_layout)
    var mask_t = TileTensor(mask_buf, mask_layout)
    var logits_t = TileTensor(logits_buf, ss_layout)
    var probs_t = TileTensor(probs_buf, ss_layout)
    var attn_perm = TileTensor(attn_perm_buf, bhsd_layout)
    var attn_flat_4d = TileTensor(attn_flat_buf, bshd_layout)
    var attn_flat_2d = TileTensor(attn_flat_buf, x_2d_layout)
    var attn_proj_2d = TileTensor(attn_proj_buf, x_2d_layout)
    var post_attn_2d = TileTensor(post_attn_buf, x_2d_layout)
    var post_norm_out_2d = TileTensor(post_norm_out_buf, x_2d_layout)
    var gate_t = TileTensor(gate_buf, intermediate_layout)
    var up_t = TileTensor(up_buf, intermediate_layout)
    var hidden_act_t = TileTensor(hidden_act_buf, intermediate_layout)
    var mlp_out_t = TileTensor(mlp_out_buf, x_2d_layout)
    var out_t = TileTensor(out_buf, x_2d_layout)

    comptime rms_k = rmsnorm_kernel[
        DType.bfloat16, type_of(x_2d_layout), type_of(norm_w_layout),
        type_of(x_2d_layout), RMS_BLOCK,
    ]
    comptime bshd_to_bhsd = bshd_to_bhsd_kernel[
        DType.bfloat16, type_of(bshd_layout), type_of(bhsd_layout),
        BATCH, SEQ, N_HEADS, HEAD_DIM,
    ]
    comptime bhsd_to_bshd = bhsd_to_bshd_kernel[
        DType.bfloat16, type_of(bhsd_layout), type_of(bshd_layout),
        BATCH, SEQ, N_HEADS, HEAD_DIM,
    ]
    comptime rope_k = rope_kernel[
        DType.bfloat16, type_of(bhsd_layout), type_of(cs_layout),
        type_of(bhsd_layout), HEAD_DIM, ROPE_HALF,
    ]
    comptime qk_k = qk_scaled_kernel[
        DType.bfloat16, type_of(bhsd_layout), type_of(bhsd_layout),
        type_of(mask_layout), type_of(ss_layout), HEAD_DIM, SEQ,
    ]
    comptime sm_k = softmax_kernel[
        DType.bfloat16, type_of(ss_layout), type_of(ss_layout), SEQ, SOFTMAX_BLOCK,
    ]
    comptime av_k = av_kernel[
        DType.bfloat16, type_of(ss_layout), type_of(bhsd_layout),
        type_of(bhsd_layout), SEQ, HEAD_DIM,
    ]
    comptime add_k = add_kernel[
        DType.bfloat16, type_of(x_2d_layout), type_of(x_2d_layout),
        type_of(x_2d_layout), POINTWISE_BLOCK,
    ]
    comptime silu_k = silu_mul_kernel[
        DType.bfloat16, type_of(intermediate_layout), type_of(intermediate_layout),
        type_of(intermediate_layout), POINTWISE_BLOCK,
    ]

    ctx.enqueue_function[rms_k, rms_k](
        norm_2d, x_2d, in_norm_t, EPS,
        grid_dim=ROWS, block_dim=RMS_BLOCK,
    )
    matmul[target="gpu"](q_flat_2d, norm_2d, qw_t, dctx)
    matmul[target="gpu"](k_flat_2d, norm_2d, kw_t, dctx)
    matmul[target="gpu"](v_flat_2d, norm_2d, vw_t, dctx)

    var n_perm_threads = BATCH * N_HEADS * SEQ
    ctx.enqueue_function[bshd_to_bhsd, bshd_to_bhsd](
        q_perm, q_flat_4d, grid_dim=n_perm_threads, block_dim=HEAD_DIM,
    )
    ctx.enqueue_function[bshd_to_bhsd, bshd_to_bhsd](
        k_perm, k_flat_4d, grid_dim=n_perm_threads, block_dim=HEAD_DIM,
    )
    ctx.enqueue_function[bshd_to_bhsd, bshd_to_bhsd](
        v_perm, v_flat_4d, grid_dim=n_perm_threads, block_dim=HEAD_DIM,
    )

    ctx.enqueue_function[rope_k, rope_k](
        q_rot, q_perm, cos_t, sin_t, N_HEADS, SEQ,
        grid_dim=BATCH * N_HEADS * SEQ, block_dim=HEAD_DIM,
    )
    ctx.enqueue_function[rope_k, rope_k](
        k_rot, k_perm, cos_t, sin_t, N_HEADS, SEQ,
        grid_dim=BATCH * N_HEADS * SEQ, block_dim=HEAD_DIM,
    )

    ctx.enqueue_function[qk_k, qk_k](
        logits_t, q_rot, k_rot, mask_t, N_HEADS, SCALE,
        grid_dim=BATCH * N_HEADS * SEQ, block_dim=SEQ,
    )
    ctx.enqueue_function[sm_k, sm_k](
        probs_t, logits_t, N_HEADS,
        grid_dim=BATCH * N_HEADS * SEQ, block_dim=SOFTMAX_BLOCK,
    )
    ctx.enqueue_function[av_k, av_k](
        attn_perm, probs_t, v_perm, N_HEADS,
        grid_dim=BATCH * N_HEADS * SEQ, block_dim=HEAD_DIM,
    )
    ctx.enqueue_function[bhsd_to_bshd, bhsd_to_bshd](
        attn_flat_4d, attn_perm, grid_dim=n_perm_threads, block_dim=HEAD_DIM,
    )

    matmul[target="gpu"](attn_proj_2d, attn_flat_2d, ow_t, dctx)

    var n_x_int = n_x
    ctx.enqueue_function[add_k, add_k](
        post_attn_2d, x_2d, attn_proj_2d, n_x_int,
        grid_dim=ceildiv(n_x_int, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )

    ctx.enqueue_function[rms_k, rms_k](
        post_norm_out_2d, post_attn_2d, post_norm_t, EPS,
        grid_dim=ROWS, block_dim=RMS_BLOCK,
    )

    matmul[target="gpu"](gate_t, post_norm_out_2d, gw_t, dctx)
    matmul[target="gpu"](up_t, post_norm_out_2d, uw_t, dctx)
    var n_silu = ROWS * INTERMEDIATE
    ctx.enqueue_function[silu_k, silu_k](
        hidden_act_t, gate_t, up_t, n_silu,
        grid_dim=ceildiv(n_silu, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )
    matmul[target="gpu"](mlp_out_t, hidden_act_t, dw_t, dctx)

    ctx.enqueue_function[add_k, add_k](
        out_t, post_attn_2d, mlp_out_t, n_x_int,
        grid_dim=ceildiv(n_x_int, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )

    ctx.synchronize()

    with out_buf.map_to_host() as h:
        for i in range(n_x):
            var got = Float32(h[i])
            var want = Float32(exp.data[i])
            # bf16 block — errors compound through ~10 ops + softmax; budget loose.
            assert_almost_equal(got, want, atol=2.0e-1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
