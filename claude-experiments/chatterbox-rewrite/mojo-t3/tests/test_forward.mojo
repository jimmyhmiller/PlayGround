"""
Full Llama-backbone forward parity test (30 layers) vs HF on real T3 weights.

Compares our Mojo output against acts_fp32["last_hidden_state"] from
oracle/extract.py (post-final-norm hidden states).

We don't load all weights for all layers into separate buffers up front
(2.9 GB on host); instead we allocate device buffers per-layer once at the
start, then upload weights lazily. After each layer we re-use the same input
buffer for the next layer's input by ping-ponging x_a / x_b.
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from layout import TileTensor, row_major
from linalg.matmul import matmul

from fixture import Tensor, TensorBF16, load_fp32, load_bf16
from rmsnorm import rmsnorm_kernel
from rope import rope_kernel
from sdpa import qk_scaled_kernel, softmax_kernel, av_kernel
from mlp import silu_mul_kernel
from util_kernels import add_kernel, bshd_to_bhsd_kernel, bhsd_to_bshd_kernel, print_rss


# T3 Llama config.
comptime BATCH = 1
comptime SEQ = 16
comptime N_HEADS = 16
comptime HEAD_DIM = 64
comptime HIDDEN = N_HEADS * HEAD_DIM        # 1024
comptime INTERMEDIATE = 4096
comptime N_LAYERS = 30
comptime SCALE: Float32 = 0.125
comptime EPS: Float32 = 1.0e-5
comptime ROPE_HALF = HEAD_DIM // 2

comptime RMS_BLOCK = 256
comptime ROWS = BATCH * SEQ
comptime POINTWISE_BLOCK = 256
comptime SOFTMAX_BLOCK = 32

comptime x_2d_layout = row_major[ROWS, HIDDEN]()
comptime w_hh_layout = row_major[HIDDEN, HIDDEN]()
comptime w_in_inter_layout = row_major[HIDDEN, INTERMEDIATE]()
comptime w_inter_in_layout = row_major[INTERMEDIATE, HIDDEN]()
comptime intermediate_layout = row_major[ROWS, INTERMEDIATE]()
comptime bshd_layout = row_major[BATCH, SEQ, N_HEADS, HEAD_DIM]()
comptime bhsd_layout = row_major[BATCH, N_HEADS, SEQ, HEAD_DIM]()
comptime cs_layout = row_major[BATCH, SEQ, HEAD_DIM]()
comptime mask_layout = row_major[SEQ, SEQ]()
comptime ss_layout = row_major[BATCH, N_HEADS, SEQ, SEQ]()
comptime norm_w_layout = row_major[HIDDEN]()


# ---- Kernel bindings (fp32) ----
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


# ---- Kernel bindings (bf16) ----
comptime rms_k_bf = rmsnorm_kernel[
    DType.bfloat16, type_of(x_2d_layout), type_of(norm_w_layout),
    type_of(x_2d_layout), RMS_BLOCK,
]
comptime bshd_to_bhsd_bf = bshd_to_bhsd_kernel[
    DType.bfloat16, type_of(bshd_layout), type_of(bhsd_layout),
    BATCH, SEQ, N_HEADS, HEAD_DIM,
]
comptime bhsd_to_bshd_bf = bhsd_to_bshd_kernel[
    DType.bfloat16, type_of(bhsd_layout), type_of(bshd_layout),
    BATCH, SEQ, N_HEADS, HEAD_DIM,
]
comptime rope_k_bf = rope_kernel[
    DType.bfloat16, type_of(bhsd_layout), type_of(cs_layout),
    type_of(bhsd_layout), HEAD_DIM, ROPE_HALF,
]
comptime qk_k_bf = qk_scaled_kernel[
    DType.bfloat16, type_of(bhsd_layout), type_of(bhsd_layout),
    type_of(mask_layout), type_of(ss_layout), HEAD_DIM, SEQ,
]
comptime sm_k_bf = softmax_kernel[
    DType.bfloat16, type_of(ss_layout), type_of(ss_layout), SEQ, SOFTMAX_BLOCK,
]
comptime av_k_bf = av_kernel[
    DType.bfloat16, type_of(ss_layout), type_of(bhsd_layout),
    type_of(bhsd_layout), SEQ, HEAD_DIM,
]
comptime add_k_bf = add_kernel[
    DType.bfloat16, type_of(x_2d_layout), type_of(x_2d_layout),
    type_of(x_2d_layout), POINTWISE_BLOCK,
]
comptime silu_k_bf = silu_mul_kernel[
    DType.bfloat16, type_of(intermediate_layout), type_of(intermediate_layout),
    type_of(intermediate_layout), POINTWISE_BLOCK,
]


def upload_fp32(ctx: DeviceContext, buf: DeviceBuffer[DType.float32], data: List[Float32]) raises:
    var n = len(data)
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def upload_bf16(ctx: DeviceContext, buf: DeviceBuffer[DType.bfloat16], data: List[BFloat16]) raises:
    var n = len(data)
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def test_forward_fp32() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/forward/"
    var x = load_fp32(fix + "input_embeds_fp32.bin")
    var cos = load_fp32(fix + "cos_fp32.bin")
    var sin = load_fp32(fix + "sin_fp32.bin")
    var mask = load_fp32(fix + "mask_fp32.bin")
    var final_norm = load_fp32(fix + "final_norm_fp32.bin")
    var exp = load_fp32(fix + "expected_fp32.bin")

    assert_equal(x.shape[0] * x.shape[1], ROWS)
    assert_equal(x.shape[2], HIDDEN)

    var n_x = ROWS * HIDDEN
    var n_cs = BATCH * SEQ * HEAD_DIM
    var n_mask = SEQ * SEQ
    var n_qkv_perm = BATCH * N_HEADS * SEQ * HEAD_DIM
    var n_ss = BATCH * N_HEADS * SEQ * SEQ
    var n_inter = ROWS * INTERMEDIATE
    var n_w_hh = HIDDEN * HIDDEN
    var n_w_in_inter = HIDDEN * INTERMEDIATE
    var n_w_inter_in = INTERMEDIATE * HIDDEN

    var ctx = DeviceContext()
    var dctx = DeviceContextPtr(ctx)

    # --- Globals ---
    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var post_attn_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var cos_buf = ctx.enqueue_create_buffer[DType.float32](n_cs)
    var sin_buf = ctx.enqueue_create_buffer[DType.float32](n_cs)
    var mask_buf = ctx.enqueue_create_buffer[DType.float32](n_mask)
    var final_norm_buf = ctx.enqueue_create_buffer[DType.float32](HIDDEN)

    # --- Per-layer scratch buffers (reused across all 30 layers) ---
    var in_norm_buf = ctx.enqueue_create_buffer[DType.float32](HIDDEN)
    var post_norm_buf = ctx.enqueue_create_buffer[DType.float32](HIDDEN)
    var qw_buf = ctx.enqueue_create_buffer[DType.float32](n_w_hh)
    var kw_buf = ctx.enqueue_create_buffer[DType.float32](n_w_hh)
    var vw_buf = ctx.enqueue_create_buffer[DType.float32](n_w_hh)
    var ow_buf = ctx.enqueue_create_buffer[DType.float32](n_w_hh)
    var gw_buf = ctx.enqueue_create_buffer[DType.float32](n_w_in_inter)
    var uw_buf = ctx.enqueue_create_buffer[DType.float32](n_w_in_inter)
    var dw_buf = ctx.enqueue_create_buffer[DType.float32](n_w_inter_in)

    # --- Block-internal scratch ---
    var norm_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var q_flat_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var k_flat_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var v_flat_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var q_perm_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv_perm)
    var k_perm_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv_perm)
    var v_perm_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv_perm)
    var q_rot_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv_perm)
    var k_rot_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv_perm)
    var logits_buf = ctx.enqueue_create_buffer[DType.float32](n_ss)
    var probs_buf = ctx.enqueue_create_buffer[DType.float32](n_ss)
    var attn_perm_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv_perm)
    var attn_flat_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var attn_proj_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var post_norm_out_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var gate_buf = ctx.enqueue_create_buffer[DType.float32](n_inter)
    var up_buf = ctx.enqueue_create_buffer[DType.float32](n_inter)
    var hidden_act_buf = ctx.enqueue_create_buffer[DType.float32](n_inter)
    var mlp_out_buf = ctx.enqueue_create_buffer[DType.float32](n_x)

    # --- TileTensor views (constant across layers; addresses stay valid) ---
    var cos_t = TileTensor(cos_buf, cs_layout)
    var sin_t = TileTensor(sin_buf, cs_layout)
    var mask_t = TileTensor(mask_buf, mask_layout)
    var final_norm_t = TileTensor(final_norm_buf, norm_w_layout)
    var in_norm_t = TileTensor(in_norm_buf, norm_w_layout)
    var post_norm_t = TileTensor(post_norm_buf, norm_w_layout)
    var qw_t = TileTensor(qw_buf, w_hh_layout)
    var kw_t = TileTensor(kw_buf, w_hh_layout)
    var vw_t = TileTensor(vw_buf, w_hh_layout)
    var ow_t = TileTensor(ow_buf, w_hh_layout)
    var gw_t = TileTensor(gw_buf, w_in_inter_layout)
    var uw_t = TileTensor(uw_buf, w_in_inter_layout)
    var dw_t = TileTensor(dw_buf, w_inter_in_layout)
    var norm_2d = TileTensor(norm_buf, x_2d_layout)
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
    var logits_t = TileTensor(logits_buf, ss_layout)
    var probs_t = TileTensor(probs_buf, ss_layout)
    var attn_perm = TileTensor(attn_perm_buf, bhsd_layout)
    var attn_flat_4d = TileTensor(attn_flat_buf, bshd_layout)
    var attn_flat_2d = TileTensor(attn_flat_buf, x_2d_layout)
    var attn_proj_2d = TileTensor(attn_proj_buf, x_2d_layout)
    var post_norm_out_2d = TileTensor(post_norm_out_buf, x_2d_layout)
    var gate_t = TileTensor(gate_buf, intermediate_layout)
    var up_t = TileTensor(up_buf, intermediate_layout)
    var hidden_act_t = TileTensor(hidden_act_buf, intermediate_layout)
    var mlp_out_t = TileTensor(mlp_out_buf, x_2d_layout)

    # --- Stage globals to device ---
    upload_fp32(ctx, x_buf, x.data)
    upload_fp32(ctx, cos_buf, cos.data)
    upload_fp32(ctx, sin_buf, sin.data)
    upload_fp32(ctx, mask_buf, mask.data)
    upload_fp32(ctx, final_norm_buf, final_norm.data)

    print_rss("before layer loop (fp32)")

    var n_perm_threads = BATCH * N_HEADS * SEQ
    var n_silu = ROWS * INTERMEDIATE

    var x_2d = TileTensor(x_buf, x_2d_layout)
    var post_attn_2d = TileTensor(post_attn_buf, x_2d_layout)

    # --- Loop over layers ---
    for L in range(N_LAYERS):
        # Load this layer's weights from disk and stage to device.
        var layer_dir = "tests/fixtures/forward/layer" + String(L) + "/"
        var in_norm = load_fp32(layer_dir + "in_norm_fp32.bin")
        var post_norm = load_fp32(layer_dir + "post_norm_fp32.bin")
        var qw = load_fp32(layer_dir + "qw_fp32.bin")
        var kw = load_fp32(layer_dir + "kw_fp32.bin")
        var vw = load_fp32(layer_dir + "vw_fp32.bin")
        var ow = load_fp32(layer_dir + "ow_fp32.bin")
        var gate_w = load_fp32(layer_dir + "gate_w_fp32.bin")
        var up_w = load_fp32(layer_dir + "up_w_fp32.bin")
        var down_w = load_fp32(layer_dir + "down_w_fp32.bin")

        upload_fp32(ctx, in_norm_buf, in_norm.data)
        upload_fp32(ctx, post_norm_buf, post_norm.data)
        upload_fp32(ctx, qw_buf, qw.data)
        upload_fp32(ctx, kw_buf, kw.data)
        upload_fp32(ctx, vw_buf, vw.data)
        upload_fp32(ctx, ow_buf, ow.data)
        upload_fp32(ctx, gw_buf, gate_w.data)
        upload_fp32(ctx, uw_buf, up_w.data)
        upload_fp32(ctx, dw_buf, down_w.data)

        # 1. norm = rmsnorm(x)
        ctx.enqueue_function[rms_k, rms_k](
            norm_2d, x_2d, in_norm_t, EPS,
            grid_dim=ROWS, block_dim=RMS_BLOCK,
        )
        # 2. q/k/v projections
        matmul[target="gpu"](q_flat_2d, norm_2d, qw_t, dctx)
        matmul[target="gpu"](k_flat_2d, norm_2d, kw_t, dctx)
        matmul[target="gpu"](v_flat_2d, norm_2d, vw_t, dctx)
        # 3. permute (B,S,H,D) → (B,H,S,D)
        ctx.enqueue_function[bshd_to_bhsd, bshd_to_bhsd](
            q_perm, q_flat_4d, grid_dim=n_perm_threads, block_dim=HEAD_DIM,
        )
        ctx.enqueue_function[bshd_to_bhsd, bshd_to_bhsd](
            k_perm, k_flat_4d, grid_dim=n_perm_threads, block_dim=HEAD_DIM,
        )
        ctx.enqueue_function[bshd_to_bhsd, bshd_to_bhsd](
            v_perm, v_flat_4d, grid_dim=n_perm_threads, block_dim=HEAD_DIM,
        )
        # 4. RoPE on q, k
        ctx.enqueue_function[rope_k, rope_k](
            q_rot, q_perm, cos_t, sin_t, N_HEADS, SEQ,
            grid_dim=BATCH * N_HEADS * SEQ, block_dim=HEAD_DIM,
        )
        ctx.enqueue_function[rope_k, rope_k](
            k_rot, k_perm, cos_t, sin_t, N_HEADS, SEQ,
            grid_dim=BATCH * N_HEADS * SEQ, block_dim=HEAD_DIM,
        )
        # 5. SDPA
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
        # 6. permute back
        ctx.enqueue_function[bhsd_to_bshd, bhsd_to_bshd](
            attn_flat_4d, attn_perm, grid_dim=n_perm_threads, block_dim=HEAD_DIM,
        )
        # 7. attn_proj = attn_flat @ ow
        matmul[target="gpu"](attn_proj_2d, attn_flat_2d, ow_t, dctx)
        # 8. post_attn = x + attn_proj (residual 1)
        ctx.enqueue_function[add_k, add_k](
            post_attn_2d, x_2d, attn_proj_2d, n_x,
            grid_dim=ceildiv(n_x, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
        )
        # 9. post_norm = rmsnorm(post_attn)
        ctx.enqueue_function[rms_k, rms_k](
            post_norm_out_2d, post_attn_2d, post_norm_t, EPS,
            grid_dim=ROWS, block_dim=RMS_BLOCK,
        )
        # 10. MLP
        matmul[target="gpu"](gate_t, post_norm_out_2d, gw_t, dctx)
        matmul[target="gpu"](up_t, post_norm_out_2d, uw_t, dctx)
        ctx.enqueue_function[silu_k, silu_k](
            hidden_act_t, gate_t, up_t, n_silu,
            grid_dim=ceildiv(n_silu, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
        )
        matmul[target="gpu"](mlp_out_t, hidden_act_t, dw_t, dctx)
        # 11. x = post_attn + mlp_out (residual 2 → back into x_buf for next layer)
        ctx.enqueue_function[add_k, add_k](
            x_2d, post_attn_2d, mlp_out_t, n_x,
            grid_dim=ceildiv(n_x, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
        )
        # Drain the queue at end of each layer so the runtime can reclaim
        # transient state (DeviceContext was holding ~all 480 kernels' worth
        # of metadata otherwise).
        ctx.synchronize()
        if L % 5 == 0:
            print_rss("after layer " + String(L))

    # After 30 layers, x_buf holds the residual stream output. Apply final RMSNorm.
    var final_out_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var final_out_2d = TileTensor(final_out_buf, x_2d_layout)
    ctx.enqueue_function[rms_k, rms_k](
        final_out_2d, x_2d, final_norm_t, EPS,
        grid_dim=ROWS, block_dim=RMS_BLOCK,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    with final_out_buf.map_to_host() as h:
        for i in range(n_x):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-2)
    print("max abs diff after 30 layers + final norm:", max_abs)


def test_forward_bf16() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/forward/"
    var x = load_bf16(fix + "input_embeds_bf16.bin")
    var cos = load_bf16(fix + "cos_bf16.bin")
    var sin = load_bf16(fix + "sin_bf16.bin")
    var mask = load_bf16(fix + "mask_bf16.bin")
    var final_norm = load_bf16(fix + "final_norm_bf16.bin")
    var exp = load_bf16(fix + "expected_bf16.bin")

    assert_equal(x.shape[0] * x.shape[1], ROWS)
    assert_equal(x.shape[2], HIDDEN)

    var n_x = ROWS * HIDDEN
    var n_cs = BATCH * SEQ * HEAD_DIM
    var n_mask = SEQ * SEQ
    var n_qkv_perm = BATCH * N_HEADS * SEQ * HEAD_DIM
    var n_ss = BATCH * N_HEADS * SEQ * SEQ
    var n_inter = ROWS * INTERMEDIATE
    var n_w_hh = HIDDEN * HIDDEN
    var n_w_in_inter = HIDDEN * INTERMEDIATE
    var n_w_inter_in = INTERMEDIATE * HIDDEN

    var ctx = DeviceContext()
    var dctx = DeviceContextPtr(ctx)

    # --- Globals ---
    var x_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_x)
    var post_attn_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_x)
    var cos_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_cs)
    var sin_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_cs)
    var mask_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_mask)
    var final_norm_buf = ctx.enqueue_create_buffer[DType.bfloat16](HIDDEN)

    # --- Per-layer reusable buffers ---
    var in_norm_buf = ctx.enqueue_create_buffer[DType.bfloat16](HIDDEN)
    var post_norm_buf = ctx.enqueue_create_buffer[DType.bfloat16](HIDDEN)
    var qw_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_w_hh)
    var kw_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_w_hh)
    var vw_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_w_hh)
    var ow_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_w_hh)
    var gw_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_w_in_inter)
    var uw_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_w_in_inter)
    var dw_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_w_inter_in)

    # --- Block-internal scratch ---
    var norm_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_x)
    var q_flat_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_x)
    var k_flat_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_x)
    var v_flat_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_x)
    var q_perm_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv_perm)
    var k_perm_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv_perm)
    var v_perm_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv_perm)
    var q_rot_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv_perm)
    var k_rot_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv_perm)
    var logits_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_ss)
    var probs_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_ss)
    var attn_perm_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv_perm)
    var attn_flat_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_x)
    var attn_proj_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_x)
    var post_norm_out_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_x)
    var gate_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_inter)
    var up_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_inter)
    var hidden_act_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_inter)
    var mlp_out_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_x)

    # --- TileTensor views ---
    var cos_t = TileTensor(cos_buf, cs_layout)
    var sin_t = TileTensor(sin_buf, cs_layout)
    var mask_t = TileTensor(mask_buf, mask_layout)
    var final_norm_t = TileTensor(final_norm_buf, norm_w_layout)
    var in_norm_t = TileTensor(in_norm_buf, norm_w_layout)
    var post_norm_t = TileTensor(post_norm_buf, norm_w_layout)
    var qw_t = TileTensor(qw_buf, w_hh_layout)
    var kw_t = TileTensor(kw_buf, w_hh_layout)
    var vw_t = TileTensor(vw_buf, w_hh_layout)
    var ow_t = TileTensor(ow_buf, w_hh_layout)
    var gw_t = TileTensor(gw_buf, w_in_inter_layout)
    var uw_t = TileTensor(uw_buf, w_in_inter_layout)
    var dw_t = TileTensor(dw_buf, w_inter_in_layout)
    var norm_2d = TileTensor(norm_buf, x_2d_layout)
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
    var logits_t = TileTensor(logits_buf, ss_layout)
    var probs_t = TileTensor(probs_buf, ss_layout)
    var attn_perm = TileTensor(attn_perm_buf, bhsd_layout)
    var attn_flat_4d = TileTensor(attn_flat_buf, bshd_layout)
    var attn_flat_2d = TileTensor(attn_flat_buf, x_2d_layout)
    var attn_proj_2d = TileTensor(attn_proj_buf, x_2d_layout)
    var post_norm_out_2d = TileTensor(post_norm_out_buf, x_2d_layout)
    var gate_t = TileTensor(gate_buf, intermediate_layout)
    var up_t = TileTensor(up_buf, intermediate_layout)
    var hidden_act_t = TileTensor(hidden_act_buf, intermediate_layout)
    var mlp_out_t = TileTensor(mlp_out_buf, x_2d_layout)

    upload_bf16(ctx, x_buf, x.data)
    upload_bf16(ctx, cos_buf, cos.data)
    upload_bf16(ctx, sin_buf, sin.data)
    upload_bf16(ctx, mask_buf, mask.data)
    upload_bf16(ctx, final_norm_buf, final_norm.data)

    var n_perm_threads = BATCH * N_HEADS * SEQ
    var n_silu = ROWS * INTERMEDIATE

    var x_2d = TileTensor(x_buf, x_2d_layout)
    var post_attn_2d = TileTensor(post_attn_buf, x_2d_layout)

    for L in range(N_LAYERS):
        var layer_dir = "tests/fixtures/forward/layer" + String(L) + "/"
        var in_norm = load_bf16(layer_dir + "in_norm_bf16.bin")
        var post_norm = load_bf16(layer_dir + "post_norm_bf16.bin")
        var qw = load_bf16(layer_dir + "qw_bf16.bin")
        var kw = load_bf16(layer_dir + "kw_bf16.bin")
        var vw = load_bf16(layer_dir + "vw_bf16.bin")
        var ow = load_bf16(layer_dir + "ow_bf16.bin")
        var gate_w = load_bf16(layer_dir + "gate_w_bf16.bin")
        var up_w = load_bf16(layer_dir + "up_w_bf16.bin")
        var down_w = load_bf16(layer_dir + "down_w_bf16.bin")

        upload_bf16(ctx, in_norm_buf, in_norm.data)
        upload_bf16(ctx, post_norm_buf, post_norm.data)
        upload_bf16(ctx, qw_buf, qw.data)
        upload_bf16(ctx, kw_buf, kw.data)
        upload_bf16(ctx, vw_buf, vw.data)
        upload_bf16(ctx, ow_buf, ow.data)
        upload_bf16(ctx, gw_buf, gate_w.data)
        upload_bf16(ctx, uw_buf, up_w.data)
        upload_bf16(ctx, dw_buf, down_w.data)

        ctx.enqueue_function[rms_k_bf, rms_k_bf](
            norm_2d, x_2d, in_norm_t, EPS,
            grid_dim=ROWS, block_dim=RMS_BLOCK,
        )
        matmul[target="gpu"](q_flat_2d, norm_2d, qw_t, dctx)
        matmul[target="gpu"](k_flat_2d, norm_2d, kw_t, dctx)
        matmul[target="gpu"](v_flat_2d, norm_2d, vw_t, dctx)
        ctx.enqueue_function[bshd_to_bhsd_bf, bshd_to_bhsd_bf](
            q_perm, q_flat_4d, grid_dim=n_perm_threads, block_dim=HEAD_DIM,
        )
        ctx.enqueue_function[bshd_to_bhsd_bf, bshd_to_bhsd_bf](
            k_perm, k_flat_4d, grid_dim=n_perm_threads, block_dim=HEAD_DIM,
        )
        ctx.enqueue_function[bshd_to_bhsd_bf, bshd_to_bhsd_bf](
            v_perm, v_flat_4d, grid_dim=n_perm_threads, block_dim=HEAD_DIM,
        )
        ctx.enqueue_function[rope_k_bf, rope_k_bf](
            q_rot, q_perm, cos_t, sin_t, N_HEADS, SEQ,
            grid_dim=BATCH * N_HEADS * SEQ, block_dim=HEAD_DIM,
        )
        ctx.enqueue_function[rope_k_bf, rope_k_bf](
            k_rot, k_perm, cos_t, sin_t, N_HEADS, SEQ,
            grid_dim=BATCH * N_HEADS * SEQ, block_dim=HEAD_DIM,
        )
        ctx.enqueue_function[qk_k_bf, qk_k_bf](
            logits_t, q_rot, k_rot, mask_t, N_HEADS, SCALE,
            grid_dim=BATCH * N_HEADS * SEQ, block_dim=SEQ,
        )
        ctx.enqueue_function[sm_k_bf, sm_k_bf](
            probs_t, logits_t, N_HEADS,
            grid_dim=BATCH * N_HEADS * SEQ, block_dim=SOFTMAX_BLOCK,
        )
        ctx.enqueue_function[av_k_bf, av_k_bf](
            attn_perm, probs_t, v_perm, N_HEADS,
            grid_dim=BATCH * N_HEADS * SEQ, block_dim=HEAD_DIM,
        )
        ctx.enqueue_function[bhsd_to_bshd_bf, bhsd_to_bshd_bf](
            attn_flat_4d, attn_perm, grid_dim=n_perm_threads, block_dim=HEAD_DIM,
        )
        matmul[target="gpu"](attn_proj_2d, attn_flat_2d, ow_t, dctx)
        ctx.enqueue_function[add_k_bf, add_k_bf](
            post_attn_2d, x_2d, attn_proj_2d, n_x,
            grid_dim=ceildiv(n_x, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
        )
        ctx.enqueue_function[rms_k_bf, rms_k_bf](
            post_norm_out_2d, post_attn_2d, post_norm_t, EPS,
            grid_dim=ROWS, block_dim=RMS_BLOCK,
        )
        matmul[target="gpu"](gate_t, post_norm_out_2d, gw_t, dctx)
        matmul[target="gpu"](up_t, post_norm_out_2d, uw_t, dctx)
        ctx.enqueue_function[silu_k_bf, silu_k_bf](
            hidden_act_t, gate_t, up_t, n_silu,
            grid_dim=ceildiv(n_silu, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
        )
        matmul[target="gpu"](mlp_out_t, hidden_act_t, dw_t, dctx)
        ctx.enqueue_function[add_k_bf, add_k_bf](
            x_2d, post_attn_2d, mlp_out_t, n_x,
            grid_dim=ceildiv(n_x, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
        )

    var final_out_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_x)
    var final_out_2d = TileTensor(final_out_buf, x_2d_layout)
    ctx.enqueue_function[rms_k_bf, rms_k_bf](
        final_out_2d, x_2d, final_norm_t, EPS,
        grid_dim=ROWS, block_dim=RMS_BLOCK,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    var sum_abs: Float64 = 0.0
    with final_out_buf.map_to_host() as h:
        for i in range(n_x):
            var got = Float32(h[i])
            var want = Float32(exp.data[i])
            var d = got - want
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_abs += Float64(d)
            assert_almost_equal(got, want, atol=2.0e-1)
    print("bf16 forward — max abs:", max_abs, "mean abs:", sum_abs / Float64(n_x))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
