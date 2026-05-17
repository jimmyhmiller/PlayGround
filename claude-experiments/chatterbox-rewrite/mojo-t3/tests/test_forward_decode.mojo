"""
30-layer decode-step parity test vs HF on real T3 weights.

Reuses the existing per-layer prefill weights from forward/layer{L}/*.bin and
the per-layer prefill K/V caches from forward_decode/layer{L}/k_hist,v_hist.

Per layer L:
  1. norm = rmsnorm(x, in_norm[L])
  2. q,k,v = norm @ {qw,kw,vw}[L]
  3. permute (B,1,H,D) → (B,H,1,D)
  4. RoPE on (q, k_new) at position T_HIST
  5. cache_append k_rot → k_cache[L,..,T_HIST,..]; same for v
  6. attn = qk_decode → softmax_decode → av_decode (cur_len = MAX_CTX)
  7. permute back → o_proj → add residual
  8. post_norm → MLP → add residual → next layer
After all 30 layers: rmsnorm(x, final_norm), compare to expected (B,1,H).
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
from sdpa import qk_decode_kernel, softmax_decode_kernel, av_decode_kernel
from mlp import silu_mul_kernel
from util_kernels import (
    add_kernel,
    bshd_to_bhsd_kernel,
    bhsd_to_bshd_kernel,
    cache_append_kernel,
    print_rss,
)


# T3 Llama config.
comptime BATCH = 1
comptime T_HIST = 15
comptime SEQ = 1
comptime MAX_CTX = T_HIST + SEQ                # 16
comptime N_HEADS = 16
comptime HEAD_DIM = 64
comptime HIDDEN = N_HEADS * HEAD_DIM           # 1024
comptime INTERMEDIATE = 4096
comptime N_LAYERS = 30
comptime SCALE: Float32 = 0.125
comptime EPS: Float32 = 1.0e-5
comptime ROPE_HALF = HEAD_DIM // 2

comptime RMS_BLOCK = 256
comptime ROWS = BATCH * SEQ                    # 1
comptime POINTWISE_BLOCK = 256
comptime SOFTMAX_BLOCK = 32

comptime x_2d_layout = row_major[ROWS, HIDDEN]()
comptime w_hh_layout = row_major[HIDDEN, HIDDEN]()
comptime w_in_inter_layout = row_major[HIDDEN, INTERMEDIATE]()
comptime w_inter_in_layout = row_major[INTERMEDIATE, HIDDEN]()
comptime intermediate_layout = row_major[ROWS, INTERMEDIATE]()
comptime bshd_layout = row_major[BATCH, SEQ, N_HEADS, HEAD_DIM]()
comptime bhsd_layout = row_major[BATCH, N_HEADS, SEQ, HEAD_DIM]()
comptime kv_cache_layout = row_major[BATCH, N_HEADS, MAX_CTX, HEAD_DIM]()
comptime cs_layout = row_major[BATCH, SEQ, HEAD_DIM]()
comptime probs_layout = row_major[BATCH, N_HEADS, 1, MAX_CTX]()
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
comptime qk_dec = qk_decode_kernel[
    DType.float32, type_of(bhsd_layout), type_of(kv_cache_layout),
    type_of(probs_layout), HEAD_DIM, MAX_CTX,
]
comptime sm_dec = softmax_decode_kernel[
    DType.float32, type_of(probs_layout), type_of(probs_layout),
    MAX_CTX, SOFTMAX_BLOCK,
]
comptime av_dec = av_decode_kernel[
    DType.float32, type_of(probs_layout), type_of(kv_cache_layout),
    type_of(bhsd_layout), MAX_CTX, HEAD_DIM,
]
comptime add_k = add_kernel[
    DType.float32, type_of(x_2d_layout), type_of(x_2d_layout),
    type_of(x_2d_layout), POINTWISE_BLOCK,
]
comptime silu_k = silu_mul_kernel[
    DType.float32, type_of(intermediate_layout), type_of(intermediate_layout),
    type_of(intermediate_layout), POINTWISE_BLOCK,
]
comptime cache_k = cache_append_kernel[
    DType.float32, type_of(bhsd_layout), type_of(kv_cache_layout),
    BATCH, N_HEADS, HEAD_DIM,
]


def upload_fp32(ctx: DeviceContext, buf: DeviceBuffer[DType.float32], data: List[Float32]) raises:
    var n = len(data)
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def test_forward_decode_fp32() raises:
    comptime assert has_accelerator(), "Requires GPU"

    # ---- Load globals ----
    var fix = "tests/fixtures/forward_decode/"
    var x = load_fp32(fix + "x_decode_fp32.bin")
    var cos = load_fp32(fix + "cos_decode_fp32.bin")
    var sin = load_fp32(fix + "sin_decode_fp32.bin")
    var final_norm = load_fp32(fix + "final_norm_fp32.bin")
    var exp = load_fp32(fix + "expected_fp32.bin")

    var n_x = ROWS * HIDDEN                           # 1024
    var n_cs = BATCH * SEQ * HEAD_DIM                 # 64
    var n_qkv_perm = BATCH * N_HEADS * SEQ * HEAD_DIM # 1024
    var n_kv_cache = BATCH * N_HEADS * MAX_CTX * HEAD_DIM
    var n_probs = BATCH * N_HEADS * 1 * MAX_CTX
    var n_inter = ROWS * INTERMEDIATE
    var n_w_hh = HIDDEN * HIDDEN
    var n_w_in_inter = HIDDEN * INTERMEDIATE
    var n_w_inter_in = INTERMEDIATE * HIDDEN

    var ctx = DeviceContext()
    var dctx = DeviceContextPtr(ctx)

    # ---- Persistent buffers ----
    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var post_attn_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var cos_buf = ctx.enqueue_create_buffer[DType.float32](n_cs)
    var sin_buf = ctx.enqueue_create_buffer[DType.float32](n_cs)
    var final_norm_buf = ctx.enqueue_create_buffer[DType.float32](HIDDEN)

    # ---- Per-layer scratch (reused) ----
    var in_norm_buf = ctx.enqueue_create_buffer[DType.float32](HIDDEN)
    var post_norm_buf = ctx.enqueue_create_buffer[DType.float32](HIDDEN)
    var qw_buf = ctx.enqueue_create_buffer[DType.float32](n_w_hh)
    var kw_buf = ctx.enqueue_create_buffer[DType.float32](n_w_hh)
    var vw_buf = ctx.enqueue_create_buffer[DType.float32](n_w_hh)
    var ow_buf = ctx.enqueue_create_buffer[DType.float32](n_w_hh)
    var gw_buf = ctx.enqueue_create_buffer[DType.float32](n_w_in_inter)
    var uw_buf = ctx.enqueue_create_buffer[DType.float32](n_w_in_inter)
    var dw_buf = ctx.enqueue_create_buffer[DType.float32](n_w_inter_in)

    # ---- Block-internal scratch ----
    var norm_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var q_flat_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv_perm)
    var k_flat_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv_perm)
    var v_flat_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv_perm)
    var q_perm_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv_perm)
    var k_perm_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv_perm)
    var v_perm_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv_perm)
    var q_rot_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv_perm)
    var k_rot_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv_perm)
    var logits_buf = ctx.enqueue_create_buffer[DType.float32](n_probs)
    var probs_buf = ctx.enqueue_create_buffer[DType.float32](n_probs)
    var attn_perm_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv_perm)
    var attn_flat_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv_perm)
    var attn_proj_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var post_norm_out_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var gate_buf = ctx.enqueue_create_buffer[DType.float32](n_inter)
    var up_buf = ctx.enqueue_create_buffer[DType.float32](n_inter)
    var hidden_act_buf = ctx.enqueue_create_buffer[DType.float32](n_inter)
    var mlp_out_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var final_out_buf = ctx.enqueue_create_buffer[DType.float32](n_x)

    # ---- Per-layer K/V cache buffers (one set per layer) ----
    var k_cache_bufs = List[DeviceBuffer[DType.float32]]()
    var v_cache_bufs = List[DeviceBuffer[DType.float32]]()
    for _ in range(N_LAYERS):
        k_cache_bufs.append(ctx.enqueue_create_buffer[DType.float32](n_kv_cache))
        v_cache_bufs.append(ctx.enqueue_create_buffer[DType.float32](n_kv_cache))

    # ---- Stage globals ----
    upload_fp32(ctx, x_buf, x.data)
    upload_fp32(ctx, cos_buf, cos.data)
    upload_fp32(ctx, sin_buf, sin.data)
    upload_fp32(ctx, final_norm_buf, final_norm.data)

    # ---- Stage per-layer K/V history into the front of each layer's cache ----
    for L in range(N_LAYERS):
        var layer_dir = "tests/fixtures/forward_decode/layer" + String(L) + "/"
        var k_hist = load_fp32(layer_dir + "k_hist_fp32.bin")
        var v_hist = load_fp32(layer_dir + "v_hist_fp32.bin")
        with k_cache_bufs[L].map_to_host() as h:
            # Front-load slots [0..T_HIST); leave slot T_HIST for the new K.
            for b in range(BATCH):
                for hd in range(N_HEADS):
                    for s in range(T_HIST):
                        for d in range(HEAD_DIM):
                            var src = ((b * N_HEADS + hd) * T_HIST + s) * HEAD_DIM + d
                            var dst = ((b * N_HEADS + hd) * MAX_CTX + s) * HEAD_DIM + d
                            h[dst] = k_hist.data[src]
        with v_cache_bufs[L].map_to_host() as h:
            for b in range(BATCH):
                for hd in range(N_HEADS):
                    for s in range(T_HIST):
                        for d in range(HEAD_DIM):
                            var src = ((b * N_HEADS + hd) * T_HIST + s) * HEAD_DIM + d
                            var dst = ((b * N_HEADS + hd) * MAX_CTX + s) * HEAD_DIM + d
                            h[dst] = v_hist.data[src]

    print_rss("before decode layer loop (fp32)")

    # ---- Static TileTensor views ----
    var cos_t = TileTensor(cos_buf, cs_layout)
    var sin_t = TileTensor(sin_buf, cs_layout)
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
    var logits_t = TileTensor(logits_buf, probs_layout)
    var probs_t = TileTensor(probs_buf, probs_layout)
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
    var x_2d = TileTensor(x_buf, x_2d_layout)

    var n_perm_threads = BATCH * N_HEADS * SEQ
    var n_silu = ROWS * INTERMEDIATE

    # ---- Loop over layers ----
    for L in range(N_LAYERS):
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

        # Rebind cache views — buffer ptr changes per layer.
        var k_cache_t = TileTensor(k_cache_bufs[L], kv_cache_layout)
        var v_cache_t = TileTensor(v_cache_bufs[L], kv_cache_layout)

        # 1. norm
        ctx.enqueue_function[rms_k, rms_k](
            norm_2d, x_2d, in_norm_t, EPS,
            grid_dim=ROWS, block_dim=RMS_BLOCK,
        )
        # 2. qkv
        matmul[target="gpu"](q_flat_2d, norm_2d, qw_t, dctx)
        matmul[target="gpu"](k_flat_2d, norm_2d, kw_t, dctx)
        matmul[target="gpu"](v_flat_2d, norm_2d, vw_t, dctx)
        # 3. permute
        ctx.enqueue_function[bshd_to_bhsd, bshd_to_bhsd](
            q_perm, q_flat_4d, grid_dim=n_perm_threads, block_dim=HEAD_DIM,
        )
        ctx.enqueue_function[bshd_to_bhsd, bshd_to_bhsd](
            k_perm, k_flat_4d, grid_dim=n_perm_threads, block_dim=HEAD_DIM,
        )
        ctx.enqueue_function[bshd_to_bhsd, bshd_to_bhsd](
            v_perm, v_flat_4d, grid_dim=n_perm_threads, block_dim=HEAD_DIM,
        )
        # 4. RoPE
        ctx.enqueue_function[rope_k, rope_k](
            q_rot, q_perm, cos_t, sin_t, N_HEADS, SEQ,
            grid_dim=BATCH * N_HEADS * SEQ, block_dim=HEAD_DIM,
        )
        ctx.enqueue_function[rope_k, rope_k](
            k_rot, k_perm, cos_t, sin_t, N_HEADS, SEQ,
            grid_dim=BATCH * N_HEADS * SEQ, block_dim=HEAD_DIM,
        )
        # 5. Cache append.
        ctx.enqueue_function[cache_k, cache_k](
            k_cache_t, k_rot, T_HIST,
            grid_dim=BATCH * N_HEADS, block_dim=HEAD_DIM,
        )
        ctx.enqueue_function[cache_k, cache_k](
            v_cache_t, v_perm, T_HIST,
            grid_dim=BATCH * N_HEADS, block_dim=HEAD_DIM,
        )
        # 6. Decode attention.
        ctx.enqueue_function[qk_dec, qk_dec](
            logits_t, q_rot, k_cache_t, N_HEADS, MAX_CTX, SCALE,
            grid_dim=BATCH * N_HEADS, block_dim=MAX_CTX,
        )
        ctx.enqueue_function[sm_dec, sm_dec](
            probs_t, logits_t, N_HEADS,
            grid_dim=BATCH * N_HEADS, block_dim=SOFTMAX_BLOCK,
        )
        ctx.enqueue_function[av_dec, av_dec](
            attn_perm, probs_t, v_cache_t, N_HEADS, MAX_CTX,
            grid_dim=BATCH * N_HEADS, block_dim=HEAD_DIM,
        )
        # 7. Permute back.
        ctx.enqueue_function[bhsd_to_bshd, bhsd_to_bshd](
            attn_flat_4d, attn_perm, grid_dim=n_perm_threads, block_dim=HEAD_DIM,
        )
        # 8. o_proj.
        matmul[target="gpu"](attn_proj_2d, attn_flat_2d, ow_t, dctx)
        # 9. Residual 1.
        ctx.enqueue_function[add_k, add_k](
            post_attn_2d, x_2d, attn_proj_2d, n_x,
            grid_dim=ceildiv(n_x, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
        )
        # 10. Post-norm.
        ctx.enqueue_function[rms_k, rms_k](
            post_norm_out_2d, post_attn_2d, post_norm_t, EPS,
            grid_dim=ROWS, block_dim=RMS_BLOCK,
        )
        # 11. MLP.
        matmul[target="gpu"](gate_t, post_norm_out_2d, gw_t, dctx)
        matmul[target="gpu"](up_t, post_norm_out_2d, uw_t, dctx)
        ctx.enqueue_function[silu_k, silu_k](
            hidden_act_t, gate_t, up_t, n_silu,
            grid_dim=ceildiv(n_silu, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
        )
        matmul[target="gpu"](mlp_out_t, hidden_act_t, dw_t, dctx)
        # 12. Residual 2 — write back into x_buf for next layer.
        ctx.enqueue_function[add_k, add_k](
            x_2d, post_attn_2d, mlp_out_t, n_x,
            grid_dim=ceildiv(n_x, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
        )

        if L % 5 == 0:
            print_rss("after decode layer " + String(L))

    # Final RMSNorm.
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
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-4)
    print("30-layer decode fp32 — max abs:", max_abs)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
