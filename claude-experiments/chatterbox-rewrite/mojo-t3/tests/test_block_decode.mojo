"""
Single transformer block — decode step — parity test vs HF.

The oracle (dump_block_decode_case.py) runs a 15-token prefill on the same
block, dumps the resulting K/V (BHSD) as `k_hist` / `v_hist`, and dumps the
single-token decode-step output as `expected`.

This Mojo test reuses the oracle's k_hist / v_hist directly (no prefill on
the Mojo side), then runs the full decode-step block:

  norm_x  = rmsnorm(x_decode, in_norm_w)            (B, 1, H)
  q_flat  = norm_x @ qw                              (B, 1, H)
  k_flat  = norm_x @ kw
  v_flat  = norm_x @ vw
  q, k_new, v_new = reshape → permute → (B, H, 1, D)
  q, k_new = rope(q, k_new, cos_decode, sin_decode)
  k_cache = concat(k_hist, k_new) along seq dim      (B, H, T_TOTAL, D)
  v_cache = concat(v_hist, v_new)
  attn    = qk_decode → softmax_decode → av_decode   (B, H, 1, D)
  attn    = permute → flatten → (B, 1, H)
  attn    = attn @ ow
  x       = x_decode + attn
  norm_x  = rmsnorm(x, post_norm_w)
  mlp_out = silu_mul(norm_x @ gate_w, norm_x @ up_w) @ down_w
  out     = x + mlp_out

We skip the prefill pass because (a) it would require a different-SEQ
instantiation of all the kernels and (b) prefill is already covered by
test_block.mojo. The cache contents we receive from the oracle are exactly
what a correct prefill would produce.
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
from sdpa import qk_decode_kernel, softmax_decode_kernel, av_decode_kernel
from mlp import silu_mul_kernel
from util_kernels import add_kernel, bshd_to_bhsd_kernel, bhsd_to_bshd_kernel


# T3 Llama config — same as test_block.
comptime BATCH = 1
comptime T_HIST = 15
comptime SEQ = 1                    # decode-step seq length
comptime MAX_CTX = T_HIST + SEQ     # 16
comptime N_HEADS = 16
comptime HEAD_DIM = 64
comptime HIDDEN = N_HEADS * HEAD_DIM            # 1024
comptime INTERMEDIATE = 4096
comptime SCALE: Float32 = 0.125                  # 1/sqrt(64)
comptime EPS: Float32 = 1.0e-5
comptime ROPE_HALF = HEAD_DIM // 2

comptime RMS_BLOCK = 256
comptime ROWS = BATCH * SEQ                      # 1 row in decode
comptime POINTWISE_BLOCK = 256
comptime SOFTMAX_BLOCK = 32

# Layouts.
comptime x_2d_layout = row_major[ROWS, HIDDEN]()
comptime w_hh_layout = row_major[HIDDEN, HIDDEN]()
comptime w_in_inter_layout = row_major[HIDDEN, INTERMEDIATE]()
comptime w_inter_in_layout = row_major[INTERMEDIATE, HIDDEN]()
comptime intermediate_layout = row_major[ROWS, INTERMEDIATE]()
comptime bshd_layout_dec = row_major[BATCH, SEQ, N_HEADS, HEAD_DIM]()
comptime bhsd_layout_dec = row_major[BATCH, N_HEADS, SEQ, HEAD_DIM]()
comptime kv_cache_layout = row_major[BATCH, N_HEADS, MAX_CTX, HEAD_DIM]()
comptime cs_layout_dec = row_major[BATCH, SEQ, HEAD_DIM]()
comptime probs_layout = row_major[BATCH, N_HEADS, 1, MAX_CTX]()
comptime norm_w_layout = row_major[HIDDEN]()


def test_block_decode_fp32() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/block_decode/"
    var x = load_fp32(fix + "x_decode_fp32.bin")
    var cos = load_fp32(fix + "cos_decode_fp32.bin")
    var sin = load_fp32(fix + "sin_decode_fp32.bin")
    var k_hist = load_fp32(fix + "k_hist_fp32.bin")
    var v_hist = load_fp32(fix + "v_hist_fp32.bin")
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

    var n_x = ROWS * HIDDEN                          # 1024
    var n_cs = BATCH * SEQ * HEAD_DIM                # 64
    var n_qkv_flat = ROWS * HIDDEN                    # 1024
    var n_qkv_perm = BATCH * N_HEADS * SEQ * HEAD_DIM # 1024
    var n_kv_cache = BATCH * N_HEADS * MAX_CTX * HEAD_DIM
    var n_probs = BATCH * N_HEADS * 1 * MAX_CTX
    var n_inter = ROWS * INTERMEDIATE

    var ctx = DeviceContext()
    var dctx = DeviceContextPtr(ctx)

    # ---- Buffers ----
    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var cos_buf = ctx.enqueue_create_buffer[DType.float32](n_cs)
    var sin_buf = ctx.enqueue_create_buffer[DType.float32](n_cs)
    var in_norm_buf = ctx.enqueue_create_buffer[DType.float32](HIDDEN)
    var post_norm_buf = ctx.enqueue_create_buffer[DType.float32](HIDDEN)
    var qw_buf = ctx.enqueue_create_buffer[DType.float32](HIDDEN * HIDDEN)
    var kw_buf = ctx.enqueue_create_buffer[DType.float32](HIDDEN * HIDDEN)
    var vw_buf = ctx.enqueue_create_buffer[DType.float32](HIDDEN * HIDDEN)
    var ow_buf = ctx.enqueue_create_buffer[DType.float32](HIDDEN * HIDDEN)
    var gw_buf = ctx.enqueue_create_buffer[DType.float32](HIDDEN * INTERMEDIATE)
    var uw_buf = ctx.enqueue_create_buffer[DType.float32](HIDDEN * INTERMEDIATE)
    var dw_buf = ctx.enqueue_create_buffer[DType.float32](INTERMEDIATE * HIDDEN)

    var norm_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var q_flat_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv_flat)
    var k_flat_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv_flat)
    var v_flat_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv_flat)
    var q_perm_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv_perm)
    var k_perm_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv_perm)
    var v_perm_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv_perm)
    var q_rot_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv_perm)
    var k_rot_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv_perm)
    var k_cache_buf = ctx.enqueue_create_buffer[DType.float32](n_kv_cache)
    var v_cache_buf = ctx.enqueue_create_buffer[DType.float32](n_kv_cache)
    var logits_buf = ctx.enqueue_create_buffer[DType.float32](n_probs)
    var probs_buf = ctx.enqueue_create_buffer[DType.float32](n_probs)
    var attn_perm_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv_perm)
    var attn_flat_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv_flat)
    var attn_proj_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var post_attn_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var post_norm_out_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var gate_buf = ctx.enqueue_create_buffer[DType.float32](n_inter)
    var up_buf = ctx.enqueue_create_buffer[DType.float32](n_inter)
    var hidden_act_buf = ctx.enqueue_create_buffer[DType.float32](n_inter)
    var mlp_out_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_x)

    # ---- Stage inputs ----
    with x_buf.map_to_host() as h:
        for i in range(n_x): h[i] = x.data[i]
    with cos_buf.map_to_host() as h:
        for i in range(n_cs): h[i] = cos.data[i]
    with sin_buf.map_to_host() as h:
        for i in range(n_cs): h[i] = sin.data[i]
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

    # Cache buffers: first T_HIST slots from oracle k_hist/v_hist, last slot
    # left for k_new/v_new which we'll write into directly after RoPE.
    with k_cache_buf.map_to_host() as h:
        for b in range(BATCH):
            for hd in range(N_HEADS):
                for s in range(T_HIST):
                    for d in range(HEAD_DIM):
                        var src = ((b * N_HEADS + hd) * T_HIST + s) * HEAD_DIM + d
                        var dst = ((b * N_HEADS + hd) * MAX_CTX + s) * HEAD_DIM + d
                        h[dst] = k_hist.data[src]
    with v_cache_buf.map_to_host() as h:
        for b in range(BATCH):
            for hd in range(N_HEADS):
                for s in range(T_HIST):
                    for d in range(HEAD_DIM):
                        var src = ((b * N_HEADS + hd) * T_HIST + s) * HEAD_DIM + d
                        var dst = ((b * N_HEADS + hd) * MAX_CTX + s) * HEAD_DIM + d
                        h[dst] = v_hist.data[src]

    # ---- TileTensor views ----
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
    var q_flat_4d = TileTensor(q_flat_buf, bshd_layout_dec)
    var k_flat_4d = TileTensor(k_flat_buf, bshd_layout_dec)
    var v_flat_4d = TileTensor(v_flat_buf, bshd_layout_dec)
    var q_perm = TileTensor(q_perm_buf, bhsd_layout_dec)
    var k_perm = TileTensor(k_perm_buf, bhsd_layout_dec)
    var v_perm = TileTensor(v_perm_buf, bhsd_layout_dec)
    var q_rot = TileTensor(q_rot_buf, bhsd_layout_dec)
    var k_rot = TileTensor(k_rot_buf, bhsd_layout_dec)
    var k_cache_t = TileTensor(k_cache_buf, kv_cache_layout)
    var v_cache_t = TileTensor(v_cache_buf, kv_cache_layout)
    var cos_t = TileTensor(cos_buf, cs_layout_dec)
    var sin_t = TileTensor(sin_buf, cs_layout_dec)
    var logits_t = TileTensor(logits_buf, probs_layout)
    var probs_t = TileTensor(probs_buf, probs_layout)
    var attn_perm = TileTensor(attn_perm_buf, bhsd_layout_dec)
    var attn_flat_4d = TileTensor(attn_flat_buf, bshd_layout_dec)
    var attn_flat_2d = TileTensor(attn_flat_buf, x_2d_layout)
    var attn_proj_2d = TileTensor(attn_proj_buf, x_2d_layout)
    var post_attn_2d = TileTensor(post_attn_buf, x_2d_layout)
    var post_norm_out_2d = TileTensor(post_norm_out_buf, x_2d_layout)
    var gate_t = TileTensor(gate_buf, intermediate_layout)
    var up_t = TileTensor(up_buf, intermediate_layout)
    var hidden_act_t = TileTensor(hidden_act_buf, intermediate_layout)
    var mlp_out_t = TileTensor(mlp_out_buf, x_2d_layout)
    var out_t = TileTensor(out_buf, x_2d_layout)

    # ---- Kernel bindings ----
    comptime rms_k = rmsnorm_kernel[
        DType.float32, type_of(x_2d_layout), type_of(norm_w_layout),
        type_of(x_2d_layout), RMS_BLOCK,
    ]
    comptime bshd_to_bhsd = bshd_to_bhsd_kernel[
        DType.float32, type_of(bshd_layout_dec), type_of(bhsd_layout_dec),
        BATCH, SEQ, N_HEADS, HEAD_DIM,
    ]
    comptime bhsd_to_bshd = bhsd_to_bshd_kernel[
        DType.float32, type_of(bhsd_layout_dec), type_of(bshd_layout_dec),
        BATCH, SEQ, N_HEADS, HEAD_DIM,
    ]
    comptime rope_k = rope_kernel[
        DType.float32, type_of(bhsd_layout_dec), type_of(cs_layout_dec),
        type_of(bhsd_layout_dec), HEAD_DIM, ROPE_HALF,
    ]
    comptime qk_dec = qk_decode_kernel[
        DType.float32, type_of(bhsd_layout_dec), type_of(kv_cache_layout),
        type_of(probs_layout), HEAD_DIM, MAX_CTX,
    ]
    comptime sm_dec = softmax_decode_kernel[
        DType.float32, type_of(probs_layout), type_of(probs_layout),
        MAX_CTX, SOFTMAX_BLOCK,
    ]
    comptime av_dec = av_decode_kernel[
        DType.float32, type_of(probs_layout), type_of(kv_cache_layout),
        type_of(bhsd_layout_dec), MAX_CTX, HEAD_DIM,
    ]
    comptime add_k = add_kernel[
        DType.float32, type_of(x_2d_layout), type_of(x_2d_layout),
        type_of(x_2d_layout), POINTWISE_BLOCK,
    ]
    comptime silu_k = silu_mul_kernel[
        DType.float32, type_of(intermediate_layout), type_of(intermediate_layout),
        type_of(intermediate_layout), POINTWISE_BLOCK,
    ]

    # ---- Pipeline ----
    # 1. norm = rmsnorm(x_decode, in_norm)
    ctx.enqueue_function[rms_k, rms_k](
        norm_2d, x_2d, in_norm_t, EPS,
        grid_dim=ROWS, block_dim=RMS_BLOCK,
    )

    # 2. q/k/v = norm @ w_*
    matmul[target="gpu"](q_flat_2d, norm_2d, qw_t, dctx)
    matmul[target="gpu"](k_flat_2d, norm_2d, kw_t, dctx)
    matmul[target="gpu"](v_flat_2d, norm_2d, vw_t, dctx)

    # 3. (B,1,H,D) → (B,H,1,D)
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

    # 4. RoPE on q and k_new (decode-step positions).
    ctx.enqueue_function[rope_k, rope_k](
        q_rot, q_perm, cos_t, sin_t, N_HEADS, SEQ,
        grid_dim=BATCH * N_HEADS * SEQ, block_dim=HEAD_DIM,
    )
    ctx.enqueue_function[rope_k, rope_k](
        k_rot, k_perm, cos_t, sin_t, N_HEADS, SEQ,
        grid_dim=BATCH * N_HEADS * SEQ, block_dim=HEAD_DIM,
    )

    # 5. Append k_rot/v_perm to the cache at slot T_HIST.
    # The cache is (B,H,MAX_CTX,D); we synchronize host-side reads first.
    ctx.synchronize()
    with k_rot_buf.map_to_host() as kr:
        with k_cache_buf.map_to_host() as kc:
            for b in range(BATCH):
                for hd in range(N_HEADS):
                    for d in range(HEAD_DIM):
                        var src = ((b * N_HEADS + hd) * SEQ + 0) * HEAD_DIM + d
                        var dst = ((b * N_HEADS + hd) * MAX_CTX + T_HIST) * HEAD_DIM + d
                        kc[dst] = kr[src]
    with v_perm_buf.map_to_host() as vr:
        with v_cache_buf.map_to_host() as vc:
            for b in range(BATCH):
                for hd in range(N_HEADS):
                    for d in range(HEAD_DIM):
                        var src = ((b * N_HEADS + hd) * SEQ + 0) * HEAD_DIM + d
                        var dst = ((b * N_HEADS + hd) * MAX_CTX + T_HIST) * HEAD_DIM + d
                        vc[dst] = vr[src]

    # 6. Decode-attention: qk_decode → softmax → av_decode (cur_len=MAX_CTX).
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

    # 7. Permute back (B,H,1,D) → (B,1,H,D).
    ctx.enqueue_function[bhsd_to_bshd, bhsd_to_bshd](
        attn_flat_4d, attn_perm, grid_dim=n_perm_threads, block_dim=HEAD_DIM,
    )

    # 8. attn_proj = attn_flat @ ow
    matmul[target="gpu"](attn_proj_2d, attn_flat_2d, ow_t, dctx)

    # 9. post_attn = x + attn_proj (residual 1)
    ctx.enqueue_function[add_k, add_k](
        post_attn_2d, x_2d, attn_proj_2d, n_x,
        grid_dim=ceildiv(n_x, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )

    # 10. post_norm = rmsnorm(post_attn, post_norm)
    ctx.enqueue_function[rms_k, rms_k](
        post_norm_out_2d, post_attn_2d, post_norm_t, EPS,
        grid_dim=ROWS, block_dim=RMS_BLOCK,
    )

    # 11. MLP
    matmul[target="gpu"](gate_t, post_norm_out_2d, gw_t, dctx)
    matmul[target="gpu"](up_t, post_norm_out_2d, uw_t, dctx)
    var n_silu = ROWS * INTERMEDIATE
    ctx.enqueue_function[silu_k, silu_k](
        hidden_act_t, gate_t, up_t, n_silu,
        grid_dim=ceildiv(n_silu, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )
    matmul[target="gpu"](mlp_out_t, hidden_act_t, dw_t, dctx)

    # 12. out = post_attn + mlp_out (residual 2)
    ctx.enqueue_function[add_k, add_k](
        out_t, post_attn_2d, mlp_out_t, n_x,
        grid_dim=ceildiv(n_x, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n_x):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-3)
    print("block decode fp32 — max abs:", max_abs)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
