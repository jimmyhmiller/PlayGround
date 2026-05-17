"""
Decode-step SDPA parity tests vs HF eager attention.

One autoregressive decode step over a KV cache of total length MAX_CTX=16
(15 history entries + 1 new entry). The test:
  1. Loads q_new, k_hist/v_hist (T_HIST=15) and k_new/v_new (1) from the oracle.
  2. Builds a single K/V cache buffer of length MAX_CTX with the new entries at
     slot T_HIST. cur_len=MAX_CTX (cache is fully populated).
  3. Runs qk_decode → softmax_decode → av_decode and compares to expected
     output (B,H,1,D).

We also run the partial-fill case at cur_len=T_HIST+1 to make sure the sentinel
masking path is correct (here, with MAX_CTX==T_HIST+1, sentinels are trivially
unused, but the structure exercises it for when MAX_CTX > cur_len in real
decode loops).
"""

from std.math import exp
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from fixture import Tensor, TensorBF16, load_fp32, load_bf16
from sdpa import qk_decode_kernel, softmax_decode_kernel, av_decode_kernel


# Match dump_decode_sdpa_case.py.
comptime BATCH = 1
comptime N_HEADS = 4
comptime T_HIST = 15
comptime MAX_CTX = T_HIST + 1   # 16; matches the prefill SEQ.
comptime HEAD_DIM = 64
comptime SCALE: Float32 = 0.125  # 1 / sqrt(64)

comptime SOFTMAX_BLOCK = 32  # one warp; MAX_CTX=16 → plenty of slack

comptime q_layout = row_major[BATCH, N_HEADS, 1, HEAD_DIM]()
comptime kv_cache_layout = row_major[BATCH, N_HEADS, MAX_CTX, HEAD_DIM]()
comptime probs_layout = row_major[BATCH, N_HEADS, 1, MAX_CTX]()


def test_decode_sdpa_full_fp32() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/decode_sdpa/"
    var q_new = load_fp32(fix + "q_new_fp32.bin")
    var k_hist = load_fp32(fix + "k_hist_fp32.bin")
    var v_hist = load_fp32(fix + "v_hist_fp32.bin")
    var k_new = load_fp32(fix + "k_new_fp32.bin")
    var v_new = load_fp32(fix + "v_new_fp32.bin")
    var exp_out = load_fp32(fix + "expected_fp32.bin")

    var n_q = BATCH * N_HEADS * 1 * HEAD_DIM
    var n_kv_cache = BATCH * N_HEADS * MAX_CTX * HEAD_DIM
    var n_probs = BATCH * N_HEADS * 1 * MAX_CTX
    var n_out = BATCH * N_HEADS * 1 * HEAD_DIM
    var ctx = DeviceContext()
    var q_buf = ctx.enqueue_create_buffer[DType.float32](n_q)
    var k_cache_buf = ctx.enqueue_create_buffer[DType.float32](n_kv_cache)
    var v_cache_buf = ctx.enqueue_create_buffer[DType.float32](n_kv_cache)
    var logits_buf = ctx.enqueue_create_buffer[DType.float32](n_probs)
    var probs_buf = ctx.enqueue_create_buffer[DType.float32](n_probs)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_out)

    # Upload q_new directly.
    with q_buf.map_to_host() as h:
        for i in range(n_q): h[i] = q_new.data[i]

    # Stitch k_hist || k_new (and same for V) into a single (B,H,MAX_CTX,D)
    # buffer with the new entry at slot T_HIST. Per-head loop because the
    # split between hist and new is on the seq axis (dim 2 of a 4D layout).
    with k_cache_buf.map_to_host() as h:
        for b in range(BATCH):
            for hd in range(N_HEADS):
                for s in range(T_HIST):
                    for d in range(HEAD_DIM):
                        var src = ((b * N_HEADS + hd) * T_HIST + s) * HEAD_DIM + d
                        var dst = ((b * N_HEADS + hd) * MAX_CTX + s) * HEAD_DIM + d
                        h[dst] = k_hist.data[src]
                # k_new[b,hd,0,d] → k_cache[b,hd,T_HIST,d]
                for d in range(HEAD_DIM):
                    var src = ((b * N_HEADS + hd) * 1 + 0) * HEAD_DIM + d
                    var dst = ((b * N_HEADS + hd) * MAX_CTX + T_HIST) * HEAD_DIM + d
                    h[dst] = k_new.data[src]
    with v_cache_buf.map_to_host() as h:
        for b in range(BATCH):
            for hd in range(N_HEADS):
                for s in range(T_HIST):
                    for d in range(HEAD_DIM):
                        var src = ((b * N_HEADS + hd) * T_HIST + s) * HEAD_DIM + d
                        var dst = ((b * N_HEADS + hd) * MAX_CTX + s) * HEAD_DIM + d
                        h[dst] = v_hist.data[src]
                for d in range(HEAD_DIM):
                    var src = ((b * N_HEADS + hd) * 1 + 0) * HEAD_DIM + d
                    var dst = ((b * N_HEADS + hd) * MAX_CTX + T_HIST) * HEAD_DIM + d
                    h[dst] = v_new.data[src]

    var q_t = TileTensor(q_buf, q_layout)
    var k_cache_t = TileTensor(k_cache_buf, kv_cache_layout)
    var v_cache_t = TileTensor(v_cache_buf, kv_cache_layout)
    var logits_t = TileTensor(logits_buf, probs_layout)
    var probs_t = TileTensor(probs_buf, probs_layout)
    var out_t = TileTensor(out_buf, q_layout)

    comptime qk_k = qk_decode_kernel[
        DType.float32, type_of(q_layout), type_of(kv_cache_layout),
        type_of(probs_layout), HEAD_DIM, MAX_CTX,
    ]
    comptime sm_k = softmax_decode_kernel[
        DType.float32, type_of(probs_layout), type_of(probs_layout),
        MAX_CTX, SOFTMAX_BLOCK,
    ]
    comptime av_k = av_decode_kernel[
        DType.float32, type_of(probs_layout), type_of(kv_cache_layout),
        type_of(q_layout), MAX_CTX, HEAD_DIM,
    ]

    # cur_len = MAX_CTX (cache fully populated).
    ctx.enqueue_function[qk_k, qk_k](
        logits_t, q_t, k_cache_t, N_HEADS, MAX_CTX, SCALE,
        grid_dim=BATCH * N_HEADS, block_dim=MAX_CTX,
    )
    ctx.enqueue_function[sm_k, sm_k](
        probs_t, logits_t, N_HEADS,
        grid_dim=BATCH * N_HEADS, block_dim=SOFTMAX_BLOCK,
    )
    ctx.enqueue_function[av_k, av_k](
        out_t, probs_t, v_cache_t, N_HEADS, MAX_CTX,
        grid_dim=BATCH * N_HEADS, block_dim=HEAD_DIM,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n_out):
            var d = h[i] - exp_out.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            assert_almost_equal(h[i], exp_out.data[i], atol=1.0e-5)
    print("decode SDPA fp32 — max abs:", max_abs)


def test_decode_sdpa_partial_fp32() raises:
    """Same data, but build a MAX_CTX-sized cache with garbage in the last
    slot and set cur_len=T_HIST. We're masking off the last slot, so the
    answer must match a "no-new-token" attention — i.e., HF eager-SDPA over
    (q_new, k_hist, v_hist) alone.

    This validates the sentinel/cur_len masking path.
    """
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/decode_sdpa/"
    var q_new = load_fp32(fix + "q_new_fp32.bin")
    var k_hist = load_fp32(fix + "k_hist_fp32.bin")
    var v_hist = load_fp32(fix + "v_hist_fp32.bin")
    var k_new = load_fp32(fix + "k_new_fp32.bin")
    var v_new = load_fp32(fix + "v_new_fp32.bin")
    # Note: we don't have an oracle for "hist-only" output, so this test does
    # a host-side oracle computation. It's a sanity check on masking — small
    # head/seq sizes so the CPU cost is fine.

    var n_q = BATCH * N_HEADS * 1 * HEAD_DIM
    var n_kv_cache = BATCH * N_HEADS * MAX_CTX * HEAD_DIM
    var n_probs = BATCH * N_HEADS * 1 * MAX_CTX
    var n_out = BATCH * N_HEADS * 1 * HEAD_DIM

    var ctx = DeviceContext()
    var q_buf = ctx.enqueue_create_buffer[DType.float32](n_q)
    var k_cache_buf = ctx.enqueue_create_buffer[DType.float32](n_kv_cache)
    var v_cache_buf = ctx.enqueue_create_buffer[DType.float32](n_kv_cache)
    var logits_buf = ctx.enqueue_create_buffer[DType.float32](n_probs)
    var probs_buf = ctx.enqueue_create_buffer[DType.float32](n_probs)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_out)

    with q_buf.map_to_host() as h:
        for i in range(n_q): h[i] = q_new.data[i]

    # Last slot deliberately set to a huge garbage value to ensure masking
    # actually drops it (if masking were broken, the output would explode).
    with k_cache_buf.map_to_host() as h:
        for b in range(BATCH):
            for hd in range(N_HEADS):
                for s in range(T_HIST):
                    for d in range(HEAD_DIM):
                        var src = ((b * N_HEADS + hd) * T_HIST + s) * HEAD_DIM + d
                        var dst = ((b * N_HEADS + hd) * MAX_CTX + s) * HEAD_DIM + d
                        h[dst] = k_hist.data[src]
                for d in range(HEAD_DIM):
                    var dst = ((b * N_HEADS + hd) * MAX_CTX + T_HIST) * HEAD_DIM + d
                    h[dst] = 1.0e6
    with v_cache_buf.map_to_host() as h:
        for b in range(BATCH):
            for hd in range(N_HEADS):
                for s in range(T_HIST):
                    for d in range(HEAD_DIM):
                        var src = ((b * N_HEADS + hd) * T_HIST + s) * HEAD_DIM + d
                        var dst = ((b * N_HEADS + hd) * MAX_CTX + s) * HEAD_DIM + d
                        h[dst] = v_hist.data[src]
                for d in range(HEAD_DIM):
                    var dst = ((b * N_HEADS + hd) * MAX_CTX + T_HIST) * HEAD_DIM + d
                    h[dst] = 1.0e6

    var q_t = TileTensor(q_buf, q_layout)
    var k_cache_t = TileTensor(k_cache_buf, kv_cache_layout)
    var v_cache_t = TileTensor(v_cache_buf, kv_cache_layout)
    var logits_t = TileTensor(logits_buf, probs_layout)
    var probs_t = TileTensor(probs_buf, probs_layout)
    var out_t = TileTensor(out_buf, q_layout)

    comptime qk_k = qk_decode_kernel[
        DType.float32, type_of(q_layout), type_of(kv_cache_layout),
        type_of(probs_layout), HEAD_DIM, MAX_CTX,
    ]
    comptime sm_k = softmax_decode_kernel[
        DType.float32, type_of(probs_layout), type_of(probs_layout),
        MAX_CTX, SOFTMAX_BLOCK,
    ]
    comptime av_k = av_decode_kernel[
        DType.float32, type_of(probs_layout), type_of(kv_cache_layout),
        type_of(q_layout), MAX_CTX, HEAD_DIM,
    ]

    ctx.enqueue_function[qk_k, qk_k](
        logits_t, q_t, k_cache_t, N_HEADS, T_HIST, SCALE,
        grid_dim=BATCH * N_HEADS, block_dim=MAX_CTX,
    )
    ctx.enqueue_function[sm_k, sm_k](
        probs_t, logits_t, N_HEADS,
        grid_dim=BATCH * N_HEADS, block_dim=SOFTMAX_BLOCK,
    )
    ctx.enqueue_function[av_k, av_k](
        out_t, probs_t, v_cache_t, N_HEADS, T_HIST,
        grid_dim=BATCH * N_HEADS, block_dim=HEAD_DIM,
    )
    ctx.synchronize()

    # CPU-side oracle: eager SDPA over (q_new, k_hist, v_hist) only.
    # logits[b,h,sk] = dot(q_new[b,h,:], k_hist[b,h,sk,:]) * scale
    # probs = softmax(logits, axis=-1)
    # out[b,h,d] = sum_sk probs[b,h,sk] * v_hist[b,h,sk,d]
    var max_abs: Float32 = 0.0
    with out_buf.map_to_host() as got:
        for b in range(BATCH):
            for hd in range(N_HEADS):
                # Compute logits over T_HIST positions.
                var logits = List[Float32]()
                for _ in range(T_HIST):
                    logits.append(0.0)
                for sk in range(T_HIST):
                    var s: Float32 = 0.0
                    for d in range(HEAD_DIM):
                        var qi = ((b * N_HEADS + hd) * 1 + 0) * HEAD_DIM + d
                        var ki = ((b * N_HEADS + hd) * T_HIST + sk) * HEAD_DIM + d
                        s += q_new.data[qi] * k_hist.data[ki]
                    logits[sk] = s * SCALE
                # softmax
                var m: Float32 = logits[0]
                for sk in range(1, T_HIST):
                    if logits[sk] > m: m = logits[sk]
                var ss: Float32 = 0.0
                for sk in range(T_HIST):
                    logits[sk] = exp(logits[sk] - m)
                    ss += logits[sk]
                for sk in range(T_HIST):
                    logits[sk] = logits[sk] / ss
                # AV
                for d in range(HEAD_DIM):
                    var acc: Float32 = 0.0
                    for sk in range(T_HIST):
                        var vi = ((b * N_HEADS + hd) * T_HIST + sk) * HEAD_DIM + d
                        acc += logits[sk] * v_hist.data[vi]
                    var oi = ((b * N_HEADS + hd) * 1 + 0) * HEAD_DIM + d
                    var diff = got[oi] - acc
                    if diff < 0.0: diff = -diff
                    if diff > max_abs: max_abs = diff
                    assert_almost_equal(Float32(got[oi]), acc, atol=1.0e-5)
    print("decode SDPA partial-fp32 — max abs:", max_abs)


def test_decode_sdpa_full_bf16() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/decode_sdpa/"
    var q_new = load_bf16(fix + "q_new_bf16.bin")
    var k_hist = load_bf16(fix + "k_hist_bf16.bin")
    var v_hist = load_bf16(fix + "v_hist_bf16.bin")
    var k_new = load_bf16(fix + "k_new_bf16.bin")
    var v_new = load_bf16(fix + "v_new_bf16.bin")
    var exp_out = load_bf16(fix + "expected_bf16.bin")

    var n_q = BATCH * N_HEADS * 1 * HEAD_DIM
    var n_kv_cache = BATCH * N_HEADS * MAX_CTX * HEAD_DIM
    var n_probs = BATCH * N_HEADS * 1 * MAX_CTX
    var n_out = BATCH * N_HEADS * 1 * HEAD_DIM

    var ctx = DeviceContext()
    var q_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_q)
    var k_cache_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_kv_cache)
    var v_cache_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_kv_cache)
    var logits_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_probs)
    var probs_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_probs)
    var out_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_out)

    with q_buf.map_to_host() as h:
        for i in range(n_q): h[i] = q_new.data[i]
    with k_cache_buf.map_to_host() as h:
        for b in range(BATCH):
            for hd in range(N_HEADS):
                for s in range(T_HIST):
                    for d in range(HEAD_DIM):
                        var src = ((b * N_HEADS + hd) * T_HIST + s) * HEAD_DIM + d
                        var dst = ((b * N_HEADS + hd) * MAX_CTX + s) * HEAD_DIM + d
                        h[dst] = k_hist.data[src]
                for d in range(HEAD_DIM):
                    var src = ((b * N_HEADS + hd) * 1 + 0) * HEAD_DIM + d
                    var dst = ((b * N_HEADS + hd) * MAX_CTX + T_HIST) * HEAD_DIM + d
                    h[dst] = k_new.data[src]
    with v_cache_buf.map_to_host() as h:
        for b in range(BATCH):
            for hd in range(N_HEADS):
                for s in range(T_HIST):
                    for d in range(HEAD_DIM):
                        var src = ((b * N_HEADS + hd) * T_HIST + s) * HEAD_DIM + d
                        var dst = ((b * N_HEADS + hd) * MAX_CTX + s) * HEAD_DIM + d
                        h[dst] = v_hist.data[src]
                for d in range(HEAD_DIM):
                    var src = ((b * N_HEADS + hd) * 1 + 0) * HEAD_DIM + d
                    var dst = ((b * N_HEADS + hd) * MAX_CTX + T_HIST) * HEAD_DIM + d
                    h[dst] = v_new.data[src]

    var q_t = TileTensor(q_buf, q_layout)
    var k_cache_t = TileTensor(k_cache_buf, kv_cache_layout)
    var v_cache_t = TileTensor(v_cache_buf, kv_cache_layout)
    var logits_t = TileTensor(logits_buf, probs_layout)
    var probs_t = TileTensor(probs_buf, probs_layout)
    var out_t = TileTensor(out_buf, q_layout)

    comptime qk_k = qk_decode_kernel[
        DType.bfloat16, type_of(q_layout), type_of(kv_cache_layout),
        type_of(probs_layout), HEAD_DIM, MAX_CTX,
    ]
    comptime sm_k = softmax_decode_kernel[
        DType.bfloat16, type_of(probs_layout), type_of(probs_layout),
        MAX_CTX, SOFTMAX_BLOCK,
    ]
    comptime av_k = av_decode_kernel[
        DType.bfloat16, type_of(probs_layout), type_of(kv_cache_layout),
        type_of(q_layout), MAX_CTX, HEAD_DIM,
    ]

    ctx.enqueue_function[qk_k, qk_k](
        logits_t, q_t, k_cache_t, N_HEADS, MAX_CTX, SCALE,
        grid_dim=BATCH * N_HEADS, block_dim=MAX_CTX,
    )
    ctx.enqueue_function[sm_k, sm_k](
        probs_t, logits_t, N_HEADS,
        grid_dim=BATCH * N_HEADS, block_dim=SOFTMAX_BLOCK,
    )
    ctx.enqueue_function[av_k, av_k](
        out_t, probs_t, v_cache_t, N_HEADS, MAX_CTX,
        grid_dim=BATCH * N_HEADS, block_dim=HEAD_DIM,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    var sum_abs: Float64 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n_out):
            var got = Float32(h[i])
            var want = Float32(exp_out.data[i])
            var d = got - want
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_abs += Float64(d)
            # Match test_sdpa.mojo's bf16 atol budget (3 fused ops + softmax).
            assert_almost_equal(got, want, atol=5.0e-2)
    print("decode SDPA bf16 — max abs:", max_abs, "mean abs:", sum_abs / Float64(n_out))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
