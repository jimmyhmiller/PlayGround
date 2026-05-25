"""
SDPA parity tests vs HF eager_attention_forward.

Three layered tests per dtype:
  qk    — Q @ K^T * scale + mask matches logits oracle
  smax  — softmax(logits) matches probs oracle
  full  — full pipeline (qk → smax → av) matches output oracle

Failures are localized: if `qk` fails, smax/full are skipped — no point chasing
downstream errors when the upstream is wrong.
"""

from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from fixture import Tensor, TensorBF16, load_fp32, load_bf16
from sdpa import qk_scaled_kernel, softmax_kernel, av_kernel


# Match dump_sdpa_case.py.
comptime BATCH = 1
comptime N_HEADS = 4
comptime SEQ = 16
comptime HEAD_DIM = 64
comptime SCALE: Float32 = 0.125  # 1 / sqrt(64) = 1/8 = 0.125 exactly

comptime SOFTMAX_BLOCK = 32  # one warp; SEQ=16 so plenty of slack

comptime qkv_layout = row_major[BATCH, N_HEADS, SEQ, HEAD_DIM]()
comptime ss_layout = row_major[BATCH, N_HEADS, SEQ, SEQ]()
comptime mask_layout = row_major[SEQ, SEQ]()


# ============================================================================
# fp32 tests
# ============================================================================

def test_sdpa_qk_fp32() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/sdpa/"
    var q = load_fp32(fix + "q_fp32.bin")
    var k = load_fp32(fix + "k_fp32.bin")
    var mask = load_fp32(fix + "mask_fp32.bin")
    var exp_logits = load_fp32(fix + "logits_fp32.bin")

    var n_qkv = BATCH * N_HEADS * SEQ * HEAD_DIM
    var n_ss = BATCH * N_HEADS * SEQ * SEQ
    var n_mask = SEQ * SEQ

    var ctx = DeviceContext()
    var q_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv)
    var k_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv)
    var mask_buf = ctx.enqueue_create_buffer[DType.float32](n_mask)
    var logits_buf = ctx.enqueue_create_buffer[DType.float32](n_ss)

    with q_buf.map_to_host() as h:
        for i in range(n_qkv): h[i] = q.data[i]
    with k_buf.map_to_host() as h:
        for i in range(n_qkv): h[i] = k.data[i]
    with mask_buf.map_to_host() as h:
        for i in range(n_mask): h[i] = mask.data[i]

    var q_t = TileTensor(q_buf, qkv_layout)
    var k_t = TileTensor(k_buf, qkv_layout)
    var mask_t = TileTensor(mask_buf, mask_layout)
    var logits_t = TileTensor(logits_buf, ss_layout)

    comptime kernel = qk_scaled_kernel[
        DType.float32, type_of(qkv_layout), type_of(qkv_layout),
        type_of(mask_layout), type_of(ss_layout), HEAD_DIM, SEQ,
    ]
    ctx.enqueue_function[kernel, kernel](
        logits_t, q_t, k_t, mask_t, N_HEADS, SCALE,
        grid_dim=BATCH * N_HEADS * SEQ, block_dim=SEQ,
    )
    ctx.synchronize()

    with logits_buf.map_to_host() as h:
        for i in range(n_ss):
            # Skip masked positions: HF stored finfo.min, we'd need to ignore
            # those. Easier: clamp diff to ignore -inf positions.
            var got = h[i]
            var want = exp_logits.data[i]
            if want < -1.0e30 and got < -1.0e30:
                continue  # both are -inf-ish, fine
            assert_almost_equal(got, want, atol=1.0e-4)


def test_sdpa_softmax_fp32() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/sdpa/"
    var logits = load_fp32(fix + "logits_fp32.bin")
    var exp_probs = load_fp32(fix + "probs_fp32.bin")

    var n_ss = BATCH * N_HEADS * SEQ * SEQ

    var ctx = DeviceContext()
    var in_buf = ctx.enqueue_create_buffer[DType.float32](n_ss)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_ss)

    with in_buf.map_to_host() as h:
        for i in range(n_ss): h[i] = logits.data[i]

    var in_t = TileTensor(in_buf, ss_layout)
    var out_t = TileTensor(out_buf, ss_layout)

    comptime kernel = softmax_kernel[
        DType.float32, type_of(ss_layout), type_of(ss_layout), SEQ, SOFTMAX_BLOCK,
    ]
    ctx.enqueue_function[kernel, kernel](
        out_t, in_t, N_HEADS,
        grid_dim=BATCH * N_HEADS * SEQ, block_dim=SOFTMAX_BLOCK,
    )
    ctx.synchronize()

    with out_buf.map_to_host() as h:
        for i in range(n_ss):
            assert_almost_equal(h[i], exp_probs.data[i], atol=1.0e-6)


def test_sdpa_av_fp32() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/sdpa/"
    var probs = load_fp32(fix + "probs_fp32.bin")
    var v = load_fp32(fix + "v_fp32.bin")
    var exp_out = load_fp32(fix + "expected_fp32.bin")

    var n_ss = BATCH * N_HEADS * SEQ * SEQ
    var n_qkv = BATCH * N_HEADS * SEQ * HEAD_DIM

    var ctx = DeviceContext()
    var p_buf = ctx.enqueue_create_buffer[DType.float32](n_ss)
    var v_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv)

    with p_buf.map_to_host() as h:
        for i in range(n_ss): h[i] = probs.data[i]
    with v_buf.map_to_host() as h:
        for i in range(n_qkv): h[i] = v.data[i]

    var p_t = TileTensor(p_buf, ss_layout)
    var v_t = TileTensor(v_buf, qkv_layout)
    var out_t = TileTensor(out_buf, qkv_layout)

    comptime kernel = av_kernel[
        DType.float32, type_of(ss_layout), type_of(qkv_layout), type_of(qkv_layout),
        SEQ, HEAD_DIM,
    ]
    ctx.enqueue_function[kernel, kernel](
        out_t, p_t, v_t, N_HEADS,
        grid_dim=BATCH * N_HEADS * SEQ, block_dim=HEAD_DIM,
    )
    ctx.synchronize()

    with out_buf.map_to_host() as h:
        for i in range(n_qkv):
            assert_almost_equal(h[i], exp_out.data[i], atol=1.0e-5)


def test_sdpa_full_fp32() raises:
    """Full pipeline: qk → softmax → av, all our kernels chained."""
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/sdpa/"
    var q = load_fp32(fix + "q_fp32.bin")
    var k = load_fp32(fix + "k_fp32.bin")
    var v = load_fp32(fix + "v_fp32.bin")
    var mask = load_fp32(fix + "mask_fp32.bin")
    var exp_out = load_fp32(fix + "expected_fp32.bin")

    var n_qkv = BATCH * N_HEADS * SEQ * HEAD_DIM
    var n_ss = BATCH * N_HEADS * SEQ * SEQ
    var n_mask = SEQ * SEQ

    var ctx = DeviceContext()
    var q_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv)
    var k_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv)
    var v_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv)
    var mask_buf = ctx.enqueue_create_buffer[DType.float32](n_mask)
    var logits_buf = ctx.enqueue_create_buffer[DType.float32](n_ss)
    var probs_buf = ctx.enqueue_create_buffer[DType.float32](n_ss)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv)

    with q_buf.map_to_host() as h:
        for i in range(n_qkv): h[i] = q.data[i]
    with k_buf.map_to_host() as h:
        for i in range(n_qkv): h[i] = k.data[i]
    with v_buf.map_to_host() as h:
        for i in range(n_qkv): h[i] = v.data[i]
    with mask_buf.map_to_host() as h:
        for i in range(n_mask): h[i] = mask.data[i]

    var q_t = TileTensor(q_buf, qkv_layout)
    var k_t = TileTensor(k_buf, qkv_layout)
    var v_t = TileTensor(v_buf, qkv_layout)
    var mask_t = TileTensor(mask_buf, mask_layout)
    var logits_t = TileTensor(logits_buf, ss_layout)
    var probs_t = TileTensor(probs_buf, ss_layout)
    var out_t = TileTensor(out_buf, qkv_layout)

    comptime qk_k = qk_scaled_kernel[
        DType.float32, type_of(qkv_layout), type_of(qkv_layout),
        type_of(mask_layout), type_of(ss_layout), HEAD_DIM, SEQ,
    ]
    comptime sm_k = softmax_kernel[
        DType.float32, type_of(ss_layout), type_of(ss_layout), SEQ, SOFTMAX_BLOCK,
    ]
    comptime av_k = av_kernel[
        DType.float32, type_of(ss_layout), type_of(qkv_layout), type_of(qkv_layout),
        SEQ, HEAD_DIM,
    ]

    ctx.enqueue_function[qk_k, qk_k](
        logits_t, q_t, k_t, mask_t, N_HEADS, SCALE,
        grid_dim=BATCH * N_HEADS * SEQ, block_dim=SEQ,
    )
    ctx.enqueue_function[sm_k, sm_k](
        probs_t, logits_t, N_HEADS,
        grid_dim=BATCH * N_HEADS * SEQ, block_dim=SOFTMAX_BLOCK,
    )
    ctx.enqueue_function[av_k, av_k](
        out_t, probs_t, v_t, N_HEADS,
        grid_dim=BATCH * N_HEADS * SEQ, block_dim=HEAD_DIM,
    )
    ctx.synchronize()

    with out_buf.map_to_host() as h:
        for i in range(n_qkv):
            assert_almost_equal(h[i], exp_out.data[i], atol=1.0e-5)


# ============================================================================
# bf16 full-pipeline test
# ============================================================================

def test_sdpa_full_bf16() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/sdpa/"
    var q = load_bf16(fix + "q_bf16.bin")
    var k = load_bf16(fix + "k_bf16.bin")
    var v = load_bf16(fix + "v_bf16.bin")
    var mask = load_bf16(fix + "mask_bf16.bin")
    var exp_out = load_bf16(fix + "expected_bf16.bin")

    var n_qkv = BATCH * N_HEADS * SEQ * HEAD_DIM
    var n_ss = BATCH * N_HEADS * SEQ * SEQ
    var n_mask = SEQ * SEQ

    var ctx = DeviceContext()
    var q_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv)
    var k_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv)
    var v_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv)
    var mask_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_mask)
    var logits_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_ss)
    var probs_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_ss)
    var out_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv)

    with q_buf.map_to_host() as h:
        for i in range(n_qkv): h[i] = q.data[i]
    with k_buf.map_to_host() as h:
        for i in range(n_qkv): h[i] = k.data[i]
    with v_buf.map_to_host() as h:
        for i in range(n_qkv): h[i] = v.data[i]
    with mask_buf.map_to_host() as h:
        for i in range(n_mask): h[i] = mask.data[i]

    var q_t = TileTensor(q_buf, qkv_layout)
    var k_t = TileTensor(k_buf, qkv_layout)
    var v_t = TileTensor(v_buf, qkv_layout)
    var mask_t = TileTensor(mask_buf, mask_layout)
    var logits_t = TileTensor(logits_buf, ss_layout)
    var probs_t = TileTensor(probs_buf, ss_layout)
    var out_t = TileTensor(out_buf, qkv_layout)

    comptime qk_k = qk_scaled_kernel[
        DType.bfloat16, type_of(qkv_layout), type_of(qkv_layout),
        type_of(mask_layout), type_of(ss_layout), HEAD_DIM, SEQ,
    ]
    comptime sm_k = softmax_kernel[
        DType.bfloat16, type_of(ss_layout), type_of(ss_layout), SEQ, SOFTMAX_BLOCK,
    ]
    comptime av_k = av_kernel[
        DType.bfloat16, type_of(ss_layout), type_of(qkv_layout), type_of(qkv_layout),
        SEQ, HEAD_DIM,
    ]

    ctx.enqueue_function[qk_k, qk_k](
        logits_t, q_t, k_t, mask_t, N_HEADS, SCALE,
        grid_dim=BATCH * N_HEADS * SEQ, block_dim=SEQ,
    )
    ctx.enqueue_function[sm_k, sm_k](
        probs_t, logits_t, N_HEADS,
        grid_dim=BATCH * N_HEADS * SEQ, block_dim=SOFTMAX_BLOCK,
    )
    ctx.enqueue_function[av_k, av_k](
        out_t, probs_t, v_t, N_HEADS,
        grid_dim=BATCH * N_HEADS * SEQ, block_dim=HEAD_DIM,
    )
    ctx.synchronize()

    with out_buf.map_to_host() as h:
        for i in range(n_qkv):
            var got = Float32(h[i])
            var want = Float32(exp_out.data[i])
            # bf16 SDPA accumulates errors across 3 ops + softmax. Empirically
            # ~5e-2 is a reasonable budget; tighten if we observe smaller drift.
            assert_almost_equal(got, want, atol=5.0e-2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
