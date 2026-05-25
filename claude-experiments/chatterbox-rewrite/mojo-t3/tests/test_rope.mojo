"""
Parity tests: Mojo GPU RoPE kernel vs HF apply_rotary_pos_emb.

Two cases (fp32 and bf16) loaded from oracle/dump_rope_case.py.
"""

from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from fixture import Tensor, TensorBF16, load_fp32, load_bf16
from rope import rope_kernel


# Match dump_rope_case.py.
comptime BATCH = 1
comptime N_HEADS = 4
comptime SEQ = 16
comptime HEAD_DIM = 64
comptime HALF = HEAD_DIM // 2

comptime q_layout = row_major[BATCH, N_HEADS, SEQ, HEAD_DIM]()
comptime cs_layout = row_major[BATCH, SEQ, HEAD_DIM]()


def test_rope_fp32() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/rope/"
    var q = load_fp32(fix + "q_fp32.bin")
    var cos = load_fp32(fix + "cos_fp32.bin")
    var sin = load_fp32(fix + "sin_fp32.bin")
    var exp = load_fp32(fix + "expected_fp32.bin")

    assert_equal(q.shape[0], BATCH)
    assert_equal(q.shape[1], N_HEADS)
    assert_equal(q.shape[2], SEQ)
    assert_equal(q.shape[3], HEAD_DIM)
    assert_equal(cos.shape[0], BATCH)
    assert_equal(cos.shape[1], SEQ)
    assert_equal(cos.shape[2], HEAD_DIM)

    var n_q = BATCH * N_HEADS * SEQ * HEAD_DIM
    var n_cs = BATCH * SEQ * HEAD_DIM

    var ctx = DeviceContext()
    var q_buf = ctx.enqueue_create_buffer[DType.float32](n_q)
    var cos_buf = ctx.enqueue_create_buffer[DType.float32](n_cs)
    var sin_buf = ctx.enqueue_create_buffer[DType.float32](n_cs)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_q)

    with q_buf.map_to_host() as h:
        for i in range(n_q):
            h[i] = q.data[i]
    with cos_buf.map_to_host() as h:
        for i in range(n_cs):
            h[i] = cos.data[i]
    with sin_buf.map_to_host() as h:
        for i in range(n_cs):
            h[i] = sin.data[i]

    var q_t = TileTensor(q_buf, q_layout)
    var cos_t = TileTensor(cos_buf, cs_layout)
    var sin_t = TileTensor(sin_buf, cs_layout)
    var out_t = TileTensor(out_buf, q_layout)

    comptime kernel = rope_kernel[
        DType.float32,
        type_of(q_layout),
        type_of(cs_layout),
        type_of(q_layout),
        HEAD_DIM,
        HALF,
    ]
    ctx.enqueue_function[kernel, kernel](
        out_t, q_t, cos_t, sin_t, N_HEADS, SEQ,
        grid_dim=BATCH * N_HEADS * SEQ,
        block_dim=HEAD_DIM,
    )
    ctx.synchronize()

    with out_buf.map_to_host() as h:
        for i in range(n_q):
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-5)


def test_rope_bf16() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/rope/"
    var q = load_bf16(fix + "q_bf16.bin")
    var cos = load_bf16(fix + "cos_bf16.bin")
    var sin = load_bf16(fix + "sin_bf16.bin")
    var exp = load_bf16(fix + "expected_bf16.bin")

    var n_q = BATCH * N_HEADS * SEQ * HEAD_DIM
    var n_cs = BATCH * SEQ * HEAD_DIM

    var ctx = DeviceContext()
    var q_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_q)
    var cos_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_cs)
    var sin_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_cs)
    var out_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_q)

    with q_buf.map_to_host() as h:
        for i in range(n_q):
            h[i] = q.data[i]
    with cos_buf.map_to_host() as h:
        for i in range(n_cs):
            h[i] = cos.data[i]
    with sin_buf.map_to_host() as h:
        for i in range(n_cs):
            h[i] = sin.data[i]

    var q_t = TileTensor(q_buf, q_layout)
    var cos_t = TileTensor(cos_buf, cs_layout)
    var sin_t = TileTensor(sin_buf, cs_layout)
    var out_t = TileTensor(out_buf, q_layout)

    comptime kernel = rope_kernel[
        DType.bfloat16,
        type_of(q_layout),
        type_of(cs_layout),
        type_of(q_layout),
        HEAD_DIM,
        HALF,
    ]
    ctx.enqueue_function[kernel, kernel](
        out_t, q_t, cos_t, sin_t, N_HEADS, SEQ,
        grid_dim=BATCH * N_HEADS * SEQ,
        block_dim=HEAD_DIM,
    )
    ctx.synchronize()

    with out_buf.map_to_host() as h:
        for i in range(n_q):
            var got = Float32(h[i])
            var want = Float32(exp.data[i])
            assert_almost_equal(got, want, atol=2.0e-2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
