"""
T3 head/embedding kernel parity tests.

  embed_lookup_kernel: gather speech_emb rows by token id
  add_pos_emb_kernel : add learned position embedding at base_pos..base_pos+S
  argmax_kernel      : argmax over V=8194 from speech_head logits
"""

from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from fixture import Tensor, TensorI64, load_fp32, load_i64
from heads import embed_lookup_kernel, add_pos_emb_kernel, argmax_kernel


comptime V_SPEECH = 8194
comptime D = 1024
comptime P_SPEECH = 4100
comptime ARGMAX_BLOCK = 256
comptime EMB_SEQ = 8        # matches dump_heads_case.py
comptime POS_SEQ = 8
comptime ARGMAX_SEQ = 4


def test_embed_lookup_fp32() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/heads/"
    var table = load_fp32(fix + "embed_lookup_table_fp32.bin")
    var ids = load_i64(fix + "embed_lookup_ids.bin")
    var exp = load_fp32(fix + "embed_lookup_expected_fp32.bin")

    assert_equal(table.shape[0], V_SPEECH)
    assert_equal(table.shape[1], D)
    assert_equal(ids.shape[0], 1)
    assert_equal(ids.shape[1], EMB_SEQ)

    var n_table = V_SPEECH * D
    var n_ids = 1 * EMB_SEQ
    var n_out = 1 * EMB_SEQ * D

    var ctx = DeviceContext()
    var table_buf = ctx.enqueue_create_buffer[DType.float32](n_table)
    var ids_buf = ctx.enqueue_create_buffer[DType.int64](n_ids)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_out)

    with table_buf.map_to_host() as h:
        for i in range(n_table): h[i] = table.data[i]
    with ids_buf.map_to_host() as h:
        for i in range(n_ids): h[i] = ids.data[i]

    comptime table_layout = row_major[V_SPEECH, D]()
    comptime ids_layout = row_major[1, EMB_SEQ]()
    comptime out_layout = row_major[1, EMB_SEQ, D]()

    var table_t = TileTensor(table_buf, table_layout)
    var ids_t = TileTensor(ids_buf, ids_layout)
    var out_t = TileTensor(out_buf, out_layout)

    comptime kernel = embed_lookup_kernel[
        DType.float32, type_of(ids_layout), type_of(table_layout),
        type_of(out_layout), D,
    ]
    ctx.enqueue_function[kernel, kernel](
        out_t, ids_t, table_t, 1, EMB_SEQ,
        grid_dim=1 * EMB_SEQ, block_dim=D,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n_out):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            assert_almost_equal(h[i], exp.data[i], atol=0.0)
    print("embed_lookup fp32 — max abs:", max_abs)


def test_pos_emb_add_fp32() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/heads/"
    var pos_table = load_fp32(fix + "pos_table_fp32.bin")
    var x_in = load_fp32(fix + "pos_x_fp32.bin")
    var meta = load_i64(fix + "pos_meta.bin")
    var exp = load_fp32(fix + "pos_expected_fp32.bin")

    assert_equal(pos_table.shape[0], P_SPEECH)
    assert_equal(pos_table.shape[1], D)
    var base_pos = Int(meta.data[0])
    assert_equal(Int(meta.data[1]), POS_SEQ)

    var n_pos = P_SPEECH * D
    var n_x = 1 * POS_SEQ * D

    var ctx = DeviceContext()
    var pos_buf = ctx.enqueue_create_buffer[DType.float32](n_pos)
    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_x)

    with pos_buf.map_to_host() as h:
        for i in range(n_pos): h[i] = pos_table.data[i]
    with x_buf.map_to_host() as h:
        for i in range(n_x): h[i] = x_in.data[i]

    comptime pos_layout = row_major[P_SPEECH, D]()
    comptime x_layout = row_major[1, POS_SEQ, D]()

    var pos_t = TileTensor(pos_buf, pos_layout)
    var x_t = TileTensor(x_buf, x_layout)
    var out_t = TileTensor(out_buf, x_layout)

    comptime kernel = add_pos_emb_kernel[
        DType.float32, type_of(x_layout), type_of(pos_layout), type_of(x_layout), D,
    ]
    ctx.enqueue_function[kernel, kernel](
        out_t, x_t, pos_t, 1, POS_SEQ, base_pos,
        grid_dim=1 * POS_SEQ, block_dim=D,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n_x):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-6)
    print("add_pos_emb fp32 — max abs:", max_abs)


def test_argmax_fp32() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/heads/"
    var logits = load_fp32(fix + "argmax_logits_fp32.bin")
    var exp = load_i64(fix + "argmax_expected.bin")

    assert_equal(logits.shape[2], V_SPEECH)
    assert_equal(logits.shape[1], ARGMAX_SEQ)
    var n_logits = 1 * ARGMAX_SEQ * V_SPEECH
    var n_out = 1 * ARGMAX_SEQ

    var ctx = DeviceContext()
    var logits_buf = ctx.enqueue_create_buffer[DType.float32](n_logits)
    var out_buf = ctx.enqueue_create_buffer[DType.int64](n_out)

    with logits_buf.map_to_host() as h:
        for i in range(n_logits): h[i] = logits.data[i]

    comptime logits_layout = row_major[1, ARGMAX_SEQ, V_SPEECH]()
    comptime out_layout = row_major[1, ARGMAX_SEQ]()

    var logits_t = TileTensor(logits_buf, logits_layout)
    var out_t = TileTensor(out_buf, out_layout)

    comptime kernel = argmax_kernel[
        DType.float32, type_of(logits_layout), type_of(out_layout),
        V_SPEECH, ARGMAX_BLOCK,
    ]
    ctx.enqueue_function[kernel, kernel](
        out_t, logits_t, 1, ARGMAX_SEQ,
        grid_dim=1 * ARGMAX_SEQ, block_dim=ARGMAX_BLOCK,
    )
    ctx.synchronize()

    with out_buf.map_to_host() as h:
        for i in range(n_out):
            assert_equal(Int(h[i]), Int(exp.data[i]))
    print("argmax fp32 — all", n_out, "indices match")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
