"""Parity test: text_tokens → embed + add_pos_emb → input embeddings."""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import load_fp32
from heads import embed_lookup_kernel, add_pos_emb_kernel


comptime B = 1
comptime T_TEXT = 17
comptime VOCAB = 704
comptime D = 1024


def upload_f32(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def upload_i64(buf: DeviceBuffer[DType.int64], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = Int64(Int(data[i]))


def test_text_prefix() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/text_prefix/"
    var ctx = DeviceContext()

    var tokens_t = load_fp32(fix + "text_tokens.bin")
    var exp = load_fp32(fix + "out.bin")
    var emb_w_t = load_fp32(fix + "text_emb_w.bin")
    var pos_w_t = load_fp32(fix + "text_pos_w.bin")

    var tokens_buf = ctx.enqueue_create_buffer[DType.int64](B * T_TEXT)
    upload_i64(tokens_buf, tokens_t.data, B * T_TEXT)
    var emb_w = ctx.enqueue_create_buffer[DType.float32](VOCAB * D)
    upload_f32(emb_w, emb_w_t.data, VOCAB * D)
    var pos_w = ctx.enqueue_create_buffer[DType.float32](2050 * D)
    upload_f32(pos_w, pos_w_t.data, 2050 * D)

    var text_emb_buf = ctx.enqueue_create_buffer[DType.float32](B * T_TEXT * D)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](B * T_TEXT * D)

    comptime tokens_layout = row_major[B, T_TEXT]()
    comptime emb_table_layout = row_major[VOCAB, D]()
    comptime pos_table_layout = row_major[2050, D]()
    comptime emb_out_layout = row_major[B, T_TEXT, D]()

    var tokens_tt = TileTensor(tokens_buf, tokens_layout)
    var emb_w_tt = TileTensor(emb_w, emb_table_layout)
    var pos_w_tt = TileTensor(pos_w, pos_table_layout)
    var text_emb_tt = TileTensor(text_emb_buf, emb_out_layout)
    var out_tt = TileTensor(out_buf, emb_out_layout)

    # Step 1: text_emb = embed_table[text_tokens].
    comptime kemb = embed_lookup_kernel[
        DType.float32, type_of(tokens_layout), type_of(emb_table_layout),
        type_of(emb_out_layout), D,
    ]
    ctx.enqueue_function[kemb, kemb](
        text_emb_tt, tokens_tt, emb_w_tt, B, T_TEXT,
        grid_dim=B * T_TEXT, block_dim=D,
    )

    # Step 2: out = text_emb + pos_emb[0..T_TEXT-1].
    comptime kpos = add_pos_emb_kernel[
        DType.float32, type_of(emb_out_layout), type_of(pos_table_layout),
        type_of(emb_out_layout), D,
    ]
    ctx.enqueue_function[kpos, kpos](
        out_tt, text_emb_tt, pos_w_tt, B, T_TEXT, 0,
        grid_dim=B * T_TEXT, block_dim=D,
    )
    ctx.synchronize()

    var n_out = B * T_TEXT * D
    var max_abs: Float32 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n_out):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            if i < 8:
                print("tp[", i, "]: mojo=", h[i], "  torch=", exp.data[i], "  diff=", d)
    print("Text prefix — max abs:", max_abs)
    assert_almost_equal(max_abs, 0.0, atol=1.0e-5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
