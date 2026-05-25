"""
Parity test for the small flow "glue" pieces:
  - flow.input_embedding(tokens) — vocab lookup (6561, 512).
  - flow.spk_embed_affine_layer(xvector_normalized) — Linear 192 → 80.

Inputs:
  flow_token_in.bin (1, 376) int64
  embedding_normed.bin (1, 192) — already F.normalize'd xvector
Targets:
  flow_input_emb_out.bin (1, 376, 512)
  spk_affine_out.bin (1, 80)
"""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import load_fp32, load_i64
from layernorm import linear_kernel
from flow_glue import embedding_lookup_kernel


comptime B = 1
comptime T_TOK = 376
comptime VOCAB = 6561
comptime D_EMB = 512
comptime D_SPK_IN = 192
comptime D_SPK_OUT = 80
comptime BLOCK = 256


def upload_fp32(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def upload_i64(buf: DeviceBuffer[DType.int64], data: List[Int64], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def test_input_embedding() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/s3gen/"
    var ctx = DeviceContext()

    var tokens = load_i64(fix + "flow_token_in.bin")
    var w = load_fp32(fix + "weights/flow__input_embedding__weight.bin")
    var exp = load_fp32(fix + "flow_input_emb_out.bin")

    var n_tok = B * T_TOK
    var n_w = VOCAB * D_EMB
    var n_out = B * T_TOK * D_EMB

    var tok_buf = ctx.enqueue_create_buffer[DType.int64](n_tok)
    var w_buf = ctx.enqueue_create_buffer[DType.float32](n_w)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    upload_i64(tok_buf, tokens.data, n_tok)
    upload_fp32(w_buf, w.data, n_w)

    comptime tok_layout = row_major[B, T_TOK]()
    comptime w_layout = row_major[VOCAB, D_EMB]()
    comptime out_layout = row_major[B, T_TOK, D_EMB]()
    var tok_t = TileTensor(tok_buf, tok_layout)
    var w_t = TileTensor(w_buf, w_layout)
    var out_t = TileTensor(out_buf, out_layout)

    comptime k = embedding_lookup_kernel[
        DType.float32, type_of(out_layout), type_of(tok_layout), type_of(w_layout),
        D_EMB, BLOCK,
    ]
    ctx.enqueue_function[k, k](
        out_t, tok_t, w_t, B, T_TOK,
        grid_dim=B * T_TOK, block_dim=BLOCK,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n_out):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-6)
    print("flow.input_embedding — max abs:", max_abs)


def test_spk_embed_affine() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/s3gen/"
    var ctx = DeviceContext()

    var xv = load_fp32(fix + "embedding_normed.bin")
    var w = load_fp32(fix + "weights/flow__spk_embed_affine_layer__weight.bin")
    var b = load_fp32(fix + "weights/flow__spk_embed_affine_layer__bias.bin")
    var exp = load_fp32(fix + "spk_affine_out.bin")

    var n_in = B * D_SPK_IN
    var n_w = D_SPK_OUT * D_SPK_IN
    var n_out = B * D_SPK_OUT

    var xv_buf = ctx.enqueue_create_buffer[DType.float32](n_in)
    var w_buf = ctx.enqueue_create_buffer[DType.float32](n_w)
    var b_buf = ctx.enqueue_create_buffer[DType.float32](D_SPK_OUT)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    upload_fp32(xv_buf, xv.data, n_in)
    upload_fp32(w_buf, w.data, n_w)
    upload_fp32(b_buf, b.data, D_SPK_OUT)

    # linear_kernel expects (B, T, D_in) → (B, T, D_out). Use T=1.
    comptime in_layout = row_major[B, 1, D_SPK_IN]()
    comptime w_layout = row_major[D_SPK_OUT, D_SPK_IN]()
    comptime out_layout = row_major[B, 1, D_SPK_OUT]()
    comptime p_layout = row_major[D_SPK_OUT]()
    var xv_t = TileTensor(xv_buf, in_layout)
    var w_t = TileTensor(w_buf, w_layout)
    var b_t = TileTensor(b_buf, p_layout)
    var out_t = TileTensor(out_buf, out_layout)

    comptime lin_k = linear_kernel[
        DType.float32, type_of(in_layout), type_of(w_layout),
        type_of(p_layout), type_of(out_layout),
        True, BLOCK,
    ]
    ctx.enqueue_function[lin_k, lin_k](
        out_t, xv_t, w_t, b_t, B, 1, D_SPK_IN, D_SPK_OUT,
        grid_dim=B, block_dim=BLOCK,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n_out):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            if i < 8:
                print("spk[", i, "]: mojo=", h[i], "  torch=", exp.data[i], "  diff=", d)
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-5)
    print("flow.spk_embed_affine_layer — max abs:", max_abs)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
