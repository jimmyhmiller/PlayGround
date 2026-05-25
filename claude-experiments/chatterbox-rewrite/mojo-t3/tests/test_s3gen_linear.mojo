"""
S3Gen flow parity tests for the three linear/embedding layers around the
encoder + decoder. These are small but they validate that:
  - the dumped S3Gen weights load correctly
  - matmul + bias_add produces the right output at real S3Gen shapes
  - embed_lookup handles the real flow vocab/dim

  1. spk_embed_affine_layer:   (1, 192) -> (1, 80)         Linear
  2. flow.input_embedding:      (1, 376) i64 -> (1, 376, 512) Embedding
  3. flow.encoder_proj:         (1, 752, 512) -> (1, 752, 80)  Linear
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from layout import TileTensor, row_major
from linalg.matmul import matmul

from fixture import Tensor, TensorI64, load_fp32, load_i64
from conv import bias_add_2d_kernel
from heads import embed_lookup_kernel


comptime POINTWISE_BLOCK = 256

# Shapes captured from dump_s3gen_intermediates.py
comptime SPK_IN = 192
comptime SPK_OUT = 80
comptime VOCAB = 6561
comptime FLOW_DIM = 512
comptime FLOW_T = 376
comptime ENC_T = 752
comptime ENC_C = 512
comptime PROJ_OUT = 80


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def upload_i64(buf: DeviceBuffer[DType.int64], data: List[Int64], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def test_spk_embed_affine_fp32() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/s3gen/"
    var x = load_fp32(fix + "embedding_normed.bin")          # (1, 192)
    var w = load_fp32(fix + "weights/spk_embed_affine_layer__weight.bin")  # (80, 192)
    var b = load_fp32(fix + "weights/spk_embed_affine_layer__bias.bin")    # (80,)
    var exp = load_fp32(fix + "embedding_affine.bin")        # (1, 80)

    assert_equal(x.shape[0], 1)
    assert_equal(x.shape[1], SPK_IN)
    assert_equal(w.shape[0], SPK_OUT)
    assert_equal(w.shape[1], SPK_IN)

    var n_x = SPK_IN
    var n_w = SPK_OUT * SPK_IN
    var n_out = SPK_OUT

    var ctx = DeviceContext()
    var dctx = DeviceContextPtr(ctx)

    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    # Linear weight is (out, in); for `out = x @ w.T` we need w.T (in, out).
    var w_t_buf = ctx.enqueue_create_buffer[DType.float32](n_w)
    var b_buf = ctx.enqueue_create_buffer[DType.float32](SPK_OUT)
    var mm_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_out)

    upload(x_buf, x.data, n_x)
    # Transpose (out, in) -> (in, out) on the host.
    with w_t_buf.map_to_host() as h:
        for o in range(SPK_OUT):
            for i in range(SPK_IN):
                h[i * SPK_OUT + o] = w.data[o * SPK_IN + i]
    upload(b_buf, b.data, SPK_OUT)

    comptime x_layout = row_major[1, SPK_IN]()
    comptime w_t_layout = row_major[SPK_IN, SPK_OUT]()
    comptime b_layout = row_major[SPK_OUT]()
    comptime out_layout = row_major[1, SPK_OUT]()

    var x_t = TileTensor(x_buf, x_layout)
    var w_t = TileTensor(w_t_buf, w_t_layout)
    var b_t = TileTensor(b_buf, b_layout)
    var mm_t = TileTensor(mm_buf, out_layout)
    var out_t = TileTensor(out_buf, out_layout)

    matmul[target="gpu"](mm_t, x_t, w_t, dctx)
    comptime bias_k = bias_add_2d_kernel[
        DType.float32, type_of(out_layout), type_of(b_layout),
        type_of(out_layout), POINTWISE_BLOCK,
    ]
    ctx.enqueue_function[bias_k, bias_k](
        out_t, mm_t, b_t, 1, SPK_OUT,
        grid_dim=ceildiv(SPK_OUT, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n_out):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-4)
    print("spk_embed_affine_layer fp32 (real S3Gen weights) — max abs:", max_abs)


def test_flow_input_embedding_fp32() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/s3gen/"
    var ids = load_i64(fix + "flow_token_in.bin")            # (1, 376)
    var table = load_fp32(fix + "weights/flow__input_embedding__weight.bin")  # (6561, 512)
    var exp = load_fp32(fix + "flow_token_emb.bin")          # (1, 376, 512)

    assert_equal(ids.shape[0], 1)
    assert_equal(ids.shape[1], FLOW_T)
    assert_equal(table.shape[0], VOCAB)
    assert_equal(table.shape[1], FLOW_DIM)

    var n_table = VOCAB * FLOW_DIM
    var n_ids = FLOW_T
    var n_out = FLOW_T * FLOW_DIM

    var ctx = DeviceContext()
    var table_buf = ctx.enqueue_create_buffer[DType.float32](n_table)
    var ids_buf = ctx.enqueue_create_buffer[DType.int64](n_ids)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_out)

    upload(table_buf, table.data, n_table)
    upload_i64(ids_buf, ids.data, n_ids)

    comptime table_layout = row_major[VOCAB, FLOW_DIM]()
    comptime ids_layout = row_major[1, FLOW_T]()
    comptime out_layout = row_major[1, FLOW_T, FLOW_DIM]()

    var table_t = TileTensor(table_buf, table_layout)
    var ids_t = TileTensor(ids_buf, ids_layout)
    var out_t = TileTensor(out_buf, out_layout)

    comptime kernel = embed_lookup_kernel[
        DType.float32, type_of(ids_layout), type_of(table_layout),
        type_of(out_layout), FLOW_DIM,
    ]
    ctx.enqueue_function[kernel, kernel](
        out_t, ids_t, table_t, 1, FLOW_T,
        grid_dim=FLOW_T, block_dim=FLOW_DIM,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n_out):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            assert_almost_equal(h[i], exp.data[i], atol=0.0)
    print("flow.input_embedding fp32 (real S3Gen weights) — max abs:", max_abs)


def test_flow_encoder_proj_fp32() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/s3gen/"
    var x = load_fp32(fix + "encoder_h.bin")                 # (1, 752, 512)
    var w = load_fp32(fix + "weights/flow__encoder_proj__weight.bin")  # (80, 512)
    var b = load_fp32(fix + "weights/flow__encoder_proj__bias.bin")    # (80,)
    var exp = load_fp32(fix + "encoder_proj_h.bin")          # (1, 752, 80)

    assert_equal(x.shape[1], ENC_T)
    assert_equal(x.shape[2], ENC_C)

    var n_x = ENC_T * ENC_C
    var n_w = PROJ_OUT * ENC_C
    var n_out = ENC_T * PROJ_OUT

    var ctx = DeviceContext()
    var dctx = DeviceContextPtr(ctx)

    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var w_t_buf = ctx.enqueue_create_buffer[DType.float32](n_w)
    var b_buf = ctx.enqueue_create_buffer[DType.float32](PROJ_OUT)
    var mm_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    upload(x_buf, x.data, n_x)
    with w_t_buf.map_to_host() as h:
        for o in range(PROJ_OUT):
            for i in range(ENC_C):
                h[i * PROJ_OUT + o] = w.data[o * ENC_C + i]
    upload(b_buf, b.data, PROJ_OUT)

    comptime x_layout = row_major[ENC_T, ENC_C]()
    comptime w_t_layout = row_major[ENC_C, PROJ_OUT]()
    comptime b_layout = row_major[PROJ_OUT]()
    comptime out_layout = row_major[ENC_T, PROJ_OUT]()

    var x_t = TileTensor(x_buf, x_layout)
    var w_t = TileTensor(w_t_buf, w_t_layout)
    var b_t = TileTensor(b_buf, b_layout)
    var mm_t = TileTensor(mm_buf, out_layout)
    var out_t = TileTensor(out_buf, out_layout)

    matmul[target="gpu"](mm_t, x_t, w_t, dctx)
    comptime bias_k = bias_add_2d_kernel[
        DType.float32, type_of(out_layout), type_of(b_layout),
        type_of(out_layout), POINTWISE_BLOCK,
    ]
    ctx.enqueue_function[bias_k, bias_k](
        out_t, mm_t, b_t, ENC_T, PROJ_OUT,
        grid_dim=ceildiv(ENC_T * PROJ_OUT, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n_out):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-4)
    print("flow.encoder_proj fp32 (real S3Gen weights) — max abs:", max_abs)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
