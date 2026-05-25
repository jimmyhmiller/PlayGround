"""
Time embedding parity tests for the CFM ConditionalDecoder.
  1. SinusoidalPosEmb(t=t_span[0]=0) → (2, 320)         time_emb_out
  2. TimestepEmbedding(time_emb_out) → (2, 1024)         time_mlp_out
     = linear_1 + silu + linear_2
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from layout import TileTensor, row_major
from linalg.matmul import matmul

from fixture import Tensor, load_fp32
from time_emb import sinusoidal_pos_emb_kernel, silu_kernel
from conv import bias_add_2d_kernel


comptime BATCH = 2          # CFG doubled
comptime IN_DIM = 320
comptime HALF_DIM = IN_DIM // 2
comptime HIDDEN = 1024
comptime POINTWISE_BLOCK = 256
comptime SCALE: Float32 = 1000.0


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def test_sinusoidal_pos_emb_fp32() raises:
    comptime assert has_accelerator(), "Requires GPU"

    # At step 0, t = 0 (broadcast to both CFG halves). Upstream calls
    # time_embeddings on a tensor of shape (2,) with values both 0.
    var fix = "tests/fixtures/s3gen/"
    var exp = load_fp32(fix + "estimator_time_emb_out.bin")
    assert_equal(exp.shape[0], BATCH)
    assert_equal(exp.shape[1], IN_DIM)

    var n_t = BATCH
    var n_out = BATCH * IN_DIM

    var ctx = DeviceContext()
    var t_buf = ctx.enqueue_create_buffer[DType.float32](n_t)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_out)

    # t = [0, 0]
    var t_data = List[Float32](capacity=BATCH)
    for _ in range(BATCH):
        t_data.append(0.0)
    upload(t_buf, t_data, n_t)

    comptime t_layout = row_major[BATCH]()
    comptime out_layout = row_major[BATCH, IN_DIM]()

    var t_t = TileTensor(t_buf, t_layout)
    var out_t = TileTensor(out_buf, out_layout)

    comptime kernel = sinusoidal_pos_emb_kernel[
        DType.float32, type_of(t_layout), type_of(out_layout),
        IN_DIM, HALF_DIM,
    ]
    ctx.enqueue_function[kernel, kernel](
        out_t, t_t, BATCH, SCALE,
        grid_dim=BATCH, block_dim=IN_DIM,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n_out):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-6)
    print("SinusoidalPosEmb fp32 (t=0) — max abs:", max_abs)


def test_time_mlp_fp32() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/s3gen/"
    var x = load_fp32(fix + "estimator_time_emb_out.bin")     # (2, 320)
    var w1 = load_fp32(fix + "weights/flow__decoder__estimator__time_mlp__linear_1__weight.bin")  # (1024, 320)
    var b1 = load_fp32(fix + "weights/flow__decoder__estimator__time_mlp__linear_1__bias.bin")    # (1024,)
    var w2 = load_fp32(fix + "weights/flow__decoder__estimator__time_mlp__linear_2__weight.bin")  # (1024, 1024)
    var b2 = load_fp32(fix + "weights/flow__decoder__estimator__time_mlp__linear_2__bias.bin")    # (1024,)
    var exp = load_fp32(fix + "estimator_time_mlp_out.bin")   # (2, 1024)

    var n_x = BATCH * IN_DIM
    var n_w1 = HIDDEN * IN_DIM
    var n_h = BATCH * HIDDEN
    var n_w2 = HIDDEN * HIDDEN

    var ctx = DeviceContext()
    var dctx = DeviceContextPtr(ctx)

    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var w1t_buf = ctx.enqueue_create_buffer[DType.float32](n_w1)
    var b1_buf = ctx.enqueue_create_buffer[DType.float32](HIDDEN)
    var mm1_buf = ctx.enqueue_create_buffer[DType.float32](n_h)
    var h1_buf = ctx.enqueue_create_buffer[DType.float32](n_h)
    var act_buf = ctx.enqueue_create_buffer[DType.float32](n_h)
    var w2t_buf = ctx.enqueue_create_buffer[DType.float32](n_w2)
    var b2_buf = ctx.enqueue_create_buffer[DType.float32](HIDDEN)
    var mm2_buf = ctx.enqueue_create_buffer[DType.float32](n_h)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_h)

    upload(x_buf, x.data, n_x)
    upload(b1_buf, b1.data, HIDDEN)
    upload(b2_buf, b2.data, HIDDEN)
    # Transpose linear weights from (out, in) to (in, out) for matmul.
    with w1t_buf.map_to_host() as h:
        for o in range(HIDDEN):
            for i in range(IN_DIM):
                h[i * HIDDEN + o] = w1.data[o * IN_DIM + i]
    with w2t_buf.map_to_host() as h:
        for o in range(HIDDEN):
            for i in range(HIDDEN):
                h[i * HIDDEN + o] = w2.data[o * HIDDEN + i]

    comptime x_layout = row_major[BATCH, IN_DIM]()
    comptime w1t_layout = row_major[IN_DIM, HIDDEN]()
    comptime h_layout = row_major[BATCH, HIDDEN]()
    comptime h_flat_layout = row_major[BATCH * HIDDEN]()
    comptime w2t_layout = row_major[HIDDEN, HIDDEN]()
    comptime bias_layout = row_major[HIDDEN]()

    var x_t = TileTensor(x_buf, x_layout)
    var w1t_t = TileTensor(w1t_buf, w1t_layout)
    var b1_t = TileTensor(b1_buf, bias_layout)
    var mm1_t = TileTensor(mm1_buf, h_layout)
    var h1_t = TileTensor(h1_buf, h_layout)
    var h1_flat = TileTensor(h1_buf, h_flat_layout)
    var act_t = TileTensor(act_buf, h_layout)
    var act_flat = TileTensor(act_buf, h_flat_layout)
    var w2t_t = TileTensor(w2t_buf, w2t_layout)
    var b2_t = TileTensor(b2_buf, bias_layout)
    var mm2_t = TileTensor(mm2_buf, h_layout)
    var out_t = TileTensor(out_buf, h_layout)

    comptime bias_k = bias_add_2d_kernel[
        DType.float32, type_of(h_layout), type_of(bias_layout),
        type_of(h_layout), POINTWISE_BLOCK,
    ]
    comptime silu_k = silu_kernel[
        DType.float32, type_of(h_flat_layout), type_of(h_flat_layout),
        POINTWISE_BLOCK,
    ]

    # linear_1: mm = x @ w1.T; h1 = mm + b1
    matmul[target="gpu"](mm1_t, x_t, w1t_t, dctx)
    ctx.enqueue_function[bias_k, bias_k](
        h1_t, mm1_t, b1_t, BATCH, HIDDEN,
        grid_dim=ceildiv(n_h, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )
    # silu
    ctx.enqueue_function[silu_k, silu_k](
        act_flat, h1_flat, n_h,
        grid_dim=ceildiv(n_h, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )
    # linear_2: mm = act @ w2.T; out = mm + b2
    matmul[target="gpu"](mm2_t, act_t, w2t_t, dctx)
    ctx.enqueue_function[bias_k, bias_k](
        out_t, mm2_t, b2_t, BATCH, HIDDEN,
        grid_dim=ceildiv(n_h, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n_h):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-4)
    print("TimestepEmbedding fp32 (real S3Gen) — max abs:", max_abs)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
