"""
Parity test for SinusoidalPosEmb + TimestepEmbedding in the CFM estimator.

Input:  cfm_step_00_input_t.bin     (2,)
Target: estimator_time_emb_out.bin  (2, 320)
        estimator_time_mlp_out.bin  (2, 1024)
"""
from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import load_fp32
from layernorm import linear_kernel
from attention import swish_kernel
from decoder_kernels import sinusoidal_pos_emb_kernel


comptime B = 2
comptime IN_DIM = 320
comptime OUT_DIM = 1024
comptime BLOCK = 256


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def test_time_emb_and_mlp() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/s3gen/"
    var ctx = DeviceContext()

    var t_in = load_fp32(fix + "cfm_step_00_input_t.bin")
    var exp_emb = load_fp32(fix + "estimator_time_emb_out_real.bin")
    var exp_mlp = load_fp32(fix + "estimator_time_mlp_out_real.bin")
    var w1 = load_fp32(fix + "weights/flow__decoder__estimator__time_mlp__linear_1__weight.bin")
    var b1 = load_fp32(fix + "weights/flow__decoder__estimator__time_mlp__linear_1__bias.bin")
    var w2 = load_fp32(fix + "weights/flow__decoder__estimator__time_mlp__linear_2__weight.bin")
    var b2 = load_fp32(fix + "weights/flow__decoder__estimator__time_mlp__linear_2__bias.bin")

    var t_buf = ctx.enqueue_create_buffer[DType.float32](B * 1)
    upload(t_buf, t_in.data, B)

    var emb_buf = ctx.enqueue_create_buffer[DType.float32](B * IN_DIM)
    var h1_buf = ctx.enqueue_create_buffer[DType.float32](B * OUT_DIM)
    var act_buf = ctx.enqueue_create_buffer[DType.float32](B * OUT_DIM)
    var mlp_buf = ctx.enqueue_create_buffer[DType.float32](B * OUT_DIM)
    var w1_buf = ctx.enqueue_create_buffer[DType.float32](OUT_DIM * IN_DIM)
    var b1_buf = ctx.enqueue_create_buffer[DType.float32](OUT_DIM)
    var w2_buf = ctx.enqueue_create_buffer[DType.float32](OUT_DIM * OUT_DIM)
    var b2_buf = ctx.enqueue_create_buffer[DType.float32](OUT_DIM)
    upload(w1_buf, w1.data, OUT_DIM * IN_DIM)
    upload(b1_buf, b1.data, OUT_DIM)
    upload(w2_buf, w2.data, OUT_DIM * OUT_DIM)
    upload(b2_buf, b2.data, OUT_DIM)

    comptime t_layout = row_major[B, 1]()
    comptime emb_layout = row_major[B, IN_DIM]()
    comptime mlp_layout = row_major[B, OUT_DIM]()
    comptime w1_layout = row_major[OUT_DIM, IN_DIM]()
    comptime w2_layout = row_major[OUT_DIM, OUT_DIM]()
    comptime p1_layout = row_major[OUT_DIM]()
    comptime flat_mlp = row_major[B * OUT_DIM]()

    var t_t = TileTensor(t_buf, t_layout)
    var emb_t = TileTensor(emb_buf, emb_layout)
    var w1_t = TileTensor(w1_buf, w1_layout)
    var b1_t = TileTensor(b1_buf, p1_layout)
    var h1_t = TileTensor(h1_buf, mlp_layout)
    var h1_flat = TileTensor(h1_buf, flat_mlp)
    var act_t = TileTensor(act_buf, mlp_layout)
    var act_flat = TileTensor(act_buf, flat_mlp)
    var w2_t = TileTensor(w2_buf, w2_layout)
    var b2_t = TileTensor(b2_buf, p1_layout)
    var mlp_t = TileTensor(mlp_buf, mlp_layout)

    # ---- SinusoidalPosEmb.
    comptime emb_k = sinusoidal_pos_emb_kernel[
        DType.float32, type_of(emb_layout), type_of(t_layout), IN_DIM, BLOCK,
    ]
    ctx.enqueue_function[emb_k, emb_k](
        emb_t, t_t, B, Float32(1000.0),
        grid_dim=B, block_dim=BLOCK,
    )
    ctx.synchronize()
    # Verify sinusoidal output against the dumped time_emb_out.
    var max_emb: Float32 = 0.0
    with emb_buf.map_to_host() as h:
        for i in range(B * IN_DIM):
            var d = h[i] - exp_emb.data[i]
            if d < 0.0: d = -d
            if d > max_emb: max_emb = d
            assert_almost_equal(h[i], exp_emb.data[i], atol=1.0e-4)
    print("SinusoidalPosEmb — max abs:", max_emb)

    # ---- TimestepEmbedding: Linear(IN_DIM, OUT_DIM) → SiLU → Linear(OUT_DIM, OUT_DIM).
    # Linear over rank-2 input: treat as (B, T=1, IN_DIM) for the linear_kernel.
    comptime t_btd = row_major[B, 1, IN_DIM]()
    comptime mlp_btd = row_major[B, 1, OUT_DIM]()
    var emb_btd_t = TileTensor(emb_buf, t_btd)
    var h1_btd_t = TileTensor(h1_buf, mlp_btd)
    var mlp_btd_t = TileTensor(mlp_buf, mlp_btd)
    comptime lin_k = linear_kernel[
        DType.float32, type_of(t_btd), type_of(w1_layout),
        type_of(p1_layout), type_of(mlp_btd),
        True, BLOCK,
    ]
    ctx.enqueue_function[lin_k, lin_k](
        h1_btd_t, emb_btd_t, w1_t, b1_t, B, 1, IN_DIM, OUT_DIM,
        grid_dim=B, block_dim=BLOCK,
    )
    # SiLU (== Swish).
    comptime sw_k = swish_kernel[
        DType.float32, type_of(flat_mlp), type_of(flat_mlp), BLOCK,
    ]
    ctx.enqueue_function[sw_k, sw_k](
        act_flat, h1_flat, B * OUT_DIM,
        grid_dim=ceildiv(B * OUT_DIM, BLOCK), block_dim=BLOCK,
    )
    # Linear OUT_DIM → OUT_DIM.
    comptime act_btd = row_major[B, 1, OUT_DIM]()
    var act_btd_t = TileTensor(act_buf, act_btd)
    comptime lin2_k = linear_kernel[
        DType.float32, type_of(act_btd), type_of(w2_layout),
        type_of(p1_layout), type_of(mlp_btd),
        True, BLOCK,
    ]
    ctx.enqueue_function[lin2_k, lin2_k](
        mlp_btd_t, act_btd_t, w2_t, b2_t, B, 1, OUT_DIM, OUT_DIM,
        grid_dim=B, block_dim=BLOCK,
    )
    ctx.synchronize()
    var max_mlp: Float32 = 0.0
    with mlp_buf.map_to_host() as h:
        for i in range(B * OUT_DIM):
            var d = h[i] - exp_mlp.data[i]
            if d < 0.0: d = -d
            if d > max_mlp: max_mlp = d
            assert_almost_equal(h[i], exp_mlp.data[i], atol=1.0e-3)
    print("TimestepEmbedding — max abs:", max_mlp)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
