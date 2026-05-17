"""
Parity tests: SwiGLU MLP using linalg.matmul + our silu_mul kernel,
vs HF LlamaMLP forward on real T3 layer-0 weights.
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.gpu.host import DeviceContext
from std.runtime.asyncrt import DeviceContextPtr
from layout import TileTensor, row_major
from linalg.matmul import matmul

from fixture import Tensor, TensorBF16, load_fp32, load_bf16
from mlp import silu_mul_kernel


# Match dump_mlp_case.py.
comptime ROWS = 16
comptime HIDDEN = 1024
comptime INTERMEDIATE = 4096
comptime SILU_BLOCK = 256

comptime x_layout = row_major[ROWS, HIDDEN]()
comptime w_in_to_inter_layout = row_major[HIDDEN, INTERMEDIATE]()
comptime w_inter_to_in_layout = row_major[INTERMEDIATE, HIDDEN]()
comptime intermediate_layout = row_major[ROWS, INTERMEDIATE]()
comptime out_layout = row_major[ROWS, HIDDEN]()


def test_mlp_fp32() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/mlp/"
    var x = load_fp32(fix + "x_fp32.bin")
    var gw = load_fp32(fix + "gate_w_fp32.bin")
    var uw = load_fp32(fix + "up_w_fp32.bin")
    var dw = load_fp32(fix + "down_w_fp32.bin")
    var exp = load_fp32(fix + "expected_fp32.bin")

    assert_equal(x.shape[0], ROWS)
    assert_equal(x.shape[1], HIDDEN)
    assert_equal(gw.shape[0], HIDDEN)
    assert_equal(gw.shape[1], INTERMEDIATE)
    assert_equal(dw.shape[0], INTERMEDIATE)
    assert_equal(dw.shape[1], HIDDEN)

    var n_x = ROWS * HIDDEN
    var n_w_in = HIDDEN * INTERMEDIATE
    var n_w_out = INTERMEDIATE * HIDDEN
    var n_inter = ROWS * INTERMEDIATE

    var ctx = DeviceContext()
    var dctx = DeviceContextPtr(ctx)

    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var gw_buf = ctx.enqueue_create_buffer[DType.float32](n_w_in)
    var uw_buf = ctx.enqueue_create_buffer[DType.float32](n_w_in)
    var dw_buf = ctx.enqueue_create_buffer[DType.float32](n_w_out)
    var gate_buf = ctx.enqueue_create_buffer[DType.float32](n_inter)
    var up_buf = ctx.enqueue_create_buffer[DType.float32](n_inter)
    var hidden_buf = ctx.enqueue_create_buffer[DType.float32](n_inter)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_x)

    with x_buf.map_to_host() as h:
        for i in range(n_x): h[i] = x.data[i]
    with gw_buf.map_to_host() as h:
        for i in range(n_w_in): h[i] = gw.data[i]
    with uw_buf.map_to_host() as h:
        for i in range(n_w_in): h[i] = uw.data[i]
    with dw_buf.map_to_host() as h:
        for i in range(n_w_out): h[i] = dw.data[i]

    var x_t = TileTensor(x_buf, x_layout)
    var gw_t = TileTensor(gw_buf, w_in_to_inter_layout)
    var uw_t = TileTensor(uw_buf, w_in_to_inter_layout)
    var dw_t = TileTensor(dw_buf, w_inter_to_in_layout)
    var gate_t = TileTensor(gate_buf, intermediate_layout)
    var up_t = TileTensor(up_buf, intermediate_layout)
    var hidden_t = TileTensor(hidden_buf, intermediate_layout)
    var out_t = TileTensor(out_buf, out_layout)

    # gate = x @ gate_w
    matmul[target="gpu"](gate_t, x_t, gw_t, dctx)
    # up = x @ up_w
    matmul[target="gpu"](up_t, x_t, uw_t, dctx)

    # hidden = silu(gate) * up
    comptime silu_k = silu_mul_kernel[
        DType.float32, type_of(intermediate_layout), type_of(intermediate_layout),
        type_of(intermediate_layout), SILU_BLOCK,
    ]
    var n_silu = ROWS * INTERMEDIATE
    ctx.enqueue_function[silu_k, silu_k](
        hidden_t, gate_t, up_t, n_silu,
        grid_dim=ceildiv(n_silu, SILU_BLOCK), block_dim=SILU_BLOCK,
    )

    # out = hidden @ down_w
    matmul[target="gpu"](out_t, hidden_t, dw_t, dctx)
    ctx.synchronize()

    with out_buf.map_to_host() as h:
        for i in range(n_x):
            # MLP composes 3 matmuls; expect ~1e-4 drift accumulation.
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-4)


def test_mlp_bf16() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/mlp/"
    var x = load_bf16(fix + "x_bf16.bin")
    var gw = load_bf16(fix + "gate_w_bf16.bin")
    var uw = load_bf16(fix + "up_w_bf16.bin")
    var dw = load_bf16(fix + "down_w_bf16.bin")
    var exp = load_bf16(fix + "expected_bf16.bin")

    var n_x = ROWS * HIDDEN
    var n_w_in = HIDDEN * INTERMEDIATE
    var n_w_out = INTERMEDIATE * HIDDEN
    var n_inter = ROWS * INTERMEDIATE

    var ctx = DeviceContext()
    var dctx = DeviceContextPtr(ctx)

    var x_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_x)
    var gw_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_w_in)
    var uw_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_w_in)
    var dw_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_w_out)
    var gate_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_inter)
    var up_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_inter)
    var hidden_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_inter)
    var out_buf = ctx.enqueue_create_buffer[DType.bfloat16](n_x)

    with x_buf.map_to_host() as h:
        for i in range(n_x): h[i] = x.data[i]
    with gw_buf.map_to_host() as h:
        for i in range(n_w_in): h[i] = gw.data[i]
    with uw_buf.map_to_host() as h:
        for i in range(n_w_in): h[i] = uw.data[i]
    with dw_buf.map_to_host() as h:
        for i in range(n_w_out): h[i] = dw.data[i]

    var x_t = TileTensor(x_buf, x_layout)
    var gw_t = TileTensor(gw_buf, w_in_to_inter_layout)
    var uw_t = TileTensor(uw_buf, w_in_to_inter_layout)
    var dw_t = TileTensor(dw_buf, w_inter_to_in_layout)
    var gate_t = TileTensor(gate_buf, intermediate_layout)
    var up_t = TileTensor(up_buf, intermediate_layout)
    var hidden_t = TileTensor(hidden_buf, intermediate_layout)
    var out_t = TileTensor(out_buf, out_layout)

    matmul[target="gpu"](gate_t, x_t, gw_t, dctx)
    matmul[target="gpu"](up_t, x_t, uw_t, dctx)

    comptime silu_k = silu_mul_kernel[
        DType.bfloat16, type_of(intermediate_layout), type_of(intermediate_layout),
        type_of(intermediate_layout), SILU_BLOCK,
    ]
    var n_silu = ROWS * INTERMEDIATE
    ctx.enqueue_function[silu_k, silu_k](
        hidden_t, gate_t, up_t, n_silu,
        grid_dim=ceildiv(n_silu, SILU_BLOCK), block_dim=SILU_BLOCK,
    )

    matmul[target="gpu"](out_t, hidden_t, dw_t, dctx)
    ctx.synchronize()

    with out_buf.map_to_host() as h:
        for i in range(n_x):
            var got = Float32(h[i])
            var want = Float32(exp.data[i])
            # bf16 MLP: 2 matmuls (each O(K) sums in bf16) + silu*up + final matmul.
            # Errors compound; start with a generous budget and tighten if observed.
            assert_almost_equal(got, want, atol=5.0e-2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
