"""
Parity tests: Mojo GPU RMSNorm kernel vs HF LlamaRMSNorm.

Two cases:
  fp32: input/weight/output all fp32, atol 1e-6 (the loose part is bf16).
  bf16: input/weight/output all bf16; tests the cast-point sequence matches
        HF (cast in→fp32, fp32 reduce/rsqrt, cast back→bf16, bf16 weight mul).

Tolerance for bf16 comes from per-layer drift observed in oracle extraction
(bf16 vs fp32 layer-0 RMSNorm differs by ~1e-3 max abs).
"""

from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from fixture import Tensor, TensorBF16, load_fp32, load_bf16
from rmsnorm import rmsnorm_kernel


# Matches T3 Llama config: hidden_size=1024. Fixture has rows = batch*seq = 16.
comptime HIDDEN = 1024
comptime ROWS = 16
comptime BLOCK = 256

comptime inp_layout_2d = row_major[ROWS, HIDDEN]()
comptime wgt_layout_1d = row_major[HIDDEN]()


def test_rmsnorm_fp32() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/rmsnorm/"
    var inp = load_fp32(fix + "input_fp32.bin")
    var wgt = load_fp32(fix + "weight_fp32.bin")
    var exp = load_fp32(fix + "expected_fp32.bin")

    assert_equal(inp.shape[0] * inp.shape[1], ROWS)
    assert_equal(inp.shape[2], HIDDEN)
    assert_equal(wgt.shape[0], HIDDEN)

    var n = ROWS * HIDDEN
    var ctx = DeviceContext()

    var inp_buf = ctx.enqueue_create_buffer[DType.float32](n)
    var wgt_buf = ctx.enqueue_create_buffer[DType.float32](HIDDEN)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n)

    with inp_buf.map_to_host() as h:
        for i in range(n):
            h[i] = inp.data[i]
    with wgt_buf.map_to_host() as h:
        for i in range(HIDDEN):
            h[i] = wgt.data[i]

    var inp_t = TileTensor(inp_buf, inp_layout_2d)
    var wgt_t = TileTensor(wgt_buf, wgt_layout_1d)
    var out_t = TileTensor(out_buf, inp_layout_2d)

    comptime kernel = rmsnorm_kernel[
        DType.float32,
        type_of(inp_layout_2d),
        type_of(wgt_layout_1d),
        type_of(inp_layout_2d),
        BLOCK,
    ]
    ctx.enqueue_function[kernel, kernel](
        out_t, inp_t, wgt_t, Float32(1.0e-5),
        grid_dim=ROWS, block_dim=BLOCK,
    )
    ctx.synchronize()

    with out_buf.map_to_host() as h:
        for i in range(n):
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-6)


def test_rmsnorm_bf16() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/rmsnorm/"
    var inp = load_bf16(fix + "input_bf16.bin")
    var wgt = load_bf16(fix + "weight_bf16.bin")
    var exp = load_bf16(fix + "expected_bf16.bin")

    assert_equal(inp.shape[0] * inp.shape[1], ROWS)
    assert_equal(inp.shape[2], HIDDEN)
    assert_equal(wgt.shape[0], HIDDEN)

    var n = ROWS * HIDDEN
    var ctx = DeviceContext()

    var inp_buf = ctx.enqueue_create_buffer[DType.bfloat16](n)
    var wgt_buf = ctx.enqueue_create_buffer[DType.bfloat16](HIDDEN)
    var out_buf = ctx.enqueue_create_buffer[DType.bfloat16](n)

    with inp_buf.map_to_host() as h:
        for i in range(n):
            h[i] = inp.data[i]
    with wgt_buf.map_to_host() as h:
        for i in range(HIDDEN):
            h[i] = wgt.data[i]

    var inp_t = TileTensor(inp_buf, inp_layout_2d)
    var wgt_t = TileTensor(wgt_buf, wgt_layout_1d)
    var out_t = TileTensor(out_buf, inp_layout_2d)

    comptime kernel = rmsnorm_kernel[
        DType.bfloat16,
        type_of(inp_layout_2d),
        type_of(wgt_layout_1d),
        type_of(inp_layout_2d),
        BLOCK,
    ]
    ctx.enqueue_function[kernel, kernel](
        out_t, inp_t, wgt_t, Float32(1.0e-5),
        grid_dim=ROWS, block_dim=BLOCK,
    )
    ctx.synchronize()

    # bf16 RMSNorm parity: per-element bf16 has ~7 mantissa bits, so on values
    # ~0.2 the smallest representable step is ~1.5e-3. We compare in fp32 space
    # with that as our atol budget.
    with out_buf.map_to_host() as h:
        for i in range(n):
            var got = Float32(h[i])
            var want = Float32(exp.data[i])
            assert_almost_equal(got, want, atol=2.0e-3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
