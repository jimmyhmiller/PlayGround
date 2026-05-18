"""
Parity test for the encoder.embed Linear + LayerNorm + xscale pipeline.

Input:  flow_token_emb.bin   (1, 376, 512)   — encoder input
Target: enc_embed_xs.bin     (1, 376, 512)   — embed output (Linear → LayerNorm → * sqrt(d_model))

The dropout is a no-op in eval mode.
"""
from std.math import ceildiv, sqrt
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu import block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major

from fixture import load_fp32
from layernorm import layernorm_kernel, linear_kernel


comptime B = 1
comptime T = 376
comptime D = 512
comptime BLOCK = 256
comptime EPS: Float32 = 1.0e-5
comptime XSCALE: Float32 = 22.627417   # sqrt(512)


def scale_kernel[
    dtype: DType, InLayout: TensorLayout, OutLayout: TensorLayout, BLOCK_: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],
    n: Int, scale: Float32,
):
    """Pointwise out = inp * scale."""
    comptime assert inp.flat_rank == 1
    comptime assert output.flat_rank == 1
    var idx = block_idx.x * BLOCK_ + thread_idx.x
    if idx >= n:
        return
    var v = rebind[Scalar[dtype]](inp[idx]).cast[DType.float32]()
    output[idx] = rebind[output.ElementType]((v * scale).cast[dtype]())


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def test_encoder_embed() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/s3gen/"
    var ctx = DeviceContext()

    var x_in = load_fp32(fix + "flow_token_emb.bin")  # (1, 376, 512)
    var w_lin = load_fp32(fix + "weights/flow__encoder__embed__out__0__weight.bin")  # (512, 512)
    var b_lin = load_fp32(fix + "weights/flow__encoder__embed__out__0__bias.bin")
    var w_ln = load_fp32(fix + "weights/flow__encoder__embed__out__1__weight.bin")
    var b_ln = load_fp32(fix + "weights/flow__encoder__embed__out__1__bias.bin")
    var exp = load_fp32(fix + "enc_embed_xs.bin")     # (1, 376, 512)

    var n_x = B * T * D
    var n_w = D * D

    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var w_lin_buf = ctx.enqueue_create_buffer[DType.float32](n_w)
    var b_lin_buf = ctx.enqueue_create_buffer[DType.float32](D)
    var w_ln_buf = ctx.enqueue_create_buffer[DType.float32](D)
    var b_ln_buf = ctx.enqueue_create_buffer[DType.float32](D)
    var lin_out = ctx.enqueue_create_buffer[DType.float32](n_x)
    var ln_out = ctx.enqueue_create_buffer[DType.float32](n_x)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_x)

    upload(x_buf, x_in.data, n_x)
    upload(w_lin_buf, w_lin.data, n_w)
    upload(b_lin_buf, b_lin.data, D)
    upload(w_ln_buf, w_ln.data, D)
    upload(b_ln_buf, b_ln.data, D)

    comptime in_layout = row_major[B, T, D]()
    comptime w_layout = row_major[D, D]()
    comptime p_layout = row_major[D]()
    comptime flat = row_major[B * T * D]()

    var x_t = TileTensor(x_buf, in_layout)
    var w_lin_t = TileTensor(w_lin_buf, w_layout)
    var b_lin_t = TileTensor(b_lin_buf, p_layout)
    var w_ln_t = TileTensor(w_ln_buf, p_layout)
    var b_ln_t = TileTensor(b_ln_buf, p_layout)
    var lin_out_t = TileTensor(lin_out, in_layout)
    var ln_out_t = TileTensor(ln_out, in_layout)
    var ln_out_flat = TileTensor(ln_out, flat)
    var out_flat = TileTensor(out_buf, flat)

    comptime lin_k = linear_kernel[
        DType.float32, type_of(in_layout), type_of(w_layout),
        type_of(p_layout), type_of(in_layout),
        True, BLOCK,
    ]
    ctx.enqueue_function[lin_k, lin_k](
        lin_out_t, x_t, w_lin_t, b_lin_t,
        B, T, D, D,
        grid_dim=B * T, block_dim=BLOCK,
    )
    comptime ln_k = layernorm_kernel[
        DType.float32, type_of(in_layout), type_of(p_layout),
        type_of(in_layout), BLOCK,
    ]
    ctx.enqueue_function[ln_k, ln_k](
        ln_out_t, lin_out_t, w_ln_t, b_ln_t,
        B, T, D, EPS,
        grid_dim=B * T, block_dim=BLOCK,
    )
    comptime sc_k = scale_kernel[
        DType.float32, type_of(flat), type_of(flat), BLOCK,
    ]
    ctx.enqueue_function[sc_k, sc_k](
        out_flat, ln_out_flat, n_x, XSCALE,
        grid_dim=ceildiv(n_x, BLOCK), block_dim=BLOCK,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    var sum_abs: Float64 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n_x):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_abs += Float64(d)
            if i < 8:
                print("emb[", i, "]: mojo=", h[i], "  torch=", exp.data[i], "  diff=", d)
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-3)
    print("encoder.embed (Linear + LayerNorm + xscale) — max abs:", max_abs,
          " mean:", sum_abs / Float64(n_x))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
