"""
Parity test for ConvRNNF0Predictor: mel (1, 80, T) → f0 (1, T).

Structure: 5× (Conv1d(k=3, pad=1) → ELU) → transpose(1,2) → Linear(C, 1) → squeeze → abs.

Channels: 80 → 512 → 512 → 512 → 512 → 512 → 1.

Input:  tests/fixtures/real/real_mel.bin    (1, 80, 262)
Target: tests/fixtures/real/f0_out.bin       (1, 262)
"""
from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import load_fp32
from conv import conv1d_kernel_fast
from layernorm import (
    linear_kernel, transpose_btc_to_bct_kernel, transpose_bct_to_btc_kernel,
)
from decoder_kernels import elu_kernel, abs_kernel


comptime B = 1
comptime IN_C = 80
comptime D = 512
comptime T = 262
comptime BLOCK = 256


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def upload_w(mut ctx: DeviceContext, fix: String, name: String) raises -> DeviceBuffer[DType.float32]:
    var t = load_fp32(fix + name)
    var n = len(t.data)
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    upload(buf, t.data, n)
    return buf^


def conv_elu[
    IN_: Int, OUT_: Int,
](
    mut ctx: DeviceContext,
    mut x_buf: DeviceBuffer[DType.float32],
    mut out_buf: DeviceBuffer[DType.float32],
    mut w_buf: DeviceBuffer[DType.float32],
    mut b_buf: DeviceBuffer[DType.float32],
) raises:
    """Conv1d(IN_ → OUT_, k=3, pad=1) + ELU. In-place on out_buf."""
    var n_x = B * IN_ * T
    var n_out = B * OUT_ * T
    var pre_buf = ctx.enqueue_create_buffer[DType.float32](n_out)

    comptime in_layout = row_major[B, IN_, T]()
    comptime out_layout = row_major[B, OUT_, T]()
    comptime w_layout = row_major[OUT_, IN_, 3]()
    comptime p_layout = row_major[OUT_]()
    comptime flat_out = row_major[B * OUT_ * T]()

    var x_t = TileTensor(x_buf, in_layout)
    var pre_t = TileTensor(pre_buf, out_layout)
    var w_t = TileTensor(w_buf, w_layout)
    var b_t = TileTensor(b_buf, p_layout)
    var pre_flat = TileTensor(pre_buf, flat_out)
    var out_flat = TileTensor(out_buf, flat_out)

    comptime conv_k = conv1d_kernel_fast[
        DType.float32, type_of(in_layout), type_of(w_layout),
        type_of(p_layout), type_of(out_layout),
        3, True, BLOCK,
    ]
    ctx.enqueue_function[conv_k, conv_k](
        pre_t, x_t, w_t, b_t, B, IN_, OUT_, T, T, 1, 1, 1,
        grid_dim=B * OUT_, block_dim=BLOCK,
    )
    comptime elu_k = elu_kernel[
        DType.float32, type_of(flat_out), type_of(flat_out), BLOCK,
    ]
    ctx.enqueue_function[elu_k, elu_k](
        out_flat, pre_flat, n_out,
        grid_dim=ceildiv(n_out, BLOCK), block_dim=BLOCK,
    )


def test_f0_predictor() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/real/"
    var ctx = DeviceContext()

    var mel = load_fp32(fix + "real_mel.bin")
    var exp = load_fp32(fix + "f0_out.bin")

    var w0 = upload_w(ctx, fix, "weights/mel2wav__f0_predictor__condnet__0__weight.bin")
    var b0 = upload_w(ctx, fix, "weights/mel2wav__f0_predictor__condnet__0__bias.bin")
    var w2 = upload_w(ctx, fix, "weights/mel2wav__f0_predictor__condnet__2__weight.bin")
    var b2 = upload_w(ctx, fix, "weights/mel2wav__f0_predictor__condnet__2__bias.bin")
    var w4 = upload_w(ctx, fix, "weights/mel2wav__f0_predictor__condnet__4__weight.bin")
    var b4 = upload_w(ctx, fix, "weights/mel2wav__f0_predictor__condnet__4__bias.bin")
    var w6 = upload_w(ctx, fix, "weights/mel2wav__f0_predictor__condnet__6__weight.bin")
    var b6 = upload_w(ctx, fix, "weights/mel2wav__f0_predictor__condnet__6__bias.bin")
    var w8 = upload_w(ctx, fix, "weights/mel2wav__f0_predictor__condnet__8__weight.bin")
    var b8 = upload_w(ctx, fix, "weights/mel2wav__f0_predictor__condnet__8__bias.bin")
    var cls_w = upload_w(ctx, fix, "weights/mel2wav__f0_predictor__classifier__weight.bin")
    var cls_b = upload_w(ctx, fix, "weights/mel2wav__f0_predictor__classifier__bias.bin")

    var n_in = B * IN_C * T
    var n_d = B * D * T
    var n_out = B * T

    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_in)
    var a = ctx.enqueue_create_buffer[DType.float32](n_d)
    var b = ctx.enqueue_create_buffer[DType.float32](n_d)
    var c = ctx.enqueue_create_buffer[DType.float32](n_d)
    var d = ctx.enqueue_create_buffer[DType.float32](n_d)
    var e = ctx.enqueue_create_buffer[DType.float32](n_d)
    var btc_buf = ctx.enqueue_create_buffer[DType.float32](n_d)
    var pre_abs = ctx.enqueue_create_buffer[DType.float32](n_out)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    upload(x_buf, mel.data, n_in)

    # condnet: 5 (Conv1d + ELU) layers. Note PyTorch Sequential indices 0,2,4,6,8 are conv1d.
    conv_elu[IN_C, D](ctx, x_buf, a, w0, b0)
    conv_elu[D, D](ctx, a, b, w2, b2)
    conv_elu[D, D](ctx, b, c, w4, b4)
    conv_elu[D, D](ctx, c, d, w6, b6)
    conv_elu[D, D](ctx, d, e, w8, b8)

    # transpose (B, D, T) → (B, T, D) for Linear.
    comptime bdt_layout = row_major[B, D, T]()
    comptime btd_layout = row_major[B, T, D]()
    var e_t = TileTensor(e, bdt_layout)
    var btc_t = TileTensor(btc_buf, btd_layout)
    comptime tp_k = transpose_bct_to_btc_kernel[
        DType.float32, type_of(bdt_layout), type_of(btd_layout), BLOCK,
    ]
    ctx.enqueue_function[tp_k, tp_k](
        btc_t, e_t, B, D, T, grid_dim=B * T, block_dim=BLOCK,
    )
    # Linear(D → 1, with bias). Output shape (B, T, 1).
    comptime pre_layout = row_major[B, T, 1]()
    comptime w_layout = row_major[1, D]()
    comptime p_layout = row_major[1]()
    var cls_w_t = TileTensor(cls_w, w_layout)
    var cls_b_t = TileTensor(cls_b, p_layout)
    var pre_t = TileTensor(pre_abs, pre_layout)
    comptime lin_k = linear_kernel[
        DType.float32, type_of(btd_layout), type_of(w_layout),
        type_of(p_layout), type_of(pre_layout),
        True, BLOCK,
    ]
    ctx.enqueue_function[lin_k, lin_k](
        pre_t, btc_t, cls_w_t, cls_b_t, B, T, D, 1,
        grid_dim=B * T, block_dim=BLOCK,
    )
    # abs.
    comptime flat = row_major[B * T]()
    var pre_flat = TileTensor(pre_abs, flat)
    var out_flat = TileTensor(out_buf, flat)
    comptime abs_k = abs_kernel[
        DType.float32, type_of(flat), type_of(flat), BLOCK,
    ]
    ctx.enqueue_function[abs_k, abs_k](
        out_flat, pre_flat, n_out,
        grid_dim=ceildiv(n_out, BLOCK), block_dim=BLOCK,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    var sum_abs: Float64 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n_out):
            var dv = h[i] - exp.data[i]
            if dv < 0.0: dv = -dv
            if dv > max_abs: max_abs = dv
            sum_abs += Float64(dv)
            if i < 8:
                print("f0[", i, "]: mojo=", h[i], "  torch=", exp.data[i], "  diff=", dv)
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-3)
    print("f0_predictor — max abs:", max_abs, " mean:", sum_abs / Float64(n_out))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
