"""
Parity test for the full CAMPPlus xvector.block1 (12 dense TDNN layers).

Input:  tdnn_out.bin       (1, 128, 499)  — output of xvector.tdnn
Target: block1_out.bin     (1, 512, 499)  — 128 + 12*32 = 512

Uses cam_dense_tdnn_layer helper from src/campplus.mojo.
"""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer

from fixture import load_fp32
from campplus import cam_dense_tdnn_layer, DenseTdnnWeights


comptime B = 1
comptime T = 499
comptime KERNEL = 3
comptime DILATION = 1     # block1 has dilation=1
comptime SEG_LEN = 100
comptime GROWTH = 32
comptime BN_C = 128
comptime HALF_BN = 64
comptime NUM_LAYERS = 12
comptime INIT_C = 128


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


def load_dense_weights(mut ctx: DeviceContext, fix: String, layer_idx: Int) raises -> DenseTdnnWeights:
    """Load weights for xvector.block1.tdnnd{layer_idx+1}."""
    var prefix = "weights/xvector__block1__tdnnd" + String(layer_idx + 1) + "__"
    return DenseTdnnWeights(
        upload_w(ctx, fix, prefix + "nonlinear1__batchnorm__weight.bin"),
        upload_w(ctx, fix, prefix + "nonlinear1__batchnorm__bias.bin"),
        upload_w(ctx, fix, prefix + "nonlinear1__batchnorm__running_mean.bin"),
        upload_w(ctx, fix, prefix + "nonlinear1__batchnorm__running_var.bin"),
        upload_w(ctx, fix, prefix + "linear1__weight.bin"),
        upload_w(ctx, fix, prefix + "nonlinear2__batchnorm__weight.bin"),
        upload_w(ctx, fix, prefix + "nonlinear2__batchnorm__bias.bin"),
        upload_w(ctx, fix, prefix + "nonlinear2__batchnorm__running_mean.bin"),
        upload_w(ctx, fix, prefix + "nonlinear2__batchnorm__running_var.bin"),
        upload_w(ctx, fix, prefix + "cam_layer__linear_local__weight.bin"),
        upload_w(ctx, fix, prefix + "cam_layer__linear1__weight.bin"),
        upload_w(ctx, fix, prefix + "cam_layer__linear1__bias.bin"),
        upload_w(ctx, fix, prefix + "cam_layer__linear2__weight.bin"),
        upload_w(ctx, fix, prefix + "cam_layer__linear2__bias.bin"),
    )


def test_block1() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/campplus/"
    var ctx = DeviceContext()

    var x_in = load_fp32(fix + "tdnn_out.bin")
    var exp = load_fp32(fix + "block1_out.bin")
    var n_in = B * INIT_C * T
    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_in)
    upload(x_buf, x_in.data, n_in)

    # Allocate one buffer per layer output. Each layer i takes IN_C = INIT_C+i*GROWTH
    # and produces OUT_C = INIT_C+(i+1)*GROWTH.
    # We can't reuse a single buffer because each call needs distinct buffers.
    var w0 = load_dense_weights(ctx, fix, 0)
    var w1 = load_dense_weights(ctx, fix, 1)
    var w2 = load_dense_weights(ctx, fix, 2)
    var w3 = load_dense_weights(ctx, fix, 3)
    var w4 = load_dense_weights(ctx, fix, 4)
    var w5 = load_dense_weights(ctx, fix, 5)
    var w6 = load_dense_weights(ctx, fix, 6)
    var w7 = load_dense_weights(ctx, fix, 7)
    var w8 = load_dense_weights(ctx, fix, 8)
    var w9 = load_dense_weights(ctx, fix, 9)
    var w10 = load_dense_weights(ctx, fix, 10)
    var w11 = load_dense_weights(ctx, fix, 11)

    # One dummy bn buffer per layer (mut-aliasing).
    var d0 = ctx.enqueue_create_buffer[DType.float32](BN_C)
    var d1 = ctx.enqueue_create_buffer[DType.float32](BN_C)
    var d2 = ctx.enqueue_create_buffer[DType.float32](BN_C)
    var d3 = ctx.enqueue_create_buffer[DType.float32](BN_C)
    var d4 = ctx.enqueue_create_buffer[DType.float32](BN_C)
    var d5 = ctx.enqueue_create_buffer[DType.float32](BN_C)
    var d6 = ctx.enqueue_create_buffer[DType.float32](BN_C)
    var d7 = ctx.enqueue_create_buffer[DType.float32](BN_C)
    var d8 = ctx.enqueue_create_buffer[DType.float32](BN_C)
    var d9 = ctx.enqueue_create_buffer[DType.float32](BN_C)
    var d10 = ctx.enqueue_create_buffer[DType.float32](BN_C)
    var d11 = ctx.enqueue_create_buffer[DType.float32](BN_C)

    # Allocate output buffers for each of the 12 layers.
    # IN_C_i = 128 + i*32; OUT_C_i = IN_C_{i+1} = 160 + i*32
    var out0 = ctx.enqueue_create_buffer[DType.float32](B * 160 * T)
    var out1 = ctx.enqueue_create_buffer[DType.float32](B * 192 * T)
    var out2 = ctx.enqueue_create_buffer[DType.float32](B * 224 * T)
    var out3 = ctx.enqueue_create_buffer[DType.float32](B * 256 * T)
    var out4 = ctx.enqueue_create_buffer[DType.float32](B * 288 * T)
    var out5 = ctx.enqueue_create_buffer[DType.float32](B * 320 * T)
    var out6 = ctx.enqueue_create_buffer[DType.float32](B * 352 * T)
    var out7 = ctx.enqueue_create_buffer[DType.float32](B * 384 * T)
    var out8 = ctx.enqueue_create_buffer[DType.float32](B * 416 * T)
    var out9 = ctx.enqueue_create_buffer[DType.float32](B * 448 * T)
    var out10 = ctx.enqueue_create_buffer[DType.float32](B * 480 * T)
    var out11 = ctx.enqueue_create_buffer[DType.float32](B * 512 * T)

    cam_dense_tdnn_layer[B, 128, GROWTH, BN_C, HALF_BN, T, KERNEL, DILATION, SEG_LEN](
        ctx, x_buf, out0, w0, d0)
    cam_dense_tdnn_layer[B, 160, GROWTH, BN_C, HALF_BN, T, KERNEL, DILATION, SEG_LEN](
        ctx, out0, out1, w1, d1)
    cam_dense_tdnn_layer[B, 192, GROWTH, BN_C, HALF_BN, T, KERNEL, DILATION, SEG_LEN](
        ctx, out1, out2, w2, d2)
    cam_dense_tdnn_layer[B, 224, GROWTH, BN_C, HALF_BN, T, KERNEL, DILATION, SEG_LEN](
        ctx, out2, out3, w3, d3)
    cam_dense_tdnn_layer[B, 256, GROWTH, BN_C, HALF_BN, T, KERNEL, DILATION, SEG_LEN](
        ctx, out3, out4, w4, d4)
    cam_dense_tdnn_layer[B, 288, GROWTH, BN_C, HALF_BN, T, KERNEL, DILATION, SEG_LEN](
        ctx, out4, out5, w5, d5)
    cam_dense_tdnn_layer[B, 320, GROWTH, BN_C, HALF_BN, T, KERNEL, DILATION, SEG_LEN](
        ctx, out5, out6, w6, d6)
    cam_dense_tdnn_layer[B, 352, GROWTH, BN_C, HALF_BN, T, KERNEL, DILATION, SEG_LEN](
        ctx, out6, out7, w7, d7)
    cam_dense_tdnn_layer[B, 384, GROWTH, BN_C, HALF_BN, T, KERNEL, DILATION, SEG_LEN](
        ctx, out7, out8, w8, d8)
    cam_dense_tdnn_layer[B, 416, GROWTH, BN_C, HALF_BN, T, KERNEL, DILATION, SEG_LEN](
        ctx, out8, out9, w9, d9)
    cam_dense_tdnn_layer[B, 448, GROWTH, BN_C, HALF_BN, T, KERNEL, DILATION, SEG_LEN](
        ctx, out9, out10, w10, d10)
    cam_dense_tdnn_layer[B, 480, GROWTH, BN_C, HALF_BN, T, KERNEL, DILATION, SEG_LEN](
        ctx, out10, out11, w11, d11)

    ctx.synchronize()

    var n_out = B * 512 * T
    var max_abs: Float32 = 0.0
    var sum_abs: Float64 = 0.0
    with out11.map_to_host() as h:
        for i in range(n_out):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_abs += Float64(d)
            assert_almost_equal(h[i], exp.data[i], atol=2.0e-3)
    print("CAMPPlus block1 (12 dense TDNN layers) — max abs:", max_abs,
          " mean:", sum_abs / Float64(n_out))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
