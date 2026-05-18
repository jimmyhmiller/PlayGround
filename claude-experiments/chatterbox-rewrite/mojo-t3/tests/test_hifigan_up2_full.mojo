"""
HiFiGAN upsample stage 2 in Mojo (the final upsample, i=2 in decode()).

Input:  stage_after_up1.bin   (1, 128, 1280)
Output: stage_after_up2.bin   (1, 64, 3841)

Notable: this is the last upsample, so reflection_pad((1, 0)) bumps T 3840 -> 3841
BEFORE the +si add. Then resblocks and mean operate at T=3841.

Stage config:
  ups[2]: 128 -> 64, K=7, stride=3, padding=2, T 1280 -> 3840
  reflection_pad((1, 0)): T 3840 -> 3841
  source_downs[2]: K=1, stride=1, padding=0, T 3841 -> 3841
  source_resblocks[2]: K=11, dilations=[1,3,5] at C=64
  resblocks.6 K=3, resblocks.7 K=7, resblocks.8 K=11 at C=64
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import Tensor, load_fp32
from conv import (
    conv1d_kernel, transposed_conv1d_kernel, leaky_relu_kernel, snake_kernel,
    reflection_pad_left1_kernel,
)
from util_kernels import add_kernel


comptime BATCH = 1
comptime IN_C = 128
comptime IN_T = 1280
comptime OUT_C = 64
comptime PRE_PAD_T = 3840    # T after ups[2], before reflection_pad
comptime OUT_T = 3841        # T after reflection_pad
comptime UP_K = 7
comptime UP_STRIDE = 3
comptime UP_PAD = 2
comptime S_STFT_C = 18
comptime S_STFT_T = 3841
comptime SRC_DOWN_K = 1
comptime SRC_DOWN_STRIDE = 1
comptime SRC_DOWN_PAD = 0
comptime POINTWISE_BLOCK = 256
comptime SNAKE_BLOCK = 256


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def copy_device_buf(mut ctx: DeviceContext,
                    mut src: DeviceBuffer[DType.float32],
                    mut dst: DeviceBuffer[DType.float32],
                    n: Int) raises:
    ctx.synchronize()
    with src.map_to_host() as s:
        with dst.map_to_host() as d:
            for i in range(n):
                d[i] = s[i]


def _run_resblock_chain[K: Int](
    mut ctx: DeviceContext,
    mut rb_x_buf: DeviceBuffer[DType.float32],
    mut rb_next_buf: DeviceBuffer[DType.float32],
    mut rb_xt_buf: DeviceBuffer[DType.float32],
    mut rb_xt2_buf: DeviceBuffer[DType.float32],
    weight_prefix: String,
    dil0: Int, dil1: Int, dil2: Int,
) raises:
    var C = OUT_C
    var T = OUT_T
    var n = BATCH * C * T
    var n_w = C * C * K

    comptime x_layout = row_major[BATCH, OUT_C, OUT_T]()
    comptime w_layout = row_major[OUT_C, OUT_C, K]()
    comptime b_layout = row_major[OUT_C]()
    comptime alpha_layout = row_major[OUT_C]()
    comptime flat_layout = row_major[1, BATCH * OUT_C * OUT_T]()

    var w_buf = ctx.enqueue_create_buffer[DType.float32](n_w)
    var b_buf = ctx.enqueue_create_buffer[DType.float32](C)
    var alpha_buf = ctx.enqueue_create_buffer[DType.float32](C)

    var rb_x_t = TileTensor(rb_x_buf, x_layout)
    var rb_x_flat = TileTensor(rb_x_buf, flat_layout)
    var rb_next_flat = TileTensor(rb_next_buf, flat_layout)
    var rb_xt_t = TileTensor(rb_xt_buf, x_layout)
    var rb_xt2_t = TileTensor(rb_xt2_buf, x_layout)
    var rb_xt2_flat = TileTensor(rb_xt2_buf, flat_layout)
    var w_t = TileTensor(w_buf, w_layout)
    var b_t = TileTensor(b_buf, b_layout)
    var alpha_t = TileTensor(alpha_buf, alpha_layout)

    comptime conv_k = conv1d_kernel[
        DType.float32, type_of(x_layout), type_of(w_layout),
        type_of(b_layout), type_of(x_layout), K, True,
    ]
    comptime snake_k = snake_kernel[
        DType.float32, type_of(x_layout), type_of(alpha_layout), type_of(x_layout),
        SNAKE_BLOCK,
    ]
    comptime add_k = add_kernel[
        DType.float32, type_of(flat_layout), type_of(flat_layout),
        type_of(flat_layout), POINTWISE_BLOCK,
    ]

    var dils = List[Int]()
    dils.append(dil0); dils.append(dil1); dils.append(dil2)

    for j in range(3):
        var dil = dils[j]
        var pad1 = ((K - 1) * dil) // 2
        var pad2 = ((K - 1) * 1) // 2

        var w1 = load_fp32(weight_prefix + "convs1__" + String(j) + "__weight.bin")
        var b1 = load_fp32(weight_prefix + "convs1__" + String(j) + "__bias.bin")
        var w2 = load_fp32(weight_prefix + "convs2__" + String(j) + "__weight.bin")
        var b2 = load_fp32(weight_prefix + "convs2__" + String(j) + "__bias.bin")
        var a1 = load_fp32(weight_prefix + "activations1__" + String(j) + "__alpha.bin")
        var a2 = load_fp32(weight_prefix + "activations2__" + String(j) + "__alpha.bin")

        upload(alpha_buf, a1.data, C)
        ctx.enqueue_function[snake_k, snake_k](
            rb_xt_t, rb_x_t, alpha_t, BATCH, C, T,
            grid_dim=BATCH * C, block_dim=SNAKE_BLOCK,
        )
        upload(w_buf, w1.data, n_w)
        upload(b_buf, b1.data, C)
        ctx.enqueue_function[conv_k, conv_k](
            rb_xt2_t, rb_xt_t, w_t, b_t,
            BATCH, C, C, T, T, 1, pad1, dil,
            grid_dim=BATCH * C * T, block_dim=1,
        )
        upload(alpha_buf, a2.data, C)
        ctx.enqueue_function[snake_k, snake_k](
            rb_xt_t, rb_xt2_t, alpha_t, BATCH, C, T,
            grid_dim=BATCH * C, block_dim=SNAKE_BLOCK,
        )
        upload(w_buf, w2.data, n_w)
        upload(b_buf, b2.data, C)
        ctx.enqueue_function[conv_k, conv_k](
            rb_xt2_t, rb_xt_t, w_t, b_t,
            BATCH, C, C, T, T, 1, pad2, 1,
            grid_dim=BATCH * C * T, block_dim=1,
        )
        ctx.enqueue_function[add_k, add_k](
            rb_next_flat, rb_x_flat, rb_xt2_flat, n,
            grid_dim=ceildiv(n, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
        )
        ctx.synchronize()
        with rb_next_buf.map_to_host() as src:
            with rb_x_buf.map_to_host() as dst:
                for k in range(n):
                    dst[k] = src[k]


def test_up2_full_fp32() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/hifigan/"
    var x_in = load_fp32(fix + "stage_after_up1.bin")
    var s_stft = load_fp32(fix + "stage_s_stft_cat.bin")
    var exp = load_fp32(fix + "stage_after_up2.bin")

    var n_x_in = BATCH * IN_C * IN_T
    var n_pre_pad = BATCH * OUT_C * PRE_PAD_T   # output of ups[2] before reflection_pad
    var n_x_out = BATCH * OUT_C * OUT_T
    var n_s_stft = BATCH * S_STFT_C * S_STFT_T

    var ctx = DeviceContext()
    var x_in_buf = ctx.enqueue_create_buffer[DType.float32](n_x_in)
    var x_lrelu_buf = ctx.enqueue_create_buffer[DType.float32](n_x_in)
    var x_up_buf = ctx.enqueue_create_buffer[DType.float32](n_pre_pad)
    var x_padded_buf = ctx.enqueue_create_buffer[DType.float32](n_x_out)
    var s_stft_buf = ctx.enqueue_create_buffer[DType.float32](n_s_stft)
    var si_after_down_buf = ctx.enqueue_create_buffer[DType.float32](n_x_out)
    var si_buf = ctx.enqueue_create_buffer[DType.float32](n_x_out)
    var x_plus_si_buf = ctx.enqueue_create_buffer[DType.float32](n_x_out)

    var rb_x_buf = ctx.enqueue_create_buffer[DType.float32](n_x_out)
    var rb_next_buf = ctx.enqueue_create_buffer[DType.float32](n_x_out)
    var rb_xt_buf = ctx.enqueue_create_buffer[DType.float32](n_x_out)
    var rb_xt2_buf = ctx.enqueue_create_buffer[DType.float32](n_x_out)
    var rb_out_acc_buf = ctx.enqueue_create_buffer[DType.float32](n_x_out)
    var rb_out_acc2_buf = ctx.enqueue_create_buffer[DType.float32](n_x_out)

    upload(x_in_buf, x_in.data, n_x_in)
    upload(s_stft_buf, s_stft.data, n_s_stft)

    comptime x_in_layout = row_major[BATCH, IN_C, IN_T]()
    comptime x_in_lrelu_layout = row_major[BATCH * IN_C * IN_T]()
    comptime pre_pad_layout = row_major[BATCH, OUT_C, PRE_PAD_T]()
    comptime x_out_layout = row_major[BATCH, OUT_C, OUT_T]()
    comptime x_out_flat_layout = row_major[1, BATCH * OUT_C * OUT_T]()
    comptime s_stft_layout = row_major[BATCH, S_STFT_C, S_STFT_T]()
    comptime up_w_layout = row_major[IN_C, OUT_C, UP_K]()
    comptime src_down_w_layout = row_major[OUT_C, S_STFT_C, SRC_DOWN_K]()
    comptime bias_layout = row_major[OUT_C]()

    var x_in_t = TileTensor(x_in_buf, x_in_layout)
    var x_in_lrelu = TileTensor(x_in_buf, x_in_lrelu_layout)
    var x_lrelu_t = TileTensor(x_lrelu_buf, x_in_layout)
    var x_lrelu_lrelu = TileTensor(x_lrelu_buf, x_in_lrelu_layout)
    var x_up_pre_pad_t = TileTensor(x_up_buf, pre_pad_layout)
    var x_padded_t = TileTensor(x_padded_buf, x_out_layout)
    var x_padded_flat = TileTensor(x_padded_buf, x_out_flat_layout)
    var s_stft_t = TileTensor(s_stft_buf, s_stft_layout)
    var si_after_down_t = TileTensor(si_after_down_buf, x_out_layout)
    var si_flat = TileTensor(si_buf, x_out_flat_layout)
    var x_plus_si_flat = TileTensor(x_plus_si_buf, x_out_flat_layout)
    var rb_x_flat = TileTensor(rb_x_buf, x_out_flat_layout)
    var rb_out_acc_flat = TileTensor(rb_out_acc_buf, x_out_flat_layout)
    var rb_out_acc2_flat = TileTensor(rb_out_acc2_buf, x_out_flat_layout)

    comptime lrelu_k = leaky_relu_kernel[
        DType.float32, type_of(x_in_lrelu_layout), type_of(x_in_lrelu_layout),
        POINTWISE_BLOCK,
    ]
    comptime up_k = transposed_conv1d_kernel[
        DType.float32, type_of(x_in_layout), type_of(up_w_layout),
        type_of(bias_layout), type_of(pre_pad_layout), UP_K, True,
    ]
    comptime refpad_k = reflection_pad_left1_kernel[
        DType.float32, type_of(pre_pad_layout), type_of(x_out_layout),
    ]
    comptime src_down_k = conv1d_kernel[
        DType.float32, type_of(s_stft_layout), type_of(src_down_w_layout),
        type_of(bias_layout), type_of(x_out_layout), SRC_DOWN_K, True,
    ]
    comptime add_out_k = add_kernel[
        DType.float32, type_of(x_out_flat_layout), type_of(x_out_flat_layout),
        type_of(x_out_flat_layout), POINTWISE_BLOCK,
    ]

    # Weights.
    var up_w_buf = ctx.enqueue_create_buffer[DType.float32](IN_C * OUT_C * UP_K)
    var up_b_buf = ctx.enqueue_create_buffer[DType.float32](OUT_C)
    var src_down_w_buf = ctx.enqueue_create_buffer[DType.float32](OUT_C * S_STFT_C * SRC_DOWN_K)
    var src_down_b_buf = ctx.enqueue_create_buffer[DType.float32](OUT_C)
    var up_w = load_fp32(fix + "weights/ups__2__weight.bin")
    var up_b = load_fp32(fix + "weights/ups__2__bias.bin")
    var src_down_w = load_fp32(fix + "weights/source_downs__2__weight.bin")
    var src_down_b = load_fp32(fix + "weights/source_downs__2__bias.bin")
    upload(up_w_buf, up_w.data, IN_C * OUT_C * UP_K)
    upload(up_b_buf, up_b.data, OUT_C)
    upload(src_down_w_buf, src_down_w.data, OUT_C * S_STFT_C * SRC_DOWN_K)
    upload(src_down_b_buf, src_down_b.data, OUT_C)
    var up_w_t = TileTensor(up_w_buf, up_w_layout)
    var up_b_t = TileTensor(up_b_buf, bias_layout)
    var src_down_w_t = TileTensor(src_down_w_buf, src_down_w_layout)
    var src_down_b_t = TileTensor(src_down_b_buf, bias_layout)

    # 1. lrelu.
    ctx.enqueue_function[lrelu_k, lrelu_k](
        x_lrelu_lrelu, x_in_lrelu, n_x_in, Float32(0.1),
        grid_dim=ceildiv(n_x_in, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )
    # 2. ups[2]: 1280 -> 3840.
    ctx.enqueue_function[up_k, up_k](
        x_up_pre_pad_t, x_lrelu_t, up_w_t, up_b_t,
        BATCH, IN_C, OUT_C, IN_T, PRE_PAD_T, UP_STRIDE, UP_PAD, 1,
        grid_dim=BATCH * OUT_C * PRE_PAD_T, block_dim=1,
    )
    # 3. reflection_pad((1, 0)): 3840 -> 3841.
    ctx.enqueue_function[refpad_k, refpad_k](
        x_padded_t, x_up_pre_pad_t, BATCH, OUT_C, PRE_PAD_T,
        grid_dim=BATCH * OUT_C * OUT_T, block_dim=1,
    )
    # 4. source_downs[2]: K=1, stride=1, padding=0.
    ctx.enqueue_function[src_down_k, src_down_k](
        si_after_down_t, s_stft_t, src_down_w_t, src_down_b_t,
        BATCH, S_STFT_C, OUT_C, S_STFT_T, OUT_T, SRC_DOWN_STRIDE, SRC_DOWN_PAD, 1,
        grid_dim=BATCH * OUT_C * OUT_T, block_dim=1,
    )
    # 5. source_resblocks[2]: K=11, dils=[1,3,5].
    copy_device_buf(ctx, si_after_down_buf, rb_x_buf, n_x_out)
    _run_resblock_chain[11](ctx, rb_x_buf, rb_next_buf, rb_xt_buf, rb_xt2_buf,
                             fix + "weights/source_resblocks__2__", 1, 3, 5)
    copy_device_buf(ctx, rb_x_buf, si_buf, n_x_out)

    # 6. x_plus_si = x_padded + si.
    ctx.enqueue_function[add_out_k, add_out_k](
        x_plus_si_flat, x_padded_flat, si_flat, n_x_out,
        grid_dim=ceildiv(n_x_out, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )

    # 7. resblocks[6,7,8] (K=3, K=7, K=11) at C=64.
    copy_device_buf(ctx, x_plus_si_buf, rb_x_buf, n_x_out)
    _run_resblock_chain[3](ctx, rb_x_buf, rb_next_buf, rb_xt_buf, rb_xt2_buf,
                            fix + "weights/resblocks__6__", 1, 3, 5)
    copy_device_buf(ctx, rb_x_buf, rb_out_acc_buf, n_x_out)

    copy_device_buf(ctx, x_plus_si_buf, rb_x_buf, n_x_out)
    _run_resblock_chain[7](ctx, rb_x_buf, rb_next_buf, rb_xt_buf, rb_xt2_buf,
                            fix + "weights/resblocks__7__", 1, 3, 5)
    ctx.enqueue_function[add_out_k, add_out_k](
        rb_out_acc2_flat, rb_out_acc_flat, rb_x_flat, n_x_out,
        grid_dim=ceildiv(n_x_out, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )
    copy_device_buf(ctx, rb_out_acc2_buf, rb_out_acc_buf, n_x_out)

    copy_device_buf(ctx, x_plus_si_buf, rb_x_buf, n_x_out)
    _run_resblock_chain[11](ctx, rb_x_buf, rb_next_buf, rb_xt_buf, rb_xt2_buf,
                             fix + "weights/resblocks__8__", 1, 3, 5)
    ctx.enqueue_function[add_out_k, add_out_k](
        rb_out_acc2_flat, rb_out_acc_flat, rb_x_flat, n_x_out,
        grid_dim=ceildiv(n_x_out, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )
    copy_device_buf(ctx, rb_out_acc2_buf, rb_out_acc_buf, n_x_out)

    # 8. x = sum / 3, compare to stage_after_up2.
    ctx.synchronize()
    var max_abs: Float32 = 0.0
    var sum_abs: Float64 = 0.0
    var inv = Float32(1.0 / 3.0)
    with rb_out_acc_buf.map_to_host() as h:
        for i in range(n_x_out):
            var v = h[i] * inv
            var d = v - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_abs += Float64(d)
            assert_almost_equal(v, exp.data[i], atol=5.0e-3)
    print("HiFiGAN up2 full stage fp32 — max abs:", max_abs,
          " mean abs:", sum_abs / Float64(n_x_out))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
