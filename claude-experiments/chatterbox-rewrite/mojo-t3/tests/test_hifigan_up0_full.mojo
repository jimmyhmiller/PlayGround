"""
Full HiFiGAN upsample stage 0 in Mojo.

Mirrors the body of decode()'s for-loop at i=0:
  x = lrelu(x, 0.1)
  x = ups[0](x)
  si = source_downs[0](s_stft_cat)
  si = source_resblocks[0](si)
  x = x + si
  xs = sum_j(resblocks[i*num_kernels + j](x))
  x = xs / num_kernels

Input:  stage_after_conv_pre.bin   (1, 512, 32)
        stage_s_stft_cat.bin       (1, 18, 3841)   STFT(s=zeros)
Output: stage_after_up0.bin        (1, 256, 256)
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import Tensor, load_fp32
from conv import (
    conv1d_kernel, transposed_conv1d_kernel, leaky_relu_kernel, snake_kernel,
)
from util_kernels import add_kernel


comptime BATCH = 1
comptime IN_C = 512
comptime IN_T = 32
comptime OUT_C = 256
comptime OUT_T = 256
comptime UP_K = 16
comptime UP_STRIDE = 8
comptime UP_PAD = 4
comptime S_STFT_C = 18
comptime S_STFT_T = 3841
comptime SRC_DOWN_K = 30
comptime SRC_DOWN_STRIDE = 15
comptime SRC_DOWN_PAD = 7

# Residual blocks at this stage:
#   resblocks.0: K=3, dilations=[1, 3, 5]
#   resblocks.1: K=7, dilations=[1, 3, 5]
#   resblocks.2: K=11, dilations=[1, 3, 5]
#   source_resblocks.0: K=7, dilations=[1, 3, 5]
comptime NUM_KERNELS = 3
comptime POINTWISE_BLOCK = 256


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def copy_device_buf(ctx: DeviceContext, src: DeviceBuffer[DType.float32],
                    dst: DeviceBuffer[DType.float32], n: Int) raises:
    ctx.synchronize()
    with src.map_to_host() as s:
        with dst.map_to_host() as d:
            for i in range(n):
                d[i] = s[i]


def test_up0_full_fp32() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/hifigan/"
    var x_pre = load_fp32(fix + "stage_after_conv_pre.bin")
    var s_stft = load_fp32(fix + "stage_s_stft_cat.bin")
    var exp = load_fp32(fix + "stage_after_up0.bin")

    var n_x_pre = BATCH * IN_C * IN_T
    var n_x_out = BATCH * OUT_C * OUT_T
    var n_s_stft = BATCH * S_STFT_C * S_STFT_T

    var ctx = DeviceContext()

    # ---- Persistent buffers ----
    var x_in_buf = ctx.enqueue_create_buffer[DType.float32](n_x_pre)
    var x_lrelu_buf = ctx.enqueue_create_buffer[DType.float32](n_x_pre)
    var x_up_buf = ctx.enqueue_create_buffer[DType.float32](n_x_out)        # post-ups[0]
    var s_stft_buf = ctx.enqueue_create_buffer[DType.float32](n_s_stft)
    var si_after_down_buf = ctx.enqueue_create_buffer[DType.float32](n_x_out)
    var si_buf = ctx.enqueue_create_buffer[DType.float32](n_x_out)          # post-source_resblock
    var x_plus_si_buf = ctx.enqueue_create_buffer[DType.float32](n_x_out)

    # Residual-stream scratch for ResBlock chains:
    var rb_x_buf = ctx.enqueue_create_buffer[DType.float32](n_x_out)
    var rb_next_buf = ctx.enqueue_create_buffer[DType.float32](n_x_out)
    var rb_xt_buf = ctx.enqueue_create_buffer[DType.float32](n_x_out)
    var rb_xt2_buf = ctx.enqueue_create_buffer[DType.float32](n_x_out)

    var rb_out_acc_buf = ctx.enqueue_create_buffer[DType.float32](n_x_out)
    var rb_out_acc2_buf = ctx.enqueue_create_buffer[DType.float32](n_x_out)

    upload(x_in_buf, x_pre.data, n_x_pre)
    upload(s_stft_buf, s_stft.data, n_s_stft)

    # ---- Layouts ----
    comptime x_in_layout = row_major[BATCH, IN_C, IN_T]()
    comptime x_in_lrelu_layout = row_major[BATCH * IN_C * IN_T]()        # rank-1 (for lrelu)
    comptime x_in_flat_layout = row_major[1, BATCH * IN_C * IN_T]()      # rank-2 (for add)
    comptime x_out_layout = row_major[BATCH, OUT_C, OUT_T]()
    comptime x_out_flat_layout = row_major[1, BATCH * OUT_C * OUT_T]()    # rank-2 (for add)
    comptime s_stft_layout = row_major[BATCH, S_STFT_C, S_STFT_T]()

    comptime up_w_layout = row_major[IN_C, OUT_C, UP_K]()
    comptime src_down_w_layout = row_major[OUT_C, S_STFT_C, SRC_DOWN_K]()
    comptime bias_layout = row_major[OUT_C]()

    # ---- Views ----
    var x_in_t = TileTensor(x_in_buf, x_in_layout)
    var x_in_lrelu = TileTensor(x_in_buf, x_in_lrelu_layout)
    var x_lrelu_t = TileTensor(x_lrelu_buf, x_in_layout)
    var x_lrelu_lrelu = TileTensor(x_lrelu_buf, x_in_lrelu_layout)
    var x_up_t = TileTensor(x_up_buf, x_out_layout)
    var x_up_flat = TileTensor(x_up_buf, x_out_flat_layout)
    var s_stft_t = TileTensor(s_stft_buf, s_stft_layout)
    var si_after_down_t = TileTensor(si_after_down_buf, x_out_layout)
    var si_t = TileTensor(si_buf, x_out_layout)
    var x_plus_si_t = TileTensor(x_plus_si_buf, x_out_layout)
    var x_plus_si_flat = TileTensor(x_plus_si_buf, x_out_flat_layout)
    var rb_x_t = TileTensor(rb_x_buf, x_out_layout)
    var rb_x_flat = TileTensor(rb_x_buf, x_out_flat_layout)
    var rb_next_t = TileTensor(rb_next_buf, x_out_layout)
    var rb_next_flat = TileTensor(rb_next_buf, x_out_flat_layout)
    var rb_xt_t = TileTensor(rb_xt_buf, x_out_layout)
    var rb_xt2_t = TileTensor(rb_xt2_buf, x_out_layout)
    var rb_xt2_flat = TileTensor(rb_xt2_buf, x_out_flat_layout)
    var rb_out_acc_t = TileTensor(rb_out_acc_buf, x_out_layout)
    var rb_out_acc_flat = TileTensor(rb_out_acc_buf, x_out_flat_layout)
    var rb_out_acc2_flat = TileTensor(rb_out_acc2_buf, x_out_flat_layout)

    # ---- Kernel bindings ----
    comptime lrelu_in_k = leaky_relu_kernel[
        DType.float32, type_of(x_in_lrelu_layout), type_of(x_in_lrelu_layout),
        POINTWISE_BLOCK,
    ]
    comptime up_k = transposed_conv1d_kernel[
        DType.float32, type_of(x_in_layout), type_of(up_w_layout),
        type_of(bias_layout), type_of(x_out_layout), UP_K, True,
    ]
    comptime src_down_k = conv1d_kernel[
        DType.float32, type_of(s_stft_layout), type_of(src_down_w_layout),
        type_of(bias_layout), type_of(x_out_layout), SRC_DOWN_K, True,
    ]
    comptime add_out_k = add_kernel[
        DType.float32, type_of(x_out_flat_layout), type_of(x_out_flat_layout),
        type_of(x_out_flat_layout), POINTWISE_BLOCK,
    ]

    # ---- Weight buffers (reused across resblocks) ----
    var up_w_buf = ctx.enqueue_create_buffer[DType.float32](IN_C * OUT_C * UP_K)
    var up_b_buf = ctx.enqueue_create_buffer[DType.float32](OUT_C)
    var src_down_w_buf = ctx.enqueue_create_buffer[DType.float32](OUT_C * S_STFT_C * SRC_DOWN_K)
    var src_down_b_buf = ctx.enqueue_create_buffer[DType.float32](OUT_C)

    var up_w = load_fp32(fix + "weights/ups__0__weight.bin")
    var up_b = load_fp32(fix + "weights/ups__0__bias.bin")
    var src_down_w = load_fp32(fix + "weights/source_downs__0__weight.bin")
    var src_down_b = load_fp32(fix + "weights/source_downs__0__bias.bin")
    upload(up_w_buf, up_w.data, IN_C * OUT_C * UP_K)
    upload(up_b_buf, up_b.data, OUT_C)
    upload(src_down_w_buf, src_down_w.data, OUT_C * S_STFT_C * SRC_DOWN_K)
    upload(src_down_b_buf, src_down_b.data, OUT_C)
    var up_w_t = TileTensor(up_w_buf, up_w_layout)
    var up_b_t = TileTensor(up_b_buf, bias_layout)
    var src_down_w_t = TileTensor(src_down_w_buf, src_down_w_layout)
    var src_down_b_t = TileTensor(src_down_b_buf, bias_layout)

    # ============================================================
    # STEP 1: x = lrelu(x_pre, 0.1)
    # ============================================================
    ctx.enqueue_function[lrelu_in_k, lrelu_in_k](
        x_lrelu_lrelu, x_in_lrelu, n_x_pre, Float32(0.1),
        grid_dim=ceildiv(n_x_pre, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )

    # ============================================================
    # STEP 2: x = ups[0](x)
    # ============================================================
    ctx.enqueue_function[up_k, up_k](
        x_up_t, x_lrelu_t, up_w_t, up_b_t,
        BATCH, IN_C, OUT_C, IN_T, OUT_T, UP_STRIDE, UP_PAD, 1,
        grid_dim=BATCH * OUT_C * OUT_T, block_dim=1,
    )

    # ============================================================
    # STEP 3: si = source_downs[0](s_stft_cat) — conv1d (18 → 256), K=30, stride=15
    # Output time dim: (3841 + 14 - 29 - 1)//15 + 1 = 256 ✓
    # ============================================================
    ctx.enqueue_function[src_down_k, src_down_k](
        si_after_down_t, s_stft_t, src_down_w_t, src_down_b_t,
        BATCH, S_STFT_C, OUT_C, S_STFT_T, OUT_T, SRC_DOWN_STRIDE, SRC_DOWN_PAD, 1,
        grid_dim=BATCH * OUT_C * OUT_T, block_dim=1,
    )

    # ============================================================
    # STEP 4: si = source_resblocks[0](si)  — K=7, dils=[1,3,5], C=256
    # Implement as: copy si_after_down into rb_x, run 3-dilation ResBlock, leave result in rb_x.
    # Then copy rb_x into si_buf.
    # ============================================================
    copy_device_buf(ctx, si_after_down_buf, rb_x_buf, n_x_out)

    _run_resblock_chain[7](ctx, rb_x_buf, rb_next_buf, rb_xt_buf, rb_xt2_buf,
                           fix + "weights/source_resblocks__0__",
                           1, 3, 5)
    copy_device_buf(ctx, rb_x_buf, si_buf, n_x_out)

    # ============================================================
    # STEP 5: x = x_up + si
    # ============================================================
    # add_kernel requires non-aliased inputs/outputs; x_up_buf is read while
    # x_plus_si_buf is written so we're safe.
    var x_up_flat_a = TileTensor(x_up_buf, x_out_flat_layout)
    var si_flat = TileTensor(si_buf, x_out_flat_layout)
    ctx.enqueue_function[add_out_k, add_out_k](
        x_plus_si_flat, x_up_flat_a, si_flat, n_x_out,
        grid_dim=ceildiv(n_x_out, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )

    # ============================================================
    # STEP 6: xs = resblocks[0](x) + resblocks[1](x) + resblocks[2](x)
    # ============================================================

    # j = 0: resblocks.0 K=3 → rb_out_acc holds first resblock output.
    copy_device_buf(ctx, x_plus_si_buf, rb_x_buf, n_x_out)
    _run_resblock_chain[3](ctx, rb_x_buf, rb_next_buf, rb_xt_buf, rb_xt2_buf,
                           fix + "weights/resblocks__0__", 1, 3, 5)
    copy_device_buf(ctx, rb_x_buf, rb_out_acc_buf, n_x_out)

    # j = 1: resblocks.1 K=7; sum into rb_out_acc2 = rb_out_acc + rb_x; swap.
    copy_device_buf(ctx, x_plus_si_buf, rb_x_buf, n_x_out)
    _run_resblock_chain[7](ctx, rb_x_buf, rb_next_buf, rb_xt_buf, rb_xt2_buf,
                           fix + "weights/resblocks__1__", 1, 3, 5)
    ctx.enqueue_function[add_out_k, add_out_k](
        rb_out_acc2_flat, rb_out_acc_flat, rb_x_flat, n_x_out,
        grid_dim=ceildiv(n_x_out, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )
    copy_device_buf(ctx, rb_out_acc2_buf, rb_out_acc_buf, n_x_out)

    # j = 2: resblocks.2 K=11; rb_out_acc2 = rb_out_acc + rb_x; copy.
    copy_device_buf(ctx, x_plus_si_buf, rb_x_buf, n_x_out)
    _run_resblock_chain[11](ctx, rb_x_buf, rb_next_buf, rb_xt_buf, rb_xt2_buf,
                            fix + "weights/resblocks__2__", 1, 3, 5)
    ctx.enqueue_function[add_out_k, add_out_k](
        rb_out_acc2_flat, rb_out_acc_flat, rb_x_flat, n_x_out,
        grid_dim=ceildiv(n_x_out, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )
    copy_device_buf(ctx, rb_out_acc2_buf, rb_out_acc_buf, n_x_out)

    # ============================================================
    # STEP 7: x = xs / 3   (final output of this upsample stage)
    # ============================================================
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
            assert_almost_equal(v, exp.data[i], atol=2.0e-3)
    print("HiFiGAN up0 full stage fp32 — max abs:", max_abs,
          " mean abs:", sum_abs / Float64(n_x_out))


# ---- Helper: run a K-aware ResBlock chain (3 dilations, in-place residual) ----
# Comptime K lets us instantiate the conv1d_kernel correctly. We pass mutable
# device buffers (not TileTensor views) so we can ping-pong with map_to_host.
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
    comptime SNAKE_BLOCK = 256
    comptime snake_k = snake_kernel[
        DType.float32, type_of(x_layout), type_of(alpha_layout), type_of(x_layout),
        SNAKE_BLOCK,
    ]
    comptime add_k = add_kernel[
        DType.float32, type_of(flat_layout), type_of(flat_layout),
        type_of(flat_layout), POINTWISE_BLOCK,
    ]

    var dilations = List[Int]()
    dilations.append(dil0)
    dilations.append(dil1)
    dilations.append(dil2)

    for j in range(3):
        var dil = dilations[j]
        var pad1 = ((K - 1) * dil) // 2
        var pad2 = ((K - 1) * 1) // 2

        var w1 = load_fp32(weight_prefix + "convs1__" + String(j) + "__weight.bin")
        var b1 = load_fp32(weight_prefix + "convs1__" + String(j) + "__bias.bin")
        var w2 = load_fp32(weight_prefix + "convs2__" + String(j) + "__weight.bin")
        var b2 = load_fp32(weight_prefix + "convs2__" + String(j) + "__bias.bin")
        var a1 = load_fp32(weight_prefix + "activations1__" + String(j) + "__alpha.bin")
        var a2 = load_fp32(weight_prefix + "activations2__" + String(j) + "__alpha.bin")

        # xt = snake(rb_x, a1)
        upload(alpha_buf, a1.data, C)
        ctx.enqueue_function[snake_k, snake_k](
            rb_xt_t, rb_x_t, alpha_t, BATCH, C, T,
            grid_dim=BATCH * C, block_dim=SNAKE_BLOCK,
        )
        # xt2 = conv1d(xt, w1, b1, dilation=dil, padding=pad1)
        upload(w_buf, w1.data, n_w)
        upload(b_buf, b1.data, C)
        ctx.enqueue_function[conv_k, conv_k](
            rb_xt2_t, rb_xt_t, w_t, b_t,
            BATCH, C, C, T, T, 1, pad1, dil,
            grid_dim=BATCH * C * T, block_dim=1,
        )
        # xt = snake(xt2, a2)
        upload(alpha_buf, a2.data, C)
        ctx.enqueue_function[snake_k, snake_k](
            rb_xt_t, rb_xt2_t, alpha_t, BATCH, C, T,
            grid_dim=BATCH * C, block_dim=SNAKE_BLOCK,
        )
        # xt2 = conv1d(xt, w2, b2, dilation=1, padding=pad2)
        upload(w_buf, w2.data, n_w)
        upload(b_buf, b2.data, C)
        ctx.enqueue_function[conv_k, conv_k](
            rb_xt2_t, rb_xt_t, w_t, b_t,
            BATCH, C, C, T, T, 1, pad2, 1,
            grid_dim=BATCH * C * T, block_dim=1,
        )
        # rb_next = rb_x + rb_xt2
        ctx.enqueue_function[add_k, add_k](
            rb_next_flat, rb_x_flat, rb_xt2_flat, n,
            grid_dim=ceildiv(n, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
        )
        # Copy rb_next → rb_x for the next dilation iteration.
        ctx.synchronize()
        with rb_next_buf.map_to_host() as src:
            with rb_x_buf.map_to_host() as dst:
                for k in range(n):
                    dst[k] = src[k]


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
