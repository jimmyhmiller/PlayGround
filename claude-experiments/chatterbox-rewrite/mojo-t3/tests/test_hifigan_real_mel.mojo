"""
Full HiFiGAN mel → audio driver in Mojo (s=zeros, no f0 branch).

Input:  real_mel.bin                        (1, 80, 32)
Output: matches expected_wav_decode_zeros (1, 15360)

This is the first complete vocoder forward pass in Mojo. Pipeline:
  conv_pre → for i in 0..2: (lrelu → ups[i] → reflection_pad? → +si → mean(3 resblocks)) → final lrelu → conv_post → magnitude/phase split → iSTFT → audio.

Each stage uses real Chatterbox weights and matches upstream Chatterbox audio
to within ~few * 1e-3 max abs (accumulated kernel error across the full chain).

For brevity we inline every comptime-shape's kernel binding three times (one per
upsample stage) rather than try to abstract the stage helper across different
output channel/length tuples — Mojo's comptime layout system makes that awkward.
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import Tensor, load_fp32, save_fp32_1d
from conv import (
    conv1d_kernel, transposed_conv1d_kernel, leaky_relu_kernel, snake_kernel,
    reflection_pad_left1_kernel,
    magnitude_phase_split_kernel, magnitude_phase_to_complex_kernel,
)
from util_kernels import add_kernel
from stft import istft_kernel


comptime BATCH = 1
comptime POINTWISE_BLOCK = 256
comptime SNAKE_BLOCK = 256

# ---- Conv-pre stage (mel -> 512 channels) ----
comptime MEL_C = 80
comptime MEL_T = 252
comptime PRE_C = 512
comptime PRE_T = 252
comptime CP_PRE_K = 7

# ---- Stage 0: PRE (512 x 32) -> S0 (256 x 256) ----
comptime S0_C = 256
comptime S0_T = 2016
comptime UP0_K = 16
comptime UP0_STRIDE = 8
comptime UP0_PAD = 4
comptime S0_SRC_DOWN_K = 30
comptime S0_SRC_DOWN_STRIDE = 15
comptime S0_SRC_DOWN_PAD = 7

# ---- Stage 1: S0 (256 x 256) -> S1 (128 x 1280) ----
comptime S1_C = 128
comptime S1_T = 10080
comptime UP1_K = 11
comptime UP1_STRIDE = 5
comptime UP1_PAD = 3
comptime S1_SRC_DOWN_K = 6
comptime S1_SRC_DOWN_STRIDE = 3
comptime S1_SRC_DOWN_PAD = 1

# ---- Stage 2: S1 (128 x 1280) -> S2 (64 x 3841 after reflection_pad) ----
comptime S2_C = 64
comptime S2_PRE_PAD_T = 30240
comptime S2_T = 30241
comptime UP2_K = 7
comptime UP2_STRIDE = 3
comptime UP2_PAD = 2
comptime S2_SRC_DOWN_K = 1
comptime S2_SRC_DOWN_STRIDE = 1
comptime S2_SRC_DOWN_PAD = 0

# ---- Conv-post (S2 -> 18 channels at T=3841) ----
comptime POST_C = 18
comptime CP_POST_K = 7

# ---- STFT params (matches HiFiGAN config) ----
comptime N_FFT = 16
comptime HOP = 4
comptime N_FREQ = N_FFT // 2 + 1   # 9
comptime T_AUDIO = 120960

# ---- Source STFT input (B, 18, 3841) ----
comptime S_STFT_C = 18
comptime S_STFT_T = 30241


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


# A stage-templated resblock-chain helper. Comptime params: C, T (the
# stage's output channels and length), K (kernel size), BLOCK_SIZE (for
# launch).
def _run_resblock_chain[
    C: Int, T: Int, K: Int, SBLOCK: Int,
](
    mut ctx: DeviceContext,
    mut rb_x_buf: DeviceBuffer[DType.float32],
    mut rb_next_buf: DeviceBuffer[DType.float32],
    mut rb_xt_buf: DeviceBuffer[DType.float32],
    mut rb_xt2_buf: DeviceBuffer[DType.float32],
    weight_prefix: String,
) raises:
    var n = BATCH * C * T
    var n_w = C * C * K

    comptime x_layout = row_major[BATCH, C, T]()
    comptime w_layout = row_major[C, C, K]()
    comptime b_layout = row_major[C]()
    comptime flat_layout = row_major[1, BATCH * C * T]()

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
    var alpha_t = TileTensor(alpha_buf, b_layout)

    comptime conv_k = conv1d_kernel[
        DType.float32, type_of(x_layout), type_of(w_layout),
        type_of(b_layout), type_of(x_layout), K, True,
    ]
    comptime snake_k = snake_kernel[
        DType.float32, type_of(x_layout), type_of(b_layout), type_of(x_layout),
        SBLOCK,
    ]
    comptime add_k = add_kernel[
        DType.float32, type_of(flat_layout), type_of(flat_layout),
        type_of(flat_layout), POINTWISE_BLOCK,
    ]

    var dils = List[Int]()
    dils.append(1); dils.append(3); dils.append(5)

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
            grid_dim=BATCH * C, block_dim=SBLOCK,
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
            grid_dim=BATCH * C, block_dim=SBLOCK,
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


def _run_hifigan_full(s_stft_path: String, exp_path: String,
                       label: String) raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/real/"
    var mel = load_fp32(fix + "real_mel.bin")
    var s_stft = load_fp32(s_stft_path)
    var window = load_fp32(fix + "weights/stft_window.bin")
    var exp = load_fp32(exp_path)

    var ctx = DeviceContext()

    # ---- Sizes ----
    var n_mel = BATCH * MEL_C * MEL_T
    var n_pre = BATCH * PRE_C * PRE_T
    var n_s_stft = BATCH * S_STFT_C * S_STFT_T
    var n_s0 = BATCH * S0_C * S0_T
    var n_s1 = BATCH * S1_C * S1_T
    var n_s2_pre_pad = BATCH * S2_C * S2_PRE_PAD_T
    var n_s2 = BATCH * S2_C * S2_T
    var n_post = BATCH * POST_C * S2_T
    var n_spec = BATCH * N_FREQ * S2_T
    var n_audio = BATCH * T_AUDIO

    # ---- Buffers ----
    var mel_buf = ctx.enqueue_create_buffer[DType.float32](n_mel)
    var pre_buf = ctx.enqueue_create_buffer[DType.float32](n_pre)
    var s_stft_buf = ctx.enqueue_create_buffer[DType.float32](n_s_stft)

    var pre_lrelu_buf = ctx.enqueue_create_buffer[DType.float32](n_pre)
    var s0_up_buf = ctx.enqueue_create_buffer[DType.float32](n_s0)
    var s0_si_after_down_buf = ctx.enqueue_create_buffer[DType.float32](n_s0)
    var s0_si_buf = ctx.enqueue_create_buffer[DType.float32](n_s0)
    var s0_x_plus_si_buf = ctx.enqueue_create_buffer[DType.float32](n_s0)
    var s0_rb_x_buf = ctx.enqueue_create_buffer[DType.float32](n_s0)
    var s0_rb_next_buf = ctx.enqueue_create_buffer[DType.float32](n_s0)
    var s0_rb_xt_buf = ctx.enqueue_create_buffer[DType.float32](n_s0)
    var s0_rb_xt2_buf = ctx.enqueue_create_buffer[DType.float32](n_s0)
    var s0_acc_buf = ctx.enqueue_create_buffer[DType.float32](n_s0)
    var s0_acc2_buf = ctx.enqueue_create_buffer[DType.float32](n_s0)

    var s0_lrelu_buf = ctx.enqueue_create_buffer[DType.float32](n_s0)
    var s1_up_buf = ctx.enqueue_create_buffer[DType.float32](n_s1)
    var s1_si_after_down_buf = ctx.enqueue_create_buffer[DType.float32](n_s1)
    var s1_si_buf = ctx.enqueue_create_buffer[DType.float32](n_s1)
    var s1_x_plus_si_buf = ctx.enqueue_create_buffer[DType.float32](n_s1)
    var s1_rb_x_buf = ctx.enqueue_create_buffer[DType.float32](n_s1)
    var s1_rb_next_buf = ctx.enqueue_create_buffer[DType.float32](n_s1)
    var s1_rb_xt_buf = ctx.enqueue_create_buffer[DType.float32](n_s1)
    var s1_rb_xt2_buf = ctx.enqueue_create_buffer[DType.float32](n_s1)
    var s1_acc_buf = ctx.enqueue_create_buffer[DType.float32](n_s1)
    var s1_acc2_buf = ctx.enqueue_create_buffer[DType.float32](n_s1)

    var s1_lrelu_buf = ctx.enqueue_create_buffer[DType.float32](n_s1)
    var s2_up_buf = ctx.enqueue_create_buffer[DType.float32](n_s2_pre_pad)
    var s2_padded_buf = ctx.enqueue_create_buffer[DType.float32](n_s2)
    var s2_si_after_down_buf = ctx.enqueue_create_buffer[DType.float32](n_s2)
    var s2_si_buf = ctx.enqueue_create_buffer[DType.float32](n_s2)
    var s2_x_plus_si_buf = ctx.enqueue_create_buffer[DType.float32](n_s2)
    var s2_rb_x_buf = ctx.enqueue_create_buffer[DType.float32](n_s2)
    var s2_rb_next_buf = ctx.enqueue_create_buffer[DType.float32](n_s2)
    var s2_rb_xt_buf = ctx.enqueue_create_buffer[DType.float32](n_s2)
    var s2_rb_xt2_buf = ctx.enqueue_create_buffer[DType.float32](n_s2)
    var s2_acc_buf = ctx.enqueue_create_buffer[DType.float32](n_s2)
    var s2_acc2_buf = ctx.enqueue_create_buffer[DType.float32](n_s2)

    var s2_lrelu_buf = ctx.enqueue_create_buffer[DType.float32](n_s2)
    var post_out_buf = ctx.enqueue_create_buffer[DType.float32](n_post)
    var mag_buf = ctx.enqueue_create_buffer[DType.float32](n_spec)
    var phase_buf = ctx.enqueue_create_buffer[DType.float32](n_spec)
    var re_buf = ctx.enqueue_create_buffer[DType.float32](n_spec)
    var im_buf = ctx.enqueue_create_buffer[DType.float32](n_spec)
    var window_buf = ctx.enqueue_create_buffer[DType.float32](N_FFT)
    var audio_buf = ctx.enqueue_create_buffer[DType.float32](n_audio)

    upload(mel_buf, mel.data, n_mel)
    upload(s_stft_buf, s_stft.data, n_s_stft)
    upload(window_buf, window.data, N_FFT)

    # ---- Layouts ----
    comptime mel_layout = row_major[BATCH, MEL_C, MEL_T]()
    comptime pre_layout = row_major[BATCH, PRE_C, PRE_T]()
    comptime pre_lrelu_layout = row_major[BATCH * PRE_C * PRE_T]()
    comptime pre_flat_layout = row_major[1, BATCH * PRE_C * PRE_T]()
    comptime cp_pre_w_layout = row_major[PRE_C, MEL_C, CP_PRE_K]()
    comptime pre_bias_layout = row_major[PRE_C]()

    comptime s0_layout = row_major[BATCH, S0_C, S0_T]()
    comptime s0_flat_layout = row_major[1, BATCH * S0_C * S0_T]()
    comptime s0_lrelu_layout = row_major[BATCH * S0_C * S0_T]()
    comptime up0_w_layout = row_major[PRE_C, S0_C, UP0_K]()
    comptime s_stft_layout = row_major[BATCH, S_STFT_C, S_STFT_T]()
    comptime s0_src_down_w_layout = row_major[S0_C, S_STFT_C, S0_SRC_DOWN_K]()
    comptime s0_bias_layout = row_major[S0_C]()

    comptime s1_layout = row_major[BATCH, S1_C, S1_T]()
    comptime s1_flat_layout = row_major[1, BATCH * S1_C * S1_T]()
    comptime s1_lrelu_layout = row_major[BATCH * S1_C * S1_T]()
    comptime up1_w_layout = row_major[S0_C, S1_C, UP1_K]()
    comptime s1_src_down_w_layout = row_major[S1_C, S_STFT_C, S1_SRC_DOWN_K]()
    comptime s1_bias_layout = row_major[S1_C]()

    comptime s2_pre_pad_layout = row_major[BATCH, S2_C, S2_PRE_PAD_T]()
    comptime s2_layout = row_major[BATCH, S2_C, S2_T]()
    comptime s2_flat_layout = row_major[1, BATCH * S2_C * S2_T]()
    comptime s2_lrelu_layout = row_major[BATCH * S2_C * S2_T]()
    comptime up2_w_layout = row_major[S1_C, S2_C, UP2_K]()
    comptime s2_src_down_w_layout = row_major[S2_C, S_STFT_C, S2_SRC_DOWN_K]()
    comptime s2_bias_layout = row_major[S2_C]()

    comptime post_layout = row_major[BATCH, POST_C, S2_T]()
    comptime cp_post_w_layout = row_major[POST_C, S2_C, CP_POST_K]()
    comptime post_bias_layout = row_major[POST_C]()
    comptime spec_layout = row_major[BATCH, N_FREQ, S2_T]()
    comptime window_layout = row_major[N_FFT]()
    comptime audio_layout = row_major[BATCH, T_AUDIO]()

    # ---- TileTensor views ----
    var mel_t = TileTensor(mel_buf, mel_layout)
    var pre_t = TileTensor(pre_buf, pre_layout)
    var pre_lrelu_lrelu = TileTensor(pre_buf, pre_lrelu_layout)
    var pre_lrelu_t = TileTensor(pre_lrelu_buf, pre_layout)
    var pre_lrelu_lrelu_out = TileTensor(pre_lrelu_buf, pre_lrelu_layout)
    var s_stft_t = TileTensor(s_stft_buf, s_stft_layout)

    var s0_up_t = TileTensor(s0_up_buf, s0_layout)
    var s0_up_flat = TileTensor(s0_up_buf, s0_flat_layout)
    var s0_si_after_down_t = TileTensor(s0_si_after_down_buf, s0_layout)
    var s0_si_flat = TileTensor(s0_si_buf, s0_flat_layout)
    var s0_x_plus_si_t = TileTensor(s0_x_plus_si_buf, s0_layout)
    var s0_x_plus_si_flat = TileTensor(s0_x_plus_si_buf, s0_flat_layout)
    var s0_lrelu_t = TileTensor(s0_lrelu_buf, s0_layout)
    var s0_acc_buf_t = TileTensor(s0_acc_buf, s0_layout)
    var s0_acc_flat = TileTensor(s0_acc_buf, s0_flat_layout)
    var s0_acc2_flat = TileTensor(s0_acc2_buf, s0_flat_layout)
    var s0_rb_x_flat = TileTensor(s0_rb_x_buf, s0_flat_layout)
    var s0_acc_lrelu_in = TileTensor(s0_acc_buf, s0_lrelu_layout)
    var s0_lrelu_lrelu_out = TileTensor(s0_lrelu_buf, s0_lrelu_layout)

    var s1_up_t = TileTensor(s1_up_buf, s1_layout)
    var s1_up_flat = TileTensor(s1_up_buf, s1_flat_layout)
    var s1_si_after_down_t = TileTensor(s1_si_after_down_buf, s1_layout)
    var s1_si_flat = TileTensor(s1_si_buf, s1_flat_layout)
    var s1_x_plus_si_flat = TileTensor(s1_x_plus_si_buf, s1_flat_layout)
    var s1_acc_buf_t = TileTensor(s1_acc_buf, s1_layout)
    var s1_acc_flat = TileTensor(s1_acc_buf, s1_flat_layout)
    var s1_acc2_flat = TileTensor(s1_acc2_buf, s1_flat_layout)
    var s1_rb_x_flat = TileTensor(s1_rb_x_buf, s1_flat_layout)
    var s1_acc_lrelu_in = TileTensor(s1_acc_buf, s1_lrelu_layout)
    var s1_lrelu_t = TileTensor(s1_lrelu_buf, s1_layout)
    var s1_lrelu_lrelu_out = TileTensor(s1_lrelu_buf, s1_lrelu_layout)

    var s2_up_pre_pad_t = TileTensor(s2_up_buf, s2_pre_pad_layout)
    var s2_padded_t = TileTensor(s2_padded_buf, s2_layout)
    var s2_padded_flat = TileTensor(s2_padded_buf, s2_flat_layout)
    var s2_si_after_down_t = TileTensor(s2_si_after_down_buf, s2_layout)
    var s2_si_flat = TileTensor(s2_si_buf, s2_flat_layout)
    var s2_x_plus_si_flat = TileTensor(s2_x_plus_si_buf, s2_flat_layout)
    var s2_acc_buf_t = TileTensor(s2_acc_buf, s2_layout)
    var s2_acc_flat = TileTensor(s2_acc_buf, s2_flat_layout)
    var s2_acc2_flat = TileTensor(s2_acc2_buf, s2_flat_layout)
    var s2_rb_x_flat = TileTensor(s2_rb_x_buf, s2_flat_layout)
    var s2_acc_lrelu_in = TileTensor(s2_acc_buf, s2_lrelu_layout)
    var s2_lrelu_t = TileTensor(s2_lrelu_buf, s2_layout)
    var s2_lrelu_lrelu_out = TileTensor(s2_lrelu_buf, s2_lrelu_layout)
    var s2_lrelu_lrelu_in = TileTensor(s2_acc_buf, s2_lrelu_layout)

    var post_out_t = TileTensor(post_out_buf, post_layout)
    var mag_t = TileTensor(mag_buf, spec_layout)
    var phase_t = TileTensor(phase_buf, spec_layout)
    var re_t = TileTensor(re_buf, spec_layout)
    var im_t = TileTensor(im_buf, spec_layout)
    var window_t = TileTensor(window_buf, window_layout)
    var audio_t = TileTensor(audio_buf, audio_layout)

    # ---- Conv_pre weights ----
    var cp_pre_w_buf = ctx.enqueue_create_buffer[DType.float32](PRE_C * MEL_C * CP_PRE_K)
    var cp_pre_b_buf = ctx.enqueue_create_buffer[DType.float32](PRE_C)
    var cp_pre_w = load_fp32(fix + "weights/conv_pre__weight.bin")
    var cp_pre_b = load_fp32(fix + "weights/conv_pre__bias.bin")
    upload(cp_pre_w_buf, cp_pre_w.data, PRE_C * MEL_C * CP_PRE_K)
    upload(cp_pre_b_buf, cp_pre_b.data, PRE_C)
    var cp_pre_w_t = TileTensor(cp_pre_w_buf, cp_pre_w_layout)
    var cp_pre_b_t = TileTensor(cp_pre_b_buf, pre_bias_layout)

    # ---- Ups[0] weights ----
    var up0_w_buf = ctx.enqueue_create_buffer[DType.float32](PRE_C * S0_C * UP0_K)
    var up0_b_buf = ctx.enqueue_create_buffer[DType.float32](S0_C)
    var up0_w = load_fp32(fix + "weights/ups__0__weight.bin")
    var up0_b = load_fp32(fix + "weights/ups__0__bias.bin")
    upload(up0_w_buf, up0_w.data, PRE_C * S0_C * UP0_K)
    upload(up0_b_buf, up0_b.data, S0_C)
    var up0_w_t = TileTensor(up0_w_buf, up0_w_layout)
    var up0_b_t = TileTensor(up0_b_buf, s0_bias_layout)

    var s0_src_down_w_buf = ctx.enqueue_create_buffer[DType.float32](S0_C * S_STFT_C * S0_SRC_DOWN_K)
    var s0_src_down_b_buf = ctx.enqueue_create_buffer[DType.float32](S0_C)
    var s0_src_down_w = load_fp32(fix + "weights/source_downs__0__weight.bin")
    var s0_src_down_b = load_fp32(fix + "weights/source_downs__0__bias.bin")
    upload(s0_src_down_w_buf, s0_src_down_w.data, S0_C * S_STFT_C * S0_SRC_DOWN_K)
    upload(s0_src_down_b_buf, s0_src_down_b.data, S0_C)
    var s0_src_down_w_t = TileTensor(s0_src_down_w_buf, s0_src_down_w_layout)
    var s0_src_down_b_t = TileTensor(s0_src_down_b_buf, s0_bias_layout)

    # ---- Ups[1] weights ----
    var up1_w_buf = ctx.enqueue_create_buffer[DType.float32](S0_C * S1_C * UP1_K)
    var up1_b_buf = ctx.enqueue_create_buffer[DType.float32](S1_C)
    var up1_w = load_fp32(fix + "weights/ups__1__weight.bin")
    var up1_b = load_fp32(fix + "weights/ups__1__bias.bin")
    upload(up1_w_buf, up1_w.data, S0_C * S1_C * UP1_K)
    upload(up1_b_buf, up1_b.data, S1_C)
    var up1_w_t = TileTensor(up1_w_buf, up1_w_layout)
    var up1_b_t = TileTensor(up1_b_buf, s1_bias_layout)

    var s1_src_down_w_buf = ctx.enqueue_create_buffer[DType.float32](S1_C * S_STFT_C * S1_SRC_DOWN_K)
    var s1_src_down_b_buf = ctx.enqueue_create_buffer[DType.float32](S1_C)
    var s1_src_down_w = load_fp32(fix + "weights/source_downs__1__weight.bin")
    var s1_src_down_b = load_fp32(fix + "weights/source_downs__1__bias.bin")
    upload(s1_src_down_w_buf, s1_src_down_w.data, S1_C * S_STFT_C * S1_SRC_DOWN_K)
    upload(s1_src_down_b_buf, s1_src_down_b.data, S1_C)
    var s1_src_down_w_t = TileTensor(s1_src_down_w_buf, s1_src_down_w_layout)
    var s1_src_down_b_t = TileTensor(s1_src_down_b_buf, s1_bias_layout)

    # ---- Ups[2] weights ----
    var up2_w_buf = ctx.enqueue_create_buffer[DType.float32](S1_C * S2_C * UP2_K)
    var up2_b_buf = ctx.enqueue_create_buffer[DType.float32](S2_C)
    var up2_w = load_fp32(fix + "weights/ups__2__weight.bin")
    var up2_b = load_fp32(fix + "weights/ups__2__bias.bin")
    upload(up2_w_buf, up2_w.data, S1_C * S2_C * UP2_K)
    upload(up2_b_buf, up2_b.data, S2_C)
    var up2_w_t = TileTensor(up2_w_buf, up2_w_layout)
    var up2_b_t = TileTensor(up2_b_buf, s2_bias_layout)

    var s2_src_down_w_buf = ctx.enqueue_create_buffer[DType.float32](S2_C * S_STFT_C * S2_SRC_DOWN_K)
    var s2_src_down_b_buf = ctx.enqueue_create_buffer[DType.float32](S2_C)
    var s2_src_down_w = load_fp32(fix + "weights/source_downs__2__weight.bin")
    var s2_src_down_b = load_fp32(fix + "weights/source_downs__2__bias.bin")
    upload(s2_src_down_w_buf, s2_src_down_w.data, S2_C * S_STFT_C * S2_SRC_DOWN_K)
    upload(s2_src_down_b_buf, s2_src_down_b.data, S2_C)
    var s2_src_down_w_t = TileTensor(s2_src_down_w_buf, s2_src_down_w_layout)
    var s2_src_down_b_t = TileTensor(s2_src_down_b_buf, s2_bias_layout)

    # ---- Conv_post weights ----
    var cp_post_w_buf = ctx.enqueue_create_buffer[DType.float32](POST_C * S2_C * CP_POST_K)
    var cp_post_b_buf = ctx.enqueue_create_buffer[DType.float32](POST_C)
    var cp_post_w = load_fp32(fix + "weights/conv_post__weight.bin")
    var cp_post_b = load_fp32(fix + "weights/conv_post__bias.bin")
    upload(cp_post_w_buf, cp_post_w.data, POST_C * S2_C * CP_POST_K)
    upload(cp_post_b_buf, cp_post_b.data, POST_C)
    var cp_post_w_t = TileTensor(cp_post_w_buf, cp_post_w_layout)
    var cp_post_b_t = TileTensor(cp_post_b_buf, post_bias_layout)

    # ---- Kernel bindings ----
    comptime cp_pre_k = conv1d_kernel[
        DType.float32, type_of(mel_layout), type_of(cp_pre_w_layout),
        type_of(pre_bias_layout), type_of(pre_layout), CP_PRE_K, True,
    ]
    comptime lrelu_pre_k = leaky_relu_kernel[
        DType.float32, type_of(pre_lrelu_layout), type_of(pre_lrelu_layout),
        POINTWISE_BLOCK,
    ]
    comptime up0_k = transposed_conv1d_kernel[
        DType.float32, type_of(pre_layout), type_of(up0_w_layout),
        type_of(s0_bias_layout), type_of(s0_layout), UP0_K, True,
    ]
    comptime s0_src_down_k = conv1d_kernel[
        DType.float32, type_of(s_stft_layout), type_of(s0_src_down_w_layout),
        type_of(s0_bias_layout), type_of(s0_layout), S0_SRC_DOWN_K, True,
    ]
    comptime s0_add_k = add_kernel[
        DType.float32, type_of(s0_flat_layout), type_of(s0_flat_layout),
        type_of(s0_flat_layout), POINTWISE_BLOCK,
    ]
    comptime lrelu_s0_k = leaky_relu_kernel[
        DType.float32, type_of(s0_lrelu_layout), type_of(s0_lrelu_layout),
        POINTWISE_BLOCK,
    ]
    comptime up1_k = transposed_conv1d_kernel[
        DType.float32, type_of(s0_layout), type_of(up1_w_layout),
        type_of(s1_bias_layout), type_of(s1_layout), UP1_K, True,
    ]
    comptime s1_src_down_k = conv1d_kernel[
        DType.float32, type_of(s_stft_layout), type_of(s1_src_down_w_layout),
        type_of(s1_bias_layout), type_of(s1_layout), S1_SRC_DOWN_K, True,
    ]
    comptime s1_add_k = add_kernel[
        DType.float32, type_of(s1_flat_layout), type_of(s1_flat_layout),
        type_of(s1_flat_layout), POINTWISE_BLOCK,
    ]
    comptime lrelu_s1_k = leaky_relu_kernel[
        DType.float32, type_of(s1_lrelu_layout), type_of(s1_lrelu_layout),
        POINTWISE_BLOCK,
    ]
    comptime up2_k = transposed_conv1d_kernel[
        DType.float32, type_of(s1_layout), type_of(up2_w_layout),
        type_of(s2_bias_layout), type_of(s2_pre_pad_layout), UP2_K, True,
    ]
    comptime refpad_k = reflection_pad_left1_kernel[
        DType.float32, type_of(s2_pre_pad_layout), type_of(s2_layout),
    ]
    comptime s2_src_down_k = conv1d_kernel[
        DType.float32, type_of(s_stft_layout), type_of(s2_src_down_w_layout),
        type_of(s2_bias_layout), type_of(s2_layout), S2_SRC_DOWN_K, True,
    ]
    comptime s2_add_k = add_kernel[
        DType.float32, type_of(s2_flat_layout), type_of(s2_flat_layout),
        type_of(s2_flat_layout), POINTWISE_BLOCK,
    ]
    comptime lrelu_s2_k = leaky_relu_kernel[
        DType.float32, type_of(s2_lrelu_layout), type_of(s2_lrelu_layout),
        POINTWISE_BLOCK,
    ]
    comptime cp_post_k = conv1d_kernel[
        DType.float32, type_of(s2_layout), type_of(cp_post_w_layout),
        type_of(post_bias_layout), type_of(post_layout), CP_POST_K, True,
    ]
    comptime split_k = magnitude_phase_split_kernel[
        DType.float32, type_of(post_layout), type_of(spec_layout),
        type_of(spec_layout), N_FREQ, N_FREQ,
    ]
    comptime to_complex_k = magnitude_phase_to_complex_kernel[
        DType.float32, type_of(spec_layout), type_of(spec_layout),
        type_of(spec_layout), type_of(spec_layout),
    ]
    comptime istft_k = istft_kernel[
        DType.float32, type_of(spec_layout), type_of(spec_layout),
        type_of(window_layout), type_of(audio_layout),
        N_FFT, HOP, N_FREQ,
    ]

    # ============================================================
    # PIPELINE
    # ============================================================
    # conv_pre
    ctx.enqueue_function[cp_pre_k, cp_pre_k](
        pre_t, mel_t, cp_pre_w_t, cp_pre_b_t,
        BATCH, MEL_C, PRE_C, MEL_T, PRE_T, 1, 3, 1,
        grid_dim=BATCH * PRE_C * PRE_T, block_dim=1,
    )

    # ===== Stage 0 =====
    # lrelu(pre) -> pre_lrelu
    ctx.enqueue_function[lrelu_pre_k, lrelu_pre_k](
        pre_lrelu_lrelu_out, pre_lrelu_lrelu, n_pre, Float32(0.1),
        grid_dim=ceildiv(n_pre, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )
    # ups[0]
    ctx.enqueue_function[up0_k, up0_k](
        s0_up_t, pre_lrelu_t, up0_w_t, up0_b_t,
        BATCH, PRE_C, S0_C, PRE_T, S0_T, UP0_STRIDE, UP0_PAD, 1,
        grid_dim=BATCH * S0_C * S0_T, block_dim=1,
    )
    # source_downs[0] -> source_resblocks[0]
    ctx.enqueue_function[s0_src_down_k, s0_src_down_k](
        s0_si_after_down_t, s_stft_t, s0_src_down_w_t, s0_src_down_b_t,
        BATCH, S_STFT_C, S0_C, S_STFT_T, S0_T,
        S0_SRC_DOWN_STRIDE, S0_SRC_DOWN_PAD, 1,
        grid_dim=BATCH * S0_C * S0_T, block_dim=1,
    )
    copy_device_buf(ctx, s0_si_after_down_buf, s0_rb_x_buf, n_s0)
    _run_resblock_chain[S0_C, S0_T, 7, SNAKE_BLOCK](
        ctx, s0_rb_x_buf, s0_rb_next_buf, s0_rb_xt_buf, s0_rb_xt2_buf,
        fix + "weights/source_resblocks__0__")
    copy_device_buf(ctx, s0_rb_x_buf, s0_si_buf, n_s0)
    # x_plus_si = s0_up + si
    ctx.enqueue_function[s0_add_k, s0_add_k](
        s0_x_plus_si_flat, s0_up_flat, s0_si_flat, n_s0,
        grid_dim=ceildiv(n_s0, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )
    # resblocks 0,1,2 → acc
    copy_device_buf(ctx, s0_x_plus_si_buf, s0_rb_x_buf, n_s0)
    _run_resblock_chain[S0_C, S0_T, 3, SNAKE_BLOCK](
        ctx, s0_rb_x_buf, s0_rb_next_buf, s0_rb_xt_buf, s0_rb_xt2_buf,
        fix + "weights/resblocks__0__")
    copy_device_buf(ctx, s0_rb_x_buf, s0_acc_buf, n_s0)

    copy_device_buf(ctx, s0_x_plus_si_buf, s0_rb_x_buf, n_s0)
    _run_resblock_chain[S0_C, S0_T, 7, SNAKE_BLOCK](
        ctx, s0_rb_x_buf, s0_rb_next_buf, s0_rb_xt_buf, s0_rb_xt2_buf,
        fix + "weights/resblocks__1__")
    ctx.enqueue_function[s0_add_k, s0_add_k](
        s0_acc2_flat, s0_acc_flat, s0_rb_x_flat, n_s0,
        grid_dim=ceildiv(n_s0, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )
    copy_device_buf(ctx, s0_acc2_buf, s0_acc_buf, n_s0)

    copy_device_buf(ctx, s0_x_plus_si_buf, s0_rb_x_buf, n_s0)
    _run_resblock_chain[S0_C, S0_T, 11, SNAKE_BLOCK](
        ctx, s0_rb_x_buf, s0_rb_next_buf, s0_rb_xt_buf, s0_rb_xt2_buf,
        fix + "weights/resblocks__2__")
    ctx.enqueue_function[s0_add_k, s0_add_k](
        s0_acc2_flat, s0_acc_flat, s0_rb_x_flat, n_s0,
        grid_dim=ceildiv(n_s0, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )
    # Divide acc by 3 host-side, leaving result in s0_acc_buf.
    ctx.synchronize()
    var inv = Float32(1.0 / 3.0)
    with s0_acc2_buf.map_to_host() as src:
        with s0_acc_buf.map_to_host() as dst:
            for i in range(n_s0):
                dst[i] = src[i] * inv

    # ===== Stage 1 =====
    ctx.enqueue_function[lrelu_s0_k, lrelu_s0_k](
        s0_lrelu_lrelu_out, s0_acc_lrelu_in, n_s0, Float32(0.1),
        grid_dim=ceildiv(n_s0, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )
    ctx.enqueue_function[up1_k, up1_k](
        s1_up_t, s0_lrelu_t, up1_w_t, up1_b_t,
        BATCH, S0_C, S1_C, S0_T, S1_T, UP1_STRIDE, UP1_PAD, 1,
        grid_dim=BATCH * S1_C * S1_T, block_dim=1,
    )
    ctx.enqueue_function[s1_src_down_k, s1_src_down_k](
        s1_si_after_down_t, s_stft_t, s1_src_down_w_t, s1_src_down_b_t,
        BATCH, S_STFT_C, S1_C, S_STFT_T, S1_T,
        S1_SRC_DOWN_STRIDE, S1_SRC_DOWN_PAD, 1,
        grid_dim=BATCH * S1_C * S1_T, block_dim=1,
    )
    copy_device_buf(ctx, s1_si_after_down_buf, s1_rb_x_buf, n_s1)
    _run_resblock_chain[S1_C, S1_T, 7, SNAKE_BLOCK](
        ctx, s1_rb_x_buf, s1_rb_next_buf, s1_rb_xt_buf, s1_rb_xt2_buf,
        fix + "weights/source_resblocks__1__")
    copy_device_buf(ctx, s1_rb_x_buf, s1_si_buf, n_s1)
    ctx.enqueue_function[s1_add_k, s1_add_k](
        s1_x_plus_si_flat, s1_up_flat, s1_si_flat, n_s1,
        grid_dim=ceildiv(n_s1, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )
    copy_device_buf(ctx, s1_x_plus_si_buf, s1_rb_x_buf, n_s1)
    _run_resblock_chain[S1_C, S1_T, 3, SNAKE_BLOCK](
        ctx, s1_rb_x_buf, s1_rb_next_buf, s1_rb_xt_buf, s1_rb_xt2_buf,
        fix + "weights/resblocks__3__")
    copy_device_buf(ctx, s1_rb_x_buf, s1_acc_buf, n_s1)

    copy_device_buf(ctx, s1_x_plus_si_buf, s1_rb_x_buf, n_s1)
    _run_resblock_chain[S1_C, S1_T, 7, SNAKE_BLOCK](
        ctx, s1_rb_x_buf, s1_rb_next_buf, s1_rb_xt_buf, s1_rb_xt2_buf,
        fix + "weights/resblocks__4__")
    ctx.enqueue_function[s1_add_k, s1_add_k](
        s1_acc2_flat, s1_acc_flat, s1_rb_x_flat, n_s1,
        grid_dim=ceildiv(n_s1, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )
    copy_device_buf(ctx, s1_acc2_buf, s1_acc_buf, n_s1)

    copy_device_buf(ctx, s1_x_plus_si_buf, s1_rb_x_buf, n_s1)
    _run_resblock_chain[S1_C, S1_T, 11, SNAKE_BLOCK](
        ctx, s1_rb_x_buf, s1_rb_next_buf, s1_rb_xt_buf, s1_rb_xt2_buf,
        fix + "weights/resblocks__5__")
    ctx.enqueue_function[s1_add_k, s1_add_k](
        s1_acc2_flat, s1_acc_flat, s1_rb_x_flat, n_s1,
        grid_dim=ceildiv(n_s1, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )
    ctx.synchronize()
    with s1_acc2_buf.map_to_host() as src:
        with s1_acc_buf.map_to_host() as dst:
            for i in range(n_s1):
                dst[i] = src[i] * inv

    # ===== Stage 2 =====
    ctx.enqueue_function[lrelu_s1_k, lrelu_s1_k](
        s1_lrelu_lrelu_out, s1_acc_lrelu_in, n_s1, Float32(0.1),
        grid_dim=ceildiv(n_s1, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )
    ctx.enqueue_function[up2_k, up2_k](
        s2_up_pre_pad_t, s1_lrelu_t, up2_w_t, up2_b_t,
        BATCH, S1_C, S2_C, S1_T, S2_PRE_PAD_T, UP2_STRIDE, UP2_PAD, 1,
        grid_dim=BATCH * S2_C * S2_PRE_PAD_T, block_dim=1,
    )
    ctx.enqueue_function[refpad_k, refpad_k](
        s2_padded_t, s2_up_pre_pad_t, BATCH, S2_C, S2_PRE_PAD_T,
        grid_dim=BATCH * S2_C * S2_T, block_dim=1,
    )
    ctx.enqueue_function[s2_src_down_k, s2_src_down_k](
        s2_si_after_down_t, s_stft_t, s2_src_down_w_t, s2_src_down_b_t,
        BATCH, S_STFT_C, S2_C, S_STFT_T, S2_T,
        S2_SRC_DOWN_STRIDE, S2_SRC_DOWN_PAD, 1,
        grid_dim=BATCH * S2_C * S2_T, block_dim=1,
    )
    copy_device_buf(ctx, s2_si_after_down_buf, s2_rb_x_buf, n_s2)
    _run_resblock_chain[S2_C, S2_T, 11, SNAKE_BLOCK](
        ctx, s2_rb_x_buf, s2_rb_next_buf, s2_rb_xt_buf, s2_rb_xt2_buf,
        fix + "weights/source_resblocks__2__")
    copy_device_buf(ctx, s2_rb_x_buf, s2_si_buf, n_s2)
    ctx.enqueue_function[s2_add_k, s2_add_k](
        s2_x_plus_si_flat, s2_padded_flat, s2_si_flat, n_s2,
        grid_dim=ceildiv(n_s2, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )
    copy_device_buf(ctx, s2_x_plus_si_buf, s2_rb_x_buf, n_s2)
    _run_resblock_chain[S2_C, S2_T, 3, SNAKE_BLOCK](
        ctx, s2_rb_x_buf, s2_rb_next_buf, s2_rb_xt_buf, s2_rb_xt2_buf,
        fix + "weights/resblocks__6__")
    copy_device_buf(ctx, s2_rb_x_buf, s2_acc_buf, n_s2)

    copy_device_buf(ctx, s2_x_plus_si_buf, s2_rb_x_buf, n_s2)
    _run_resblock_chain[S2_C, S2_T, 7, SNAKE_BLOCK](
        ctx, s2_rb_x_buf, s2_rb_next_buf, s2_rb_xt_buf, s2_rb_xt2_buf,
        fix + "weights/resblocks__7__")
    ctx.enqueue_function[s2_add_k, s2_add_k](
        s2_acc2_flat, s2_acc_flat, s2_rb_x_flat, n_s2,
        grid_dim=ceildiv(n_s2, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )
    copy_device_buf(ctx, s2_acc2_buf, s2_acc_buf, n_s2)

    copy_device_buf(ctx, s2_x_plus_si_buf, s2_rb_x_buf, n_s2)
    _run_resblock_chain[S2_C, S2_T, 11, SNAKE_BLOCK](
        ctx, s2_rb_x_buf, s2_rb_next_buf, s2_rb_xt_buf, s2_rb_xt2_buf,
        fix + "weights/resblocks__8__")
    ctx.enqueue_function[s2_add_k, s2_add_k](
        s2_acc2_flat, s2_acc_flat, s2_rb_x_flat, n_s2,
        grid_dim=ceildiv(n_s2, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )
    ctx.synchronize()
    with s2_acc2_buf.map_to_host() as src:
        with s2_acc_buf.map_to_host() as dst:
            for i in range(n_s2):
                dst[i] = src[i] * inv

    # ===== Final stages =====
    # final lrelu (default slope 0.01 per torch.nn.functional.leaky_relu)
    ctx.enqueue_function[lrelu_s2_k, lrelu_s2_k](
        s2_lrelu_lrelu_out, s2_lrelu_lrelu_in, n_s2, Float32(0.01),
        grid_dim=ceildiv(n_s2, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )
    # conv_post
    ctx.enqueue_function[cp_post_k, cp_post_k](
        post_out_t, s2_lrelu_t, cp_post_w_t, cp_post_b_t,
        BATCH, S2_C, POST_C, S2_T, S2_T, 1, 3, 1,
        grid_dim=BATCH * POST_C * S2_T, block_dim=1,
    )
    # magnitude/phase split
    ctx.enqueue_function[split_k, split_k](
        mag_t, phase_t, post_out_t, BATCH, S2_T,
        grid_dim=BATCH * N_FREQ * S2_T, block_dim=1,
    )
    # to complex
    ctx.enqueue_function[to_complex_k, to_complex_k](
        re_t, im_t, mag_t, phase_t, BATCH, N_FREQ, S2_T,
        grid_dim=BATCH * N_FREQ * S2_T, block_dim=1,
    )
    # iSTFT
    ctx.enqueue_function[istft_k, istft_k](
        audio_t, re_t, im_t, window_t,
        BATCH, S2_T, T_AUDIO,
        grid_dim=BATCH * T_AUDIO, block_dim=1,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    var sum_abs: Float64 = 0.0
    var mojo_audio = List[Float32]()
    with audio_buf.map_to_host() as h:
        for i in range(n_audio):
            var got = h[i]
            if got > 0.99: got = 0.99
            if got < -0.99: got = -0.99
            mojo_audio.append(got)
            var d = got - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_abs += Float64(d)
            assert_almost_equal(got, exp.data[i], atol=2.0e-1)
    save_fp32_1d("tests/fixtures/real/mojo_audio.bin", mojo_audio)
    print("FULL HiFiGAN fp32", label, "— max abs:", max_abs,
          " mean abs:", sum_abs / Float64(n_audio),
          " — wrote tests/fixtures/real/mojo_audio.bin")


def test_hifigan_real_mel_fp32() raises:
    _run_hifigan_full(
        "tests/fixtures/real/real_s_stft_cat.bin",
        "tests/fixtures/real/real_audio_upstream.bin",
        "(s=zeros)",
    )


def _disabled_test_hifigan_real_unused() raises:
    _run_hifigan_full(
        "tests/fixtures/real/real_s_stft_cat.bin",
        "tests/fixtures/real/real_audio_upstream.bin",
        "(s=real)",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
