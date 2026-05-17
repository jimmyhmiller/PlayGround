"""
HiFiGAN parity tests against real Chatterbox HiFTGenerator weights:
  1. conv_pre(mel)                            — conv1d at (1,80,32) -> (1,512,32)
  2. leaky_relu(.)                             — checks our LR kernel on real data
  3. ups[0] transposed_conv1d                  — first upsample, 32 -> 256 samples
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import Tensor, load_fp32
from conv import (
    conv1d_kernel, transposed_conv1d_kernel, leaky_relu_kernel,
    magnitude_phase_split_kernel, magnitude_phase_to_complex_kernel,
)
from stft import istft_kernel


comptime BATCH = 1
comptime C_IN = 80
comptime C_OUT = 512
comptime T_MEL = 32
comptime K = 7
comptime STRIDE = 1
comptime PADDING = 3
comptime DILATION = 1


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def test_conv_pre_fp32() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/hifigan/"
    var mel = load_fp32(fix + "mel.bin")
    var w = load_fp32(fix + "weights/conv_pre__weight.bin")
    var bias = load_fp32(fix + "weights/conv_pre__bias.bin")
    var exp = load_fp32(fix + "stage_after_conv_pre.bin")

    assert_equal(mel.shape[0], BATCH)
    assert_equal(mel.shape[1], C_IN)
    assert_equal(mel.shape[2], T_MEL)
    assert_equal(w.shape[0], C_OUT)
    assert_equal(w.shape[1], C_IN)
    assert_equal(w.shape[2], K)
    assert_equal(exp.shape[1], C_OUT)
    assert_equal(exp.shape[2], T_MEL)

    var n_x = BATCH * C_IN * T_MEL
    var n_w = C_OUT * C_IN * K
    var n_out = BATCH * C_OUT * T_MEL

    var ctx = DeviceContext()
    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var w_buf = ctx.enqueue_create_buffer[DType.float32](n_w)
    var bias_buf = ctx.enqueue_create_buffer[DType.float32](C_OUT)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    upload(x_buf, mel.data, n_x)
    upload(w_buf, w.data, n_w)
    upload(bias_buf, bias.data, C_OUT)

    comptime x_layout = row_major[BATCH, C_IN, T_MEL]()
    comptime w_layout = row_major[C_OUT, C_IN, K]()
    comptime bias_layout = row_major[C_OUT]()
    comptime out_layout = row_major[BATCH, C_OUT, T_MEL]()

    var x_t = TileTensor(x_buf, x_layout)
    var w_t = TileTensor(w_buf, w_layout)
    var bias_t = TileTensor(bias_buf, bias_layout)
    var out_t = TileTensor(out_buf, out_layout)

    comptime kernel = conv1d_kernel[
        DType.float32, type_of(x_layout), type_of(w_layout),
        type_of(bias_layout), type_of(out_layout), K, True,
    ]
    ctx.enqueue_function[kernel, kernel](
        out_t, x_t, w_t, bias_t,
        BATCH, C_IN, C_OUT, T_MEL, T_MEL, STRIDE, PADDING, DILATION,
        grid_dim=BATCH * C_OUT * T_MEL, block_dim=1,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n_out):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            # Real-weight conv1d: tolerance covers fp32 reduction-order
            # differences vs PyTorch (still small).
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-4)
    print("conv_pre fp32 (real T3 weights) — max abs:", max_abs)


comptime POINTWISE_BLOCK = 256

# First upsample stage: ups[0] = ConvTranspose1d(512 -> 256, K=16, stride=8, padding=4).
comptime UPS0_C_IN = 512
comptime UPS0_C_OUT = 256
comptime UPS0_K = 16
comptime UPS0_STRIDE = 8
comptime UPS0_PADDING = 4
comptime UPS0_T_IN = 32     # = T_MEL
comptime UPS0_T_OUT = 256   # (32 - 1) * 8 - 8 + 16 = 256


def test_leaky_relu_after_conv_pre_fp32() raises:
    """Verify leaky_relu(conv_pre(mel)) matches stage_up0_after_lrelu."""
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/hifigan/"
    var x_in = load_fp32(fix + "stage_after_conv_pre.bin")
    var exp = load_fp32(fix + "stage_up0_after_lrelu.bin")

    var n = BATCH * C_OUT * T_MEL

    var ctx = DeviceContext()
    var x_buf = ctx.enqueue_create_buffer[DType.float32](n)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n)
    upload(x_buf, x_in.data, n)

    comptime flat_layout = row_major[BATCH * C_OUT * T_MEL]()
    var x_t = TileTensor(x_buf, flat_layout)
    var out_t = TileTensor(out_buf, flat_layout)

    comptime kernel = leaky_relu_kernel[
        DType.float32, type_of(flat_layout), type_of(flat_layout), POINTWISE_BLOCK,
    ]
    ctx.enqueue_function[kernel, kernel](
        out_t, x_t, n, Float32(0.1),
        grid_dim=ceildiv(n, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            assert_almost_equal(h[i], exp.data[i], atol=0.0)
    print("leaky_relu fp32 (after conv_pre) — max abs:", max_abs)


def test_ups0_fp32() raises:
    """Verify ups[0] transposed_conv1d matches stage_up0_after_transposed_conv.

    Uses the upstream-dumped lrelu output as input (so this test isolates
    the transposed conv).
    """
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/hifigan/"
    var x_in = load_fp32(fix + "stage_up0_after_lrelu.bin")
    var w = load_fp32(fix + "weights/ups__0__weight.bin")
    var bias = load_fp32(fix + "weights/ups__0__bias.bin")
    var exp = load_fp32(fix + "stage_up0_after_transposed_conv.bin")

    assert_equal(w.shape[0], UPS0_C_IN)
    assert_equal(w.shape[1], UPS0_C_OUT)
    assert_equal(w.shape[2], UPS0_K)
    assert_equal(bias.shape[0], UPS0_C_OUT)
    assert_equal(exp.shape[1], UPS0_C_OUT)
    assert_equal(exp.shape[2], UPS0_T_OUT)

    var n_x = BATCH * UPS0_C_IN * UPS0_T_IN
    var n_w = UPS0_C_IN * UPS0_C_OUT * UPS0_K
    var n_out = BATCH * UPS0_C_OUT * UPS0_T_OUT

    var ctx = DeviceContext()
    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var w_buf = ctx.enqueue_create_buffer[DType.float32](n_w)
    var bias_buf = ctx.enqueue_create_buffer[DType.float32](UPS0_C_OUT)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    upload(x_buf, x_in.data, n_x)
    upload(w_buf, w.data, n_w)
    upload(bias_buf, bias.data, UPS0_C_OUT)

    comptime x_layout = row_major[BATCH, UPS0_C_IN, UPS0_T_IN]()
    comptime w_layout = row_major[UPS0_C_IN, UPS0_C_OUT, UPS0_K]()
    comptime bias_layout = row_major[UPS0_C_OUT]()
    comptime out_layout = row_major[BATCH, UPS0_C_OUT, UPS0_T_OUT]()

    var x_t = TileTensor(x_buf, x_layout)
    var w_t = TileTensor(w_buf, w_layout)
    var bias_t = TileTensor(bias_buf, bias_layout)
    var out_t = TileTensor(out_buf, out_layout)

    comptime kernel = transposed_conv1d_kernel[
        DType.float32, type_of(x_layout), type_of(w_layout),
        type_of(bias_layout), type_of(out_layout), UPS0_K, True,
    ]
    ctx.enqueue_function[kernel, kernel](
        out_t, x_t, w_t, bias_t,
        BATCH, UPS0_C_IN, UPS0_C_OUT, UPS0_T_IN, UPS0_T_OUT,
        UPS0_STRIDE, UPS0_PADDING, 1,
        grid_dim=BATCH * UPS0_C_OUT * UPS0_T_OUT, block_dim=1,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n_out):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-4)
    print("ups[0] transposed_conv1d fp32 (real weights) — max abs:", max_abs)


comptime CP_C_IN = 64
comptime CP_C_OUT = 18
comptime CP_K = 7
comptime CP_T = 3841   # padded length after reflection_pad in upstream
comptime N_FFT = 16
comptime HOP = 4
comptime N_FREQ = N_FFT // 2 + 1   # 9
comptime T_AUDIO = 15360


def test_final_stages_fp32() raises:
    """End of HiFiGAN: conv_post → magnitude/phase → iSTFT → audio.

    Input: stage_after_final_lrelu (1, 64, 3841) from upstream.
    Output: expected_wav (s=zeros path, 1, 15360).
    """
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/hifigan/"
    var x_in = load_fp32(fix + "stage_after_final_lrelu.bin")
    var w = load_fp32(fix + "weights/conv_post__weight.bin")
    var bias = load_fp32(fix + "weights/conv_post__bias.bin")
    var window = load_fp32(fix + "weights/stft_window.bin")
    var exp_audio = load_fp32(fix + "expected_wav_decode_zeros.bin")

    assert_equal(x_in.shape[1], CP_C_IN)
    assert_equal(x_in.shape[2], CP_T)
    assert_equal(w.shape[0], CP_C_OUT)
    assert_equal(w.shape[1], CP_C_IN)
    assert_equal(w.shape[2], CP_K)
    assert_equal(window.shape[0], N_FFT)
    assert_equal(exp_audio.shape[1], T_AUDIO)

    var n_x = BATCH * CP_C_IN * CP_T
    var n_w = CP_C_OUT * CP_C_IN * CP_K
    var n_cp_out = BATCH * CP_C_OUT * CP_T
    var n_spec = BATCH * N_FREQ * CP_T
    var n_audio = BATCH * T_AUDIO

    var ctx = DeviceContext()
    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var w_buf = ctx.enqueue_create_buffer[DType.float32](n_w)
    var bias_buf = ctx.enqueue_create_buffer[DType.float32](CP_C_OUT)
    var cp_out_buf = ctx.enqueue_create_buffer[DType.float32](n_cp_out)
    var mag_buf = ctx.enqueue_create_buffer[DType.float32](n_spec)
    var phase_buf = ctx.enqueue_create_buffer[DType.float32](n_spec)
    var re_buf = ctx.enqueue_create_buffer[DType.float32](n_spec)
    var im_buf = ctx.enqueue_create_buffer[DType.float32](n_spec)
    var window_buf = ctx.enqueue_create_buffer[DType.float32](N_FFT)
    var audio_buf = ctx.enqueue_create_buffer[DType.float32](n_audio)

    upload(x_buf, x_in.data, n_x)
    upload(w_buf, w.data, n_w)
    upload(bias_buf, bias.data, CP_C_OUT)
    upload(window_buf, window.data, N_FFT)

    comptime x_layout = row_major[BATCH, CP_C_IN, CP_T]()
    comptime w_layout = row_major[CP_C_OUT, CP_C_IN, CP_K]()
    comptime bias_layout = row_major[CP_C_OUT]()
    comptime cp_out_layout = row_major[BATCH, CP_C_OUT, CP_T]()
    comptime spec_layout = row_major[BATCH, N_FREQ, CP_T]()
    comptime window_layout = row_major[N_FFT]()
    comptime audio_layout = row_major[BATCH, T_AUDIO]()

    var x_t = TileTensor(x_buf, x_layout)
    var w_t = TileTensor(w_buf, w_layout)
    var bias_t = TileTensor(bias_buf, bias_layout)
    var cp_out_t = TileTensor(cp_out_buf, cp_out_layout)
    var mag_t = TileTensor(mag_buf, spec_layout)
    var phase_t = TileTensor(phase_buf, spec_layout)
    var re_t = TileTensor(re_buf, spec_layout)
    var im_t = TileTensor(im_buf, spec_layout)
    var window_t = TileTensor(window_buf, window_layout)
    var audio_t = TileTensor(audio_buf, audio_layout)

    comptime conv_k = conv1d_kernel[
        DType.float32, type_of(x_layout), type_of(w_layout),
        type_of(bias_layout), type_of(cp_out_layout), CP_K, True,
    ]
    comptime split_k = magnitude_phase_split_kernel[
        DType.float32, type_of(cp_out_layout), type_of(spec_layout),
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

    # 1. conv_post.
    ctx.enqueue_function[conv_k, conv_k](
        cp_out_t, x_t, w_t, bias_t,
        BATCH, CP_C_IN, CP_C_OUT, CP_T, CP_T, 1, 3, 1,
        grid_dim=BATCH * CP_C_OUT * CP_T, block_dim=1,
    )
    # 2. magnitude_phase_split.
    ctx.enqueue_function[split_k, split_k](
        mag_t, phase_t, cp_out_t, BATCH, CP_T,
        grid_dim=BATCH * N_FREQ * CP_T, block_dim=1,
    )
    # 3. magnitude/phase → real/imag.
    ctx.enqueue_function[to_complex_k, to_complex_k](
        re_t, im_t, mag_t, phase_t, BATCH, N_FREQ, CP_T,
        grid_dim=BATCH * N_FREQ * CP_T, block_dim=1,
    )
    # 4. iSTFT.
    ctx.enqueue_function[istft_k, istft_k](
        audio_t, re_t, im_t, window_t,
        BATCH, CP_T, T_AUDIO,
        grid_dim=BATCH * T_AUDIO, block_dim=1,
    )
    ctx.synchronize()

    # Compare audio. Apply the same clamp(±0.99) upstream does.
    var max_abs: Float32 = 0.0
    var sum_abs: Float64 = 0.0
    with audio_buf.map_to_host() as h:
        for i in range(n_audio):
            var got = h[i]
            if got > 0.99: got = 0.99
            if got < -0.99: got = -0.99
            var d = got - exp_audio.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_abs += Float64(d)
            # iSTFT amplifies any conv error; allow a slack budget for now.
            assert_almost_equal(got, exp_audio.data[i], atol=1.0e-2)
    print("HiFiGAN final stages fp32 (real weights, s=zeros) — max abs:", max_abs,
          " mean abs:", sum_abs / Float64(n_audio))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
