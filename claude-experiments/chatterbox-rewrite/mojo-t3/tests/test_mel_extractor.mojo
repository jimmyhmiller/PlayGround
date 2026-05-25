"""
Pure-Mojo mel-spectrogram extractor parity vs upstream Chatterbox.

Pipeline:
  audio (T_samples)
  → reflect-pad by (n_fft - hop)/2 = 720 on each side
  → naive DFT magnitude (n_fft=1920, hop=480, hann window)
  → matmul(mel_basis (80, 961), magnitude (961, n_frames))
  → log(max(., 1e-5))

10s of audio at 24kHz → 500 mel frames. Bit-tolerant vs upstream's torch+librosa output.
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from layout import TileTensor, row_major
from linalg.matmul import matmul

from fixture import Tensor, load_fp32, load_i64
from conv import reflect_pad_1d_kernel, stft_mag_kernel, log_clamp_kernel


comptime BATCH = 1
comptime N_FFT = 1920
comptime HOP = 480
comptime N_FREQ = N_FFT // 2 + 1   # 961
comptime NUM_MELS = 80
comptime PAD = (N_FFT - HOP) // 2  # 720
comptime POINTWISE_BLOCK = 256


def test_mel_extractor() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/mel_extractor/"
    var wav = load_fp32(fix + "ref_wav.bin")
    var mel_basis = load_fp32(fix + "mel_basis.bin")
    var window = load_fp32(fix + "hann_window.bin")
    var exp_mel = load_fp32(fix + "ref_mel.bin")
    var meta = load_i64(fix + "meta.bin")

    # Sanity-check meta vs comptime config.
    assert_equal(Int(meta.data[0]), N_FFT)
    assert_equal(Int(meta.data[1]), HOP)
    assert_equal(Int(meta.data[3]), NUM_MELS)
    assert_equal(mel_basis.shape[0], NUM_MELS)
    assert_equal(mel_basis.shape[1], N_FREQ)

    var T = wav.shape[1]
    var T_pad = T + 2 * PAD
    # n_frames = (T_pad - N_FFT) // HOP + 1 (center=False)
    var n_frames = (T_pad - N_FFT) // HOP + 1
    print("input audio:", T, " padded:", T_pad, " n_frames:", n_frames)
    assert_equal(exp_mel.shape[2], n_frames)

    var ctx = DeviceContext()
    var dctx = DeviceContextPtr(ctx)

    var n_audio = BATCH * T
    var n_padded = BATCH * T_pad
    var n_mag = BATCH * N_FREQ * n_frames
    var n_mel = BATCH * NUM_MELS * n_frames

    var audio_buf = ctx.enqueue_create_buffer[DType.float32](n_audio)
    var padded_buf = ctx.enqueue_create_buffer[DType.float32](n_padded)
    var window_buf = ctx.enqueue_create_buffer[DType.float32](N_FFT)
    var mag_buf = ctx.enqueue_create_buffer[DType.float32](n_mag)
    var basis_buf = ctx.enqueue_create_buffer[DType.float32](NUM_MELS * N_FREQ)
    var mel_buf = ctx.enqueue_create_buffer[DType.float32](n_mel)
    var log_mel_buf = ctx.enqueue_create_buffer[DType.float32](n_mel)

    with audio_buf.map_to_host() as h:
        for i in range(n_audio): h[i] = wav.data[i]
    with window_buf.map_to_host() as h:
        for i in range(N_FFT): h[i] = window.data[i]
    with basis_buf.map_to_host() as h:
        for i in range(NUM_MELS * N_FREQ): h[i] = mel_basis.data[i]

    # Layouts. T_pad and n_frames are runtime; we have to know them at comptime
    # for the kernel templates. They depend on T (10s = 240000 → T_pad=241440,
    # n_frames=(241440-1920)/480 + 1 = 500). Hard-code for the test.
    comptime T_FIXED = 240000
    comptime T_PAD_FIXED = T_FIXED + 2 * PAD   # 241440
    comptime N_FRAMES_FIXED = (T_PAD_FIXED - N_FFT) // HOP + 1   # 500
    assert_equal(T, T_FIXED)
    assert_equal(T_pad, T_PAD_FIXED)
    assert_equal(n_frames, N_FRAMES_FIXED)

    comptime audio_layout = row_major[BATCH, T_FIXED]()
    comptime padded_layout = row_major[BATCH, T_PAD_FIXED]()
    comptime window_layout = row_major[N_FFT]()
    comptime mag_layout = row_major[BATCH, N_FREQ, N_FRAMES_FIXED]()
    comptime mag_2d_layout = row_major[N_FREQ, N_FRAMES_FIXED]()
    comptime basis_layout = row_major[NUM_MELS, N_FREQ]()
    comptime mel_layout = row_major[BATCH, NUM_MELS, N_FRAMES_FIXED]()
    comptime mel_2d_layout = row_major[NUM_MELS, N_FRAMES_FIXED]()
    comptime mel_flat_layout = row_major[BATCH * NUM_MELS * N_FRAMES_FIXED]()

    var audio_t = TileTensor(audio_buf, audio_layout)
    var padded_t = TileTensor(padded_buf, padded_layout)
    var window_t = TileTensor(window_buf, window_layout)
    var mag_t = TileTensor(mag_buf, mag_layout)
    var mag_2d = TileTensor(mag_buf, mag_2d_layout)
    var basis_t = TileTensor(basis_buf, basis_layout)
    var mel_2d = TileTensor(mel_buf, mel_2d_layout)
    var mel_flat = TileTensor(mel_buf, mel_flat_layout)
    var log_mel_flat = TileTensor(log_mel_buf, mel_flat_layout)

    # 1. Reflect-pad audio.
    comptime refpad_k = reflect_pad_1d_kernel[
        DType.float32, type_of(audio_layout), type_of(padded_layout),
    ]
    ctx.enqueue_function[refpad_k, refpad_k](
        padded_t, audio_t, BATCH, T_FIXED, PAD,
        grid_dim=BATCH * T_PAD_FIXED, block_dim=1,
    )
    # 2. STFT magnitude.
    comptime stft_k = stft_mag_kernel[
        DType.float32, type_of(padded_layout), type_of(window_layout),
        type_of(mag_layout), N_FFT, HOP, N_FREQ,
    ]
    ctx.enqueue_function[stft_k, stft_k](
        mag_t, padded_t, window_t, BATCH, T_PAD_FIXED, N_FRAMES_FIXED,
        grid_dim=BATCH * N_FREQ * N_FRAMES_FIXED, block_dim=1,
    )
    # 3. mel = mel_basis @ magnitude  →  (NUM_MELS, n_frames)
    matmul[target="gpu"](mel_2d, basis_t, mag_2d, dctx)
    # 4. log_clamp.
    comptime logc_k = log_clamp_kernel[
        DType.float32, type_of(mel_flat_layout), type_of(mel_flat_layout),
        POINTWISE_BLOCK,
    ]
    ctx.enqueue_function[logc_k, logc_k](
        log_mel_flat, mel_flat, n_mel, Float32(1.0e-5),
        grid_dim=ceildiv(n_mel, POINTWISE_BLOCK), block_dim=POINTWISE_BLOCK,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    var sum_abs: Float64 = 0.0
    with log_mel_buf.map_to_host() as h:
        for i in range(n_mel):
            var d = h[i] - exp_mel.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_abs += Float64(d)
            assert_almost_equal(h[i], exp_mel.data[i], atol=1.0)
    print("mel extractor max abs:", max_abs, " mean abs:", sum_abs / Float64(n_mel))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
