"""
Parity test for mel_spectrogram_24k Mojo port.

Input:  ref_wav_24k.bin     (1, 240000) — 10s at 24kHz
        mel_basis_24k.bin   (80, 961) — librosa mel filterbank
Target: prompt_feat.bin     (1, 500, 80)
"""
from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import load_fp32
from mel_24k import (
    hann_window_n1920, reflect_pad_kernel, stft_24k_magnitude_kernel,
    mel_filter_log_kernel, transpose_bct_to_btc_2d_kernel,
)


comptime B = 1
comptime L = 240000
comptime N_FFT = 1920
comptime HOP = 480
comptime PAD = (N_FFT - HOP) // 2   # 720
comptime L_PADDED = L + 2 * PAD     # 241440
comptime N_BINS = N_FFT // 2 + 1    # 961
comptime N_MEL = 80
comptime T_FRAMES = (L_PADDED - N_FFT) // HOP + 1   # 500
comptime BLOCK = 256


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def test_mel_24k() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var wav = load_fp32("tests/fixtures/s3gen/ref_wav_24k.bin")
    var mel_basis = load_fp32("tests/fixtures/s3gen/mel_basis_24k.bin")
    var exp = load_fp32("tests/fixtures/s3gen/prompt_feat.bin")

    var win = hann_window_n1920()

    var wav_buf = ctx.enqueue_create_buffer[DType.float32](B * L)
    var pad_buf = ctx.enqueue_create_buffer[DType.float32](B * L_PADDED)
    var win_buf = ctx.enqueue_create_buffer[DType.float32](N_FFT)
    var mag_buf = ctx.enqueue_create_buffer[DType.float32](B * N_BINS * T_FRAMES)
    var bank_buf = ctx.enqueue_create_buffer[DType.float32](N_MEL * N_BINS)
    var log_mel_bct = ctx.enqueue_create_buffer[DType.float32](B * N_MEL * T_FRAMES)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](B * T_FRAMES * N_MEL)

    upload(wav_buf, wav.data, B * L)
    upload(win_buf, win, N_FFT)
    upload(bank_buf, mel_basis.data, N_MEL * N_BINS)

    comptime wav_layout = row_major[B, L]()
    comptime pad_layout = row_major[B, L_PADDED]()
    comptime win_layout = row_major[N_FFT]()
    comptime mag_layout = row_major[B, N_BINS, T_FRAMES]()
    comptime bank_layout = row_major[N_MEL, N_BINS]()
    comptime log_mel_bct_layout = row_major[B, N_MEL, T_FRAMES]()
    comptime out_layout = row_major[B, T_FRAMES, N_MEL]()

    var wav_t = TileTensor(wav_buf, wav_layout)
    var pad_t = TileTensor(pad_buf, pad_layout)
    var win_t = TileTensor(win_buf, win_layout)
    var mag_t = TileTensor(mag_buf, mag_layout)
    var bank_t = TileTensor(bank_buf, bank_layout)
    var log_mel_bct_t = TileTensor(log_mel_bct, log_mel_bct_layout)
    var out_t = TileTensor(out_buf, out_layout)

    # 1. Reflect pad.
    comptime pad_k = reflect_pad_kernel[
        DType.float32, type_of(wav_layout), type_of(pad_layout), BLOCK,
    ]
    ctx.enqueue_function[pad_k, pad_k](
        pad_t, wav_t, B, L, PAD,
        grid_dim=B, block_dim=BLOCK,
    )

    # 2. STFT → magnitude.
    comptime stft_k = stft_24k_magnitude_kernel[
        DType.float32, type_of(pad_layout), type_of(win_layout), type_of(mag_layout),
        N_FFT, HOP, N_BINS, BLOCK,
    ]
    ctx.enqueue_function[stft_k, stft_k](
        mag_t, pad_t, win_t, B, L_PADDED, T_FRAMES,
        grid_dim=B * T_FRAMES, block_dim=BLOCK,
    )

    # 3. Mel filterbank + log.
    comptime mel_k = mel_filter_log_kernel[
        DType.float32, type_of(mag_layout), type_of(bank_layout), type_of(log_mel_bct_layout),
        N_BINS, N_MEL, BLOCK,
    ]
    ctx.enqueue_function[mel_k, mel_k](
        log_mel_bct_t, mag_t, bank_t, B, T_FRAMES, Float32(1.0e-5),
        grid_dim=B * T_FRAMES, block_dim=BLOCK,
    )

    # 4. Transpose (B, 80, T) → (B, T, 80).
    comptime tp_k = transpose_bct_to_btc_2d_kernel[
        DType.float32, type_of(log_mel_bct_layout), type_of(out_layout), BLOCK,
    ]
    ctx.enqueue_function[tp_k, tp_k](
        out_t, log_mel_bct_t, B, N_MEL, T_FRAMES,
        grid_dim=B * T_FRAMES, block_dim=BLOCK,
    )

    ctx.synchronize()

    var n_out = B * T_FRAMES * N_MEL
    var max_abs: Float32 = 0.0
    var sum_abs: Float64 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n_out):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_abs += Float64(d)
            if i < 8:
                print("pf[", i, "]: mojo=", h[i], "  torch=", exp.data[i], "  diff=", d)
            assert_almost_equal(h[i], exp.data[i], atol=1.0)
    print("mel_spectrogram_24k — max abs:", max_abs,
          " mean:", sum_abs / Float64(n_out))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
