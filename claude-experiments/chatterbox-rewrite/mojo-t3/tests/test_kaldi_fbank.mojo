"""
Parity test for Kaldi fbank in pure Mojo.

Input:  ref_wav_16k.bin   (1, 160000) — 10s of 16kHz mono PCM as float32
Target: fbank_feat.bin    (1, 998, 80) — centered log-mel features

We verify Mojo's frame_preprocess + naive_rfft_power + mel_filterbank_log +
subtract_per_utterance_mean against torchaudio.compliance.kaldi.fbank +
per-utterance mean subtraction.
"""
from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import load_fp32
from kaldi_fbank import (
    frame_preprocess_kernel,
    naive_rfft_power_kernel,
    mel_filterbank_log_kernel,
    subtract_per_utterance_mean_kernel,
    povey_window_fp32,
    mel_filterbank_fp32,
)


comptime N_SAMPLES = 160000
comptime WINDOW_SIZE = 400
comptime WINDOW_SHIFT = 160
comptime PADDED = 512
comptime NUM_BINS = PADDED // 2 + 1   # 257
comptime NUM_MEL = 80
comptime SAMPLE_RATE: Float32 = 16000.0
comptime LOW_FREQ: Float32 = 20.0
comptime HIGH_FREQ: Float32 = 0.0     # 0 => nyquist
comptime PREEMPHASIS: Float32 = 0.97
comptime EPS: Float32 = 1.1920929e-7
comptime BLOCK = 256

# Number of frames with snip_edges=True.
comptime M = 1 + (N_SAMPLES - WINDOW_SIZE) // WINDOW_SHIFT   # = 998


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def test_kaldi_fbank() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/campplus/"
    var ctx = DeviceContext()

    var wav = load_fp32(fix + "ref_wav_16k.bin")    # (1, 160000)
    var exp = load_fp32(fix + "fbank_feat.bin")     # (1, 998, 80) after subtract_mean

    var n_w = N_SAMPLES
    var n_frames = M * PADDED
    var n_power = M * NUM_BINS
    var n_mel = M * NUM_MEL

    # ---- Host: precompute Povey window and mel filterbank.
    var win = povey_window_fp32(WINDOW_SIZE)
    var bank = mel_filterbank_fp32(NUM_MEL, PADDED, SAMPLE_RATE, LOW_FREQ, HIGH_FREQ)

    var wav_buf = ctx.enqueue_create_buffer[DType.float32](n_w)
    var win_buf = ctx.enqueue_create_buffer[DType.float32](WINDOW_SIZE)
    var bank_buf = ctx.enqueue_create_buffer[DType.float32](NUM_MEL * NUM_BINS)
    var frames_buf = ctx.enqueue_create_buffer[DType.float32](n_frames)
    var power_buf = ctx.enqueue_create_buffer[DType.float32](n_power)
    var mel_buf = ctx.enqueue_create_buffer[DType.float32](n_mel)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_mel)

    upload(wav_buf, wav.data, n_w)
    upload(win_buf, win, WINDOW_SIZE)
    upload(bank_buf, bank, NUM_MEL * NUM_BINS)

    comptime wav_layout = row_major[N_SAMPLES]()
    comptime win_layout = row_major[WINDOW_SIZE]()
    comptime bank_layout = row_major[NUM_MEL, NUM_BINS]()
    comptime frames_layout = row_major[M, PADDED]()
    comptime power_layout = row_major[M, NUM_BINS]()
    comptime mel_layout = row_major[M, NUM_MEL]()

    var wav_t = TileTensor(wav_buf, wav_layout)
    var win_t = TileTensor(win_buf, win_layout)
    var bank_t = TileTensor(bank_buf, bank_layout)
    var frames_t = TileTensor(frames_buf, frames_layout)
    var power_t = TileTensor(power_buf, power_layout)
    var mel_t = TileTensor(mel_buf, mel_layout)
    var out_t = TileTensor(out_buf, mel_layout)

    # ---- Kernel 1: per-frame preprocess.
    comptime pre_k = frame_preprocess_kernel[
        DType.float32, type_of(wav_layout), type_of(frames_layout),
        type_of(win_layout),
        WINDOW_SIZE, PADDED, BLOCK,
    ]
    ctx.enqueue_function[pre_k, pre_k](
        frames_t, wav_t, win_t, M, WINDOW_SHIFT, PREEMPHASIS,
        grid_dim=M, block_dim=BLOCK,
    )

    # ---- Kernel 2: naive rFFT power spectrum.
    comptime fft_k = naive_rfft_power_kernel[
        DType.float32, type_of(frames_layout), type_of(power_layout),
        PADDED, NUM_BINS, BLOCK,
    ]
    ctx.enqueue_function[fft_k, fft_k](
        power_t, frames_t, M,
        grid_dim=M, block_dim=BLOCK,
    )

    # ---- Kernel 3: mel filterbank + log.
    comptime mel_k = mel_filterbank_log_kernel[
        DType.float32, type_of(power_layout), type_of(bank_layout), type_of(mel_layout),
        NUM_BINS, NUM_MEL, BLOCK,
    ]
    ctx.enqueue_function[mel_k, mel_k](
        mel_t, power_t, bank_t, M, EPS,
        grid_dim=M, block_dim=BLOCK,
    )

    # ---- Kernel 4: per-utterance mean subtraction.
    comptime sub_k = subtract_per_utterance_mean_kernel[
        DType.float32, type_of(mel_layout), type_of(mel_layout),
        NUM_MEL, BLOCK,
    ]
    ctx.enqueue_function[sub_k, sub_k](
        out_t, mel_t, M,
        grid_dim=NUM_MEL, block_dim=BLOCK,
    )
    ctx.synchronize()

    var n_total = n_mel
    var max_abs: Float32 = 0.0
    var sum_abs: Float64 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n_total):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_abs += Float64(d)
            if i < 8:
                print("fbank[", i, "]: mojo=", h[i], "  torch=", exp.data[i], "  diff=", d)
            assert_almost_equal(h[i], exp.data[i], atol=5.0e-2)
    print("Kaldi fbank — max abs:", max_abs, " mean:", sum_abs / Float64(n_total))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
