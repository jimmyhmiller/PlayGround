"""Parity test for VE mel extractor (reflect-pad → STFT power → mel filter).

Compares Mojo output against the numpy reference dumped by
oracle/dump_ve_mel_case.py for 2 seconds of the default-voice reference
audio resampled to 16 kHz.
"""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import load_fp32
from mel_24k import reflect_pad_kernel
from ve_mel import hann_window, stft_power_kernel, mel_filter_amp_kernel


comptime B = 1
comptime L = 32000
comptime PAD = 200
comptime L_PADDED = L + 2 * PAD
comptime N_FFT = 400
comptime HOP = 160
comptime N_BINS = N_FFT // 2 + 1     # 201
comptime N_MEL = 40
comptime T_FRAMES = 1 + (L_PADDED - N_FFT) // HOP   # 201
comptime BLOCK = 128


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def test_ve_mel() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/ve_mel/"
    var ctx = DeviceContext()

    var wav = load_fp32(fix + "wav.bin")
    var bank = load_fp32(fix + "bank.bin")
    var exp = load_fp32(fix + "mel.bin")

    var wav_buf = ctx.enqueue_create_buffer[DType.float32](B * L)
    var pad_buf = ctx.enqueue_create_buffer[DType.float32](B * L_PADDED)
    var spec_buf = ctx.enqueue_create_buffer[DType.float32](B * N_BINS * T_FRAMES)
    var mel_buf = ctx.enqueue_create_buffer[DType.float32](B * T_FRAMES * N_MEL)
    var win_buf = ctx.enqueue_create_buffer[DType.float32](N_FFT)
    var bank_buf = ctx.enqueue_create_buffer[DType.float32](N_MEL * N_BINS)

    upload(wav_buf, wav.data, B * L)
    upload(bank_buf, bank.data, N_MEL * N_BINS)
    var win = hann_window(N_FFT)
    upload(win_buf, win, N_FFT)

    comptime wav_layout = row_major[B, L]()
    comptime pad_layout = row_major[B, L_PADDED]()
    comptime spec_layout = row_major[B, N_BINS, T_FRAMES]()
    comptime mel_layout = row_major[B, T_FRAMES, N_MEL]()
    comptime win_layout = row_major[N_FFT]()
    comptime bank_layout = row_major[N_MEL, N_BINS]()

    var wav_t = TileTensor(wav_buf, wav_layout)
    var pad_t = TileTensor(pad_buf, pad_layout)
    var spec_t = TileTensor(spec_buf, spec_layout)
    var mel_t = TileTensor(mel_buf, mel_layout)
    var win_t = TileTensor(win_buf, win_layout)
    var bank_t = TileTensor(bank_buf, bank_layout)

    # 1. reflect-pad.
    comptime kp = reflect_pad_kernel[
        DType.float32, type_of(wav_layout), type_of(pad_layout), BLOCK,
    ]
    ctx.enqueue_function[kp, kp](
        pad_t, wav_t, B, L, PAD,
        grid_dim=B, block_dim=BLOCK,
    )

    # 2. STFT power = re^2 + im^2 (mel_power=2).
    comptime ks = stft_power_kernel[
        DType.float32, type_of(pad_layout), type_of(win_layout),
        type_of(spec_layout),
        N_FFT, HOP, N_BINS, BLOCK,
    ]
    ctx.enqueue_function[ks, ks](
        spec_t, pad_t, win_t,
        B, L_PADDED, T_FRAMES,
        grid_dim=B * T_FRAMES, block_dim=BLOCK,
    )

    # 3. Mel filterbank — output is (B, T, N_MEL) layout.
    comptime km = mel_filter_amp_kernel[
        DType.float32, type_of(spec_layout), type_of(bank_layout),
        type_of(mel_layout),
        N_BINS, N_MEL, BLOCK,
    ]
    ctx.enqueue_function[km, km](
        mel_t, spec_t, bank_t,
        B, T_FRAMES,
        grid_dim=B * T_FRAMES, block_dim=BLOCK,
    )
    ctx.synchronize()

    var n_out = B * T_FRAMES * N_MEL
    var max_abs: Float32 = 0.0
    var max_rel: Float32 = 0.0
    var max_rel_t: Float32 = 0.0
    with mel_buf.map_to_host() as h:
        for i in range(n_out):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            # relative error
            var r0 = exp.data[i]
            if r0 < 0.0: r0 = -r0
            if r0 > 1.0e-3:
                var r = d / r0
                if r > max_rel:
                    max_rel = r
                    max_rel_t = exp.data[i]
            if i < 8:
                print("ve_mel[", i, "]: mojo=", h[i], "  torch=", exp.data[i], "  diff=", d)
    print("VE mel — max abs:", max_abs, "  max_rel:", max_rel, "  at val:", max_rel_t)
    assert_almost_equal(max_abs, 0.0, atol=2.0e-3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
