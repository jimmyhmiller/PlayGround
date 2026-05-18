"""VE mel extractor parity test (16k, 40 bins, mel_power=2, no log, no normalize)."""
from std.math import cos, pi
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer

from fixture import load_fp32
from mel_extractor import reflect_pad_1d, stft_magnitude_power, mel_filter_apply


comptime B = 1
comptime L = 32000
comptime PAD = 200
comptime L_PADDED = L + 2 * PAD
comptime N_FFT = 400
comptime HOP = 160
comptime N_BINS = N_FFT // 2 + 1     # 201
comptime N_MEL = 40
comptime T_FRAMES = 1 + (L_PADDED - N_FFT) // HOP   # 201


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def hann_window_periodic(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(0.5 * (1.0 - cos(2.0 * Float32(pi) * Float32(i) / Float32(n))))
    return out^


def test_ve_mel() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "../mojo-t3/tests/fixtures/ve_mel/"
    var ctx = DeviceContext()

    var wav_t = load_fp32(fix + "wav.bin")
    var bank_t = load_fp32(fix + "bank.bin")
    var exp_t = load_fp32(fix + "mel.bin")

    var wav_buf = ctx.enqueue_create_buffer[DType.float32](B * L)
    var pad_buf = ctx.enqueue_create_buffer[DType.float32](B * L_PADDED)
    var win_buf = ctx.enqueue_create_buffer[DType.float32](N_FFT)
    var spec_buf = ctx.enqueue_create_buffer[DType.float32](B * N_BINS * T_FRAMES)
    var bank_buf = ctx.enqueue_create_buffer[DType.float32](N_MEL * N_BINS)
    var mel_buf = ctx.enqueue_create_buffer[DType.float32](B * T_FRAMES * N_MEL)

    upload(wav_buf, wav_t.data, B * L)
    upload(bank_buf, bank_t.data, N_MEL * N_BINS)
    var win = hann_window_periodic(N_FFT)
    upload(win_buf, win, N_FFT)

    reflect_pad_1d(ctx, wav_buf, pad_buf, B, L, PAD)
    stft_magnitude_power(ctx, pad_buf, win_buf, spec_buf,
                          B, L_PADDED, N_FFT, HOP, N_BINS, T_FRAMES, 2)
    mel_filter_apply(ctx, spec_buf, bank_buf, mel_buf,
                     B, N_BINS, N_MEL, T_FRAMES)
    ctx.synchronize()

    var n_out = B * T_FRAMES * N_MEL
    var max_abs: Float32 = 0.0
    with mel_buf.map_to_host() as h:
        for i in range(n_out):
            var d = h[i] - exp_t.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            if i < 4:
                print("mel[", i, "]: mojo=", h[i], " torch=", exp_t.data[i], " diff=", d)
    print("VE mel — max abs:", max_abs)
    assert_almost_equal(max_abs, 0.0, atol=2.0e-3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
