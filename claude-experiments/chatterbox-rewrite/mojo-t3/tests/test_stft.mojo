"""
STFT / iSTFT parity tests vs torch.stft / torch.istft for n_fft=16, hop=4.
"""

from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import Tensor, TensorI64, load_fp32, load_i64
from stft import stft_kernel, istft_kernel


comptime N_FFT = 16
comptime HOP = 4
comptime N_FREQ = N_FFT // 2 + 1   # 9


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def test_stft_fp32() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/stft/"
    var x = load_fp32(fix + "x.bin")
    var win = load_fp32(fix + "window.bin")
    var exp_real = load_fp32(fix + "real.bin")
    var exp_imag = load_fp32(fix + "imag.bin")
    var meta = load_i64(fix + "meta.bin")

    var B = Int(meta.data[0])
    var T = Int(meta.data[1])
    assert_equal(Int(meta.data[2]), N_FFT)
    assert_equal(Int(meta.data[3]), HOP)
    assert_equal(Int(meta.data[4]), N_FREQ)
    var n_frames = Int(meta.data[5])
    assert_equal(B, 1)
    assert_equal(T, 64)

    var n_x = B * T
    var n_spec = B * N_FREQ * n_frames

    var ctx = DeviceContext()
    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var win_buf = ctx.enqueue_create_buffer[DType.float32](N_FFT)
    var real_buf = ctx.enqueue_create_buffer[DType.float32](n_spec)
    var imag_buf = ctx.enqueue_create_buffer[DType.float32](n_spec)
    upload(x_buf, x.data, n_x)
    upload(win_buf, win.data, N_FFT)

    comptime x_layout = row_major[1, 64]()
    comptime win_layout = row_major[N_FFT]()
    # We instantiate layout with the runtime n_frames=17 hard-coded as comptime,
    # which is fine because we control the oracle.
    comptime spec_layout = row_major[1, N_FREQ, 17]()

    var x_t = TileTensor(x_buf, x_layout)
    var win_t = TileTensor(win_buf, win_layout)
    var real_t = TileTensor(real_buf, spec_layout)
    var imag_t = TileTensor(imag_buf, spec_layout)

    comptime kernel = stft_kernel[
        DType.float32, type_of(x_layout), type_of(win_layout),
        type_of(spec_layout), type_of(spec_layout),
        N_FFT, HOP, N_FREQ,
    ]
    ctx.enqueue_function[kernel, kernel](
        real_t, imag_t, x_t, win_t,
        B, T, n_frames,
        grid_dim=B * N_FREQ * n_frames, block_dim=1,
    )
    ctx.synchronize()

    var max_re: Float32 = 0.0
    var max_im: Float32 = 0.0
    with real_buf.map_to_host() as h:
        for i in range(n_spec):
            var d = h[i] - exp_real.data[i]
            if d < 0.0: d = -d
            if d > max_re: max_re = d
            assert_almost_equal(h[i], exp_real.data[i], atol=2.0e-6)
    with imag_buf.map_to_host() as h:
        for i in range(n_spec):
            var d = h[i] - exp_imag.data[i]
            if d < 0.0: d = -d
            if d > max_im: max_im = d
            assert_almost_equal(h[i], exp_imag.data[i], atol=2.0e-6)
    print("stft fp32 — max abs real:", max_re, " imag:", max_im)


def test_istft_fp32() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var fix = "tests/fixtures/stft/"
    var real_in = load_fp32(fix + "real.bin")
    var imag_in = load_fp32(fix + "imag.bin")
    var win = load_fp32(fix + "window.bin")
    var exp = load_fp32(fix + "istft_expected.bin")
    var meta = load_i64(fix + "meta.bin")

    var B = Int(meta.data[0])
    var T = Int(meta.data[1])
    var n_frames = Int(meta.data[5])

    var n_x = B * T
    var n_spec = B * N_FREQ * n_frames

    var ctx = DeviceContext()
    var real_buf = ctx.enqueue_create_buffer[DType.float32](n_spec)
    var imag_buf = ctx.enqueue_create_buffer[DType.float32](n_spec)
    var win_buf = ctx.enqueue_create_buffer[DType.float32](N_FFT)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    upload(real_buf, real_in.data, n_spec)
    upload(imag_buf, imag_in.data, n_spec)
    upload(win_buf, win.data, N_FFT)

    comptime out_layout = row_major[1, 64]()
    comptime win_layout = row_major[N_FFT]()
    comptime spec_layout = row_major[1, N_FREQ, 17]()

    var real_t = TileTensor(real_buf, spec_layout)
    var imag_t = TileTensor(imag_buf, spec_layout)
    var win_t = TileTensor(win_buf, win_layout)
    var out_t = TileTensor(out_buf, out_layout)

    comptime kernel = istft_kernel[
        DType.float32, type_of(spec_layout), type_of(spec_layout),
        type_of(win_layout), type_of(out_layout),
        N_FFT, HOP, N_FREQ,
    ]
    ctx.enqueue_function[kernel, kernel](
        out_t, real_t, imag_t, win_t,
        B, n_frames, T,
        grid_dim=B * T, block_dim=1,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n_x):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-5)
    print("istft fp32 — max abs:", max_abs)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
