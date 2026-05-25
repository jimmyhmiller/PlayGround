"""s3tokenizer log-mel parity vs upstream."""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.math import sqrt
from std.gpu.host import DeviceContext, DeviceBuffer

from fixture import load_fp32
from weights import upload_fp32
from mel_s3tok import (
    log_mel_s3tok_forward, build_hann_window_full, build_librosa_mel_filterbank_s3tok,
)


def _diff(name: String, mut mojo: DeviceBuffer[DType.float32], ref_path: String) raises:
    var reference = load_fp32(ref_path)
    var ref_n = reference.numel()
    var max_abs: Float32 = 0.0
    var sum_diff_sq: Float32 = 0.0
    var sum_ref_sq: Float32 = 0.0
    with mojo.map_to_host() as h:
        for i in range(ref_n):
            var dd = h[i] - reference.data[i]
            var d = dd
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_diff_sq += dd * dd
            sum_ref_sq += reference.data[i] * reference.data[i]
    var rel = sqrt(sum_diff_sq / sum_ref_sq) if sum_ref_sq > 0.0 else Float32(0.0)
    print("[mel-s3tok]", name, ": max-abs=", max_abs, " rel_l2=", rel, " (n=", ref_n, ")")


def test_mel_s3tok_parity() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var fix = "weights/s3gen_prompt/s3tok_diag/"

    var wav = upload_fp32(ctx, fix + "wav_16k.bin")
    var n_samples = 192000
    var n_fft = 400
    var hop = 160
    var pad = n_fft // 2
    var n_padded = n_samples + 2 * pad
    var t_stft = (n_padded - n_fft) // hop + 1
    var t_used = t_stft - 1   # upstream drops last frame
    print("[mel-s3tok] t_used=", t_used)

    var window = ctx.enqueue_create_buffer[DType.float32](n_fft)
    build_hann_window_full(ctx, window, n_fft)

    var n_bins = n_fft // 2 + 1
    var mel_fb = ctx.enqueue_create_buffer[DType.float32](128 * n_bins)
    build_librosa_mel_filterbank_s3tok(ctx, mel_fb, 128, n_fft, Float64(16000.0))
    ctx.synchronize()

    var out = ctx.enqueue_create_buffer[DType.float32](128 * t_used)
    log_mel_s3tok_forward(ctx, wav, window, mel_fb, out, n_samples, t_used)
    ctx.synchronize()
    _diff("log_mel", out, fix + "log_mel_16k.bin")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
