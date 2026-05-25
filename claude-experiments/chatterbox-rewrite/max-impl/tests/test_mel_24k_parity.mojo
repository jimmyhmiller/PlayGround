"""24kHz mel parity vs upstream."""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.math import sqrt
from std.gpu.host import DeviceContext, DeviceBuffer

from fixture import load_fp32
from weights import upload_fp32
from mel_24k import (
    mel_24k_forward, build_hann_window, build_librosa_mel_filterbank,
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
    print("[mel24k]", name, ": max-abs=", max_abs, " rel_l2=", rel, " (n=", ref_n, ")")


def test_mel_24k() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var fix = "weights/s3gen_prompt/mel24k_diag/"

    # Wav has shape (24000,). After pad reflect 720 each side: 24000+1440=25440.
    # With n_fft=1920, hop=480, T_frames = (25440 - 1920) // 480 + 1 = 50.
    var n_samples = 24000
    var n_fft = 1920
    var hop = 480
    var pad = (n_fft - hop) // 2   # 720
    var n_padded = n_samples + 2 * pad
    var T = (n_padded - n_fft) // hop + 1
    print("[mel24k] T_frames=", T)

    var wav = upload_fp32(ctx, fix + "wav_24k.bin")

    var window = ctx.enqueue_create_buffer[DType.float32](n_fft)
    build_hann_window(ctx, window, n_fft)

    var n_bins = n_fft // 2 + 1
    var mel_fb = ctx.enqueue_create_buffer[DType.float32](80 * n_bins)
    build_librosa_mel_filterbank(ctx, mel_fb, 80, n_fft,
                                  Float64(24000.0), Float64(0.0), Float64(8000.0))
    ctx.synchronize()

    var out = ctx.enqueue_create_buffer[DType.float32](80 * T)
    mel_24k_forward(ctx, wav, window, mel_fb, out, n_samples, T,
                     n_fft=n_fft, hop=hop, n_mels=80)
    ctx.synchronize()

    _diff("mel_24k", out, fix + "mel_24k.bin")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
