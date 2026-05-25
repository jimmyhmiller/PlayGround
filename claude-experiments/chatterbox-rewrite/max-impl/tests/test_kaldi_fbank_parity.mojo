"""Kaldi fbank parity test."""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.math import sqrt
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr

from fixture import load_fp32
from weights import upload_fp32
from kaldi_fbank import (
    kaldi_fbank_forward, kaldi_subtract_column_mean,
    build_povey_window, build_kaldi_mel_filterbank,
)


def _diff(name: String, mut mojo: DeviceBuffer[DType.float32], ref_path: String) raises:
    var reference = load_fp32(ref_path)
    var ref_n = reference.numel()
    var max_abs: Float32 = 0.0
    var max_abs_idx: Int = -1
    var sum_diff_sq: Float32 = 0.0
    var sum_ref_sq: Float32 = 0.0
    with mojo.map_to_host() as h:
        for i in range(ref_n):
            var dd = h[i] - reference.data[i]
            var d = dd
            if d < 0.0: d = -d
            if d > max_abs:
                max_abs = d
                max_abs_idx = i
            sum_diff_sq += dd * dd
            sum_ref_sq += reference.data[i] * reference.data[i]
    var rel = sqrt(sum_diff_sq / sum_ref_sq) if sum_ref_sq > 0.0 else Float32(0.0)
    print("[kaldi]", name, ": max-abs=", max_abs, " (i=", max_abs_idx, ") rel_l2=", rel, " (n=", ref_n, ")")


def test_kaldi_fbank() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var fix = "weights/s3gen_prompt/kaldi_diag/"

    # Wav has shape (16240,). Frames: snip_edges → (16240 - 400) // 160 + 1 = 100.
    var n_samples = 16240
    var t_frames = (n_samples - 400) // 160 + 1   # 99? or 100? Let me see: (16240-400)/160 = 99
    var T = (n_samples - 400) // 160 + 1
    print("[kaldi] computed T_frames=", T)

    var wav = upload_fp32(ctx, fix + "wav.bin")

    var window = ctx.enqueue_create_buffer[DType.float32](400)
    build_povey_window(ctx, window, 400)
    ctx.synchronize()

    var num_bins = 80
    var pad_ws = 512
    var n_fft_bins = pad_ws // 2 + 1
    var mel_fb = ctx.enqueue_create_buffer[DType.float32](num_bins * n_fft_bins)
    build_kaldi_mel_filterbank(ctx, mel_fb, num_bins, pad_ws, Float64(16000.0), Float64(20.0), Float64(0.0))
    ctx.synchronize()

    var out = ctx.enqueue_create_buffer[DType.float32](T * num_bins)
    kaldi_fbank_forward(ctx, wav, window, mel_fb, out, n_samples, T)
    ctx.synchronize()

    _diff("fbank_raw", out, fix + "fbank_raw.bin")

    # Apply per-column mean subtraction.
    kaldi_subtract_column_mean(ctx, out, T, num_bins)
    ctx.synchronize()
    _diff("fbank_after_mean", out, fix + "fbank_after_mean.bin")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
