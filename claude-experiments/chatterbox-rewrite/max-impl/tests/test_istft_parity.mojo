"""Compare Mojo iSTFT vs upstream torch.istft. Feed upstream's exact conv_post_out
(which is the iSTFT input in mag/phase form) and compare audio output."""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.math import sqrt
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr

from fixture import load_fp32
from weights import upload_fp32
from hift_generator import istft_forward, hann_window_periodic_fill


def _diff(name: String, mut mojo: DeviceBuffer[DType.float32], ref_path: String) raises:
    var reference = load_fp32(ref_path)
    var ref_n = reference.numel()
    var max_abs: Float32 = 0.0
    var max_abs_idx: Int = -1
    var sum_diff_sq: Float32 = 0.0
    var sum_ref_sq: Float32 = 0.0
    var first_count = 0
    with mojo.map_to_host() as h:
        for i in range(ref_n):
            var dd = h[i] - reference.data[i]
            var d = dd
            if d < 0.0: d = -d
            if d > max_abs:
                max_abs = d
                max_abs_idx = i
            if first_count < 10 and i < 32:
                print("  i=", i, " mojo=", h[i], " ref=", reference.data[i], " diff=", dd)
                first_count += 1
            sum_diff_sq += dd * dd
            sum_ref_sq += reference.data[i] * reference.data[i]
    var rel = sqrt(sum_diff_sq / sum_ref_sq) if sum_ref_sq > 0.0 else Float32(0.0)
    print("[parity]", name, ": max-abs=", max_abs, " (at i=", max_abs_idx, ") rel_l2=", rel, " (n=", ref_n, ")")


def test_istft_parity() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var fix = "weights/s3gen_prompt/hift_dump/"

    var B = 1
    var N_FFT = 16
    var T_FRAMES = 12241
    var T_AUDIO = 48960

    # Upload upstream's exact conv_post_out (bit-exact match with Mojo's anyway).
    var spec = upload_fp32(ctx, fix + "conv_post_out.bin")
    var window = ctx.enqueue_create_buffer[DType.float32](N_FFT)
    hann_window_periodic_fill(ctx, window, N_FFT)
    var audio = ctx.enqueue_create_buffer[DType.float32](B * T_AUDIO)

    istft_forward(ctx, spec, window, audio, B, N_FFT, T_FRAMES, T_AUDIO)
    ctx.synchronize()
    _diff("istft_out", audio, fix + "istft_out.bin")
    _diff("audio (vs upstream final)", audio, fix + "audio.bin")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
