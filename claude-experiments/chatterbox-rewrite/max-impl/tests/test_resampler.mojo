"""Test Mojo polyphase resampler 24k → 16k against librosa output."""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.math import sqrt
from std.gpu.host import DeviceContext, DeviceBuffer

from fixture import load_fp32
from weights import upload_fp32
from resampler import resample_24k_to_16k


def _diff(name: String, mut mojo: DeviceBuffer[DType.float32], ref_path: String) raises:
    var reference = load_fp32(ref_path)
    var ref_n = reference.numel()
    var max_abs: Float32 = 0.0
    var sum_diff_sq: Float32 = 0.0
    var sum_ref_sq: Float32 = 0.0
    var mojo_peak: Float32 = 0.0
    var ref_peak: Float32 = 0.0
    with mojo.map_to_host() as h:
        for i in range(ref_n):
            var dd = h[i] - reference.data[i]
            var d = dd
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_diff_sq += dd * dd
            sum_ref_sq += reference.data[i] * reference.data[i]
            var av = h[i]
            if av < 0.0: av = -av
            if av > mojo_peak: mojo_peak = av
            var ar = reference.data[i]
            if ar < 0.0: ar = -ar
            if ar > ref_peak: ref_peak = ar
    var rel = sqrt(sum_diff_sq / sum_ref_sq) if sum_ref_sq > 0.0 else Float32(0.0)
    print("[resample]", name, ": max-abs=", max_abs, " rel_l2=", rel,
          " mojo_peak=", mojo_peak, " ref_peak=", ref_peak)


def test_resampler() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var fix = "weights/s3gen_prompt/resample_diag/"

    var wav_24 = upload_fp32(ctx, fix + "wav_24k.bin")
    # 24k → 16k: ratio 2/3. For N_in=288000, N_out=192000.
    var n_in = 288000
    var n_out = 192000
    var out = ctx.enqueue_create_buffer[DType.float32](n_out)
    resample_24k_to_16k(ctx, wav_24, out, n_in, n_out)
    ctx.synchronize()

    _diff("vs librosa soxr_hq", out, fix + "wav_16k_soxr.bin")
    _diff("vs librosa scipy",    out, fix + "wav_16k_scipy.bin")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
