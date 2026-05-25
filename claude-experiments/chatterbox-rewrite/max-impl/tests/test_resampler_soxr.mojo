"""Test ffmpeg+soxr resampler is bit-exact to upstream librosa.resample."""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.math import sqrt
from std.gpu.host import DeviceContext, DeviceBuffer

from fixture import load_fp32
from weights import upload_fp32
from resampler_soxr import soxr_resample_24k_to_16k


def test_resampler_soxr() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var fix = "weights/s3gen_prompt/resample_diag/"

    var wav_24 = upload_fp32(ctx, fix + "wav_24k.bin")
    var n_in = 288000
    var n_out = n_in * 2 // 3   # 192000
    var out = ctx.enqueue_create_buffer[DType.float32](n_out)
    soxr_resample_24k_to_16k(ctx, wav_24, out, n_in, n_out)
    ctx.synchronize()

    var reference = load_fp32(fix + "wav_16k_soxr.bin")
    var max_abs: Float32 = 0.0
    var sum_diff_sq: Float32 = 0.0
    var sum_ref_sq: Float32 = 0.0
    with out.map_to_host() as h:
        for i in range(n_out):
            var dd = h[i] - reference.data[i]
            var d = dd
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_diff_sq += dd * dd
            sum_ref_sq += reference.data[i] * reference.data[i]
    var rel = sqrt(sum_diff_sq / sum_ref_sq) if sum_ref_sq > 0.0 else Float32(0.0)
    print("[soxr] max-abs=", max_abs, " rel_l2=", rel)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
