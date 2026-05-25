"""FCM (feature compression module) parity test vs upstream."""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.math import sqrt
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr

from fixture import load_fp32
from weights import load_fcm, upload_fp32
from fcm import fcm_forward


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
    print("[fcm-parity]", name, ": max-abs=", max_abs, " rel_l2=", rel, " (n=", ref_n, ")")


def test_fcm_parity() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var fix = "weights/s3gen_prompt/fcm_diag/"

    print("[fcm-parity] loading FCM...")
    var fcm = load_fcm(ctx, "weights/s3gen/speaker_encoder/head")

    var mel = upload_fp32(ctx, fix + "fcm_input.bin")
    var B = 1
    var T = 64
    var out = ctx.enqueue_create_buffer[DType.float32](B * 320 * T)

    print("[fcm-parity] running FCM forward...")
    fcm_forward(ctx, fcm, mel, out, B, T)
    ctx.synchronize()

    _diff("fcm_output", out, fix + "fcm_output.bin")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
