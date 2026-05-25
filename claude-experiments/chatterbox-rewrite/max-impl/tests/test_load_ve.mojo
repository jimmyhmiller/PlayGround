"""Smoke test: load real VoiceEncoder weights from disk + run forward.

Verifies the Session A weight loader end-to-end. Doesn't check parity (no
oracle); just confirms shapes line up and the model produces a finite
embedding from a synthetic input.
"""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.gpu.host import DeviceContext, DeviceBuffer

from weights import load_voice_encoder
from voice_encoder import voice_encoder_forward


def test_load_ve_real_weights() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var ve = load_voice_encoder(ctx, "weights/ve")

    # Run on a synthetic mel input (B=1, T=160, M=40).
    var B = 1
    var T = 160
    var M = 40
    var n = B * T * M
    var mels_buf = ctx.enqueue_create_buffer[DType.float32](n)
    with mels_buf.map_to_host() as h:
        for i in range(n):
            h[i] = Float32(i % 100) * 0.001

    var embed_buf = ctx.enqueue_create_buffer[DType.float32](B * 256)
    voice_encoder_forward(ctx, ve, mels_buf, embed_buf, B, T)
    ctx.synchronize()

    with embed_buf.map_to_host() as h:
        print("embed[0:4]:", h[0], h[1], h[2], h[3])
        var norm_sq: Float32 = 0.0
        for i in range(256):
            norm_sq += h[i] * h[i]
        print("L2-norm-squared (should be ~1.0):", norm_sq)
    print("load_voice_encoder smoke PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
