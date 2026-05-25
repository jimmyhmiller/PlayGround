"""Smoke test: load real upstream HiFTGenerator (NSF-HiFiGAN) weights.

Validates structural completeness of `load_hift_generator()` — all 6 weight
groups (conv_pre, ups×3, resblocks×9, source_downs×3, source_resblocks×3,
conv_post) + m_source + f0_predictor populate without missing keys.
"""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true
from std.gpu.host import DeviceContext

from weights import load_hift_generator


def test_load_hift_real_weights() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    print("[hift] loading from weights/s3gen/mel2wav/...")
    var hift = load_hift_generator(ctx, "weights/s3gen/mel2wav")
    ctx.synchronize()
    print("[hift] loaded: ups=", len(hift.ups),
          " resblocks=", len(hift.resblocks),
          " source_downs=", len(hift.source_downs),
          " source_resblocks=", len(hift.source_resblocks))
    assert_true(len(hift.ups) == 3, "should have 3 ups stages")
    assert_true(len(hift.resblocks) == 9, "should have 9 MRF resblocks total")
    assert_true(len(hift.source_downs) == 3, "should have 3 source_downs")
    assert_true(len(hift.source_resblocks) == 3, "should have 3 source_resblocks")
    assert_true(hift.n_fft == 16, "n_fft should be 16")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
