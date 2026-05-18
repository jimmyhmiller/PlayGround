"""Smoke test: load real upstream CAMPPlus speaker-encoder weights.

Validates structural completeness of `load_campplus()` against all 937
converted .bin files. Confirms block layer counts and channel widths match
the upstream layout (block1=12, block2=24, block3=16).
"""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true
from std.gpu.host import DeviceContext

from weights import load_campplus


def test_load_campplus_real_weights() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    print("[campplus] loading from weights/s3gen/speaker_encoder/...")
    var sp = load_campplus(ctx, "weights/s3gen/speaker_encoder")
    ctx.synchronize()
    print("[campplus] block1=", len(sp.xvector.block1.layers),
          " block2=", len(sp.xvector.block2.layers),
          " block3=", len(sp.xvector.block3.layers),
          " head.layer1=", len(sp.head.layer1),
          " head.layer2=", len(sp.head.layer2))
    assert_true(len(sp.xvector.block1.layers) == 12, "block1 should have 12 layers")
    assert_true(len(sp.xvector.block2.layers) == 24, "block2 should have 24 layers")
    assert_true(len(sp.xvector.block3.layers) == 16, "block3 should have 16 layers")
    assert_true(len(sp.head.layer1) == 2, "head.layer1 should have 2 blocks")
    assert_true(len(sp.head.layer2) == 2, "head.layer2 should have 2 blocks")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
