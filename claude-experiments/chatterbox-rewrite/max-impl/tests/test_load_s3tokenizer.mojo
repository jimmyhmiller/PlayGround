"""Smoke test: load real upstream S3TokenizerV2 weights from disk.

Validates structural completeness of `load_s3tokenizer()` against all 103
converted .bin files. Does NOT yet exercise forward pass — that comes after
this verifies all weights are present and shape-compatible.
"""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true
from std.gpu.host import DeviceContext

from weights import load_s3tokenizer


def test_load_s3tokenizer_real_weights() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    print("[s3t] loading from weights/s3t/...")
    var s3t = load_s3tokenizer(ctx, "weights/s3t")
    ctx.synchronize()
    print("[s3t] loaded:",
          "n_mels=", s3t.n_mels,
          "n_state=", s3t.n_state,
          "n_layers=", s3t.n_layers,
          "blocks=", len(s3t.blocks))
    assert_true(s3t.n_state == 1280, "n_state should be 1280")
    assert_true(s3t.n_layers == 6, "n_layers should be 6")
    assert_true(len(s3t.blocks) == 6, "should have 6 blocks loaded")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
