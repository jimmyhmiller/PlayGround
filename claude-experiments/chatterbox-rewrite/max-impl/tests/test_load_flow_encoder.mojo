"""Smoke test: load real upstream s3gen flow encoder layers.

Validates structural completeness against the .bin layout produced by
`convert_weights.py --s3gen`. Loads all 6 `encoders/` + 4 `up_encoders/`.
"""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true
from std.gpu.host import DeviceContext

from weights import load_s3gen_flow_encoder_layers


def test_load_flow_encoders_real_weights() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    # Layer config inferred from upstream shapes:
    #   d_model=512, intermediate=2048, n_heads=8, head_dim=64.
    var D = 512
    var INT = 2048
    var H = 8
    var DH = 64

    print("[flow-enc] loading 6 layers from weights/s3gen/flow/encoder/encoders/...")
    var pre_layers = load_s3gen_flow_encoder_layers(
        ctx, "weights/s3gen/flow/encoder/encoders", 6, D, INT, H, DH,
    )
    print("[flow-enc] loading 4 layers from weights/s3gen/flow/encoder/up_encoders/...")
    var post_layers = load_s3gen_flow_encoder_layers(
        ctx, "weights/s3gen/flow/encoder/up_encoders", 4, D, INT, H, DH,
    )
    ctx.synchronize()
    print("[flow-enc] pre=", len(pre_layers), "post=", len(post_layers))
    assert_true(len(pre_layers) == 6, "should have 6 pre-upsample layers")
    assert_true(len(post_layers) == 4, "should have 4 post-upsample layers")
    assert_true(pre_layers[0].d_model == 512, "d_model should be 512")
    assert_true(pre_layers[0].intermediate == 2048, "intermediate should be 2048")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
