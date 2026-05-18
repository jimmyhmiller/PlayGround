"""Smoke test: load real upstream CFM estimator weights.

Validates `load_cfm_estimator_real()` populates all 1 down + 12 mid + 1 up
stages plus time_mlp, final_block, final_proj.
"""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true
from std.gpu.host import DeviceContext

from weights import load_cfm_estimator_real


def test_load_cfm_real_weights() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    print("[cfm] loading from weights/s3gen/flow/decoder/estimator/...")
    var cfm = load_cfm_estimator_real(ctx, "weights/s3gen/flow/decoder/estimator")
    ctx.synchronize()
    print("[cfm] down=", len(cfm.down_blocks),
          " mid=", len(cfm.mid_blocks),
          " up=", len(cfm.up_blocks),
          " mid[0].transformers=", len(cfm.mid_blocks[0].transformers))
    assert_true(len(cfm.down_blocks) == 1, "1 down stage")
    assert_true(len(cfm.mid_blocks) == 12, "12 mid stages")
    assert_true(len(cfm.up_blocks) == 1, "1 up stage")
    assert_true(len(cfm.mid_blocks[0].transformers) == 4, "4 transformers per mid stage")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
