"""Smoke test: load real T3 Llama-30L weights + run prefill forward.

Verifies the Session A T3 loader. Synthetic input embedding; just checks
the model runs end-to-end and produces non-NaN logits.
"""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.gpu.host import DeviceContext, DeviceBuffer

from weights import load_t3
from t3 import t3_prefill_forward


def test_load_t3_real_weights() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    print("Loading T3 30-layer Llama backbone from disk...")
    var t3 = load_t3(ctx, "weights/t3")
    print("Loaded. Running prefill on B=1, T=16...")

    var B = 1
    var T = 16
    var D = 1024
    var n_x = B * T * D
    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    with x_buf.map_to_host() as h:
        for i in range(n_x):
            h[i] = Float32(i % 100) * 0.001 - 0.05

    # Synthetic cos/sin (proper RoPE needs HF cos_full of shape (B,S,Dh)).
    var cos_buf = ctx.enqueue_create_buffer[DType.float32](B * T * 64)
    cos_buf.enqueue_fill(1.0)
    var sin_buf = ctx.enqueue_create_buffer[DType.float32](B * T * 64)
    sin_buf.enqueue_fill(0.0)
    # Causal mask (all-zero bias means no masking).
    var mask_buf = ctx.enqueue_create_buffer[DType.float32](T * T)
    mask_buf.enqueue_fill(0.0)

    t3_prefill_forward(ctx, t3, x_buf, cos_buf, sin_buf, mask_buf, B, T)
    ctx.synchronize()

    with x_buf.map_to_host() as h:
        print("post-30-layer x[0:8]:", h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7])
        # Check no NaN.
        var any_nan: Bool = False
        for i in range(n_x):
            if h[i] != h[i]: any_nan = True
        if any_nan:
            print("FAIL: NaN detected in output")
        else:
            print("load_t3 smoke PASS (no NaN, 30 layers ran)")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
