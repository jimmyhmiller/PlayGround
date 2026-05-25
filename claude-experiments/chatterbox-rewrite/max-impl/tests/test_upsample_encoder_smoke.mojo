"""Runtime smoke test for the full UpsampleConformerEncoder forward.

Loads real upstream weights and runs the full pipeline:
  token_ids (B, T_in) → input_embedding lookup → embed.out → pre_lookahead
    → 6 RelPos transformer layers → up_layer → up_embed.out
    → 4 RelPos transformer layers → after_norm → encoder_proj
    → mu (B, 80, 2*T_in)

Small T_in = 4 so the doubled length stays modest.
"""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true
from std.gpu.host import DeviceContext

from weights import load_upsample_conformer_encoder
from upsample_encoder import upsample_conformer_forward


comptime B = 1
comptime T_IN = 4
comptime T_UP = 2 * T_IN
comptime MEL = 80


def test_upsample_encoder_smoke() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    print("[enc] loading UpsampleConformerEncoder from weights/s3gen/flow/...")
    var enc = load_upsample_conformer_encoder(ctx, "weights/s3gen/flow")

    # Random small token_ids in valid range [0, vocab).
    var token_ids = ctx.enqueue_create_buffer[DType.int64](B * T_IN)
    with token_ids.map_to_host() as h:
        for i in range(T_IN):
            h[i] = Int64((i * 137 + 42) % 6562)

    var mu_out = ctx.enqueue_create_buffer[DType.float32](B * MEL * T_UP)
    print("[enc] running forward (T_in=", T_IN, " → T_up=", T_UP, ")...")
    upsample_conformer_forward(ctx, enc, token_ids, mu_out, B, T_IN)
    ctx.synchronize()

    var n_nan = 0
    var sum_abs: Float32 = 0.0
    with mu_out.map_to_host() as h:
        for i in range(B * MEL * T_UP):
            var v = h[i]
            if v != v: n_nan += 1
            if v < 0.0: sum_abs -= v
            else:       sum_abs += v
    print("[enc] mu output mean-abs=",
          sum_abs / Float32(B * MEL * T_UP), " nan_count=", n_nan)
    assert_true(n_nan == 0, "no NaNs in encoder output")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
