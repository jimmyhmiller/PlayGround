"""Runtime smoke test: load real upstream flow-encoder weights and run
TransformerEncoderLayer forward end-to-end with a small T=8 fake input +
fake position embedding. Validates that the new RelPos MHA + FF pipeline
doesn't crash on real weight shapes — does NOT yet check torch parity (that
requires an oracle dump of upstream encoder activations).
"""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true
from std.gpu.host import DeviceContext

from weights import load_s3gen_flow_encoder_layers
from conformer import transformer_encoder_layer_forward


comptime B = 1
comptime T = 8
comptime T_POS = 2 * T - 1   # 15
comptime D = 512
comptime H = 8
comptime DH = 64
comptime INTER = 2048


def test_relpos_mha_smoke() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    print("[relpos] loading flow encoder layers...")
    var layers = load_s3gen_flow_encoder_layers(
        ctx, "weights/s3gen/flow/encoder/encoders", 1, D, INTER, H, DH,
    )

    # Build a deterministic input: x[b,t,d] = sin((t + 0.1*d) * 0.01).
    var x_buf = ctx.enqueue_create_buffer[DType.float32](B * T * D)
    from std.math import sin
    with x_buf.map_to_host() as h:
        for t_i in range(T):
            for d_i in range(D):
                h[t_i * D + d_i] = sin(Float32(t_i) * 0.1 + Float32(d_i) * 0.001)

    # Fake position embedding (1, T_POS, D).
    var pos_buf = ctx.enqueue_create_buffer[DType.float32](T_POS * D)
    with pos_buf.map_to_host() as h:
        for p in range(T_POS):
            for d in range(D):
                h[p * D + d] = sin(Float32(p) * 0.05 + Float32(d) * 0.002)

    print("[relpos] running encoder layer forward (T=", T, " T_POS=", T_POS, ")...")
    transformer_encoder_layer_forward(
        ctx, layers[0], x_buf, pos_buf, B, T, T_POS,
    )
    ctx.synchronize()

    # Sanity check the output isn't all NaN or zero.
    var n_nan = 0
    var sum_abs: Float32 = 0.0
    with x_buf.map_to_host() as h:
        for i in range(B * T * D):
            var v = h[i]
            if v != v: n_nan += 1
            if v < 0.0: sum_abs -= v
            else:       sum_abs += v
    print("[relpos] output mean-abs=", sum_abs / Float32(B * T * D), " nan_count=", n_nan)
    assert_true(n_nan == 0, "no NaNs in encoder output")
    assert_true(sum_abs > 0.0, "encoder output should not be all zero")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
