"""SDPA building blocks smoke test (small B=1, H=2, S=4, D=8)."""
from std.math import exp, sqrt
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer

from attention import qk_scaled_and_masked, softmax_2d, av_matmul


comptime B = 1
comptime H = 2
comptime S = 4
comptime D = 8
comptime BH = B * H


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def test_sdpa_smoke() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var n_qkv = BH * S * D
    var n_logits = BH * S * S

    # Synthetic Q, K, V.
    var q_data = List[Float32]()
    var k_data = List[Float32]()
    var v_data = List[Float32]()
    for i in range(n_qkv):
        q_data.append(Float32(i) * 0.01 - 0.1)
        k_data.append(Float32(i) * 0.005 + 0.05)
        v_data.append(Float32(i) * 0.02 - 0.05)
    # Mask: all zeros (no causal mask).
    var mask_data = List[Float32]()
    for i in range(S * S):
        mask_data.append(Float32(0.0))

    var q_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv)
    var k_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv)
    var v_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv)
    var mask_buf = ctx.enqueue_create_buffer[DType.float32](S * S)
    var logits_buf = ctx.enqueue_create_buffer[DType.float32](n_logits)
    var probs_buf  = ctx.enqueue_create_buffer[DType.float32](n_logits)
    var out_buf    = ctx.enqueue_create_buffer[DType.float32](n_qkv)
    upload(q_buf, q_data, n_qkv)
    upload(k_buf, k_data, n_qkv)
    upload(v_buf, v_data, n_qkv)
    upload(mask_buf, mask_data, S * S)

    var scale: Float32 = 1.0 / sqrt(Float32(D))
    qk_scaled_and_masked(ctx, q_buf, k_buf, mask_buf, logits_buf,
                          BH, S, S, D, scale, True)
    softmax_2d(ctx, logits_buf, probs_buf, BH * S, S)
    av_matmul(ctx, probs_buf, v_buf, out_buf, BH, S, S, D)
    ctx.synchronize()

    # CPU reference.
    var expected = List[Float32]()
    for _ in range(n_qkv): expected.append(Float32(0.0))
    for bh in range(BH):
        for iq in range(S):
            var row_logits = List[Float32]()
            var row_max: Float32 = -1.0e30
            for ik in range(S):
                var s: Float32 = 0.0
                for di in range(D):
                    s += q_data[bh * S * D + iq * D + di] * k_data[bh * S * D + ik * D + di]
                s *= scale
                row_logits.append(s)
                if s > row_max: row_max = s
            var sum_exp: Float32 = 0.0
            var row_probs = List[Float32]()
            for ik in range(S):
                var e = exp(row_logits[ik] - row_max)
                row_probs.append(e)
                sum_exp += e
            for di in range(D):
                var acc: Float32 = 0.0
                for ik in range(S):
                    acc += (row_probs[ik] / sum_exp) * v_data[bh * S * D + ik * D + di]
                expected[bh * S * D + iq * D + di] = acc

    var max_abs: Float32 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n_qkv):
            var d_ = h[i] - expected[i]
            if d_ < 0.0: d_ = -d_
            if d_ > max_abs: max_abs = d_
            assert_almost_equal(h[i], expected[i], atol=1.0e-5)
    print("SDPA smoke PASS — max abs:", max_abs)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
