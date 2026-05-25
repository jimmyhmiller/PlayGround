"""HF-style RoPE elementwise smoke test (B=1, H=2, S=4, D=8).

Verifies the apply_rope_hf_style helper matches a CPU reference.
"""
from std.math import cos as scos, sin as ssin, pi
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer

from transformer_blocks import apply_rope_hf_style


comptime B = 1
comptime H = 2
comptime S = 4
comptime D = 8
comptime HALF = D // 2


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def test_rope_hf() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var n = B * H * S * D
    var x_data = List[Float32]()
    for i in range(n):
        x_data.append(Float32(i) * 0.05 - 0.5)

    # cos/sin (B, S, D) — synthetic per (s, d) frequency.
    var cs_n = B * S * D
    var cos_data = List[Float32]()
    var sin_data = List[Float32]()
    for i in range(cs_n):
        var s = (i % (S * D)) // D
        var dim = i % D
        var freq: Float32 = Float32(s) / (10000.0 ** (Float32(dim % HALF) / Float32(HALF)))
        cos_data.append(scos(freq))
        sin_data.append(ssin(freq))

    var x_buf = ctx.enqueue_create_buffer[DType.float32](n)
    var y_buf = ctx.enqueue_create_buffer[DType.float32](n)
    var cos_buf = ctx.enqueue_create_buffer[DType.float32](cs_n)
    var sin_buf = ctx.enqueue_create_buffer[DType.float32](cs_n)
    upload(x_buf, x_data, n)
    upload(cos_buf, cos_data, cs_n)
    upload(sin_buf, sin_data, cs_n)

    apply_rope_hf_style(ctx, x_buf, y_buf, cos_buf, sin_buf, B, H, S, D)
    ctx.synchronize()

    # CPU reference.
    var expected = List[Float32]()
    for _ in range(n): expected.append(Float32(0.0))
    for bi in range(B):
        for hi in range(H):
            for si in range(S):
                for di in range(D):
                    var idx = bi * H * S * D + hi * S * D + si * D + di
                    var c = cos_data[bi * S * D + si * D + di]
                    var sn = sin_data[bi * S * D + si * D + di]
                    var x_i = x_data[idx]
                    var pair_di = di + HALF if di < HALF else di - HALF
                    var pair_src = bi * H * S * D + hi * S * D + si * D + pair_di
                    var paired = x_data[pair_src]
                    var rh: Float32 = -paired if di < HALF else paired
                    expected[idx] = x_i * c + rh * sn

    var max_abs: Float32 = 0.0
    with y_buf.map_to_host() as h:
        for i in range(n):
            var d = h[i] - expected[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            assert_almost_equal(h[i], expected[i], atol=1.0e-5)
    print("RoPE (HF style) — max abs:", max_abs)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
