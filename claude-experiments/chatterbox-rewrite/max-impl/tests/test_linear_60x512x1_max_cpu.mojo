"""Same shape as test_linear_60x512x1, but with target='cpu' instead of 'gpu'.

If the CPU matmul produces the correct full output and the GPU one only
writes row 0, that demonstrates the bug is in the AMD/N=1 GPU dispatch
specifically — not in our call.
"""
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.testing import TestSuite

from layout import Idx, TileTensor, row_major
from linalg.matmul import matmul as nn_matmul


def test_cpu() raises:
    var ctx = DeviceContext()
    var M = 60
    var K = 512

    var a = ctx.enqueue_create_buffer[DType.float32](M * K)
    var b = ctx.enqueue_create_buffer[DType.float32](1 * K)
    var c = ctx.enqueue_create_buffer[DType.float32](M * 1)
    c.enqueue_fill(0.0)
    b.enqueue_fill(1.0)
    with a.map_to_host() as h:
        for i in range(M * K):
            h[i] = Float32(i)

    var a_t = TileTensor(a, row_major(Idx(M), Idx(K)))
    var b_t = TileTensor(b, row_major(Idx(1), Idx(K)))
    var c_t = TileTensor(c, row_major(Idx(M), Idx(1)))

    nn_matmul[target="cpu", transpose_b=True](c_t, a_t, b_t, DeviceContextPtr(ctx))
    ctx.synchronize()

    print("Target = CPU, M=60, K=512, N=1, transpose_b=True")
    with c.map_to_host() as h:
        for i in range(8):
            # Expected: sum_{k=0..K-1} (i*K + k) = i*K^2 + K*(K-1)/2
            var expected = Float32(i) * Float32(K) * Float32(K) + Float32(K) * Float32(K - 1) / 2.0
            print("  C[", i, "] =", h[i], "  expected =", expected)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
