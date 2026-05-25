"""Test linear (60, 512) @ (1, 512).T → (60, 1)."""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.gpu.host import DeviceContext, DeviceBuffer

from modules import Linear, linear_forward


def test_lin() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var M = 60
    var K = 512

    # Build a Linear with weight = ones, bias = 0.
    var w = ctx.enqueue_create_buffer[DType.float32](K)
    w.enqueue_fill(1.0)
    var b = ctx.enqueue_create_buffer[DType.float32](1)
    b.enqueue_fill(0.0)
    var lin = Linear(w^, b^, K, 1, True)

    # x = arange(M*K) → expected y[i] = sum_k x[i,k] = sum over [i*K, (i+1)*K).
    var x = ctx.enqueue_create_buffer[DType.float32](M * K)
    with x.map_to_host() as h:
        for i in range(M * K):
            h[i] = Float32(i)

    var y = ctx.enqueue_create_buffer[DType.float32](M)
    linear_forward(ctx, lin, x, y, M)
    ctx.synchronize()

    with y.map_to_host() as h:
        for i in range(8):
            var expected: Float32 = 0.0
            for k in range(K):
                expected += Float32(i * K + k)
            print("y[", i, "] got=", h[i], " want=", expected)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
