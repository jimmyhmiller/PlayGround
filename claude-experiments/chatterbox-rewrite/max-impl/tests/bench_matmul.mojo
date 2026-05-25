"""Microbench MAX nn_matmul at our decode-step shapes."""
from std.gpu.host import DeviceContext
from std.runtime.asyncrt import DeviceContextPtr
from std.time import perf_counter_ns
from layout import Idx, TileTensor, row_major
from linalg.matmul import matmul as nn_matmul


def bench(mut ctx: DeviceContext, M: Int, K: Int, N: Int, iters: Int = 200, warmup: Int = 20) raises:
    var a = ctx.enqueue_create_buffer[DType.float32](M * K)
    var b = ctx.enqueue_create_buffer[DType.float32](N * K)   # transposed: (N, K)
    var c = ctx.enqueue_create_buffer[DType.float32](M * N)
    a.enqueue_fill(0.5)
    b.enqueue_fill(0.5)
    c.enqueue_fill(0.0)

    var dctx = DeviceContextPtr(ctx)
    var at = TileTensor(a, row_major(Idx(M), Idx(K)))
    var bt = TileTensor(b, row_major(Idx(N), Idx(K)))
    var ct = TileTensor(c, row_major(Idx(M), Idx(N)))

    for _ in range(warmup):
        nn_matmul[target="gpu", transpose_b=True](ct, at, bt, dctx)
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(iters):
        nn_matmul[target="gpu", transpose_b=True](ct, at, bt, dctx)
    ctx.synchronize()
    var dt_ns = (perf_counter_ns() - t0) // UInt(iters)
    var dt_us = Float64(Int(dt_ns)) / 1000.0
    var flops = 2.0 * Float64(M) * Float64(K) * Float64(N)
    var gflops = flops / (Float64(Int(dt_ns)) * 1e-9) / 1e9
    print("M=", M, " K=", K, " N=", N, ": ", dt_us, " us  ", gflops, " GFLOP/s")


def main() raises:
    var ctx = DeviceContext()
    bench(ctx, 1, 1024, 1024)
    bench(ctx, 2, 1024, 1024)
    print()
    bench(ctx, 1, 1024, 4096)
    bench(ctx, 2, 1024, 4096)
    print()
    bench(ctx, 1, 4096, 1024)
    bench(ctx, 2, 4096, 1024)
