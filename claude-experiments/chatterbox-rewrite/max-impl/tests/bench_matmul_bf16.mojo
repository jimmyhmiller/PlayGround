"""Microbench MAX nn_matmul at our decode-step shapes — bf16 weights."""
from std.gpu.host import DeviceContext
from std.runtime.asyncrt import DeviceContextPtr
from std.time import perf_counter_ns
from layout import Idx, TileTensor, row_major
from linalg.matmul import matmul as nn_matmul


def bench_f32(mut ctx: DeviceContext, M: Int, K: Int, N: Int, iters: Int = 200, warmup: Int = 20) raises:
    var a = ctx.enqueue_create_buffer[DType.float32](M * K)
    var b = ctx.enqueue_create_buffer[DType.float32](N * K)
    var c = ctx.enqueue_create_buffer[DType.float32](M * N)
    a.enqueue_fill(0.5)
    b.enqueue_fill(0.5)
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
    print("f32 M=", M, " K=", K, " N=", N, ": ", dt_us, " us")


def bench_bf16(mut ctx: DeviceContext, M: Int, K: Int, N: Int, iters: Int = 200, warmup: Int = 20) raises:
    var a = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    var b = ctx.enqueue_create_buffer[DType.bfloat16](N * K)
    var c = ctx.enqueue_create_buffer[DType.float32](M * N)
    a.enqueue_fill(BFloat16(0.5))
    b.enqueue_fill(BFloat16(0.5))
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
    print("bf16 M=", M, " K=", K, " N=", N, ": ", dt_us, " us")


def main() raises:
    var ctx = DeviceContext()
    print("--- f32 ---")
    bench_f32(ctx, 1, 1024, 1024)
    bench_f32(ctx, 2, 1024, 1024)
    bench_f32(ctx, 2, 1024, 3072)
    bench_f32(ctx, 2, 1024, 4096)
    bench_f32(ctx, 2, 1024, 8192)
    bench_f32(ctx, 2, 4096, 1024)
    print()
    print("--- bf16 ---")
    bench_bf16(ctx, 1, 1024, 1024)
    bench_bf16(ctx, 2, 1024, 1024)
    bench_bf16(ctx, 2, 1024, 3072)
    bench_bf16(ctx, 2, 1024, 4096)
    bench_bf16(ctx, 2, 1024, 8192)
    bench_bf16(ctx, 2, 4096, 1024)

    print()
    print("--- CFM large-M shapes ---")
    # b=2, t=1000 mel frames → M=2000. K,N from CFM Linears (d_model=256, ff=1024).
    bench_f32(ctx, 2000, 256, 256)        # to_q,k,v inner
    bench_bf16(ctx, 2000, 256, 256)
    bench_f32(ctx, 2000, 256, 512)        # to_q,k,v → inner=512
    bench_bf16(ctx, 2000, 256, 512)
    bench_f32(ctx, 2000, 512, 256)        # to_out
    bench_bf16(ctx, 2000, 512, 256)
    bench_f32(ctx, 2000, 256, 2048)       # ff w1: GEGLU doubles intermediate
    bench_bf16(ctx, 2000, 256, 2048)
    bench_f32(ctx, 2000, 1024, 256)       # ff w2
    bench_bf16(ctx, 2000, 1024, 256)

# Also test large-M shapes that CFM uses.
