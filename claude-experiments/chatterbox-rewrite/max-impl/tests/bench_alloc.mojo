"""Measure DeviceBuffer alloc cost: 100 fresh small allocs vs 1 large reused."""
from std.gpu.host import DeviceContext, DeviceBuffer
from std.time import perf_counter_ns

def main() raises:
    var ctx = DeviceContext()

    # Warmup.
    for _ in range(50):
        var b = ctx.enqueue_create_buffer[DType.float32](1024)
        _ = b
    ctx.synchronize()

    # Bench: 100 fresh allocs of 1024 floats.
    var t0 = perf_counter_ns()
    for _ in range(100):
        var b = ctx.enqueue_create_buffer[DType.float32](1024)
        _ = b
    ctx.synchronize()
    var t1 = perf_counter_ns()
    print("100 fresh allocs of 1024 floats: ", Float64(t1 - t0) / 1e6, "ms total, avg",
          Float64(t1 - t0) / 1e5, "us each")

    # Bench: 1000 fresh allocs (simulating one T3 generation: 30 layers × 15 allocs × 180 tokens = 81000 — let's do 1000 to keep test small).
    t0 = perf_counter_ns()
    for _ in range(1000):
        var b = ctx.enqueue_create_buffer[DType.float32](1024)
        _ = b
    ctx.synchronize()
    t1 = perf_counter_ns()
    print("1000 fresh allocs of 1024 floats: ", Float64(t1 - t0) / 1e6, "ms total, avg",
          Float64(t1 - t0) / 1e4, "us each")

    # Compare to reusing a buffer 1000 times.
    var buf = ctx.enqueue_create_buffer[DType.float32](1024)
    t0 = perf_counter_ns()
    for _ in range(1000):
        buf.enqueue_fill(0.0)
    ctx.synchronize()
    t1 = perf_counter_ns()
    print("1000 fills of same buffer: ", Float64(t1 - t0) / 1e6, "ms total, avg",
          Float64(t1 - t0) / 1e4, "us each")
