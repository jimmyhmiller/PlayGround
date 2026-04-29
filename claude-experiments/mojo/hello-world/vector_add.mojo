from std.math import ceildiv
from std.sys import has_accelerator
from std.gpu import block_idx, block_dim, thread_idx
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

comptime dtype = DType.float32
comptime N = 1 << 20
comptime BLOCK = 256
comptime layout = row_major[N]()

def add_kernel(
    a: TileTensor[dtype, type_of(layout), MutAnyOrigin],
    b: TileTensor[dtype, type_of(layout), MutAnyOrigin],
    c: TileTensor[dtype, type_of(layout), MutAnyOrigin],
):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    if tid < N:
        c[tid] = a[tid] + b[tid]

def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    print("device:", ctx.name())

    var a_buf = ctx.enqueue_create_buffer[dtype](N)
    var b_buf = ctx.enqueue_create_buffer[dtype](N)
    var c_buf = ctx.enqueue_create_buffer[dtype](N)

    with a_buf.map_to_host() as ah, b_buf.map_to_host() as bh:
        var at = TileTensor(ah, layout)
        var bt = TileTensor(bh, layout)
        for i in range(N):
            at[i] = Float32(i)
            bt[i] = Float32(2 * i)

    var a = TileTensor(a_buf, layout)
    var b = TileTensor(b_buf, layout)
    var c = TileTensor(c_buf, layout)

    ctx.enqueue_function[add_kernel, add_kernel](
        a, b, c,
        grid_dim=ceildiv(N, BLOCK),
        block_dim=BLOCK,
    )

    with c_buf.map_to_host() as ch:
        var ct = TileTensor(ch, layout)
        print("c[0]      =", ct[0],      " (expected 0)")
        print("c[1]      =", ct[1],      " (expected 3)")
        print("c[1024]   =", ct[1024],   " (expected", 3 * 1024, ")")
        print("c[N-1]    =", ct[N - 1],  " (expected", 3 * (N - 1), ")")
