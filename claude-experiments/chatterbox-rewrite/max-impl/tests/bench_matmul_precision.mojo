"""Verify MAX bf16 matmul precision at N=8192 (fused gate_up shape) vs N=4096 (unfused gate/up).

If the fused (8192, 1024) matmul produces different results than concat(matmul_gate_4096, matmul_up_4096),
that's a MAX kernel bug. If they're bit-identical, our problem is elsewhere (e.g., calling code).
"""
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList
from std.math import sqrt
from std.random import seed, random_float64
from linalg.matmul import matmul as nn_matmul
from layout import TileTensor, row_major, Idx

def main() raises:
    var ctx = DeviceContext()
    var M = 1
    var K = 1024
    var N_half = 4096
    var N_full = 8192

    # Build random f32 weights for gate and up.
    seed(42)
    var gate_w_host = ctx.enqueue_create_host_buffer[DType.float32](N_half * K)
    var up_w_host = ctx.enqueue_create_host_buffer[DType.float32](N_half * K)
    var x_host = ctx.enqueue_create_host_buffer[DType.float32](M * K)
    for i in range(N_half * K):
        gate_w_host[i] = Float32(random_float64() - 0.5) * 0.1
        up_w_host[i] = Float32(random_float64() - 0.5) * 0.1
    for i in range(M * K):
        x_host[i] = Float32(random_float64() - 0.5) * 0.5

    # Upload to GPU as bf16 (round-cast on host).
    var gate_w_bf = ctx.enqueue_create_buffer[DType.bfloat16](N_half * K)
    var up_w_bf = ctx.enqueue_create_buffer[DType.bfloat16](N_half * K)
    var gate_up_bf = ctx.enqueue_create_buffer[DType.bfloat16](N_full * K)
    var x_bf = ctx.enqueue_create_buffer[DType.bfloat16](M * K)

    with gate_w_bf.map_to_host() as h:
        for i in range(N_half * K): h[i] = BFloat16(gate_w_host[i])
    with up_w_bf.map_to_host() as h:
        for i in range(N_half * K): h[i] = BFloat16(up_w_host[i])
    # Fused: gate rows first, then up rows.
    with gate_up_bf.map_to_host() as h:
        for i in range(N_half * K): h[i] = BFloat16(gate_w_host[i])
        for i in range(N_half * K): h[N_half * K + i] = BFloat16(up_w_host[i])
    with x_bf.map_to_host() as h:
        for i in range(M * K): h[i] = BFloat16(x_host[i])

    ctx.synchronize()

    # Three matmuls: separate gate, separate up, fused.
    var y_gate = ctx.enqueue_create_buffer[DType.float32](M * N_half)
    var y_up = ctx.enqueue_create_buffer[DType.float32](M * N_half)
    var y_fused = ctx.enqueue_create_buffer[DType.float32](M * N_full)

    var dctx = DeviceContextPtr(ctx)
    var x_t = TileTensor(x_bf, row_major(Idx(M), Idx(K)))
    var g_t = TileTensor(gate_w_bf, row_major(Idx(N_half), Idx(K)))
    var u_t = TileTensor(up_w_bf, row_major(Idx(N_half), Idx(K)))
    var gu_t = TileTensor(gate_up_bf, row_major(Idx(N_full), Idx(K)))
    var yg_t = TileTensor(y_gate, row_major(Idx(M), Idx(N_half)))
    var yu_t = TileTensor(y_up, row_major(Idx(M), Idx(N_half)))
    var yf_t = TileTensor(y_fused, row_major(Idx(M), Idx(N_full)))

    nn_matmul[target="gpu", transpose_b=True](yg_t, x_t, g_t, dctx)
    nn_matmul[target="gpu", transpose_b=True](yu_t, x_t, u_t, dctx)
    nn_matmul[target="gpu", transpose_b=True](yf_t, x_t, gu_t, dctx)
    ctx.synchronize()

    # Read back.
    var yg_h = ctx.enqueue_create_host_buffer[DType.float32](M * N_half)
    var yu_h = ctx.enqueue_create_host_buffer[DType.float32](M * N_half)
    var yf_h = ctx.enqueue_create_host_buffer[DType.float32](M * N_full)
    ctx.enqueue_copy(yg_h, y_gate)
    ctx.enqueue_copy(yu_h, y_up)
    ctx.enqueue_copy(yf_h, y_fused)
    ctx.synchronize()

    # Compare: fused[0:N_half] vs y_gate, fused[N_half:N_full] vs y_up.
    var max_diff_gate: Float32 = 0.0
    var max_diff_up: Float32 = 0.0
    var n_gate_diff = 0
    var n_up_diff = 0
    for i in range(N_half):
        var d = yf_h[i] - yg_h[i]
        if d < 0: d = -d
        if d > max_diff_gate: max_diff_gate = d
        if d > Float32(0.0): n_gate_diff += 1
    for i in range(N_half):
        var d = yf_h[N_half + i] - yu_h[i]
        if d < 0: d = -d
        if d > max_diff_up: max_diff_up = d
        if d > Float32(0.0): n_up_diff += 1
    print("M=1, K=1024, N_half=4096, N_full=8192")
    print("fused[0:4096] vs separate gate matmul: max_diff =", max_diff_gate,
          ", non-zero diff count =", n_gate_diff)
    print("fused[4096:8192] vs separate up matmul: max_diff =", max_diff_up,
          ", non-zero diff count =", n_up_diff)
