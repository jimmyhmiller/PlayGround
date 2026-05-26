"""Test on REAL T3 weights: fused gate_up matmul vs separate gate/up matmuls.
Uses the actual linear_forward path (with split-M etc) so we exercise the
same code as production T3 decode.
"""
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList
from std.random import seed, random_float64
from linalg.matmul import matmul as nn_matmul
from layout import TileTensor, row_major, Idx
from modules import Linear, linear_forward
from weights import upload_fp32, upload_bf16
from std.collections.list import List
from fixture import load_bf16 as fixture_load_bf16, TensorBF16


def upload_bf16_concat_rows(
    mut ctx: DeviceContext, paths: List[String]
) raises -> DeviceBuffer[DType.bfloat16]:
    var total = 0
    var tensors = List[TensorBF16]()
    for i in range(len(paths)):
        var t = fixture_load_bf16(paths[i])
        total += len(t.data)
        tensors.append(t^)
    var buf = ctx.enqueue_create_buffer[DType.bfloat16](total)
    with buf.map_to_host() as h:
        var off = 0
        for ti in range(len(tensors)):
            ref data = tensors[ti].data
            var n = len(data)
            for k in range(n):
                h[off + k] = data[k]
            off += n
    return buf^

def main() raises:
    var ctx = DeviceContext()

    var BASE = "weights/t3/layer0"
    var hidden = 1024
    var inter = 4096

    # Load f32 + bf16 of gate_w, up_w.
    var gw_f = upload_fp32(ctx, BASE + "/gate_w.bin")
    var uw_f = upload_fp32(ctx, BASE + "/up_w.bin")
    var gw_b = upload_bf16(ctx, BASE + "/gate_w.bf16.bin")
    var uw_b = upload_bf16(ctx, BASE + "/up_w.bf16.bin")

    var gu_paths = List[String]()
    gu_paths.append(BASE + "/gate_w.bin")
    gu_paths.append(BASE + "/up_w.bin")
    # f32 fused — not used since gate_up is bf16-only in T3 bf16 path.
    var gu_b_paths = List[String]()
    gu_b_paths.append(BASE + "/gate_w.bf16.bin")
    gu_b_paths.append(BASE + "/up_w.bf16.bin")
    var gu_b = upload_bf16_concat_rows(ctx, gu_b_paths)

    # Build Linear objects with bf16 attached.
    var zero_inter = ctx.enqueue_create_buffer[DType.float32](inter)
    zero_inter.enqueue_fill(0.0)
    var zero_2inter = ctx.enqueue_create_buffer[DType.float32](2 * inter)
    zero_2inter.enqueue_fill(0.0)

    var gate = Linear(gw_f^, zero_inter, hidden, inter, False, gw_b^)
    var up = Linear(uw_f^, zero_inter.copy(), hidden, inter, False, uw_b^)
    # We also need a fresh fp32 for fused (won't be used) and the bf16 fused.
    var dummy_gu_f = ctx.enqueue_create_buffer[DType.float32](2 * inter * hidden)
    dummy_gu_f.enqueue_fill(0.0)
    var gate_up = Linear(dummy_gu_f^, zero_2inter, hidden, 2 * inter, False, gu_b^)

    # Build a realistic input: random with small magnitude (matches RMS-normed residual stream).
    seed(123)
    var x_buf = ctx.enqueue_create_buffer[DType.float32](hidden)
    with x_buf.map_to_host() as h:
        for i in range(hidden):
            h[i] = Float32(random_float64() - 0.5) * 0.2
    ctx.synchronize()

    # Run fused.
    var y_fused = ctx.enqueue_create_buffer[DType.float32](2 * inter)
    linear_forward(ctx, gate_up, x_buf, y_fused, 1)
    ctx.synchronize()

    # Run separate gate + up.
    var y_gate = ctx.enqueue_create_buffer[DType.float32](inter)
    var y_up = ctx.enqueue_create_buffer[DType.float32](inter)
    linear_forward(ctx, gate, x_buf, y_gate, 1)
    linear_forward(ctx, up, x_buf, y_up, 1)
    ctx.synchronize()

    # Compare.
    var yf_h = ctx.enqueue_create_host_buffer[DType.float32](2 * inter)
    var yg_h = ctx.enqueue_create_host_buffer[DType.float32](inter)
    var yu_h = ctx.enqueue_create_host_buffer[DType.float32](inter)
    ctx.enqueue_copy(yf_h, y_fused)
    ctx.enqueue_copy(yg_h, y_gate)
    ctx.enqueue_copy(yu_h, y_up)
    ctx.synchronize()

    var max_diff_gate: Float32 = 0.0
    var max_diff_up: Float32 = 0.0
    var n_diff_gate = 0
    var n_diff_up = 0
    var first_g_diff_i = -1
    var first_u_diff_i = -1

    for i in range(inter):
        var d_g = yf_h[i] - yg_h[i]
        if d_g < 0: d_g = -d_g
        if d_g > Float32(0.0):
            n_diff_gate += 1
            if first_g_diff_i < 0:
                first_g_diff_i = i
        if d_g > max_diff_gate: max_diff_gate = d_g

        var d_u = yf_h[inter + i] - yu_h[i]
        if d_u < 0: d_u = -d_u
        if d_u > Float32(0.0):
            n_diff_up += 1
            if first_u_diff_i < 0:
                first_u_diff_i = i
        if d_u > max_diff_up: max_diff_up = d_u

    print("REAL T3 weights, M=1, hidden=1024, inter=4096:")
    print("fused[0:inter] vs gate matmul: max_diff =", max_diff_gate,
          ", #non-zero diff =", n_diff_gate, "/", inter, ", first @", first_g_diff_i)
    print("fused[inter:2*inter] vs up matmul: max_diff =", max_diff_up,
          ", #non-zero diff =", n_diff_up, "/", inter, ", first @", first_u_diff_i)

    if first_g_diff_i >= 0:
        print("  yf[gate@", first_g_diff_i, "] =", yf_h[first_g_diff_i],
              "vs yg =", yg_h[first_g_diff_i])
    if first_u_diff_i >= 0:
        print("  yf[up@", first_u_diff_i, "] =", yf_h[inter + first_u_diff_i],
              "vs yu =", yu_h[first_u_diff_i])
