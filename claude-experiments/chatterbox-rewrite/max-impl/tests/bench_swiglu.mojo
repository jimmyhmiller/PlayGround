"""Test the actual swiglu output (silu(gate) * up) — fused vs unfused."""
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList
from std.math import exp as mexp
from std.random import seed, random_float64
from modules import Linear, linear_forward, silu
from weights import upload_fp32, upload_bf16
from fixture import load_bf16 as fixture_load_bf16, TensorBF16
from std.collections.list import List


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


def swiglu_combine_fused(
    mut ctx: DeviceContext,
    mut gate_up: DeviceBuffer[DType.float32],
    mut out: DeviceBuffer[DType.float32],
    b: Int, inter: Int,
) raises:
    var gup = gate_up.unsafe_ptr()
    var op = out.unsafe_ptr()
    @always_inline
    @parameter
    @__copy_capture(gup, op, inter)
    def swfn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // inter
        var ii = i - bi * inter
        var base = bi * 2 * inter
        var g = gup[base + ii]
        var u = gup[base + inter + ii]
        var sig = Float32(1.0) / (Float32(1.0) + mexp(-g))
        op[i] = (g * sig) * u
    elementwise[swfn, simd_width=1, target="gpu"](
        IndexList[1](b * inter), DeviceContextPtr(ctx),
    )


def main() raises:
    var ctx = DeviceContext()
    var BASE = "weights/t3/layer0"
    var hidden = 1024
    var inter = 4096

    var gw_f = upload_fp32(ctx, BASE + "/gate_w.bin")
    var uw_f = upload_fp32(ctx, BASE + "/up_w.bin")
    var gw_b = upload_bf16(ctx, BASE + "/gate_w.bf16.bin")
    var uw_b = upload_bf16(ctx, BASE + "/up_w.bf16.bin")

    var gu_b_paths = List[String]()
    gu_b_paths.append(BASE + "/gate_w.bf16.bin")
    gu_b_paths.append(BASE + "/up_w.bf16.bin")
    var gu_b = upload_bf16_concat_rows(ctx, gu_b_paths)

    var zero_inter = ctx.enqueue_create_buffer[DType.float32](inter)
    zero_inter.enqueue_fill(0.0)
    var zero_2inter = ctx.enqueue_create_buffer[DType.float32](2 * inter)
    zero_2inter.enqueue_fill(0.0)
    var dummy_gu_f = ctx.enqueue_create_buffer[DType.float32](2 * inter * hidden)
    dummy_gu_f.enqueue_fill(0.0)

    var gate = Linear(gw_f^, zero_inter, hidden, inter, False, gw_b^)
    var up = Linear(uw_f^, zero_inter.copy(), hidden, inter, False, uw_b^)
    var gate_up = Linear(dummy_gu_f^, zero_2inter, hidden, 2 * inter, False, gu_b^)

    seed(123)
    var x_buf = ctx.enqueue_create_buffer[DType.float32](hidden)
    with x_buf.map_to_host() as h:
        for i in range(hidden):
            h[i] = Float32(random_float64() - 0.5) * 0.2
    ctx.synchronize()

    # Fused: matmul + swiglu.
    var fused_mm = ctx.enqueue_create_buffer[DType.float32](2 * inter)
    var out_fused = ctx.enqueue_create_buffer[DType.float32](inter)
    linear_forward(ctx, gate_up, x_buf, fused_mm, 1)
    swiglu_combine_fused(ctx, fused_mm, out_fused, 1, inter)
    ctx.synchronize()

    # Unfused: gate matmul, up matmul, silu, mul.
    var gate_h = ctx.enqueue_create_buffer[DType.float32](inter)
    var up_h = ctx.enqueue_create_buffer[DType.float32](inter)
    var act_h = ctx.enqueue_create_buffer[DType.float32](inter)
    var out_unfused = ctx.enqueue_create_buffer[DType.float32](inter)
    linear_forward(ctx, gate, x_buf, gate_h, 1)
    linear_forward(ctx, up, x_buf, up_h, 1)
    silu(ctx, gate_h, act_h, inter)
    var ap = act_h.unsafe_ptr()
    var upp = up_h.unsafe_ptr()
    var pp = out_unfused.unsafe_ptr()
    @always_inline
    @parameter
    @__copy_capture(ap, upp, pp)
    def mul_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var a = ap.load[width=width, alignment=alignment](i)
        var u = upp.load[width=width, alignment=alignment](i)
        pp.store[width=width, alignment=alignment](i, a * u)
    elementwise[mul_func, simd_width=4, target="gpu"](
        IndexList[1](inter), DeviceContextPtr(ctx),
    )
    ctx.synchronize()

    var f_h = ctx.enqueue_create_host_buffer[DType.float32](inter)
    var u_h = ctx.enqueue_create_host_buffer[DType.float32](inter)
    ctx.enqueue_copy(f_h, out_fused)
    ctx.enqueue_copy(u_h, out_unfused)
    ctx.synchronize()

    var max_diff: Float32 = 0.0
    var n_diff = 0
    var first_diff = -1
    for i in range(inter):
        var d = f_h[i] - u_h[i]
        if d < 0: d = -d
        if d > Float32(0.0):
            n_diff += 1
            if first_diff < 0: first_diff = i
        if d > max_diff: max_diff = d
    print("SwiGLU output, M=1, inter=4096:")
    print("  fused vs unfused: max_diff =", max_diff, ", #non-zero =", n_diff, "/", inter, ", first @", first_diff)
    if first_diff >= 0:
        print("  fused[", first_diff, "]=", f_h[first_diff], "vs unfused=", u_h[first_diff])
