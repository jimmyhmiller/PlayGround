"""Phase 0 sanity test: call nn.softmax.softmax on GPU and verify vs CPU.

Proves that we can wire up MAX `nn.*` kernel functions from Mojo project code.
"""
from std.math import exp
from std.sys import has_accelerator
from std.sys.info import simd_width_of
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import IndexList
from layout import Idx, TileTensor, row_major

from nn.softmax import softmax as nn_softmax


comptime BATCH = 4
comptime AXIS_DIM = 16


def cpu_softmax(data: List[Float32], batch: Int, axis_dim: Int) -> List[Float32]:
    var out = List[Float32](capacity=batch * axis_dim)
    for _ in range(batch * axis_dim):
        out.append(Float32(0.0))
    for row in range(batch):
        var off = row * axis_dim
        var mx: Float32 = data[off]
        for i in range(1, axis_dim):
            if data[off + i] > mx: mx = data[off + i]
        var s: Float32 = 0.0
        for i in range(axis_dim):
            var v = exp(data[off + i] - mx)
            out[off + i] = v
            s += v
        for i in range(axis_dim):
            out[off + i] /= s
    return out^


def test_sanity_softmax_gpu() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var n = BATCH * AXIS_DIM

    # Construct host data.
    var host_data = List[Float32](capacity=n)
    for i in range(n):
        host_data.append(Float32(i) * 0.1)

    # Upload to GPU.
    var in_buf = ctx.enqueue_create_buffer[DType.float32](n)
    with in_buf.map_to_host() as h:
        for i in range(n):
            h[i] = host_data[i]

    var out_buf = ctx.enqueue_create_buffer[DType.float32](n)

    # Compute CPU reference.
    var expected = cpu_softmax(host_data, BATCH, AXIS_DIM)

    # Call nn.softmax via callback pattern (mirrors _interpreter_ops/softmax_ops.mojo).
    var in_ptr = in_buf.unsafe_ptr()
    var out_ptr = out_buf.unsafe_ptr()
    var shape = IndexList[2](BATCH, AXIS_DIM)

    @always_inline
    @parameter
    @__copy_capture(in_ptr)
    def input_fn[width: Int, rank: Int](coords: IndexList[rank]) -> SIMD[DType.float32, width]:
        var c = rebind[IndexList[2]](coords)
        var idx = c[0] * AXIS_DIM + c[1]
        return in_ptr.load[width=width](idx)

    var out_tt = TileTensor(out_ptr, row_major(Idx(BATCH), Idx(AXIS_DIM)))
    var dctx = DeviceContextPtr(ctx)

    nn_softmax[
        DType.float32,
        simd_width_of[DType.float32](),
        2,
        input_fn,
        target="gpu",
    ](shape, out_tt, 1, dctx)

    ctx.synchronize()

    var max_abs: Float32 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n):
            var d = h[i] - expected[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            if i < 4:
                print("sm[", i, "]: gpu=", h[i], " cpu=", expected[i], " diff=", d)
            assert_almost_equal(h[i], expected[i], atol=1.0e-6)
    print("nn.softmax sanity PASS — max abs:", max_abs)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
