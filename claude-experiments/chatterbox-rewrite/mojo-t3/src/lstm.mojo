"""
LSTM cell + multi-layer forward, matching PyTorch nn.LSTM semantics.

PyTorch convention: weight_ih_l{N} is (4*H, I) and weight_hh_l{N} is (4*H, H).
The 4 gates in order: i, f, g, o (input, forget, candidate, output).

Single-step (cell) update:
    pre_gates = x @ W_ih.T + b_ih + h_prev @ W_hh.T + b_hh   # (B, 4H)
    i = sigmoid(pre_gates[:, 0:H])
    f = sigmoid(pre_gates[:, H:2H])
    g = tanh(pre_gates[:, 2H:3H])
    o = sigmoid(pre_gates[:, 3H:4H])
    c_new = f * c_prev + i * g
    h_new = o * tanh(c_new)

Sequence forward: iterate cell over time, optionally batch_first.
"""
from std.math import exp, tanh
from std.gpu import block_idx, thread_idx
from layout import TileTensor, TensorLayout, row_major


def lstm_layer_n_first_kernel[
    dtype: DType,
    XLayout: TensorLayout,
    HSeqLayout: TensorLayout,
    WIHLayout: TensorLayout,
    WHHLayout: TensorLayout,
    BLayout: TensorLayout,
    HIDDEN: Int,
    INPUT: Int,
    BLOCK: Int,
](
    h_out: TileTensor[dtype, HSeqLayout, MutAnyOrigin],   # (B, T, HIDDEN)
    x: TileTensor[dtype, XLayout, MutAnyOrigin],           # (B, T, INPUT)
    w_ih: TileTensor[dtype, WIHLayout, MutAnyOrigin],
    w_hh: TileTensor[dtype, WHHLayout, MutAnyOrigin],
    b_ih: TileTensor[dtype, BLayout, MutAnyOrigin],
    b_hh: TileTensor[dtype, BLayout, MutAnyOrigin],
    batch: Int, time: Int,
):
    """Single LSTM layer with INPUT possibly != HIDDEN. Same as lstm_layer_first_kernel
    but with separate INPUT and HIDDEN compile-time parameters."""
    comptime assert x.flat_rank == 3
    comptime assert h_out.flat_rank == 3
    comptime assert w_ih.flat_rank == 2
    comptime assert w_hh.flat_rank == 2
    comptime assert b_ih.flat_rank == 1
    comptime assert b_hh.flat_rank == 1
    from std.gpu.memory import AddressSpace
    from std.gpu.sync import barrier
    from layout import stack_allocation
    var b = block_idx.x
    var tid = thread_idx.x
    var h_prev = stack_allocation[DType.float32,
        address_space=AddressSpace.SHARED](row_major[HIDDEN]())
    var c_prev = stack_allocation[DType.float32,
        address_space=AddressSpace.SHARED](row_major[HIDDEN]())
    var hh = tid
    while hh < HIDDEN:
        h_prev[hh] = Float32(0.0)
        c_prev[hh] = Float32(0.0)
        hh += BLOCK
    barrier()

    for t in range(time):
        var h2 = tid
        while h2 < HIDDEN:
            var pre_i: Float32 = rebind[Scalar[dtype]](b_ih[h2]).cast[DType.float32]() + rebind[Scalar[dtype]](b_hh[h2]).cast[DType.float32]()
            var pre_f: Float32 = rebind[Scalar[dtype]](b_ih[HIDDEN + h2]).cast[DType.float32]() + rebind[Scalar[dtype]](b_hh[HIDDEN + h2]).cast[DType.float32]()
            var pre_g: Float32 = rebind[Scalar[dtype]](b_ih[2 * HIDDEN + h2]).cast[DType.float32]() + rebind[Scalar[dtype]](b_hh[2 * HIDDEN + h2]).cast[DType.float32]()
            var pre_o: Float32 = rebind[Scalar[dtype]](b_ih[3 * HIDDEN + h2]).cast[DType.float32]() + rebind[Scalar[dtype]](b_hh[3 * HIDDEN + h2]).cast[DType.float32]()
            for i in range(INPUT):
                var xv = rebind[Scalar[dtype]](x[b, t, i]).cast[DType.float32]()
                pre_i += xv * rebind[Scalar[dtype]](w_ih[h2, i]).cast[DType.float32]()
                pre_f += xv * rebind[Scalar[dtype]](w_ih[HIDDEN + h2, i]).cast[DType.float32]()
                pre_g += xv * rebind[Scalar[dtype]](w_ih[2 * HIDDEN + h2, i]).cast[DType.float32]()
                pre_o += xv * rebind[Scalar[dtype]](w_ih[3 * HIDDEN + h2, i]).cast[DType.float32]()
            for j in range(HIDDEN):
                var hv = rebind[Scalar[DType.float32]](h_prev[j])
                pre_i += hv * rebind[Scalar[dtype]](w_hh[h2, j]).cast[DType.float32]()
                pre_f += hv * rebind[Scalar[dtype]](w_hh[HIDDEN + h2, j]).cast[DType.float32]()
                pre_g += hv * rebind[Scalar[dtype]](w_hh[2 * HIDDEN + h2, j]).cast[DType.float32]()
                pre_o += hv * rebind[Scalar[dtype]](w_hh[3 * HIDDEN + h2, j]).cast[DType.float32]()
            var i_g: Float32 = 1.0 / (1.0 + exp(-pre_i))
            var f_g: Float32 = 1.0 / (1.0 + exp(-pre_f))
            var g_g: Float32 = tanh(pre_g)
            var o_g: Float32 = 1.0 / (1.0 + exp(-pre_o))
            var c_p: Float32 = rebind[Scalar[DType.float32]](c_prev[h2])
            var c_n: Float32 = f_g * c_p + i_g * g_g
            var h_n: Float32 = o_g * tanh(c_n)
            c_prev[h2] = c_n
            h_out[b, t, h2] = rebind[h_out.ElementType](h_n.cast[dtype]())
            h2 += BLOCK
        barrier()
        var h3 = tid
        while h3 < HIDDEN:
            h_prev[h3] = rebind[Scalar[DType.float32]](h_out[b, t, h3].cast[DType.float32]())
            h3 += BLOCK
        barrier()


def lstm_cell_kernel[
    dtype: DType,
    XLayout: TensorLayout,
    HLayout: TensorLayout,
    WIHLayout: TensorLayout,
    WHHLayout: TensorLayout,
    BLayout: TensorLayout,
    HIDDEN: Int,
    INPUT: Int,
    BLOCK: Int,
](
    h_new: TileTensor[dtype, HLayout, MutAnyOrigin],     # (B, HIDDEN)
    c_new: TileTensor[dtype, HLayout, MutAnyOrigin],     # (B, HIDDEN)
    x: TileTensor[dtype, XLayout, MutAnyOrigin],          # (B, INPUT)
    h_prev: TileTensor[dtype, HLayout, MutAnyOrigin],     # (B, HIDDEN)
    c_prev: TileTensor[dtype, HLayout, MutAnyOrigin],     # (B, HIDDEN)
    w_ih: TileTensor[dtype, WIHLayout, MutAnyOrigin],     # (4*HIDDEN, INPUT)
    w_hh: TileTensor[dtype, WHHLayout, MutAnyOrigin],     # (4*HIDDEN, HIDDEN)
    b_ih: TileTensor[dtype, BLayout, MutAnyOrigin],       # (4*HIDDEN,)
    b_hh: TileTensor[dtype, BLayout, MutAnyOrigin],       # (4*HIDDEN,)
    batch: Int,
):
    """One LSTM cell step. Launch: grid = B, block_dim = BLOCK (thread strides over HIDDEN)."""
    comptime assert x.flat_rank == 2
    comptime assert h_prev.flat_rank == 2
    comptime assert c_prev.flat_rank == 2
    comptime assert h_new.flat_rank == 2
    comptime assert c_new.flat_rank == 2
    comptime assert w_ih.flat_rank == 2
    comptime assert w_hh.flat_rank == 2
    comptime assert b_ih.flat_rank == 1
    comptime assert b_hh.flat_rank == 1

    var b = block_idx.x
    var tid = thread_idx.x
    var h = tid
    while h < HIDDEN:
        # Compute pre_gates for the 4 gates at position h.
        var pre_i: Float32 = rebind[Scalar[dtype]](b_ih[h]).cast[DType.float32]() + rebind[Scalar[dtype]](b_hh[h]).cast[DType.float32]()
        var pre_f: Float32 = rebind[Scalar[dtype]](b_ih[HIDDEN + h]).cast[DType.float32]() + rebind[Scalar[dtype]](b_hh[HIDDEN + h]).cast[DType.float32]()
        var pre_g: Float32 = rebind[Scalar[dtype]](b_ih[2 * HIDDEN + h]).cast[DType.float32]() + rebind[Scalar[dtype]](b_hh[2 * HIDDEN + h]).cast[DType.float32]()
        var pre_o: Float32 = rebind[Scalar[dtype]](b_ih[3 * HIDDEN + h]).cast[DType.float32]() + rebind[Scalar[dtype]](b_hh[3 * HIDDEN + h]).cast[DType.float32]()

        # Accumulate x @ W_ih.T (W_ih is (4H, INPUT), W_ih[row, i]).
        for i in range(INPUT):
            var xv = rebind[Scalar[dtype]](x[b, i]).cast[DType.float32]()
            pre_i += xv * rebind[Scalar[dtype]](w_ih[h, i]).cast[DType.float32]()
            pre_f += xv * rebind[Scalar[dtype]](w_ih[HIDDEN + h, i]).cast[DType.float32]()
            pre_g += xv * rebind[Scalar[dtype]](w_ih[2 * HIDDEN + h, i]).cast[DType.float32]()
            pre_o += xv * rebind[Scalar[dtype]](w_ih[3 * HIDDEN + h, i]).cast[DType.float32]()
        # Accumulate h @ W_hh.T.
        for j in range(HIDDEN):
            var hv = rebind[Scalar[dtype]](h_prev[b, j]).cast[DType.float32]()
            pre_i += hv * rebind[Scalar[dtype]](w_hh[h, j]).cast[DType.float32]()
            pre_f += hv * rebind[Scalar[dtype]](w_hh[HIDDEN + h, j]).cast[DType.float32]()
            pre_g += hv * rebind[Scalar[dtype]](w_hh[2 * HIDDEN + h, j]).cast[DType.float32]()
            pre_o += hv * rebind[Scalar[dtype]](w_hh[3 * HIDDEN + h, j]).cast[DType.float32]()

        # Activations.
        var i_g: Float32 = 1.0 / (1.0 + exp(-pre_i))
        var f_g: Float32 = 1.0 / (1.0 + exp(-pre_f))
        var g_g: Float32 = tanh(pre_g)
        var o_g: Float32 = 1.0 / (1.0 + exp(-pre_o))

        # New cell state.
        var c_p: Float32 = rebind[Scalar[dtype]](c_prev[b, h]).cast[DType.float32]()
        var c_n: Float32 = f_g * c_p + i_g * g_g
        var h_n: Float32 = o_g * tanh(c_n)

        c_new[b, h] = rebind[c_new.ElementType](c_n.cast[dtype]())
        h_new[b, h] = rebind[h_new.ElementType](h_n.cast[dtype]())
        h += BLOCK


def lstm_layer_first_kernel[
    dtype: DType,
    XLayout: TensorLayout,
    HSeqLayout: TensorLayout,
    WIHLayout: TensorLayout,
    WHHLayout: TensorLayout,
    BLayout: TensorLayout,
    HIDDEN: Int,
    INPUT: Int,
    BLOCK: Int,
](
    h_out: TileTensor[dtype, HSeqLayout, MutAnyOrigin],   # (B, T, HIDDEN) — sequence of hidden states
    x: TileTensor[dtype, XLayout, MutAnyOrigin],           # (B, T, INPUT)
    w_ih: TileTensor[dtype, WIHLayout, MutAnyOrigin],
    w_hh: TileTensor[dtype, WHHLayout, MutAnyOrigin],
    b_ih: TileTensor[dtype, BLayout, MutAnyOrigin],
    b_hh: TileTensor[dtype, BLayout, MutAnyOrigin],
    batch: Int, time: Int,
):
    """Single LSTM layer over a full sequence. h_prev and c_prev start as zeros.
    Launch: grid=B, block_dim=BLOCK. Sequential along T inside the kernel.
    """
    comptime assert x.flat_rank == 3
    comptime assert h_out.flat_rank == 3
    comptime assert w_ih.flat_rank == 2
    comptime assert w_hh.flat_rank == 2
    comptime assert b_ih.flat_rank == 1
    comptime assert b_hh.flat_rank == 1
    from std.gpu.memory import AddressSpace
    from layout import stack_allocation
    var b = block_idx.x
    var tid = thread_idx.x

    # Shared memory for h_prev/c_prev per timestep.
    var h_prev = stack_allocation[DType.float32,
        address_space=AddressSpace.SHARED](row_major[HIDDEN]())
    var c_prev = stack_allocation[DType.float32,
        address_space=AddressSpace.SHARED](row_major[HIDDEN]())
    # Initialize to zero.
    var hh = tid
    while hh < HIDDEN:
        h_prev[hh] = Float32(0.0)
        c_prev[hh] = Float32(0.0)
        hh += BLOCK
    from std.gpu.sync import barrier
    barrier()

    for t in range(time):
        # Each thread computes a slice of HIDDEN dim.
        var h2 = tid
        while h2 < HIDDEN:
            var pre_i: Float32 = rebind[Scalar[dtype]](b_ih[h2]).cast[DType.float32]() + rebind[Scalar[dtype]](b_hh[h2]).cast[DType.float32]()
            var pre_f: Float32 = rebind[Scalar[dtype]](b_ih[HIDDEN + h2]).cast[DType.float32]() + rebind[Scalar[dtype]](b_hh[HIDDEN + h2]).cast[DType.float32]()
            var pre_g: Float32 = rebind[Scalar[dtype]](b_ih[2 * HIDDEN + h2]).cast[DType.float32]() + rebind[Scalar[dtype]](b_hh[2 * HIDDEN + h2]).cast[DType.float32]()
            var pre_o: Float32 = rebind[Scalar[dtype]](b_ih[3 * HIDDEN + h2]).cast[DType.float32]() + rebind[Scalar[dtype]](b_hh[3 * HIDDEN + h2]).cast[DType.float32]()
            for i in range(INPUT):
                var xv = rebind[Scalar[dtype]](x[b, t, i]).cast[DType.float32]()
                pre_i += xv * rebind[Scalar[dtype]](w_ih[h2, i]).cast[DType.float32]()
                pre_f += xv * rebind[Scalar[dtype]](w_ih[HIDDEN + h2, i]).cast[DType.float32]()
                pre_g += xv * rebind[Scalar[dtype]](w_ih[2 * HIDDEN + h2, i]).cast[DType.float32]()
                pre_o += xv * rebind[Scalar[dtype]](w_ih[3 * HIDDEN + h2, i]).cast[DType.float32]()
            for j in range(HIDDEN):
                var hv = rebind[Scalar[DType.float32]](h_prev[j])
                pre_i += hv * rebind[Scalar[dtype]](w_hh[h2, j]).cast[DType.float32]()
                pre_f += hv * rebind[Scalar[dtype]](w_hh[HIDDEN + h2, j]).cast[DType.float32]()
                pre_g += hv * rebind[Scalar[dtype]](w_hh[2 * HIDDEN + h2, j]).cast[DType.float32]()
                pre_o += hv * rebind[Scalar[dtype]](w_hh[3 * HIDDEN + h2, j]).cast[DType.float32]()

            var i_g: Float32 = 1.0 / (1.0 + exp(-pre_i))
            var f_g: Float32 = 1.0 / (1.0 + exp(-pre_f))
            var g_g: Float32 = tanh(pre_g)
            var o_g: Float32 = 1.0 / (1.0 + exp(-pre_o))
            var c_p: Float32 = rebind[Scalar[DType.float32]](c_prev[h2])
            var c_n: Float32 = f_g * c_p + i_g * g_g
            var h_n: Float32 = o_g * tanh(c_n)
            # Write new c into shared (deferred — must barrier before write to h_prev).
            c_prev[h2] = c_n
            # Save h_n into output.
            h_out[b, t, h2] = rebind[h_out.ElementType](h_n.cast[dtype]())
            h2 += BLOCK
        barrier()
        # Update h_prev for next timestep.
        var h3 = tid
        while h3 < HIDDEN:
            h_prev[h3] = rebind[Scalar[DType.float32]](h_out[b, t, h3].cast[DType.float32]())
            h3 += BLOCK
        barrier()
