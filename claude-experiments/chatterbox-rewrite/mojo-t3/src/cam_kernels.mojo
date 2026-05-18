"""
Kernels needed by CAMPPlus CAMLayer and CAMDenseTDNNBlock:
  mean_along_t_kernel   — (B, C, T) -> (B, C, 1)   mean reduction over T
  seg_pool_kernel       — segment-mean pooling with seg_len=100, ceil_mode=True,
                          then upsample back to T by repeating each chunk
  sigmoid_kernel        — pointwise sigmoid
  broadcast_mul_t_kernel — out[b,c,t] = a[b,c,t] * m[b,c,t]
  add_ct_to_t_kernel    — out[b,c,t] = a[b,c,t] + b[b,c,0]   (broadcast over T)
  channel_concat_kernel — out[b,:cA,:] = A, out[b,cA:,:] = B
"""
from std.gpu import block_idx, thread_idx
from std.math import exp
from layout import TileTensor, TensorLayout


def mean_along_t_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    OutLayout: TensorLayout,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, C, 1)
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],        # (B, C, T)
    batch: Int, channels: Int, length: Int,
):
    """For each (b, c), compute mean(x[b, c, :]) and write to output[b, c, 0].

    Launch: grid = B * C, block_dim = BLOCK. Threads strided over T, then
    block-reduction via shared memory.
    """
    comptime assert inp.flat_rank == 3
    comptime assert output.flat_rank == 3

    var bid = block_idx.x
    var tid = thread_idx.x
    var c = bid % channels
    var b = bid // channels

    var acc: Float32 = 0.0
    var t = tid
    while t < length:
        acc += rebind[Scalar[dtype]](inp[b, c, t]).cast[DType.float32]()
        t += BLOCK

    # Naive: serial accumulation across lanes via shared memory.
    from std.gpu.memory import AddressSpace
    from layout import row_major, stack_allocation
    var smem = stack_allocation[DType.float32,
        address_space=AddressSpace.SHARED](row_major[BLOCK]())
    smem[tid] = acc
    from std.gpu.sync import barrier
    barrier()

    if tid == 0:
        var total: Float32 = 0.0
        for i in range(BLOCK):
            total += rebind[Scalar[DType.float32]](smem[i])
        var mean_val = total / Float32(length)
        output[b, c, 0] = rebind[output.ElementType](mean_val.cast[dtype]())


def seg_pool_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    OutLayout: TensorLayout,
    SEG_LEN: Int,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, C, T)
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],        # (B, C, T)
    batch: Int, channels: Int, length: Int,
):
    """Replicates F.avg_pool1d(x, kernel_size=SEG_LEN, stride=SEG_LEN, ceil_mode=True),
    then upsamples by repeating each chunk SEG_LEN times, then truncates back
    to original length.

    For ceil_mode=True with stride=kernel, the number of output segments is
    ceil(length / SEG_LEN). Each output segment averages SEG_LEN inputs, but
    the last segment may be partial (avg over fewer elements — torch's
    avg_pool1d divides by SEG_LEN even with ceil_mode if count_include_pad
    defaults are used... but actually for avg_pool1d the default
    count_include_pad=True but ceil_mode just changes output size). Replicates
    PyTorch's actual behavior: divides each segment by SEG_LEN regardless.

    Launch: grid = B * C, block_dim = BLOCK. Each thread strides over T.
    """
    comptime assert inp.flat_rank == 3
    comptime assert output.flat_rank == 3

    var bid = block_idx.x
    var tid = thread_idx.x
    var c = bid % channels
    var b = bid // channels

    # number of segments
    var n_seg = (length + SEG_LEN - 1) // SEG_LEN

    var t = tid
    while t < length:
        var seg_i = t // SEG_LEN
        if seg_i >= n_seg:
            seg_i = n_seg - 1
        var start = seg_i * SEG_LEN
        var end = start + SEG_LEN
        if end > length:
            end = length
        var acc: Float32 = 0.0
        for k in range(start, end):
            acc += rebind[Scalar[dtype]](inp[b, c, k]).cast[DType.float32]()
        # Empirically: PyTorch avg_pool1d with ceil_mode=True divides each
        # output segment by its *actual* number of valid input elements,
        # not by kernel_size (verified against torch reference).
        var mean_val = acc / Float32(end - start)
        output[b, c, t] = rebind[output.ElementType](mean_val.cast[dtype]())
        t += BLOCK


def sigmoid_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    OutLayout: TensorLayout,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],
    n_elems: Int,
):
    """Pointwise sigmoid(x) = 1 / (1 + exp(-x))."""
    comptime assert inp.flat_rank == 1
    comptime assert output.flat_rank == 1
    var idx = block_idx.x * BLOCK + thread_idx.x
    if idx >= n_elems:
        return
    var v = rebind[Scalar[dtype]](inp[idx]).cast[DType.float32]()
    var y: Float32 = 1.0 / (1.0 + exp(-v))
    output[idx] = rebind[output.ElementType](y.cast[dtype]())


def broadcast_mul_t_kernel[
    dtype: DType,
    ALayout: TensorLayout,
    MLayout: TensorLayout,
    OutLayout: TensorLayout,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, C, T)
    a: TileTensor[dtype, ALayout, MutAnyOrigin],           # (B, C, T)
    m: TileTensor[dtype, MLayout, MutAnyOrigin],           # (B, C, T)
    batch: Int, channels: Int, length: Int,
):
    """Element-wise out[b,c,t] = a[b,c,t] * m[b,c,t]."""
    comptime assert a.flat_rank == 3
    comptime assert m.flat_rank == 3
    comptime assert output.flat_rank == 3
    var bid = block_idx.x
    var tid = thread_idx.x
    var c = bid % channels
    var b = bid // channels
    var t = tid
    while t < length:
        var av = rebind[Scalar[dtype]](a[b, c, t]).cast[DType.float32]()
        var mv = rebind[Scalar[dtype]](m[b, c, t]).cast[DType.float32]()
        output[b, c, t] = rebind[output.ElementType]((av * mv).cast[dtype]())
        t += BLOCK


def add_t_with_bc1_kernel[
    dtype: DType,
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    OutLayout: TensorLayout,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, C, T)
    a: TileTensor[dtype, ALayout, MutAnyOrigin],           # (B, C, T)
    b_bc1: TileTensor[dtype, BLayout, MutAnyOrigin],       # (B, C, 1)
    batch: Int, channels: Int, length: Int,
):
    """out[b,c,t] = a[b,c,t] + b_bc1[b,c,0]   (broadcast over T)."""
    comptime assert a.flat_rank == 3
    comptime assert b_bc1.flat_rank == 3
    comptime assert output.flat_rank == 3
    var bid = block_idx.x
    var tid = thread_idx.x
    var c = bid % channels
    var b = bid // channels
    var bias_v = rebind[Scalar[dtype]](b_bc1[b, c, 0]).cast[DType.float32]()
    var t = tid
    while t < length:
        var av = rebind[Scalar[dtype]](a[b, c, t]).cast[DType.float32]()
        output[b, c, t] = rebind[output.ElementType]((av + bias_v).cast[dtype]())
        t += BLOCK


def channel_concat_kernel[
    dtype: DType,
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    OutLayout: TensorLayout,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, C_A + C_B, T)
    a: TileTensor[dtype, ALayout, MutAnyOrigin],           # (B, C_A, T)
    b: TileTensor[dtype, BLayout, MutAnyOrigin],           # (B, C_B, T)
    batch: Int, c_a: Int, c_b: Int, length: Int,
):
    """out[:, :c_a, :] = a; out[:, c_a:, :] = b. Launch grid = B*(c_a+c_b)."""
    comptime assert a.flat_rank == 3
    comptime assert b.flat_rank == 3
    comptime assert output.flat_rank == 3
    var bid = block_idx.x
    var tid = thread_idx.x
    var c_total = c_a + c_b
    var c = bid % c_total
    var ba = bid // c_total
    var t = tid
    while t < length:
        if c < c_a:
            var av = rebind[Scalar[dtype]](a[ba, c, t]).cast[DType.float32]()
            output[ba, c, t] = rebind[output.ElementType](av.cast[dtype]())
        else:
            var bv = rebind[Scalar[dtype]](b[ba, c - c_a, t]).cast[DType.float32]()
            output[ba, c, t] = rebind[output.ElementType](bv.cast[dtype]())
        t += BLOCK
