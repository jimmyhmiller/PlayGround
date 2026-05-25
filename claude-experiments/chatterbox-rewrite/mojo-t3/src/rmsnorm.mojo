"""
RMSNorm GPU kernel for the Llama backbone of Chatterbox T3.

Spec (matches transformers.LlamaRMSNorm exactly):
  input dtype is `dtype` (fp32 or bf16). Variance is always computed in fp32:
    x_fp32   = x.cast[fp32]
    inv_rms  = 1 / sqrt(mean(x_fp32^2) + eps)        # fp32
    x_norm   = (x_fp32 * inv_rms).cast[dtype]        # back to input dtype
    out      = weight * x_norm                       # in input dtype

Layout: inp/out are 2D (rows, hidden), weight is 1D (hidden,). Same dtype.

Launch: one block per row, BLOCK threads per block. Each thread strides over
hidden/BLOCK elements. Reduction is two-stage:
  1. warp.sum within each warp → one partial per warp
  2. warp 0 sums those partials and broadcasts inv_rms via shared memory
"""

from std.math import sqrt
from std.gpu import barrier, block_idx, thread_idx, lane_id, WARP_SIZE
from std.gpu.primitives import warp
from std.gpu.memory import AddressSpace
from layout import TileTensor, TensorLayout, row_major, stack_allocation


def rmsnorm_kernel[
    dtype: DType,
    InpLayout: TensorLayout,
    WgtLayout: TensorLayout,
    OutLayout: TensorLayout,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],
    inp: TileTensor[dtype, InpLayout, MutAnyOrigin],
    weight: TileTensor[dtype, WgtLayout, MutAnyOrigin],
    eps: Float32,
):
    comptime assert inp.flat_rank == 2, "expected 2D input"
    comptime assert output.flat_rank == 2, "expected 2D output"
    comptime assert weight.flat_rank == 1, "expected 1D weight"
    comptime NUM_WARPS = BLOCK // WARP_SIZE

    var row = block_idx.x
    var tid = thread_idx.x
    var hidden = Int(inp.dim[1]())

    # Stage 1: per-thread partial sum-of-squares.
    # Always accumulate in fp32 — bf16's 7 mantissa bits would lose precision
    # over a 1024-element reduction.
    var partial: Float32 = 0.0
    var j = tid
    while j < hidden:
        var v = rebind[Scalar[dtype]](inp[row, j]).cast[DType.float32]()
        partial += v * v
        j += BLOCK

    # Stage 2: warp-level reduction.
    var warp_sum = warp.sum(partial)

    # Stage 3: cross-warp reduction in shared memory.
    var warp_sums = stack_allocation[
        DType.float32, address_space=AddressSpace.SHARED
    ](row_major[NUM_WARPS]())
    if lane_id() == 0:
        warp_sums[tid // WARP_SIZE] = warp_sum
    barrier()

    # Stage 4: warp 0 sums the per-warp values, computes inv_rms, broadcasts.
    var inv_rms_shared = stack_allocation[
        DType.float32, address_space=AddressSpace.SHARED
    ](row_major[1]())
    if tid < WARP_SIZE:
        var v: Float32 = 0.0
        if tid < NUM_WARPS:
            v = rebind[Scalar[DType.float32]](warp_sums[tid])
        var total = warp.sum(v)
        if tid == 0:
            var variance = total / Float32(hidden)
            inv_rms_shared[0] = 1.0 / sqrt(variance + eps)
    barrier()

    var inv_rms = rebind[Scalar[DType.float32]](inv_rms_shared[0])

    # Stage 5: normalize and write.
    # Cast x to fp32 for the inv_rms multiply, then cast back to input dtype
    # before the weight multiply (HF does the weight mul in input dtype).
    j = tid
    while j < hidden:
        var v_fp32 = rebind[Scalar[dtype]](inp[row, j]).cast[DType.float32]()
        var x_norm = (v_fp32 * inv_rms).cast[dtype]()
        var w = rebind[Scalar[dtype]](weight[j])
        output[row, j] = rebind[output.ElementType](w * x_norm)
        j += BLOCK
