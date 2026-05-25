"""
Kernels for the CFM ConditionalDecoder (estimator):
  group_norm_kernel       — GroupNorm over (B, C, T) with 8 groups.
  mish_kernel             — y = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x)).
  gelu_kernel             — y = 0.5 * x * (1 + erf(x / sqrt(2)))
  geglu_proj_split_kernel — Linear(D, 2*F) then split into (out, gate), output = out * gelu(gate)
  multiply_mask_3d_kernel — out = x * mask (broadcast mask over channels)
  sinusoidal_pos_emb_kernel — produce sinusoidal position embedding for a scalar t.
  conv1d_causal_left_pad_kernel — Convenience for causal Conv1d (left-pad k-1, no right pad).

We also reuse `linear_kernel`, `layernorm_kernel`, `add_4d_kernel`, `swish_kernel`, etc.
from existing modules.
"""
from std.math import sqrt, exp, tanh, log, cos, sin, pi, erf
from std.gpu import block_idx, thread_idx
from std.gpu.sync import barrier
from std.gpu.memory import AddressSpace
from layout import TileTensor, TensorLayout, row_major, stack_allocation


def group_norm_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    PLayout: TensorLayout,
    OutLayout: TensorLayout,
    GROUPS: Int,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, C, T)
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],        # (B, C, T)
    weight: TileTensor[dtype, PLayout, MutAnyOrigin],      # (C,)
    bias: TileTensor[dtype, PLayout, MutAnyOrigin],        # (C,)
    batch: Int, channels: Int, time: Int, eps: Float32,
):
    """Compute mean+var over (C/GROUPS, T) per group, then normalize and apply affine.

    Launch: grid = B * GROUPS, block_dim = BLOCK. Threads cooperate via shared memory.
    """
    comptime assert inp.flat_rank == 3
    comptime assert output.flat_rank == 3
    comptime assert weight.flat_rank == 1

    var bid = block_idx.x
    var tid = thread_idx.x
    var g = bid % GROUPS
    var b = bid // GROUPS

    var ch_per_group = channels // GROUPS
    var group_size = ch_per_group * time
    var c_start = g * ch_per_group

    var smem = stack_allocation[DType.float32,
        address_space=AddressSpace.SHARED](row_major[BLOCK]())

    # Pass 1: sum.
    var s: Float32 = 0.0
    var idx = tid
    while idx < group_size:
        var c = c_start + idx // time
        var t = idx % time
        s += rebind[Scalar[dtype]](inp[b, c, t]).cast[DType.float32]()
        idx += BLOCK
    smem[tid] = s
    barrier()
    if tid == 0:
        var total: Float32 = 0.0
        for i in range(BLOCK):
            total += rebind[Scalar[DType.float32]](smem[i])
        smem[0] = total / Float32(group_size)
    barrier()
    var mean_val = rebind[Scalar[DType.float32]](smem[0])

    # Pass 2: variance.
    var s2: Float32 = 0.0
    var idx2 = tid
    while idx2 < group_size:
        var c = c_start + idx2 // time
        var t = idx2 % time
        var d = rebind[Scalar[dtype]](inp[b, c, t]).cast[DType.float32]() - mean_val
        s2 += d * d
        idx2 += BLOCK
    smem[tid] = s2
    barrier()
    if tid == 0:
        var total2: Float32 = 0.0
        for i in range(BLOCK):
            total2 += rebind[Scalar[DType.float32]](smem[i])
        smem[1] = total2 / Float32(group_size)
    barrier()
    var var_val = rebind[Scalar[DType.float32]](smem[1])
    var inv_std: Float32 = 1.0 / sqrt(var_val + eps)

    # Pass 3: write.
    var idx3 = tid
    while idx3 < group_size:
        var c = c_start + idx3 // time
        var t = idx3 % time
        var x = rebind[Scalar[dtype]](inp[b, c, t]).cast[DType.float32]()
        var g_w = rebind[Scalar[dtype]](weight[c]).cast[DType.float32]()
        var b_b = rebind[Scalar[dtype]](bias[c]).cast[DType.float32]()
        var y = (x - mean_val) * inv_std * g_w + b_b
        output[b, c, t] = rebind[output.ElementType](y.cast[dtype]())
        idx3 += BLOCK


def elu_kernel[
    dtype: DType, InLayout: TensorLayout, OutLayout: TensorLayout, BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],
    n: Int,
):
    """ELU: y = x if x > 0 else exp(x) - 1."""
    comptime assert inp.flat_rank == 1
    comptime assert output.flat_rank == 1
    var idx = block_idx.x * BLOCK + thread_idx.x
    if idx >= n: return
    var x = rebind[Scalar[dtype]](inp[idx]).cast[DType.float32]()
    var y: Float32 = x if x > 0.0 else (exp(x) - 1.0)
    output[idx] = rebind[output.ElementType](y.cast[dtype]())


def abs_kernel[
    dtype: DType, InLayout: TensorLayout, OutLayout: TensorLayout, BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],
    n: Int,
):
    """y = abs(x)."""
    comptime assert inp.flat_rank == 1
    comptime assert output.flat_rank == 1
    var idx = block_idx.x * BLOCK + thread_idx.x
    if idx >= n: return
    var x = rebind[Scalar[dtype]](inp[idx]).cast[DType.float32]()
    var y: Float32 = x if x >= 0.0 else -x
    output[idx] = rebind[output.ElementType](y.cast[dtype]())


def mish_kernel[
    dtype: DType, InLayout: TensorLayout, OutLayout: TensorLayout, BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],
    n: Int,
):
    """y = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x))). Operates on flat 1D buffer."""
    comptime assert inp.flat_rank == 1
    comptime assert output.flat_rank == 1
    var idx = block_idx.x * BLOCK + thread_idx.x
    if idx >= n: return
    var x = rebind[Scalar[dtype]](inp[idx]).cast[DType.float32]()
    # softplus(x). Use a numerically stable form: softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    var ax: Float32 = x
    if ax < 0.0: ax = -ax
    var max_x: Float32 = x
    if max_x < 0.0: max_x = 0.0
    var sp: Float32 = max_x + log(1.0 + exp(-ax))
    var y: Float32 = x * tanh(sp)
    output[idx] = rebind[output.ElementType](y.cast[dtype]())


def gelu_kernel[
    dtype: DType, InLayout: TensorLayout, OutLayout: TensorLayout, BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],
    n: Int,
):
    """Exact GELU: y = 0.5 * x * (1 + erf(x / sqrt(2)))."""
    comptime assert inp.flat_rank == 1
    comptime assert output.flat_rank == 1
    var idx = block_idx.x * BLOCK + thread_idx.x
    if idx >= n: return
    var x = rebind[Scalar[dtype]](inp[idx]).cast[DType.float32]()
    var inv_sqrt2: Float32 = 0.70710678118654752440
    var y: Float32 = 0.5 * x * (1.0 + erf(x * inv_sqrt2))
    output[idx] = rebind[output.ElementType](y.cast[dtype]())


def multiply_mask_3d_kernel[
    dtype: DType, InLayout: TensorLayout, MaskLayout: TensorLayout, OutLayout: TensorLayout,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, C, T)
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],        # (B, C, T)
    mask: TileTensor[dtype, MaskLayout, MutAnyOrigin],     # (B, 1, T)
    batch: Int, channels: Int, time: Int,
):
    """out[b, c, t] = inp[b, c, t] * mask[b, 0, t]."""
    comptime assert inp.flat_rank == 3
    comptime assert mask.flat_rank == 3
    comptime assert output.flat_rank == 3
    var bid = block_idx.x
    var tid = thread_idx.x
    var c = bid % channels
    var b = bid // channels
    var t = tid
    while t < time:
        var x = rebind[Scalar[dtype]](inp[b, c, t]).cast[DType.float32]()
        var m = rebind[Scalar[dtype]](mask[b, 0, t]).cast[DType.float32]()
        output[b, c, t] = rebind[output.ElementType]((x * m).cast[dtype]())
        t += BLOCK


def add_3d_time_emb_kernel[
    dtype: DType,
    InLayout: TensorLayout, TLayout: TensorLayout, OutLayout: TensorLayout,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, C, T)
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],        # (B, C, T)
    t_emb: TileTensor[dtype, TLayout, MutAnyOrigin],       # (B, C)
    batch: Int, channels: Int, time: Int,
):
    """out[b, c, t] = inp[b, c, t] + t_emb[b, c]. Used after `mlp(time_emb)` in ResnetBlock1D."""
    comptime assert inp.flat_rank == 3
    comptime assert t_emb.flat_rank == 2
    comptime assert output.flat_rank == 3
    var bid = block_idx.x
    var tid = thread_idx.x
    var c = bid % channels
    var b = bid // channels
    var bias = rebind[Scalar[dtype]](t_emb[b, c]).cast[DType.float32]()
    var t = tid
    while t < time:
        var v = rebind[Scalar[dtype]](inp[b, c, t]).cast[DType.float32]()
        output[b, c, t] = rebind[output.ElementType]((v + bias).cast[dtype]())
        t += BLOCK


def sinusoidal_pos_emb_kernel[
    dtype: DType, OutLayout: TensorLayout, TLayout: TensorLayout, DIM: Int, BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, DIM)
    t_in: TileTensor[dtype, TLayout, MutAnyOrigin],        # (B, 1) scalar per batch
    batch: Int,
    scale: Float32,
):
    """SinusoidalPosEmb(dim):
      emb_factor = log(10000) / (dim/2 - 1)
      coeffs = exp(arange(dim/2) * -emb_factor)
      out = cat(sin(scale*t*coeffs), cos(scale*t*coeffs))

    Launch: grid=B, block_dim=BLOCK over DIM/2.
    """
    comptime assert output.flat_rank == 2
    comptime assert t_in.flat_rank == 2
    var b = block_idx.x
    var tid = thread_idx.x
    var half = DIM // 2
    var t_val = rebind[Scalar[dtype]](t_in[b, 0]).cast[DType.float32]() * scale
    var emb_factor: Float32 = Float32(log(Float32(10000.0))) / (Float32(half) - 1.0)
    var i = tid
    while i < half:
        var coeff: Float32 = exp(-emb_factor * Float32(i))
        var arg: Float32 = t_val * coeff
        output[b, i] = rebind[output.ElementType](sin(arg).cast[dtype]())
        output[b, half + i] = rebind[output.ElementType](cos(arg).cast[dtype]())
        i += BLOCK
