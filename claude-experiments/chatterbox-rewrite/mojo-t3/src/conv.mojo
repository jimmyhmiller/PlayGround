"""
1-D convolution kernels for the S3Gen vocoder port.

  conv1d_kernel:         general conv1d (stride, padding, dilation, groups=1)
  transposed_conv1d_kernel: transposed conv1d for upsampling layers
  leaky_relu_kernel:     y = x if x > 0 else slope * x

These are deliberately simple, one-output-element-per-thread kernels. Vocoder
convs are small enough that this is fine; we can specialize later.

Tensor layouts:
  Input:   (B, C_in, L_in)
  Weight:  (C_out, C_in, K)
  Bias:    (C_out,)              (optional via separate add kernel if needed)
  Output:  (B, C_out, L_out)

For conv1d:
  L_out = (L_in + 2*pad - dilation*(K-1) - 1) // stride + 1

For transposed_conv1d (PyTorch nn.ConvTranspose1d):
  L_out = (L_in - 1) * stride - 2 * pad + dilation * (K - 1) + 1 + output_pad

We pass L_out, stride, padding, dilation as runtime args so the same kernel can
serve multiple HiFiGAN layers.
"""

from std.gpu import block_idx, thread_idx
from std.math import sin
from layout import TileTensor, TensorLayout


def conv1d_kernel[
    dtype: DType,
    XLayout: TensorLayout,
    WLayout: TensorLayout,
    BiasLayout: TensorLayout,
    OutLayout: TensorLayout,
    K: Int,           # kernel size, comptime
    HAS_BIAS: Bool,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, C_out, L_out)
    x: TileTensor[dtype, XLayout, MutAnyOrigin],           # (B, C_in, L_in)
    w: TileTensor[dtype, WLayout, MutAnyOrigin],           # (C_out, C_in, K)
    bias: TileTensor[dtype, BiasLayout, MutAnyOrigin],     # (C_out,) — pass dummy when HAS_BIAS=False
    batch: Int,
    c_in: Int,
    c_out: Int,
    l_in: Int,
    l_out: Int,
    stride: Int,
    padding: Int,
    dilation: Int,
):
    """out[b, co, lo] = bias[co] + sum_{ci, k} x[b, ci, lo*stride + k*dilation - padding] * w[co, ci, k].

    Launch: grid = B * C_out * L_out, block_dim = 1 (one thread per output).

    Realistically you'd want at least block_dim = WARP_SIZE for occupancy;
    we keep block_dim=1 for simplicity and let the grid be large. The
    first-implementation goal is correctness; tuning comes later.
    """
    comptime assert x.flat_rank == 3
    comptime assert w.flat_rank == 3
    comptime assert output.flat_rank == 3

    var idx = block_idx.x  # 0..B*C_out*L_out-1
    var lo = idx % l_out
    var co = (idx // l_out) % c_out
    var b = idx // (l_out * c_out)

    var acc: Float32 = 0.0
    if HAS_BIAS:
        comptime assert bias.flat_rank == 1
        acc = rebind[Scalar[dtype]](bias[co]).cast[DType.float32]()

    for ci in range(c_in):
        for k in range(K):
            var li = lo * stride + k * dilation - padding
            if li >= 0 and li < l_in:
                var xv = rebind[Scalar[dtype]](x[b, ci, li]).cast[DType.float32]()
                var wv = rebind[Scalar[dtype]](w[co, ci, k]).cast[DType.float32]()
                acc += xv * wv

    output[b, co, lo] = rebind[output.ElementType](acc.cast[dtype]())


def transposed_conv1d_kernel[
    dtype: DType,
    XLayout: TensorLayout,
    WLayout: TensorLayout,
    BiasLayout: TensorLayout,
    OutLayout: TensorLayout,
    K: Int,
    HAS_BIAS: Bool,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, C_out, L_out)
    x: TileTensor[dtype, XLayout, MutAnyOrigin],           # (B, C_in, L_in)
    w: TileTensor[dtype, WLayout, MutAnyOrigin],           # (C_in, C_out, K)
    bias: TileTensor[dtype, BiasLayout, MutAnyOrigin],     # (C_out,)
    batch: Int,
    c_in: Int,
    c_out: Int,
    l_in: Int,
    l_out: Int,
    stride: Int,
    padding: Int,
    dilation: Int,
):
    """Transposed conv1d, equivalent to torch.nn.ConvTranspose1d.

    out[b, co, lo] = bias[co]
                   + sum over (ci, li, k) such that
                     lo + padding == li * stride + k * dilation
                     of x[b, ci, li] * w[ci, co, k].

    Note the weight tensor is laid out (C_in, C_out, K) — the opposite of
    conv1d — matching PyTorch's nn.ConvTranspose1d.weight shape.

    Launch: grid = B * C_out * L_out, block_dim = 1.
    """
    comptime assert x.flat_rank == 3
    comptime assert w.flat_rank == 3
    comptime assert output.flat_rank == 3

    var idx = block_idx.x
    var lo = idx % l_out
    var co = (idx // l_out) % c_out
    var b = idx // (l_out * c_out)

    var acc: Float32 = 0.0
    if HAS_BIAS:
        comptime assert bias.flat_rank == 1
        acc = rebind[Scalar[dtype]](bias[co]).cast[DType.float32]()

    # For each (k, ci), check if there's a valid li satisfying
    # lo + padding = li * stride + k * dilation
    for ci in range(c_in):
        for k in range(K):
            var num = lo + padding - k * dilation
            if num >= 0 and (num % stride) == 0:
                var li = num // stride
                if li >= 0 and li < l_in:
                    var xv = rebind[Scalar[dtype]](x[b, ci, li]).cast[DType.float32]()
                    var wv = rebind[Scalar[dtype]](w[ci, co, k]).cast[DType.float32]()
                    acc += xv * wv

    output[b, co, lo] = rebind[output.ElementType](acc.cast[dtype]())


def snake_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    AlphaLayout: TensorLayout,
    OutLayout: TensorLayout,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, C, T)
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],        # (B, C, T)
    alpha: TileTensor[dtype, AlphaLayout, MutAnyOrigin],   # (C,)
    batch: Int, channels: Int, length: Int,
):
    """Snake activation:
        y = x + (1 / (alpha[c] + 1e-9)) * sin(x * alpha[c])^2

    Used by HiFiGAN ResBlocks. alpha is per-channel, learned.

    Launch: grid = B * C, block_dim = length (one thread per time step).
    Realistically we'd want length-parallel within a block; we keep
    block_dim = length for simplicity (sizes are modest).
    """
    comptime assert inp.flat_rank == 3
    comptime assert alpha.flat_rank == 1
    comptime assert output.flat_rank == 3

    var bid = block_idx.x
    var t = thread_idx.x

    var c = bid % channels
    var b = bid // channels

    if t >= length:
        return

    var x_val = rebind[Scalar[dtype]](inp[b, c, t]).cast[DType.float32]()
    var a = rebind[Scalar[dtype]](alpha[c]).cast[DType.float32]()
    var s = sin(x_val * a)
    var y = x_val + (1.0 / (a + 1.0e-9)) * s * s
    output[b, c, t] = rebind[output.ElementType](y.cast[dtype]())


def leaky_relu_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    OutLayout: TensorLayout,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],
    n_elems: Int,
    slope: Float32,
):
    """Pointwise leaky_relu: y = x if x > 0 else slope * x.

    Operates over the flattened buffer (rank-1 view).
    Launch: grid = ceildiv(n_elems, BLOCK), block_dim = BLOCK.
    """
    comptime assert inp.flat_rank == 1
    comptime assert output.flat_rank == 1

    var idx = block_idx.x * BLOCK + thread_idx.x
    if idx >= n_elems:
        return

    var v = rebind[Scalar[dtype]](inp[idx]).cast[DType.float32]()
    var y = v if v > 0.0 else slope * v
    output[idx] = rebind[output.ElementType](y.cast[dtype]())
