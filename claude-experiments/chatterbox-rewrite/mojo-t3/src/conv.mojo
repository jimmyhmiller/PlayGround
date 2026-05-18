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
from std.math import sin, cos, exp
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
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, C, T)
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],        # (B, C, T)
    alpha: TileTensor[dtype, AlphaLayout, MutAnyOrigin],   # (C,)
    batch: Int, channels: Int, length: Int,
):
    """Snake activation:
        y = x + (1 / (alpha[c] + 1e-9)) * sin(x * alpha[c])^2

    Used by HiFiGAN ResBlocks. alpha is per-channel, learned.

    Launch: grid = B * C, block_dim = BLOCK. Each thread processes
    `length / BLOCK` (rounded up) time steps via strided iteration so we
    can handle T > 1024 (the AMD GPU block_dim limit).
    """
    comptime assert inp.flat_rank == 3
    comptime assert alpha.flat_rank == 1
    comptime assert output.flat_rank == 3

    var bid = block_idx.x
    var tid = thread_idx.x

    var c = bid % channels
    var b = bid // channels

    var a = rebind[Scalar[dtype]](alpha[c]).cast[DType.float32]()
    var inv_a = 1.0 / (a + 1.0e-9)

    var t = tid
    while t < length:
        var x_val = rebind[Scalar[dtype]](inp[b, c, t]).cast[DType.float32]()
        var s = sin(x_val * a)
        var y = x_val + inv_a * s * s
        output[b, c, t] = rebind[output.ElementType](y.cast[dtype]())
        t += BLOCK


def magnitude_phase_split_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    MLayout: TensorLayout,
    PLayout: TensorLayout,
    N_FREQ: Int,
    SECOND_HALF_OFFSET: Int,    # = N_FREQ
](
    magnitude: TileTensor[dtype, MLayout, MutAnyOrigin],     # (B, N_FREQ, T)
    phase: TileTensor[dtype, PLayout, MutAnyOrigin],          # (B, N_FREQ, T)
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],            # (B, 2*N_FREQ, T)
    batch: Int, time: Int,
):
    """For HiFiGAN's conv_post output:
        magnitude[b,k,t] = exp(inp[b, k, t])              for k in [0, N_FREQ)
        phase[b,k,t]     = sin(inp[b, N_FREQ + k, t])     for k in [0, N_FREQ)

    Launch: grid = B * N_FREQ * T, block_dim = 1.
    Also clips magnitude at 1e2 to match upstream's `torch.clip(magnitude, max=1e2)`.
    """
    comptime assert inp.flat_rank == 3
    comptime assert magnitude.flat_rank == 3
    comptime assert phase.flat_rank == 3

    var idx = block_idx.x
    var t = idx % time
    var k = (idx // time) % N_FREQ
    var b = idx // (time * N_FREQ)

    var m_in = rebind[Scalar[dtype]](inp[b, k, t]).cast[DType.float32]()
    var p_in = rebind[Scalar[dtype]](inp[b, SECOND_HALF_OFFSET + k, t]).cast[DType.float32]()

    var m_val = exp(m_in)
    if m_val > 100.0:
        m_val = 100.0
    var p_val = sin(p_in)

    magnitude[b, k, t] = rebind[magnitude.ElementType](m_val.cast[dtype]())
    phase[b, k, t] = rebind[phase.ElementType](p_val.cast[dtype]())


def magnitude_phase_to_complex_kernel[
    dtype: DType,
    MLayout: TensorLayout,
    PLayout: TensorLayout,
    RLayout: TensorLayout,
    ILayout: TensorLayout,
](
    real_out: TileTensor[dtype, RLayout, MutAnyOrigin],     # (B, N_FREQ, T)
    imag_out: TileTensor[dtype, ILayout, MutAnyOrigin],     # (B, N_FREQ, T)
    magnitude: TileTensor[dtype, MLayout, MutAnyOrigin],   # (B, N_FREQ, T)
    phase: TileTensor[dtype, PLayout, MutAnyOrigin],        # (B, N_FREQ, T)
    batch: Int, n_freq: Int, time: Int,
):
    """real = magnitude * cos(phase), imag = magnitude * sin(phase).
    Used to bridge HiFiGAN's magnitude/phase output into istft_kernel's
    real/imag input. Launch: grid = B*N_FREQ*T, block_dim = 1.
    """
    comptime assert real_out.flat_rank == 3
    comptime assert imag_out.flat_rank == 3
    comptime assert magnitude.flat_rank == 3
    comptime assert phase.flat_rank == 3

    var idx = block_idx.x
    var t = idx % time
    var k = (idx // time) % n_freq
    var b = idx // (time * n_freq)

    var m = rebind[Scalar[dtype]](magnitude[b, k, t]).cast[DType.float32]()
    var p = rebind[Scalar[dtype]](phase[b, k, t]).cast[DType.float32]()
    var re = m * cos(p)
    var im = m * sin(p)
    real_out[b, k, t] = rebind[real_out.ElementType](re.cast[dtype]())
    imag_out[b, k, t] = rebind[imag_out.ElementType](im.cast[dtype]())


def reflection_pad_left1_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    OutLayout: TensorLayout,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, C, T_in + 1)
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],        # (B, C, T_in)
    batch: Int, channels: Int, t_in: Int,
):
    """ReflectionPad1d((1, 0)): pad 1 on the left via reflection.
    out[b,c,0]      = in[b,c,1]
    out[b,c,1..]    = in[b,c,0..T-1]

    Launch: grid = B * C * (T_in + 1), block_dim = 1.
    """
    comptime assert inp.flat_rank == 3
    comptime assert output.flat_rank == 3

    var idx = block_idx.x
    var t_out = idx % (t_in + 1)
    var c = (idx // (t_in + 1)) % channels
    var b = idx // ((t_in + 1) * channels)

    var src_t: Int
    if t_out == 0:
        src_t = 1
    else:
        src_t = t_out - 1
    var v = rebind[Scalar[dtype]](inp[b, c, src_t])
    output[b, c, t_out] = rebind[output.ElementType](v)


def bias_add_2d_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    BiasLayout: TensorLayout,
    OutLayout: TensorLayout,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],     # (R, C)
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],          # (R, C)
    bias: TileTensor[dtype, BiasLayout, MutAnyOrigin],       # (C,)
    rows: Int, cols: Int,
):
    """Add a per-column bias to a 2D tensor: out[r, c] = inp[r, c] + bias[c].

    Launch: grid = ceildiv(rows * cols, BLOCK), block_dim = BLOCK.
    """
    comptime assert inp.flat_rank == 2
    comptime assert bias.flat_rank == 1
    comptime assert output.flat_rank == 2

    var idx = block_idx.x * BLOCK + thread_idx.x
    var n = rows * cols
    if idx >= n:
        return
    var r = idx // cols
    var c = idx % cols
    var x = rebind[Scalar[dtype]](inp[r, c]).cast[DType.float32]()
    var b = rebind[Scalar[dtype]](bias[c]).cast[DType.float32]()
    var y = x + b
    output[r, c] = rebind[output.ElementType](y.cast[dtype]())


def elu_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    OutLayout: TensorLayout,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],
    n_elems: Int,
    alpha: Float32,
):
    """Pointwise ELU: y = x if x > 0 else alpha * (exp(x) - 1).

    Matches torch.nn.functional.elu(x, alpha=1.0). Operates on a rank-1
    view of any buffer.
    Launch: grid = ceildiv(n_elems, BLOCK), block_dim = BLOCK.
    """
    comptime assert inp.flat_rank == 1
    comptime assert output.flat_rank == 1

    var idx = block_idx.x * BLOCK + thread_idx.x
    if idx >= n_elems:
        return

    var v = rebind[Scalar[dtype]](inp[idx]).cast[DType.float32]()
    var y = v if v > 0.0 else alpha * (exp(v) - 1.0)
    output[idx] = rebind[output.ElementType](y.cast[dtype]())


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
