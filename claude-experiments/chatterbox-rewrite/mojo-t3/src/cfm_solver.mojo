"""
CFM Euler step kernel — combines CFG mix with Euler step.

  out[b, c, t] = x[b, c, t] + dt * ((1 + cfg) * dxdt[b] - cfg * dxdt[B+b])
"""
from std.gpu import block_idx, thread_idx
from layout import TileTensor, TensorLayout, row_major


def cfm_euler_step_kernel[
    dtype: DType,
    XLayout: TensorLayout,
    EstLayout: TensorLayout,
    OutLayout: TensorLayout,
    B: Int, C: Int, T: Int,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],     # (B, C, T)
    x: TileTensor[dtype, XLayout, MutAnyOrigin],             # (B, C, T)
    estimator_out: TileTensor[dtype, EstLayout, MutAnyOrigin],  # (2*B, C, T)
    dt: Float32,
    cfg_rate: Float32,
):
    """x_new = x + dt * ((1+r)*dxdt - r*cfg_dxdt) where dxdt = est[:B], cfg_dxdt = est[B:]."""
    comptime assert x.flat_rank == 3
    comptime assert estimator_out.flat_rank == 3
    comptime assert output.flat_rank == 3

    var idx = block_idx.x * BLOCK + thread_idx.x
    var n = B * C * T
    if idx >= n: return

    var t = idx % T
    var c = (idx // T) % C
    var b = idx // (T * C)

    var xv = rebind[Scalar[dtype]](x[b, c, t]).cast[DType.float32]()
    var d_cond = rebind[Scalar[dtype]](estimator_out[b, c, t]).cast[DType.float32]()
    var d_uncond = rebind[Scalar[dtype]](estimator_out[b + B, c, t]).cast[DType.float32]()
    var dxdt = (1.0 + cfg_rate) * d_cond - cfg_rate * d_uncond
    var y = xv + dt * dxdt
    output[b, c, t] = rebind[output.ElementType](y.cast[dtype]())


def build_cfg_inputs_kernel[
    dtype: DType,
    SrcLayout: TensorLayout, OutLayout: TensorLayout,
    B: Int, C: Int, T: Int, BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (2*B, C, T)
    src: TileTensor[dtype, SrcLayout, MutAnyOrigin],       # (B, C, T)
    zero_uncond: Int,   # 1 = zero out upper half (used for mu, spks, cond); 0 = duplicate (used for x, mask, t)
):
    """Builds CFG-doubled input: first half = src, second half = src (zero_uncond=0) or zeros (zero_uncond=1).

    Launch: grid = 2 * B * C, block_dim = BLOCK over T.
    """
    comptime assert src.flat_rank == 3
    comptime assert output.flat_rank == 3

    var bid = block_idx.x
    var tid = thread_idx.x
    var c = bid % C
    var b2 = bid // C
    var b = b2 % B
    var is_uncond = b2 >= B

    var t = tid
    while t < T:
        var v: Float32 = 0.0
        if not is_uncond:
            v = rebind[Scalar[dtype]](src[b, c, t]).cast[DType.float32]()
        elif zero_uncond == 0:
            # Duplicate (used for x, mask): read the conditional copy.
            v = rebind[Scalar[dtype]](src[b, c, t]).cast[DType.float32]()
        # else: zero (default v=0).
        output[b2, c, t] = rebind[output.ElementType](v.cast[dtype]())
        t += BLOCK


def pack_xmsc_kernel[
    dtype: DType, OutLayout: TensorLayout,
    XLayout: TensorLayout, MuLayout: TensorLayout,
    SpksLayout: TensorLayout, CondLayout: TensorLayout,
    BLOCK_: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, 320, T)
    x_in: TileTensor[dtype, XLayout, MutAnyOrigin],
    mu_in: TileTensor[dtype, MuLayout, MutAnyOrigin],
    spks_in: TileTensor[dtype, SpksLayout, MutAnyOrigin],
    cond_in: TileTensor[dtype, CondLayout, MutAnyOrigin],
    batch: Int, time: Int,
):
    """Pack [x, mu, spks_expand, cond] into (B, 320, T)."""
    comptime assert x_in.flat_rank == 3
    comptime assert mu_in.flat_rank == 3
    comptime assert spks_in.flat_rank == 2
    comptime assert cond_in.flat_rank == 3
    comptime assert output.flat_rank == 3
    var bid = block_idx.x
    var tid = thread_idx.x
    var c = bid % 320
    var b = bid // 320
    var t = tid
    while t < time:
        var v: Float32 = 0.0
        if c < 80:
            v = rebind[Scalar[dtype]](x_in[b, c, t]).cast[DType.float32]()
        elif c < 160:
            v = rebind[Scalar[dtype]](mu_in[b, c - 80, t]).cast[DType.float32]()
        elif c < 240:
            v = rebind[Scalar[dtype]](spks_in[b, c - 160]).cast[DType.float32]()
        else:
            v = rebind[Scalar[dtype]](cond_in[b, c - 240, t]).cast[DType.float32]()
        output[b, c, t] = rebind[output.ElementType](v.cast[dtype]())
        t += BLOCK_


def build_cfg_inputs_2d_kernel[
    dtype: DType,
    SrcLayout: TensorLayout, OutLayout: TensorLayout,
    B: Int, C: Int, BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (2*B, C)
    src: TileTensor[dtype, SrcLayout, MutAnyOrigin],       # (B, C)
    zero_uncond: Int,
):
    """Same as build_cfg_inputs_kernel but for 2D (B, C) tensors (used for spks)."""
    comptime assert src.flat_rank == 2
    comptime assert output.flat_rank == 2
    var idx = block_idx.x * BLOCK + thread_idx.x
    if idx >= 2 * B * C: return
    var c = idx % C
    var b2 = idx // C
    var b = b2 % B
    var is_uncond = b2 >= B
    var v: Float32 = 0.0
    if not is_uncond or zero_uncond == 0:
        v = rebind[Scalar[dtype]](src[b, c]).cast[DType.float32]()
    output[b2, c] = rebind[output.ElementType](v.cast[dtype]())
