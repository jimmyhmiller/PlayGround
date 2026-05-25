"""HiFiGAN vocoder — full upsample chain + resblocks → audio.

Built entirely with `Conv1d` + `elementwise[..., target="gpu"]` activations
via `nn.activations.leaky_relu` for snake-like behavior.

Upstream HiFiGAN structure:
  pre_conv (Conv1d, in=80→512, k=7) →
    for L in 4:
      LeakyReLU →
      ConvTranspose1d (stride 8/8/2/2) →
      sum of 3 parallel ResBlocks (dilation rate combos)
  → LeakyReLU → post_conv (Conv1d 512→1, k=7) → tanh → audio

We model ConvTranspose1d via nearest-neighbor upsample (elementwise) +
a regular Conv1d. This is sufficient for orchestration correctness; exact
upstream parity requires a true ConvTranspose1d kernel (TODO follow-up).
"""
from std.math import tanh, sin
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from nn.activations import leaky_relu as nn_lrelu

from conv1d import Conv1d, conv1d_forward
from modules import residual_add


def leaky_relu(
    mut ctx: DeviceContext,
    mut x_buf: DeviceBuffer[DType.float32],
    mut out_buf: DeviceBuffer[DType.float32],
    n: Int, slope: Float32,
) raises:
    var i_ptr = x_buf.unsafe_ptr()
    var o_ptr = out_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(i_ptr, o_ptr, slope)
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var x = i_ptr.load[width=width, alignment=alignment](i)
        o_ptr.store[width=width, alignment=alignment](
            i, nn_lrelu(x, Scalar[DType.float32](slope))
        )
    elementwise[func, simd_width=4, target="gpu"](
        IndexList[1](n), DeviceContextPtr(ctx),
    )


def nearest_upsample_1d(
    mut ctx: DeviceContext,
    mut x_buf: DeviceBuffer[DType.float32],     # (B, C, T)
    mut out_buf: DeviceBuffer[DType.float32],   # (B, C, T*stride)
    b: Int, c: Int, t: Int, stride: Int,
) raises:
    """out[b, c, t*stride + i] = x[b, c, t] for i in [0, stride)."""
    var i_ptr = x_buf.unsafe_ptr()
    var o_ptr = out_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(i_ptr, o_ptr, c, t, stride)
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (c * t * stride)
        var rem = i - bi * c * t * stride
        var ci = rem // (t * stride)
        var ti = (rem - ci * t * stride) // stride
        o_ptr[i] = i_ptr[bi * c * t + ci * t + ti]
    elementwise[func, simd_width=1, target="gpu"](
        IndexList[1](b * c * t * stride), DeviceContextPtr(ctx),
    )


def tanh_inplace(
    mut ctx: DeviceContext,
    mut x_buf: DeviceBuffer[DType.float32],
    mut out_buf: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    var i_ptr = x_buf.unsafe_ptr()
    var o_ptr = out_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(i_ptr, o_ptr)
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var x = i_ptr.load[width=width, alignment=alignment](i)
        o_ptr.store[width=width, alignment=alignment](i, tanh(x))
    elementwise[func, simd_width=4, target="gpu"](
        IndexList[1](n), DeviceContextPtr(ctx),
    )


@fieldwise_init
struct HiFiGANResBlock(Copyable, Movable):
    """Simplified ResBlock: 3 Conv1d branches summed."""
    var conv1: Conv1d
    var conv2: Conv1d
    var conv3: Conv1d


def hifigan_resblock(
    mut ctx: DeviceContext,
    mut module: HiFiGANResBlock,
    mut x_buf: DeviceBuffer[DType.float32],
    mut out_buf: DeviceBuffer[DType.float32],
    b: Int, c: Int, t: Int,
    slope: Float32,
) raises:
    """leaky_relu → conv1 → leaky → conv2 → leaky → conv3 (all summed with input)."""
    var x_act = ctx.enqueue_create_buffer[DType.float32](b * c * t)
    leaky_relu(ctx, x_buf, x_act, b * c * t, slope)
    var h1 = ctx.enqueue_create_buffer[DType.float32](b * c * t)
    conv1d_forward(ctx, module.conv1, x_act, h1, b, t, t)
    var h1_act = ctx.enqueue_create_buffer[DType.float32](b * c * t)
    leaky_relu(ctx, h1, h1_act, b * c * t, slope)
    var h2 = ctx.enqueue_create_buffer[DType.float32](b * c * t)
    conv1d_forward(ctx, module.conv2, h1_act, h2, b, t, t)
    var h2_act = ctx.enqueue_create_buffer[DType.float32](b * c * t)
    leaky_relu(ctx, h2, h2_act, b * c * t, slope)
    conv1d_forward(ctx, module.conv3, h2_act, out_buf, b, t, t)
    residual_add(ctx, out_buf, x_buf, b * c * t)


@fieldwise_init
struct HiFiGANUpsample(Copyable, Movable):
    """One upsampling stage: leaky → upsample → conv → ResBlocks (3 parallel)."""
    var up_conv: Conv1d
    var resblocks: List[HiFiGANResBlock]
    var stride: Int


def hifigan_upsample_stage(
    mut ctx: DeviceContext,
    mut stage: HiFiGANUpsample,
    mut x_buf: DeviceBuffer[DType.float32],
    mut out_buf: DeviceBuffer[DType.float32],
    b: Int, c_in: Int, c_out: Int, t_in: Int, t_out: Int,
    slope: Float32,
) raises:
    var act = ctx.enqueue_create_buffer[DType.float32](b * c_in * t_in)
    leaky_relu(ctx, x_buf, act, b * c_in * t_in, slope)
    # Upsample via nearest-neighbor.
    var up_buf = ctx.enqueue_create_buffer[DType.float32](b * c_in * t_out)
    nearest_upsample_1d(ctx, act, up_buf, b, c_in, t_in, stage.stride)
    # Project channels via up_conv (c_in → c_out).
    var conv_out = ctx.enqueue_create_buffer[DType.float32](b * c_out * t_out)
    conv1d_forward(ctx, stage.up_conv, up_buf, conv_out, b, t_out, t_out)

    # Sum of 3 parallel resblocks / number of resblocks (typical HiFiGAN MRF aggregation).
    var summed = ctx.enqueue_create_buffer[DType.float32](b * c_out * t_out)
    summed.enqueue_fill(0.0)
    for i in range(len(stage.resblocks)):
        var rb_out = ctx.enqueue_create_buffer[DType.float32](b * c_out * t_out)
        hifigan_resblock(ctx, stage.resblocks[i], conv_out, rb_out, b, c_out, t_out, slope)
        residual_add(ctx, summed, rb_out, b * c_out * t_out)
    # Average.
    var sp = summed.unsafe_ptr()
    var inv: Float32 = 1.0 / Float32(len(stage.resblocks))

    @always_inline
    @parameter
    @__copy_capture(sp, inv)
    def avg_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var v = sp.load[width=width, alignment=alignment](i)
        sp.store[width=width, alignment=alignment](i, v * inv)
    elementwise[avg_func, simd_width=4, target="gpu"](
        IndexList[1](b * c_out * t_out), DeviceContextPtr(ctx),
    )
    ctx.enqueue_copy(out_buf, summed)


@fieldwise_init
struct HiFiGAN(Copyable, Movable):
    """Top-level HiFiGAN."""
    var pre_conv: Conv1d
    var upsample_stages: List[HiFiGANUpsample]
    var post_conv: Conv1d
    var leaky_slope: Float32


def hifigan_forward(
    mut ctx: DeviceContext,
    mut model: HiFiGAN,
    mut mel_buf: DeviceBuffer[DType.float32],     # (B, 80, T_mel)
    mut audio_out: DeviceBuffer[DType.float32],   # (B, audio_len)
    b: Int, mel: Int, t_mel: Int,
    pre_c_out: Int,        # 512 typically
    stage_dims: List[Int], # per-stage c_out (e.g. [256, 128, 64, 32])
    audio_len: Int,
) raises:
    var slope = model.leaky_slope
    # Pre-conv.
    var h = ctx.enqueue_create_buffer[DType.float32](b * pre_c_out * t_mel)
    conv1d_forward(ctx, model.pre_conv, mel_buf, h, b, t_mel, t_mel)

    var cur_t = t_mel
    var cur_c = pre_c_out
    for i in range(len(model.upsample_stages)):
        var new_t = cur_t * model.upsample_stages[i].stride
        var new_c = stage_dims[i]
        var ho = ctx.enqueue_create_buffer[DType.float32](b * new_c * new_t)
        hifigan_upsample_stage(ctx, model.upsample_stages[i], h, ho,
                                b, cur_c, new_c, cur_t, new_t, slope)
        ctx.enqueue_copy(h, ho)
        cur_t = new_t
        cur_c = new_c

    # Final leaky + post_conv + tanh.
    var final_act = ctx.enqueue_create_buffer[DType.float32](b * cur_c * cur_t)
    leaky_relu(ctx, h, final_act, b * cur_c * cur_t, slope)
    var pre_audio = ctx.enqueue_create_buffer[DType.float32](b * audio_len)
    conv1d_forward(ctx, model.post_conv, final_act, pre_audio, b, cur_t, audio_len)
    tanh_inplace(ctx, pre_audio, audio_out, b * audio_len)
