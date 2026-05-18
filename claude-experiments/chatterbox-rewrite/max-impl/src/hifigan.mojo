"""HiFiGAN vocoder built with MAX Conv1d + activations.

Structure (mirrors chatterbox upstream):
  pre_conv (Conv1d) → for L in upsamples:
      up_conv (TransposedConv1d via stride'd Conv1d trick) → for R in resblocks:
          resblock(snake/leaky_relu + Conv1d + ...)
  post_conv (Conv1d) + tanh → audio

Per upstream, also takes a `source` signal concatenated channel-wise; for
brevity here we only model the mel→audio path. A full SourceModuleHnNSF can
be added incrementally.
"""
from std.math import tanh, sin
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from modules import Linear
from conv1d import Conv1d, conv1d_forward


@fieldwise_init
struct HiFiGANResBlock(Copyable, Movable):
    """Two parallel Conv1d branches with leaky-relu, summed residually."""
    var conv1: Conv1d
    var conv2: Conv1d
    var conv3: Conv1d


def leaky_relu(
    mut ctx: DeviceContext,
    mut x_buf: DeviceBuffer[DType.float32],
    mut out_buf: DeviceBuffer[DType.float32],
    n: Int, slope: Float32,
) raises:
    """y = x if x >= 0 else slope * x. Uses `nn.activations.leaky_relu`."""
    from nn.activations import leaky_relu as nn_lrelu
    var i_ptr = x_buf.unsafe_ptr()
    var o_ptr = out_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(i_ptr, o_ptr, slope)
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var x = i_ptr.load[width=width, alignment=alignment](i)
        o_ptr.store[width=width, alignment=alignment](i, nn_lrelu(x, Scalar[DType.float32](slope)))
    elementwise[func, simd_width=4, target="gpu"](
        IndexList[1](n), DeviceContextPtr(ctx),
    )


def snake(
    mut ctx: DeviceContext,
    mut x_buf: DeviceBuffer[DType.float32],
    mut alpha_buf: DeviceBuffer[DType.float32],   # (C,) per-channel alpha
    mut out_buf: DeviceBuffer[DType.float32],
    b: Int, c: Int, t: Int,
) raises:
    """Snake activation: y = x + (1/alpha) * sin^2(alpha * x)."""
    var i_ptr = x_buf.unsafe_ptr()
    var a_ptr = alpha_buf.unsafe_ptr()
    var o_ptr = out_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(i_ptr, a_ptr, o_ptr, c, t)
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (c * t)
        var rem = i - bi * c * t
        var ci = rem // t
        var alpha = a_ptr[ci]
        var safe_alpha: Float32 = alpha if alpha > 1.0e-6 else 1.0e-6
        var x = i_ptr[i]
        var s = sin(safe_alpha * x)
        o_ptr[i] = x + (1.0 / safe_alpha) * s * s
    elementwise[func, simd_width=1, target="gpu"](
        IndexList[1](b * c * t), DeviceContextPtr(ctx),
    )


@fieldwise_init
struct HiFiGAN(Copyable, Movable):
    """Top-level HiFiGAN vocoder skeleton."""
    var pre_conv: Conv1d
    var post_conv: Conv1d
    # Upsample stack omitted — added as `up_convs: List[Conv1d]` + `resblocks: List[HiFiGANResBlock]`
    # in a follow-up commit. The pre/post path is enough to demonstrate the
    # MAX-abstraction wiring.


def hifigan_forward_stub(
    mut ctx: DeviceContext,
    mut model: HiFiGAN,
    mut mel_buf: DeviceBuffer[DType.float32],    # (B, mel, T_mel)
    mut audio_out: DeviceBuffer[DType.float32],  # (B, audio_len)
    b: Int, mel: Int, t_mel: Int, audio_len: Int,
) raises:
    """Skeleton HiFiGAN forward — pre_conv → (deferred) → post_conv.

    A full forward is a chain of:
      pre_conv → leaky_relu/snake → for L in 4:
          ConvTranspose1d (stride 2..8) → leaky_relu → 3× ResBlock → ...
      → post_conv → tanh → audio.
    All composed from `Conv1d`, `leaky_relu`, `snake` defined here, plus
    `linalg.matmul` where helpful. No hand kernels.
    """
    var h = ctx.enqueue_create_buffer[DType.float32](b * mel * t_mel)
    conv1d_forward(ctx, model.pre_conv, mel_buf, h, b, t_mel, t_mel)
    # ... full upsampling stack deferred ...
    conv1d_forward(ctx, model.post_conv, h, audio_out, b, t_mel, audio_len)
