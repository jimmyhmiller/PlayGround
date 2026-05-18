"""NSF-HiFiGAN / HiFTGenerator (Neural Source Filter + iSTFTNet) — struct
definitions matching real upstream Chatterbox `mel2wav` weights.

Architecture summary (from chatterbox/models/s3gen/hifigan.py HiFTGenerator):

  conv_pre (mel→512, k=7)
  for stage in 0,1,2:
    x = lrelu(x); x = ups[stage](x)        # transposed conv
    si = source_downs[stage](source_stft)
    si = source_resblocks[stage](si)
    x = x + si
    sum over MRF resblocks (3 per stage, kernel sizes [3,7,11], dilation [1,3,5])
  x = lrelu(x); x = conv_post(x)            # → 18 channels (n_fft+2)
  mag, phase = exp(x[:9]), sin(x[9:18])
  audio = iSTFT(mag, phase)                 # n_fft=16, hop=4

Each ResBlock has:
  3 dilated convs (convs1) each followed by Snake activation (alpha per channel)
  3 1×1 convs (convs2) each followed by Snake activation
  Output is x + summed branches

NSF source path:
  m_source (l_linear (1, 9)) generates harmonic+noise → STFT (real, imag) → cat
  → source_downs[stage] (Conv1d 18 → ch, varying kernel/stride per stage)
  → source_resblocks[stage] (single MRF resblock with kernel from
    source_resblock_kernel_sizes [7, 11, ...])

f0_predictor: a Conv1d stack (condnet 5 layers) + linear classifier → F0 per frame

This file currently provides STRUCTS only; forward wiring TBD in follow-up.
"""
from std.gpu.host import DeviceContext, DeviceBuffer

from modules import Linear
from conv1d import Conv1d


@fieldwise_init
struct SnakeActivation(Copyable, Movable):
    """Snake activation: y = x + (1/alpha) * sin^2(alpha * x).

    alpha is a learnable per-channel parameter, shape (C,).
    """
    var alpha: DeviceBuffer[DType.float32]
    var channels: Int


@fieldwise_init
struct HiFTResBlock(Copyable, Movable):
    """One MRF ResBlock: 3 dilated convs (convs1) + 3 1x1 convs (convs2),
    each preceded by a Snake activation. All weight_norm collapsed at load.
    """
    var convs1: List[Conv1d]                # 3 dilated convs
    var convs2: List[Conv1d]                # 3 1x1 post convs
    var activations1: List[SnakeActivation] # 3 snake activations before convs1
    var activations2: List[SnakeActivation] # 3 snake activations before convs2
    var channels: Int


@fieldwise_init
struct F0Predictor(Copyable, Movable):
    """5 weight-normed conv1d layers (condnet/0,2,4,6,8) + linear classifier
    that maps 512 features → F0 prediction (1 channel).
    """
    var condnet: List[Conv1d]    # 5 conv layers; activations between are inferred
    var classifier: Linear       # (1, 512) + bias (1,)


@fieldwise_init
struct MSource(Copyable, Movable):
    """NSF source generator's `l_linear` mixer: (1, 9) — 9 harmonic inputs to
    1 output, mixed linearly with bias.
    """
    var l_linear: Linear         # in=9 out=1 + bias


@fieldwise_init
struct HiFTGenerator(Copyable, Movable):
    """Full NSF-HiFiGAN / HiFTGenerator.

    - conv_pre: Conv1d (80→512, k=7)
    - ups: 3 transposed Conv1d stages [512→256, 256→128, 128→64]
    - resblocks: 9 (3 per ups stage) — MRF kernels [3,7,11]
    - source_downs: 3 (one per ups stage, in=18 channels = n_fft+2)
    - source_resblocks: 3 (one per stage)
    - conv_post: Conv1d (64 → 18, k=7) — outputs STFT magnitude+phase
    - m_source: NSF harmonic source generator (l_linear 1×9)
    - f0_predictor: 5 conv layers + classifier
    """
    var conv_pre: Conv1d
    var ups: List[Conv1d]               # NOTE: weight is ConvTranspose1d — loader handles
    var resblocks: List[HiFTResBlock]
    var source_downs: List[Conv1d]
    var source_resblocks: List[HiFTResBlock]
    var conv_post: Conv1d
    var m_source: MSource
    var f0_predictor: F0Predictor
    var n_fft: Int                      # 16
    var hop_len: Int                    # 4
    var lrelu_slope: Float32            # 0.1
