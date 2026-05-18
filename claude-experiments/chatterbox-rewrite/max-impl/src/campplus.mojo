"""CAMPPlus speaker encoder — struct definitions matching real upstream
Chatterbox `speaker_encoder` weights.

Architecture (from chatterbox/models/s3gen/CAMPPlus):

  xvector
    tdnn      : 1×1 Conv1d 320→128 (k=5) + BN + nonlinear
    block1    : CAMDenseTDNNBlock (12 layers tdnnd1..tdnnd12), growth=32, in=128
    transit1  : 1×1 Conv1d 128 + 12*32 → 256 + BN
    block2    : CAMDenseTDNNBlock (24 layers), in=256, growth=32
    transit2  : Conv1d 256 + 24*32 → 512 + BN
    block3    : CAMDenseTDNNBlock (16 layers), in=512, growth=32
    transit3  : Conv1d 512 + 16*32 → 1024 + BN
    out_nonlinear  : BN(1024)
    dense          : Linear (192, 1024) + BN

  head : ResNetHead with bn1+conv1+bn2+conv2 + layer1 (2 ResBlocks) + layer2 (2 ResBlocks)

Each `tdnnd*` (CAMDenseTDNNLayer) is:
  nonlinear1 (BN) → linear1 (Conv1d 1x1) → nonlinear2 (BN) → cam_layer
  where cam_layer = CAMLayer:
    linear_local : Conv1d (32, 128, 3)         — context conv
    linear1      : Conv1d (64, 128, 1)         — point conv
    linear2      : Conv1d (32, 64, 1)          — output conv

Loader populates buffers; forward implementation is a follow-up.
"""
from std.gpu.host import DeviceContext, DeviceBuffer

from modules import Linear
from conv1d import Conv1d


@fieldwise_init
struct BatchNorm1d(Copyable, Movable):
    """PyTorch BatchNorm1d: weight (γ), bias (β), running_mean, running_var.
    forward = (x - mean) / sqrt(var + eps) * γ + β
    """
    var weight: DeviceBuffer[DType.float32]
    var bias: DeviceBuffer[DType.float32]
    var running_mean: DeviceBuffer[DType.float32]
    var running_var: DeviceBuffer[DType.float32]
    var channels: Int
    var eps: Float32


@fieldwise_init
struct CAMLayer(Copyable, Movable):
    """Context-Aware Masking layer used inside CAMDenseTDNNLayer."""
    var linear_local: Conv1d  # (32, 128, 3) — depthwise-like context conv
    var linear1: Conv1d        # (64, 128, 1) — point conv expanding
    var linear2: Conv1d        # (32, 64, 1)  — point conv contracting


@fieldwise_init
struct CAMDenseTDNNLayer(Copyable, Movable):
    """One tdnnd* layer = BN + Conv1d (1x1 128→128) + BN + CAMLayer."""
    var nonlinear1: BatchNorm1d
    var linear1: Conv1d
    var nonlinear2: BatchNorm1d
    var cam_layer: CAMLayer


@fieldwise_init
struct CAMDenseTDNNBlock(Copyable, Movable):
    """A block of N CAMDenseTDNNLayers with growth channels concatenated."""
    var layers: List[CAMDenseTDNNLayer]


@fieldwise_init
struct TransitLayer(Copyable, Movable):
    """Transit layer between dense TDNN blocks: BN + 1x1 Conv1d to expand channels."""
    var nonlinear: BatchNorm1d
    var linear: Conv1d


@fieldwise_init
struct TDNN(Copyable, Movable):
    """First TDNN layer: BN + 1x1 Conv1d (320→128) with kernel 5."""
    var linear: Conv1d
    var nonlinear: BatchNorm1d


@fieldwise_init
struct DenseLayer(Copyable, Movable):
    """Final dense head: Linear(1024 → 192) + BN(192)."""
    var linear: Conv1d   # implemented as 1x1 Conv1d in upstream
    var nonlinear: BatchNorm1d


@fieldwise_init
struct XVectorBackbone(Copyable, Movable):
    """xvector backbone:
       tdnn → block1 → transit1 → block2 → transit2 → block3 → transit3
       → out_nonlinear → dense
    """
    var tdnn: TDNN
    var block1: CAMDenseTDNNBlock
    var transit1: TransitLayer
    var block2: CAMDenseTDNNBlock
    var transit2: TransitLayer
    var block3: CAMDenseTDNNBlock
    var transit3: TransitLayer
    var out_nonlinear: BatchNorm1d
    var dense: DenseLayer


@fieldwise_init
struct ResNetBasicBlock(Copyable, Movable):
    """One residual block in ResNet head: bn1+conv1+bn2+conv2 (+ optional downsample)."""
    var bn1: BatchNorm1d
    var conv1: Conv1d
    var bn2: BatchNorm1d
    var conv2: Conv1d


@fieldwise_init
struct ResNetHead(Copyable, Movable):
    """Top of CAMPPlus: stem (bn1+conv1+bn2+conv2) + layer1 (2 blocks) + layer2 (2 blocks).

    Note: the head receives the dense feature and refines speaker embedding.
    """
    var bn1: BatchNorm1d
    var conv1: Conv1d
    var bn2: BatchNorm1d
    var conv2: Conv1d
    var layer1: List[ResNetBasicBlock]   # 2 blocks
    var layer2: List[ResNetBasicBlock]   # 2 blocks


@fieldwise_init
struct CAMPPlus(Copyable, Movable):
    var xvector: XVectorBackbone
    var head: ResNetHead
