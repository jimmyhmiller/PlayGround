"""CFM estimator structs matching real upstream Chatterbox weights.

Architecture (from chatterbox/models/s3gen/flow_matching matcha decoder):

  estimator
    time_mlp        : Linear 320 → 1024, Linear 1024 → 1024
    down_blocks[0]  : (Resnet1D, [4 BasicTransformerBlocks], Downsample Conv1d)
    mid_blocks[0..11]: each (Resnet1D, [4 BasicTransformerBlocks])
    up_blocks[0]    : (Resnet1D, [4 BasicTransformerBlocks], Upsample Conv1d)
    final_block     : Block1D (Conv1d + GroupNorm + Activation)
    final_proj      : Conv1d 256 → 80

  Resnet1D = (block1: Block1D, block2: Block1D, mlp: Linear 1024 → 256, res_conv: Conv1d 1x1)
  Block1D  = block (Sequential: Conv1d (k=3), Mish, GroupNorm)
  BasicTransformerBlock = norm1 + self-attn (q/k/v dim=512, out dim=256) + norm3 + FF (GEGLU 256→1024→256)
"""
from std.gpu.host import DeviceContext, DeviceBuffer

from modules import Linear, LayerNorm
from conv1d import Conv1d


@fieldwise_init
struct GroupNorm1d(Copyable, Movable):
    """PyTorch GroupNorm: weight (γ), bias (β), num_groups.
    forward = group-normalized x * γ + β
    """
    var weight: DeviceBuffer[DType.float32]
    var bias: DeviceBuffer[DType.float32]
    var channels: Int
    var num_groups: Int
    var eps: Float32


@fieldwise_init
struct Block1D(Copyable, Movable):
    """Conv1d (k=3) → Mish → GroupNorm.

    Upstream `block.block.0` = Conv1d, `block.block.2` = GroupNorm
    (index 1 is the Mish activation with no weights).
    """
    var conv: Conv1d
    var group_norm: GroupNorm1d


@fieldwise_init
struct Resnet1D(Copyable, Movable):
    """Resnet block with time-MLP injection: block1 + (+mlp(t)) + block2 + res_conv."""
    var block1: Block1D
    var block2: Block1D
    var mlp: Linear           # (channels, time_emb_dim) — projects t_emb into block1 output
    var res_conv: Conv1d      # 1x1 residual projection (input dim ≠ output dim case)


@fieldwise_init
struct CFMAttention(Copyable, Movable):
    """Self-attention with Q/K/V dim = 512 and output dim = 256."""
    var to_q: Conv1d   # Stored as Conv1d 1x1 OR Linear — upstream uses Linear (no bias on qkv)
    var to_k: Conv1d
    var to_v: Conv1d
    var to_out: Conv1d   # 256, 512
    var n_heads: Int
    var head_dim: Int


@fieldwise_init
struct CFMFeedForward(Copyable, Movable):
    """GEGLU FF: net.0.proj (256 → 1024 — projects to gate+value pair),
                net.2 (1024 → 256 — output projection).
    """
    var net0_proj: Linear       # (1024, 256) + bias
    var net2: Linear            # (256, 1024) + bias


@fieldwise_init
struct BasicTransformerBlock(Copyable, Movable):
    """One transformer block in the CFM stack."""
    var norm1: LayerNorm
    var attn1: CFMAttention
    var norm3: LayerNorm
    var ff: CFMFeedForward


@fieldwise_init
struct CFMDownStage(Copyable, Movable):
    var resnet: Resnet1D
    var transformers: List[BasicTransformerBlock]   # 4 blocks
    var downsampler: Conv1d                         # 1x1 stride-2 (k=3)


@fieldwise_init
struct CFMMidStage(Copyable, Movable):
    var resnet: Resnet1D
    var transformers: List[BasicTransformerBlock]   # 4 blocks


@fieldwise_init
struct CFMUpStage(Copyable, Movable):
    var resnet: Resnet1D
    var transformers: List[BasicTransformerBlock]
    var upsampler: Conv1d


@fieldwise_init
struct CFMEstimatorReal(Copyable, Movable):
    """Real-upstream-shape CFM estimator. Separate from the old `CFMEstimator`
    struct in `cfm.mojo` so the old forward path still compiles.
    """
    var time_mlp1: Linear
    var time_mlp2: Linear
    var down_blocks: List[CFMDownStage]   # 1 stage
    var mid_blocks: List[CFMMidStage]     # 12 stages
    var up_blocks: List[CFMUpStage]       # 1 stage
    var final_block: Block1D
    var final_proj: Conv1d                # 256 → 80, k=1
