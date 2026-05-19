"""FCM (Feature Compression Module) — front-end of CAMPPlus speaker encoder.

Architecture:
    conv1 (1→32, k=3, s=1, p=1) → bn1 → relu
    layer1: BasicResBlock(32→32, s=2) → BasicResBlock(32→32, s=1)
    layer2: BasicResBlock(32→32, s=2) → BasicResBlock(32→32, s=1)
    conv2 (32→32, k=3, s=(2,1), p=1) → bn2 → relu
    reshape (B, 32, H/8, W) → (B, 320, W)

Input: (B, 80, T) — mel; unsqueeze to (B, 1, 80, T) for 2D conv.
Output: (B, 320, T) — fed to XVectorBackbone.

BasicResBlock(in_planes, planes, stride):
    conv1 (in_planes→planes, k=3, s=(stride,1), p=1, bias=False) → bn1 → relu
    conv2 (planes→planes, k=3, s=1, p=1, bias=False) → bn2
    shortcut: identity if stride==1 and in==out; else conv(in→out, k=1, s=(stride,1)) + bn
    out += shortcut(x); relu
"""
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr

from conv2d import (
    Conv2d, BatchNorm2d, conv2d_forward, batchnorm2d_forward,
    relu_inplace_2d, add_inplace_2d,
)


@fieldwise_init
struct BasicResBlock2d(Copyable, Movable):
    var conv1: Conv2d
    var bn1: BatchNorm2d
    var conv2: Conv2d
    var bn2: BatchNorm2d
    # Shortcut may be a no-op when stride=1 and channels match.
    var has_shortcut: Bool
    var shortcut_conv: Conv2d
    var shortcut_bn: BatchNorm2d
    var stride: Int
    var in_planes: Int
    var planes: Int


@fieldwise_init
struct FCM(Copyable, Movable):
    var conv1: Conv2d
    var bn1: BatchNorm2d
    var layer1_block0: BasicResBlock2d
    var layer1_block1: BasicResBlock2d
    var layer2_block0: BasicResBlock2d
    var layer2_block1: BasicResBlock2d
    var conv2: Conv2d
    var bn2: BatchNorm2d
    var feat_dim: Int   # 80
    var m_channels: Int # 32


def basic_resblock_2d_forward(
    mut ctx: DeviceContext,
    mut block: BasicResBlock2d,
    mut x_buf: DeviceBuffer[DType.float32],     # (B, in_planes, h_in, w_in)
    mut out_buf: DeviceBuffer[DType.float32],   # (B, planes, h_out, w_out)
    batch: Int, h_in: Int, w_in: Int, h_out: Int, w_out: Int,
) raises:
    """Run BasicResBlock with stride only on H. h_out = (h_in - 1) // stride + 1 (since k=3, p=1)."""
    var in_planes = block.in_planes
    var planes = block.planes

    # Branch 1: conv1 → bn1 → relu → conv2 → bn2.
    var branch = ctx.enqueue_create_buffer[DType.float32](batch * planes * h_out * w_out)
    conv2d_forward(ctx, block.conv1, x_buf, branch, batch, h_in, w_in, h_out, w_out)
    var branch_bn = ctx.enqueue_create_buffer[DType.float32](batch * planes * h_out * w_out)
    batchnorm2d_forward(ctx, block.bn1, branch, branch_bn, batch, h_out, w_out)
    relu_inplace_2d(ctx, branch_bn, batch * planes * h_out * w_out)
    var branch_c2 = ctx.enqueue_create_buffer[DType.float32](batch * planes * h_out * w_out)
    conv2d_forward(ctx, block.conv2, branch_bn, branch_c2, batch, h_out, w_out, h_out, w_out)
    var branch_b2 = ctx.enqueue_create_buffer[DType.float32](batch * planes * h_out * w_out)
    batchnorm2d_forward(ctx, block.bn2, branch_c2, branch_b2, batch, h_out, w_out)

    # Branch 2: shortcut.
    if block.has_shortcut:
        var sc = ctx.enqueue_create_buffer[DType.float32](batch * planes * h_out * w_out)
        conv2d_forward(ctx, block.shortcut_conv, x_buf, sc, batch, h_in, w_in, h_out, w_out)
        var sc_bn = ctx.enqueue_create_buffer[DType.float32](batch * planes * h_out * w_out)
        batchnorm2d_forward(ctx, block.shortcut_bn, sc, sc_bn, batch, h_out, w_out)
        add_inplace_2d(ctx, branch_b2, sc_bn, batch * planes * h_out * w_out)
    else:
        # Identity: x is already (B, planes, h_in=h_out, w_in=w_out).
        add_inplace_2d(ctx, branch_b2, x_buf, batch * planes * h_out * w_out)

    relu_inplace_2d(ctx, branch_b2, batch * planes * h_out * w_out)
    ctx.enqueue_copy(out_buf, branch_b2)


def fcm_forward(
    mut ctx: DeviceContext,
    mut model: FCM,
    mut mel_buf: DeviceBuffer[DType.float32],   # (B, 80, T) — interpreted as (B, 1, 80, T)
    mut out_buf: DeviceBuffer[DType.float32],   # (B, 320, T) — output for x-vector
    batch: Int, t_in: Int,
) raises:
    """Run FCM forward. Input is (B, feat_dim=80, T); reshape to (B, 1, 80, T)."""
    var feat = model.feat_dim
    var m = model.m_channels

    # x.unsqueeze(1) is a no-op since (B, 80, T) and (B, 1, 80, T) have same memory layout.

    # conv1: (B, 1, 80, T) → (B, 32, 80, T)
    var h1 = ctx.enqueue_create_buffer[DType.float32](batch * m * feat * t_in)
    conv2d_forward(ctx, model.conv1, mel_buf, h1, batch, feat, t_in, feat, t_in)
    var h1_bn = ctx.enqueue_create_buffer[DType.float32](batch * m * feat * t_in)
    batchnorm2d_forward(ctx, model.bn1, h1, h1_bn, batch, feat, t_in)
    relu_inplace_2d(ctx, h1_bn, batch * m * feat * t_in)

    # layer1 block0 (stride=2): H 80 → 40
    var h2_h = feat // 2
    var h2 = ctx.enqueue_create_buffer[DType.float32](batch * m * h2_h * t_in)
    basic_resblock_2d_forward(ctx, model.layer1_block0, h1_bn, h2,
                                batch, feat, t_in, h2_h, t_in)
    # layer1 block1 (stride=1): no shape change.
    var h3 = ctx.enqueue_create_buffer[DType.float32](batch * m * h2_h * t_in)
    basic_resblock_2d_forward(ctx, model.layer1_block1, h2, h3,
                                batch, h2_h, t_in, h2_h, t_in)

    # layer2 block0 (stride=2): H 40 → 20
    var h4_h = h2_h // 2
    var h4 = ctx.enqueue_create_buffer[DType.float32](batch * m * h4_h * t_in)
    basic_resblock_2d_forward(ctx, model.layer2_block0, h3, h4,
                                batch, h2_h, t_in, h4_h, t_in)
    var h5 = ctx.enqueue_create_buffer[DType.float32](batch * m * h4_h * t_in)
    basic_resblock_2d_forward(ctx, model.layer2_block1, h4, h5,
                                batch, h4_h, t_in, h4_h, t_in)

    # conv2: stride=(2,1), H 20 → 10
    var h6_h = h4_h // 2
    var h6 = ctx.enqueue_create_buffer[DType.float32](batch * m * h6_h * t_in)
    conv2d_forward(ctx, model.conv2, h5, h6, batch, h4_h, t_in, h6_h, t_in)
    var h6_bn = ctx.enqueue_create_buffer[DType.float32](batch * m * h6_h * t_in)
    batchnorm2d_forward(ctx, model.bn2, h6, h6_bn, batch, h6_h, t_in)
    relu_inplace_2d(ctx, h6_bn, batch * m * h6_h * t_in)

    # Reshape (B, 32, 10, T) → (B, 320, T). Memory layout is already contiguous;
    # just need to copy (or alias) into out_buf.
    ctx.enqueue_copy(out_buf, h6_bn)
