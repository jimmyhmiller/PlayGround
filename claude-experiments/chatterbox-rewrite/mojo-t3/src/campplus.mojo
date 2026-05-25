"""
CAMPPlus speaker encoder — pure-Mojo port.

Currently provides the FCM (front-end conv module) host-side orchestration
that feeds a 1×80×T log-mel-fbank window through:
  conv1 (1→32, 3x3, pad=1)
  bn1
  relu
  layer1 = [BasicResBlock(stride=2,shortcut), BasicResBlock(stride=1,no shortcut)]
  layer2 = [BasicResBlock(stride=2,shortcut), BasicResBlock(stride=1,no shortcut)]
  conv2 (32→32, 3x3, stride=(2,1), pad=1)
  bn2
  relu
  reshape (B, C, H, T) → (B, C*H, T)        # H collapses from 80→10 so C*H=320

The TDNN trunk + StatsPool + DenseLayer pieces will plug in as additional
host-side helpers in a follow-up.
"""
from std.math import ceildiv
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major, Idx

from conv import conv1d_kernel_fast, conv2d_kernel, batchnorm1d_kernel, batchnorm2d_kernel, relu_kernel
from util_kernels import add_kernel
from cam_kernels import (
    seg_pool_kernel,
    mean_along_t_kernel,
    add_t_with_bc1_kernel,
    sigmoid_kernel,
    broadcast_mul_t_kernel,
    channel_concat_kernel,
)


comptime EPS_BN: Float32 = 1.0e-5
comptime BLOCK_PW: Int = 256


@fieldwise_init
struct DenseTdnnWeights(Copyable, Movable):
    """Bundle of all weights for one CAMDenseTDNNLayer (1 dense TDNN layer)."""
    var nl1_w: DeviceBuffer[DType.float32]
    var nl1_b: DeviceBuffer[DType.float32]
    var nl1_m: DeviceBuffer[DType.float32]
    var nl1_v: DeviceBuffer[DType.float32]
    var lin1_w: DeviceBuffer[DType.float32]
    var nl2_w: DeviceBuffer[DType.float32]
    var nl2_b: DeviceBuffer[DType.float32]
    var nl2_m: DeviceBuffer[DType.float32]
    var nl2_v: DeviceBuffer[DType.float32]
    var cam_local_w: DeviceBuffer[DType.float32]
    var cam_l1_w: DeviceBuffer[DType.float32]
    var cam_l1_b: DeviceBuffer[DType.float32]
    var cam_l2_w: DeviceBuffer[DType.float32]
    var cam_l2_b: DeviceBuffer[DType.float32]


def cam_dense_tdnn_layer[
    B: Int, IN_C: Int, OUT_C: Int, BN_C: Int, HALF_BN: Int, T: Int,
    KERNEL: Int, DILATION: Int, SEG_LEN: Int,
](
    mut ctx: DeviceContext,
    mut x_buf: DeviceBuffer[DType.float32],         # (B, IN_C, T)
    mut out_buf: DeviceBuffer[DType.float32],       # (B, IN_C + OUT_C, T)
    mut w: DenseTdnnWeights,
    mut dummy_bn_buf: DeviceBuffer[DType.float32],
) raises:
    """One CAMDenseTDNNLayer forward.

    bn_function: BN1d(IN_C) → ReLU → Conv1d(IN_C → BN_C, k=1, no bias)
    nonlinear2:  BN1d(BN_C) → ReLU
    CAMLayer:
        linear_local: Conv1d(BN_C → OUT_C, k=KERNEL, dilation=DILATION, pad=auto, no bias)
        ctx_mean = mean(nl2, dim=-1, keepdim=True)
        ctx_seg = seg_pool(nl2, SEG_LEN)
        ctx = ctx_seg + ctx_mean (broadcast)
        ctx_l1 = ReLU(Conv1d(BN_C → HALF_BN, k=1, bias)(ctx))
        m = sigmoid(Conv1d(HALF_BN → OUT_C, k=1, bias)(ctx_l1))
        cam_out = linear_local(nl2) * m
    out = concat([x_buf, cam_out], dim=1)
    """
    var n_x = B * IN_C * T
    var n_bn = B * BN_C * T
    var n_local = B * OUT_C * T
    var n_half = B * HALF_BN * T
    var n_out = B * (IN_C + OUT_C) * T

    var nl1_pre = ctx.enqueue_create_buffer[DType.float32](n_x)
    var nl1_out = ctx.enqueue_create_buffer[DType.float32](n_x)
    var lin1_out = ctx.enqueue_create_buffer[DType.float32](n_bn)
    var nl2_pre = ctx.enqueue_create_buffer[DType.float32](n_bn)
    var nl2_out = ctx.enqueue_create_buffer[DType.float32](n_bn)
    var y_local = ctx.enqueue_create_buffer[DType.float32](n_local)
    var ctx_mean = ctx.enqueue_create_buffer[DType.float32](B * BN_C * 1)
    var ctx_seg = ctx.enqueue_create_buffer[DType.float32](n_bn)
    var ctx_sum = ctx.enqueue_create_buffer[DType.float32](n_bn)
    var ctx_l1_pre = ctx.enqueue_create_buffer[DType.float32](n_half)
    var ctx_l1_out = ctx.enqueue_create_buffer[DType.float32](n_half)
    var ctx_l2_out = ctx.enqueue_create_buffer[DType.float32](n_local)
    var m_buf = ctx.enqueue_create_buffer[DType.float32](n_local)
    var cam_out = ctx.enqueue_create_buffer[DType.float32](n_local)

    comptime in_layout = row_major[B, IN_C, T]()
    comptime bn_layout = row_major[B, BN_C, T]()
    comptime local_layout = row_major[B, OUT_C, T]()
    comptime half_layout = row_major[B, HALF_BN, T]()
    comptime ctx1_layout = row_major[B, BN_C, 1]()
    comptime out_layout = row_major[B, IN_C + OUT_C, T]()
    comptime w_lin1_layout = row_major[BN_C, IN_C, 1]()
    comptime w_local_layout = row_major[OUT_C, BN_C, KERNEL]()
    comptime w_l1_layout = row_major[HALF_BN, BN_C, 1]()
    comptime w_l2_layout = row_major[OUT_C, HALF_BN, 1]()
    comptime p_in_layout = row_major[IN_C]()
    comptime p_bn_layout = row_major[BN_C]()
    comptime p_half_layout = row_major[HALF_BN]()
    comptime p_out_layout = row_major[OUT_C]()
    comptime flat_in = row_major[B * IN_C * T]()
    comptime flat_bn = row_major[B * BN_C * T]()
    comptime flat_half = row_major[B * HALF_BN * T]()
    comptime flat_local = row_major[B * OUT_C * T]()

    var x_t = TileTensor(x_buf, in_layout)
    var nl1_w_t = TileTensor(w.nl1_w, p_in_layout)
    var nl1_b_t = TileTensor(w.nl1_b, p_in_layout)
    var nl1_m_t = TileTensor(w.nl1_m, p_in_layout)
    var nl1_v_t = TileTensor(w.nl1_v, p_in_layout)
    var nl1_pre_t = TileTensor(nl1_pre, in_layout)
    var nl1_out_t = TileTensor(nl1_out, in_layout)
    var nl1_pre_flat = TileTensor(nl1_pre, flat_in)
    var nl1_out_flat = TileTensor(nl1_out, flat_in)
    var dummy_bn_t = TileTensor(dummy_bn_buf, p_bn_layout)
    var lin1_w_t = TileTensor(w.lin1_w, w_lin1_layout)
    var lin1_out_t = TileTensor(lin1_out, bn_layout)

    var nl2_w_t = TileTensor(w.nl2_w, p_bn_layout)
    var nl2_b_t = TileTensor(w.nl2_b, p_bn_layout)
    var nl2_m_t = TileTensor(w.nl2_m, p_bn_layout)
    var nl2_v_t = TileTensor(w.nl2_v, p_bn_layout)
    var nl2_pre_t = TileTensor(nl2_pre, bn_layout)
    var nl2_out_t = TileTensor(nl2_out, bn_layout)
    var nl2_pre_flat = TileTensor(nl2_pre, flat_bn)
    var nl2_out_flat = TileTensor(nl2_out, flat_bn)

    var cam_local_w_t = TileTensor(w.cam_local_w, w_local_layout)
    var y_local_t = TileTensor(y_local, local_layout)
    var ctx_mean_t = TileTensor(ctx_mean, ctx1_layout)
    var ctx_seg_t = TileTensor(ctx_seg, bn_layout)
    var ctx_sum_t = TileTensor(ctx_sum, bn_layout)
    var cam_l1_w_t = TileTensor(w.cam_l1_w, w_l1_layout)
    var cam_l1_b_t = TileTensor(w.cam_l1_b, p_half_layout)
    var ctx_l1_pre_t = TileTensor(ctx_l1_pre, half_layout)
    var ctx_l1_pre_flat = TileTensor(ctx_l1_pre, flat_half)
    var ctx_l1_out_flat = TileTensor(ctx_l1_out, flat_half)
    var ctx_l1_out_t = TileTensor(ctx_l1_out, half_layout)
    var cam_l2_w_t = TileTensor(w.cam_l2_w, w_l2_layout)
    var cam_l2_b_t = TileTensor(w.cam_l2_b, p_out_layout)
    var ctx_l2_out_t = TileTensor(ctx_l2_out, local_layout)
    var ctx_l2_out_flat = TileTensor(ctx_l2_out, flat_local)
    var m_t = TileTensor(m_buf, local_layout)
    var m_flat = TileTensor(m_buf, flat_local)
    var cam_out_t = TileTensor(cam_out, local_layout)
    var out_t = TileTensor(out_buf, out_layout)

    # ---- nonlinear1: BN1d(IN_C) + ReLU.
    comptime bn_in_k = batchnorm1d_kernel[
        DType.float32, type_of(in_layout), type_of(p_in_layout),
        type_of(in_layout), BLOCK_PW,
    ]
    ctx.enqueue_function[bn_in_k, bn_in_k](
        nl1_pre_t, x_t, nl1_w_t, nl1_b_t, nl1_m_t, nl1_v_t,
        B, IN_C, T, EPS_BN,
        grid_dim=B * IN_C, block_dim=BLOCK_PW,
    )
    comptime relu_in_k = relu_kernel[
        DType.float32, type_of(flat_in), type_of(flat_in), BLOCK_PW,
    ]
    ctx.enqueue_function[relu_in_k, relu_in_k](
        nl1_out_flat, nl1_pre_flat, n_x,
        grid_dim=ceildiv(n_x, BLOCK_PW), block_dim=BLOCK_PW,
    )
    # ---- linear1: Conv1d(IN_C → BN_C, k=1, no bias).
    comptime conv_lin1_k = conv1d_kernel_fast[
        DType.float32, type_of(in_layout), type_of(w_lin1_layout),
        type_of(p_bn_layout), type_of(bn_layout),
        1, False, BLOCK_PW,
    ]
    ctx.enqueue_function[conv_lin1_k, conv_lin1_k](
        lin1_out_t, nl1_out_t, lin1_w_t, dummy_bn_t,
        B, IN_C, BN_C, T, T, 1, 0, 1,
        grid_dim=B * BN_C, block_dim=BLOCK_PW,
    )
    # ---- nonlinear2: BN1d(BN_C) + ReLU.
    comptime bn_bn_k = batchnorm1d_kernel[
        DType.float32, type_of(bn_layout), type_of(p_bn_layout),
        type_of(bn_layout), BLOCK_PW,
    ]
    ctx.enqueue_function[bn_bn_k, bn_bn_k](
        nl2_pre_t, lin1_out_t, nl2_w_t, nl2_b_t, nl2_m_t, nl2_v_t,
        B, BN_C, T, EPS_BN,
        grid_dim=B * BN_C, block_dim=BLOCK_PW,
    )
    comptime relu_bn_k = relu_kernel[
        DType.float32, type_of(flat_bn), type_of(flat_bn), BLOCK_PW,
    ]
    ctx.enqueue_function[relu_bn_k, relu_bn_k](
        nl2_out_flat, nl2_pre_flat, n_bn,
        grid_dim=ceildiv(n_bn, BLOCK_PW), block_dim=BLOCK_PW,
    )
    # ---- CAMLayer.linear_local: Conv1d(BN_C → OUT_C, k=KERNEL, dilation=DILATION).
    # pad = (KERNEL-1)/2 * DILATION  (for kernel=3, dil=1: pad=1; dil=2: pad=2)
    comptime PAD = ((KERNEL - 1) // 2) * DILATION
    comptime conv_local_k = conv1d_kernel_fast[
        DType.float32, type_of(bn_layout), type_of(w_local_layout),
        type_of(p_bn_layout), type_of(local_layout),
        KERNEL, False, BLOCK_PW,
    ]
    ctx.enqueue_function[conv_local_k, conv_local_k](
        y_local_t, nl2_out_t, cam_local_w_t, dummy_bn_t,
        B, BN_C, OUT_C, T, T, 1, PAD, DILATION,
        grid_dim=B * OUT_C, block_dim=BLOCK_PW,
    )
    # ---- mean(nl2, dim=-1, keepdim=True).
    comptime mean_k = mean_along_t_kernel[
        DType.float32, type_of(bn_layout), type_of(ctx1_layout), BLOCK_PW,
    ]
    ctx.enqueue_function[mean_k, mean_k](
        ctx_mean_t, nl2_out_t, B, BN_C, T,
        grid_dim=B * BN_C, block_dim=BLOCK_PW,
    )
    # ---- seg_pool.
    comptime seg_k = seg_pool_kernel[
        DType.float32, type_of(bn_layout), type_of(bn_layout),
        SEG_LEN, BLOCK_PW,
    ]
    ctx.enqueue_function[seg_k, seg_k](
        ctx_seg_t, nl2_out_t, B, BN_C, T,
        grid_dim=B * BN_C, block_dim=BLOCK_PW,
    )
    # ---- ctx_sum = ctx_seg + broadcast(ctx_mean).
    comptime add_bc_k = add_t_with_bc1_kernel[
        DType.float32, type_of(bn_layout), type_of(ctx1_layout),
        type_of(bn_layout), BLOCK_PW,
    ]
    ctx.enqueue_function[add_bc_k, add_bc_k](
        ctx_sum_t, ctx_seg_t, ctx_mean_t, B, BN_C, T,
        grid_dim=B * BN_C, block_dim=BLOCK_PW,
    )
    # ---- CAMLayer.linear1: Conv1d(BN_C → HALF_BN, k=1, bias).
    comptime conv_l1_k = conv1d_kernel_fast[
        DType.float32, type_of(bn_layout), type_of(w_l1_layout),
        type_of(p_half_layout), type_of(half_layout),
        1, True, BLOCK_PW,
    ]
    ctx.enqueue_function[conv_l1_k, conv_l1_k](
        ctx_l1_pre_t, ctx_sum_t, cam_l1_w_t, cam_l1_b_t,
        B, BN_C, HALF_BN, T, T, 1, 0, 1,
        grid_dim=B * HALF_BN, block_dim=BLOCK_PW,
    )
    comptime relu_half_k = relu_kernel[
        DType.float32, type_of(flat_half), type_of(flat_half), BLOCK_PW,
    ]
    ctx.enqueue_function[relu_half_k, relu_half_k](
        ctx_l1_out_flat, ctx_l1_pre_flat, n_half,
        grid_dim=ceildiv(n_half, BLOCK_PW), block_dim=BLOCK_PW,
    )
    # ---- CAMLayer.linear2: Conv1d(HALF_BN → OUT_C, k=1, bias).
    comptime conv_l2_k = conv1d_kernel_fast[
        DType.float32, type_of(half_layout), type_of(w_l2_layout),
        type_of(p_out_layout), type_of(local_layout),
        1, True, BLOCK_PW,
    ]
    ctx.enqueue_function[conv_l2_k, conv_l2_k](
        ctx_l2_out_t, ctx_l1_out_t, cam_l2_w_t, cam_l2_b_t,
        B, HALF_BN, OUT_C, T, T, 1, 0, 1,
        grid_dim=B * OUT_C, block_dim=BLOCK_PW,
    )
    # ---- m = sigmoid; cam_out = y_local * m.
    comptime sig_k = sigmoid_kernel[
        DType.float32, type_of(flat_local), type_of(flat_local), BLOCK_PW,
    ]
    ctx.enqueue_function[sig_k, sig_k](
        m_flat, ctx_l2_out_flat, n_local,
        grid_dim=ceildiv(n_local, BLOCK_PW), block_dim=BLOCK_PW,
    )
    comptime mul_k = broadcast_mul_t_kernel[
        DType.float32, type_of(local_layout), type_of(local_layout),
        type_of(local_layout), BLOCK_PW,
    ]
    ctx.enqueue_function[mul_k, mul_k](
        cam_out_t, y_local_t, m_t, B, OUT_C, T,
        grid_dim=B * OUT_C, block_dim=BLOCK_PW,
    )
    # ---- out = concat([x, cam_out], dim=1).
    comptime cat_k = channel_concat_kernel[
        DType.float32, type_of(in_layout), type_of(local_layout),
        type_of(out_layout), BLOCK_PW,
    ]
    ctx.enqueue_function[cat_k, cat_k](
        out_t, x_t, cam_out_t, B, IN_C, OUT_C, T,
        grid_dim=B * (IN_C + OUT_C), block_dim=BLOCK_PW,
    )


# Helper: BasicResBlock generic over (in_channels, out_channels, H_in, H_out, W,
# has_shortcut, stride_h). The kernels themselves are parameterized over the
# *layout types*; we pass layouts in at call sites so the kernel monomorphizes
# correctly per-block shape.
def basic_resblock[
    BATCH: Int, IN_C: Int, OUT_C: Int, H_IN: Int, H_OUT: Int, W: Int,
    HAS_SHORTCUT: Bool, STRIDE_H: Int,
](
    mut ctx: DeviceContext,
    # Inputs.
    mut x_buf: DeviceBuffer[DType.float32],         # (B, IN_C,  H_IN,  W)
    mut out_buf: DeviceBuffer[DType.float32],       # (B, OUT_C, H_OUT, W)  ReLU output
    # 3x3 convs.
    mut w_c1_buf: DeviceBuffer[DType.float32],      # (OUT_C, IN_C,  3, 3)
    mut w_c2_buf: DeviceBuffer[DType.float32],      # (OUT_C, OUT_C, 3, 3)
    # BN1, BN2 affines.
    mut bn1_w_buf: DeviceBuffer[DType.float32],
    mut bn1_b_buf: DeviceBuffer[DType.float32],
    mut bn1_m_buf: DeviceBuffer[DType.float32],
    mut bn1_v_buf: DeviceBuffer[DType.float32],
    mut bn2_w_buf: DeviceBuffer[DType.float32],
    mut bn2_b_buf: DeviceBuffer[DType.float32],
    mut bn2_m_buf: DeviceBuffer[DType.float32],
    mut bn2_v_buf: DeviceBuffer[DType.float32],
    # Shortcut 1x1 conv + BN (only used if HAS_SHORTCUT).
    mut sc_w_buf: DeviceBuffer[DType.float32],
    mut sc_bn_w_buf: DeviceBuffer[DType.float32],
    mut sc_bn_b_buf: DeviceBuffer[DType.float32],
    mut sc_bn_m_buf: DeviceBuffer[DType.float32],
    mut sc_bn_v_buf: DeviceBuffer[DType.float32],
    # Dummy bias (HAS_BIAS=False on convs but we still need a tensor handle).
    mut dummy_buf: DeviceBuffer[DType.float32],
) raises:
    var n_in = BATCH * IN_C * H_IN * W
    var n_mid = BATCH * OUT_C * H_OUT * W   # output of conv1 (already strided)
    var n_out = BATCH * OUT_C * H_OUT * W

    comptime in_layout = row_major[BATCH, IN_C, H_IN, W]()
    comptime out_layout = row_major[BATCH, OUT_C, H_OUT, W]()
    comptime w_3x3_in_layout = row_major[OUT_C, IN_C, 3, 3]()
    comptime w_3x3_same_layout = row_major[OUT_C, OUT_C, 3, 3]()
    comptime w_1x1_layout = row_major[OUT_C, IN_C, 1, 1]()
    comptime p_in_layout = row_major[IN_C]()
    comptime p_out_layout = row_major[OUT_C]()
    comptime flat2d = row_major[1, BATCH * OUT_C * H_OUT * W]()
    comptime out_flat_1d = row_major[BATCH * OUT_C * H_OUT * W]()

    # Allocate intermediate buffers.
    var conv1_buf = ctx.enqueue_create_buffer[DType.float32](n_mid)
    var bn1_buf = ctx.enqueue_create_buffer[DType.float32](n_mid)
    var relu1_buf = ctx.enqueue_create_buffer[DType.float32](n_mid)
    var conv2_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    var bn2_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    var sc_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    var sc_bn_buf = ctx.enqueue_create_buffer[DType.float32](n_out)
    var sum_buf = ctx.enqueue_create_buffer[DType.float32](n_out)

    var x_t = TileTensor(x_buf, in_layout)
    var w_c1_t = TileTensor(w_c1_buf, w_3x3_in_layout)
    var w_c2_t = TileTensor(w_c2_buf, w_3x3_same_layout)
    var dummy_in_t = TileTensor(dummy_buf, p_in_layout)
    var dummy_out_t = TileTensor(dummy_buf, p_out_layout)
    var bn1_w_t = TileTensor(bn1_w_buf, p_out_layout)
    var bn1_b_t = TileTensor(bn1_b_buf, p_out_layout)
    var bn1_m_t = TileTensor(bn1_m_buf, p_out_layout)
    var bn1_v_t = TileTensor(bn1_v_buf, p_out_layout)
    var bn2_w_t = TileTensor(bn2_w_buf, p_out_layout)
    var bn2_b_t = TileTensor(bn2_b_buf, p_out_layout)
    var bn2_m_t = TileTensor(bn2_m_buf, p_out_layout)
    var bn2_v_t = TileTensor(bn2_v_buf, p_out_layout)

    var conv1_t = TileTensor(conv1_buf, out_layout)
    var bn1_t = TileTensor(bn1_buf, out_layout)
    var relu1_t = TileTensor(relu1_buf, out_layout)
    var conv2_t = TileTensor(conv2_buf, out_layout)
    var bn2_t = TileTensor(bn2_buf, out_layout)
    var sc_t = TileTensor(sc_buf, out_layout)
    var sc_bn_t = TileTensor(sc_bn_buf, out_layout)
    var sum_t = TileTensor(sum_buf, out_layout)
    var out_t = TileTensor(out_buf, out_layout)

    var bn1_flat = TileTensor(bn1_buf, row_major[1, BATCH * OUT_C * H_OUT * W]())
    var relu1_flat = TileTensor(relu1_buf, row_major[1, BATCH * OUT_C * H_OUT * W]())
    var bn2_flat_2d = TileTensor(bn2_buf, flat2d)
    var sc_bn_flat_2d = TileTensor(sc_bn_buf, flat2d)
    var sum_flat_2d = TileTensor(sum_buf, flat2d)

    var sum_flat_1d = TileTensor(sum_buf, out_flat_1d)
    var out_flat_1d_view = TileTensor(out_buf, out_flat_1d)
    var bn1_flat_1d = TileTensor(bn1_buf, out_flat_1d)
    var relu1_flat_1d = TileTensor(relu1_buf, out_flat_1d)

    # conv1: 3x3 stride=(STRIDE_H, 1) pad=1, no bias.
    comptime conv1_k = conv2d_kernel[
        DType.float32, type_of(in_layout), type_of(w_3x3_in_layout),
        type_of(p_in_layout), type_of(out_layout),
        3, 3, False, 256,
    ]
    ctx.enqueue_function[conv1_k, conv1_k](
        conv1_t, x_t, w_c1_t, dummy_in_t,
        BATCH, IN_C, OUT_C, H_IN, W, H_OUT, W, STRIDE_H, 1, 1, 1,
        grid_dim=BATCH * OUT_C * H_OUT, block_dim=256,
    )
    # bn1.
    comptime bn_k = batchnorm2d_kernel[
        DType.float32, type_of(out_layout), type_of(p_out_layout),
        type_of(out_layout), 256,
    ]
    ctx.enqueue_function[bn_k, bn_k](
        bn1_t, conv1_t, bn1_w_t, bn1_b_t, bn1_m_t, bn1_v_t,
        BATCH, OUT_C, H_OUT, W, EPS_BN,
        grid_dim=BATCH * OUT_C * H_OUT, block_dim=256,
    )
    # relu1: write into relu1_buf.
    comptime relu_k = relu_kernel[
        DType.float32, type_of(out_flat_1d), type_of(out_flat_1d), BLOCK_PW,
    ]
    ctx.enqueue_function[relu_k, relu_k](
        relu1_flat_1d, bn1_flat_1d, n_mid,
        grid_dim=ceildiv(n_mid, BLOCK_PW), block_dim=BLOCK_PW,
    )
    # conv2: 3x3 stride=1 pad=1, no bias (input has already been strided).
    comptime conv2_k = conv2d_kernel[
        DType.float32, type_of(out_layout), type_of(w_3x3_same_layout),
        type_of(p_out_layout), type_of(out_layout),
        3, 3, False, 256,
    ]
    ctx.enqueue_function[conv2_k, conv2_k](
        conv2_t, relu1_t, w_c2_t, dummy_out_t,
        BATCH, OUT_C, OUT_C, H_OUT, W, H_OUT, W, 1, 1, 1, 1,
        grid_dim=BATCH * OUT_C * H_OUT, block_dim=256,
    )
    # bn2.
    ctx.enqueue_function[bn_k, bn_k](
        bn2_t, conv2_t, bn2_w_t, bn2_b_t, bn2_m_t, bn2_v_t,
        BATCH, OUT_C, H_OUT, W, EPS_BN,
        grid_dim=BATCH * OUT_C * H_OUT, block_dim=256,
    )

    comptime if HAS_SHORTCUT:
        # Shortcut: Conv2d 1x1 stride=(STRIDE_H,1) on x, then BN2d.
        var w_sc_t = TileTensor(sc_w_buf, w_1x1_layout)
        var sc_bn_w_t = TileTensor(sc_bn_w_buf, p_out_layout)
        var sc_bn_b_t = TileTensor(sc_bn_b_buf, p_out_layout)
        var sc_bn_m_t = TileTensor(sc_bn_m_buf, p_out_layout)
        var sc_bn_v_t = TileTensor(sc_bn_v_buf, p_out_layout)

        comptime conv_sc_k = conv2d_kernel[
            DType.float32, type_of(in_layout), type_of(w_1x1_layout),
            type_of(p_in_layout), type_of(out_layout),
            1, 1, False, 256,
        ]
        ctx.enqueue_function[conv_sc_k, conv_sc_k](
            sc_t, x_t, w_sc_t, dummy_in_t,
            BATCH, IN_C, OUT_C, H_IN, W, H_OUT, W, STRIDE_H, 1, 0, 0,
            grid_dim=BATCH * OUT_C * H_OUT, block_dim=256,
        )
        ctx.enqueue_function[bn_k, bn_k](
            sc_bn_t, sc_t, sc_bn_w_t, sc_bn_b_t, sc_bn_m_t, sc_bn_v_t,
            BATCH, OUT_C, H_OUT, W, EPS_BN,
            grid_dim=BATCH * OUT_C * H_OUT, block_dim=256,
        )
        # sum = bn2 + sc_bn
        comptime add_k = add_kernel[
            DType.float32, type_of(flat2d), type_of(flat2d),
            type_of(flat2d), BLOCK_PW,
        ]
        ctx.enqueue_function[add_k, add_k](
            sum_flat_2d, bn2_flat_2d, sc_bn_flat_2d, n_out,
            grid_dim=ceildiv(n_out, BLOCK_PW), block_dim=BLOCK_PW,
        )
    else:
        # No shortcut: residual is x (which has same shape as out — IN_C==OUT_C,
        # H_IN==H_OUT, STRIDE_H==1 must hold for the non-shortcut branch).
        comptime assert IN_C == OUT_C and H_IN == H_OUT and STRIDE_H == 1,
            "non-shortcut BasicResBlock requires matching shapes"
        var x_flat_2d = TileTensor(x_buf, flat2d)
        comptime add_k2 = add_kernel[
            DType.float32, type_of(flat2d), type_of(flat2d),
            type_of(flat2d), BLOCK_PW,
        ]
        ctx.enqueue_function[add_k2, add_k2](
            sum_flat_2d, bn2_flat_2d, x_flat_2d, n_out,
            grid_dim=ceildiv(n_out, BLOCK_PW), block_dim=BLOCK_PW,
        )
    # out = relu(sum)
    ctx.enqueue_function[relu_k, relu_k](
        out_flat_1d_view, sum_flat_1d, n_out,
        grid_dim=ceildiv(n_out, BLOCK_PW), block_dim=BLOCK_PW,
    )


def transit_layer[
    B: Int, IN_C: Int, OUT_C: Int, T: Int,
    HAS_BIAS: Bool,
](
    mut ctx: DeviceContext,
    mut x_buf: DeviceBuffer[DType.float32],         # (B, IN_C, T)
    mut out_buf: DeviceBuffer[DType.float32],       # (B, OUT_C, T)
    mut bn_w_buf: DeviceBuffer[DType.float32],      # (IN_C,)
    mut bn_b_buf: DeviceBuffer[DType.float32],      # (IN_C,)
    mut bn_m_buf: DeviceBuffer[DType.float32],
    mut bn_v_buf: DeviceBuffer[DType.float32],
    mut lin_w_buf: DeviceBuffer[DType.float32],     # (OUT_C, IN_C, 1)
    mut lin_b_buf: DeviceBuffer[DType.float32],     # (OUT_C,) — only used if HAS_BIAS
    mut dummy_buf: DeviceBuffer[DType.float32],
) raises:
    """TransitLayer: BN1d(IN_C) → ReLU → Conv1d(IN_C → OUT_C, k=1, bias=HAS_BIAS)."""
    var n_in = B * IN_C * T
    var n_out = B * OUT_C * T

    var bn_pre = ctx.enqueue_create_buffer[DType.float32](n_in)
    var bn_out = ctx.enqueue_create_buffer[DType.float32](n_in)

    comptime in_layout = row_major[B, IN_C, T]()
    comptime out_layout = row_major[B, OUT_C, T]()
    comptime w_layout = row_major[OUT_C, IN_C, 1]()
    comptime p_in_layout = row_major[IN_C]()
    comptime p_out_layout = row_major[OUT_C]()
    comptime flat_in = row_major[B * IN_C * T]()

    var x_t = TileTensor(x_buf, in_layout)
    var bn_w_t = TileTensor(bn_w_buf, p_in_layout)
    var bn_b_t = TileTensor(bn_b_buf, p_in_layout)
    var bn_m_t = TileTensor(bn_m_buf, p_in_layout)
    var bn_v_t = TileTensor(bn_v_buf, p_in_layout)
    var bn_pre_t = TileTensor(bn_pre, in_layout)
    var bn_pre_flat = TileTensor(bn_pre, flat_in)
    var bn_out_t = TileTensor(bn_out, in_layout)
    var bn_out_flat = TileTensor(bn_out, flat_in)
    var lin_w_t = TileTensor(lin_w_buf, w_layout)
    var lin_b_t = TileTensor(lin_b_buf, p_out_layout)
    var dummy_t = TileTensor(dummy_buf, p_out_layout)
    var out_t = TileTensor(out_buf, out_layout)

    comptime bn_k = batchnorm1d_kernel[
        DType.float32, type_of(in_layout), type_of(p_in_layout),
        type_of(in_layout), BLOCK_PW,
    ]
    ctx.enqueue_function[bn_k, bn_k](
        bn_pre_t, x_t, bn_w_t, bn_b_t, bn_m_t, bn_v_t,
        B, IN_C, T, EPS_BN,
        grid_dim=B * IN_C, block_dim=BLOCK_PW,
    )
    comptime relu_k = relu_kernel[
        DType.float32, type_of(flat_in), type_of(flat_in), BLOCK_PW,
    ]
    ctx.enqueue_function[relu_k, relu_k](
        bn_out_flat, bn_pre_flat, n_in,
        grid_dim=ceildiv(n_in, BLOCK_PW), block_dim=BLOCK_PW,
    )
    comptime conv_k = conv1d_kernel_fast[
        DType.float32, type_of(in_layout), type_of(w_layout),
        type_of(p_out_layout), type_of(out_layout),
        1, HAS_BIAS, BLOCK_PW,
    ]
    comptime if HAS_BIAS:
        ctx.enqueue_function[conv_k, conv_k](
            out_t, bn_out_t, lin_w_t, lin_b_t,
            B, IN_C, OUT_C, T, T, 1, 0, 1,
            grid_dim=B * OUT_C, block_dim=BLOCK_PW,
        )
    else:
        ctx.enqueue_function[conv_k, conv_k](
            out_t, bn_out_t, lin_w_t, dummy_t,
            B, IN_C, OUT_C, T, T, 1, 0, 1,
            grid_dim=B * OUT_C, block_dim=BLOCK_PW,
        )
