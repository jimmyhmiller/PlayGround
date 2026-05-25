"""
Full CAMPPlus parity test: fbank features → 192-d xvector.

Input:  fbank_feat.bin     (1, 998, 80) — kaldi fbank features (centered)
        (For now we use fcm_out.bin as direct input to xvector trunk
         to validate the trunk independently of FCM.)
Target: xvector.bin        (1, 192)

This test exercises:
  - FCM (already verified, but we run it again to chain)
  - xvector.tdnn (TDNNLayer)
  - xvector.block1..3 (CAMDenseTDNNBlock × 3)
  - xvector.transit1..3 (TransitLayer × 3, no bias)
  - xvector.out_nonlinear (BN + ReLU)
  - xvector.stats (StatsPool: mean + std)
  - xvector.dense (Conv1d 1x1 → BN no-affine)
"""
from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import load_fp32
from conv import conv1d_kernel_fast, batchnorm1d_kernel, relu_kernel
from campplus import (
    cam_dense_tdnn_layer, transit_layer, DenseTdnnWeights,
)
from stats_kernels import stats_pool_kernel, bn_no_affine_2d_kernel


comptime B = 1
comptime T = 499
comptime GROWTH = 32
comptime BN_C = 128
comptime HALF_BN = 64
comptime KERNEL = 3
comptime SEG_LEN = 100
comptime BLOCK = 256
comptime EPS: Float32 = 1.0e-5


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def upload_w(mut ctx: DeviceContext, fix: String, name: String) raises -> DeviceBuffer[DType.float32]:
    var t = load_fp32(fix + name)
    var n = len(t.data)
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    upload(buf, t.data, n)
    return buf^


def load_dense(mut ctx: DeviceContext, fix: String, block_id: Int, layer_idx: Int) raises -> DenseTdnnWeights:
    var prefix = "weights/xvector__block" + String(block_id) + "__tdnnd" + String(layer_idx + 1) + "__"
    return DenseTdnnWeights(
        upload_w(ctx, fix, prefix + "nonlinear1__batchnorm__weight.bin"),
        upload_w(ctx, fix, prefix + "nonlinear1__batchnorm__bias.bin"),
        upload_w(ctx, fix, prefix + "nonlinear1__batchnorm__running_mean.bin"),
        upload_w(ctx, fix, prefix + "nonlinear1__batchnorm__running_var.bin"),
        upload_w(ctx, fix, prefix + "linear1__weight.bin"),
        upload_w(ctx, fix, prefix + "nonlinear2__batchnorm__weight.bin"),
        upload_w(ctx, fix, prefix + "nonlinear2__batchnorm__bias.bin"),
        upload_w(ctx, fix, prefix + "nonlinear2__batchnorm__running_mean.bin"),
        upload_w(ctx, fix, prefix + "nonlinear2__batchnorm__running_var.bin"),
        upload_w(ctx, fix, prefix + "cam_layer__linear_local__weight.bin"),
        upload_w(ctx, fix, prefix + "cam_layer__linear1__weight.bin"),
        upload_w(ctx, fix, prefix + "cam_layer__linear1__bias.bin"),
        upload_w(ctx, fix, prefix + "cam_layer__linear2__weight.bin"),
        upload_w(ctx, fix, prefix + "cam_layer__linear2__bias.bin"),
    )


def test_full_xvector_trunk() raises:
    """Tests xvector trunk: fcm_out -> 192-d xvector. Uses fcm_out.bin as
    input (already verified by test_campplus_fcm.mojo)."""
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/campplus/"
    var ctx = DeviceContext()

    # ---- Input.
    var x_in = load_fp32(fix + "fcm_out.bin")     # (1, 320, 998)
    var exp = load_fp32(fix + "xvector.bin")      # (1, 192)
    var n_fcm = B * 320 * 998
    var fcm_buf = ctx.enqueue_create_buffer[DType.float32](n_fcm)
    upload(fcm_buf, x_in.data, n_fcm)

    # ============================================================
    # xvector.tdnn: Conv1d(320 -> 128, k=5, s=2, pad=2, dil=1) + BN + ReLU
    # ============================================================
    var w_tdnn = upload_w(ctx, fix, "weights/xvector__tdnn__linear__weight.bin")
    var tdnn_bn_w = upload_w(ctx, fix, "weights/xvector__tdnn__nonlinear__batchnorm__weight.bin")
    var tdnn_bn_b = upload_w(ctx, fix, "weights/xvector__tdnn__nonlinear__batchnorm__bias.bin")
    var tdnn_bn_m = upload_w(ctx, fix, "weights/xvector__tdnn__nonlinear__batchnorm__running_mean.bin")
    var tdnn_bn_v = upload_w(ctx, fix, "weights/xvector__tdnn__nonlinear__batchnorm__running_var.bin")
    var tdnn_bias_dummy = ctx.enqueue_create_buffer[DType.float32](128)

    var n_tdnn = B * 128 * T
    var tdnn_pre = ctx.enqueue_create_buffer[DType.float32](n_tdnn)
    var tdnn_bn = ctx.enqueue_create_buffer[DType.float32](n_tdnn)
    var tdnn_out = ctx.enqueue_create_buffer[DType.float32](n_tdnn)

    comptime fcm_layout = row_major[B, 320, 998]()
    comptime tdnn_w_layout = row_major[128, 320, 5]()
    comptime tdnn_layout = row_major[B, 128, T]()
    comptime tdnn_p_layout = row_major[128]()
    comptime tdnn_flat = row_major[B * 128 * T]()

    var fcm_t = TileTensor(fcm_buf, fcm_layout)
    var w_tdnn_t = TileTensor(w_tdnn, tdnn_w_layout)
    var tdnn_bias_t = TileTensor(tdnn_bias_dummy, tdnn_p_layout)
    var tdnn_pre_t = TileTensor(tdnn_pre, tdnn_layout)
    var tdnn_bn_w_t = TileTensor(tdnn_bn_w, tdnn_p_layout)
    var tdnn_bn_b_t = TileTensor(tdnn_bn_b, tdnn_p_layout)
    var tdnn_bn_m_t = TileTensor(tdnn_bn_m, tdnn_p_layout)
    var tdnn_bn_v_t = TileTensor(tdnn_bn_v, tdnn_p_layout)
    var tdnn_bn_t = TileTensor(tdnn_bn, tdnn_layout)
    var tdnn_bn_flat = TileTensor(tdnn_bn, tdnn_flat)
    var tdnn_out_flat = TileTensor(tdnn_out, tdnn_flat)

    comptime tdnn_conv_k = conv1d_kernel_fast[
        DType.float32, type_of(fcm_layout), type_of(tdnn_w_layout),
        type_of(tdnn_p_layout), type_of(tdnn_layout),
        5, False, BLOCK,
    ]
    ctx.enqueue_function[tdnn_conv_k, tdnn_conv_k](
        tdnn_pre_t, fcm_t, w_tdnn_t, tdnn_bias_t,
        B, 320, 128, 998, T, 2, 2, 1,
        grid_dim=B * 128, block_dim=BLOCK,
    )
    comptime tdnn_bn_k = batchnorm1d_kernel[
        DType.float32, type_of(tdnn_layout), type_of(tdnn_p_layout),
        type_of(tdnn_layout), BLOCK,
    ]
    ctx.enqueue_function[tdnn_bn_k, tdnn_bn_k](
        tdnn_bn_t, tdnn_pre_t, tdnn_bn_w_t, tdnn_bn_b_t, tdnn_bn_m_t, tdnn_bn_v_t,
        B, 128, T, EPS,
        grid_dim=B * 128, block_dim=BLOCK,
    )
    comptime tdnn_relu_k = relu_kernel[
        DType.float32, type_of(tdnn_flat), type_of(tdnn_flat), BLOCK,
    ]
    ctx.enqueue_function[tdnn_relu_k, tdnn_relu_k](
        tdnn_out_flat, tdnn_bn_flat, n_tdnn,
        grid_dim=ceildiv(n_tdnn, BLOCK), block_dim=BLOCK,
    )

    # ============================================================
    # block1: 12 dense TDNN layers, dilation=1
    # IN_C goes 128, 160, ..., 480; output is (B, 512, T)
    # ============================================================
    var b1_w = List[DenseTdnnWeights]()
    for i in range(12):
        b1_w.append(load_dense(ctx, fix, 1, i))
    var b1_dummy = List[DeviceBuffer[DType.float32]]()
    for i in range(12):
        b1_dummy.append(ctx.enqueue_create_buffer[DType.float32](BN_C))

    var b1_out0 = ctx.enqueue_create_buffer[DType.float32](B * 160 * T)
    var b1_out1 = ctx.enqueue_create_buffer[DType.float32](B * 192 * T)
    var b1_out2 = ctx.enqueue_create_buffer[DType.float32](B * 224 * T)
    var b1_out3 = ctx.enqueue_create_buffer[DType.float32](B * 256 * T)
    var b1_out4 = ctx.enqueue_create_buffer[DType.float32](B * 288 * T)
    var b1_out5 = ctx.enqueue_create_buffer[DType.float32](B * 320 * T)
    var b1_out6 = ctx.enqueue_create_buffer[DType.float32](B * 352 * T)
    var b1_out7 = ctx.enqueue_create_buffer[DType.float32](B * 384 * T)
    var b1_out8 = ctx.enqueue_create_buffer[DType.float32](B * 416 * T)
    var b1_out9 = ctx.enqueue_create_buffer[DType.float32](B * 448 * T)
    var b1_out10 = ctx.enqueue_create_buffer[DType.float32](B * 480 * T)
    var b1_out11 = ctx.enqueue_create_buffer[DType.float32](B * 512 * T)

    cam_dense_tdnn_layer[B, 128, GROWTH, BN_C, HALF_BN, T, KERNEL, 1, SEG_LEN](
        ctx, tdnn_out, b1_out0, b1_w[0], b1_dummy[0])
    cam_dense_tdnn_layer[B, 160, GROWTH, BN_C, HALF_BN, T, KERNEL, 1, SEG_LEN](
        ctx, b1_out0, b1_out1, b1_w[1], b1_dummy[1])
    cam_dense_tdnn_layer[B, 192, GROWTH, BN_C, HALF_BN, T, KERNEL, 1, SEG_LEN](
        ctx, b1_out1, b1_out2, b1_w[2], b1_dummy[2])
    cam_dense_tdnn_layer[B, 224, GROWTH, BN_C, HALF_BN, T, KERNEL, 1, SEG_LEN](
        ctx, b1_out2, b1_out3, b1_w[3], b1_dummy[3])
    cam_dense_tdnn_layer[B, 256, GROWTH, BN_C, HALF_BN, T, KERNEL, 1, SEG_LEN](
        ctx, b1_out3, b1_out4, b1_w[4], b1_dummy[4])
    cam_dense_tdnn_layer[B, 288, GROWTH, BN_C, HALF_BN, T, KERNEL, 1, SEG_LEN](
        ctx, b1_out4, b1_out5, b1_w[5], b1_dummy[5])
    cam_dense_tdnn_layer[B, 320, GROWTH, BN_C, HALF_BN, T, KERNEL, 1, SEG_LEN](
        ctx, b1_out5, b1_out6, b1_w[6], b1_dummy[6])
    cam_dense_tdnn_layer[B, 352, GROWTH, BN_C, HALF_BN, T, KERNEL, 1, SEG_LEN](
        ctx, b1_out6, b1_out7, b1_w[7], b1_dummy[7])
    cam_dense_tdnn_layer[B, 384, GROWTH, BN_C, HALF_BN, T, KERNEL, 1, SEG_LEN](
        ctx, b1_out7, b1_out8, b1_w[8], b1_dummy[8])
    cam_dense_tdnn_layer[B, 416, GROWTH, BN_C, HALF_BN, T, KERNEL, 1, SEG_LEN](
        ctx, b1_out8, b1_out9, b1_w[9], b1_dummy[9])
    cam_dense_tdnn_layer[B, 448, GROWTH, BN_C, HALF_BN, T, KERNEL, 1, SEG_LEN](
        ctx, b1_out9, b1_out10, b1_w[10], b1_dummy[10])
    cam_dense_tdnn_layer[B, 480, GROWTH, BN_C, HALF_BN, T, KERNEL, 1, SEG_LEN](
        ctx, b1_out10, b1_out11, b1_w[11], b1_dummy[11])

    # ============================================================
    # transit1: TransitLayer(512 -> 256, no bias)
    # ============================================================
    var t1_bn_w = upload_w(ctx, fix, "weights/xvector__transit1__nonlinear__batchnorm__weight.bin")
    var t1_bn_b = upload_w(ctx, fix, "weights/xvector__transit1__nonlinear__batchnorm__bias.bin")
    var t1_bn_m = upload_w(ctx, fix, "weights/xvector__transit1__nonlinear__batchnorm__running_mean.bin")
    var t1_bn_v = upload_w(ctx, fix, "weights/xvector__transit1__nonlinear__batchnorm__running_var.bin")
    var t1_lin_w = upload_w(ctx, fix, "weights/xvector__transit1__linear__weight.bin")
    var t1_lin_b_dummy = ctx.enqueue_create_buffer[DType.float32](256)
    var t1_dummy = ctx.enqueue_create_buffer[DType.float32](256)
    var t1_out = ctx.enqueue_create_buffer[DType.float32](B * 256 * T)
    transit_layer[B, 512, 256, T, False](
        ctx, b1_out11, t1_out,
        t1_bn_w, t1_bn_b, t1_bn_m, t1_bn_v,
        t1_lin_w, t1_lin_b_dummy, t1_dummy)

    # ============================================================
    # block2: 24 dense TDNN layers, dilation=2; IN_C 256..1024-32
    # ============================================================
    var b2_w = List[DenseTdnnWeights]()
    for i in range(24):
        b2_w.append(load_dense(ctx, fix, 2, i))
    var b2_dummy = List[DeviceBuffer[DType.float32]]()
    for i in range(24):
        b2_dummy.append(ctx.enqueue_create_buffer[DType.float32](BN_C))
    var b2_out0 = ctx.enqueue_create_buffer[DType.float32](B * 288 * T)
    var b2_out1 = ctx.enqueue_create_buffer[DType.float32](B * 320 * T)
    var b2_out2 = ctx.enqueue_create_buffer[DType.float32](B * 352 * T)
    var b2_out3 = ctx.enqueue_create_buffer[DType.float32](B * 384 * T)
    var b2_out4 = ctx.enqueue_create_buffer[DType.float32](B * 416 * T)
    var b2_out5 = ctx.enqueue_create_buffer[DType.float32](B * 448 * T)
    var b2_out6 = ctx.enqueue_create_buffer[DType.float32](B * 480 * T)
    var b2_out7 = ctx.enqueue_create_buffer[DType.float32](B * 512 * T)
    var b2_out8 = ctx.enqueue_create_buffer[DType.float32](B * 544 * T)
    var b2_out9 = ctx.enqueue_create_buffer[DType.float32](B * 576 * T)
    var b2_out10 = ctx.enqueue_create_buffer[DType.float32](B * 608 * T)
    var b2_out11 = ctx.enqueue_create_buffer[DType.float32](B * 640 * T)
    var b2_out12 = ctx.enqueue_create_buffer[DType.float32](B * 672 * T)
    var b2_out13 = ctx.enqueue_create_buffer[DType.float32](B * 704 * T)
    var b2_out14 = ctx.enqueue_create_buffer[DType.float32](B * 736 * T)
    var b2_out15 = ctx.enqueue_create_buffer[DType.float32](B * 768 * T)
    var b2_out16 = ctx.enqueue_create_buffer[DType.float32](B * 800 * T)
    var b2_out17 = ctx.enqueue_create_buffer[DType.float32](B * 832 * T)
    var b2_out18 = ctx.enqueue_create_buffer[DType.float32](B * 864 * T)
    var b2_out19 = ctx.enqueue_create_buffer[DType.float32](B * 896 * T)
    var b2_out20 = ctx.enqueue_create_buffer[DType.float32](B * 928 * T)
    var b2_out21 = ctx.enqueue_create_buffer[DType.float32](B * 960 * T)
    var b2_out22 = ctx.enqueue_create_buffer[DType.float32](B * 992 * T)
    var b2_out23 = ctx.enqueue_create_buffer[DType.float32](B * 1024 * T)
    cam_dense_tdnn_layer[B, 256, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, t1_out, b2_out0, b2_w[0], b2_dummy[0])
    cam_dense_tdnn_layer[B, 288, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b2_out0, b2_out1, b2_w[1], b2_dummy[1])
    cam_dense_tdnn_layer[B, 320, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b2_out1, b2_out2, b2_w[2], b2_dummy[2])
    cam_dense_tdnn_layer[B, 352, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b2_out2, b2_out3, b2_w[3], b2_dummy[3])
    cam_dense_tdnn_layer[B, 384, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b2_out3, b2_out4, b2_w[4], b2_dummy[4])
    cam_dense_tdnn_layer[B, 416, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b2_out4, b2_out5, b2_w[5], b2_dummy[5])
    cam_dense_tdnn_layer[B, 448, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b2_out5, b2_out6, b2_w[6], b2_dummy[6])
    cam_dense_tdnn_layer[B, 480, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b2_out6, b2_out7, b2_w[7], b2_dummy[7])
    cam_dense_tdnn_layer[B, 512, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b2_out7, b2_out8, b2_w[8], b2_dummy[8])
    cam_dense_tdnn_layer[B, 544, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b2_out8, b2_out9, b2_w[9], b2_dummy[9])
    cam_dense_tdnn_layer[B, 576, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b2_out9, b2_out10, b2_w[10], b2_dummy[10])
    cam_dense_tdnn_layer[B, 608, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b2_out10, b2_out11, b2_w[11], b2_dummy[11])
    cam_dense_tdnn_layer[B, 640, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b2_out11, b2_out12, b2_w[12], b2_dummy[12])
    cam_dense_tdnn_layer[B, 672, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b2_out12, b2_out13, b2_w[13], b2_dummy[13])
    cam_dense_tdnn_layer[B, 704, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b2_out13, b2_out14, b2_w[14], b2_dummy[14])
    cam_dense_tdnn_layer[B, 736, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b2_out14, b2_out15, b2_w[15], b2_dummy[15])
    cam_dense_tdnn_layer[B, 768, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b2_out15, b2_out16, b2_w[16], b2_dummy[16])
    cam_dense_tdnn_layer[B, 800, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b2_out16, b2_out17, b2_w[17], b2_dummy[17])
    cam_dense_tdnn_layer[B, 832, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b2_out17, b2_out18, b2_w[18], b2_dummy[18])
    cam_dense_tdnn_layer[B, 864, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b2_out18, b2_out19, b2_w[19], b2_dummy[19])
    cam_dense_tdnn_layer[B, 896, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b2_out19, b2_out20, b2_w[20], b2_dummy[20])
    cam_dense_tdnn_layer[B, 928, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b2_out20, b2_out21, b2_w[21], b2_dummy[21])
    cam_dense_tdnn_layer[B, 960, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b2_out21, b2_out22, b2_w[22], b2_dummy[22])
    cam_dense_tdnn_layer[B, 992, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b2_out22, b2_out23, b2_w[23], b2_dummy[23])

    # ============================================================
    # transit2: TransitLayer(1024 -> 512, no bias)
    # ============================================================
    var t2_bn_w = upload_w(ctx, fix, "weights/xvector__transit2__nonlinear__batchnorm__weight.bin")
    var t2_bn_b = upload_w(ctx, fix, "weights/xvector__transit2__nonlinear__batchnorm__bias.bin")
    var t2_bn_m = upload_w(ctx, fix, "weights/xvector__transit2__nonlinear__batchnorm__running_mean.bin")
    var t2_bn_v = upload_w(ctx, fix, "weights/xvector__transit2__nonlinear__batchnorm__running_var.bin")
    var t2_lin_w = upload_w(ctx, fix, "weights/xvector__transit2__linear__weight.bin")
    var t2_lin_b_dummy = ctx.enqueue_create_buffer[DType.float32](512)
    var t2_dummy = ctx.enqueue_create_buffer[DType.float32](512)
    var t2_out = ctx.enqueue_create_buffer[DType.float32](B * 512 * T)
    transit_layer[B, 1024, 512, T, False](
        ctx, b2_out23, t2_out,
        t2_bn_w, t2_bn_b, t2_bn_m, t2_bn_v,
        t2_lin_w, t2_lin_b_dummy, t2_dummy)

    # ============================================================
    # block3: 16 dense TDNN layers, dilation=2; IN_C 512..1024-32
    # ============================================================
    var b3_w = List[DenseTdnnWeights]()
    for i in range(16):
        b3_w.append(load_dense(ctx, fix, 3, i))
    var b3_dummy = List[DeviceBuffer[DType.float32]]()
    for i in range(16):
        b3_dummy.append(ctx.enqueue_create_buffer[DType.float32](BN_C))
    var b3_out0 = ctx.enqueue_create_buffer[DType.float32](B * 544 * T)
    var b3_out1 = ctx.enqueue_create_buffer[DType.float32](B * 576 * T)
    var b3_out2 = ctx.enqueue_create_buffer[DType.float32](B * 608 * T)
    var b3_out3 = ctx.enqueue_create_buffer[DType.float32](B * 640 * T)
    var b3_out4 = ctx.enqueue_create_buffer[DType.float32](B * 672 * T)
    var b3_out5 = ctx.enqueue_create_buffer[DType.float32](B * 704 * T)
    var b3_out6 = ctx.enqueue_create_buffer[DType.float32](B * 736 * T)
    var b3_out7 = ctx.enqueue_create_buffer[DType.float32](B * 768 * T)
    var b3_out8 = ctx.enqueue_create_buffer[DType.float32](B * 800 * T)
    var b3_out9 = ctx.enqueue_create_buffer[DType.float32](B * 832 * T)
    var b3_out10 = ctx.enqueue_create_buffer[DType.float32](B * 864 * T)
    var b3_out11 = ctx.enqueue_create_buffer[DType.float32](B * 896 * T)
    var b3_out12 = ctx.enqueue_create_buffer[DType.float32](B * 928 * T)
    var b3_out13 = ctx.enqueue_create_buffer[DType.float32](B * 960 * T)
    var b3_out14 = ctx.enqueue_create_buffer[DType.float32](B * 992 * T)
    var b3_out15 = ctx.enqueue_create_buffer[DType.float32](B * 1024 * T)
    cam_dense_tdnn_layer[B, 512, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, t2_out, b3_out0, b3_w[0], b3_dummy[0])
    cam_dense_tdnn_layer[B, 544, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b3_out0, b3_out1, b3_w[1], b3_dummy[1])
    cam_dense_tdnn_layer[B, 576, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b3_out1, b3_out2, b3_w[2], b3_dummy[2])
    cam_dense_tdnn_layer[B, 608, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b3_out2, b3_out3, b3_w[3], b3_dummy[3])
    cam_dense_tdnn_layer[B, 640, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b3_out3, b3_out4, b3_w[4], b3_dummy[4])
    cam_dense_tdnn_layer[B, 672, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b3_out4, b3_out5, b3_w[5], b3_dummy[5])
    cam_dense_tdnn_layer[B, 704, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b3_out5, b3_out6, b3_w[6], b3_dummy[6])
    cam_dense_tdnn_layer[B, 736, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b3_out6, b3_out7, b3_w[7], b3_dummy[7])
    cam_dense_tdnn_layer[B, 768, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b3_out7, b3_out8, b3_w[8], b3_dummy[8])
    cam_dense_tdnn_layer[B, 800, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b3_out8, b3_out9, b3_w[9], b3_dummy[9])
    cam_dense_tdnn_layer[B, 832, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b3_out9, b3_out10, b3_w[10], b3_dummy[10])
    cam_dense_tdnn_layer[B, 864, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b3_out10, b3_out11, b3_w[11], b3_dummy[11])
    cam_dense_tdnn_layer[B, 896, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b3_out11, b3_out12, b3_w[12], b3_dummy[12])
    cam_dense_tdnn_layer[B, 928, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b3_out12, b3_out13, b3_w[13], b3_dummy[13])
    cam_dense_tdnn_layer[B, 960, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b3_out13, b3_out14, b3_w[14], b3_dummy[14])
    cam_dense_tdnn_layer[B, 992, GROWTH, BN_C, HALF_BN, T, KERNEL, 2, SEG_LEN](
        ctx, b3_out14, b3_out15, b3_w[15], b3_dummy[15])

    # ============================================================
    # transit3: TransitLayer(1024 -> 512, no bias)
    # ============================================================
    var t3_bn_w = upload_w(ctx, fix, "weights/xvector__transit3__nonlinear__batchnorm__weight.bin")
    var t3_bn_b = upload_w(ctx, fix, "weights/xvector__transit3__nonlinear__batchnorm__bias.bin")
    var t3_bn_m = upload_w(ctx, fix, "weights/xvector__transit3__nonlinear__batchnorm__running_mean.bin")
    var t3_bn_v = upload_w(ctx, fix, "weights/xvector__transit3__nonlinear__batchnorm__running_var.bin")
    var t3_lin_w = upload_w(ctx, fix, "weights/xvector__transit3__linear__weight.bin")
    var t3_lin_b_dummy = ctx.enqueue_create_buffer[DType.float32](512)
    var t3_dummy = ctx.enqueue_create_buffer[DType.float32](512)
    var t3_out = ctx.enqueue_create_buffer[DType.float32](B * 512 * T)
    transit_layer[B, 1024, 512, T, False](
        ctx, b3_out15, t3_out,
        t3_bn_w, t3_bn_b, t3_bn_m, t3_bn_v,
        t3_lin_w, t3_lin_b_dummy, t3_dummy)

    # ============================================================
    # out_nonlinear: BN1d(512) + ReLU.
    # ============================================================
    var on_bn_w = upload_w(ctx, fix, "weights/xvector__out_nonlinear__batchnorm__weight.bin")
    var on_bn_b = upload_w(ctx, fix, "weights/xvector__out_nonlinear__batchnorm__bias.bin")
    var on_bn_m = upload_w(ctx, fix, "weights/xvector__out_nonlinear__batchnorm__running_mean.bin")
    var on_bn_v = upload_w(ctx, fix, "weights/xvector__out_nonlinear__batchnorm__running_var.bin")

    var on_bn_buf = ctx.enqueue_create_buffer[DType.float32](B * 512 * T)
    var on_out_buf = ctx.enqueue_create_buffer[DType.float32](B * 512 * T)
    comptime t3_layout = row_major[B, 512, T]()
    comptime t3_p_layout = row_major[512]()
    comptime t3_flat = row_major[B * 512 * T]()
    var t3_out_t = TileTensor(t3_out, t3_layout)
    var on_bn_w_t = TileTensor(on_bn_w, t3_p_layout)
    var on_bn_b_t = TileTensor(on_bn_b, t3_p_layout)
    var on_bn_m_t = TileTensor(on_bn_m, t3_p_layout)
    var on_bn_v_t = TileTensor(on_bn_v, t3_p_layout)
    var on_bn_t = TileTensor(on_bn_buf, t3_layout)
    var on_bn_flat = TileTensor(on_bn_buf, t3_flat)
    var on_out_flat = TileTensor(on_out_buf, t3_flat)

    comptime on_bn_k = batchnorm1d_kernel[
        DType.float32, type_of(t3_layout), type_of(t3_p_layout),
        type_of(t3_layout), BLOCK,
    ]
    ctx.enqueue_function[on_bn_k, on_bn_k](
        on_bn_t, t3_out_t, on_bn_w_t, on_bn_b_t, on_bn_m_t, on_bn_v_t,
        B, 512, T, EPS,
        grid_dim=B * 512, block_dim=BLOCK,
    )
    comptime on_relu_k = relu_kernel[
        DType.float32, type_of(t3_flat), type_of(t3_flat), BLOCK,
    ]
    ctx.enqueue_function[on_relu_k, on_relu_k](
        on_out_flat, on_bn_flat, B * 512 * T,
        grid_dim=ceildiv(B * 512 * T, BLOCK), block_dim=BLOCK,
    )

    # ============================================================
    # stats: mean+std along T -> (B, 1024).
    # ============================================================
    var stats_buf = ctx.enqueue_create_buffer[DType.float32](B * 1024)
    var on_out_t = TileTensor(on_out_buf, t3_layout)
    comptime stats_layout = row_major[B, 1024]()
    var stats_t = TileTensor(stats_buf, stats_layout)
    comptime stats_k = stats_pool_kernel[
        DType.float32, type_of(t3_layout), type_of(stats_layout), BLOCK,
    ]
    ctx.enqueue_function[stats_k, stats_k](
        stats_t, on_out_t, B, 512, T,
        grid_dim=B * 512, block_dim=BLOCK,
    )

    # ============================================================
    # dense: Conv1d(1024 -> 192, k=1, no bias) on (B, 1024, 1)
    # then BN1d(192) with affine=False.
    # Input shape after unsqueeze: (B, 1024, 1). Output before BN: (B, 192, 1).
    # We treat as 2D since T=1.
    # ============================================================
    var d_lin_w = upload_w(ctx, fix, "weights/xvector__dense__linear__weight.bin")
    var d_bn_m = upload_w(ctx, fix, "weights/xvector__dense__nonlinear__batchnorm__running_mean.bin")
    var d_bn_v = upload_w(ctx, fix, "weights/xvector__dense__nonlinear__batchnorm__running_var.bin")
    var d_dummy = ctx.enqueue_create_buffer[DType.float32](192)
    var d_lin_out = ctx.enqueue_create_buffer[DType.float32](B * 192 * 1)
    var d_out = ctx.enqueue_create_buffer[DType.float32](B * 192)

    comptime stats_3d_layout = row_major[B, 1024, 1]()
    comptime d_w_layout = row_major[192, 1024, 1]()
    comptime d_out_3d_layout = row_major[B, 192, 1]()
    comptime d_2d_layout = row_major[B, 192]()
    comptime d_p_layout = row_major[192]()

    var stats_3d_t = TileTensor(stats_buf, stats_3d_layout)
    var d_lin_w_t = TileTensor(d_lin_w, d_w_layout)
    var d_dummy_t = TileTensor(d_dummy, d_p_layout)
    var d_lin_out_t = TileTensor(d_lin_out, d_out_3d_layout)

    comptime d_conv_k = conv1d_kernel_fast[
        DType.float32, type_of(stats_3d_layout), type_of(d_w_layout),
        type_of(d_p_layout), type_of(d_out_3d_layout),
        1, False, BLOCK,
    ]
    ctx.enqueue_function[d_conv_k, d_conv_k](
        d_lin_out_t, stats_3d_t, d_lin_w_t, d_dummy_t,
        B, 1024, 192, 1, 1, 1, 0, 1,
        grid_dim=B * 192, block_dim=BLOCK,
    )

    # BN no-affine on (B, 192) — view d_lin_out as (B, 192) since T=1.
    var d_lin_2d_t = TileTensor(d_lin_out, d_2d_layout)
    var d_bn_m_t = TileTensor(d_bn_m, d_p_layout)
    var d_bn_v_t = TileTensor(d_bn_v, d_p_layout)
    var d_out_t = TileTensor(d_out, d_2d_layout)
    comptime bn_na_k = bn_no_affine_2d_kernel[
        DType.float32, type_of(d_2d_layout), type_of(d_p_layout),
        type_of(d_2d_layout), BLOCK,
    ]
    ctx.enqueue_function[bn_na_k, bn_na_k](
        d_out_t, d_lin_2d_t, d_bn_m_t, d_bn_v_t,
        B, 192, EPS,
        grid_dim=ceildiv(B * 192, BLOCK), block_dim=BLOCK,
    )
    ctx.synchronize()

    # ---- Compare against xvector.bin (1, 192).
    var n_xvec = B * 192
    var max_abs: Float32 = 0.0
    var sum_abs: Float64 = 0.0
    with d_out.map_to_host() as h:
        for i in range(n_xvec):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_abs += Float64(d)
            if i < 16:
                print("xvec[", i, "]: mojo=", h[i], "  torch=", exp.data[i], "  diff=", d)
            assert_almost_equal(h[i], exp.data[i], atol=5.0e-3)
    print("CAMPPlus full xvector trunk — max abs:", max_abs, " mean:", sum_abs / Float64(n_xvec))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
