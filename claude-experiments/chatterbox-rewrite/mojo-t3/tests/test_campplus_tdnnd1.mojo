"""
Parity test for CAMPPlus xvector.block1.tdnnd1 (first dense TDNN layer).

Input:  tdnn_out.bin       (1, 128, 499)  — output of xvector.tdnn
Target: tdnnd1_out.bin     (1, 160, 499)  — concat([input, cam_out], dim=1)
"""
from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import load_fp32
from conv import conv1d_kernel_fast, batchnorm1d_kernel, relu_kernel
from cam_kernels import (
    seg_pool_kernel,
    mean_along_t_kernel,
    add_t_with_bc1_kernel,
    sigmoid_kernel,
    broadcast_mul_t_kernel,
    channel_concat_kernel,
)


comptime B = 1
comptime IN_C = 128       # x_tdnn channel count (initial in_channels for tdnnd1)
comptime BN_C = 128
comptime OUT_C = 32       # growth rate
comptime HALF_BN = 64
comptime T = 499
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


def test_tdnnd1() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/campplus/"
    var ctx = DeviceContext()

    var x_in = load_fp32(fix + "tdnn_out.bin")              # (1,128,499)
    var exp = load_fp32(fix + "tdnnd1_out.bin")              # (1,160,499)

    # Weights.
    var nl1_w = upload_w(ctx, fix, "weights/xvector__block1__tdnnd1__nonlinear1__batchnorm__weight.bin")
    var nl1_b = upload_w(ctx, fix, "weights/xvector__block1__tdnnd1__nonlinear1__batchnorm__bias.bin")
    var nl1_m = upload_w(ctx, fix, "weights/xvector__block1__tdnnd1__nonlinear1__batchnorm__running_mean.bin")
    var nl1_v = upload_w(ctx, fix, "weights/xvector__block1__tdnnd1__nonlinear1__batchnorm__running_var.bin")
    var lin1_w = upload_w(ctx, fix, "weights/xvector__block1__tdnnd1__linear1__weight.bin")  # (128, 128, 1)
    var nl2_w = upload_w(ctx, fix, "weights/xvector__block1__tdnnd1__nonlinear2__batchnorm__weight.bin")
    var nl2_b = upload_w(ctx, fix, "weights/xvector__block1__tdnnd1__nonlinear2__batchnorm__bias.bin")
    var nl2_m = upload_w(ctx, fix, "weights/xvector__block1__tdnnd1__nonlinear2__batchnorm__running_mean.bin")
    var nl2_v = upload_w(ctx, fix, "weights/xvector__block1__tdnnd1__nonlinear2__batchnorm__running_var.bin")
    var cam_local_w = upload_w(ctx, fix, "weights/xvector__block1__tdnnd1__cam_layer__linear_local__weight.bin")  # (32,128,3)
    var cam_l1_w = upload_w(ctx, fix, "weights/xvector__block1__tdnnd1__cam_layer__linear1__weight.bin")  # (64,128,1)
    var cam_l1_b = upload_w(ctx, fix, "weights/xvector__block1__tdnnd1__cam_layer__linear1__bias.bin")    # (64,)
    var cam_l2_w = upload_w(ctx, fix, "weights/xvector__block1__tdnnd1__cam_layer__linear2__weight.bin")  # (32,64,1)
    var cam_l2_b = upload_w(ctx, fix, "weights/xvector__block1__tdnnd1__cam_layer__linear2__bias.bin")    # (32,)

    # Sizes.
    var n_x = B * IN_C * T            # 128 * 499
    var n_bn = B * BN_C * T           # 128 * 499
    var n_local = B * OUT_C * T       # 32 * 499
    var n_half = B * HALF_BN * T      # 64 * 499
    var n_m = B * OUT_C * T
    var n_out = B * (IN_C + OUT_C) * T   # 160 * 499

    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    upload(x_buf, x_in.data, n_x)

    # Intermediate buffers.
    var nl1_pre = ctx.enqueue_create_buffer[DType.float32](n_bn)   # BN1d(x) (no ReLU)
    var nl1_out = ctx.enqueue_create_buffer[DType.float32](n_bn)   # ReLU(BN1d(x))
    var lin1_out = ctx.enqueue_create_buffer[DType.float32](n_bn)
    var nl2_pre = ctx.enqueue_create_buffer[DType.float32](n_bn)
    var nl2_out = ctx.enqueue_create_buffer[DType.float32](n_bn)
    var y_local = ctx.enqueue_create_buffer[DType.float32](n_local)
    var ctx_mean = ctx.enqueue_create_buffer[DType.float32](B * BN_C * 1)
    var ctx_seg = ctx.enqueue_create_buffer[DType.float32](n_bn)
    var ctx_sum = ctx.enqueue_create_buffer[DType.float32](n_bn)
    var ctx_l1_pre = ctx.enqueue_create_buffer[DType.float32](n_half)
    var ctx_l1_out = ctx.enqueue_create_buffer[DType.float32](n_half)  # after ReLU
    var ctx_l2_out = ctx.enqueue_create_buffer[DType.float32](n_m)     # pre-sigmoid
    var m_buf = ctx.enqueue_create_buffer[DType.float32](n_m)          # sigmoid
    var cam_out = ctx.enqueue_create_buffer[DType.float32](n_local)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_out)

    var dummy = ctx.enqueue_create_buffer[DType.float32](BN_C)

    # Layouts.
    comptime in_layout = row_major[B, IN_C, T]()
    comptime bn_layout = row_major[B, BN_C, T]()
    comptime local_layout = row_major[B, OUT_C, T]()
    comptime half_layout = row_major[B, HALF_BN, T]()
    comptime ctx1_layout = row_major[B, BN_C, 1]()
    comptime out_layout = row_major[B, IN_C + OUT_C, T]()
    comptime w_lin1_layout = row_major[BN_C, IN_C, 1]()
    comptime w_local_layout = row_major[OUT_C, BN_C, 3]()
    comptime w_l1_layout = row_major[HALF_BN, BN_C, 1]()
    comptime w_l2_layout = row_major[OUT_C, HALF_BN, 1]()
    comptime p_bn_layout = row_major[BN_C]()
    comptime p_half_layout = row_major[HALF_BN]()
    comptime p_out_layout = row_major[OUT_C]()
    comptime flat_bn = row_major[B * BN_C * T]()
    comptime flat_half = row_major[B * HALF_BN * T]()
    comptime flat_local = row_major[B * OUT_C * T]()

    # Tensors.
    var x_t = TileTensor(x_buf, in_layout)
    var nl1_w_t = TileTensor(nl1_w, p_bn_layout)
    var nl1_b_t = TileTensor(nl1_b, p_bn_layout)
    var nl1_m_t = TileTensor(nl1_m, p_bn_layout)
    var nl1_v_t = TileTensor(nl1_v, p_bn_layout)
    var nl1_pre_t = TileTensor(nl1_pre, bn_layout)
    var nl1_out_t = TileTensor(nl1_out, bn_layout)
    var nl1_pre_flat = TileTensor(nl1_pre, flat_bn)
    var nl1_out_flat = TileTensor(nl1_out, flat_bn)

    var lin1_w_t = TileTensor(lin1_w, w_lin1_layout)
    var dummy_bn_t = TileTensor(dummy, p_bn_layout)
    var lin1_out_t = TileTensor(lin1_out, bn_layout)

    var nl2_w_t = TileTensor(nl2_w, p_bn_layout)
    var nl2_b_t = TileTensor(nl2_b, p_bn_layout)
    var nl2_m_t = TileTensor(nl2_m, p_bn_layout)
    var nl2_v_t = TileTensor(nl2_v, p_bn_layout)
    var nl2_pre_t = TileTensor(nl2_pre, bn_layout)
    var nl2_out_t = TileTensor(nl2_out, bn_layout)
    var nl2_pre_flat = TileTensor(nl2_pre, flat_bn)
    var nl2_out_flat = TileTensor(nl2_out, flat_bn)

    var cam_local_w_t = TileTensor(cam_local_w, w_local_layout)
    var y_local_t = TileTensor(y_local, local_layout)
    var ctx_mean_t = TileTensor(ctx_mean, ctx1_layout)
    var ctx_seg_t = TileTensor(ctx_seg, bn_layout)
    var ctx_sum_t = TileTensor(ctx_sum, bn_layout)

    var cam_l1_w_t = TileTensor(cam_l1_w, w_l1_layout)
    var cam_l1_b_t = TileTensor(cam_l1_b, p_half_layout)
    var ctx_l1_pre_t = TileTensor(ctx_l1_pre, half_layout)
    var ctx_l1_pre_flat = TileTensor(ctx_l1_pre, flat_half)
    var ctx_l1_out_flat = TileTensor(ctx_l1_out, flat_half)
    var ctx_l1_out_t = TileTensor(ctx_l1_out, half_layout)

    var cam_l2_w_t = TileTensor(cam_l2_w, w_l2_layout)
    var cam_l2_b_t = TileTensor(cam_l2_b, p_out_layout)
    var ctx_l2_out_t = TileTensor(ctx_l2_out, local_layout)
    var ctx_l2_out_flat = TileTensor(ctx_l2_out, flat_local)
    var m_t = TileTensor(m_buf, local_layout)
    var m_flat = TileTensor(m_buf, flat_local)
    var cam_out_t = TileTensor(cam_out, local_layout)

    var out_t = TileTensor(out_buf, out_layout)

    # ---- nonlinear1: BN1d + ReLU.
    comptime bn1d_k = batchnorm1d_kernel[
        DType.float32, type_of(in_layout), type_of(p_bn_layout),
        type_of(bn_layout), BLOCK,
    ]
    ctx.enqueue_function[bn1d_k, bn1d_k](
        nl1_pre_t, x_t, nl1_w_t, nl1_b_t, nl1_m_t, nl1_v_t,
        B, BN_C, T, EPS,
        grid_dim=B * BN_C, block_dim=BLOCK,
    )
    comptime relu_bn_k = relu_kernel[
        DType.float32, type_of(flat_bn), type_of(flat_bn), BLOCK,
    ]
    ctx.enqueue_function[relu_bn_k, relu_bn_k](
        nl1_out_flat, nl1_pre_flat, n_bn,
        grid_dim=ceildiv(n_bn, BLOCK), block_dim=BLOCK,
    )

    # ---- linear1: Conv1d(128 → 128, k=1) no bias.
    comptime conv_lin1_k = conv1d_kernel_fast[
        DType.float32, type_of(bn_layout), type_of(w_lin1_layout),
        type_of(p_bn_layout), type_of(bn_layout),
        1, False, BLOCK,
    ]
    ctx.enqueue_function[conv_lin1_k, conv_lin1_k](
        lin1_out_t, nl1_out_t, lin1_w_t, dummy_bn_t,
        B, IN_C, BN_C, T, T, 1, 0, 1,
        grid_dim=B * BN_C, block_dim=BLOCK,
    )

    # ---- nonlinear2: BN1d + ReLU.
    comptime bn1d_k2 = batchnorm1d_kernel[
        DType.float32, type_of(bn_layout), type_of(p_bn_layout),
        type_of(bn_layout), BLOCK,
    ]
    ctx.enqueue_function[bn1d_k2, bn1d_k2](
        nl2_pre_t, lin1_out_t, nl2_w_t, nl2_b_t, nl2_m_t, nl2_v_t,
        B, BN_C, T, EPS,
        grid_dim=B * BN_C, block_dim=BLOCK,
    )
    ctx.enqueue_function[relu_bn_k, relu_bn_k](
        nl2_out_flat, nl2_pre_flat, n_bn,
        grid_dim=ceildiv(n_bn, BLOCK), block_dim=BLOCK,
    )

    # ---- CAMLayer.linear_local: Conv1d(128 → 32, k=3, pad=1).
    comptime conv_local_k = conv1d_kernel_fast[
        DType.float32, type_of(bn_layout), type_of(w_local_layout),
        type_of(p_bn_layout), type_of(local_layout),
        3, False, BLOCK,
    ]
    ctx.enqueue_function[conv_local_k, conv_local_k](
        y_local_t, nl2_out_t, cam_local_w_t, dummy_bn_t,
        B, BN_C, OUT_C, T, T, 1, 1, 1,
        grid_dim=B * OUT_C, block_dim=BLOCK,
    )

    # ---- mean(nl2, dim=-1, keepdim=True).
    comptime mean_k = mean_along_t_kernel[
        DType.float32, type_of(bn_layout), type_of(ctx1_layout), BLOCK,
    ]
    ctx.enqueue_function[mean_k, mean_k](
        ctx_mean_t, nl2_out_t, B, BN_C, T,
        grid_dim=B * BN_C, block_dim=BLOCK,
    )

    # ---- seg_pool(nl2, seg_len=100).
    comptime seg_k = seg_pool_kernel[
        DType.float32, type_of(bn_layout), type_of(bn_layout),
        SEG_LEN, BLOCK,
    ]
    ctx.enqueue_function[seg_k, seg_k](
        ctx_seg_t, nl2_out_t, B, BN_C, T,
        grid_dim=B * BN_C, block_dim=BLOCK,
    )

    # ---- ctx_sum = ctx_seg + ctx_mean(broadcast over T).
    comptime add_bc_k = add_t_with_bc1_kernel[
        DType.float32, type_of(bn_layout), type_of(ctx1_layout),
        type_of(bn_layout), BLOCK,
    ]
    ctx.enqueue_function[add_bc_k, add_bc_k](
        ctx_sum_t, ctx_seg_t, ctx_mean_t, B, BN_C, T,
        grid_dim=B * BN_C, block_dim=BLOCK,
    )

    # ---- CAMLayer.linear1: Conv1d(128 → 64, k=1, bias).
    comptime conv_l1_k = conv1d_kernel_fast[
        DType.float32, type_of(bn_layout), type_of(w_l1_layout),
        type_of(p_half_layout), type_of(half_layout),
        1, True, BLOCK,
    ]
    ctx.enqueue_function[conv_l1_k, conv_l1_k](
        ctx_l1_pre_t, ctx_sum_t, cam_l1_w_t, cam_l1_b_t,
        B, BN_C, HALF_BN, T, T, 1, 0, 1,
        grid_dim=B * HALF_BN, block_dim=BLOCK,
    )
    comptime relu_half_k = relu_kernel[
        DType.float32, type_of(flat_half), type_of(flat_half), BLOCK,
    ]
    ctx.enqueue_function[relu_half_k, relu_half_k](
        ctx_l1_out_flat, ctx_l1_pre_flat, n_half,
        grid_dim=ceildiv(n_half, BLOCK), block_dim=BLOCK,
    )

    # ---- CAMLayer.linear2: Conv1d(64 → 32, k=1, bias).
    comptime conv_l2_k = conv1d_kernel_fast[
        DType.float32, type_of(half_layout), type_of(w_l2_layout),
        type_of(p_out_layout), type_of(local_layout),
        1, True, BLOCK,
    ]
    ctx.enqueue_function[conv_l2_k, conv_l2_k](
        ctx_l2_out_t, ctx_l1_out_t, cam_l2_w_t, cam_l2_b_t,
        B, HALF_BN, OUT_C, T, T, 1, 0, 1,
        grid_dim=B * OUT_C, block_dim=BLOCK,
    )

    # ---- m = sigmoid(ctx_l2_out).
    comptime sig_k = sigmoid_kernel[
        DType.float32, type_of(flat_local), type_of(flat_local), BLOCK,
    ]
    ctx.enqueue_function[sig_k, sig_k](
        m_flat, ctx_l2_out_flat, n_m,
        grid_dim=ceildiv(n_m, BLOCK), block_dim=BLOCK,
    )

    # ---- cam_out = y_local * m.
    comptime mul_k = broadcast_mul_t_kernel[
        DType.float32, type_of(local_layout), type_of(local_layout),
        type_of(local_layout), BLOCK,
    ]
    ctx.enqueue_function[mul_k, mul_k](
        cam_out_t, y_local_t, m_t, B, OUT_C, T,
        grid_dim=B * OUT_C, block_dim=BLOCK,
    )

    # ---- out = concat([x, cam_out], dim=1).
    comptime cat_k = channel_concat_kernel[
        DType.float32, type_of(in_layout), type_of(local_layout),
        type_of(out_layout), BLOCK,
    ]
    ctx.enqueue_function[cat_k, cat_k](
        out_t, x_t, cam_out_t, B, IN_C, OUT_C, T,
        grid_dim=B * (IN_C + OUT_C), block_dim=BLOCK,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    var sum_abs: Float64 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n_out):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_abs += Float64(d)
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-3)
    print("CAMPPlus block1.tdnnd1 — max abs:", max_abs,
          " mean:", sum_abs / Float64(n_out))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
