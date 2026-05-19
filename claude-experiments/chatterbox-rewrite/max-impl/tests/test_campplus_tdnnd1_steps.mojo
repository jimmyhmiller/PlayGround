"""tdnnd1 step-by-step parity:
  step 1: nl1 (BN+ReLU on 128)
  step 2: linear1 (Conv1d 128 → 128 1x1)
  step 3: nl2 (BN+ReLU on 128)
  step 4: cam_layer (full CAM)
"""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt

from fixture import load_fp32
from weights import load_campplus, upload_fp32
from campplus import (
    batchnorm1d_forward, relu_inplace_bct, cam_layer_forward,
)
from conv1d import conv1d_forward


comptime B = 1
comptime T = 8
comptime IN_CH = 128
comptime GROWTH = 32


def stats(name: String, got: DeviceBuffer[DType.float32], expected: List[Float32], n: Int) raises:
    var max_abs: Float32 = 0.0
    var sum_diff_sq: Float32 = 0.0
    var sum_ref_sq: Float32 = 0.0
    with got.map_to_host() as h:
        for i in range(n):
            var d = h[i] - expected[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            var dd = h[i] - expected[i]
            sum_diff_sq += dd * dd
            sum_ref_sq += expected[i] * expected[i]
        for i in range(4):
            print("  [", i, "] got=", h[i], " want=", expected[i])
    var rel_l2 = sqrt(sum_diff_sq / sum_ref_sq) if sum_ref_sq > 0.0 else Float32(0.0)
    print("[", name, "] max-abs =", max_abs, " rel_l2 =", rel_l2)


def test_tdnnd1_steps() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var cp = load_campplus(ctx, "weights/s3gen/speaker_encoder")

    var tdnn_out = upload_fp32(ctx, "weights/campplus_parity/tdnn_out.bin")

    # nl1: BN(128) + ReLU
    var h1 = ctx.enqueue_create_buffer[DType.float32](B * IN_CH * T)
    batchnorm1d_forward(ctx, cp.xvector.block1.layers[0].nonlinear1, tdnn_out, h1, B, IN_CH, T)
    relu_inplace_bct(ctx, h1, B * IN_CH * T)
    ctx.synchronize()
    var exp_nl1 = load_fp32("weights/campplus_parity/tdnnd1_nl1_out.bin")
    stats(String("nl1"), h1, exp_nl1.data, B * IN_CH * T)

    # linear1: 1x1 Conv 128 → 128 (bn_channels = 128)
    var h2 = ctx.enqueue_create_buffer[DType.float32](B * 128 * T)
    conv1d_forward(ctx, cp.xvector.block1.layers[0].linear1, h1, h2, B, T, T)
    ctx.synchronize()
    var exp_lin1 = load_fp32("weights/campplus_parity/tdnnd1_lin1_out.bin")
    stats(String("lin1"), h2, exp_lin1.data, B * 128 * T)

    # nl2: BN(128) + ReLU
    var h3 = ctx.enqueue_create_buffer[DType.float32](B * 128 * T)
    batchnorm1d_forward(ctx, cp.xvector.block1.layers[0].nonlinear2, h2, h3, B, 128, T)
    relu_inplace_bct(ctx, h3, B * 128 * T)
    ctx.synchronize()
    var exp_nl2 = load_fp32("weights/campplus_parity/tdnnd1_nl2_out.bin")
    stats(String("nl2"), h3, exp_nl2.data, B * 128 * T)

    # cam_layer: 128 → growth=32 (with k=3, pad=1, dil=1)
    var h4 = ctx.enqueue_create_buffer[DType.float32](B * GROWTH * T)
    cam_layer_forward(ctx, cp.xvector.block1.layers[0].cam_layer, h3, h4,
                      B, 128, GROWTH, T, 3, 1, 1)
    ctx.synchronize()
    var exp_cam = load_fp32("weights/campplus_parity/tdnnd1_cam_out.bin")
    stats(String("cam"), h4, exp_cam.data, B * GROWTH * T)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
