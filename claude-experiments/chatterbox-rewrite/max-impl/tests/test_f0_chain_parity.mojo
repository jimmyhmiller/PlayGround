"""Check the f0_predictor conv0 → elu → conv1 chain."""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt

from fixture import load_fp32
from weights import load_hift_generator, upload_fp32
from conv1d import conv1d_forward
from hift_generator import elu_inplace


comptime B = 1
comptime T_MEL = 60


def stats(name: String, got: DeviceBuffer[DType.float32], expected: List[Float32], n: Int) raises:
    var max_abs: Float32 = 0.0
    var sum_diff_sq: Float32 = 0.0
    var sum_ref_sq: Float32 = 0.0
    with got.map_to_host() as h:
        for i in range(n):
            var dd = h[i] - expected[i]
            var d = dd
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_diff_sq += dd * dd
            sum_ref_sq += expected[i] * expected[i]
        for i in range(4):
            print("  ", name, "[", i, "] got=", h[i], " want=", expected[i])
    var rel_l2 = sqrt(sum_diff_sq / sum_ref_sq) if sum_ref_sq > 0.0 else Float32(0.0)
    print("[", name, "] max-abs =", max_abs, " rel_l2 =", rel_l2)


def test_chain() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var hift = load_hift_generator(ctx, "weights/s3gen/mel2wav")

    var mel = upload_fp32(ctx, "weights/source_path_parity/mel.bin")

    # Step 1: conv0
    var h0 = ctx.enqueue_create_buffer[DType.float32](B * 512 * T_MEL)
    conv1d_forward(ctx, hift.f0_predictor.condnet[0], mel, h0, B, T_MEL, T_MEL)
    ctx.synchronize()
    var exp_conv0 = load_fp32("weights/source_path_parity/conv0.bin")
    stats(String("conv0"), h0, exp_conv0.data, B * 512 * T_MEL)

    # Step 2: ELU
    elu_inplace(ctx, h0, B * 512 * T_MEL)
    ctx.synchronize()
    var exp_elu0 = load_fp32("weights/source_path_parity/elu0.bin")
    stats(String("elu0"), h0, exp_elu0.data, B * 512 * T_MEL)

    # Step 3: conv1 (condnet index 1 in our loader = upstream condnet index 2)
    var h1 = ctx.enqueue_create_buffer[DType.float32](B * 512 * T_MEL)
    conv1d_forward(ctx, hift.f0_predictor.condnet[1], h0, h1, B, T_MEL, T_MEL)
    ctx.synchronize()
    var exp_conv1 = load_fp32("weights/source_path_parity/conv1.bin")
    stats(String("conv1"), h1, exp_conv1.data, B * 512 * T_MEL)

    # Now repeat using the same f0_predictor_forward call but isolate intermediates.
    # Replicate the function's internal logic step by step:
    from hift_generator import f0_predictor_forward
    var f0_out = ctx.enqueue_create_buffer[DType.float32](B * T_MEL)
    f0_predictor_forward(ctx, hift.f0_predictor, mel, f0_out, B, T_MEL)
    ctx.synchronize()
    var exp_f0 = load_fp32("weights/source_path_parity/f0.bin")
    print("[f0 from f0_predictor_forward]")
    stats(String("f0"), f0_out, exp_f0.data, B * T_MEL)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
