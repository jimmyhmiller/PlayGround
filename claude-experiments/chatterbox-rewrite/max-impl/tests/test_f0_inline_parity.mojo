"""Inline f0 forward — copy of f0_predictor_forward in the test itself."""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from fixture import load_fp32
from weights import load_hift_generator, upload_fp32
from conv1d import conv1d_forward
from hift_generator import elu_inplace
from modules import linear_forward


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
        for i in range(8):
            print("  ", name, "[", i, "] got=", h[i], " want=", expected[i])
    var rel_l2 = sqrt(sum_diff_sq / sum_ref_sq) if sum_ref_sq > 0.0 else Float32(0.0)
    print("[", name, "] max-abs =", max_abs, " rel_l2 =", rel_l2)


def test_inline() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var hift = load_hift_generator(ctx, "weights/s3gen/mel2wav")

    var mel = upload_fp32(ctx, "weights/source_path_parity/mel.bin")

    # ─ Conv0 ─
    var h0 = ctx.enqueue_create_buffer[DType.float32](B * 512 * T_MEL)
    conv1d_forward(ctx, hift.f0_predictor.condnet[0], mel, h0, B, T_MEL, T_MEL)
    elu_inplace(ctx, h0, B * 512 * T_MEL)

    # ─ Conv1 ─
    var h1 = ctx.enqueue_create_buffer[DType.float32](B * 512 * T_MEL)
    conv1d_forward(ctx, hift.f0_predictor.condnet[1], h0, h1, B, T_MEL, T_MEL)
    elu_inplace(ctx, h1, B * 512 * T_MEL)

    # ─ Conv2 ─
    var h2 = ctx.enqueue_create_buffer[DType.float32](B * 512 * T_MEL)
    conv1d_forward(ctx, hift.f0_predictor.condnet[2], h1, h2, B, T_MEL, T_MEL)
    elu_inplace(ctx, h2, B * 512 * T_MEL)

    # ─ Conv3 ─
    var h3 = ctx.enqueue_create_buffer[DType.float32](B * 512 * T_MEL)
    conv1d_forward(ctx, hift.f0_predictor.condnet[3], h2, h3, B, T_MEL, T_MEL)
    elu_inplace(ctx, h3, B * 512 * T_MEL)

    # ─ Conv4 ─
    var h4 = ctx.enqueue_create_buffer[DType.float32](B * 512 * T_MEL)
    conv1d_forward(ctx, hift.f0_predictor.condnet[4], h3, h4, B, T_MEL, T_MEL)
    elu_inplace(ctx, h4, B * 512 * T_MEL)

    # Transpose (B, 512, T) → (B, T, 512)
    var x_btc = ctx.enqueue_create_buffer[DType.float32](B * T_MEL * 512)
    var x_ptr = h4.unsafe_ptr()
    var x_btc_ptr = x_btc.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(x_ptr, x_btc_ptr)
    def tr_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (T_MEL * 512)
        var rem = i - bi * T_MEL * 512
        var ti = rem // 512
        var ci = rem - ti * 512
        x_btc_ptr[i] = x_ptr[bi * 512 * T_MEL + ci * T_MEL + ti]
    elementwise[tr_fn, simd_width=1, target="gpu"](
        IndexList[1](B * T_MEL * 512), DeviceContextPtr(ctx),
    )

    # Linear classifier 512 → 1
    var f0_raw = ctx.enqueue_create_buffer[DType.float32](B * T_MEL * 1)
    linear_forward(ctx, hift.f0_predictor.classifier, x_btc, f0_raw, B * T_MEL)

    # Abs
    var f0_out = ctx.enqueue_create_buffer[DType.float32](B * T_MEL)
    var fp = f0_raw.unsafe_ptr()
    var fop = f0_out.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(fp, fop)
    def abs_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var v = fp[i]
        if v < 0.0: v = -v
        fop[i] = v
    elementwise[abs_fn, simd_width=1, target="gpu"](
        IndexList[1](B * T_MEL), DeviceContextPtr(ctx),
    )
    ctx.synchronize()

    var exp_f0 = load_fp32("weights/source_path_parity/f0.bin")
    stats(String("inline_f0"), f0_out, exp_f0.data, B * T_MEL)

    # Diagnostic: print h4 and x_btc samples to see if transpose worked.
    print("--- raw values ---")
    with h4.map_to_host() as h:
        print("h4[ci=0, ti=0] = h4[0] =", h[0])
        print("h4[ci=0, ti=1] = h4[1] =", h[1])
        print("h4[ci=1, ti=0] = h4[T_MEL] =", h[T_MEL])
    with x_btc.map_to_host() as h:
        print("x_btc[ti=0, ci=0] = x_btc[0] =", h[0], "(want h4[0])")
        print("x_btc[ti=1, ci=0] = x_btc[512] =", h[512], "(want h4[1])")
        print("x_btc[ti=0, ci=1] = x_btc[1] =", h[1], "(want h4[T_MEL])")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
