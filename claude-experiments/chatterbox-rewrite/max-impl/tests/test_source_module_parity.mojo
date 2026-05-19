"""Tight parity test on the source module. Feed upstream's exact f0, phase_vec,
and noise; compare Mojo's sine_waves (pre-merge), theta_mat, and post-merge
against the dumped tensors."""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.math import sqrt, sin as msin
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from fixture import load_fp32
from weights import load_hift_generator, upload_fp32
from hift_generator import source_module_forward_deterministic


def _diff(name: String, mut mojo: DeviceBuffer[DType.float32], ref_path: String) raises:
    var reference = load_fp32(ref_path)
    var ref_n = reference.numel()
    var max_abs: Float32 = 0.0
    var sum_diff_sq: Float32 = 0.0
    var sum_ref_sq: Float32 = 0.0
    var first_mismatches = 0
    with mojo.map_to_host() as h:
        for i in range(ref_n):
            var dd = h[i] - reference.data[i]
            var d = dd
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            if first_mismatches < 8 and d > 1e-3:
                print("  i=", i, " mojo=", h[i], " ref=", reference.data[i], " diff=", d)
                first_mismatches += 1
            sum_diff_sq += dd * dd
            sum_ref_sq += reference.data[i] * reference.data[i]
    var rel = sqrt(sum_diff_sq / sum_ref_sq) if sum_ref_sq > 0.0 else Float32(0.0)
    print("[parity]", name, ": max-abs=", max_abs, " rel_l2=", rel, " (n=", ref_n, ")")


def compute_theta_mat_mojo(
    mut ctx: DeviceContext,
    mut f0_up: DeviceBuffer[DType.float32],
    mut theta_out: DeviceBuffer[DType.float32],
    b: Int, n_harm: Int, t_audio: Int, sample_rate: Int,
) raises:
    var sr_f = Float32(sample_rate)
    var f0p = f0_up.unsafe_ptr()
    var op = theta_out.unsafe_ptr()
    var two_pi: Float32 = 2.0 * Float32(3.141592653589793)

    @always_inline
    @parameter
    @__copy_capture(f0p, op, b, n_harm, t_audio, sr_f, two_pi)
    def theta_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // n_harm
        var h = i - bi * n_harm
        var acc: Float32 = 0.0
        for t in range(t_audio):
            var f0_val = f0p[bi * t_audio + t]
            var df: Float32 = f0_val * Float32(h + 1) / sr_f
            acc = acc + df
            while acc >= 1.0:
                acc = acc - 1.0
            while acc < 0.0:
                acc = acc + 1.0
            op[bi * n_harm * t_audio + h * t_audio + t] = two_pi * acc
    elementwise[theta_fn, simd_width=1, target="gpu"](
        IndexList[1](b * n_harm), DeviceContextPtr(ctx),
    )


def test_source_module_parity() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var fix = "weights/s3gen_prompt/hift_dump/"

    var B = 1
    var N_HARM = 9
    var T = 48960

    var hift = load_hift_generator(ctx, "weights/s3gen/mel2wav")

    # Use upstream's f0_upsampled (shape (1, 1, 48960)) as input — it's the SAME
    # tensor that upstream's SineGen.forward got, so f0 is exact.
    var f0_upstream = upload_fp32(ctx, fix + "sg_f0_upsampled.bin")
    # Flatten (1,1,T) to (B, T_audio) — same contiguous layout.
    var f0_up = ctx.enqueue_create_buffer[DType.float32](B * T)
    ctx.enqueue_copy(f0_up, f0_upstream)
    ctx.synchronize()

    # Compare theta_mat (no phase, no sin yet — just the cumulative phase).
    var theta_mojo = ctx.enqueue_create_buffer[DType.float32](B * N_HARM * T)
    compute_theta_mat_mojo(ctx, f0_up, theta_mojo, B, N_HARM, T, 24000)
    ctx.synchronize()
    _diff("theta_mat (no phase)", theta_mojo, fix + "sg_theta_mat.bin")

    # Compare Mojo's sine_waves_post_merge (B, n_harm, T) against upstream's.
    var phase_vec = upload_fp32(ctx, fix + "sg_phase_vec.bin")
    var noise_buf = upload_fp32(ctx, fix + "sg_noise.bin")

    var sine_waves_mojo = ctx.enqueue_create_buffer[DType.float32](B * N_HARM * T)
    var f0p = f0_up.unsafe_ptr()
    var pp = phase_vec.unsafe_ptr()
    var np_ = noise_buf.unsafe_ptr()
    var swp = sine_waves_mojo.unsafe_ptr()
    var sample_rate_f = Float32(24000.0)
    var sine_amp = Float32(0.1)
    var voiced_threshold = Float32(10.0)
    var two_pi_v: Float32 = 2.0 * Float32(3.141592653589793)

    @always_inline
    @parameter
    @__copy_capture(f0p, pp, np_, swp, B, N_HARM, T, sample_rate_f, sine_amp, voiced_threshold, two_pi_v)
    def sw_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // N_HARM
        var h = i - bi * N_HARM
        var phase = pp[bi * N_HARM + h]
        var theta_acc: Float32 = 0.0
        for t in range(T):
            var f0_val = f0p[bi * T + t]
            var uv: Float32 = 1.0 if f0_val > voiced_threshold else 0.0
            var df: Float32 = f0_val * Float32(h + 1) / sample_rate_f
            theta_acc = theta_acc + df
            while theta_acc >= 1.0:
                theta_acc = theta_acc - 1.0
            var theta = two_pi_v * theta_acc + phase
            var sine = sine_amp * msin(theta)
            var noise = np_[bi * N_HARM * T + h * T + t]
            swp[bi * N_HARM * T + h * T + t] = sine * uv + noise
    elementwise[sw_fn, simd_width=1, target="gpu"](
        IndexList[1](B * N_HARM), DeviceContextPtr(ctx),
    )
    ctx.synchronize()
    _diff("sine_waves_post_merge", sine_waves_mojo, fix + "sg_sine_waves_post_merge.bin")

    # Now run full deterministic source module end-to-end.
    var sine_merge = ctx.enqueue_create_buffer[DType.float32](B * 1 * T)
    source_module_forward_deterministic(
        ctx, hift.m_source, f0_up, phase_vec, noise_buf, sine_merge,
        B, T, sampling_rate=24000, harmonic_num=8,
        sine_amp=Float32(0.1), voiced_threshold=Float32(10.0),
    )
    ctx.synchronize()
    _diff("sine_merge (full deterministic)", sine_merge, fix + "sine_merge.bin")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
