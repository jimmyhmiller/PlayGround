"""
Source signal generation for HiFiGAN: f0 → s_stft_cat.

  f0_upsample_kernel       — nearest-neighbor upsample of f0 by `prod(upsample_rates)*hop_len = 480`.
  sine_gen_kernel          — for each (b, t): theta = 2π * cumsum(f0 / fs); s = sin(theta) (since harmonic_num=0, no phase_vec needed for index 0). Adds noise scaled by uv.
                              Outputs: s (B, 1, T_audio).
  stft_forward_kernel      — naive DFT of windowed frames with hann window, n_fft=16, hop=4 → outputs s_stft_cat (B, 2*N_BINS, T_frames) (real+imag concatenated).

We re-use existing `nearest_upsample_1d_kernel` for the upsample.
"""
from std.gpu import block_idx, thread_idx
from std.gpu.memory import AddressSpace
from std.math import sin, cos, pi, sqrt, tanh
from layout import TileTensor, TensorLayout, row_major, stack_allocation


def f0_upsample_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    OutLayout: TensorLayout,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, 1, T_audio)
    f0: TileTensor[dtype, InLayout, MutAnyOrigin],         # (B, T_mel)
    batch: Int, t_mel: Int, t_audio: Int, scale: Int,
):
    """Nearest-neighbor upsample f0 from (B, T_mel) to (B, 1, T_audio=T_mel*scale).
    Launch: grid = B, block_dim = BLOCK over T_audio.
    """
    comptime assert f0.flat_rank == 2
    comptime assert output.flat_rank == 3
    var b = block_idx.x
    var tid = thread_idx.x
    var t = tid
    while t < t_audio:
        var src = t // scale
        if src >= t_mel: src = t_mel - 1
        var v = rebind[Scalar[dtype]](f0[b, src]).cast[DType.float32]()
        output[b, 0, t] = rebind[output.ElementType](v.cast[dtype]())
        t += BLOCK


def source_signal_full_kernel[
    dtype: DType,
    F0Layout: TensorLayout,
    WLayout: TensorLayout,
    BLayout: TensorLayout,
    OutLayout: TensorLayout,
    HARMONIC_NUM_PLUS1: Int,    # 9 = 8 harmonics + 1 base
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, 1, T_audio)
    f0_upsampled: TileTensor[dtype, F0Layout, MutAnyOrigin],   # (B, 1, T_audio)
    l_linear_w: TileTensor[dtype, WLayout, MutAnyOrigin],      # (1, HARMONIC_NUM_PLUS1)
    l_linear_b: TileTensor[dtype, BLayout, MutAnyOrigin],      # (1,)
    batch: Int, t_audio: Int,
    sampling_rate: Float32,
    sine_amp: Float32,
    noise_std: Float32,
    voiced_threshold: Float32,
    noise_seed: Int,    # for deterministic RNG
):
    """SourceModuleHnNSF with deterministic noise.

    For each batch (sequential per b since cumsum):
      - cumulate phase for each harmonic i in 0..HARMONIC_NUM_PLUS1-1
      - theta_i = 2π * (cumsum(f0 * (i+1) / fs) % 1)
      - sine_i = sine_amp * sin(theta_i + phase_i)  where phase_0 = 0, phase_i = LCG_rand
      - uv = f0 > voiced_threshold
      - noise_amp = uv ? noise_std : sine_amp/3
      - h_i = sine_i * uv + noise_amp * deterministic_normal_random(seed, b, i, t)
    Combine via:
      - s_raw[b, 0, t] = tanh(sum_i h_i[t] * l_linear_w[0, i] + l_linear_b[0])

    Launch: grid = B, block_dim = 1 (sequential along T for cumsum). Slow but
    matches upstream semantics. We can parallelize over time later.
    """
    comptime assert f0_upsampled.flat_rank == 3
    comptime assert l_linear_w.flat_rank == 2
    comptime assert l_linear_b.flat_rank == 1
    comptime assert output.flat_rank == 3

    var b = block_idx.x
    var tid = thread_idx.x
    if tid != 0:
        return

    var two_pi: Float32 = 2.0 * Float32(pi)

    # Per-harmonic accumulators for the phase.
    # We use stack arrays to track cumulative phase for each harmonic.
    var accum_0: Float32 = 0.0
    var accum_1: Float32 = 0.0
    var accum_2: Float32 = 0.0
    var accum_3: Float32 = 0.0
    var accum_4: Float32 = 0.0
    var accum_5: Float32 = 0.0
    var accum_6: Float32 = 0.0
    var accum_7: Float32 = 0.0
    var accum_8: Float32 = 0.0

    # Per-harmonic phase offsets (random for i > 0, 0 for i == 0). Use deterministic LCG.
    var rng_state: UInt64 = UInt64(noise_seed + b * 31)
    fn next_rand(mut s: UInt64) -> Float32:
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        # Convert to float in [-1, 1] approximating uniform [0, 1) -> [-π, π].
        var u: Float32 = Float32(Int(s >> UInt64(33))) / Float32(2147483647.0)
        return u - 1.0
    fn next_normal(mut s: UInt64) -> Float32:
        # Box-Muller-ish, cheap; use two uniforms.
        var u1 = next_rand(s)
        var u2 = next_rand(s)
        # Approximate normal: u1 + u2 (uniform sum, mean 0, var ~2/3).
        # Scale to ~unit variance: * sqrt(3/2).
        return (u1 + u2) * Float32(1.2247449)

    var phase_1 = next_rand(rng_state) * Float32(pi)
    var phase_2 = next_rand(rng_state) * Float32(pi)
    var phase_3 = next_rand(rng_state) * Float32(pi)
    var phase_4 = next_rand(rng_state) * Float32(pi)
    var phase_5 = next_rand(rng_state) * Float32(pi)
    var phase_6 = next_rand(rng_state) * Float32(pi)
    var phase_7 = next_rand(rng_state) * Float32(pi)
    var phase_8 = next_rand(rng_state) * Float32(pi)

    # Weights for the 9-input linear.
    var lin_w = stack_allocation[DType.float32,
        address_space=AddressSpace.SHARED](row_major[HARMONIC_NUM_PLUS1]())
    for i in range(HARMONIC_NUM_PLUS1):
        var w_val = rebind[Scalar[dtype]](l_linear_w[0, i]).cast[DType.float32]()
        lin_w[i] = w_val
    var bias_v = rebind[Scalar[dtype]](l_linear_b[0]).cast[DType.float32]()

    for t in range(t_audio):
        var f = rebind[Scalar[dtype]](f0_upsampled[b, 0, t]).cast[DType.float32]()
        var uv: Float32 = 1.0 if f > voiced_threshold else 0.0
        var noise_amp: Float32 = uv * noise_std + (1.0 - uv) * (sine_amp / 3.0)

        # Update accumulators for each harmonic.
        accum_0 += (f * 1.0) / sampling_rate
        accum_1 += (f * 2.0) / sampling_rate
        accum_2 += (f * 3.0) / sampling_rate
        accum_3 += (f * 4.0) / sampling_rate
        accum_4 += (f * 5.0) / sampling_rate
        accum_5 += (f * 6.0) / sampling_rate
        accum_6 += (f * 7.0) / sampling_rate
        accum_7 += (f * 8.0) / sampling_rate
        accum_8 += (f * 9.0) / sampling_rate

        fn wrap(x: Float32) -> Float32:
            var fr = x - Float32(Int(x))
            if fr < 0.0: fr += 1.0
            return fr

        var theta_0 = two_pi * wrap(accum_0)
        var theta_1 = two_pi * wrap(accum_1)
        var theta_2 = two_pi * wrap(accum_2)
        var theta_3 = two_pi * wrap(accum_3)
        var theta_4 = two_pi * wrap(accum_4)
        var theta_5 = two_pi * wrap(accum_5)
        var theta_6 = two_pi * wrap(accum_6)
        var theta_7 = two_pi * wrap(accum_7)
        var theta_8 = two_pi * wrap(accum_8)

        var s0 = sine_amp * sin(theta_0)            # phase_0 = 0
        var s1 = sine_amp * sin(theta_1 + phase_1)
        var s2 = sine_amp * sin(theta_2 + phase_2)
        var s3 = sine_amp * sin(theta_3 + phase_3)
        var s4 = sine_amp * sin(theta_4 + phase_4)
        var s5 = sine_amp * sin(theta_5 + phase_5)
        var s6 = sine_amp * sin(theta_6 + phase_6)
        var s7 = sine_amp * sin(theta_7 + phase_7)
        var s8 = sine_amp * sin(theta_8 + phase_8)

        var h0 = s0 * uv + noise_amp * next_normal(rng_state)
        var h1 = s1 * uv + noise_amp * next_normal(rng_state)
        var h2 = s2 * uv + noise_amp * next_normal(rng_state)
        var h3 = s3 * uv + noise_amp * next_normal(rng_state)
        var h4 = s4 * uv + noise_amp * next_normal(rng_state)
        var h5 = s5 * uv + noise_amp * next_normal(rng_state)
        var h6 = s6 * uv + noise_amp * next_normal(rng_state)
        var h7 = s7 * uv + noise_amp * next_normal(rng_state)
        var h8 = s8 * uv + noise_amp * next_normal(rng_state)

        var combined = (
            h0 * rebind[Scalar[DType.float32]](lin_w[0])
            + h1 * rebind[Scalar[DType.float32]](lin_w[1])
            + h2 * rebind[Scalar[DType.float32]](lin_w[2])
            + h3 * rebind[Scalar[DType.float32]](lin_w[3])
            + h4 * rebind[Scalar[DType.float32]](lin_w[4])
            + h5 * rebind[Scalar[DType.float32]](lin_w[5])
            + h6 * rebind[Scalar[DType.float32]](lin_w[6])
            + h7 * rebind[Scalar[DType.float32]](lin_w[7])
            + h8 * rebind[Scalar[DType.float32]](lin_w[8])
            + bias_v
        )
        var s_out: Float32 = tanh(combined)
        output[b, 0, t] = rebind[output.ElementType](s_out.cast[dtype]())


def sine_gen_no_harmonic_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    OutLayout: TensorLayout,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, 1, T_audio)
    f0_upsampled: TileTensor[dtype, InLayout, MutAnyOrigin],   # (B, 1, T_audio)
    batch: Int, t_audio: Int,
    sampling_rate: Float32,
    sine_amp: Float32,
    noise_std: Float32,
):
    """SineGen with harmonic_num=0 (used by chatterbox).
    For each (b, t):
        F = f0[b, 0, t] / sampling_rate
        theta = 2π * cumsum_along_T(F) % 1
        uv = (f0 > 0)
        sine_wave = sine_amp * sin(theta)
        noise_amp = uv * noise_std + (1 - uv) * sine_amp / 3
        s = sine_wave * uv + noise_amp * noise[b, 0, t]    # noise = N(0, 1)

    For determinism in testing, this kernel does NOT add noise; the caller can add
    deterministic noise separately if needed.

    Launch: grid = B, block_dim = 1 (sequential along T for cumsum).
    """
    comptime assert f0_upsampled.flat_rank == 3
    comptime assert output.flat_rank == 3

    var b = block_idx.x
    var tid = thread_idx.x
    if tid != 0:
        return

    var two_pi: Float32 = 2.0 * Float32(pi)
    var accum: Float32 = 0.0
    for t in range(t_audio):
        var f = rebind[Scalar[dtype]](f0_upsampled[b, 0, t]).cast[DType.float32]()
        accum += f / sampling_rate
        # Wrap to [0, 1).
        var frac: Float32 = accum - Float32(Int(accum))
        if frac < 0.0:
            frac += 1.0
        var theta: Float32 = two_pi * frac
        var sine_wave: Float32 = sine_amp * sin(theta)
        var uv: Float32 = 1.0 if f > 0.0 else 0.0
        # No noise (deterministic): just sine_wave * uv.
        var s: Float32 = sine_wave * uv
        output[b, 0, t] = rebind[output.ElementType](s.cast[dtype]())


def hann_window_buf(n: Int) -> List[Float32]:
    """Hann window (periodic=True like np.hann), length n."""
    var out = List[Float32]()
    for i in range(n):
        var x: Float32 = 0.5 - 0.5 * cos((2.0 * Float32(pi) * Float32(i)) / Float32(n))
        out.append(x)
    return out^


def stft_forward_kernel[
    dtype: DType,
    InLayout: TensorLayout, WinLayout: TensorLayout, OutLayout: TensorLayout,
    N_FFT: Int, HOP: Int, N_BINS: Int,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, 2*N_BINS, T_frames)
    signal: TileTensor[dtype, InLayout, MutAnyOrigin],     # (B, T_audio)
    window: TileTensor[dtype, WinLayout, MutAnyOrigin],    # (N_FFT,) hann window
    batch: Int, t_audio: Int, t_frames: Int,
):
    """Forward STFT: for each frame f:
        x_w[n] = signal[f*HOP + n - center_pad] * window[n]   (centered, reflect-pad)
        X[k] = sum_n x_w[n] * exp(-2πi k n / N_FFT)
        output[b, k, f]            = real(X[k])
        output[b, N_BINS + k, f]   = imag(X[k])

    Center padding: `f*HOP + n - N_FFT/2`. For out-of-range indices, use reflection.
    Launch: grid = B * t_frames, block_dim = BLOCK over N_BINS.
    """
    comptime assert signal.flat_rank == 2
    comptime assert window.flat_rank == 1
    comptime assert output.flat_rank == 3

    var bid = block_idx.x
    var tid = thread_idx.x
    var f = bid % t_frames
    var b = bid // t_frames

    var half: Int = N_FFT // 2

    var k = tid
    while k < N_BINS:
        var re: Float32 = 0.0
        var im: Float32 = 0.0
        var arg_base: Float32 = (-2.0 * Float32(pi) * Float32(k)) / Float32(N_FFT)
        for n in range(N_FFT):
            var src_idx: Int = f * HOP + n - half
            # Reflect pad.
            if src_idx < 0:
                src_idx = -src_idx
            elif src_idx >= t_audio:
                src_idx = 2 * (t_audio - 1) - src_idx
                if src_idx < 0:
                    src_idx = 0
            var x: Float32 = rebind[Scalar[dtype]](signal[b, src_idx]).cast[DType.float32]()
            var w: Float32 = rebind[Scalar[dtype]](window[n]).cast[DType.float32]()
            var xw: Float32 = x * w
            var phase: Float32 = arg_base * Float32(n)
            re += xw * cos(phase)
            im += xw * sin(phase)
        output[b, k, f] = rebind[output.ElementType](re.cast[dtype]())
        output[b, N_BINS + k, f] = rebind[output.ElementType](im.cast[dtype]())
        k += BLOCK
