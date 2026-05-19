"""Kaldi-style log-mel filterbank features.

Matches torchaudio.compliance.kaldi.fbank(waveform, num_mel_bins=80) called
by Chatterbox's CAMPPlus speaker_encoder.inference (extract_feature).

Pipeline (sample_rate=16000, frame_length=25ms=400, frame_shift=10ms=160,
padded_window=512 = next pow 2 of 400):
  1. _get_strided: extract (T_frames, 400) frames with snip_edges=True
  2. Remove DC offset per frame (subtract frame mean)
  3. Compute raw_energy per frame (= log(max(sum(x^2), eps))) — UNUSED in our output
  4. Pre-emphasis: x[i,j] -= 0.97 * x[i, max(0, j-1)]
  5. Multiply by povey window: hann(400, periodic=False)^0.85
  6. Zero-pad to 512 (right side only)
  7. rFFT → magnitude → power (^2)
  8. Multiply by kaldi mel filterbank (80, 257) — pad right=0 to (80, 257)
  9. Log (max with epsilon)
 10. Per-utterance subtract: feat -= feat.mean(dim=0)  (DONE BY CAMPPlus extract_feature)
"""
from std.math import sin as msin, cos as mcos, log as mlog, exp as mexp, sqrt as msqrt, pi
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList


# ============================================================================
# Mel filterbank generation (host-side; tiny so we just build on CPU)
# ============================================================================

def mel_scale_scalar(freq: Float64) -> Float64:
    return 1127.0 * mlog(1.0 + freq / 700.0)


def kaldi_mel_filterbank_size(num_bins: Int, padded_window_size: Int) -> Int:
    """Return the size of the kaldi mel filterbank (num_bins * (padded_window_size//2 + 1))."""
    return num_bins * (padded_window_size // 2 + 1)


def build_kaldi_mel_filterbank(
    mut ctx: DeviceContext,
    mut filt_buf: DeviceBuffer[DType.float32],   # (num_bins, n_fft_bins+1) — padded right=0
    num_bins: Int,
    padded_window_size: Int,
    sample_freq: Float64,
    low_freq: Float64 = 20.0,
    high_freq: Float64 = 0.0,
) raises:
    """Build kaldi triangular mel filterbank into filt_buf.

    Output layout: (num_bins, padded_window_size // 2 + 1), row-major.
    Last column (the +1 bin) is zero (kaldi pads it before matmul with full STFT).
    """
    var num_fft_bins = padded_window_size // 2
    var nyquist = 0.5 * sample_freq
    var hi = high_freq
    if hi <= 0.0:
        hi = nyquist
    var fft_bin_width = sample_freq / Float64(padded_window_size)
    var mel_low = mel_scale_scalar(low_freq)
    var mel_high = mel_scale_scalar(hi)
    var mel_delta = (mel_high - mel_low) / Float64(num_bins + 1)

    var n_bins_out = num_fft_bins + 1   # including the padded zero column

    var data = List[Float32](capacity=num_bins * n_bins_out)
    for b in range(num_bins):
        var left_mel = mel_low + Float64(b) * mel_delta
        var center_mel = mel_low + Float64(b + 1) * mel_delta
        var right_mel = mel_low + Float64(b + 2) * mel_delta
        for k in range(num_fft_bins):
            var freq = fft_bin_width * Float64(k)
            var mel_k = mel_scale_scalar(freq)
            var v: Float64 = 0.0
            if mel_k > left_mel and mel_k < right_mel:
                if mel_k <= center_mel:
                    v = (mel_k - left_mel) / (center_mel - left_mel)
                else:
                    v = (right_mel - mel_k) / (right_mel - center_mel)
            data.append(Float32(v))
        data.append(Float32(0.0))   # pad right column with zero (kaldi.pad mode constant)

    with filt_buf.map_to_host() as h:
        for i in range(num_bins * n_bins_out):
            h[i] = data[i]


# ============================================================================
# Povey window: hann(N, periodic=False)^0.85
# ============================================================================

def build_povey_window(
    mut ctx: DeviceContext,
    mut win_buf: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    """torch.hann_window(N, periodic=False) ** 0.85.

    Non-periodic Hann: w[i] = 0.5 * (1 - cos(2*pi*i / (N - 1))), i in [0, N).
    """
    var data = List[Float32](capacity=n)
    var two_pi: Float64 = 2.0 * Float64(pi)
    for i in range(n):
        var h = 0.5 * (1.0 - mcos(two_pi * Float64(i) / Float64(n - 1)))
        # ^ 0.85 via exp(0.85 * log(h)) when h>0
        var v: Float64 = 0.0
        if h > 0.0:
            v = mexp(0.85 * mlog(h))
        data.append(Float32(v))
    with win_buf.map_to_host() as host:
        for i in range(n):
            host[i] = data[i]


# ============================================================================
# Kaldi fbank forward
# ============================================================================

def kaldi_fbank_forward(
    mut ctx: DeviceContext,
    mut wav: DeviceBuffer[DType.float32],         # (N,) raw waveform at 16kHz
    mut window: DeviceBuffer[DType.float32],      # (window_size=400,) povey window
    mut mel_fb: DeviceBuffer[DType.float32],      # (num_mel_bins, n_fft//2 + 1)
    mut out: DeviceBuffer[DType.float32],         # (T_frames, num_mel_bins) log-mel
    n_samples: Int,
    t_frames: Int,
    window_size: Int = 400,
    window_shift: Int = 160,
    padded_window_size: Int = 512,
    num_mel_bins: Int = 80,
    preemph: Float32 = Float32(0.97),
    eps: Float32 = Float32(1.1920928955078125e-07),   # numeric_limits<float>::epsilon()
) raises:
    """Single batch (mono). Output is log-mel-power features.

    NOTE: snip_edges=True, raw_energy=True, remove_dc_offset=True, dither=0.
    Output is (T_frames, num_mel_bins) — caller is responsible for the
    per-utterance mean subtraction in `extract_feature`.
    """
    var n_fft_bins = padded_window_size // 2 + 1

    # Step 1+2: extract strided frames, remove DC offset.
    # frames[i, j] = wav[i*shift + j] for j in [0, window_size), with
    # then mean-subtracted per frame.
    var frames = ctx.enqueue_create_buffer[DType.float32](t_frames * padded_window_size)
    frames.enqueue_fill(0.0)
    var wp = wav.unsafe_ptr()
    var fp = frames.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(wp, fp, t_frames, window_size, window_shift, padded_window_size, preemph)
    def frame_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var t = idx[0]
        # Compute frame mean for DC offset removal.
        var s: Float32 = 0.0
        var base_in = t * window_shift
        for j in range(window_size):
            s = s + wp[base_in + j]
        var mean = s / Float32(window_size)
        # Compute pre-emphasis input vector: x[j] = wav[base+j] - mean.
        # Then apply pre-emphasis: y[j] = x[j] - 0.97 * x[max(0, j-1)].
        # We process in a single pass storing into frames.
        var base_out = t * padded_window_size
        # j=0: x[-1] replicate-padded to x[0], so y[0] = x[0] - preemph * x[0] = x[0] * (1 - preemph).
        var prev_x: Float32 = wp[base_in] - mean
        fp[base_out + 0] = prev_x - preemph * prev_x
        for j in range(1, window_size):
            var x: Float32 = wp[base_in + j] - mean
            fp[base_out + j] = x - preemph * prev_x
            prev_x = x
    elementwise[frame_fn, simd_width=1, target="gpu"](
        IndexList[1](t_frames), DeviceContextPtr(ctx),
    )

    # Step 3+4: multiply by povey window. (Pre-emphasis already applied above.)
    var win_ptr = window.unsafe_ptr()
    var fp2 = frames.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(fp2, win_ptr, window_size, padded_window_size)
    def win_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var t = i // padded_window_size
        var j = i - t * padded_window_size
        if j < window_size:
            fp2[i] = fp2[i] * win_ptr[j]
        # else: already zero (padded)
    elementwise[win_fn, simd_width=1, target="gpu"](
        IndexList[1](t_frames * padded_window_size), DeviceContextPtr(ctx),
    )

    # Step 5: rFFT(strided_input) -> magnitudes -> power (squared).
    # Output: (t_frames, n_fft_bins) where each value is |X[k]|^2.
    var spec_power = ctx.enqueue_create_buffer[DType.float32](t_frames * n_fft_bins)
    var sp_ptr = spec_power.unsafe_ptr()
    var fp3 = frames.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(fp3, sp_ptr, t_frames, padded_window_size, n_fft_bins)
    def fft_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var t = i // n_fft_bins
        var k = i - t * n_fft_bins
        var base = t * padded_window_size
        var re: Float32 = 0.0
        var im: Float32 = 0.0
        var two_pi_k_over_n: Float32 = -2.0 * Float32(pi) * Float32(k) / Float32(padded_window_size)
        for n in range(padded_window_size):
            var x = fp3[base + n]
            var ang = two_pi_k_over_n * Float32(n)
            re = re + x * mcos(ang)
            im = im + x * msin(ang)
        sp_ptr[i] = re * re + im * im
    elementwise[fft_fn, simd_width=1, target="gpu"](
        IndexList[1](t_frames * n_fft_bins), DeviceContextPtr(ctx),
    )

    # Step 6: matmul mel filterbank (num_mel_bins, n_fft_bins) @ spec_power^T.
    # Output: (t_frames, num_mel_bins), then log(max(.,eps)).
    var mf_ptr = mel_fb.unsafe_ptr()
    var sp2 = spec_power.unsafe_ptr()
    var op = out.unsafe_ptr()
    var n_bins_in = n_fft_bins   # mel filterbank shape (num_bins, n_fft_bins)

    @always_inline
    @parameter
    @__copy_capture(mf_ptr, sp2, op, t_frames, num_mel_bins, n_bins_in, eps)
    def mel_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var t = i // num_mel_bins
        var m = i - t * num_mel_bins
        var s: Float32 = 0.0
        for k in range(n_bins_in):
            s = s + mf_ptr[m * n_bins_in + k] * sp2[t * n_bins_in + k]
        if s < eps:
            s = eps
        op[i] = mlog(s)
    elementwise[mel_fn, simd_width=1, target="gpu"](
        IndexList[1](t_frames * num_mel_bins), DeviceContextPtr(ctx),
    )


def kaldi_subtract_column_mean(
    mut ctx: DeviceContext,
    mut feat: DeviceBuffer[DType.float32],     # (T_frames, num_mel_bins) — modified in place
    t_frames: Int, num_mel_bins: Int,
) raises:
    """Subtract per-time-mean (extract_feature does this AFTER kaldi.fbank)."""
    # Compute mean per time frame (mean over mel-bin dim).
    # Wait — extract_feature does `feature - feature.mean(dim=0, keepdim=True)`.
    # Note: feature has shape (T, F), so dim=0 is TIME axis → mean over frames.
    # Output: per-feature mean, broadcast back. So column mean.
    var fp = feat.unsafe_ptr()
    var col_means = ctx.enqueue_create_buffer[DType.float32](num_mel_bins)
    var cmp = col_means.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(fp, cmp, t_frames, num_mel_bins)
    def col_mean_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var m = idx[0]
        var s: Float32 = 0.0
        for t in range(t_frames):
            s = s + fp[t * num_mel_bins + m]
        cmp[m] = s / Float32(t_frames)
    elementwise[col_mean_fn, simd_width=1, target="gpu"](
        IndexList[1](num_mel_bins), DeviceContextPtr(ctx),
    )

    var fp2 = feat.unsafe_ptr()
    var cmp2 = col_means.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(fp2, cmp2, t_frames, num_mel_bins)
    def sub_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var m = i - (i // num_mel_bins) * num_mel_bins
        fp2[i] = fp2[i] - cmp2[m]
    elementwise[sub_fn, simd_width=1, target="gpu"](
        IndexList[1](t_frames * num_mel_bins), DeviceContextPtr(ctx),
    )
