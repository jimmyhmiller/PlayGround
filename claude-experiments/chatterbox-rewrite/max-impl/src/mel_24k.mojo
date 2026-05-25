"""24kHz librosa-style mel-spectrogram for s3gen's prompt_feat.

Matches chatterbox.models.s3gen.utils.mel.mel_spectrogram:
    n_fft=1920, hop=480, win_size=1920, num_mels=80, fmin=0, fmax=8000, sr=24000
    pad reflect (n_fft - hop_size)/2 = 720 each side
    hann window (default torch.hann_window with periodic=True)
    STFT with center=False, return_complex
    spec = sqrt(re^2 + im^2 + 1e-9)
    mel = librosa_mel_basis(...) @ spec
    log_mel = log(clamp(mel, min=1e-5))   # spectral_normalize_torch
Output shape: (B, num_mels, T_frames) then .transpose(1,2) → (B, T_frames, num_mels)
in s3gen, but we'll output (B, num_mels, T_frames) here and let the caller
transpose if needed.
"""
from std.math import sin as msin, cos as mcos, log as mlog, exp as mexp, sqrt as msqrt, pi
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList


def librosa_hz_to_mel_htk(freq: Float64) -> Float64:
    """Librosa default htk=False uses Slaney formula.
    Below 1000Hz: linear; above: logarithmic.
    """
    # Slaney auditory toolbox formula.
    var min_log_hz: Float64 = 1000.0
    var min_log_mel: Float64 = 15.0
    var logstep: Float64 = mlog(6.4) / 27.0
    if freq >= min_log_hz:
        return min_log_mel + mlog(freq / min_log_hz) / logstep
    return 3.0 * freq / 200.0


def librosa_mel_to_hz_slaney(mel: Float64) -> Float64:
    var min_log_hz: Float64 = 1000.0
    var min_log_mel: Float64 = 15.0
    var logstep: Float64 = mlog(6.4) / 27.0
    if mel >= min_log_mel:
        return min_log_hz * mexp(logstep * (mel - min_log_mel))
    return 200.0 * mel / 3.0


def build_librosa_mel_filterbank(
    mut ctx: DeviceContext,
    mut filt_buf: DeviceBuffer[DType.float32],   # (n_mels, n_fft//2 + 1)
    n_mels: Int, n_fft: Int,
    sr: Float64, fmin: Float64, fmax: Float64,
) raises:
    """Build librosa.filters.mel(htk=False, norm='slaney') filterbank.

    Norm='slaney' divides each row by 2 * (mel_high - mel_low) of that bin.
    """
    var n_bins = n_fft // 2 + 1

    var fmel_min = librosa_hz_to_mel_htk(fmin)
    var fmel_max = librosa_hz_to_mel_htk(fmax)
    # Equally spaced mel points (n_mels + 2 to define the n_mels triangular filters).
    var n_points = n_mels + 2
    var mel_pts = List[Float64](capacity=n_points)
    for k in range(n_points):
        mel_pts.append(fmel_min + (fmel_max - fmel_min) * Float64(k) / Float64(n_points - 1))
    var hz_pts = List[Float64](capacity=n_points)
    for k in range(n_points):
        hz_pts.append(librosa_mel_to_hz_slaney(mel_pts[k]))

    # FFT bin frequencies.
    var fft_freqs = List[Float64](capacity=n_bins)
    for k in range(n_bins):
        fft_freqs.append(sr / 2.0 * Float64(k) / Float64(n_bins - 1))

    var data = List[Float32](capacity=n_mels * n_bins)
    for m in range(n_mels):
        var lo = hz_pts[m]
        var ce = hz_pts[m + 1]
        var hi = hz_pts[m + 2]
        # Slaney norm: enorm = 2 / (hi - lo)
        var enorm: Float64 = 2.0 / (hi - lo)
        for k in range(n_bins):
            var f = fft_freqs[k]
            var v: Float64 = 0.0
            if f > lo and f < hi:
                if f <= ce:
                    v = (f - lo) / (ce - lo)
                else:
                    v = (hi - f) / (hi - ce)
                v = v * enorm
            data.append(Float32(v))

    with filt_buf.map_to_host() as h:
        for i in range(n_mels * n_bins):
            h[i] = data[i]


def build_hann_window(
    mut ctx: DeviceContext,
    mut win_buf: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    """torch.hann_window(N, periodic=True) — i.e., periodic (default).
    Formula: w[i] = 0.5 * (1 - cos(2*pi*i / N)).
    """
    var two_pi: Float64 = 2.0 * Float64(pi)
    var data = List[Float32](capacity=n)
    for i in range(n):
        var v = 0.5 * (1.0 - mcos(two_pi * Float64(i) / Float64(n)))
        data.append(Float32(v))
    with win_buf.map_to_host() as host:
        for i in range(n):
            host[i] = data[i]


def reflect_pad_1d(
    mut ctx: DeviceContext,
    mut wav: DeviceBuffer[DType.float32],         # (N,)
    mut out_buf: DeviceBuffer[DType.float32],     # (N + 2*pad,)
    n: Int, pad: Int,
) raises:
    """Reflect padding on both sides. Reflect at boundary, not edge."""
    var wp = wav.unsafe_ptr()
    var op = out_buf.unsafe_ptr()
    var n_total = n + 2 * pad

    @always_inline
    @parameter
    @__copy_capture(wp, op, n, pad, n_total)
    def pad_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var src = i - pad
        if src < 0:
            src = -src      # reflect; for torch's reflection, b[-k] = b[k]
        elif src >= n:
            src = 2 * (n - 1) - src
        if src < 0: src = 0
        if src >= n: src = n - 1
        op[i] = wp[src]
    elementwise[pad_fn, simd_width=1, target="gpu"](
        IndexList[1](n_total), DeviceContextPtr(ctx),
    )


def mel_24k_forward(
    mut ctx: DeviceContext,
    mut wav: DeviceBuffer[DType.float32],          # (N,) raw 24kHz waveform
    mut window: DeviceBuffer[DType.float32],       # (n_fft,) hann periodic
    mut mel_fb: DeviceBuffer[DType.float32],       # (n_mels, n_fft//2 + 1)
    mut out: DeviceBuffer[DType.float32],          # (n_mels, T_frames) log-mel
    n_samples: Int,
    t_frames: Int,
    n_fft: Int = 1920,
    hop: Int = 480,
    n_mels: Int = 80,
    clip_val: Float32 = Float32(1.0e-5),
) raises:
    """24kHz librosa-style mel-spectrogram for s3gen prompt_feat.

    Steps:
        1. Reflect-pad wav by (n_fft - hop)/2 on each side.
        2. STFT (center=False) with hann window.
        3. Magnitude: spec = sqrt(re^2 + im^2 + 1e-9).
        4. Mel filterbank @ spec.
        5. log(max(mel, clip_val)).
    """
    var pad_amt = (n_fft - hop) // 2
    var n_padded = n_samples + 2 * pad_amt
    var n_bins = n_fft // 2 + 1

    # Step 1: reflect-pad.
    var wav_padded = ctx.enqueue_create_buffer[DType.float32](n_padded)
    reflect_pad_1d(ctx, wav, wav_padded, n_samples, pad_amt)

    # Step 2+3: STFT magnitude.
    var spec_mag = ctx.enqueue_create_buffer[DType.float32](n_bins * t_frames)
    var sp_ptr = spec_mag.unsafe_ptr()
    var wp_ptr = wav_padded.unsafe_ptr()
    var win_ptr = window.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(sp_ptr, wp_ptr, win_ptr, n_padded, n_fft, hop, n_bins, t_frames)
    def stft_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var k = i // t_frames
        var f = i - k * t_frames
        var re: Float32 = 0.0
        var im: Float32 = 0.0
        var two_pi_k_over_n: Float32 = -2.0 * Float32(pi) * Float32(k) / Float32(n_fft)
        for nn in range(n_fft):
            var sidx = f * hop + nn
            if sidx < n_padded:
                var x = wp_ptr[sidx] * win_ptr[nn]
                var phase = two_pi_k_over_n * Float32(nn)
                re += x * mcos(phase)
                im += x * msin(phase)
        sp_ptr[i] = msqrt(re * re + im * im + Float32(1.0e-9))
    elementwise[stft_fn, simd_width=1, target="gpu"](
        IndexList[1](n_bins * t_frames), DeviceContextPtr(ctx),
    )

    # Step 4: mel @ spec → (n_mels, T_frames).
    var op = out.unsafe_ptr()
    var mf_ptr = mel_fb.unsafe_ptr()
    var sp2 = spec_mag.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(op, mf_ptr, sp2, n_mels, n_bins, t_frames, clip_val)
    def mel_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var m = i // t_frames
        var f = i - m * t_frames
        var s: Float32 = 0.0
        for k in range(n_bins):
            s = s + mf_ptr[m * n_bins + k] * sp2[k * t_frames + f]
        if s < clip_val: s = clip_val
        op[i] = mlog(s)
    elementwise[mel_fn, simd_width=1, target="gpu"](
        IndexList[1](n_mels * t_frames), DeviceContextPtr(ctx),
    )
