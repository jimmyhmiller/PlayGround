"""16kHz log-mel for s3tokenizer.

Replicates `S3Tokenizer.log_mel_spectrogram`:
    stft = torch.stft(audio, n_fft=400, hop=160, window=hann_window(400),
                       return_complex=True)
    magnitudes = stft[..., :-1].abs()**2          # power; drop last time frame
    mel_spec = mel_filters @ magnitudes            # (n_mels=128, T)
    log_spec = log10(clamp(mel_spec, min=1e-10))
    log_spec = max(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
"""
from std.math import sin as msin, cos as mcos, log as mlog, sqrt as msqrt, pi
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList


def build_hann_window_full(
    mut ctx: DeviceContext,
    mut win_buf: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    """torch.hann_window(N) default periodic=True: 0.5*(1 - cos(2*pi*i/N))."""
    var two_pi = 2.0 * Float64(pi)
    var data = List[Float32](capacity=n)
    for i in range(n):
        var v = 0.5 * (1.0 - mcos(two_pi * Float64(i) / Float64(n)))
        data.append(Float32(v))
    with win_buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def build_librosa_mel_filterbank_s3tok(
    mut ctx: DeviceContext,
    mut filt_buf: DeviceBuffer[DType.float32],   # (n_mels, n_fft//2 + 1)
    n_mels: Int, n_fft: Int, sr: Float64,
) raises:
    """librosa.filters.mel(sr=16000, n_fft=400, n_mels=128) — htk=False, slaney norm."""
    from mel_24k import build_librosa_mel_filterbank
    build_librosa_mel_filterbank(ctx, filt_buf, n_mels, n_fft, sr,
                                   Float64(0.0), sr / 2.0)


def log_mel_s3tok_forward(
    mut ctx: DeviceContext,
    mut wav: DeviceBuffer[DType.float32],         # (N,) raw 16kHz mono
    mut window: DeviceBuffer[DType.float32],      # (n_fft=400,) hann
    mut mel_fb: DeviceBuffer[DType.float32],      # (n_mels=128, n_fft//2 + 1)
    mut out: DeviceBuffer[DType.float32],         # (n_mels, T_mel) log-mel
    n_samples: Int,
    t_mel: Int,   # = ceil(n_samples / hop)
    n_fft: Int = 400,
    hop: Int = 160,
    n_mels: Int = 128,
) raises:
    """torch.stft (center=True, reflect-pad), magnitude**2, mel filter,
    log10(clamp 1e-10), max with global_max - 8, (+4)/4 normalize."""
    var pad = n_fft // 2
    var n_padded = n_samples + 2 * pad
    var n_bins = n_fft // 2 + 1

    # Step 1: reflect-pad.
    var wav_padded = ctx.enqueue_create_buffer[DType.float32](n_padded)
    var wp = wav.unsafe_ptr()
    var pp = wav_padded.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(wp, pp, n_samples, pad)
    def pad_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var src = i - pad
        if src < 0:
            src = -src
        elif src >= n_samples:
            src = 2 * (n_samples - 1) - src
        if src < 0: src = 0
        if src >= n_samples: src = n_samples - 1
        pp[i] = wp[src]
    elementwise[pad_fn, simd_width=1, target="gpu"](
        IndexList[1](n_padded), DeviceContextPtr(ctx),
    )

    # Step 2+3: STFT → |spec|^2 (power). Drop the last time frame from STFT (upstream does [..., :-1]).
    # T_frames_total = (n_padded - n_fft) // hop + 1 = upstream's STFT output; we drop the last.
    var t_stft_total = (n_padded - n_fft) // hop + 1
    var t_used = t_stft_total - 1   # = upstream's mel T after [..., :-1]
    # Caller passes t_mel = t_used. We compute spec_power of (n_bins, t_used).
    var spec_power = ctx.enqueue_create_buffer[DType.float32](n_bins * t_used)
    var sp_ptr = spec_power.unsafe_ptr()
    var wp2 = wav_padded.unsafe_ptr()
    var wn_ptr = window.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(sp_ptr, wp2, wn_ptr, n_padded, n_fft, hop, n_bins, t_used)
    def stft_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var k = i // t_used
        var f = i - k * t_used
        var re: Float32 = 0.0
        var im: Float32 = 0.0
        var two_pi_k_over_n: Float32 = -2.0 * Float32(pi) * Float32(k) / Float32(n_fft)
        for nn in range(n_fft):
            var sidx = f * hop + nn
            if sidx < n_padded:
                var x = wp2[sidx] * wn_ptr[nn]
                var phase = two_pi_k_over_n * Float32(nn)
                re += x * mcos(phase)
                im += x * msin(phase)
        sp_ptr[i] = re * re + im * im
    elementwise[stft_fn, simd_width=1, target="gpu"](
        IndexList[1](n_bins * t_used), DeviceContextPtr(ctx),
    )

    # Step 4: mel @ spec_power → (n_mels, t_used). Use t_used as t_mel.
    var op_ptr = out.unsafe_ptr()
    var mf_ptr = mel_fb.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(op_ptr, mf_ptr, sp_ptr, n_mels, n_bins, t_used)
    def mel_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var m = i // t_used
        var f = i - m * t_used
        var s: Float32 = 0.0
        for k in range(n_bins):
            s = s + mf_ptr[m * n_bins + k] * sp_ptr[k * t_used + f]
        if s < Float32(1.0e-10):
            s = Float32(1.0e-10)
        # log10 = ln(x) / ln(10) = ln(x) * 0.43429448
        op_ptr[i] = mlog(s) * Float32(0.43429448190325176)
    elementwise[mel_fn, simd_width=1, target="gpu"](
        IndexList[1](n_mels * t_used), DeviceContextPtr(ctx),
    )

    # Step 5: max with (global max - 8); then (+4)/4.
    # Two passes: first find global max, then apply.
    var n_total = n_mels * t_used
    var max_acc = ctx.enqueue_create_buffer[DType.float32](1)
    max_acc.enqueue_fill(Float32(-1.0e30))
    var ma_ptr = max_acc.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(op_ptr, ma_ptr, n_total)
    def find_max[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        # Single-threaded reduction (small T).
        var best: Float32 = -1.0e30
        for i in range(n_total):
            var v = op_ptr[i]
            if v > best: best = v
        ma_ptr[0] = best
    elementwise[find_max, simd_width=1, target="gpu"](
        IndexList[1](1), DeviceContextPtr(ctx),
    )

    var ma_ptr2 = max_acc.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(op_ptr, ma_ptr2)
    def normalize[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var v = op_ptr[i]
        var floor = ma_ptr2[0] - Float32(8.0)
        if v < floor: v = floor
        op_ptr[i] = (v + Float32(4.0)) / Float32(4.0)
    elementwise[normalize, simd_width=1, target="gpu"](
        IndexList[1](n_total), DeviceContextPtr(ctx),
    )
