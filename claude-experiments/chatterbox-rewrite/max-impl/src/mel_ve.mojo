"""VoiceEncoder mel: 16kHz librosa-style, n_mels=40, fmin=0, fmax=8000.

Matches chatterbox.models.voice_encoder.melspec.melspectrogram with
VoiceEncConfig:
    n_fft=400, hop=160, win=400, n_mels=40, fmin=0, fmax=8000,
    preemphasis=0 (skipped), mel_power=2, mel_type='amp', normalized_mels=False.

Output is (T, n_mels=40) — same layout as upstream's mel.T.
"""
from std.math import sin as msin, cos as mcos, log as mlog, sqrt as msqrt, pi
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList


def mel_ve_forward(
    mut ctx: DeviceContext,
    mut wav: DeviceBuffer[DType.float32],         # (N,) 16k mono
    mut window: DeviceBuffer[DType.float32],      # (n_fft=400,) hann periodic
    mut mel_fb: DeviceBuffer[DType.float32],      # (n_mels=40, n_fft//2+1=201)
    mut out: DeviceBuffer[DType.float32],         # (T, n_mels) — transposed to upstream's mel.T
    n_samples: Int,
    t_frames: Int,
    n_fft: Int = 400,
    hop: Int = 160,
    n_mels: Int = 40,
) raises:
    """librosa STFT (center=True reflect-pad), |spec|^2, mel filter, no log."""
    var pad = n_fft // 2
    var n_padded = n_samples + 2 * pad
    var n_bins = n_fft // 2 + 1

    # Reflect-pad.
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

    # STFT → power.
    var spec_power = ctx.enqueue_create_buffer[DType.float32](n_bins * t_frames)
    var sp = spec_power.unsafe_ptr()
    var wp2 = wav_padded.unsafe_ptr()
    var win_ptr = window.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(sp, wp2, win_ptr, n_padded, n_fft, hop, n_bins, t_frames)
    def stft_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var k = i // t_frames
        var f = i - k * t_frames
        var re: Float32 = 0.0
        var im: Float32 = 0.0
        var two_pi_k_over_n: Float32 = -2.0 * Float32(pi) * Float32(k) / Float32(n_fft)
        for n in range(n_fft):
            var sidx = f * hop + n
            if sidx < n_padded:
                var x = wp2[sidx] * win_ptr[n]
                var phase = two_pi_k_over_n * Float32(n)
                re += x * mcos(phase)
                im += x * msin(phase)
        sp[i] = re * re + im * im
    elementwise[stft_fn, simd_width=1, target="gpu"](
        IndexList[1](n_bins * t_frames), DeviceContextPtr(ctx),
    )

    # Mel @ power → (n_mels, T). Transpose to (T, n_mels).
    var op = out.unsafe_ptr()
    var mf = mel_fb.unsafe_ptr()
    var sp2 = spec_power.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(op, mf, sp2, n_mels, n_bins, t_frames)
    def mel_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        # i indexes (T, n_mels).
        var t = i // n_mels
        var m = i - t * n_mels
        var s: Float32 = 0.0
        for k in range(n_bins):
            s = s + mf[m * n_bins + k] * sp2[k * t_frames + t]
        op[i] = s
    elementwise[mel_fn, simd_width=1, target="gpu"](
        IndexList[1](n_mels * t_frames), DeviceContextPtr(ctx),
    )
