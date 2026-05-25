"""
Naive STFT / iSTFT kernels matching torch.stft / torch.istft for small n_fft.

Matches the HiFiGAN configuration (n_fft=16, hop_len=4, center=True, hann window).
Computes the DFT directly (no FFT) — n_fft is tiny so O(N^2) per frame is fine.

  stft_kernel:   (B, T) signal → (B, n_freq, n_frames) real & imag spectrogram
  istft_kernel:  (B, n_freq, n_frames) real & imag → (B, T) reconstructed signal

n_freq = n_fft // 2 + 1 (we only store the non-redundant half — torch does too).
"""

from std.math import sin, cos
from std.gpu import block_idx, thread_idx
from layout import TileTensor, TensorLayout


# Reflection padding for center=True torch.stft: index i in [-n_fft//2, T+n_fft//2)
# maps to a valid index in [0, T) by reflection. Torch uses "reflect" mode which
# bounces off the edges without repeating the boundary sample:
#   pad_left:  signal[i] for i<0 becomes signal[-i]
#   pad_right: signal[T+i] for i>=0 becomes signal[T-2-i]  (note: not T-1-i)
def _reflect_index(i: Int, T: Int) -> Int:
    if i < 0:
        return -i
    if i >= T:
        return 2 * T - 2 - i
    return i


def stft_kernel[
    dtype: DType,
    XLayout: TensorLayout,
    WLayout: TensorLayout,
    RLayout: TensorLayout,
    ILayout: TensorLayout,
    N_FFT: Int,
    HOP: Int,
    N_FREQ: Int,    # n_fft // 2 + 1
](
    real_out: TileTensor[dtype, RLayout, MutAnyOrigin],   # (B, n_freq, n_frames)
    imag_out: TileTensor[dtype, ILayout, MutAnyOrigin],   # (B, n_freq, n_frames)
    x: TileTensor[dtype, XLayout, MutAnyOrigin],           # (B, T)
    window: TileTensor[dtype, WLayout, MutAnyOrigin],      # (N_FFT,)
    batch: Int,
    T: Int,
    n_frames: Int,
):
    """STFT with center=True (reflect pad by n_fft//2 on each side).

    real[b,k,t] = sum_{n=0..N-1} window[n] * x[b, t*hop - n_fft//2 + n] * cos(2π k n / N)
    imag[b,k,t] = -sum_{n=0..N-1} window[n] * x[b, t*hop - n_fft//2 + n] * sin(2π k n / N)

    Launch: grid = B * n_freq * n_frames, block_dim = 1.
    (We could parallelize n inside a block, but N_FFT is tiny.)
    """
    comptime assert x.flat_rank == 2
    comptime assert window.flat_rank == 1
    comptime assert real_out.flat_rank == 3
    comptime assert imag_out.flat_rank == 3

    var idx = block_idx.x
    var t = idx % n_frames
    var k = (idx // n_frames) % N_FREQ
    var b = idx // (n_frames * N_FREQ)

    var pad = N_FFT // 2
    var sample_start = t * HOP - pad
    var two_pi_k_over_n = 6.283185307179586 * Float32(k) / Float32(N_FFT)

    var re_acc: Float32 = 0.0
    var im_acc: Float32 = 0.0
    for n in range(N_FFT):
        var raw_i = sample_start + n
        var si = _reflect_index(raw_i, T)
        var s = rebind[Scalar[dtype]](x[b, si]).cast[DType.float32]()
        var w = rebind[Scalar[dtype]](window[n]).cast[DType.float32]()
        var sw = s * w
        var ang = two_pi_k_over_n * Float32(n)
        re_acc += sw * cos(ang)
        im_acc += -sw * sin(ang)

    real_out[b, k, t] = rebind[real_out.ElementType](re_acc.cast[dtype]())
    imag_out[b, k, t] = rebind[imag_out.ElementType](im_acc.cast[dtype]())


def istft_kernel[
    dtype: DType,
    RLayout: TensorLayout,
    ILayout: TensorLayout,
    WLayout: TensorLayout,
    XLayout: TensorLayout,
    N_FFT: Int,
    HOP: Int,
    N_FREQ: Int,
](
    output: TileTensor[dtype, XLayout, MutAnyOrigin],      # (B, T_OUT) output signal
    real_in: TileTensor[dtype, RLayout, MutAnyOrigin],    # (B, n_freq, n_frames)
    imag_in: TileTensor[dtype, ILayout, MutAnyOrigin],    # (B, n_freq, n_frames)
    window: TileTensor[dtype, WLayout, MutAnyOrigin],      # (N_FFT,)
    batch: Int,
    n_frames: Int,
    T_out: Int,
):
    """Inverse STFT (one thread per output sample).

    For each output sample t_out (in the un-centered signal of length T_out):
      t_padded = t_out + pad             (account for center=True padding)
      sum = 0
      norm = 0
      for each frame f covering t_padded:
        local_n = t_padded - f * HOP     (0 <= local_n < N_FFT)
        frame_sample = (1/N_FFT) * sum_{k=0..N_FFT-1} spec[f, k] * exp(2πi k n / N).real
                    only stored bins are k in [0, N_FREQ); reconstruct conjugates
                    for k in [N_FREQ, N_FFT) on the fly.
        sum  += window[local_n] * frame_sample
        norm += window[local_n]^2
      out[t_out] = sum / norm

    Match torch.istft's normalization (sum of squared windows). Length parameter
    drops the centered padding by trimming to T_out from the un-padded coordinate
    `pad` to `pad + T_out`.
    """
    comptime assert output.flat_rank == 2
    comptime assert real_in.flat_rank == 3
    comptime assert imag_in.flat_rank == 3
    comptime assert window.flat_rank == 1

    var idx = block_idx.x
    var t_out = idx % T_out
    var b = idx // T_out

    var pad = N_FFT // 2
    var t_padded = t_out + pad

    # First frame whose window may cover t_padded: f_min = ceil((t_padded - N + 1)/HOP)
    # Last frame: f_max = t_padded // HOP
    var f_min_raw = t_padded - N_FFT + 1
    var f_min: Int
    if f_min_raw <= 0:
        f_min = 0
    else:
        f_min = (f_min_raw + HOP - 1) // HOP
    var f_max = t_padded // HOP
    if f_max >= n_frames:
        f_max = n_frames - 1

    var sum: Float32 = 0.0
    var norm: Float32 = 0.0
    var inv_n = 1.0 / Float32(N_FFT)

    for f in range(f_min, f_max + 1):
        var local_n = t_padded - f * HOP
        if local_n < 0 or local_n >= N_FFT:
            continue
        # Reconstruct the n-th time-domain sample of frame f via inverse DFT,
        # filling in the conjugate bins [N_FREQ..N_FFT) on the fly.
        var two_pi_n_over_n = 6.283185307179586 * Float32(local_n) / Float32(N_FFT)
        var frame_sample: Float32 = 0.0
        # k = 0 (purely real bin)
        var r0 = rebind[Scalar[dtype]](real_in[b, 0, f]).cast[DType.float32]()
        frame_sample += r0
        # k = 1..N_FREQ-1
        for k in range(1, N_FREQ):
            var re_v = rebind[Scalar[dtype]](real_in[b, k, f]).cast[DType.float32]()
            var im_v = rebind[Scalar[dtype]](imag_in[b, k, f]).cast[DType.float32]()
            var ang = two_pi_n_over_n * Float32(k)
            var c = cos(ang)
            var s_ = sin(ang)
            # Forward bin k contributes:   re * c - im * s
            # Mirror bin (N - k) contributes the conjugate: same real, negated imag.
            # Mirror angle is 2π * (N - k) * n / N = 2π n - ang  →
            # cos(2π n - ang) = cos(ang), sin(2π n - ang) = -sin(ang).
            # So mirror contributes (re) * cos(ang) - (-im) * (-sin(ang))
            #                     = re * cos(ang) - im * sin(ang).
            # Combined: 2 * (re * c - im * s)
            # Special case: k == N_FFT/2 has no distinct conjugate (it equals itself
            # when N_FFT is even). Don't double-count.
            if k == N_FFT // 2:
                frame_sample += re_v * c - im_v * s_
            else:
                frame_sample += 2.0 * (re_v * c - im_v * s_)
        frame_sample *= inv_n

        var w = rebind[Scalar[dtype]](window[local_n]).cast[DType.float32]()
        sum += w * frame_sample
        norm += w * w

    var y: Float32 = 0.0
    if norm > 1.0e-11:
        y = sum / norm
    output[b, t_out] = rebind[output.ElementType](y.cast[dtype]())
