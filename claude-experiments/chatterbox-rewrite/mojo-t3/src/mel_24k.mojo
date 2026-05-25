"""
Pure-Mojo mel_spectrogram at 24kHz, matching chatterbox's mel_extractor
(used to produce prompt_feat from ref_wav_24k).

  n_fft=1920, num_mels=80, hop=480, win=1920, hann window, center=False,
  reflect-pad ((n_fft-hop)/2)=720 on both sides.
  magnitude = sqrt(re^2 + im^2 + 1e-9)
  output = log(clamp(mel_basis @ magnitude, min=1e-5))
"""
from std.math import sin, cos, pi, sqrt, log
from std.gpu import block_idx, thread_idx
from layout import TileTensor, TensorLayout, row_major


def hann_window_n1920() -> List[Float32]:
    """torch.hann_window(1920) — periodic=True (default)."""
    var n = 1920
    var out = List[Float32]()
    for i in range(n):
        # torch.hann_window default is `periodic=True`, divisor = n.
        var x: Float32 = 0.5 * (1.0 - cos((2.0 * Float32(pi) * Float32(i)) / Float32(n)))
        out.append(x)
    return out^


def reflect_pad_kernel[
    dtype: DType, InLayout: TensorLayout, OutLayout: TensorLayout, BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, L + 2*pad)
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],        # (B, L)
    batch: Int, l: Int, pad: Int,
):
    """Reflect-pad along the last dim: output[b, t] = inp[b, reflect(t - pad)]."""
    comptime assert inp.flat_rank == 2
    comptime assert output.flat_rank == 2
    var bid = block_idx.x
    var tid = thread_idx.x
    var b = bid
    var l_out = l + 2 * pad
    var t = tid
    while t < l_out:
        var src = t - pad
        if src < 0:
            src = -src
        elif src >= l:
            src = 2 * (l - 1) - src
        if src < 0: src = 0
        if src >= l: src = l - 1
        var v = rebind[Scalar[dtype]](inp[b, src]).cast[DType.float32]()
        output[b, t] = rebind[output.ElementType](v.cast[dtype]())
        t += BLOCK


def stft_24k_magnitude_kernel[
    dtype: DType,
    InLayout: TensorLayout, WinLayout: TensorLayout, OutLayout: TensorLayout,
    N_FFT: Int, HOP: Int, N_BINS: Int, BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, N_BINS, T_frames) — magnitude
    signal: TileTensor[dtype, InLayout, MutAnyOrigin],     # (B, L_padded)
    window: TileTensor[dtype, WinLayout, MutAnyOrigin],    # (N_FFT,)
    batch: Int, l_padded: Int, t_frames: Int,
):
    """STFT(center=False) → magnitude. center=False means frame f starts at f*HOP (no center shift).

    For each (b, f, k):
        x_w[n] = signal[b, f*HOP + n] * window[n]   for n in 0..N_FFT-1
        X[k] = sum_n x_w[n] * exp(-2πi k n / N_FFT)
        |X[k]| = sqrt(re² + im² + 1e-9)
    Launch: grid = B * t_frames, block_dim = BLOCK over N_BINS.
    """
    comptime assert signal.flat_rank == 2
    comptime assert window.flat_rank == 1
    comptime assert output.flat_rank == 3
    var bid = block_idx.x
    var tid = thread_idx.x
    var f = bid % t_frames
    var b = bid // t_frames
    var k = tid
    while k < N_BINS:
        var re: Float32 = 0.0
        var im: Float32 = 0.0
        var arg_base: Float32 = (-2.0 * Float32(pi) * Float32(k)) / Float32(N_FFT)
        for n in range(N_FFT):
            var src_idx: Int = f * HOP + n
            if src_idx >= l_padded:
                continue
            var x: Float32 = rebind[Scalar[dtype]](signal[b, src_idx]).cast[DType.float32]()
            var w: Float32 = rebind[Scalar[dtype]](window[n]).cast[DType.float32]()
            var xw: Float32 = x * w
            var phase: Float32 = arg_base * Float32(n)
            re += xw * cos(phase)
            im += xw * sin(phase)
        var mag: Float32 = sqrt(re * re + im * im + 1.0e-9)
        output[b, k, f] = rebind[output.ElementType](mag.cast[dtype]())
        k += BLOCK


def mel_filter_log_kernel[
    dtype: DType,
    InLayout: TensorLayout, BankLayout: TensorLayout, OutLayout: TensorLayout,
    N_BINS: Int, N_MEL: Int, BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, N_MEL, T_frames) — log-mel
    spec: TileTensor[dtype, InLayout, MutAnyOrigin],       # (B, N_BINS, T_frames) — magnitude
    bank: TileTensor[dtype, BankLayout, MutAnyOrigin],     # (N_MEL, N_BINS)
    batch: Int, t_frames: Int, min_clamp: Float32,
):
    """mel = bank @ spec → log(clamp(mel, min=min_clamp)).
    Launch: grid = B*t_frames, block_dim = BLOCK over N_MEL.
    """
    comptime assert spec.flat_rank == 3
    comptime assert bank.flat_rank == 2
    comptime assert output.flat_rank == 3
    var bid = block_idx.x
    var tid = thread_idx.x
    var f = bid % t_frames
    var b = bid // t_frames
    var m = tid
    while m < N_MEL:
        var acc: Float32 = 0.0
        for k in range(N_BINS):
            var x = rebind[Scalar[dtype]](spec[b, k, f]).cast[DType.float32]()
            var w = rebind[Scalar[dtype]](bank[m, k]).cast[DType.float32]()
            acc += w * x
        if acc < min_clamp:
            acc = min_clamp
        var y = log(acc)
        output[b, m, f] = rebind[output.ElementType](y.cast[dtype]())
        m += BLOCK


def transpose_bct_to_btc_2d_kernel[
    dtype: DType, InLayout: TensorLayout, OutLayout: TensorLayout, BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, T, C)
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],        # (B, C, T)
    batch: Int, c: Int, t: Int,
):
    """Transpose (B, C, T) → (B, T, C). Specialized for the prompt_feat output."""
    comptime assert inp.flat_rank == 3
    comptime assert output.flat_rank == 3
    var bid = block_idx.x
    var tid = thread_idx.x
    var tt = bid % t
    var b = bid // t
    var cc = tid
    while cc < c:
        var v = rebind[Scalar[dtype]](inp[b, cc, tt]).cast[DType.float32]()
        output[b, tt, cc] = rebind[output.ElementType](v.cast[dtype]())
        cc += BLOCK
