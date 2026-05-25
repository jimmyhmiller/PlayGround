"""VoiceEncoder mel extractor (16 kHz).

Matches chatterbox VoiceEncoder melspec.py:
  n_fft=400, hop=160, win=400, mel_power=2.0, num_mels=40, fmin=0, fmax=8000
  center=True (n_fft//2 reflect pad), no preemphasis (hp.preemphasis=0)
  mel_type="amp" (no log), normalized_mels=False (no min/max scale)

Mel filter bank uses librosa's slaney convention (htk=False, norm='slaney').
Bank is precomputed in Python and passed in.

Output is (B, T_frames, num_mels) — VoiceEncoder forward expects (B, T, M).
"""
from std.math import sin, cos, pi, sqrt
from std.gpu import block_idx, thread_idx
from layout import TileTensor, TensorLayout, row_major


def hann_window(n: Int) -> List[Float32]:
    """torch.hann_window(n, periodic=True) — divisor n, not n-1."""
    var out = List[Float32]()
    for i in range(n):
        var x: Float32 = 0.5 * (1.0 - cos((2.0 * Float32(pi) * Float32(i)) / Float32(n)))
        out.append(x)
    return out^


def stft_power_kernel[
    dtype: DType,
    InLayout: TensorLayout, WinLayout: TensorLayout, OutLayout: TensorLayout,
    N_FFT: Int, HOP: Int, N_BINS: Int, BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, N_BINS, T_frames)
    signal: TileTensor[dtype, InLayout, MutAnyOrigin],     # (B, L_padded)
    window: TileTensor[dtype, WinLayout, MutAnyOrigin],    # (N_FFT,)
    batch: Int, l_padded: Int, t_frames: Int,
):
    """librosa-style STFT |X[k]|^2 (mel_power=2). Output is re^2 + im^2."""
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
        var p: Float32 = re * re + im * im
        output[b, k, f] = rebind[output.ElementType](p.cast[dtype]())
        k += BLOCK


def mel_filter_amp_kernel[
    dtype: DType,
    InLayout: TensorLayout, BankLayout: TensorLayout, OutLayout: TensorLayout,
    N_BINS: Int, N_MEL: Int, BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, T_frames, N_MEL)
    spec: TileTensor[dtype, InLayout, MutAnyOrigin],       # (B, N_BINS, T_frames)
    bank: TileTensor[dtype, BankLayout, MutAnyOrigin],     # (N_MEL, N_BINS)
    batch: Int, t_frames: Int,
):
    """mel = bank @ spec. Output is transposed straight to (B, T, M) — the
    layout VoiceEncoder forward expects."""
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
        # Write transposed: (B, T_frames, N_MEL).
        output[b, f, m] = rebind[output.ElementType](acc.cast[dtype]())
        m += BLOCK
