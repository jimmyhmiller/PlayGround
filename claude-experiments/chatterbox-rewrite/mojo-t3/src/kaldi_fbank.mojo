"""
Pure-Mojo Kaldi fbank feature extraction matching torchaudio.compliance.kaldi.fbank
with the defaults used by chatterbox.extract_feature():
  num_mel_bins = 80
  frame_length_ms = 25 -> window_size = 400 samples @ 16kHz
  frame_shift_ms  = 10 -> window_shift = 160 samples
  padded_window_size = 512 (next power of 2)
  low_freq = 20, high_freq = nyquist (8000)
  window_type = 'povey' (hann ** 0.85, periodic=False)
  preemphasis = 0.97 (with replicate-pad at j=0)
  remove_dc_offset = True
  use_log_fbank = True (with epsilon floor = 1.1920929e-7)
  subtract_mean = False  (we DO subtract mean post-hoc in extract_feature)
  snip_edges = True

Output shape: (T_frames, 80) in row-major; then we subtract per-frame mean to
match extract_feature.
"""
from std.gpu import block_idx, thread_idx
from std.gpu.sync import barrier
from std.gpu.memory import AddressSpace
from std.math import cos, sin, log, sqrt, pi
from layout import TileTensor, TensorLayout, row_major, stack_allocation


def frame_preprocess_kernel[
    dtype: DType,
    WLayout: TensorLayout,
    OutLayout: TensorLayout,
    WindowLayout: TensorLayout,
    WINDOW_SIZE: Int,
    PADDED: Int,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],     # (M, PADDED)
    waveform: TileTensor[dtype, WLayout, MutAnyOrigin],     # (N,) flat
    window: TileTensor[dtype, WindowLayout, MutAnyOrigin],  # (WINDOW_SIZE,) Povey window
    m: Int,            # number of frames
    window_shift: Int, # 160
    preemphasis: Float32,
):
    """Per-frame: DC removal, preemphasis (with replicate at j=0), Povey window,
    zero-pad to PADDED. Launch: grid = M, block_dim = BLOCK. Threads cooperate
    to compute the mean, then preprocess all WINDOW_SIZE samples in parallel.
    """
    comptime assert waveform.flat_rank == 1
    comptime assert window.flat_rank == 1
    comptime assert output.flat_rank == 2

    var frame_idx = block_idx.x
    var tid = thread_idx.x

    var start = frame_idx * window_shift

    # ---- Compute mean of WINDOW_SIZE samples via shared memory reduction.
    var smem = stack_allocation[DType.float32,
        address_space=AddressSpace.SHARED](row_major[BLOCK]())
    var s: Float32 = 0.0
    var j = tid
    while j < WINDOW_SIZE:
        s += rebind[Scalar[dtype]](waveform[start + j]).cast[DType.float32]()
        j += BLOCK
    smem[tid] = s
    barrier()
    if tid == 0:
        var total: Float32 = 0.0
        for i in range(BLOCK):
            total += rebind[Scalar[DType.float32]](smem[i])
        smem[0] = total / Float32(WINDOW_SIZE)
    barrier()
    var mean_val = rebind[Scalar[DType.float32]](smem[0])

    # Need the centered (DC-removed) frame values to apply preemphasis correctly.
    # Cache the centered frame in shared memory so preemphasis can look at
    # the prior sample. We need WINDOW_SIZE entries; if WINDOW_SIZE > BLOCK we
    # store strided. Use a separate shared buffer of size WINDOW_SIZE.
    var centered = stack_allocation[DType.float32,
        address_space=AddressSpace.SHARED](row_major[WINDOW_SIZE]())
    var j2 = tid
    while j2 < WINDOW_SIZE:
        var v = rebind[Scalar[dtype]](waveform[start + j2]).cast[DType.float32]()
        centered[j2] = v - mean_val
        j2 += BLOCK
    barrier()

    # ---- Preemphasis + window + zero-pad.
    var j3 = tid
    while j3 < PADDED:
        var y: Float32 = 0.0
        if j3 < WINDOW_SIZE:
            # preemphasis: y[j] = x[j] - 0.97 * x[max(0, j-1)]
            var prev_idx = j3 - 1
            if prev_idx < 0:
                prev_idx = 0
            var x_curr = rebind[Scalar[DType.float32]](centered[j3])
            var x_prev = rebind[Scalar[DType.float32]](centered[prev_idx])
            var pre = x_curr - preemphasis * x_prev
            var w = rebind[Scalar[dtype]](window[j3]).cast[DType.float32]()
            y = pre * w
        # Else: zero-pad.
        output[frame_idx, j3] = rebind[output.ElementType](y.cast[dtype]())
        j3 += BLOCK


def naive_rfft_power_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    OutLayout: TensorLayout,
    PADDED: Int,
    NUM_BINS: Int,   # PADDED // 2 + 1
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (M, NUM_BINS)  power spectrum
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],        # (M, PADDED)
    m: Int,
):
    """Naive real DFT magnitude squared.

    For each (frame, k) where k in 0..NUM_BINS-1:
      X[k] = sum_{n=0..PADDED-1} x[n] * exp(-2π i k n / PADDED)
      |X[k]|^2 = re^2 + im^2

    Launch: grid = M, block_dim = BLOCK. Each thread handles a stride of k values.
    """
    comptime assert inp.flat_rank == 2
    comptime assert output.flat_rank == 2

    var frame_idx = block_idx.x
    var tid = thread_idx.x

    var k = tid
    while k < NUM_BINS:
        var re: Float32 = 0.0
        var im: Float32 = 0.0
        var twoPiK_N: Float32 = (-2.0 * Float32(pi) * Float32(k)) / Float32(PADDED)
        for n in range(PADDED):
            var x = rebind[Scalar[dtype]](inp[frame_idx, n]).cast[DType.float32]()
            var phase = twoPiK_N * Float32(n)
            re += x * cos(phase)
            im += x * sin(phase)
        var p = re * re + im * im
        output[frame_idx, k] = rebind[output.ElementType](p.cast[dtype]())
        k += BLOCK


def mel_filterbank_log_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    BankLayout: TensorLayout,
    OutLayout: TensorLayout,
    NUM_BINS: Int,         # 257 (padded//2 + 1)
    NUM_MEL: Int,          # 80
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (M, NUM_MEL)
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],        # (M, NUM_BINS) power spectrum
    bank: TileTensor[dtype, BankLayout, MutAnyOrigin],     # (NUM_MEL, NUM_BINS)
    m: Int,
    eps: Float32,
):
    """For each (frame, mel_bin): sum_k inp[frame, k] * bank[mel_bin, k], then
    log(max(., eps)). Launch: grid = M, block_dim = BLOCK; threads stride NUM_MEL.
    """
    comptime assert inp.flat_rank == 2
    comptime assert bank.flat_rank == 2
    comptime assert output.flat_rank == 2

    var frame_idx = block_idx.x
    var tid = thread_idx.x

    var b = tid
    while b < NUM_MEL:
        var acc: Float32 = 0.0
        for k in range(NUM_BINS):
            var x = rebind[Scalar[dtype]](inp[frame_idx, k]).cast[DType.float32]()
            var w = rebind[Scalar[dtype]](bank[b, k]).cast[DType.float32]()
            acc += x * w
        var clamped = acc
        if clamped < eps:
            clamped = eps
        var y = log(clamped)
        output[frame_idx, b] = rebind[output.ElementType](y.cast[dtype]())
        b += BLOCK


def subtract_per_utterance_mean_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    OutLayout: TensorLayout,
    NUM_MEL: Int,
    BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (M, NUM_MEL)
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],        # (M, NUM_MEL)
    m: Int,
):
    """For each mel bin b: out[:, b] = inp[:, b] - mean(inp[:, b]).
    This matches extract_feature's `feature - feature.mean(dim=0, keepdim=True)`.

    Launch: grid = NUM_MEL, block_dim = BLOCK. Threads cooperate to compute
    column mean, then subtract.
    """
    comptime assert inp.flat_rank == 2
    comptime assert output.flat_rank == 2

    var b_idx = block_idx.x
    var tid = thread_idx.x

    var smem = stack_allocation[DType.float32,
        address_space=AddressSpace.SHARED](row_major[BLOCK]())
    var s: Float32 = 0.0
    var f = tid
    while f < m:
        s += rebind[Scalar[dtype]](inp[f, b_idx]).cast[DType.float32]()
        f += BLOCK
    smem[tid] = s
    barrier()
    if tid == 0:
        var total: Float32 = 0.0
        for i in range(BLOCK):
            total += rebind[Scalar[DType.float32]](smem[i])
        smem[0] = total / Float32(m)
    barrier()
    var mean_val = rebind[Scalar[DType.float32]](smem[0])

    var f2 = tid
    while f2 < m:
        var v = rebind[Scalar[dtype]](inp[f2, b_idx]).cast[DType.float32]()
        var y = v - mean_val
        output[f2, b_idx] = rebind[output.ElementType](y.cast[dtype]())
        f2 += BLOCK


# ===========================================================================
# Host-side helpers: precompute Povey window and mel-filterbank as fp32 arrays.
# ===========================================================================

def povey_window_fp32(window_size: Int) -> List[Float32]:
    """Hann window (periodic=False, divisor=window_size-1) raised to power 0.85.

    torch.hann_window(periodic=False) := 0.5 - 0.5 * cos(2π n / (N-1)) for n in 0..N-1.
    """
    var out = List[Float32]()
    var denom = Float32(window_size - 1)
    for n in range(window_size):
        var phase = (2.0 * Float32(pi) * Float32(n)) / denom
        var hann = 0.5 - 0.5 * cos(phase)
        # Povey: ^0.85.
        var p: Float32 = hann ** Float32(0.85)
        out.append(p)
    return out^


def mel_scale_scalar(freq: Float32) -> Float32:
    """Kaldi mel-scale: 1127 * ln(1 + freq/700)."""
    return Float32(1127.0) * log(Float32(1.0) + freq / Float32(700.0))


def mel_filterbank_fp32(
    num_mel: Int, padded: Int, sample_rate: Float32, low_freq: Float32, high_freq_arg: Float32,
) -> List[Float32]:
    """Build (num_mel * num_bins) row-major mel filterbank. Matches Kaldi get_mel_banks
    with the right-edge zero-pad in fbank() that makes it (num_mel, num_bins=num_fft_bins+1).
    """
    var num_fft_bins = padded // 2          # excludes Nyquist; we pad with 0 to padded//2+1
    var num_bins = padded // 2 + 1
    var nyquist = Float32(0.5) * sample_rate
    var high_freq = high_freq_arg
    if high_freq <= 0.0:
        high_freq = high_freq + nyquist
    var fft_bin_width = sample_rate / Float32(padded)
    var mel_low = mel_scale_scalar(low_freq)
    var mel_high = mel_scale_scalar(high_freq)
    var mel_delta = (mel_high - mel_low) / Float32(num_mel + 1)

    var out = List[Float32]()
    for _ in range(num_mel * num_bins):
        out.append(Float32(0.0))

    for b in range(num_mel):
        var left_mel = mel_low + Float32(b) * mel_delta
        var center_mel = mel_low + Float32(b + 1) * mel_delta
        var right_mel = mel_low + Float32(b + 2) * mel_delta
        for k in range(num_fft_bins):
            var freq = fft_bin_width * Float32(k)
            var mel = mel_scale_scalar(freq)
            var up = (mel - left_mel) / (center_mel - left_mel)
            var down = (right_mel - mel) / (right_mel - center_mel)
            var w = up
            if down < w:
                w = down
            if w < 0.0:
                w = 0.0
            out[b * num_bins + k] = w
        # Right-edge bin (k=num_fft_bins) stays 0 from initialization.
    return out^
