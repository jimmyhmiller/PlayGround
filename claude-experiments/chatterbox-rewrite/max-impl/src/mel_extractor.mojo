"""Mel-spectrogram extractor built from MAX `elementwise` + `linalg.matmul`.

Two configurations are supported:

  16k VE-style (used by Chatterbox VoiceEncoder):
    n_fft=400, hop=160, win=400, mel_power=2.0, n_mels=40,
    fmin=0, fmax=8000, center=True (reflect pad).
    No log, no normalize — raw amp^2 mel.

  16k s3tokenizer-style:
    n_fft=400, hop=160, win=400, mel_power=2.0, n_mels=128,
    center=False (handled by caller's padding).
    log10(clamp(mel, 1e-10)); max-shift; (log_mel + 4)/4.

All ops dispatched via `elementwise[..., target="gpu"]` / `linalg.matmul`.
Mel filter bank is built on the host (librosa-compatible) and passed in
as a DeviceBuffer.
"""
from std.math import sin, cos, pi, sqrt, log, exp
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList
from layout import Idx, TileTensor, row_major

from linalg.matmul import matmul as nn_matmul


def reflect_pad_1d(
    mut ctx: DeviceContext,
    mut x_buf: DeviceBuffer[DType.float32],     # (B, L)
    mut out_buf: DeviceBuffer[DType.float32],   # (B, L + 2*pad)
    batch: Int, l: Int, pad: Int,
) raises:
    """Reflect-pad along the last dim. Used to emulate librosa's center=True STFT."""
    var in_ptr = x_buf.unsafe_ptr()
    var out_ptr = out_buf.unsafe_ptr()
    var l_out = l + 2 * pad
    var total = batch * l_out
    var dctx = DeviceContextPtr(ctx)

    @always_inline
    @parameter
    @__copy_capture(in_ptr, out_ptr, l, l_out, pad)
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var b = i // l_out
        var t = i - b * l_out
        var src = t - pad
        if src < 0: src = -src
        elif src >= l: src = 2 * (l - 1) - src
        if src < 0: src = 0
        if src >= l: src = l - 1
        out_ptr[i] = in_ptr[b * l + src]

    elementwise[func, simd_width=1, target="gpu"](
        IndexList[1](total), dctx,
    )


def stft_magnitude_power(
    mut ctx: DeviceContext,
    mut signal_buf: DeviceBuffer[DType.float32],   # (B, L_padded)
    mut window_buf: DeviceBuffer[DType.float32],   # (N_FFT,)
    mut out_buf: DeviceBuffer[DType.float32],      # (B, N_BINS, T_frames)
    batch: Int, l_padded: Int, n_fft: Int, hop: Int,
    n_bins: Int, t_frames: Int,
    mel_power: Int,    # 1 = magnitude, 2 = power (re^2 + im^2)
) raises:
    """Compute STFT and produce magnitude^mel_power. Pure-elementwise:
    one capturing closure that does the naive DFT for each output (b, k, f).
    """
    var s_ptr = signal_buf.unsafe_ptr()
    var w_ptr = window_buf.unsafe_ptr()
    var o_ptr = out_buf.unsafe_ptr()
    var total = batch * n_bins * t_frames
    var dctx = DeviceContextPtr(ctx)

    @always_inline
    @parameter
    @__copy_capture(s_ptr, w_ptr, o_ptr, batch, l_padded, n_fft, hop, n_bins, t_frames, mel_power)
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var b = i // (n_bins * t_frames)
        var rem = i - b * n_bins * t_frames
        var k = rem // t_frames
        var f = rem - k * t_frames
        var re: Float32 = 0.0
        var im: Float32 = 0.0
        var arg_base: Float32 = (-2.0 * Float32(pi) * Float32(k)) / Float32(n_fft)
        for n in range(n_fft):
            var sidx = f * hop + n
            if sidx < l_padded:
                var x = s_ptr[b * l_padded + sidx]
                var w = w_ptr[n]
                var xw = x * w
                var phase = arg_base * Float32(n)
                re += xw * cos(phase)
                im += xw * sin(phase)
        var p: Float32 = re * re + im * im
        if mel_power == 1:
            p = sqrt(p)
        # mel_power == 2 keeps power.
        o_ptr[i] = p

    elementwise[func, simd_width=1, target="gpu"](
        IndexList[1](total), dctx,
    )


def mel_filter_apply(
    mut ctx: DeviceContext,
    mut spec_buf: DeviceBuffer[DType.float32],   # (B, N_BINS, T_frames)
    mut bank_buf: DeviceBuffer[DType.float32],   # (N_MEL, N_BINS)
    mut out_buf:  DeviceBuffer[DType.float32],   # (B, T_frames, N_MEL) — transposed to (B, T, M)
    batch: Int, n_bins: Int, n_mels: Int, t_frames: Int,
) raises:
    """Apply mel filter: mel[b, t, m] = sum_k bank[m, k] * spec[b, k, t].

    We dispatch one elementwise pass that computes each output element. Output
    is laid out (B, T, N_MEL) directly — convenient for downstream VoiceEncoder
    which expects (B, T, M).
    """
    var s_ptr = spec_buf.unsafe_ptr()
    var b_ptr = bank_buf.unsafe_ptr()
    var o_ptr = out_buf.unsafe_ptr()
    var total = batch * t_frames * n_mels
    var dctx = DeviceContextPtr(ctx)

    @always_inline
    @parameter
    @__copy_capture(s_ptr, b_ptr, o_ptr, batch, n_bins, n_mels, t_frames)
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var b = i // (t_frames * n_mels)
        var rem = i - b * t_frames * n_mels
        var t = rem // n_mels
        var m = rem - t * n_mels
        var acc: Float32 = 0.0
        for k in range(n_bins):
            acc += b_ptr[m * n_bins + k] * s_ptr[b * n_bins * t_frames + k * t_frames + t]
        o_ptr[i] = acc

    elementwise[func, simd_width=1, target="gpu"](
        IndexList[1](total), dctx,
    )


def log_mel_normalize(
    mut ctx: DeviceContext,
    mut x_buf: DeviceBuffer[DType.float32],   # (B, T, M)
    mut out_buf: DeviceBuffer[DType.float32],
    batch: Int, t_frames: Int, n_mels: Int,
) raises:
    """s3tokenizer-style log-mel normalisation:
       y = log10(max(x, 1e-10))
       y = max(y, y.max() - 8)
       y = (y + 4) / 4
    We do this in two passes since (y.max()) is a reduction.
    """
    # Phase 1: log10 + clamp into out_buf.
    var x_ptr = x_buf.unsafe_ptr()
    var o_ptr = out_buf.unsafe_ptr()
    var n = batch * t_frames * n_mels
    var dctx = DeviceContextPtr(ctx)

    @always_inline
    @parameter
    @__copy_capture(x_ptr, o_ptr)
    def log_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var v = x_ptr[i]
        if v < 1.0e-10: v = 1.0e-10
        o_ptr[i] = log(v) * 0.43429448190325176  # log10(x) = ln(x) / ln(10)
    elementwise[log_func, simd_width=1, target="gpu"](
        IndexList[1](n), dctx,
    )
    # Phase 2: compute global max per-batch.
    # (For s3tokenizer batch=1; we treat it as a single max across all elements.)
    var max_buf = ctx.enqueue_create_buffer[DType.float32](batch)
    max_buf.enqueue_fill(-1.0e30)
    var max_ptr = max_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(o_ptr, max_ptr, t_frames, n_mels)
    def max_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var bi = idx[0]
        var best: Float32 = -1.0e30
        var off = bi * t_frames * n_mels
        for i in range(t_frames * n_mels):
            var v = o_ptr[off + i]
            if v > best: best = v
        max_ptr[bi] = best
    elementwise[max_func, simd_width=1, target="gpu"](
        IndexList[1](batch), dctx,
    )

    # Phase 3: shift and rescale.
    @always_inline
    @parameter
    @__copy_capture(o_ptr, max_ptr, t_frames, n_mels)
    def norm_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (t_frames * n_mels)
        var v = o_ptr[i]
        var floor_v = max_ptr[bi] - 8.0
        if v < floor_v: v = floor_v
        o_ptr[i] = (v + 4.0) / 4.0
    elementwise[norm_func, simd_width=1, target="gpu"](
        IndexList[1](n), dctx,
    )
