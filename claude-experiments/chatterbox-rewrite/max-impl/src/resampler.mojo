"""Polyphase rational resampler with Kaiser-windowed sinc filter.

For 24kHz → 16kHz (ratio 2/3): up=2, down=3.
Algorithm:
  1. Design lowpass FIR: cutoff = 0.5 * min(up,down)/max(up,down) * 2*pi (relative to up*orig_fs).
  2. Resample formula: y[n] = sum_k h[k] * x[idx(n,k)] / up
     where idx(n,k) maps polyphase positions.

This produces 'reasonably accurate' resampling — not bit-exact to soxr_hq,
but suitable for speaker-encoder inputs where small phase/aliasing errors
don't materially change the 192-d xvector.
"""
from std.math import sin as msin, cos as mcos, pi, sqrt, log as mlog, exp as mexp
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList


def _bessel_i0(x: Float64) -> Float64:
    """Modified Bessel function of the first kind, order 0 (Kaiser window)."""
    var y = x / 3.75
    if x < 3.75:
        var t = y * y
        return 1.0 + t * (3.5156229 + t * (3.0899424 + t * (1.2067492 +
               t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))))
    var ax = x if x > 0.0 else -x
    var w = 1.0 / ax
    var s = 0.39894228 + w * (0.01328592 + w * (0.00225319 - w * 0.00157565 +
            w * (0.00916281 - w * (0.02057706 - w * (0.02635537 - w * (0.01647633 -
            w * 0.00392377))))))
    return mexp(ax) / sqrt(ax) * s


def design_kaiser_lowpass(
    n_taps: Int, cutoff: Float64, beta: Float64,
) -> List[Float32]:
    """Design symmetric Kaiser-windowed sinc lowpass filter, length=n_taps (odd).
    cutoff is normalized [0, 1] (1.0 = Nyquist of the *interpolated* signal).
    """
    var half = (n_taps - 1) // 2
    var i0_beta = _bessel_i0(beta)
    var taps = List[Float32](capacity=n_taps)
    for n in range(n_taps):
        var k = Float64(n - half)
        # sinc
        var sinc: Float64 = 0.0
        if k == 0.0:
            sinc = 1.0
        else:
            var arg = pi * cutoff * k
            sinc = msin(arg) / arg
        sinc = sinc * cutoff
        # Kaiser window
        var ratio = (k / Float64(half)) if half > 0 else 0.0
        var arg_in = 1.0 - ratio * ratio
        if arg_in < 0.0: arg_in = 0.0
        var w = _bessel_i0(beta * sqrt(arg_in)) / i0_beta
        taps.append(Float32(sinc * w))
    return taps^


def resample_polyphase(
    mut ctx: DeviceContext,
    mut x_buf: DeviceBuffer[DType.float32],     # (N_in,)
    mut out_buf: DeviceBuffer[DType.float32],   # (N_out,)
    n_in: Int, n_out: Int,
    up: Int, down: Int,
    filter_taps_buf: DeviceBuffer[DType.float32],
    n_taps: Int,
) raises:
    """Polyphase resample.

    y[n] = sum_k h[k] * x[(n*down + center_in*up - k) // up]   if (n*down + center_in*up - k) % up == 0
       (the input sample contributes only when index is integer)

    Equivalent vectorized form:
      For each output sample n, compute m = n * down
      For each input position p, contribute h[(p*up - m + half_filt)] * x[p]
      when (p*up - m + half_filt) in [0, n_taps).

    We implement: for each output n, the input p that contributes runs over the
    set {p : (p*up - m + half) >= 0, (p*up - m + half) < n_taps, ...}.
    """
    var xp = x_buf.unsafe_ptr()
    var op = out_buf.unsafe_ptr()
    var hp = filter_taps_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(xp, op, hp, n_in, up, down, n_taps)
    def res_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var n = idx[0]
        var half_filt = (n_taps - 1) // 2
        # Center of output sample n in input grid (at rate up*orig_fs):
        # interpolated_pos = n * down (in units of orig_fs samples ÷ up)
        # → input position (in units of orig_fs samples) = n * down / up
        # The filter is centered at interpolated_pos in the up-sampled grid.
        # For each input sample p, the filter index is (p * up - n * down) + half_filt.
        var center_int = n * down
        # Range of valid p: filter_idx in [0, n_taps).
        # 0 <= p*up - n*down + half <= n_taps - 1
        # → (n*down - half) <= p*up <= n*down - half + n_taps - 1
        # → p_min = ceil((n*down - half) / up), p_max = floor(...)
        var p_min_num = center_int - half_filt
        var p_max_num = center_int - half_filt + n_taps - 1
        var p_min: Int
        if p_min_num >= 0:
            p_min = (p_min_num + up - 1) // up
        else:
            p_min = -((-p_min_num) // up)
        var p_max = p_max_num // up
        if p_min < 0: p_min = 0
        if p_max >= n_in: p_max = n_in - 1

        var acc: Float32 = 0.0
        for p in range(p_min, p_max + 1):
            var f_idx = p * up - center_int + half_filt
            if f_idx >= 0 and f_idx < n_taps:
                acc = acc + hp[f_idx] * xp[p]
        # Scale by up (sinc filter has integral 1, but we zero-stuff by up so
        # we need to multiply by up to compensate the divide-by-up averaging).
        op[n] = acc * Float32(up)
    elementwise[res_fn, simd_width=1, target="gpu"](
        IndexList[1](n_out), DeviceContextPtr(ctx),
    )


def resample_24k_to_16k(
    mut ctx: DeviceContext,
    mut x_buf: DeviceBuffer[DType.float32],
    mut out_buf: DeviceBuffer[DType.float32],
    n_in: Int,
    n_out: Int,
) raises:
    """Convenience wrapper for 24k → 16k (up=2, down=3)."""
    var up = 2
    var down = 3
    # Filter design: cutoff = min(up,down)/max(up,down) = 2/3.
    # Typical n_taps = 2 * up * 10 + 1 = 41 (per-leg of 10 taps). For higher
    # quality, use longer; scipy default is 2*max(up,down)*10 + 1 = 61.
    var n_taps = 121  # ~10 taps per polyphase leg per output sample (high quality)
    var cutoff = 2.0 / 3.0   # normalized to (interpolated) Nyquist
    var beta = 8.6  # Kaiser beta — 8.6 gives ~80dB stopband
    var taps = design_kaiser_lowpass(n_taps, cutoff, beta)
    var taps_buf = ctx.enqueue_create_buffer[DType.float32](n_taps)
    with taps_buf.map_to_host() as h:
        for i in range(n_taps):
            h[i] = taps[i]

    resample_polyphase(ctx, x_buf, out_buf, n_in, n_out, up, down, taps_buf, n_taps)
