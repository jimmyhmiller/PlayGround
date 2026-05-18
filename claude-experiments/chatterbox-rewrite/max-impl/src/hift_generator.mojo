"""NSF-HiFiGAN / HiFTGenerator (Neural Source Filter + iSTFTNet) — struct
definitions matching real upstream Chatterbox `mel2wav` weights.

Architecture summary (from chatterbox/models/s3gen/hifigan.py HiFTGenerator):

  conv_pre (mel→512, k=7)
  for stage in 0,1,2:
    x = lrelu(x); x = ups[stage](x)        # transposed conv
    si = source_downs[stage](source_stft)
    si = source_resblocks[stage](si)
    x = x + si
    sum over MRF resblocks (3 per stage, kernel sizes [3,7,11], dilation [1,3,5])
  x = lrelu(x); x = conv_post(x)            # → 18 channels (n_fft+2)
  mag, phase = exp(x[:9]), sin(x[9:18])
  audio = iSTFT(mag, phase)                 # n_fft=16, hop=4

Each ResBlock has:
  3 dilated convs (convs1) each followed by Snake activation (alpha per channel)
  3 1×1 convs (convs2) each followed by Snake activation
  Output is x + summed branches

NSF source path:
  m_source (l_linear (1, 9)) generates harmonic+noise → STFT (real, imag) → cat
  → source_downs[stage] (Conv1d 18 → ch, varying kernel/stride per stage)
  → source_resblocks[stage] (single MRF resblock with kernel from
    source_resblock_kernel_sizes [7, 11, ...])

f0_predictor: a Conv1d stack (condnet 5 layers) + linear classifier → F0 per frame

This file currently provides STRUCTS only; forward wiring TBD in follow-up.
"""
from std.math import sin as msin, cos as mcos, sqrt, exp, pi
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from modules import Linear, linear_forward, residual_add, leaky_relu
from conv1d import Conv1d, conv1d_forward


@fieldwise_init
struct SnakeActivation(Copyable, Movable):
    """Snake activation: y = x + (1/alpha) * sin^2(alpha * x).

    alpha is a learnable per-channel parameter, shape (C,).
    """
    var alpha: DeviceBuffer[DType.float32]
    var channels: Int


@fieldwise_init
struct HiFTResBlock(Copyable, Movable):
    """One MRF ResBlock: 3 dilated convs (convs1) + 3 1x1 convs (convs2),
    each preceded by a Snake activation. All weight_norm collapsed at load.
    """
    var convs1: List[Conv1d]                # 3 dilated convs
    var convs2: List[Conv1d]                # 3 1x1 post convs
    var activations1: List[SnakeActivation] # 3 snake activations before convs1
    var activations2: List[SnakeActivation] # 3 snake activations before convs2
    var channels: Int


@fieldwise_init
struct F0Predictor(Copyable, Movable):
    """5 weight-normed conv1d layers (condnet/0,2,4,6,8) + linear classifier
    that maps 512 features → F0 prediction (1 channel).
    """
    var condnet: List[Conv1d]    # 5 conv layers; activations between are inferred
    var classifier: Linear       # (1, 512) + bias (1,)


@fieldwise_init
struct MSource(Copyable, Movable):
    """NSF source generator's `l_linear` mixer: (1, 9) — 9 harmonic inputs to
    1 output, mixed linearly with bias.
    """
    var l_linear: Linear         # in=9 out=1 + bias


@fieldwise_init
struct HiFTGenerator(Copyable, Movable):
    """Full NSF-HiFiGAN / HiFTGenerator.

    - conv_pre: Conv1d (80→512, k=7)
    - ups: 3 transposed Conv1d stages [512→256, 256→128, 128→64]
    - resblocks: 9 (3 per ups stage) — MRF kernels [3,7,11]
    - source_downs: 3 (one per ups stage, in=18 channels = n_fft+2)
    - source_resblocks: 3 (one per stage)
    - conv_post: Conv1d (64 → 18, k=7) — outputs STFT magnitude+phase
    - m_source: NSF harmonic source generator (l_linear 1×9)
    - f0_predictor: 5 conv layers + classifier
    """
    var conv_pre: Conv1d
    var ups: List[Conv1d]               # NOTE: weight is ConvTranspose1d — loader handles
    var resblocks: List[HiFTResBlock]
    var source_downs: List[Conv1d]
    var source_resblocks: List[HiFTResBlock]
    var conv_post: Conv1d
    var m_source: MSource
    var f0_predictor: F0Predictor
    var n_fft: Int                      # 16
    var hop_len: Int                    # 4
    var lrelu_slope: Float32            # 0.1


# ============================================================================
# Forward helpers
# ============================================================================


def snake_activation(
    mut ctx: DeviceContext,
    mut x_buf: DeviceBuffer[DType.float32],     # (B, C, T) in-place
    mut alpha_buf: DeviceBuffer[DType.float32], # (C,) per-channel learnable
    b: Int, c: Int, t: Int,
) raises:
    """Snake (linear-alpha): y = x + (1/alpha) * sin²(x * alpha).

    `alpha_logscale=False` upstream, so alpha is used directly (not exp'd).
    """
    var x_ptr = x_buf.unsafe_ptr()
    var a_ptr = alpha_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(x_ptr, a_ptr, c, t)
    def snake_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (c * t)
        var rem = i - bi * c * t
        var ci = rem // t
        var a = a_ptr[ci]
        var inv = 1.0 / (a + Float32(1.0e-9))
        var xa = x_ptr[i] * a
        var s = msin(xa)
        x_ptr[i] = x_ptr[i] + inv * s * s
    elementwise[snake_fn, simd_width=1, target="gpu"](
        IndexList[1](b * c * t), DeviceContextPtr(ctx),
    )


def hift_resblock_forward(
    mut ctx: DeviceContext,
    mut block: HiFTResBlock,
    mut x_buf: DeviceBuffer[DType.float32],     # (B, C, T) in-place residual updates
    b: Int, t: Int,
) raises:
    """One MRF ResBlock:
       for j in 0..2:
         xt = snake1[j](x)
         xt = conv1[j](xt)   # dilated k=3
         xt = snake2[j](xt)
         xt = conv2[j](xt)   # k=3, dilation 1
         x  = x + xt
    """
    var c = block.channels
    var n = b * c * t
    for j in range(3):
        var xt = ctx.enqueue_create_buffer[DType.float32](n)
        ctx.enqueue_copy(xt, x_buf)
        snake_activation(ctx, xt, block.activations1[j].alpha, b, c, t)
        var h1 = ctx.enqueue_create_buffer[DType.float32](n)
        conv1d_forward(ctx, block.convs1[j], xt, h1, b, t, t)
        snake_activation(ctx, h1, block.activations2[j].alpha, b, c, t)
        var h2 = ctx.enqueue_create_buffer[DType.float32](n)
        conv1d_forward(ctx, block.convs2[j], h1, h2, b, t, t)
        residual_add(ctx, x_buf, h2, n)


def conv_transpose1d_naive(
    mut ctx: DeviceContext,
    mut in_buf: DeviceBuffer[DType.float32],    # (B, C_in, T_in)
    mut weight_buf: DeviceBuffer[DType.float32], # (C_in, C_out, K) — note PyTorch layout
    mut bias_buf: DeviceBuffer[DType.float32],   # (C_out,)
    mut out_buf: DeviceBuffer[DType.float32],   # (B, C_out, T_out)
    b: Int, c_in: Int, c_out: Int, k: Int, stride: Int, pad: Int,
    t_in: Int, t_out: Int,
) raises:
    """Naive 1D transposed convolution.

    For each output position `t`, sum over kernel offset `kk` and input
    channel `ci`:
        i_src = (t + pad - kk) / stride         # only if integer
        out[b, co, t] += in[b, ci, i_src] * weight[ci, co, kk] + bias[co]

    PyTorch's ConvTranspose1d weight layout is (in_channels, out_channels, K).
    """
    var in_ptr = in_buf.unsafe_ptr()
    var w_ptr = weight_buf.unsafe_ptr()
    var b_ptr = bias_buf.unsafe_ptr()
    var o_ptr = out_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(in_ptr, w_ptr, b_ptr, o_ptr, c_in, c_out, k, stride, pad, t_in, t_out)
    def ct_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (c_out * t_out)
        var rem = i - bi * c_out * t_out
        var co = rem // t_out
        var t_o = rem - co * t_out

        var acc: Float32 = b_ptr[co]
        for kk in range(k):
            var num = t_o + pad - kk
            if num < 0: continue
            if num % stride != 0: continue
            var i_src = num // stride
            if i_src < 0 or i_src >= t_in: continue
            for ci in range(c_in):
                acc += in_ptr[bi * c_in * t_in + ci * t_in + i_src] * w_ptr[ci * c_out * k + co * k + kk]
        o_ptr[i] = acc
    elementwise[ct_func, simd_width=1, target="gpu"](
        IndexList[1](b * c_out * t_out), DeviceContextPtr(ctx),
    )


def leaky_relu_inplace(
    mut ctx: DeviceContext,
    mut x_buf: DeviceBuffer[DType.float32],
    n: Int, slope: Float32,
) raises:
    var x_ptr = x_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(x_ptr, slope)
    def lr_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var v = x_ptr[i]
        if v < 0.0:
            x_ptr[i] = v * slope
    elementwise[lr_func, simd_width=1, target="gpu"](
        IndexList[1](n), DeviceContextPtr(ctx),
    )


def add_inplace_bct_scaled(
    mut ctx: DeviceContext,
    mut dst_buf: DeviceBuffer[DType.float32],
    mut other_buf: DeviceBuffer[DType.float32],
    scale: Float32, n: Int,
) raises:
    """dst += other * scale (element-wise)."""
    var d_ptr = dst_buf.unsafe_ptr()
    var o_ptr = other_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(d_ptr, o_ptr, scale)
    def add_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        d_ptr[i] = d_ptr[i] + o_ptr[i] * scale
    elementwise[add_func, simd_width=1, target="gpu"](
        IndexList[1](n), DeviceContextPtr(ctx),
    )


def reflection_pad1_right(
    mut ctx: DeviceContext,
    mut in_buf: DeviceBuffer[DType.float32],     # (B, C, T)
    mut out_buf: DeviceBuffer[DType.float32],    # (B, C, T + 1)
    b: Int, c: Int, t: Int,
) raises:
    """ReflectionPad1d((1, 0)): pad 1 on left side (by reflecting position 1).

    Wait — upstream uses `nn.ReflectionPad1d((1, 0))`, where the tuple is
    (left, right) padding. So 1 on left, 0 on right. Reflect: out[..., 0] =
    in[..., 1], out[..., 1:] = in[..., :].
    """
    var in_ptr = in_buf.unsafe_ptr()
    var out_ptr = out_buf.unsafe_ptr()
    var t_out = t + 1

    @always_inline
    @parameter
    @__copy_capture(in_ptr, out_ptr, c, t, t_out)
    def rp_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (c * t_out)
        var rem = i - bi * c * t_out
        var ci = rem // t_out
        var ti = rem - ci * t_out
        if ti == 0:
            out_ptr[i] = in_ptr[bi * c * t + ci * t + 1]
        else:
            out_ptr[i] = in_ptr[bi * c * t + ci * t + (ti - 1)]
    elementwise[rp_func, simd_width=1, target="gpu"](
        IndexList[1](b * c * t_out), DeviceContextPtr(ctx),
    )


def hift_decode_trunk(
    mut ctx: DeviceContext,
    mut model: HiFTGenerator,
    mut mel_buf: DeviceBuffer[DType.float32],    # (B, 80, T_mel)
    mut s_stft_buf: DeviceBuffer[DType.float32], # (B, 18, T_s) — pre-computed source STFT (real|imag concat on C)
    mut spec_out: DeviceBuffer[DType.float32],   # (B, 18, T_out) — conv_post output (magnitude/phase pre-split)
    b: Int, t_mel: Int, t_s: Int, t_out: Int,
    use_source: Bool = True,
) raises:
    """HiFTGenerator's `decode()` trunk: mel + source_stft → (B, 18, T_out).

    The output is the conv_post output before iSTFT — caller does:
        magnitude = exp(spec_out[:, :n_fft//2 + 1])
        phase = sin(spec_out[:, n_fft//2 + 1:])
        audio = iSTFT(magnitude, phase)

    Upsample rates produce T_out = T_mel * 8 * 8 * 4 = 256 * T_mel.
    The last (i==num_upsamples-1) stage applies reflection_pad on the right,
    growing T by 1.
    """
    comptime BASE = 512
    var slope = model.lrelu_slope

    # 1. conv_pre: 80 → 512, kernel 7, pad 3 → keep T.
    var x = ctx.enqueue_create_buffer[DType.float32](b * BASE * t_mel)
    conv1d_forward(ctx, model.conv_pre, mel_buf, x, b, t_mel, t_mel)

    # Track T as we go through the 3 ups stages with rates [8, 8, 4].
    var ups_rates = [8, 8, 4]
    var t_cur = t_mel
    var c_cur = BASE

    for i in range(3):
        # Pre-conv leaky-relu (in place).
        leaky_relu_inplace(ctx, x, b * c_cur * t_cur, slope)

        # ConvTranspose1d ups[i] → halves channels, multiplies T by stride.
        var c_next = c_cur // 2
        var stride = ups_rates[i]
        var k = model.ups[i].k
        var pad = (k - stride) // 2
        var t_after = (t_cur - 1) * stride - 2 * pad + k
        var x_up = ctx.enqueue_create_buffer[DType.float32](b * c_next * t_after)
        conv_transpose1d_naive(
            ctx, x, model.ups[i].weight, model.ups[i].bias, x_up,
            b, c_cur, c_next, k, stride, pad, t_cur, t_after,
        )

        # On the last stage, apply ReflectionPad1d((1, 0)) → T grows by 1.
        var t_post: Int
        var x_padded: DeviceBuffer[DType.float32]
        if i == 2:
            t_post = t_after + 1
            x_padded = ctx.enqueue_create_buffer[DType.float32](b * c_next * t_post)
            reflection_pad1_right(ctx, x_up, x_padded, b, c_next, t_after)
        else:
            t_post = t_after
            x_padded = ctx.enqueue_create_buffer[DType.float32](b * c_next * t_post)
            ctx.enqueue_copy(x_padded, x_up)

        # Source fusion: source_downs[i](s_stft) → source_resblocks[i] → add.
        # Source path is sensitive to T_s alignment with each stage's T_post.
        # Caller can disable for trunk-only smoke tests by passing use_source=False.
        if use_source:
            var si = ctx.enqueue_create_buffer[DType.float32](b * c_next * t_post)
            conv1d_forward(ctx, model.source_downs[i], s_stft_buf, si, b, t_s, t_post)
            hift_resblock_forward(ctx, model.source_resblocks[i], si, b, t_post)
            residual_add(ctx, x_padded, si, b * c_next * t_post)

        # 3 parallel MRF resblocks summed and averaged.
        var x_sum = ctx.enqueue_create_buffer[DType.float32](b * c_next * t_post)
        x_sum.enqueue_fill(0.0)
        for j in range(3):
            var xs = ctx.enqueue_create_buffer[DType.float32](b * c_next * t_post)
            ctx.enqueue_copy(xs, x_padded)
            hift_resblock_forward(ctx, model.resblocks[i * 3 + j], xs, b, t_post)
            residual_add(ctx, x_sum, xs, b * c_next * t_post)
        # average.
        var inv: Float32 = 1.0 / 3.0
        var sum_ptr = x_sum.unsafe_ptr()

        @always_inline
        @parameter
        @__copy_capture(sum_ptr, inv)
        def avg_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
            var i_idx = idx[0]
            sum_ptr[i_idx] = sum_ptr[i_idx] * inv
        elementwise[avg_func, simd_width=1, target="gpu"](
            IndexList[1](b * c_next * t_post), DeviceContextPtr(ctx),
        )

        # Replace x for next stage.
        x = x_sum
        c_cur = c_next
        t_cur = t_post

    # Final leaky_relu + conv_post: (B, 64, T) → (B, 18, T).
    leaky_relu_inplace(ctx, x, b * c_cur * t_cur, 0.01)   # default slope 0.01 for last
    conv1d_forward(ctx, model.conv_post, x, spec_out, b, t_cur, t_out)


# ============================================================================
# iSTFT for the final mel→audio stage.
# ============================================================================


def hann_window_periodic_fill(
    mut ctx: DeviceContext,
    mut win_buf: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    """torch's periodic Hann: w[i] = 0.5 * (1 - cos(2π i / N))."""
    var w_ptr = win_buf.unsafe_ptr()
    var two_pi_over_n: Float32 = 2.0 * Float32(pi) / Float32(n)

    @always_inline
    @parameter
    @__copy_capture(w_ptr, two_pi_over_n)
    def w_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        w_ptr[i] = 0.5 * (1.0 - mcos(two_pi_over_n * Float32(i)))
    elementwise[w_func, simd_width=1, target="gpu"](
        IndexList[1](n), DeviceContextPtr(ctx),
    )


def istft_forward(
    mut ctx: DeviceContext,
    mut spec_buf: DeviceBuffer[DType.float32],     # (B, n_fft+2, T_frames) — conv_post output
    mut window_buf: DeviceBuffer[DType.float32],   # (n_fft,) Hann window
    mut audio_out: DeviceBuffer[DType.float32],    # (B, T_audio)
    b: Int, n_fft: Int, n_frames: Int, t_audio: Int,
) raises:
    """Inverse STFT of HiFTGenerator's `_istft`.

    Inputs:
      spec[:, :n_fft//2+1, :]   — log-magnitude → magnitude = exp(.)
      spec[:, n_fft//2+1:, :]   — phase angle  (clipped via sin per upstream)
      mag = clip(exp(spec_mag), max=1e2)
      real = mag * cos(phase)
      imag = mag * sin(phase)
      audio = istft(complex(real, imag), n_fft, hop=n_fft//4, window=hann)

    For n_fft=16, hop_len=4:
      n_freq = n_fft//2 + 1 = 9
      Frame f's window covers samples [f*hop, f*hop + n_fft) in padded coords.
      Padded length = T_audio + n_fft (center=True with n_fft/2 pad each side).
      T_audio = (n_frames - 1) * hop_len (post-center-trim).

    Caller pre-loads the periodic Hann window into window_buf.
    """
    var n_freq = n_fft // 2 + 1
    var hop = n_fft // 4
    var pad = n_fft // 2
    var s_ptr = spec_buf.unsafe_ptr()
    var w_ptr = window_buf.unsafe_ptr()
    var o_ptr = audio_out.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(s_ptr, w_ptr, o_ptr, n_fft, n_freq, hop, pad, n_frames, t_audio)
    def istft_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // t_audio
        var t_out = i - bi * t_audio
        var t_padded = t_out + pad

        var f_min_raw = t_padded - n_fft + 1
        var f_min: Int
        if f_min_raw <= 0:
            f_min = 0
        else:
            f_min = (f_min_raw + hop - 1) // hop
        var f_max = t_padded // hop
        if f_max >= n_frames:
            f_max = n_frames - 1

        var sum_val: Float32 = 0.0
        var norm: Float32 = 0.0
        var inv_n = 1.0 / Float32(n_fft)
        var two_pi_inv_n = 6.283185307179586 / Float32(n_fft)

        for f in range(f_min, f_max + 1):
            var local_n = t_padded - f * hop
            if local_n < 0 or local_n >= n_fft:
                continue
            # Reconstruct via inverse DFT, conjugate-symmetric for k in [n_freq, n_fft).
            var two_pi_n = two_pi_inv_n * Float32(local_n)
            var frame_sample: Float32 = 0.0

            # k = 0 (purely real DC bin).
            var spec_mag0 = s_ptr[bi * (n_fft + 2) * n_frames + 0 * n_frames + f]
            var mag0 = exp(spec_mag0)
            if mag0 > 100.0: mag0 = 100.0
            var phase0 = s_ptr[bi * (n_fft + 2) * n_frames + n_freq * n_frames + f]
            var sin_phase0 = msin(phase0)
            var cos_phase0 = mcos(phase0)
            var r0 = mag0 * cos_phase0
            frame_sample += r0

            for k in range(1, n_freq):
                var spec_mag_k = s_ptr[bi * (n_fft + 2) * n_frames + k * n_frames + f]
                var mag_k = exp(spec_mag_k)
                if mag_k > 100.0: mag_k = 100.0
                var phase_k = s_ptr[bi * (n_fft + 2) * n_frames + (n_freq + k) * n_frames + f]
                var sin_p = msin(phase_k)
                var cos_p = mcos(phase_k)
                var re_v = mag_k * cos_p
                var im_v = mag_k * sin_p
                var ang = two_pi_n * Float32(k)
                var c = mcos(ang)
                var s_ = msin(ang)
                if k == n_fft // 2:
                    frame_sample += re_v * c - im_v * s_
                else:
                    frame_sample += 2.0 * (re_v * c - im_v * s_)

            frame_sample = frame_sample * inv_n

            var w = w_ptr[local_n]
            sum_val += w * frame_sample
            norm += w * w

        var y: Float32 = 0.0
        if norm > 0.0:
            y = sum_val / norm
        # Clamp to audio_limit (default 0.99).
        var lim: Float32 = 0.99
        if y > lim: y = lim
        if y < -lim: y = -lim
        o_ptr[i] = y
    elementwise[istft_func, simd_width=1, target="gpu"](
        IndexList[1](b * t_audio), DeviceContextPtr(ctx),
    )


def hift_decode_full(
    mut ctx: DeviceContext,
    mut model: HiFTGenerator,
    mut mel_buf: DeviceBuffer[DType.float32],     # (B, 80, T_mel)
    mut s_stft_buf: DeviceBuffer[DType.float32],
    mut window_buf: DeviceBuffer[DType.float32],  # (n_fft,) hann
    mut audio_out: DeviceBuffer[DType.float32],   # (B, T_audio)
    b: Int, t_mel: Int, t_s: Int, t_pre_istft: Int, t_audio: Int,
    use_source: Bool = True,
) raises:
    """End-to-end NSF-HiFiGAN: mel → conv_pre/ups/MRF → conv_post → iSTFT → audio."""
    var n_fft = model.n_fft
    var n_out_channels = n_fft + 2
    var spec = ctx.enqueue_create_buffer[DType.float32](b * n_out_channels * t_pre_istft)
    hift_decode_trunk(
        ctx, model, mel_buf, s_stft_buf, spec,
        b, t_mel, t_s, t_pre_istft, use_source=use_source,
    )
    istft_forward(ctx, spec, window_buf, audio_out, b, n_fft, t_pre_istft, t_audio)
