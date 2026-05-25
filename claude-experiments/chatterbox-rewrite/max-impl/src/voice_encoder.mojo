"""VoiceEncoder forward built strictly from MAX abstractions.

  forward(mels: (B, T, M=40)) -> embed: (B, E=256)
    h_seq_l2, (h3, c3) = LSTM[40→256, 3 layers](mels)
    proj  = Linear[256→256](h3[-1])
    proj  = ReLU(proj)                    # nn.activations.relu via elementwise
    embed = proj / sqrt(sum_d(proj^2))   # std.algorithm.sum reduction + elementwise
"""
from std.math import sqrt
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList
from std.algorithm import sum as std_sum
from layout import Idx, TileTensor, row_major

from modules import Linear, linear_forward, relu
from lstm import LSTMLayer, lstm_layer_forward


@fieldwise_init
struct VoiceEncoder(Copyable, Movable):
    """Three-layer LSTM + Linear projection + ReLU + L2 normalize."""
    var lstm_l0: LSTMLayer
    var lstm_l1: LSTMLayer
    var lstm_l2: LSTMLayer
    var proj:    Linear
    var dim:     Int    # hidden / embed dim (typically 256)


def l2_normalize_rows(
    mut ctx: DeviceContext,
    mut x_buf: DeviceBuffer[DType.float32],
    mut out_buf: DeviceBuffer[DType.float32],
    batch: Int, feature_dim: Int,
) raises:
    """Per-row L2 normalize. Input (B, D), output (B, D).

    Uses `std.algorithm.sum` to reduce x*x over the last axis, then an
    `elementwise` pass to divide each element by the row's norm.
    """
    var norm_sq = ctx.enqueue_create_buffer[DType.float32](batch)
    var in_ptr = x_buf.unsafe_ptr()
    var out_ptr = out_buf.unsafe_ptr()
    var norm_ptr = norm_sq.unsafe_ptr()
    var dctx = DeviceContextPtr(ctx)
    var input_shape = IndexList[2, element_type=DType.int64](batch, feature_dim)

    @always_inline
    @parameter
    @__copy_capture(in_ptr, feature_dim)
    def sq_input_fn[width: Int, rank: Int](coords: IndexList[rank]) -> SIMD[DType.float32, width]:
        var c = rebind[IndexList[2]](coords)
        var idx = c[0] * feature_dim + c[1]
        var v = in_ptr.load[width=width](idx)
        return v * v

    @always_inline
    @parameter
    @__copy_capture(norm_ptr)
    def sum_output_fn[width: Int, rank: Int](coords: IndexList[rank], val: SIMD[DType.float32, width]):
        # Reduce sum over axis=1 keeps the rank — output coords are (batch, 0).
        var c = rebind[IndexList[2]](coords)
        norm_ptr.store[width=width](c[0], val)

    std_sum[DType.float32, sq_input_fn, sum_output_fn, target="gpu"](
        input_shape, 1, dctx,
    )

    @always_inline
    @parameter
    @__copy_capture(in_ptr, out_ptr, norm_ptr, feature_dim)
    def div_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var b = i // feature_dim
        var inv: Float32 = 1.0 / sqrt(norm_ptr[b] + 1.0e-12)
        var v = in_ptr.load[width=width, alignment=alignment](i)
        out_ptr.store[width=width, alignment=alignment](i, v * inv)
    elementwise[div_func, simd_width=1, target="gpu"](
        IndexList[1](batch * feature_dim), dctx,
    )


def voice_encoder_forward(
    mut ctx: DeviceContext,
    mut ve: VoiceEncoder,
    mut mels_buf: DeviceBuffer[DType.float32],   # (B, T, 40)
    mut embed_buf: DeviceBuffer[DType.float32],  # (B, 256) output
    batch: Int, time: Int,
) raises:
    """Run VoiceEncoder forward. Output is L2-normalized embedding."""
    var D = ve.dim
    var IN = ve.lstm_l0.input_size
    var H = D    # all 3 LSTM layers have hidden=D

    # Per-layer sequence buffers.
    var l0_seq = ctx.enqueue_create_buffer[DType.float32](batch * time * H)
    var l1_seq = ctx.enqueue_create_buffer[DType.float32](batch * time * H)
    var l2_seq = ctx.enqueue_create_buffer[DType.float32](batch * time * H)

    # Per-layer state.
    var l0_h = ctx.enqueue_create_buffer[DType.float32](batch * H)
    var l0_c = ctx.enqueue_create_buffer[DType.float32](batch * H)
    var l1_h = ctx.enqueue_create_buffer[DType.float32](batch * H)
    var l1_c = ctx.enqueue_create_buffer[DType.float32](batch * H)
    var l2_h = ctx.enqueue_create_buffer[DType.float32](batch * H)
    var l2_c = ctx.enqueue_create_buffer[DType.float32](batch * H)
    l0_h.enqueue_fill(0.0); l0_c.enqueue_fill(0.0)
    l1_h.enqueue_fill(0.0); l1_c.enqueue_fill(0.0)
    l2_h.enqueue_fill(0.0); l2_c.enqueue_fill(0.0)

    # Scratch shared across layers.
    var pre_xw_buf = ctx.enqueue_create_buffer[DType.float32](batch * 4 * H)
    var pre_hw_buf = ctx.enqueue_create_buffer[DType.float32](batch * 4 * H)

    # Layer 0 (IN=40 → H=256).
    lstm_layer_forward(
        ctx, ve.lstm_l0, mels_buf, l0_seq, l0_h, l0_c, pre_xw_buf, pre_hw_buf,
        batch, time,
    )
    # Layer 1 (H=256 → H=256).
    lstm_layer_forward(
        ctx, ve.lstm_l1, l0_seq, l1_seq, l1_h, l1_c, pre_xw_buf, pre_hw_buf,
        batch, time,
    )
    # Layer 2.
    lstm_layer_forward(
        ctx, ve.lstm_l2, l1_seq, l2_seq, l2_h, l2_c, pre_xw_buf, pre_hw_buf,
        batch, time,
    )

    # Take last hidden state h_state[-1] from layer 2. That's `l2_h` after the
    # final timestep. Run Linear → ReLU → L2 normalize.
    var proj_buf = ctx.enqueue_create_buffer[DType.float32](batch * D)
    linear_forward(ctx, ve.proj, l2_h, proj_buf, batch)
    var relu_buf = ctx.enqueue_create_buffer[DType.float32](batch * D)
    relu(ctx, proj_buf, relu_buf, batch * D)
    l2_normalize_rows(ctx, relu_buf, embed_buf, batch, D)


def voice_encoder_inference(
    mut ctx: DeviceContext,
    mut ve: VoiceEncoder,
    mut mel_full_buf: DeviceBuffer[DType.float32],   # (T, 40) full-utterance mel
    mut embed_out: DeviceBuffer[DType.float32],      # (256,) final speaker embedding
    t_frames: Int,
    frame_step: Int = 77,         # default = round((16000/1.3)/160) for rate=1.3
    partial_frames: Int = 160,
    n_mels: Int = 40,
    embed_dim: Int = 256,
) raises:
    """Multi-partial VoiceEncoder inference matching upstream's
    `VoiceEncoder.inference(mels, mel_lens, overlap=0.5, rate=1.3)`.

    Steps:
      1. Extract overlapping partials of size partial_frames at stride frame_step.
      2. Run VoiceEncoder forward on each (each output is L2-normalized).
      3. Mean across partials → 256-d vector.
      4. L2-normalize the mean → final speaker embedding.

    For T=1001, frame_step=77, partial_frames=160: n_partials = 11 + edge.
    """
    # Compute n_partials with min_coverage=0.8 heuristic from upstream:
    #   n_wins, remainder = divmod(max(T - W + step, 0), step)
    #   if remainder + (W - step) / W >= min_coverage: n_wins += 1
    # We simplify: n_partials = max(1, (t_frames - partial_frames) // frame_step + 1)
    var n_partials = 1
    if t_frames > partial_frames:
        n_partials = (t_frames - partial_frames) // frame_step + 1

    # Run VE forward at batch=1 for each partial, collecting embeddings.
    # The slice_fn and copy_fn closures are defined ONCE outside the loop;
    # they read `pi_box[0]` from a host-side scratch buffer so we don't
    # specialize the kernel per iteration.
    var partial_embeds = ctx.enqueue_create_buffer[DType.float32](n_partials * embed_dim)
    var ve_in = ctx.enqueue_create_buffer[DType.float32](partial_frames * n_mels)
    var ve_out = ctx.enqueue_create_buffer[DType.float32](embed_dim)
    var pi_box = ctx.enqueue_create_buffer[DType.int32](1)

    var mp = mel_full_buf.unsafe_ptr()
    var vp = ve_in.unsafe_ptr()
    var pip = pi_box.unsafe_ptr()
    var pep = partial_embeds.unsafe_ptr()
    var vop = ve_out.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(mp, vp, pip, partial_frames, frame_step, n_mels, t_frames)
    def slice_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var fi = i // n_mels
        var mi = i - fi * n_mels
        var p_start = Int(pip[0]) * frame_step
        var src_t = p_start + fi
        if src_t < t_frames:
            vp[i] = mp[src_t * n_mels + mi]
        else:
            vp[i] = 0.0

    @always_inline
    @parameter
    @__copy_capture(pep, vop, pip, embed_dim)
    def copy_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var d = idx[0]
        pep[Int(pip[0]) * embed_dim + d] = vop[d]

    for pi in range(n_partials):
        with pi_box.map_to_host() as h:
            h[0] = Int32(pi)
        elementwise[slice_fn, simd_width=1, target="gpu"](
            IndexList[1](partial_frames * n_mels), DeviceContextPtr(ctx),
        )
        voice_encoder_forward(ctx, ve, ve_in, ve_out, 1, partial_frames)
        elementwise[copy_fn, simd_width=1, target="gpu"](
            IndexList[1](embed_dim), DeviceContextPtr(ctx),
        )
    ctx.synchronize()

    # Mean across partials → 256-d.
    var mean_buf = ctx.enqueue_create_buffer[DType.float32](embed_dim)
    var pep_mean = partial_embeds.unsafe_ptr()
    var mbp = mean_buf.unsafe_ptr()
    var inv_n = Float32(1.0) / Float32(n_partials)

    @always_inline
    @parameter
    @__copy_capture(pep_mean, mbp, n_partials, embed_dim, inv_n)
    def mean_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var d = idx[0]
        var s: Float32 = 0.0
        for p in range(n_partials):
            s += pep_mean[p * embed_dim + d]
        mbp[d] = s * inv_n
    elementwise[mean_fn, simd_width=1, target="gpu"](
        IndexList[1](embed_dim), DeviceContextPtr(ctx),
    )

    # L2-normalize the mean → final embed.
    var sumsq: Float32 = 0.0
    with mean_buf.map_to_host() as h:
        for d in range(embed_dim):
            sumsq += h[d] * h[d]
    var inv_norm = Float32(1.0) / sqrt(sumsq) if sumsq > 0.0 else Float32(1.0)

    var mbp2 = mean_buf.unsafe_ptr()
    var op = embed_out.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(mbp2, op, inv_norm)
    def norm_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var d = idx[0]
        op[d] = mbp2[d] * inv_norm
    elementwise[norm_fn, simd_width=1, target="gpu"](
        IndexList[1](embed_dim), DeviceContextPtr(ctx),
    )
