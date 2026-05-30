"""T3 single-step decode (one new token, KV-cached) — built from MAX abstractions.

For autoregressive generation:
  Q is shape (B, H, 1, Dh)        — the new token's query
  K_cache/V_cache: (B, H, MAX_CTX, Dh) — pre-allocated cache buffers
  cur_len: how many slots [0..cur_len) are valid in the cache

Per layer:
  rmsnorm → Q/K/V Linear → reshape to (B,H,1,Dh) → RoPE on Q,K
  → write new K, V into K_cache[..., cur_len, :], V_cache[..., cur_len, :]
  → attention with effective S_k = cur_len + 1
  → out Linear + residual → rmsnorm → SwiGLU + residual
"""
from std.math import sqrt
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList
from std.os import getenv

from modules import (
    Linear, linear_forward, linear_forward_with_scratch,
    linear_forward_bf16_shape,
    RMSNorm, rms_norm_forward,
    rms_norm_fused_residual_add_forward, silu, residual_add,
)


# T3 model constants — known at compile time. Asserted against runtime values
# in t3_decode_step so the comptime-shape linear path is safe to use.
comptime T3_D = 1024
comptime T3_INTER = 4096
from attention import qk_scaled_and_masked, softmax_2d, av_matmul
from transformer_blocks import (
    reshape_bsd_to_bhsd, reshape_bhsd_to_bsd, apply_rope_hf_style,
)
from t3_block import T3Block


@fieldwise_init
struct T3DecodeScratch(Movable):
    """All per-step scratch buffers for `t3_decode_step`, allocated once at the
    start of the decode loop and reused across all (token × layer) calls.

    Sizes are baked in for batch=b, heads=H, head_dim=Dh, model_dim=D=H*Dh,
    mlp_inter, and the maximum K-cache prefix length max_ctx. The same scratch
    serves every layer (weights differ per layer; activations match).

    Eliminates ~37 enqueue_create_buffer calls per layer per token. At b=2,
    H=16, Dh=64, D=1024, inter=4096 → ~256 KB total scratch per CFG-batched
    decode step.
    """
    # Norms
    var x_norm:    DeviceBuffer[DType.float32]    # (b*D,) — pre-attn rms output
    var x_norm2:   DeviceBuffer[DType.float32]    # (b*D,) — post-attn rms output (fused with residual)
    # Q/K/V
    var qkv_out:   DeviceBuffer[DType.float32]    # (b*3*D,) — fused-QKV linear output
    var q_lin:     DeviceBuffer[DType.float32]    # (b*D,) — split path only
    var k_lin:     DeviceBuffer[DType.float32]    # (b*D,)
    var v_lin:     DeviceBuffer[DType.float32]    # (b*D,)
    var q_perm:    DeviceBuffer[DType.float32]    # (b*H*Dh,)
    var k_perm:    DeviceBuffer[DType.float32]    # (b*H*Dh,)
    var v_perm:    DeviceBuffer[DType.float32]    # (b*H*Dh,)
    var q_rope:    DeviceBuffer[DType.float32]    # (b*H*Dh,)
    var k_rope:    DeviceBuffer[DType.float32]    # (b*H*Dh,)
    # Attention
    var logits:    DeviceBuffer[DType.float32]    # (b*H*max_ctx,) — sized to max prefix
    var probs:     DeviceBuffer[DType.float32]    # (b*H*max_ctx,)
    var av:        DeviceBuffer[DType.float32]    # (b*H*Dh,)
    var av_flat:   DeviceBuffer[DType.float32]    # (b*D,)
    var attn_out:  DeviceBuffer[DType.float32]    # (b*D,)
    # MLP
    var gate_up_out: DeviceBuffer[DType.float32]  # (b*2*inter,)
    var gate_h:    DeviceBuffer[DType.float32]    # (b*inter,) — split path only
    var up_h:      DeviceBuffer[DType.float32]    # (b*inter,)
    var act_h:     DeviceBuffer[DType.float32]    # (b*inter,)
    var prod_h:    DeviceBuffer[DType.float32]    # (b*inter,)
    var mlp_out:   DeviceBuffer[DType.float32]    # (b*D,)
    # bf16 cast scratch for linear_forward. Sized to b * max(in_features) across all linears.
    # T3 has linears with in_features ∈ {D=1024, inter=4096}, so max = b * inter.
    var lin_x_bf16: DeviceBuffer[DType.bfloat16]


def make_t3_decode_scratch(
    mut ctx: DeviceContext,
    b: Int, H: Int, Dh: Int, intermediate: Int, max_ctx: Int,
) raises -> T3DecodeScratch:
    """Allocate all per-step scratch buffers once. Reused across the entire
    decode loop. Sizes match the largest needs across all layers."""
    var D = H * Dh
    var bf16_max_in = max(D, intermediate)
    return T3DecodeScratch(
        x_norm    = ctx.enqueue_create_buffer[DType.float32](b * D),
        x_norm2   = ctx.enqueue_create_buffer[DType.float32](b * D),
        qkv_out   = ctx.enqueue_create_buffer[DType.float32](b * 3 * D),
        q_lin     = ctx.enqueue_create_buffer[DType.float32](b * D),
        k_lin     = ctx.enqueue_create_buffer[DType.float32](b * D),
        v_lin     = ctx.enqueue_create_buffer[DType.float32](b * D),
        q_perm    = ctx.enqueue_create_buffer[DType.float32](b * H * Dh),
        k_perm    = ctx.enqueue_create_buffer[DType.float32](b * H * Dh),
        v_perm    = ctx.enqueue_create_buffer[DType.float32](b * H * Dh),
        q_rope    = ctx.enqueue_create_buffer[DType.float32](b * H * Dh),
        k_rope    = ctx.enqueue_create_buffer[DType.float32](b * H * Dh),
        logits    = ctx.enqueue_create_buffer[DType.float32](b * H * max_ctx),
        probs     = ctx.enqueue_create_buffer[DType.float32](b * H * max_ctx),
        av        = ctx.enqueue_create_buffer[DType.float32](b * H * Dh),
        av_flat   = ctx.enqueue_create_buffer[DType.float32](b * D),
        attn_out  = ctx.enqueue_create_buffer[DType.float32](b * D),
        gate_up_out = ctx.enqueue_create_buffer[DType.float32](b * 2 * intermediate),
        gate_h    = ctx.enqueue_create_buffer[DType.float32](b * intermediate),
        up_h      = ctx.enqueue_create_buffer[DType.float32](b * intermediate),
        act_h     = ctx.enqueue_create_buffer[DType.float32](b * intermediate),
        prod_h    = ctx.enqueue_create_buffer[DType.float32](b * intermediate),
        mlp_out   = ctx.enqueue_create_buffer[DType.float32](b * D),
        lin_x_bf16 = ctx.enqueue_create_buffer[DType.bfloat16](b * bf16_max_in),
    )


def decode_qk_against_cache(
    mut ctx: DeviceContext,
    mut q_buf: DeviceBuffer[DType.float32],         # (B, H, 1, Dh) — query for this step
    mut k_cache: DeviceBuffer[DType.float32],       # (B, H, MAX_CTX, Dh) — full cache
    mut logits_buf: DeviceBuffer[DType.float32],    # (B, H, s_k) — output logits
    b: Int, h: Int, dh: Int, max_ctx: Int, s_k: Int,
    scale: Float32,
) raises:
    """logits[bi, hi, ki] = scale * sum_d Q[bi, hi, 0, d] * K_cache[bi, hi, ki, d].

    One thread per (bi, hi, ki); reads cache directly with no copy.
    """
    var qp = q_buf.unsafe_ptr()
    var kp = k_cache.unsafe_ptr()
    var lp = logits_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(qp, kp, lp, h, dh, max_ctx, s_k, scale)
    def qk_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (h * s_k)
        var rem = i - bi * h * s_k
        var hi = rem // s_k
        var ki = rem - hi * s_k
        var q_base = bi * h * dh + hi * dh
        var k_base = bi * h * max_ctx * dh + hi * max_ctx * dh + ki * dh
        var acc: Float32 = 0.0
        for d in range(dh):
            acc += qp[q_base + d] * kp[k_base + d]
        lp[i] = acc * scale
    elementwise[qk_fn, simd_width=1, target="gpu"](
        IndexList[1](b * h * s_k), DeviceContextPtr(ctx),
    )


def decode_av_against_cache(
    mut ctx: DeviceContext,
    mut probs_buf: DeviceBuffer[DType.float32],     # (B, H, s_k) — softmax probs
    mut v_cache: DeviceBuffer[DType.float32],       # (B, H, MAX_CTX, Dh) — full cache
    mut out_buf: DeviceBuffer[DType.float32],       # (B, H, 1, Dh) — attention output
    b: Int, h: Int, dh: Int, max_ctx: Int, s_k: Int,
) raises:
    """out[bi, hi, 0, di] = sum_k probs[bi, hi, k] * V_cache[bi, hi, k, di].

    One thread per (bi, hi, di); reads cache directly with no copy.
    """
    var pp = probs_buf.unsafe_ptr()
    var vp = v_cache.unsafe_ptr()
    var op = out_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(pp, vp, op, h, dh, max_ctx, s_k)
    def av_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (h * dh)
        var rem = i - bi * h * dh
        var hi = rem // dh
        var di = rem - hi * dh
        var p_base = bi * h * s_k + hi * s_k
        var v_base = bi * h * max_ctx * dh + hi * max_ctx * dh
        var acc: Float32 = 0.0
        for k in range(s_k):
            acc += pp[p_base + k] * vp[v_base + k * dh + di]
        op[i] = acc
    elementwise[av_fn, simd_width=1, target="gpu"](
        IndexList[1](b * h * dh), DeviceContextPtr(ctx),
    )


def _qkv_split_reshape(
    mut ctx: DeviceContext,
    mut qkv_in: DeviceBuffer[DType.float32],   # (B, 3*D) — concatenated Q|K|V per row
    mut q_out: DeviceBuffer[DType.float32],    # (B, H, 1, Dh)
    mut k_out: DeviceBuffer[DType.float32],    # (B, H, 1, Dh)
    mut v_out: DeviceBuffer[DType.float32],    # (B, H, 1, Dh)
    b: Int, h: Int, dh: Int,
) raises:
    """Split fused QKV (B, 3D) into 3 separate (B, H, 1, Dh) tensors in one kernel."""
    var ip = qkv_in.unsafe_ptr()
    var qp = q_out.unsafe_ptr()
    var kp = k_out.unsafe_ptr()
    var vp = v_out.unsafe_ptr()
    var d = h * dh

    @always_inline
    @parameter
    @__copy_capture(ip, qp, kp, vp, h, dh, d)
    def split_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (h * dh)
        var rem = i - bi * h * dh
        var hi = rem // dh
        var di = rem - hi * dh
        var src_base = bi * 3 * d + hi * dh + di
        qp[i] = ip[src_base]
        kp[i] = ip[src_base + d]
        vp[i] = ip[src_base + 2 * d]
    elementwise[split_fn, simd_width=1, target="gpu"](
        IndexList[1](b * h * dh), DeviceContextPtr(ctx),
    )


def _swiglu_combine(
    mut ctx: DeviceContext,
    mut gate_up: DeviceBuffer[DType.float32],   # (B, 2*inter) — [b][0:inter]=gate, [b][inter:]=up
    mut out: DeviceBuffer[DType.float32],       # (B, inter)
    b: Int, inter: Int,
) raises:
    """Compute out[bi, i] = silu(gate_up[bi, i]) * gate_up[bi, inter + i].
    Fuses the silu activation with the gate*up product in one kernel.
    """
    from std.math import exp as mexp
    var gup = gate_up.unsafe_ptr()
    var op = out.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(gup, op, inter)
    def swiglu_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // inter
        var ii = i - bi * inter
        var base = bi * 2 * inter
        var g = gup[base + ii]
        var u = gup[base + inter + ii]
        # Match exactly the unfused modules.silu()*up FP order to avoid
        # off-by-one logit drift in T3 sampling (which can land tokens in
        # CFM-pathological regions → loud "thump" artifacts in audio).
        var sig = Float32(1.0) / (Float32(1.0) + mexp(-g))
        op[i] = (g * sig) * u
    elementwise[swiglu_fn, simd_width=1, target="gpu"](
        IndexList[1](b * inter), DeviceContextPtr(ctx),
    )


def cache_write_step(
    mut ctx: DeviceContext,
    mut step_buf: DeviceBuffer[DType.float32],   # (B, H, 1, Dh)
    mut cache_buf: DeviceBuffer[DType.float32],  # (B, H, MAX_CTX, Dh)
    b: Int, h: Int, max_ctx: Int, dh: Int,
    cur_len: Int,
) raises:
    """Copy step_buf into cache_buf at slot `cur_len`."""
    var sp = step_buf.unsafe_ptr()
    var cp = cache_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(sp, cp, max_ctx, dh, cur_len)
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (h * dh)
        var rem = i - bi * h * dh
        var hi = rem // dh
        var di = rem - hi * dh
        var dst_idx = bi * h * max_ctx * dh + hi * max_ctx * dh + cur_len * dh + di
        cp[dst_idx] = sp[i]
    elementwise[func, simd_width=1, target="gpu"](
        IndexList[1](b * h * dh), DeviceContextPtr(ctx),
    )


def t3_decode_step(
    mut ctx: DeviceContext,
    mut block: T3Block,
    mut x_buf: DeviceBuffer[DType.float32],          # (B, 1, D) — new token embedding, updated in-place
    mut k_cache: DeviceBuffer[DType.float32],        # (B, H, MAX_CTX, Dh)
    mut v_cache: DeviceBuffer[DType.float32],        # (B, H, MAX_CTX, Dh)
    mut cos_step: DeviceBuffer[DType.float32],       # (B, 1, Dh) — cos at current position
    mut sin_step: DeviceBuffer[DType.float32],
    mut scratch: T3DecodeScratch,                    # pre-allocated per-step buffers
    b: Int, max_ctx: Int, cur_len: Int,
) raises:
    """One decode step for a single transformer layer with KV caching.

    Mutates x_buf, k_cache, v_cache. cur_len is the number of valid cache
    entries BEFORE this step (so the new token writes to slot `cur_len` and
    attends over [0..cur_len]).

    All per-step temporaries (q_perm, k_perm, x_norm, logits, etc.) come from
    `scratch`, which is shared across every (token × layer) call. Eliminates
    ~37 enqueue_create_buffer calls per invocation.
    """
    var H = block.n_heads
    var Dh = block.head_dim
    var D = H * Dh
    var s_k = cur_len + 1     # K spans cache + new token

    # Comptime-shape linears require D and INTER match T3_D/T3_INTER.
    if D != T3_D or block.mlp.intermediate != T3_INTER:
        raise Error("t3_decode_step: expected D=T3_D and intermediate=T3_INTER")

    # 1. RMSNorm.
    rms_norm_forward(ctx, block.in_norm, x_buf, scratch.x_norm, b)

    # 2. Q/K/V — fused matmul (when env CHATTERBOX_T3_FUSE_QKV=1) or 3 separate.
    var fuse_qkv = False
    try:
        fuse_qkv = getenv("CHATTERBOX_T3_FUSE_QKV") == "1"
    except:
        fuse_qkv = False

    if fuse_qkv:
        # qkv: K=T3_D, N=3*T3_D, has_bias=False (Llama-style)
        linear_forward_bf16_shape[T3_D, 3 * T3_D, False](
            ctx, block.qkv, scratch.x_norm, scratch.qkv_out, scratch.lin_x_bf16, b,
        )
        _qkv_split_reshape(ctx, scratch.qkv_out, scratch.q_perm, scratch.k_perm, scratch.v_perm, b, H, Dh)
    else:
        linear_forward_bf16_shape[T3_D, T3_D, False](ctx, block.to_q, scratch.x_norm, scratch.q_lin, scratch.lin_x_bf16, b)
        linear_forward_bf16_shape[T3_D, T3_D, False](ctx, block.to_k, scratch.x_norm, scratch.k_lin, scratch.lin_x_bf16, b)
        linear_forward_bf16_shape[T3_D, T3_D, False](ctx, block.to_v, scratch.x_norm, scratch.v_lin, scratch.lin_x_bf16, b)
        reshape_bsd_to_bhsd(ctx, scratch.q_lin, scratch.q_perm, b, 1, H, Dh)
        reshape_bsd_to_bhsd(ctx, scratch.k_lin, scratch.k_perm, b, 1, H, Dh)
        reshape_bsd_to_bhsd(ctx, scratch.v_lin, scratch.v_perm, b, 1, H, Dh)

    # 4. RoPE on Q and K (single timestep with cos/sin at cur_len position).
    apply_rope_hf_style(ctx, scratch.q_perm, scratch.q_rope, cos_step, sin_step, b, H, 1, Dh)
    apply_rope_hf_style(ctx, scratch.k_perm, scratch.k_rope, cos_step, sin_step, b, H, 1, Dh)

    # 5. Append k_rope, v_perm to caches at slot cur_len.
    cache_write_step(ctx, scratch.k_rope, k_cache, b, H, max_ctx, Dh, cur_len)
    cache_write_step(ctx, scratch.v_perm, v_cache, b, H, max_ctx, Dh, cur_len)

    # 6. Attention directly against cache buffers — no copy. Reads the
    #    (B, H, MAX_CTX, Dh) cache buffers using only the valid prefix [0..s_k).
    var scale: Float32 = 1.0 / sqrt(Float32(Dh))

    decode_qk_against_cache(ctx, scratch.q_rope, k_cache, scratch.logits, b, H, Dh, max_ctx, s_k, scale)
    softmax_2d(ctx, scratch.logits, scratch.probs, b * H, s_k)
    decode_av_against_cache(ctx, scratch.probs, v_cache, scratch.av, b, H, Dh, max_ctx, s_k)

    reshape_bhsd_to_bsd(ctx, scratch.av, scratch.av_flat, b, H, 1, Dh)
    linear_forward_bf16_shape[T3_D, T3_D, False](ctx, block.to_out, scratch.av_flat, scratch.attn_out, scratch.lin_x_bf16, b)

    # 7. MLP — fused gate+up matmul (when env CHATTERBOX_T3_FUSE_MLP=1) or separate.
    # Fused: residual_add(x, attn_out) + rms_norm(x) → single launch.
    rms_norm_fused_residual_add_forward(ctx, block.post_norm, x_buf, scratch.attn_out, scratch.x_norm2, b)
    var inter = block.mlp.intermediate

    var fuse_mlp = False
    try:
        fuse_mlp = getenv("CHATTERBOX_T3_FUSE_MLP") == "1"
    except:
        fuse_mlp = False

    if fuse_mlp:
        # gate_up: K=T3_D, N=2*T3_INTER, has_bias=False
        linear_forward_bf16_shape[T3_D, 2 * T3_INTER, False](
            ctx, block.gate_up, scratch.x_norm2, scratch.gate_up_out, scratch.lin_x_bf16, b,
        )
        _swiglu_combine(ctx, scratch.gate_up_out, scratch.prod_h, b, inter)
    else:
        linear_forward_bf16_shape[T3_D, T3_INTER, False](ctx, block.mlp.gate, scratch.x_norm2, scratch.gate_h, scratch.lin_x_bf16, b)
        linear_forward_bf16_shape[T3_D, T3_INTER, False](ctx, block.mlp.up,   scratch.x_norm2, scratch.up_h,   scratch.lin_x_bf16, b)
        silu(ctx, scratch.gate_h, scratch.act_h, b * inter)
        var ap = scratch.act_h.unsafe_ptr()
        var upp = scratch.up_h.unsafe_ptr()
        var pp = scratch.prod_h.unsafe_ptr()

        @always_inline
        @parameter
        @__copy_capture(ap, upp, pp)
        def mul_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
            var i = idx[0]
            var a = ap.load[width=width, alignment=alignment](i)
            var u = upp.load[width=width, alignment=alignment](i)
            pp.store[width=width, alignment=alignment](i, a * u)
        elementwise[mul_func, simd_width=4, target="gpu"](
            IndexList[1](b * inter), DeviceContextPtr(ctx),
        )

    # down: K=T3_INTER, N=T3_D, has_bias=False
    linear_forward_bf16_shape[T3_INTER, T3_D, False](
        ctx, block.mlp.down, scratch.prod_h, scratch.mlp_out, scratch.lin_x_bf16, b,
    )
    residual_add(ctx, x_buf, scratch.mlp_out, b * D)


def t3_decode_step_with_attn(
    mut ctx: DeviceContext,
    mut block: T3Block,
    mut x_buf: DeviceBuffer[DType.float32],
    mut k_cache: DeviceBuffer[DType.float32],
    mut v_cache: DeviceBuffer[DType.float32],
    mut cos_step: DeviceBuffer[DType.float32],
    mut sin_step: DeviceBuffer[DType.float32],
    mut attn_probs_out: DeviceBuffer[DType.float32],   # (b, H, 1, s_k) — softmax probs
    b: Int, max_ctx: Int, cur_len: Int,
) raises:
    """Same as t3_decode_step but ALSO writes the post-softmax attention probs
    into attn_probs_out, shape (b, H, 1, s_k) where s_k = cur_len + 1.

    Caller is responsible for allocating attn_probs_out as b*H*1*s_k elements.
    """
    var H = block.n_heads
    var Dh = block.head_dim
    var D = H * Dh
    var s_k = cur_len + 1

    var x_norm = ctx.enqueue_create_buffer[DType.float32](b * D)
    rms_norm_forward(ctx, block.in_norm, x_buf, x_norm, b)

    # Fused QKV.
    var qkv_out = ctx.enqueue_create_buffer[DType.float32](b * 3 * D)
    linear_forward(ctx, block.qkv, x_norm, qkv_out, b)
    var q_perm = ctx.enqueue_create_buffer[DType.float32](b * H * 1 * Dh)
    var k_perm = ctx.enqueue_create_buffer[DType.float32](b * H * 1 * Dh)
    var v_perm = ctx.enqueue_create_buffer[DType.float32](b * H * 1 * Dh)
    _qkv_split_reshape(ctx, qkv_out, q_perm, k_perm, v_perm, b, H, Dh)

    var q_rope = ctx.enqueue_create_buffer[DType.float32](b * H * 1 * Dh)
    var k_rope = ctx.enqueue_create_buffer[DType.float32](b * H * 1 * Dh)
    apply_rope_hf_style(ctx, q_perm, q_rope, cos_step, sin_step, b, H, 1, Dh)
    apply_rope_hf_style(ctx, k_perm, k_rope, cos_step, sin_step, b, H, 1, Dh)

    cache_write_step(ctx, k_rope, k_cache, b, H, max_ctx, Dh, cur_len)
    cache_write_step(ctx, v_perm, v_cache, b, H, max_ctx, Dh, cur_len)

    var scale: Float32 = 1.0 / sqrt(Float32(Dh))
    var logits = ctx.enqueue_create_buffer[DType.float32](b * H * s_k)
    var av     = ctx.enqueue_create_buffer[DType.float32](b * H * 1 * Dh)
    decode_qk_against_cache(ctx, q_rope, k_cache, logits, b, H, Dh, max_ctx, s_k, scale)
    softmax_2d(ctx, logits, attn_probs_out, b * H, s_k)
    decode_av_against_cache(ctx, attn_probs_out, v_cache, av, b, H, Dh, max_ctx, s_k)

    var av_flat = ctx.enqueue_create_buffer[DType.float32](b * D)
    reshape_bhsd_to_bsd(ctx, av, av_flat, b, H, 1, Dh)
    var attn_out_buf = ctx.enqueue_create_buffer[DType.float32](b * D)
    linear_forward(ctx, block.to_out, av_flat, attn_out_buf, b)
    residual_add(ctx, x_buf, attn_out_buf, b * D)

    # Fused MLP gate+up + SwiGLU.
    var x_norm2 = ctx.enqueue_create_buffer[DType.float32](b * D)
    rms_norm_forward(ctx, block.post_norm, x_buf, x_norm2, b)
    var inter = block.mlp.intermediate
    var gate_up_out = ctx.enqueue_create_buffer[DType.float32](b * 2 * inter)
    var prod_h = ctx.enqueue_create_buffer[DType.float32](b * inter)
    linear_forward(ctx, block.gate_up, x_norm2, gate_up_out, b)
    _swiglu_combine(ctx, gate_up_out, prod_h, b, inter)

    var mlp_out = ctx.enqueue_create_buffer[DType.float32](b * D)
    linear_forward(ctx, block.mlp.down, prod_h, mlp_out, b)
    residual_add(ctx, x_buf, mlp_out, b * D)
