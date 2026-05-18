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

from modules import (
    Linear, linear_forward, RMSNorm, rms_norm_forward, silu, residual_add,
)
from attention import qk_scaled_and_masked, softmax_2d, av_matmul
from transformer_blocks import (
    reshape_bsd_to_bhsd, reshape_bhsd_to_bsd, apply_rope_hf_style,
)
from t3_block import T3Block


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
    b: Int, max_ctx: Int, cur_len: Int,
) raises:
    """One decode step for a single transformer layer with KV caching.

    Mutates x_buf, k_cache, v_cache. cur_len is the number of valid cache
    entries BEFORE this step (so the new token writes to slot `cur_len` and
    attends over [0..cur_len])."""
    var H = block.n_heads
    var Dh = block.head_dim
    var D = H * Dh
    var s_k = cur_len + 1     # K spans cache + new token

    # 1. RMSNorm.
    var x_norm = ctx.enqueue_create_buffer[DType.float32](b * D)
    rms_norm_forward(ctx, block.in_norm, x_buf, x_norm, b)

    # 2. Q/K/V projections (single timestep — Linear with M=B).
    var q_lin = ctx.enqueue_create_buffer[DType.float32](b * D)
    var k_lin = ctx.enqueue_create_buffer[DType.float32](b * D)
    var v_lin = ctx.enqueue_create_buffer[DType.float32](b * D)
    linear_forward(ctx, block.to_q, x_norm, q_lin, b)
    linear_forward(ctx, block.to_k, x_norm, k_lin, b)
    linear_forward(ctx, block.to_v, x_norm, v_lin, b)

    # 3. Reshape (B, D) → (B, 1, H, Dh) → (B, H, 1, Dh) via reshape helper.
    var q_perm = ctx.enqueue_create_buffer[DType.float32](b * H * 1 * Dh)
    var k_perm = ctx.enqueue_create_buffer[DType.float32](b * H * 1 * Dh)
    var v_perm = ctx.enqueue_create_buffer[DType.float32](b * H * 1 * Dh)
    reshape_bsd_to_bhsd(ctx, q_lin, q_perm, b, 1, H, Dh)
    reshape_bsd_to_bhsd(ctx, k_lin, k_perm, b, 1, H, Dh)
    reshape_bsd_to_bhsd(ctx, v_lin, v_perm, b, 1, H, Dh)

    # 4. RoPE on Q and K (single timestep with cos/sin at cur_len position).
    var q_rope = ctx.enqueue_create_buffer[DType.float32](b * H * 1 * Dh)
    var k_rope = ctx.enqueue_create_buffer[DType.float32](b * H * 1 * Dh)
    apply_rope_hf_style(ctx, q_perm, q_rope, cos_step, sin_step, b, H, 1, Dh)
    apply_rope_hf_style(ctx, k_perm, k_rope, cos_step, sin_step, b, H, 1, Dh)

    # 5. Append k_rope, v_perm to caches at slot cur_len.
    cache_write_step(ctx, k_rope, k_cache, b, H, max_ctx, Dh, cur_len)
    cache_write_step(ctx, v_perm, v_cache, b, H, max_ctx, Dh, cur_len)

    # 6. Attention over the FULL cache up through cur_len. Need an attention
    #    that uses K/V as (B,H,s_k,Dh) views of the cache prefix. Easiest:
    #    copy the prefix into a contiguous buffer and reuse `qk_scaled_and_masked`.
    var k_view = ctx.enqueue_create_buffer[DType.float32](b * H * s_k * Dh)
    var v_view = ctx.enqueue_create_buffer[DType.float32](b * H * s_k * Dh)
    var kcp = k_cache.unsafe_ptr()
    var vcp = v_cache.unsafe_ptr()
    var kvp = k_view.unsafe_ptr()
    var vvp = v_view.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(kcp, kvp, max_ctx, s_k, Dh, H)
    def copy_k[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (H * s_k * Dh)
        var rem = i - bi * H * s_k * Dh
        var hi = rem // (s_k * Dh)
        var rem2 = rem - hi * s_k * Dh
        var si = rem2 // Dh
        var di = rem2 - si * Dh
        var src = bi * H * max_ctx * Dh + hi * max_ctx * Dh + si * Dh + di
        kvp[i] = kcp[src]
    elementwise[copy_k, simd_width=1, target="gpu"](
        IndexList[1](b * H * s_k * Dh), DeviceContextPtr(ctx),
    )

    @always_inline
    @parameter
    @__copy_capture(vcp, vvp, max_ctx, s_k, Dh, H)
    def copy_v[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (H * s_k * Dh)
        var rem = i - bi * H * s_k * Dh
        var hi = rem // (s_k * Dh)
        var rem2 = rem - hi * s_k * Dh
        var si = rem2 // Dh
        var di = rem2 - si * Dh
        var src = bi * H * max_ctx * Dh + hi * max_ctx * Dh + si * Dh + di
        vvp[i] = vcp[src]
    elementwise[copy_v, simd_width=1, target="gpu"](
        IndexList[1](b * H * s_k * Dh), DeviceContextPtr(ctx),
    )

    # No mask (causal mask is implicit since cache only has positions ≤ cur_len).
    var no_mask = ctx.enqueue_create_buffer[DType.float32](1 * s_k)
    no_mask.enqueue_fill(0.0)

    var scale: Float32 = 1.0 / sqrt(Float32(Dh))
    var logits = ctx.enqueue_create_buffer[DType.float32](b * H * 1 * s_k)
    var probs  = ctx.enqueue_create_buffer[DType.float32](b * H * 1 * s_k)
    var av     = ctx.enqueue_create_buffer[DType.float32](b * H * 1 * Dh)
    qk_scaled_and_masked(ctx, q_rope, k_view, no_mask, logits,
                          b * H, 1, s_k, Dh, scale, False)
    softmax_2d(ctx, logits, probs, b * H * 1, s_k)
    av_matmul(ctx, probs, v_view, av, b * H, 1, s_k, Dh)

    var av_flat = ctx.enqueue_create_buffer[DType.float32](b * D)
    reshape_bhsd_to_bsd(ctx, av, av_flat, b, H, 1, Dh)
    var attn_out = ctx.enqueue_create_buffer[DType.float32](b * D)
    linear_forward(ctx, block.to_out, av_flat, attn_out, b)
    residual_add(ctx, x_buf, attn_out, b * D)

    # 7. MLP.
    var x_norm2 = ctx.enqueue_create_buffer[DType.float32](b * D)
    rms_norm_forward(ctx, block.post_norm, x_buf, x_norm2, b)
    var inter = block.mlp.intermediate
    var gate_h = ctx.enqueue_create_buffer[DType.float32](b * inter)
    var up_h   = ctx.enqueue_create_buffer[DType.float32](b * inter)
    var act_h  = ctx.enqueue_create_buffer[DType.float32](b * inter)
    var prod_h = ctx.enqueue_create_buffer[DType.float32](b * inter)
    linear_forward(ctx, block.mlp.gate, x_norm2, gate_h, b)
    linear_forward(ctx, block.mlp.up,   x_norm2, up_h,   b)
    silu(ctx, gate_h, act_h, b * inter)

    var ap = act_h.unsafe_ptr()
    var upp = up_h.unsafe_ptr()
    var pp = prod_h.unsafe_ptr()

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

    var mlp_out = ctx.enqueue_create_buffer[DType.float32](b * D)
    linear_forward(ctx, block.mlp.down, prod_h, mlp_out, b)
    residual_add(ctx, x_buf, mlp_out, b * D)
