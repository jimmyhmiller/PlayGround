"""T3 Llama-style transformer block (RMSNorm + MHA + RMSNorm + SwiGLU).

Uses MAX abstractions throughout:
  - `nn.normalization.rms_norm` via `modules.rms_norm_forward`
  - `linalg.matmul` for Q/K/V/out projections
  - `nn.rope.apply_rope` for rotary position embedding (or our helper)
  - `linalg.bmm.batched_matmul` for SDPA matmuls
  - `nn.softmax.softmax` for attention probs
  - `elementwise[..., target="gpu"]` for elementwise ops (residual, scale, mask)
"""
from std.math import sqrt
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from modules import (
    Linear, linear_forward,
    RMSNorm, rms_norm_forward,
    residual_add,
)
from attention import qk_scaled_and_masked, softmax_2d, av_matmul
from transformer_blocks import (
    reshape_bsd_to_bhsd, reshape_bhsd_to_bsd,
    apply_rope_s3_style, apply_rope_hf_style,
    LlamaMLP, llama_mlp_forward,
)
from std.io.file import open as _open_for_dump


def _maybe_dump_block_intermediates(
    tag: String, ctx: DeviceContext,
    buf: DeviceBuffer[DType.float32], n: Int,
):
    """If /tmp/t3_dump/DUMP_L0_<tag> marker exists, dump buf. Marker is removed
    after writing so we only dump once per tag.
    """
    var marker_path = String("/tmp/t3_dump/DUMP_L0_") + tag
    var done_path = String("/tmp/t3_dump/DUMP_L0_") + tag + String("_DONE")
    var should_dump = False
    try:
        var f = _open_for_dump(marker_path, "r")
        f.close()
        # Also check DONE doesn't exist (so we only dump once).
        try:
            var fd = _open_for_dump(done_path, "r")
            fd.close()
            # DONE exists; skip.
            should_dump = False
        except:
            should_dump = True
    except:
        should_dump = False
    if not should_dump:
        return

    try:
        var path = String("/tmp/t3_dump/mojo_l0_") + tag + String(".bin")
        var f = _open_for_dump(path, "w")
        with buf.map_to_host() as h:
            var bufs = List[UInt8](capacity=n * 4)
            for k in range(n):
                var v = h[k]
                var p = UnsafePointer(to=v).bitcast[UInt32]()
                var bits = p[0]
                for bb in range(4):
                    bufs.append(UInt8(Int((bits >> UInt32(8 * bb)) & 0xFF)))
            f.write_bytes(Span(bufs))
        f.close()
        # Remove marker so future calls skip — use os.remove not present in std?
        # Easiest: write a sentinel file that we treat as "dump-done".
        var done = _open_for_dump(marker_path + String("_DONE"), "w")
        done.write_bytes(Span(List[UInt8]()))
        done.close()
        print("[l0dump]", tag, "size", n)
    except e:
        print("dump failed for", tag, ":", e)


@fieldwise_init
struct T3Block(Copyable, Movable):
    """One Llama-30L transformer block.

    Includes fused Linears for fast decode:
      qkv:     concatenated to_q/to_k/to_v weights, shape (3*D, D).
      gate_up: concatenated gate/up weights, shape (2*intermediate, D).
    The individual to_q/to_k/to_v/mlp.gate/mlp.up Linears are retained for
    the prefill path and other call sites.
    """
    var in_norm: RMSNorm
    var post_norm: RMSNorm
    var to_q: Linear
    var to_k: Linear
    var to_v: Linear
    var to_out: Linear
    var mlp: LlamaMLP
    var qkv: Linear
    var gate_up: Linear
    var n_heads: Int
    var head_dim: Int


def t3_block_forward(
    mut ctx: DeviceContext,
    mut module: T3Block,
    mut x_buf: DeviceBuffer[DType.float32],   # (B, S, D) — also output (in-place residual)
    mut cos_buf: DeviceBuffer[DType.float32], # (S, HALF)
    mut sin_buf: DeviceBuffer[DType.float32],
    mut mask_buf: DeviceBuffer[DType.float32], # (S, S) bias
    b: Int, s: Int,
    has_mask: Bool,
) raises:
    """Forward pass for one T3 block.

    x_buf is modified in-place via two residual adds (pre-attn + pre-mlp).
    """
    var H = module.n_heads
    var Dh = module.head_dim
    var D = H * Dh

    # ---- attention ----
    var x_norm = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    rms_norm_forward(ctx, module.in_norm, x_buf, x_norm, b * s)

    var q_lin = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    var k_lin = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    var v_lin = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    linear_forward(ctx, module.to_q, x_norm, q_lin, b * s)
    linear_forward(ctx, module.to_k, x_norm, k_lin, b * s)
    linear_forward(ctx, module.to_v, x_norm, v_lin, b * s)

    # Reshape (B, S, D) → (B, H, S, Dh) for SDPA (matches HF Llama layout).
    var q_perm = ctx.enqueue_create_buffer[DType.float32](b * H * s * Dh)
    var k_perm = ctx.enqueue_create_buffer[DType.float32](b * H * s * Dh)
    var v_perm = ctx.enqueue_create_buffer[DType.float32](b * H * s * Dh)
    reshape_bsd_to_bhsd(ctx, q_lin, q_perm, b, s, H, Dh)
    reshape_bsd_to_bhsd(ctx, k_lin, k_perm, b, s, H, Dh)
    reshape_bsd_to_bhsd(ctx, v_lin, v_perm, b, s, H, Dh)
    # Apply HF Llama RoPE on (B, H, S, Dh) with cos/sin of shape (B, S, Dh).
    var q_rope = ctx.enqueue_create_buffer[DType.float32](b * H * s * Dh)
    var k_rope = ctx.enqueue_create_buffer[DType.float32](b * H * s * Dh)
    apply_rope_hf_style(ctx, q_perm, q_rope, cos_buf, sin_buf, b, H, s, Dh)
    apply_rope_hf_style(ctx, k_perm, k_rope, cos_buf, sin_buf, b, H, s, Dh)

    var scale: Float32 = 1.0 / sqrt(Float32(Dh))
    var logits = ctx.enqueue_create_buffer[DType.float32](b * H * s * s)
    var probs  = ctx.enqueue_create_buffer[DType.float32](b * H * s * s)
    var attn_perm = ctx.enqueue_create_buffer[DType.float32](b * H * s * Dh)
    qk_scaled_and_masked(ctx, q_rope, k_rope, mask_buf, logits,
                          b * H, s, s, Dh, scale, has_mask)
    softmax_2d(ctx, logits, probs, b * H * s, s)
    av_matmul(ctx, probs, v_perm, attn_perm, b * H, s, s, Dh)
    var attn_flat = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    reshape_bhsd_to_bsd(ctx, attn_perm, attn_flat, b, H, s, Dh)
    var attn_out = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    linear_forward(ctx, module.to_out, attn_flat, attn_out, b * s)

    # Residual: x += attn_out.
    residual_add(ctx, x_buf, attn_out, b * s * D)

    # ---- MLP ----
    var x_norm2 = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    rms_norm_forward(ctx, module.post_norm, x_buf, x_norm2, b * s)
    var mlp_out = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    llama_mlp_forward(ctx, module.mlp, x_norm2, mlp_out, b * s)
    residual_add(ctx, x_buf, mlp_out, b * s * D)


def t3_block_prefill(
    mut ctx: DeviceContext,
    mut module: T3Block,
    mut x_buf: DeviceBuffer[DType.float32],   # (B, S, D) — updated in-place
    mut cos_buf: DeviceBuffer[DType.float32],
    mut sin_buf: DeviceBuffer[DType.float32],
    mut mask_buf: DeviceBuffer[DType.float32],
    mut k_cache: DeviceBuffer[DType.float32],   # (B, H, MAX_CTX, Dh) — populated at slots [0, S)
    mut v_cache: DeviceBuffer[DType.float32],
    b: Int, s: Int, max_ctx: Int,
) raises:
    """Same as `t3_block_forward` (with causal mask) but ALSO writes the
    post-RoPE k_perm and v_perm into the per-layer cache at slots [0..S)."""
    var H = module.n_heads
    var Dh = module.head_dim
    var D = H * Dh

    # ---- attention ----
    var x_norm = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    rms_norm_forward(ctx, module.in_norm, x_buf, x_norm, b * s)

    var q_lin = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    var k_lin = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    var v_lin = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    linear_forward(ctx, module.to_q, x_norm, q_lin, b * s)
    linear_forward(ctx, module.to_k, x_norm, k_lin, b * s)
    linear_forward(ctx, module.to_v, x_norm, v_lin, b * s)

    # DEBUG: dump if env var set. Only dumps when L0_DUMP=1 file marker exists.
    _maybe_dump_block_intermediates("xnorm", ctx, x_norm, b * s * D)
    _maybe_dump_block_intermediates("qlin", ctx, q_lin, b * s * D)
    _maybe_dump_block_intermediates("klin", ctx, k_lin, b * s * D)
    _maybe_dump_block_intermediates("vlin", ctx, v_lin, b * s * D)

    # Reshape (B, S, D) → (B, H, S, Dh).
    var q_perm = ctx.enqueue_create_buffer[DType.float32](b * H * s * Dh)
    var k_perm = ctx.enqueue_create_buffer[DType.float32](b * H * s * Dh)
    var v_perm = ctx.enqueue_create_buffer[DType.float32](b * H * s * Dh)
    reshape_bsd_to_bhsd(ctx, q_lin, q_perm, b, s, H, Dh)
    reshape_bsd_to_bhsd(ctx, k_lin, k_perm, b, s, H, Dh)
    reshape_bsd_to_bhsd(ctx, v_lin, v_perm, b, s, H, Dh)
    var q_rope = ctx.enqueue_create_buffer[DType.float32](b * H * s * Dh)
    var k_rope = ctx.enqueue_create_buffer[DType.float32](b * H * s * Dh)
    apply_rope_hf_style(ctx, q_perm, q_rope, cos_buf, sin_buf, b, H, s, Dh)
    apply_rope_hf_style(ctx, k_perm, k_rope, cos_buf, sin_buf, b, H, s, Dh)

    _maybe_dump_block_intermediates("qrope", ctx, q_rope, b * H * s * Dh)
    _maybe_dump_block_intermediates("krope", ctx, k_rope, b * H * s * Dh)
    _maybe_dump_block_intermediates("vperm", ctx, v_perm, b * H * s * Dh)

    # Populate K/V cache at slots [0..S).
    var kp = k_rope.unsafe_ptr()
    var vp = v_perm.unsafe_ptr()
    var kcp = k_cache.unsafe_ptr()
    var vcp = v_cache.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(kp, vp, kcp, vcp, H, s, Dh, max_ctx)
    def cache_write[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (H * s * Dh)
        var rem = i - bi * H * s * Dh
        var hi = rem // (s * Dh)
        var rem2 = rem - hi * s * Dh
        var si = rem2 // Dh
        var di = rem2 - si * Dh
        var dst = bi * H * max_ctx * Dh + hi * max_ctx * Dh + si * Dh + di
        kcp[dst] = kp[i]
        vcp[dst] = vp[i]
    elementwise[cache_write, simd_width=1, target="gpu"](
        IndexList[1](b * H * s * Dh), DeviceContextPtr(ctx),
    )

    var scale: Float32 = 1.0 / sqrt(Float32(Dh))
    var logits = ctx.enqueue_create_buffer[DType.float32](b * H * s * s)
    var probs  = ctx.enqueue_create_buffer[DType.float32](b * H * s * s)
    var attn_perm = ctx.enqueue_create_buffer[DType.float32](b * H * s * Dh)
    qk_scaled_and_masked(ctx, q_rope, k_rope, mask_buf, logits,
                          b * H, s, s, Dh, scale, True)
    _maybe_dump_block_intermediates("qklogits", ctx, logits, b * H * s * s)
    softmax_2d(ctx, logits, probs, b * H * s, s)
    _maybe_dump_block_intermediates("attnprobs", ctx, probs, b * H * s * s)
    av_matmul(ctx, probs, v_perm, attn_perm, b * H, s, s, Dh)
    _maybe_dump_block_intermediates("av", ctx, attn_perm, b * H * s * Dh)
    var attn_flat = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    reshape_bhsd_to_bsd(ctx, attn_perm, attn_flat, b, H, s, Dh)
    var attn_out = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    linear_forward(ctx, module.to_out, attn_flat, attn_out, b * s)
    _maybe_dump_block_intermediates("attnout", ctx, attn_out, b * s * D)

    residual_add(ctx, x_buf, attn_out, b * s * D)
    _maybe_dump_block_intermediates("postattn", ctx, x_buf, b * s * D)

    var x_norm2 = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    rms_norm_forward(ctx, module.post_norm, x_buf, x_norm2, b * s)
    var mlp_out = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    llama_mlp_forward(ctx, module.mlp, x_norm2, mlp_out, b * s)
    residual_add(ctx, x_buf, mlp_out, b * s * D)


def t3_block_prefill_with_attn(
    mut ctx: DeviceContext,
    mut module: T3Block,
    mut x_buf: DeviceBuffer[DType.float32],
    mut cos_buf: DeviceBuffer[DType.float32],
    mut sin_buf: DeviceBuffer[DType.float32],
    mut mask_buf: DeviceBuffer[DType.float32],
    mut k_cache: DeviceBuffer[DType.float32],
    mut v_cache: DeviceBuffer[DType.float32],
    mut attn_probs_out: DeviceBuffer[DType.float32],   # (b, H, s, s) — softmax probs
    b: Int, s: Int, max_ctx: Int,
) raises:
    """Same as t3_block_prefill but ALSO writes (b, H, s, s) attention probs
    into attn_probs_out."""
    var H = module.n_heads
    var Dh = module.head_dim
    var D = H * Dh

    var x_norm = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    rms_norm_forward(ctx, module.in_norm, x_buf, x_norm, b * s)

    var q_lin = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    var k_lin = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    var v_lin = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    linear_forward(ctx, module.to_q, x_norm, q_lin, b * s)
    linear_forward(ctx, module.to_k, x_norm, k_lin, b * s)
    linear_forward(ctx, module.to_v, x_norm, v_lin, b * s)

    var q_perm = ctx.enqueue_create_buffer[DType.float32](b * H * s * Dh)
    var k_perm = ctx.enqueue_create_buffer[DType.float32](b * H * s * Dh)
    var v_perm = ctx.enqueue_create_buffer[DType.float32](b * H * s * Dh)
    reshape_bsd_to_bhsd(ctx, q_lin, q_perm, b, s, H, Dh)
    reshape_bsd_to_bhsd(ctx, k_lin, k_perm, b, s, H, Dh)
    reshape_bsd_to_bhsd(ctx, v_lin, v_perm, b, s, H, Dh)
    var q_rope = ctx.enqueue_create_buffer[DType.float32](b * H * s * Dh)
    var k_rope = ctx.enqueue_create_buffer[DType.float32](b * H * s * Dh)
    apply_rope_hf_style(ctx, q_perm, q_rope, cos_buf, sin_buf, b, H, s, Dh)
    apply_rope_hf_style(ctx, k_perm, k_rope, cos_buf, sin_buf, b, H, s, Dh)

    var kp = k_rope.unsafe_ptr()
    var vp = v_perm.unsafe_ptr()
    var kcp = k_cache.unsafe_ptr()
    var vcp = v_cache.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(kp, vp, kcp, vcp, H, s, Dh, max_ctx)
    def cache_write_attn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (H * s * Dh)
        var rem = i - bi * H * s * Dh
        var hi = rem // (s * Dh)
        var rem2 = rem - hi * s * Dh
        var si = rem2 // Dh
        var di = rem2 - si * Dh
        var dst = bi * H * max_ctx * Dh + hi * max_ctx * Dh + si * Dh + di
        kcp[dst] = kp[i]
        vcp[dst] = vp[i]
    elementwise[cache_write_attn, simd_width=1, target="gpu"](
        IndexList[1](b * H * s * Dh), DeviceContextPtr(ctx),
    )

    var scale: Float32 = 1.0 / sqrt(Float32(Dh))
    var logits = ctx.enqueue_create_buffer[DType.float32](b * H * s * s)
    var attn_perm = ctx.enqueue_create_buffer[DType.float32](b * H * s * Dh)
    qk_scaled_and_masked(ctx, q_rope, k_rope, mask_buf, logits,
                          b * H, s, s, Dh, scale, True)
    softmax_2d(ctx, logits, attn_probs_out, b * H * s, s)
    av_matmul(ctx, attn_probs_out, v_perm, attn_perm, b * H, s, s, Dh)
    var attn_flat = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    reshape_bhsd_to_bsd(ctx, attn_perm, attn_flat, b, H, s, Dh)
    var attn_out_w = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    linear_forward(ctx, module.to_out, attn_flat, attn_out_w, b * s)
    residual_add(ctx, x_buf, attn_out_w, b * s * D)

    var x_norm2_w = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    rms_norm_forward(ctx, module.post_norm, x_buf, x_norm2_w, b * s)
    var mlp_out_w = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    llama_mlp_forward(ctx, module.mlp, x_norm2_w, mlp_out_w, b * s)
    residual_add(ctx, x_buf, mlp_out_w, b * s * D)
