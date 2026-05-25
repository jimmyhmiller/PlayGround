"""s3tokenizer ResidualAttentionBlock (FSMN attention + MLP) using MAX abstractions.

Per-block computation:
  x_pre  = LayerNorm(x)
  q, k, v = Q(x_pre), K(x_pre), V(x_pre)             # Q/V have bias; K does not
  q4 = view(q, B, S, H, Dh); k4 = view(k, ...); v4 = view(v, ...)
  q4, k4 = apply_rope_s3_style(q4, k4, freqs_cis)
  v_masked = v.flatten(2) * mask_pad
  fsmn_conv = depthwise_conv1d(v_masked, kernel=31, pad=15)
  fsmn_mem  = (fsmn_conv + v_masked) * mask_pad
  scale = (Dh)^-0.25; q_perm = q4.permute(0,2,1,3) * scale; k_perm same
  logits = q_perm @ k_perm.T + mask           # (B,H,S,S)
  probs  = softmax(logits)
  attn   = probs @ v_perm                     # v_perm has no extra scale
  out_attn = combine_heads(attn)
  attn_out = out_proj(out_attn) + fsmn_mem
  x       = x + attn_out
  x_pre2  = LayerNorm(x)
  x       = x + MLP(x_pre2)        # Linear→GELU→Linear
"""
from std.math import sqrt
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from modules import (
    Linear, linear_forward,
    LayerNorm, layer_norm_forward,
    residual_add,
)
from attention import qk_scaled_and_masked, softmax_2d, av_matmul
from conv1d import Conv1d, conv1d_forward
from transformer_blocks import (
    reshape_bsd_to_bhsd, reshape_bhsd_to_bsd, apply_rope_s3_style,
    MLP, mlp_forward,
)


@fieldwise_init
struct FSMNAttention(Copyable, Movable):
    """FSMN multi-head attention used by s3tokenizer."""
    var to_q: Linear
    var to_k: Linear     # bias=False (caller sets has_bias=False)
    var to_v: Linear
    var to_out: Linear
    var fsmn_conv: Conv1d   # depthwise (groups=D), kernel=31, pad=0; caller pads input
    var n_heads: Int
    var head_dim: Int


def fsmn_attention_forward(
    mut ctx: DeviceContext,
    mut module: FSMNAttention,
    mut x_buf: DeviceBuffer[DType.float32],       # (B, S, D)
    mut out_buf: DeviceBuffer[DType.float32],     # (B, S, D)
    mut cos_buf: DeviceBuffer[DType.float32],     # (S, HALF)
    mut sin_buf: DeviceBuffer[DType.float32],
    mut mask_pad_buf: DeviceBuffer[DType.float32], # (B, S, 1) — 1 valid, 0 pad
    mut mask_buf: DeviceBuffer[DType.float32],    # (S, S) bias for attn — zero for full
    b: Int, s: Int,
    has_attn_mask: Bool,
) raises:
    var H = module.n_heads
    var Dh = module.head_dim
    var D = H * Dh

    var q_lin = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    var k_lin = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    var v_lin = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    linear_forward(ctx, module.to_q, x_buf, q_lin, b * s)
    linear_forward(ctx, module.to_k, x_buf, k_lin, b * s)
    linear_forward(ctx, module.to_v, x_buf, v_lin, b * s)

    # v_masked = v_lin * mask_pad.
    var v_masked = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    var v_ptr = v_lin.unsafe_ptr()
    var mp_ptr = mask_pad_buf.unsafe_ptr()
    var vm_ptr = v_masked.unsafe_ptr()
    var total = b * s * D
    var dctx = DeviceContextPtr(ctx)

    @always_inline
    @parameter
    @__copy_capture(v_ptr, mp_ptr, vm_ptr, s, D)
    def mask_v_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (s * D)
        var rem = i - bi * s * D
        var si = rem // D
        var m = mp_ptr[bi * s + si]
        vm_ptr[i] = v_ptr[i] * m
    elementwise[mask_v_func, simd_width=1, target="gpu"](
        IndexList[1](total), dctx,
    )

    # FSMN depthwise conv. Input v_masked is (B, S, D) but Conv1d wants (B, C=D, L=S).
    # We permute (B, S, D) → (B, D, S) via elementwise.
    var v_perm_bds = ctx.enqueue_create_buffer[DType.float32](b * D * s)
    var vbsd_ptr = v_masked.unsafe_ptr()
    var vbds_ptr = v_perm_bds.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(vbsd_ptr, vbds_ptr, b, s, D)
    def t1_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (D * s)
        var rem = i - bi * D * s
        var di = rem // s
        var si = rem - di * s
        vbds_ptr[i] = vbsd_ptr[bi * s * D + si * D + di]
    elementwise[t1_func, simd_width=1, target="gpu"](
        IndexList[1](b * D * s), dctx,
    )

    # Depthwise conv (groups=D, padding handled internally via conv1d's pad).
    var fsmn_out_bds = ctx.enqueue_create_buffer[DType.float32](b * D * s)
    conv1d_forward(ctx, module.fsmn_conv, v_perm_bds, fsmn_out_bds, b, s, s)

    # Permute (B, D, S) → (B, S, D).
    var fsmn_out = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    var fop_ptr = fsmn_out_bds.unsafe_ptr()
    var fo_ptr = fsmn_out.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(fop_ptr, fo_ptr, b, s, D)
    def t2_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (s * D)
        var rem = i - bi * s * D
        var si = rem // D
        var di = rem - si * D
        fo_ptr[i] = fop_ptr[bi * D * s + di * s + si]
    elementwise[t2_func, simd_width=1, target="gpu"](
        IndexList[1](total), dctx,
    )

    # fsm_mem = (fsmn_out + v_masked) * mask_pad.
    var fsm_mem = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    var fm_ptr = fsm_mem.unsafe_ptr()
    var fo2_ptr = fsmn_out.unsafe_ptr()
    var vm2_ptr = v_masked.unsafe_ptr()
    var mp2_ptr = mask_pad_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(fm_ptr, fo2_ptr, vm2_ptr, mp2_ptr, s, D)
    def fsmem_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (s * D)
        var rem = i - bi * s * D
        var si = rem // D
        var m = mp2_ptr[bi * s + si]
        fm_ptr[i] = (fo2_ptr[i] + vm2_ptr[i]) * m
    elementwise[fsmem_func, simd_width=1, target="gpu"](
        IndexList[1](total), dctx,
    )

    # Attention. Reshape Q/K/V to (B, S, H, Dh) and apply RoPE on Q, K.
    var q_4d = ctx.enqueue_create_buffer[DType.float32](b * s * H * Dh)
    var k_4d = ctx.enqueue_create_buffer[DType.float32](b * s * H * Dh)
    ctx.enqueue_copy(q_4d, q_lin)
    ctx.enqueue_copy(k_4d, k_lin)
    var q_rope = ctx.enqueue_create_buffer[DType.float32](b * s * H * Dh)
    var k_rope = ctx.enqueue_create_buffer[DType.float32](b * s * H * Dh)
    apply_rope_s3_style(ctx, q_4d, q_rope, cos_buf, sin_buf, b, s, H, Dh)
    apply_rope_s3_style(ctx, k_4d, k_rope, cos_buf, sin_buf, b, s, H, Dh)

    # Permute → (B, H, S, Dh).
    var q_perm = ctx.enqueue_create_buffer[DType.float32](b * H * s * Dh)
    var k_perm = ctx.enqueue_create_buffer[DType.float32](b * H * s * Dh)
    var v_perm = ctx.enqueue_create_buffer[DType.float32](b * H * s * Dh)
    reshape_bsd_to_bhsd(ctx, q_rope, q_perm, b, s, H, Dh)
    reshape_bsd_to_bhsd(ctx, k_rope, k_perm, b, s, H, Dh)
    reshape_bsd_to_bhsd(ctx, v_lin,  v_perm, b, s, H, Dh)

    # FSMN scale is Dh^-0.25 applied to BOTH Q and K (effective sqrt-scale).
    var scale: Float32 = 1.0 / sqrt(sqrt(Float32(Dh)))
    var q_ptr = q_perm.unsafe_ptr()
    var k_ptr = k_perm.unsafe_ptr()
    var qk_total = b * H * s * Dh

    @always_inline
    @parameter
    @__copy_capture(q_ptr, k_ptr, scale)
    def scale_qk_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        q_ptr[i] *= scale
        k_ptr[i] *= scale
    elementwise[scale_qk_func, simd_width=1, target="gpu"](
        IndexList[1](qk_total), dctx,
    )

    var logits = ctx.enqueue_create_buffer[DType.float32](b * H * s * s)
    var probs  = ctx.enqueue_create_buffer[DType.float32](b * H * s * s)
    var attn_perm = ctx.enqueue_create_buffer[DType.float32](b * H * s * Dh)
    qk_scaled_and_masked(ctx, q_perm, k_perm, mask_buf, logits,
                          b * H, s, s, Dh, Float32(1.0), has_attn_mask)
    softmax_2d(ctx, logits, probs, b * H * s, s)
    av_matmul(ctx, probs, v_perm, attn_perm, b * H, s, s, Dh)
    var attn_flat = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    reshape_bhsd_to_bsd(ctx, attn_perm, attn_flat, b, H, s, Dh)
    var attn_lin = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    linear_forward(ctx, module.to_out, attn_flat, attn_lin, b * s)

    # Final: attn_lin + fsm_mem → out_buf.
    var al_ptr = attn_lin.unsafe_ptr()
    var fm2_ptr = fsm_mem.unsafe_ptr()
    var out_ptr = out_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(al_ptr, fm2_ptr, out_ptr)
    def sum_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        out_ptr[i] = al_ptr[i] + fm2_ptr[i]
    elementwise[sum_func, simd_width=1, target="gpu"](
        IndexList[1](total), dctx,
    )


@fieldwise_init
struct S3TokenizerBlock(Copyable, Movable):
    """One s3tokenizer ResidualAttentionBlock: LN + FSMN + LN + MLP."""
    var attn_ln: LayerNorm
    var mlp_ln: LayerNorm
    var attn: FSMNAttention
    var mlp: MLP


def s3tokenizer_block_forward(
    mut ctx: DeviceContext,
    mut module: S3TokenizerBlock,
    mut x_buf: DeviceBuffer[DType.float32],   # (B, S, D) in/out (residual updates)
    mut cos_buf: DeviceBuffer[DType.float32],
    mut sin_buf: DeviceBuffer[DType.float32],
    mut mask_pad_buf: DeviceBuffer[DType.float32],
    mut attn_mask_buf: DeviceBuffer[DType.float32],
    b: Int, s: Int,
    has_attn_mask: Bool,
) raises:
    var D = module.attn.n_heads * module.attn.head_dim

    # attn_ln(x) → attn.
    var ln_x = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    layer_norm_forward(ctx, module.attn_ln, x_buf, ln_x, b * s)
    var attn_out = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    fsmn_attention_forward(ctx, module.attn, ln_x, attn_out,
                            cos_buf, sin_buf, mask_pad_buf, attn_mask_buf,
                            b, s, has_attn_mask)
    residual_add(ctx, x_buf, attn_out, b * s * D)

    # mlp_ln(x) → mlp.
    var ln_x2 = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    layer_norm_forward(ctx, module.mlp_ln, x_buf, ln_x2, b * s)
    var mlp_out = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    mlp_forward(ctx, module.mlp, ln_x2, mlp_out, b * s)
    residual_add(ctx, x_buf, mlp_out, b * s * D)
