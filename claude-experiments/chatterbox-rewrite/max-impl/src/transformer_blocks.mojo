"""Transformer building blocks composed strictly from MAX abstractions.

Provides:
  - `MHASelfAttention`: standard MHA with linear Q/K/V + bmm SDPA + linear out.
  - `FSMNAttention`: variant used by s3tokenizer (depthwise Conv1d memory branch
    + standard attention with `Dh^-0.25` scale on both Q and K + RoPE).
  - `LlamaMLP`: SwiGLU MLP used by T3 (gate/up Linear + silu + down Linear).
  - `MLP`: GELU MLP used by s3tokenizer ResidualAttentionBlock.

All op-level work goes through `modules.*` wrappers, which delegate to
`nn.normalization`, `nn.softmax`, `linalg.matmul`, `linalg.bmm`, or
`elementwise[..., target="gpu"]`.
"""
from std.math import sqrt, cos, sin, pi
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList
from layout import Idx, TileTensor, row_major

from modules import (
    Linear, linear_forward,
    LayerNorm, layer_norm_forward,
    RMSNorm, rms_norm_forward,
    silu, gelu, residual_add,
)
from attention import qk_scaled_and_masked, softmax_2d, av_matmul
from conv1d import Conv1d, conv1d_forward


# ============================================================================
# Helper: split flat (M, H*D) → (M, H, D) is a no-op in memory (row-major).
# We just rebind layouts when launching downstream kernels.
# ============================================================================

def reshape_bsd_to_bhsd(
    mut ctx: DeviceContext,
    mut in_buf: DeviceBuffer[DType.float32],    # (B, S, H*D) interpreted as (B, S, H, D)
    mut out_buf: DeviceBuffer[DType.float32],   # (B, H, S, D)
    b: Int, s: Int, h: Int, d: Int,
) raises:
    """Permute (B, S, H, D) → (B, H, S, D). Pure-elementwise op."""
    var ip = in_buf.unsafe_ptr()
    var op = out_buf.unsafe_ptr()
    var total = b * h * s * d
    var dctx = DeviceContextPtr(ctx)

    @always_inline
    @parameter
    @__copy_capture(ip, op, b, s, h, d)
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        # i in (B, H, S, D) layout.
        var bi = i // (h * s * d)
        var rem = i - bi * h * s * d
        var hi = rem // (s * d)
        var rem2 = rem - hi * s * d
        var si = rem2 // d
        var di = rem2 - si * d
        # Source in (B, S, H, D).
        var src = bi * s * h * d + si * h * d + hi * d + di
        op[i] = ip[src]

    elementwise[func, simd_width=1, target="gpu"](
        IndexList[1](total), dctx,
    )


def reshape_bhsd_to_bsd(
    mut ctx: DeviceContext,
    mut in_buf: DeviceBuffer[DType.float32],    # (B, H, S, D)
    mut out_buf: DeviceBuffer[DType.float32],   # (B, S, H*D)
    b: Int, h: Int, s: Int, d: Int,
) raises:
    """Permute (B, H, S, D) → (B, S, H*D)."""
    var ip = in_buf.unsafe_ptr()
    var op = out_buf.unsafe_ptr()
    var total = b * s * h * d
    var dctx = DeviceContextPtr(ctx)

    @always_inline
    @parameter
    @__copy_capture(ip, op, b, h, s, d)
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (s * h * d)
        var rem = i - bi * s * h * d
        var si = rem // (h * d)
        var rem2 = rem - si * h * d
        var hi = rem2 // d
        var di = rem2 - hi * d
        # Source in (B, H, S, D).
        var src = bi * h * s * d + hi * s * d + si * d + di
        op[i] = ip[src]

    elementwise[func, simd_width=1, target="gpu"](
        IndexList[1](total), dctx,
    )


def apply_rope_hf_style(
    mut ctx: DeviceContext,
    mut in_buf: DeviceBuffer[DType.float32],    # (B, H, S, D) — HF Llama layout
    mut out_buf: DeviceBuffer[DType.float32],
    mut cos_buf: DeviceBuffer[DType.float32],   # (B, S, D) — full cos
    mut sin_buf: DeviceBuffer[DType.float32],   # (B, S, D) — full sin
    b: Int, h: Int, s: Int, d: Int,
) raises:
    """HF Llama RoPE on layout (B, H, S, D) with cos/sin (B, S, D).
       For i < D/2: out[..., i] = x[..., i] * cos[s, i] + (-x[..., i+HALF]) * sin[s, i]
       For i >= D/2: out[..., i] = x[..., i] * cos[s, i] + x[..., i-HALF] * sin[s, i]
    """
    var ip = in_buf.unsafe_ptr()
    var op = out_buf.unsafe_ptr()
    var cp = cos_buf.unsafe_ptr()
    var sp = sin_buf.unsafe_ptr()
    var half = d // 2
    var total = b * h * s * d
    var dctx = DeviceContextPtr(ctx)

    @always_inline
    @parameter
    @__copy_capture(ip, op, cp, sp, h, s, d, half)
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        # (B, H, S, D) layout.
        var bi = i // (h * s * d)
        var rem = i - bi * h * s * d
        var hi = rem // (s * d)
        var rem2 = rem - hi * s * d
        var si = rem2 // d
        var di = rem2 - si * d
        var c = cp[bi * s * d + si * d + di]
        var sn = sp[bi * s * d + si * d + di]
        var x_i = ip[i]
        var pair_di: Int = di + half if di < half else di - half
        var pair_src = bi * h * s * d + hi * s * d + si * d + pair_di
        var paired = ip[pair_src]
        var rh: Float32 = -paired if di < half else paired
        op[i] = x_i * c + rh * sn
    elementwise[func, simd_width=1, target="gpu"](
        IndexList[1](total), dctx,
    )


def apply_rope_s3_style(
    mut ctx: DeviceContext,
    mut in_buf: DeviceBuffer[DType.float32],    # (B, S, H, D) — half-rotation RoPE
    mut out_buf: DeviceBuffer[DType.float32],
    mut cos_buf: DeviceBuffer[DType.float32],   # (S, D) — cos duplicated halves
    mut sin_buf: DeviceBuffer[DType.float32],   # (S, D) — sin duplicated halves
    b: Int, s: Int, h: Int, d: Int,
) raises:
    """s3tokenizer-style RoPE (half rotation, cos/sin duplicated across D).

    For i < D/2:  out[..., i] = x[..., i] * cos[s, i] + (-x[..., i+HALF]) * sin[s, i]
    For i >= D/2: out[..., i] = x[..., i] * cos[s, i] + x[..., i-HALF] * sin[s, i]

    cos_buf/sin_buf are full (S, D) — i.e., cos[s, i] == cos[s, i-HALF] for i>=HALF.
    """
    var ip = in_buf.unsafe_ptr()
    var op = out_buf.unsafe_ptr()
    var cp = cos_buf.unsafe_ptr()
    var sp = sin_buf.unsafe_ptr()
    var half = d // 2
    var total = b * s * h * d
    var dctx = DeviceContextPtr(ctx)

    @always_inline
    @parameter
    @__copy_capture(ip, op, cp, sp, b, s, h, d, half)
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (s * h * d)
        var rem = i - bi * s * h * d
        var si = rem // (h * d)
        var rem2 = rem - si * h * d
        var hi = rem2 // d
        var di = rem2 - hi * d
        var c = cp[si * d + di]
        var sn = sp[si * d + di]
        var x_i = ip[i]
        var pair_di: Int = di + half if di < half else di - half
        var pair_src = bi * s * h * d + si * h * d + hi * d + pair_di
        var paired = ip[pair_src]
        var rh: Float32 = -paired if di < half else paired
        op[i] = x_i * c + rh * sn

    elementwise[func, simd_width=1, target="gpu"](
        IndexList[1](total), dctx,
    )


# ============================================================================
# Standard self-attention block (used by Perceiver, T3)
# ============================================================================

@fieldwise_init
struct MHASelfAttention(Copyable, Movable):
    """Multi-head self-attention: Q/K/V linear → SDPA → out linear."""
    var to_q: Linear
    var to_k: Linear
    var to_v: Linear
    var to_out: Linear
    var n_heads: Int
    var head_dim: Int


def mha_self_forward(
    mut ctx: DeviceContext,
    mut module: MHASelfAttention,
    mut x_buf: DeviceBuffer[DType.float32],       # (B, S, D=H*Dh)
    mut out_buf: DeviceBuffer[DType.float32],
    mut mask_buf: DeviceBuffer[DType.float32],    # (S, S) — bias
    b: Int, s: Int,
    has_mask: Bool,
) raises:
    """Self-attention forward. Scale = 1/sqrt(head_dim)."""
    var H = module.n_heads
    var Dh = module.head_dim
    var D = H * Dh

    var q_lin = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    var k_lin = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    var v_lin = ctx.enqueue_create_buffer[DType.float32](b * s * D)
    var q_perm = ctx.enqueue_create_buffer[DType.float32](b * H * s * Dh)
    var k_perm = ctx.enqueue_create_buffer[DType.float32](b * H * s * Dh)
    var v_perm = ctx.enqueue_create_buffer[DType.float32](b * H * s * Dh)
    var logits = ctx.enqueue_create_buffer[DType.float32](b * H * s * s)
    var probs  = ctx.enqueue_create_buffer[DType.float32](b * H * s * s)
    var attn_perm = ctx.enqueue_create_buffer[DType.float32](b * H * s * Dh)
    var attn_flat = ctx.enqueue_create_buffer[DType.float32](b * s * D)

    linear_forward(ctx, module.to_q, x_buf, q_lin, b * s)
    linear_forward(ctx, module.to_k, x_buf, k_lin, b * s)
    linear_forward(ctx, module.to_v, x_buf, v_lin, b * s)
    reshape_bsd_to_bhsd(ctx, q_lin, q_perm, b, s, H, Dh)
    reshape_bsd_to_bhsd(ctx, k_lin, k_perm, b, s, H, Dh)
    reshape_bsd_to_bhsd(ctx, v_lin, v_perm, b, s, H, Dh)

    var scale: Float32 = 1.0 / sqrt(Float32(Dh))
    qk_scaled_and_masked(ctx, q_perm, k_perm, mask_buf, logits,
                          b * H, s, s, Dh, scale, has_mask)
    softmax_2d(ctx, logits, probs, b * H * s, s)
    av_matmul(ctx, probs, v_perm, attn_perm, b * H, s, s, Dh)
    reshape_bhsd_to_bsd(ctx, attn_perm, attn_flat, b, H, s, Dh)

    linear_forward(ctx, module.to_out, attn_flat, out_buf, b * s)


# ============================================================================
# Llama-style MLP (gate Linear + silu + up Linear + down Linear) — used by T3
# ============================================================================

@fieldwise_init
struct LlamaMLP(Copyable, Movable):
    """SwiGLU MLP: down(silu(gate(x)) * up(x))."""
    var gate: Linear   # (intermediate, d_model)
    var up:   Linear
    var down: Linear   # (d_model, intermediate)
    var d_model: Int
    var intermediate: Int


def llama_mlp_forward(
    mut ctx: DeviceContext,
    mut module: LlamaMLP,
    mut x_buf: DeviceBuffer[DType.float32],     # (M, d_model)
    mut out_buf: DeviceBuffer[DType.float32],   # (M, d_model)
    m: Int,
) raises:
    var inter = module.intermediate
    var gate_buf = ctx.enqueue_create_buffer[DType.float32](m * inter)
    var up_buf   = ctx.enqueue_create_buffer[DType.float32](m * inter)
    var act_buf  = ctx.enqueue_create_buffer[DType.float32](m * inter)
    var prod_buf = ctx.enqueue_create_buffer[DType.float32](m * inter)
    linear_forward(ctx, module.gate, x_buf, gate_buf, m)
    linear_forward(ctx, module.up,   x_buf, up_buf,   m)
    silu(ctx, gate_buf, act_buf, m * inter)

    # prod = act * up_buf — pure elementwise.
    var ap = act_buf.unsafe_ptr()
    var up_ = up_buf.unsafe_ptr()
    var pp = prod_buf.unsafe_ptr()
    var n = m * inter

    @always_inline
    @parameter
    @__copy_capture(ap, up_, pp)
    def mul_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var a = ap.load[width=width, alignment=alignment](i)
        var u = up_.load[width=width, alignment=alignment](i)
        pp.store[width=width, alignment=alignment](i, a * u)
    elementwise[mul_func, simd_width=4, target="gpu"](
        IndexList[1](n), DeviceContextPtr(ctx),
    )

    linear_forward(ctx, module.down, prod_buf, out_buf, m)


# ============================================================================
# Standard MLP (Linear + GELU + Linear) — used by s3tokenizer
# ============================================================================

@fieldwise_init
struct MLP(Copyable, Movable):
    var fc1: Linear   # (intermediate, d_model)
    var fc2: Linear   # (d_model, intermediate)
    var d_model: Int
    var intermediate: Int


def mlp_forward(
    mut ctx: DeviceContext,
    mut module: MLP,
    mut x_buf: DeviceBuffer[DType.float32],     # (M, d_model)
    mut out_buf: DeviceBuffer[DType.float32],   # (M, d_model)
    m: Int,
) raises:
    var inter = module.intermediate
    var fc1_buf = ctx.enqueue_create_buffer[DType.float32](m * inter)
    var gelu_buf = ctx.enqueue_create_buffer[DType.float32](m * inter)
    linear_forward(ctx, module.fc1, x_buf, fc1_buf, m)
    gelu(ctx, fc1_buf, gelu_buf, m * inter)
    linear_forward(ctx, module.fc2, gelu_buf, out_buf, m)
