"""CFM Euler ODE solver + ConditionalDecoder estimator (s3gen).

Pipeline:
  x = z_init    # gaussian noise (1, 80, T)
  t_span = linspace(0, 1, N_STEPS+1)
  for step in range(N_STEPS):
      packed = build_cfg_inputs(x, mask, spks, cond)
      v_t = estimator(packed, t_span[step])
      v_cfg = (1 + cfg_rate) * v_t[cond] - cfg_rate * v_t[uncond]
      x = x + (t_span[step+1] - t_span[step]) * v_cfg

The estimator is a U-Net with down/mid/up blocks and BasicTransformerBlocks.
Each component is composed from `Conv1d`, `Linear`, `RMSNorm`, attention helpers.
"""
from std.math import sqrt, sin, cos, pi
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from modules import Linear, linear_forward, LayerNorm, layer_norm_forward, silu, gelu, residual_add
from conv1d import Conv1d, conv1d_forward
from attention import qk_scaled_and_masked, softmax_2d, av_matmul
from transformer_blocks import reshape_bsd_to_bhsd, reshape_bhsd_to_bsd


# ============================================================================
# Sinusoidal time embedding
# ============================================================================

def sinusoidal_time_emb(
    mut ctx: DeviceContext,
    mut out_buf: DeviceBuffer[DType.float32],   # (B, dim)
    t: Float32,
    b: Int, dim: Int,
) raises:
    """Standard transformer sinusoidal positional embedding for scalar t."""
    var o_ptr = out_buf.unsafe_ptr()
    var half = dim // 2

    @always_inline
    @parameter
    @__copy_capture(o_ptr, t, half, dim, b)
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // dim
        var di = i - bi * dim
        var freq: Float32 = 0.0
        if di < half:
            freq = t / (10000.0 ** (Float32(di) / Float32(half)))
            o_ptr[i] = sin(freq)
        else:
            freq = t / (10000.0 ** (Float32(di - half) / Float32(half)))
            o_ptr[i] = cos(freq)
    elementwise[func, simd_width=1, target="gpu"](
        IndexList[1](b * dim), DeviceContextPtr(ctx),
    )


# ============================================================================
# Euler step helper: x_next = x + dt * v
# ============================================================================

def euler_step(
    mut ctx: DeviceContext,
    mut x_buf: DeviceBuffer[DType.float32],
    mut v_buf: DeviceBuffer[DType.float32],
    dt: Float32, n: Int,
) raises:
    """x += dt * v."""
    var x_ptr = x_buf.unsafe_ptr()
    var v_ptr = v_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(x_ptr, v_ptr, dt)
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        x_ptr[i] = x_ptr[i] + v_ptr[i] * dt
    elementwise[func, simd_width=4, target="gpu"](
        IndexList[1](n), DeviceContextPtr(ctx),
    )


# ============================================================================
# CFM ConditionalDecoder placeholder
#
# The actual upstream module has many nested blocks (down/mid/up + transformer
# + resnet1d). For brevity we provide a struct + forward declaration that
# delegates to `linalg.matmul`/`linalg.bmm`/`conv1d`/`attention` building
# blocks. A fuller per-block port can be added incrementally — the existing
# blocks already validate the building-block correctness end-to-end.
# ============================================================================

@fieldwise_init
struct CFMEstimator(Copyable, Movable):
    """The estimator wired as a sequence of Conv1d + transformer blocks."""
    var time_mlp: Linear        # (D, time_emb_dim)
    var time_mlp2: Linear       # (D, D)
    var in_conv: Conv1d
    var out_conv: Conv1d
    var d_model: Int
    var time_emb_dim: Int


def cfm_estimator_forward(
    mut ctx: DeviceContext,
    mut module: CFMEstimator,
    mut x_buf: DeviceBuffer[DType.float32],     # (B, C, T) — concat of [z, mu, mask, spks_broadcast]
    mut t_emb_buf: DeviceBuffer[DType.float32], # (B, time_emb_dim)
    mut out_buf: DeviceBuffer[DType.float32],   # (B, mel, T)
    b: Int, c_in: Int, c_out: Int, t: Int,
) raises:
    """Single forward call of the CFM estimator.

    The full upstream estimator has down-/mid-/up-blocks with transformer
    layers and resnet1d — those individual blocks are pure compositions of
    Conv1d / Linear / attention helpers and are added in `cfm_blocks.mojo`.

    This entry point handles the time-embedding projection + the in/out
    convolutions. A subsequent commit wires the full down/mid/up cascade.
    """
    var D = module.d_model
    # Time embedding projection (silu(Linear(t_emb)) → Linear).
    var t_proj1 = ctx.enqueue_create_buffer[DType.float32](b * D)
    linear_forward(ctx, module.time_mlp, t_emb_buf, t_proj1, b)
    var t_act = ctx.enqueue_create_buffer[DType.float32](b * D)
    silu(ctx, t_proj1, t_act, b * D)
    var t_proj2 = ctx.enqueue_create_buffer[DType.float32](b * D)
    linear_forward(ctx, module.time_mlp2, t_act, t_proj2, b)

    # in_conv + out_conv stitches.
    var h = ctx.enqueue_create_buffer[DType.float32](b * D * t)
    conv1d_forward(ctx, module.in_conv, x_buf, h, b, t, t)
    # (mid blocks deferred)
    conv1d_forward(ctx, module.out_conv, h, out_buf, b, t, t)


# ============================================================================
# CFM Euler solver (top-level)
# ============================================================================

def cfm_euler_solve(
    mut ctx: DeviceContext,
    mut estimator: CFMEstimator,
    mut z_buf: DeviceBuffer[DType.float32],     # (B, mel=80, T) — starts as gaussian noise
    mut packed_x_buf: DeviceBuffer[DType.float32], # (B*2, C_concat, T) scratch for CFG packed inputs
    mut v_buf: DeviceBuffer[DType.float32],     # (B*2, mel, T) — estimator output
    mut spks_buf: DeviceBuffer[DType.float32],  # (B, 80) speaker emb
    mut cond_buf: DeviceBuffer[DType.float32],  # (B, 80, T) cond
    mut mask_buf: DeviceBuffer[DType.float32],  # (B, 1, T) mask
    mut t_emb_buf: DeviceBuffer[DType.float32],
    b: Int, mel: Int, t: Int, c_concat: Int, n_steps: Int, cfg_rate: Float32,
    time_emb_dim: Int,
) raises:
    """Iterative Euler step. Each step:
      packed = [z | cond | mask | spks_broadcast]  (cond branch)
              + uncond version
      v = estimator(packed, t_step)
      v_cfg = (1+cfg_rate) * v_cond - cfg_rate * v_uncond
      z += dt * v_cfg
    """
    var step = 0
    while step < n_steps:
        # Time embedding for this step.
        var t_cur: Float32 = Float32(step) / Float32(n_steps)
        sinusoidal_time_emb(ctx, t_emb_buf, t_cur, b * 2, time_emb_dim)
        # (build packed_x_buf — interleaving omitted in this top-level skeleton)
        cfm_estimator_forward(
            ctx, estimator, packed_x_buf, t_emb_buf, v_buf,
            b * 2, c_concat, mel, t,
        )
        # CFG mix + step:
        var dt: Float32 = 1.0 / Float32(n_steps)
        # Simplified: just use cond branch (CFG mixing requires deinterleave).
        euler_step(ctx, z_buf, v_buf, dt, b * mel * t)
        step += 1
