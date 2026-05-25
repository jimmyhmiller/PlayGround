"""CFM Euler ODE solver + ConditionalDecoder estimator (s3gen).

Top-level CFM:
  x = z_init
  for step in range(N_STEPS):
      packed = build_cfg_packed_input(x, mask, spks, cond)   # double batch
      v_t    = estimator(packed, t_emb(t_step))
      v_cfg  = (1+cfg)*v_cond - cfg*v_uncond
      x     += dt * v_cfg

The estimator is a U-Net: in_conv → 3× down (resnet1d + transformer + down)
→ 1× mid (resnet1d × 2) → 3× up (upsample + skip-cat + resnet1d + transformer)
→ out_norm + silu + out_conv.

All composed from `cfm_blocks.causal_resnet_block_1d`,
`cfm_blocks.basic_transformer_forward`, plus `conv1d.conv1d_forward`.
"""
from std.math import sqrt, sin, cos, pi
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from modules import Linear, linear_forward, LayerNorm, layer_norm_forward, silu, residual_add
from conv1d import Conv1d, conv1d_forward
from cfm_blocks import (
    CausalResnetBlock1D, causal_resnet_block_1d,
    BasicTransformerBlock, basic_transformer_forward,
    transpose_bct_to_btc, transpose_btc_to_bct,
)


# ============================================================================
# Sinusoidal time embedding
# ============================================================================

def sinusoidal_time_emb(
    mut ctx: DeviceContext,
    mut out_buf: DeviceBuffer[DType.float32],
    t: Float32,
    b: Int, dim: Int,
) raises:
    var o_ptr = out_buf.unsafe_ptr()
    var half = dim // 2

    @always_inline
    @parameter
    @__copy_capture(o_ptr, t, half, dim)
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var di = i % dim
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


def euler_step(
    mut ctx: DeviceContext,
    mut x_buf: DeviceBuffer[DType.float32],
    mut v_buf: DeviceBuffer[DType.float32],
    dt: Float32, n: Int,
) raises:
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
# CFM Estimator: U-Net
# ============================================================================

@fieldwise_init
struct CFMEstimator(Copyable, Movable):
    """Full U-Net estimator with down/mid/up blocks.

    For brevity we instantiate a 3-stage encoder/decoder. Each list entry
    is one stage's components."""
    var time_mlp1: Linear
    var time_mlp2: Linear
    var in_conv: Conv1d
    var down_resnets: List[CausalResnetBlock1D]
    var down_transformers: List[BasicTransformerBlock]
    var down_samplers: List[Conv1d]              # stride-2 conv1d for downsampling
    var mid_resnets: List[CausalResnetBlock1D]    # typically 2
    var up_resnets: List[CausalResnetBlock1D]
    var up_transformers: List[BasicTransformerBlock]
    var up_samplers: List[Conv1d]                 # transposed conv (we use stride-1 + repeat)
    var out_ln: LayerNorm
    var out_conv: Conv1d
    var d_model: Int
    var time_emb_dim: Int


def cfm_estimator_forward(
    mut ctx: DeviceContext,
    mut module: CFMEstimator,
    mut x_buf: DeviceBuffer[DType.float32],     # (B, C_in, T)
    mut t_emb_buf: DeviceBuffer[DType.float32], # (B, time_emb_dim)
    mut out_buf: DeviceBuffer[DType.float32],   # (B, C_out, T)
    b: Int, c_in: Int, c_out: Int, t: Int,
) raises:
    """U-Net forward: in_conv → down → mid → up → out_conv."""
    var D = module.d_model
    # Project time embedding.
    var tp1 = ctx.enqueue_create_buffer[DType.float32](b * D)
    linear_forward(ctx, module.time_mlp1, t_emb_buf, tp1, b)
    var tp1_act = ctx.enqueue_create_buffer[DType.float32](b * D)
    silu(ctx, tp1, tp1_act, b * D)
    var t_proj = ctx.enqueue_create_buffer[DType.float32](b * D)
    linear_forward(ctx, module.time_mlp2, tp1_act, t_proj, b)

    # In conv.
    var h = ctx.enqueue_create_buffer[DType.float32](b * D * t)
    conv1d_forward(ctx, module.in_conv, x_buf, h, b, t, t)

    # Encoder stages — each: resnet1d + transformer + downsample.
    var skip_features = List[DeviceBuffer[DType.float32]]()
    var cur_t = t
    for i in range(len(module.down_resnets)):
        var rb_out = ctx.enqueue_create_buffer[DType.float32](b * D * cur_t)
        causal_resnet_block_1d(ctx, module.down_resnets[i], h, t_proj, rb_out, b, cur_t)
        # transformer expects (B, T, D)
        var tb_btc = ctx.enqueue_create_buffer[DType.float32](b * cur_t * D)
        transpose_bct_to_btc(ctx, rb_out, tb_btc, b, D, cur_t)
        basic_transformer_forward(ctx, module.down_transformers[i], tb_btc, b, cur_t)
        var tb_bct = ctx.enqueue_create_buffer[DType.float32](b * D * cur_t)
        transpose_btc_to_bct(ctx, tb_btc, tb_bct, b, cur_t, D)
        # Save skip then downsample.
        skip_features.append(tb_bct)
        # Downsample via stride-2 conv.
        var new_t = cur_t // 2
        var ds = ctx.enqueue_create_buffer[DType.float32](b * D * new_t)
        conv1d_forward(ctx, module.down_samplers[i], tb_bct, ds, b, cur_t, new_t)
        ctx.enqueue_copy(h, ds)
        cur_t = new_t

    # Middle resnets.
    for i in range(len(module.mid_resnets)):
        var mr_out = ctx.enqueue_create_buffer[DType.float32](b * D * cur_t)
        causal_resnet_block_1d(ctx, module.mid_resnets[i], h, t_proj, mr_out, b, cur_t)
        ctx.enqueue_copy(h, mr_out)

    # Decoder stages: upsample + skip cat + resnet1d + transformer.
    var n_stages = len(module.down_resnets)
    for i in range(n_stages):
        # Upsample (nearest doubling via Conv1d-with-stride=1 and our scale).
        var new_t = cur_t * 2
        var us = ctx.enqueue_create_buffer[DType.float32](b * D * new_t)
        # Nearest-neighbor upsample via elementwise replication.
        var hp = h.unsafe_ptr()
        var up = us.unsafe_ptr()

        @always_inline
        @parameter
        @__copy_capture(hp, up, b, D, cur_t)
        def upn_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
            var ii = idx[0]
            var bi = ii // (D * cur_t * 2)
            var rem = ii - bi * D * cur_t * 2
            var di = rem // (cur_t * 2)
            var ti = (rem - di * cur_t * 2) // 2
            up[ii] = hp[bi * D * cur_t + di * cur_t + ti]
        elementwise[upn_func, simd_width=1, target="gpu"](
            IndexList[1](b * D * new_t), DeviceContextPtr(ctx),
        )
        # Add skip — concat-style via add (simplification — upstream cats then resnet).
        var skip = skip_features[n_stages - 1 - i]
        residual_add(ctx, us, skip, b * D * new_t)

        var rb_out = ctx.enqueue_create_buffer[DType.float32](b * D * new_t)
        causal_resnet_block_1d(ctx, module.up_resnets[i], us, t_proj, rb_out, b, new_t)
        var tb_btc = ctx.enqueue_create_buffer[DType.float32](b * new_t * D)
        transpose_bct_to_btc(ctx, rb_out, tb_btc, b, D, new_t)
        basic_transformer_forward(ctx, module.up_transformers[i], tb_btc, b, new_t)
        var tb_bct = ctx.enqueue_create_buffer[DType.float32](b * D * new_t)
        transpose_btc_to_bct(ctx, tb_btc, tb_bct, b, new_t, D)
        ctx.enqueue_copy(h, tb_bct)
        cur_t = new_t

    # Out norm + silu + out_conv.
    var h_btc = ctx.enqueue_create_buffer[DType.float32](b * cur_t * D)
    transpose_bct_to_btc(ctx, h, h_btc, b, D, cur_t)
    var h_ln = ctx.enqueue_create_buffer[DType.float32](b * cur_t * D)
    layer_norm_forward(ctx, module.out_ln, h_btc, h_ln, b * cur_t)
    var h_bct = ctx.enqueue_create_buffer[DType.float32](b * D * cur_t)
    transpose_btc_to_bct(ctx, h_ln, h_bct, b, cur_t, D)
    var h_act = ctx.enqueue_create_buffer[DType.float32](b * D * cur_t)
    silu(ctx, h_bct, h_act, b * D * cur_t)
    conv1d_forward(ctx, module.out_conv, h_act, out_buf, b, cur_t, cur_t)


# ============================================================================
# Top-level Euler solver
# ============================================================================

def build_cfg_packed(
    mut ctx: DeviceContext,
    mut z_buf: DeviceBuffer[DType.float32],     # (B, mel, T)
    mut cond_buf: DeviceBuffer[DType.float32],  # (B, mel, T)
    mut mask_buf: DeviceBuffer[DType.float32],  # (B, 1, T)
    mut spks_buf: DeviceBuffer[DType.float32],  # (B, mel)
    mut packed_buf: DeviceBuffer[DType.float32], # (B*2, mel+mel+1+mel, T)
    b: Int, mel: Int, t: Int,
) raises:
    """Build the packed [cond | uncond] input for classifier-free guidance.

    The packed channel dim is 2*mel+1+mel = 3*mel+1 (rows: z, cond, mask, spks-broadcast).
    For uncond we zero the cond region.
    """
    var c_concat = 3 * mel + 1
    var z_ptr = z_buf.unsafe_ptr()
    var c_ptr = cond_buf.unsafe_ptr()
    var m_ptr = mask_buf.unsafe_ptr()
    var s_ptr = spks_buf.unsafe_ptr()
    var p_ptr = packed_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(z_ptr, c_ptr, m_ptr, s_ptr, p_ptr, b, mel, t, c_concat)
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (c_concat * t)
        var rem = i - bi * c_concat * t
        var ci = rem // t
        var ti = rem - ci * t
        var b_orig = bi % b
        var is_cond = bi < b   # first half is cond, second half is uncond

        if ci < mel:
            # z channel — shared between cond and uncond.
            p_ptr[i] = z_ptr[b_orig * mel * t + ci * t + ti]
        elif ci < 2 * mel:
            # cond channel — only present for cond branch.
            var off = ci - mel
            if is_cond:
                p_ptr[i] = c_ptr[b_orig * mel * t + off * t + ti]
            else:
                p_ptr[i] = 0.0
        elif ci == 2 * mel:
            # mask.
            p_ptr[i] = m_ptr[b_orig * t + ti]
        else:
            # spks broadcasted across T.
            var off = ci - 2 * mel - 1
            p_ptr[i] = s_ptr[b_orig * mel + off]
    elementwise[func, simd_width=1, target="gpu"](
        IndexList[1](2 * b * c_concat * t), DeviceContextPtr(ctx),
    )


def cfg_mix(
    mut ctx: DeviceContext,
    mut v_buf: DeviceBuffer[DType.float32],     # (2*B, mel, T) — output of estimator on packed
    mut v_cfg_buf: DeviceBuffer[DType.float32], # (B, mel, T) — output
    b: Int, mel: Int, t: Int, cfg_rate: Float32,
) raises:
    """v_cfg = (1+cfg)*v_cond - cfg*v_uncond. cond is first B, uncond is last B."""
    var v_ptr = v_buf.unsafe_ptr()
    var o_ptr = v_cfg_buf.unsafe_ptr()
    var stride = mel * t

    @always_inline
    @parameter
    @__copy_capture(v_ptr, o_ptr, b, stride, cfg_rate)
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // stride
        var pos = i - bi * stride
        var v_cond = v_ptr[bi * stride + pos]
        var v_uncond = v_ptr[(bi + b) * stride + pos]
        o_ptr[i] = (1.0 + cfg_rate) * v_cond - cfg_rate * v_uncond
    elementwise[func, simd_width=4, target="gpu"](
        IndexList[1](b * stride), DeviceContextPtr(ctx),
    )


def cfm_euler_solve(
    mut ctx: DeviceContext,
    mut estimator: CFMEstimator,
    mut z_buf: DeviceBuffer[DType.float32],
    mut spks_buf: DeviceBuffer[DType.float32],
    mut cond_buf: DeviceBuffer[DType.float32],
    mut mask_buf: DeviceBuffer[DType.float32],
    mut packed_buf: DeviceBuffer[DType.float32],
    mut v_buf: DeviceBuffer[DType.float32],
    mut v_cfg_buf: DeviceBuffer[DType.float32],
    mut t_emb_buf: DeviceBuffer[DType.float32],
    b: Int, mel: Int, t: Int, c_concat: Int, n_steps: Int, cfg_rate: Float32,
    time_emb_dim: Int,
) raises:
    var step = 0
    while step < n_steps:
        build_cfg_packed(ctx, z_buf, cond_buf, mask_buf, spks_buf, packed_buf, b, mel, t)
        var t_cur: Float32 = Float32(step) / Float32(n_steps)
        sinusoidal_time_emb(ctx, t_emb_buf, t_cur, b * 2, time_emb_dim)
        cfm_estimator_forward(
            ctx, estimator, packed_buf, t_emb_buf, v_buf,
            b * 2, c_concat, mel, t,
        )
        cfg_mix(ctx, v_buf, v_cfg_buf, b, mel, t, cfg_rate)
        var dt: Float32 = 1.0 / Float32(n_steps)
        euler_step(ctx, z_buf, v_cfg_buf, dt, b * mel * t)
        step += 1
