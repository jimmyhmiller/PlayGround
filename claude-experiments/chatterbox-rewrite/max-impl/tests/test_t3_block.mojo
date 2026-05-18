"""Llama T3Block parity test reusing mojo-t3 block fixture.

Tests one full T3Block: RMSNorm + Q/K/V Linear + RoPE + SDPA + out Linear +
residual + RMSNorm + SwiGLU MLP + residual.

All ops route through `linalg.matmul`, `linalg.bmm`, `nn.softmax`,
`nn.normalization.rms_norm`, and `elementwise[..., target="gpu"]`.
"""
from std.math import sqrt
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from fixture import load_fp32
from modules import (
    Linear, linear_forward, RMSNorm, rms_norm_forward, silu, residual_add,
)
from attention import qk_scaled_and_masked, softmax_2d, av_matmul
from transformer_blocks import (
    reshape_bsd_to_bhsd, reshape_bhsd_to_bsd, apply_rope_hf_style,
)


comptime BATCH = 1
comptime SEQ = 16
comptime N_HEADS = 16
comptime HEAD_DIM = 64
comptime HIDDEN = N_HEADS * HEAD_DIM
comptime INTERMEDIATE = 4096


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def upload_w(mut ctx: DeviceContext, fix: String, name: String) raises -> DeviceBuffer[DType.float32]:
    var t = load_fp32(fix + name)
    var n = len(t.data)
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    upload(buf, t.data, n)
    return buf^


def upload_w_transpose(
    mut ctx: DeviceContext, fix: String, name: String, rows: Int, cols: Int,
) raises -> DeviceBuffer[DType.float32]:
    """Load (rows, cols) weight and store transposed as (cols, rows).

    Used because mojo-t3 fixtures pre-transpose Linear weights to (IN, OUT)
    for `out = x @ W` semantics, but our linear_forward uses HF convention
    `out = x @ W.T` which expects (OUT, IN).
    """
    var t = load_fp32(fix + name)
    var n = len(t.data)
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    with buf.map_to_host() as h:
        for r in range(rows):
            for c in range(cols):
                h[c * rows + r] = t.data[r * cols + c]
    return buf^


def test_t3_block() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "../mojo-t3/tests/fixtures/block/"
    var ctx = DeviceContext()

    var x_t = load_fp32(fix + "x_fp32.bin")
    var exp_t = load_fp32(fix + "expected_fp32.bin")
    var cos_t = load_fp32(fix + "cos_fp32.bin")
    var sin_t = load_fp32(fix + "sin_fp32.bin")
    var mask_t = load_fp32(fix + "mask_fp32.bin")

    var n_x = BATCH * SEQ * HIDDEN
    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    upload(x_buf, x_t.data, n_x)

    var cos_buf = ctx.enqueue_create_buffer[DType.float32](BATCH * SEQ * HEAD_DIM)
    upload(cos_buf, cos_t.data, BATCH * SEQ * HEAD_DIM)
    var sin_buf = ctx.enqueue_create_buffer[DType.float32](BATCH * SEQ * HEAD_DIM)
    upload(sin_buf, sin_t.data, BATCH * SEQ * HEAD_DIM)
    var mask_buf = ctx.enqueue_create_buffer[DType.float32](SEQ * SEQ)
    upload(mask_buf, mask_t.data, SEQ * SEQ)

    var in_norm_w = upload_w(ctx, fix, "in_norm_fp32.bin")
    var post_norm_w = upload_w(ctx, fix, "post_norm_fp32.bin")
    # mojo-t3 fixtures store all projection weights pre-transposed to (IN, OUT).
    # Re-transpose to (OUT, IN) so linear_forward's x @ W.T works.
    var qw = upload_w_transpose(ctx, fix, "qw_fp32.bin", HIDDEN, HIDDEN)
    var kw = upload_w_transpose(ctx, fix, "kw_fp32.bin", HIDDEN, HIDDEN)
    var vw = upload_w_transpose(ctx, fix, "vw_fp32.bin", HIDDEN, HIDDEN)
    var ow = upload_w_transpose(ctx, fix, "ow_fp32.bin", HIDDEN, HIDDEN)
    var gw = upload_w_transpose(ctx, fix, "gate_w_fp32.bin", HIDDEN, INTERMEDIATE)
    var uw = upload_w_transpose(ctx, fix, "up_w_fp32.bin", HIDDEN, INTERMEDIATE)
    var dw = upload_w_transpose(ctx, fix, "down_w_fp32.bin", INTERMEDIATE, HIDDEN)

    # Build modules — Llama style has no biases; create zero bias buffers.
    var zero_d = ctx.enqueue_create_buffer[DType.float32](HIDDEN)
    zero_d.enqueue_fill(0.0)
    var zero_inter = ctx.enqueue_create_buffer[DType.float32](INTERMEDIATE)
    zero_inter.enqueue_fill(0.0)
    var zero_d_2 = ctx.enqueue_create_buffer[DType.float32](HIDDEN)
    zero_d_2.enqueue_fill(0.0)

    var in_norm = RMSNorm(in_norm_w^, HIDDEN, Float32(1.0e-5))
    var post_norm = RMSNorm(post_norm_w^, HIDDEN, Float32(1.0e-5))
    var to_q = Linear(qw^, zero_d^, HIDDEN, HIDDEN, False)

    # ---- Self-attention ----
    var x_norm = ctx.enqueue_create_buffer[DType.float32](n_x)
    rms_norm_forward(ctx, in_norm, x_buf, x_norm, BATCH * SEQ)

    var q_lin = ctx.enqueue_create_buffer[DType.float32](n_x)
    var k_lin = ctx.enqueue_create_buffer[DType.float32](n_x)
    var v_lin = ctx.enqueue_create_buffer[DType.float32](n_x)
    linear_forward(ctx, to_q, x_norm, q_lin, BATCH * SEQ)
    var to_k = Linear(kw^, zero_d_2^, HIDDEN, HIDDEN, False)
    linear_forward(ctx, to_k, x_norm, k_lin, BATCH * SEQ)
    # Need a fresh zero bias for v.
    var zero_d_v = ctx.enqueue_create_buffer[DType.float32](HIDDEN)
    zero_d_v.enqueue_fill(0.0)
    var to_v = Linear(vw^, zero_d_v^, HIDDEN, HIDDEN, False)
    linear_forward(ctx, to_v, x_norm, v_lin, BATCH * SEQ)

    # Reshape (B, S, D) → (B, H, S, Dh) for RoPE: HF expects layout (B,H,S,D).
    var q_perm = ctx.enqueue_create_buffer[DType.float32](BATCH * N_HEADS * SEQ * HEAD_DIM)
    var k_perm = ctx.enqueue_create_buffer[DType.float32](BATCH * N_HEADS * SEQ * HEAD_DIM)
    var v_perm = ctx.enqueue_create_buffer[DType.float32](BATCH * N_HEADS * SEQ * HEAD_DIM)
    # First reshape (B, S, H*Dh) → (B, H, S, Dh) via reshape_bsd_to_bhsd.
    reshape_bsd_to_bhsd(ctx, q_lin, q_perm, BATCH, SEQ, N_HEADS, HEAD_DIM)
    reshape_bsd_to_bhsd(ctx, k_lin, k_perm, BATCH, SEQ, N_HEADS, HEAD_DIM)
    reshape_bsd_to_bhsd(ctx, v_lin, v_perm, BATCH, SEQ, N_HEADS, HEAD_DIM)

    var q_rope = ctx.enqueue_create_buffer[DType.float32](BATCH * N_HEADS * SEQ * HEAD_DIM)
    var k_rope = ctx.enqueue_create_buffer[DType.float32](BATCH * N_HEADS * SEQ * HEAD_DIM)
    apply_rope_hf_style(ctx, q_perm, q_rope, cos_buf, sin_buf, BATCH, N_HEADS, SEQ, HEAD_DIM)
    apply_rope_hf_style(ctx, k_perm, k_rope, cos_buf, sin_buf, BATCH, N_HEADS, SEQ, HEAD_DIM)

    # Attention.
    var scale: Float32 = 1.0 / sqrt(Float32(HEAD_DIM))
    var logits = ctx.enqueue_create_buffer[DType.float32](BATCH * N_HEADS * SEQ * SEQ)
    var probs  = ctx.enqueue_create_buffer[DType.float32](BATCH * N_HEADS * SEQ * SEQ)
    var av     = ctx.enqueue_create_buffer[DType.float32](BATCH * N_HEADS * SEQ * HEAD_DIM)
    qk_scaled_and_masked(ctx, q_rope, k_rope, mask_buf, logits,
                          BATCH * N_HEADS, SEQ, SEQ, HEAD_DIM, scale, True)
    softmax_2d(ctx, logits, probs, BATCH * N_HEADS * SEQ, SEQ)
    av_matmul(ctx, probs, v_perm, av, BATCH * N_HEADS, SEQ, SEQ, HEAD_DIM)

    var av_flat = ctx.enqueue_create_buffer[DType.float32](n_x)
    reshape_bhsd_to_bsd(ctx, av, av_flat, BATCH, N_HEADS, SEQ, HEAD_DIM)
    var zero_d_o = ctx.enqueue_create_buffer[DType.float32](HIDDEN)
    zero_d_o.enqueue_fill(0.0)
    var to_out = Linear(ow^, zero_d_o^, HIDDEN, HIDDEN, False)
    var attn_out = ctx.enqueue_create_buffer[DType.float32](n_x)
    linear_forward(ctx, to_out, av_flat, attn_out, BATCH * SEQ)
    residual_add(ctx, x_buf, attn_out, n_x)

    # ---- MLP ----
    var x_norm2 = ctx.enqueue_create_buffer[DType.float32](n_x)
    rms_norm_forward(ctx, post_norm, x_buf, x_norm2, BATCH * SEQ)

    var zero_inter1 = ctx.enqueue_create_buffer[DType.float32](INTERMEDIATE)
    zero_inter1.enqueue_fill(0.0)
    var zero_inter2 = ctx.enqueue_create_buffer[DType.float32](INTERMEDIATE)
    zero_inter2.enqueue_fill(0.0)
    var gate_lin = Linear(gw^, zero_inter1^, HIDDEN, INTERMEDIATE, False)
    var up_lin   = Linear(uw^, zero_inter2^, HIDDEN, INTERMEDIATE, False)
    var zero_d_d = ctx.enqueue_create_buffer[DType.float32](HIDDEN)
    zero_d_d.enqueue_fill(0.0)
    var down_lin = Linear(dw^, zero_d_d^, INTERMEDIATE, HIDDEN, False)

    var gate_h = ctx.enqueue_create_buffer[DType.float32](BATCH * SEQ * INTERMEDIATE)
    var up_h   = ctx.enqueue_create_buffer[DType.float32](BATCH * SEQ * INTERMEDIATE)
    var act_h  = ctx.enqueue_create_buffer[DType.float32](BATCH * SEQ * INTERMEDIATE)
    var prod_h = ctx.enqueue_create_buffer[DType.float32](BATCH * SEQ * INTERMEDIATE)
    linear_forward(ctx, gate_lin, x_norm2, gate_h, BATCH * SEQ)
    linear_forward(ctx, up_lin,   x_norm2, up_h,   BATCH * SEQ)
    silu(ctx, gate_h, act_h, BATCH * SEQ * INTERMEDIATE)

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
        IndexList[1](BATCH * SEQ * INTERMEDIATE), DeviceContextPtr(ctx),
    )

    var mlp_out = ctx.enqueue_create_buffer[DType.float32](n_x)
    linear_forward(ctx, down_lin, prod_h, mlp_out, BATCH * SEQ)
    residual_add(ctx, x_buf, mlp_out, n_x)
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    with x_buf.map_to_host() as h:
        for i in range(n_x):
            var d = h[i] - exp_t.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            if i < 8:
                print("tb[", i, "]: mojo=", h[i], "  torch=", exp_t.data[i], "  diff=", d)
            assert_almost_equal(h[i], exp_t.data[i], atol=2.0e-4)
    print("T3Block (B=1, S=16, H=16, Dh=64) — max abs:", max_abs)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
