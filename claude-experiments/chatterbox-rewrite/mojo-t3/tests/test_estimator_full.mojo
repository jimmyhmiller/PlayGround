"""
Parity test for the FULL ConditionalDecoder estimator forward pass.

Input:  cfm_step_00_input_x.bin    (2, 80, 752)
        cfm_step_00_input_mu.bin   (2, 80, 752)
        cfm_step_00_input_spks.bin (2, 80)
        cfm_step_00_input_cond.bin (2, 80, 752)
        cfm_step_00_input_mask.bin (2, 1, 752)
        estimator_time_mlp_out_real.bin (2, 1024)  [precomputed time_mlp output]
Target: estimator_final_full.bin   (2, 80, 752)
"""
from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu import block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major

from fixture import load_fp32
from conv import conv1d_kernel_fast
from layernorm import transpose_btc_to_bct_kernel, transpose_bct_to_btc_kernel
from attention import add_4d_kernel
from cfm_decoder import (
    causal_resnet_block_1d, CausalResnetWeights,
    basic_transformer_block, BasicTransformerWeights,
    causal_conv1d_with_mask, causal_block_1d,
    transpose_with_mask_bct_btc, transpose_with_mask_btc_bct,
)


comptime B = 2
comptime IN_C = 320       # packed [x:80, mu:80, spks_expand:80, cond:80] = 320
comptime D = 256
comptime D_OUT_MEL = 80
comptime T = 752
comptime TIME_EMB_DIM = 1024
comptime H = 8
comptime D_K = 64
comptime FF_INNER = 1024
comptime BLOCK = 256


def pack_xmsc_kernel[
    dtype: DType, OutLayout: TensorLayout,
    XLayout: TensorLayout, MuLayout: TensorLayout,
    SpksLayout: TensorLayout, CondLayout: TensorLayout,
    BLOCK_: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, 320, T)
    x_in: TileTensor[dtype, XLayout, MutAnyOrigin],        # (B, 80, T)
    mu_in: TileTensor[dtype, MuLayout, MutAnyOrigin],      # (B, 80, T)
    spks_in: TileTensor[dtype, SpksLayout, MutAnyOrigin],  # (B, 80)
    cond_in: TileTensor[dtype, CondLayout, MutAnyOrigin],  # (B, 80, T)
    batch: Int, time: Int,
):
    """Pack [x, mu, spks_expand, cond] along channel dim to produce (B, 320, T).
    Launch: grid = B*320, block_dim = BLOCK_ over T.
    """
    comptime assert x_in.flat_rank == 3
    comptime assert mu_in.flat_rank == 3
    comptime assert spks_in.flat_rank == 2
    comptime assert cond_in.flat_rank == 3
    comptime assert output.flat_rank == 3
    var bid = block_idx.x
    var tid = thread_idx.x
    var c = bid % 320
    var b = bid // 320
    var t = tid
    while t < time:
        var v: Float32 = 0.0
        if c < 80:
            v = rebind[Scalar[dtype]](x_in[b, c, t]).cast[DType.float32]()
        elif c < 160:
            v = rebind[Scalar[dtype]](mu_in[b, c - 80, t]).cast[DType.float32]()
        elif c < 240:
            v = rebind[Scalar[dtype]](spks_in[b, c - 160]).cast[DType.float32]()
        else:
            v = rebind[Scalar[dtype]](cond_in[b, c - 240, t]).cast[DType.float32]()
        output[b, c, t] = rebind[output.ElementType](v.cast[dtype]())
        t += BLOCK_


def channel_concat_3d_kernel[
    dtype: DType, OutLayout: TensorLayout, ALayout: TensorLayout, BLayout: TensorLayout,
    C_A: Int, C_B: Int, BLOCK_: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, C_A + C_B, T)
    a_in: TileTensor[dtype, ALayout, MutAnyOrigin],        # (B, C_A, T)
    b_in: TileTensor[dtype, BLayout, MutAnyOrigin],        # (B, C_B, T)
    batch: Int, time: Int,
):
    """out[:, :C_A, :] = a; out[:, C_A:, :] = b."""
    comptime assert a_in.flat_rank == 3
    comptime assert b_in.flat_rank == 3
    comptime assert output.flat_rank == 3
    var bid = block_idx.x
    var tid = thread_idx.x
    var c = bid % (C_A + C_B)
    var b = bid // (C_A + C_B)
    var t = tid
    while t < time:
        var v: Float32 = 0.0
        if c < C_A:
            v = rebind[Scalar[dtype]](a_in[b, c, t]).cast[DType.float32]()
        else:
            v = rebind[Scalar[dtype]](b_in[b, c - C_A, t]).cast[DType.float32]()
        output[b, c, t] = rebind[output.ElementType](v.cast[dtype]())
        t += BLOCK_


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


def load_resnet(mut ctx: DeviceContext, fix: String, prefix: String) raises -> CausalResnetWeights:
    return CausalResnetWeights(
        upload_w(ctx, fix, prefix + "block1__block__0__weight.bin"),
        upload_w(ctx, fix, prefix + "block1__block__0__bias.bin"),
        upload_w(ctx, fix, prefix + "block1__block__2__weight.bin"),
        upload_w(ctx, fix, prefix + "block1__block__2__bias.bin"),
        upload_w(ctx, fix, prefix + "block2__block__0__weight.bin"),
        upload_w(ctx, fix, prefix + "block2__block__0__bias.bin"),
        upload_w(ctx, fix, prefix + "block2__block__2__weight.bin"),
        upload_w(ctx, fix, prefix + "block2__block__2__bias.bin"),
        upload_w(ctx, fix, prefix + "mlp__1__weight.bin"),
        upload_w(ctx, fix, prefix + "mlp__1__bias.bin"),
        upload_w(ctx, fix, prefix + "res_conv__weight.bin"),
        upload_w(ctx, fix, prefix + "res_conv__bias.bin"),
    )


def load_tblock(mut ctx: DeviceContext, fix: String, prefix: String) raises -> BasicTransformerWeights:
    return BasicTransformerWeights(
        upload_w(ctx, fix, prefix + "norm1__weight.bin"),
        upload_w(ctx, fix, prefix + "norm1__bias.bin"),
        upload_w(ctx, fix, prefix + "attn1__to_q__weight.bin"),
        upload_w(ctx, fix, prefix + "attn1__to_k__weight.bin"),
        upload_w(ctx, fix, prefix + "attn1__to_v__weight.bin"),
        upload_w(ctx, fix, prefix + "attn1__to_out__0__weight.bin"),
        upload_w(ctx, fix, prefix + "attn1__to_out__0__bias.bin"),
        upload_w(ctx, fix, prefix + "norm3__weight.bin"),
        upload_w(ctx, fix, prefix + "norm3__bias.bin"),
        upload_w(ctx, fix, prefix + "ff__net__0__proj__weight.bin"),
        upload_w(ctx, fix, prefix + "ff__net__0__proj__bias.bin"),
        upload_w(ctx, fix, prefix + "ff__net__2__weight.bin"),
        upload_w(ctx, fix, prefix + "ff__net__2__bias.bin"),
    )


def run_block_with_4_tblocks[
    B_: Int, IN_C_: Int, T_: Int, D_: Int, H_: Int, D_K_: Int, FF_INNER_: Int, TIME_EMB_DIM_: Int,
](
    mut ctx: DeviceContext,
    mut x_in: DeviceBuffer[DType.float32],          # (B, IN_C_, T)
    mut mask: DeviceBuffer[DType.float32],          # (B, 1, T)
    mut t_emb: DeviceBuffer[DType.float32],         # (B, TIME_EMB_DIM_)
    mut out_buf: DeviceBuffer[DType.float32],       # (B, D_, T)
    mut rn: CausalResnetWeights,
    mut tb0: BasicTransformerWeights,
    mut tb1: BasicTransformerWeights,
    mut tb2: BasicTransformerWeights,
    mut tb3: BasicTransformerWeights,
) raises:
    """Resnet + 4 transformer blocks. Output is (B, D_, T) (bct)."""
    var n_d = B_ * D_ * T_

    var resnet_out = ctx.enqueue_create_buffer[DType.float32](n_d)
    var btc_buf = ctx.enqueue_create_buffer[DType.float32](n_d)
    var t0 = ctx.enqueue_create_buffer[DType.float32](n_d)
    var t1 = ctx.enqueue_create_buffer[DType.float32](n_d)
    var t2 = ctx.enqueue_create_buffer[DType.float32](n_d)
    var t3 = ctx.enqueue_create_buffer[DType.float32](n_d)

    causal_resnet_block_1d[B_, IN_C_, D_, T_, TIME_EMB_DIM_](
        ctx, x_in, mask, t_emb, resnet_out, rn,
    )
    transpose_with_mask_bct_btc[B_, D_, T_](ctx, resnet_out, btc_buf)
    basic_transformer_block[B_, T_, D_, H_, D_K_, FF_INNER_](ctx, btc_buf, t0, tb0)
    basic_transformer_block[B_, T_, D_, H_, D_K_, FF_INNER_](ctx, t0, t1, tb1)
    basic_transformer_block[B_, T_, D_, H_, D_K_, FF_INNER_](ctx, t1, t2, tb2)
    basic_transformer_block[B_, T_, D_, H_, D_K_, FF_INNER_](ctx, t2, t3, tb3)
    transpose_with_mask_btc_bct[B_, T_, D_](ctx, t3, out_buf)


def test_full_estimator() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/s3gen/"
    var ctx = DeviceContext()

    var x_in = load_fp32(fix + "cfm_step_00_input_x.bin")
    var mu_in = load_fp32(fix + "cfm_step_00_input_mu.bin")
    var spks_in = load_fp32(fix + "cfm_step_00_input_spks.bin")
    var cond_in = load_fp32(fix + "cfm_step_00_input_cond.bin")
    var mask_in = load_fp32(fix + "cfm_step_00_input_mask.bin")
    var t_emb = load_fp32(fix + "estimator_time_mlp_out_real.bin")
    var exp = load_fp32(fix + "estimator_final_full.bin")

    var n_x = B * 80 * T
    var n_xfull = B * IN_C * T
    var n_mask = B * 1 * T
    var n_spks = B * 80
    var n_te = B * TIME_EMB_DIM
    var n_d = B * D * T
    var n_out = B * D_OUT_MEL * T
    var n_skip_cat = B * (D + D) * T   # for up_block input: cat([mid_out, skip_from_down])

    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var mu_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var spks_buf = ctx.enqueue_create_buffer[DType.float32](n_spks)
    var cond_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var mask_buf = ctx.enqueue_create_buffer[DType.float32](n_mask)
    var t_emb_buf = ctx.enqueue_create_buffer[DType.float32](n_te)
    var x_full = ctx.enqueue_create_buffer[DType.float32](n_xfull)
    upload(x_buf, x_in.data, n_x)
    upload(mu_buf, mu_in.data, n_x)
    upload(spks_buf, spks_in.data, n_spks)
    upload(cond_buf, cond_in.data, n_x)
    upload(mask_buf, mask_in.data, n_mask)
    upload(t_emb_buf, t_emb.data, n_te)

    comptime x_layout = row_major[B, 80, T]()
    comptime spks_layout = row_major[B, 80]()
    comptime mask_layout = row_major[B, 1, T]()
    comptime xfull_layout = row_major[B, IN_C, T]()

    var x_t = TileTensor(x_buf, x_layout)
    var mu_t = TileTensor(mu_buf, x_layout)
    var spks_t = TileTensor(spks_buf, spks_layout)
    var cond_t = TileTensor(cond_buf, x_layout)
    var x_full_t = TileTensor(x_full, xfull_layout)

    # 1. Pack [x, mu, spks_expand, cond] -> (B, 320, T).
    comptime pack_k = pack_xmsc_kernel[
        DType.float32, type_of(xfull_layout),
        type_of(x_layout), type_of(x_layout),
        type_of(spks_layout), type_of(x_layout),
        BLOCK,
    ]
    ctx.enqueue_function[pack_k, pack_k](
        x_full_t, x_t, mu_t, spks_t, cond_t, B, T,
        grid_dim=B * IN_C, block_dim=BLOCK,
    )

    # 2. down_block 0.
    var rn0 = load_resnet(ctx, fix, "weights/flow__decoder__estimator__down_blocks__0__0__")
    var dn_tb0 = load_tblock(ctx, fix, "weights/flow__decoder__estimator__down_blocks__0__1__0__")
    var dn_tb1 = load_tblock(ctx, fix, "weights/flow__decoder__estimator__down_blocks__0__1__1__")
    var dn_tb2 = load_tblock(ctx, fix, "weights/flow__decoder__estimator__down_blocks__0__1__2__")
    var dn_tb3 = load_tblock(ctx, fix, "weights/flow__decoder__estimator__down_blocks__0__1__3__")
    var dn_ds_w = upload_w(ctx, fix, "weights/flow__decoder__estimator__down_blocks__0__2__weight.bin")
    var dn_ds_b = upload_w(ctx, fix, "weights/flow__decoder__estimator__down_blocks__0__2__bias.bin")

    var pre_ds = ctx.enqueue_create_buffer[DType.float32](n_d)
    var skip0 = ctx.enqueue_create_buffer[DType.float32](n_d)
    var post_ds = ctx.enqueue_create_buffer[DType.float32](n_d)
    run_block_with_4_tblocks[B, IN_C, T, D, H, D_K, FF_INNER, TIME_EMB_DIM](
        ctx, x_full, mask_buf, t_emb_buf, pre_ds, rn0, dn_tb0, dn_tb1, dn_tb2, dn_tb3,
    )
    # Save skip0 = pre_ds (the output of transformers, pre-downsample).
    # We need to copy. Use a small copy kernel.
    # Actually: skip0 = pre_ds — both have the same data. Just don't reuse pre_ds, downsample reads from it.
    causal_conv1d_with_mask[B, D, D, T, 3](
        ctx, pre_ds, mask_buf, post_ds, dn_ds_w, dn_ds_b,
    )

    # 3. mid_blocks (12 iterations of resnet + 4 transformers).
    # Each iteration takes (B, D, T) and produces (B, D, T).
    var mid_in = post_ds   # rotating reference
    var mid_buf_a = ctx.enqueue_create_buffer[DType.float32](n_d)
    var mid_buf_b = ctx.enqueue_create_buffer[DType.float32](n_d)

    @parameter
    def run_mid[I: Int](
        mut ctx_: DeviceContext,
        mut src: DeviceBuffer[DType.float32],
        mut dst: DeviceBuffer[DType.float32],
    ) raises:
        var rn = load_resnet(ctx_, fix, "weights/flow__decoder__estimator__mid_blocks__" + String(I) + "__0__")
        var tb0 = load_tblock(ctx_, fix, "weights/flow__decoder__estimator__mid_blocks__" + String(I) + "__1__0__")
        var tb1 = load_tblock(ctx_, fix, "weights/flow__decoder__estimator__mid_blocks__" + String(I) + "__1__1__")
        var tb2 = load_tblock(ctx_, fix, "weights/flow__decoder__estimator__mid_blocks__" + String(I) + "__1__2__")
        var tb3 = load_tblock(ctx_, fix, "weights/flow__decoder__estimator__mid_blocks__" + String(I) + "__1__3__")
        run_block_with_4_tblocks[B, D, T, D, H, D_K, FF_INNER, TIME_EMB_DIM](
            ctx_, src, mask_buf, t_emb_buf, dst, rn, tb0, tb1, tb2, tb3,
        )

    run_mid[0](ctx, post_ds, mid_buf_a)
    run_mid[1](ctx, mid_buf_a, mid_buf_b)
    run_mid[2](ctx, mid_buf_b, mid_buf_a)
    run_mid[3](ctx, mid_buf_a, mid_buf_b)
    run_mid[4](ctx, mid_buf_b, mid_buf_a)
    run_mid[5](ctx, mid_buf_a, mid_buf_b)
    run_mid[6](ctx, mid_buf_b, mid_buf_a)
    run_mid[7](ctx, mid_buf_a, mid_buf_b)
    run_mid[8](ctx, mid_buf_b, mid_buf_a)
    run_mid[9](ctx, mid_buf_a, mid_buf_b)
    run_mid[10](ctx, mid_buf_b, mid_buf_a)
    run_mid[11](ctx, mid_buf_a, mid_buf_b)
    # After 12 mids, output is in mid_buf_b.

    # 4. up_block 0: skip cat [mid_buf_b, skip0=pre_ds] -> (B, 2*D, T), then resnet -> 4 tblocks -> CausalConv1d.
    var up_in = ctx.enqueue_create_buffer[DType.float32](n_skip_cat)
    var up_pre_us = ctx.enqueue_create_buffer[DType.float32](n_d)
    var up_post = ctx.enqueue_create_buffer[DType.float32](n_d)

    comptime mid_layout = row_major[B, D, T]()
    comptime skip_cat_layout = row_major[B, 2 * D, T]()
    var mid_b_t = TileTensor(mid_buf_b, mid_layout)
    var skip0_t = TileTensor(pre_ds, mid_layout)
    var up_in_t = TileTensor(up_in, skip_cat_layout)
    comptime cat_k = channel_concat_3d_kernel[
        DType.float32, type_of(skip_cat_layout),
        type_of(mid_layout), type_of(mid_layout),
        D, D, BLOCK,
    ]
    ctx.enqueue_function[cat_k, cat_k](
        up_in_t, mid_b_t, skip0_t, B, T,
        grid_dim=B * (2 * D), block_dim=BLOCK,
    )

    var up_rn = load_resnet(ctx, fix, "weights/flow__decoder__estimator__up_blocks__0__0__")
    var up_tb0 = load_tblock(ctx, fix, "weights/flow__decoder__estimator__up_blocks__0__1__0__")
    var up_tb1 = load_tblock(ctx, fix, "weights/flow__decoder__estimator__up_blocks__0__1__1__")
    var up_tb2 = load_tblock(ctx, fix, "weights/flow__decoder__estimator__up_blocks__0__1__2__")
    var up_tb3 = load_tblock(ctx, fix, "weights/flow__decoder__estimator__up_blocks__0__1__3__")
    var up_us_w = upload_w(ctx, fix, "weights/flow__decoder__estimator__up_blocks__0__2__weight.bin")
    var up_us_b = upload_w(ctx, fix, "weights/flow__decoder__estimator__up_blocks__0__2__bias.bin")

    run_block_with_4_tblocks[B, 2 * D, T, D, H, D_K, FF_INNER, TIME_EMB_DIM](
        ctx, up_in, mask_buf, t_emb_buf, up_pre_us, up_rn, up_tb0, up_tb1, up_tb2, up_tb3,
    )
    causal_conv1d_with_mask[B, D, D, T, 3](
        ctx, up_pre_us, mask_buf, up_post, up_us_w, up_us_b,
    )

    # 5. final_block (CausalBlock1D): conv1d k=3 + LN + Mish, multiplied by mask.
    var fb_cw = upload_w(ctx, fix, "weights/flow__decoder__estimator__final_block__block__0__weight.bin")
    var fb_cb = upload_w(ctx, fix, "weights/flow__decoder__estimator__final_block__block__0__bias.bin")
    var fb_lw = upload_w(ctx, fix, "weights/flow__decoder__estimator__final_block__block__2__weight.bin")
    var fb_lb = upload_w(ctx, fix, "weights/flow__decoder__estimator__final_block__block__2__bias.bin")
    var fb_out = ctx.enqueue_create_buffer[DType.float32](n_d)
    causal_block_1d[B, D, D, T, 3](
        ctx, up_post, mask_buf, fb_out, fb_cw, fb_cb, fb_lw, fb_lb,
    )

    # 6. final_proj (Conv1d 1x1, 256 -> 80) on (x * mask).
    var fp_w = upload_w(ctx, fix, "weights/flow__decoder__estimator__final_proj__weight.bin")
    var fp_b = upload_w(ctx, fix, "weights/flow__decoder__estimator__final_proj__bias.bin")
    var fp_masked = ctx.enqueue_create_buffer[DType.float32](n_d)
    var fp_out = ctx.enqueue_create_buffer[DType.float32](n_out)
    var final_out = ctx.enqueue_create_buffer[DType.float32](n_out)

    causal_conv1d_with_mask[B, D, D_OUT_MEL, T, 1](
        ctx, fb_out, mask_buf, fp_out, fp_w, fp_b,
    )
    # NOTE: the actual code does `self.final_proj(x * mask)` then `* mask` again at the end.
    # Our causal_conv1d_with_mask pre-multiplies by mask, but final_proj is k=1 so causal_pad=0 (no shift).
    # That gives the right output for the first `final_proj(x * mask)` part. Now apply the trailing mask.
    comptime final_layout = row_major[B, D_OUT_MEL, T]()
    var fp_out_t = TileTensor(fp_out, final_layout)
    var final_out_t = TileTensor(final_out, final_layout)
    var mask_t = TileTensor(mask_buf, mask_layout)
    from decoder_kernels import multiply_mask_3d_kernel
    comptime mul_final_k = multiply_mask_3d_kernel[
        DType.float32, type_of(final_layout), type_of(mask_layout),
        type_of(final_layout), BLOCK,
    ]
    ctx.enqueue_function[mul_final_k, mul_final_k](
        final_out_t, fp_out_t, mask_t, B, D_OUT_MEL, T,
        grid_dim=B * D_OUT_MEL, block_dim=BLOCK,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    var sum_abs: Float64 = 0.0
    with final_out.map_to_host() as h:
        for i in range(n_out):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_abs += Float64(d)
            if i < 8:
                print("fe[", i, "]: mojo=", h[i], "  torch=", exp.data[i], "  diff=", d)
            assert_almost_equal(h[i], exp.data[i], atol=2.0e-1)
    print("FULL ConditionalDecoder estimator — max abs:", max_abs,
          " mean:", sum_abs / Float64(n_out))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
