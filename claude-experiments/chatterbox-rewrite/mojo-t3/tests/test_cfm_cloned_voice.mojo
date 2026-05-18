"""
FULL CFM solve: runs 10 Euler steps of the ConditionalCFM with the pure-Mojo
estimator and produces the final mel.

Input:  cfm_z_init.bin, cfm_t_span.bin, cfm_mu.bin, cfm_mask.bin, cfm_spks.bin, cfm_cond.bin
Target: cfm_mel_out.bin   (1, 80, 752)
"""
from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import load_fp32
from layernorm import linear_kernel
from attention import swish_kernel
from decoder_kernels import sinusoidal_pos_emb_kernel
from cfm_decoder import (
    estimator_forward,
    CausalResnetWeights, BasicTransformerWeights,
)
from cfm_solver import (
    cfm_euler_step_kernel, build_cfg_inputs_kernel, build_cfg_inputs_2d_kernel,
    pack_xmsc_kernel,
)


comptime B = 1
comptime B2 = 2
comptime MEL_C = 80
comptime SPKS_C = 80
comptime T = 762
comptime PACKED_C = 320
comptime TIME_EMB_DIM = 1024
comptime IN_DIM_TE = 320
comptime D = 256
comptime H = 8
comptime D_K = 64
comptime FF_INNER = 1024
comptime N_STEPS = 10
comptime CFG_RATE: Float32 = 0.7
comptime BLOCK = 256


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


def test_cfm_solve_full() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/real/"
    var fix_w = "tests/fixtures/s3gen/"
    var ctx = DeviceContext()

    # ---- Inputs (B=1).
    var z_init = load_fp32(fix + "e2e_cfm_z_init.bin")
    var t_span = load_fp32(fix + "e2e_cfm_t_span.bin")
    var mu = load_fp32(fix + "e2e_cfm_mu.bin")
    var mask = load_fp32(fix + "e2e_cfm_mask.bin")
    var spks = load_fp32(fix + "e2e_cfm_spks.bin")
    var cond = load_fp32(fix + "e2e_cfm_cond.bin")
    var exp = load_fp32(fix + "e2e_mel_final.bin")

    var n_x = B * MEL_C * T
    var n_x2 = B2 * MEL_C * T
    var n_mask = B * 1 * T
    var n_mask2 = B2 * 1 * T
    var n_spks = B * SPKS_C
    var n_spks2 = B2 * SPKS_C
    var n_te_emb = B2 * IN_DIM_TE
    var n_te_mlp = B2 * TIME_EMB_DIM
    var n_packed = B2 * PACKED_C * T

    # ---- Buffers (B=1).
    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var x_next_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var mu_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var mask_buf = ctx.enqueue_create_buffer[DType.float32](n_mask)
    var spks_buf = ctx.enqueue_create_buffer[DType.float32](n_spks)
    var cond_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    upload(x_buf, z_init.data, n_x)
    upload(mu_buf, mu.data, n_x)
    upload(mask_buf, mask.data, n_mask)
    upload(spks_buf, spks.data, n_spks)
    upload(cond_buf, cond.data, n_x)

    # ---- Buffers (B=2 doubled).
    var x_in2 = ctx.enqueue_create_buffer[DType.float32](n_x2)
    var mu_in2 = ctx.enqueue_create_buffer[DType.float32](n_x2)
    var spks_in2 = ctx.enqueue_create_buffer[DType.float32](n_spks2)
    var cond_in2 = ctx.enqueue_create_buffer[DType.float32](n_x2)
    var mask_in2 = ctx.enqueue_create_buffer[DType.float32](n_mask2)
    var t_in2 = ctx.enqueue_create_buffer[DType.float32](B2 * 1)
    var t_emb_in2 = ctx.enqueue_create_buffer[DType.float32](n_te_emb)
    var t_mlp_h2 = ctx.enqueue_create_buffer[DType.float32](n_te_mlp)
    var t_mlp_act2 = ctx.enqueue_create_buffer[DType.float32](n_te_mlp)
    var t_mlp_out2 = ctx.enqueue_create_buffer[DType.float32](n_te_mlp)
    var packed_in2 = ctx.enqueue_create_buffer[DType.float32](n_packed)
    var est_out = ctx.enqueue_create_buffer[DType.float32](n_x2)

    # ---- time_mlp weights.
    var tm_w1 = upload_w(ctx, fix_w, "weights/flow__decoder__estimator__time_mlp__linear_1__weight.bin")
    var tm_b1 = upload_w(ctx, fix_w, "weights/flow__decoder__estimator__time_mlp__linear_1__bias.bin")
    var tm_w2 = upload_w(ctx, fix_w, "weights/flow__decoder__estimator__time_mlp__linear_2__weight.bin")
    var tm_b2 = upload_w(ctx, fix_w, "weights/flow__decoder__estimator__time_mlp__linear_2__bias.bin")

    # ---- Estimator weights.
    var dn_rn = load_resnet(ctx, fix_w, "weights/flow__decoder__estimator__down_blocks__0__0__")
    var dn_tb0 = load_tblock(ctx, fix_w, "weights/flow__decoder__estimator__down_blocks__0__1__0__")
    var dn_tb1 = load_tblock(ctx, fix_w, "weights/flow__decoder__estimator__down_blocks__0__1__1__")
    var dn_tb2 = load_tblock(ctx, fix_w, "weights/flow__decoder__estimator__down_blocks__0__1__2__")
    var dn_tb3 = load_tblock(ctx, fix_w, "weights/flow__decoder__estimator__down_blocks__0__1__3__")
    var dn_ds_w = upload_w(ctx, fix_w, "weights/flow__decoder__estimator__down_blocks__0__2__weight.bin")
    var dn_ds_b = upload_w(ctx, fix_w, "weights/flow__decoder__estimator__down_blocks__0__2__bias.bin")

    var mid_rns = List[CausalResnetWeights]()
    var mid_tb0s = List[BasicTransformerWeights]()
    var mid_tb1s = List[BasicTransformerWeights]()
    var mid_tb2s = List[BasicTransformerWeights]()
    var mid_tb3s = List[BasicTransformerWeights]()
    for i in range(12):
        var p = "weights/flow__decoder__estimator__mid_blocks__" + String(i) + "__"
        mid_rns.append(load_resnet(ctx, fix_w, p + "0__"))
        mid_tb0s.append(load_tblock(ctx, fix_w, p + "1__0__"))
        mid_tb1s.append(load_tblock(ctx, fix_w, p + "1__1__"))
        mid_tb2s.append(load_tblock(ctx, fix_w, p + "1__2__"))
        mid_tb3s.append(load_tblock(ctx, fix_w, p + "1__3__"))

    var up_rn = load_resnet(ctx, fix_w, "weights/flow__decoder__estimator__up_blocks__0__0__")
    var up_tb0 = load_tblock(ctx, fix_w, "weights/flow__decoder__estimator__up_blocks__0__1__0__")
    var up_tb1 = load_tblock(ctx, fix_w, "weights/flow__decoder__estimator__up_blocks__0__1__1__")
    var up_tb2 = load_tblock(ctx, fix_w, "weights/flow__decoder__estimator__up_blocks__0__1__2__")
    var up_tb3 = load_tblock(ctx, fix_w, "weights/flow__decoder__estimator__up_blocks__0__1__3__")
    var up_us_w = upload_w(ctx, fix_w, "weights/flow__decoder__estimator__up_blocks__0__2__weight.bin")
    var up_us_b = upload_w(ctx, fix_w, "weights/flow__decoder__estimator__up_blocks__0__2__bias.bin")
    var fb_cw = upload_w(ctx, fix_w, "weights/flow__decoder__estimator__final_block__block__0__weight.bin")
    var fb_cb = upload_w(ctx, fix_w, "weights/flow__decoder__estimator__final_block__block__0__bias.bin")
    var fb_lw = upload_w(ctx, fix_w, "weights/flow__decoder__estimator__final_block__block__2__weight.bin")
    var fb_lb = upload_w(ctx, fix_w, "weights/flow__decoder__estimator__final_block__block__2__bias.bin")
    var fp_w = upload_w(ctx, fix_w, "weights/flow__decoder__estimator__final_proj__weight.bin")
    var fp_b = upload_w(ctx, fix_w, "weights/flow__decoder__estimator__final_proj__bias.bin")

    # ---- Build CFG-doubled mu, spks, cond, mask (constant across iterations).
    comptime x_layout = row_major[B, MEL_C, T]()
    comptime x2_layout = row_major[B2, MEL_C, T]()
    comptime mask_layout = row_major[B, 1, T]()
    comptime mask2_layout = row_major[B2, 1, T]()
    comptime spks_layout = row_major[B, SPKS_C]()
    comptime spks2_layout = row_major[B2, SPKS_C]()
    comptime packed_layout = row_major[B2, PACKED_C, T]()
    comptime t_in2_layout = row_major[B2, 1]()
    comptime t_emb_layout = row_major[B2, IN_DIM_TE]()
    comptime mlp_btd_layout = row_major[B2, 1, TIME_EMB_DIM]()
    comptime t_emb_btd_layout = row_major[B2, 1, IN_DIM_TE]()
    comptime w_te1_layout = row_major[TIME_EMB_DIM, IN_DIM_TE]()
    comptime w_te2_layout = row_major[TIME_EMB_DIM, TIME_EMB_DIM]()
    comptime p_te_layout = row_major[TIME_EMB_DIM]()
    comptime flat_mlp_layout = row_major[B2 * TIME_EMB_DIM]()

    var mu_t = TileTensor(mu_buf, x_layout)
    var mu_in2_t = TileTensor(mu_in2, x2_layout)
    var spks_t = TileTensor(spks_buf, spks_layout)
    var spks_in2_t = TileTensor(spks_in2, spks2_layout)
    var cond_t = TileTensor(cond_buf, x_layout)
    var cond_in2_t = TileTensor(cond_in2, x2_layout)
    var mask_t = TileTensor(mask_buf, mask_layout)
    var mask_in2_t = TileTensor(mask_in2, mask2_layout)
    var x_t = TileTensor(x_buf, x_layout)
    var x_in2_t = TileTensor(x_in2, x2_layout)
    var x_next_t = TileTensor(x_next_buf, x_layout)
    var packed_t = TileTensor(packed_in2, packed_layout)
    var est_out_t = TileTensor(est_out, x2_layout)
    var t_in2_t = TileTensor(t_in2, t_in2_layout)
    var t_emb_in2_t = TileTensor(t_emb_in2, t_emb_layout)
    var t_emb_btd_t = TileTensor(t_emb_in2, t_emb_btd_layout)
    var t_mlp_h2_btd_t = TileTensor(t_mlp_h2, mlp_btd_layout)
    var t_mlp_h2_flat_t = TileTensor(t_mlp_h2, flat_mlp_layout)
    var t_mlp_act2_btd_t = TileTensor(t_mlp_act2, mlp_btd_layout)
    var t_mlp_act2_flat_t = TileTensor(t_mlp_act2, flat_mlp_layout)
    var t_mlp_out2_btd_t = TileTensor(t_mlp_out2, mlp_btd_layout)
    var tm_w1_t = TileTensor(tm_w1, w_te1_layout)
    var tm_b1_t = TileTensor(tm_b1, p_te_layout)
    var tm_w2_t = TileTensor(tm_w2, w_te2_layout)
    var tm_b2_t = TileTensor(tm_b2, p_te_layout)

    comptime cfg3_k = build_cfg_inputs_kernel[
        DType.float32, type_of(x_layout), type_of(x2_layout),
        B, MEL_C, T, BLOCK,
    ]
    comptime cfg3_mask_k = build_cfg_inputs_kernel[
        DType.float32, type_of(mask_layout), type_of(mask2_layout),
        B, 1, T, BLOCK,
    ]
    comptime cfg2_k = build_cfg_inputs_2d_kernel[
        DType.float32, type_of(spks_layout), type_of(spks2_layout),
        B, SPKS_C, BLOCK,
    ]

    # Build the constant CFG-doubled inputs (mu, cond, spks, mask) once.
    ctx.enqueue_function[cfg3_k, cfg3_k](
        mu_in2_t, mu_t, 1,
        grid_dim=B2 * MEL_C, block_dim=BLOCK,
    )
    ctx.enqueue_function[cfg3_k, cfg3_k](
        cond_in2_t, cond_t, 1,
        grid_dim=B2 * MEL_C, block_dim=BLOCK,
    )
    ctx.enqueue_function[cfg2_k, cfg2_k](
        spks_in2_t, spks_t, 1,
        grid_dim=ceildiv(B2 * SPKS_C, BLOCK), block_dim=BLOCK,
    )
    ctx.enqueue_function[cfg3_mask_k, cfg3_mask_k](
        mask_in2_t, mask_t, 0,
        grid_dim=B2 * 1, block_dim=BLOCK,
    )

    # Comptime kernels.
    comptime emb_k = sinusoidal_pos_emb_kernel[
        DType.float32, type_of(t_emb_layout), type_of(t_in2_layout), IN_DIM_TE, BLOCK,
    ]
    comptime lin1_k = linear_kernel[
        DType.float32, type_of(t_emb_btd_layout), type_of(w_te1_layout),
        type_of(p_te_layout), type_of(mlp_btd_layout),
        True, BLOCK,
    ]
    comptime sw_k = swish_kernel[
        DType.float32, type_of(flat_mlp_layout), type_of(flat_mlp_layout), BLOCK,
    ]
    comptime lin2_k = linear_kernel[
        DType.float32, type_of(mlp_btd_layout), type_of(w_te2_layout),
        type_of(p_te_layout), type_of(mlp_btd_layout),
        True, BLOCK,
    ]
    comptime pack_k = pack_xmsc_kernel[
        DType.float32, type_of(packed_layout),
        type_of(x2_layout), type_of(x2_layout),
        type_of(spks2_layout), type_of(x2_layout),
        BLOCK,
    ]
    comptime step_k = cfm_euler_step_kernel[
        DType.float32, type_of(x_layout), type_of(x2_layout), type_of(x_layout),
        B, MEL_C, T, BLOCK,
    ]

    # Run 10 Euler steps.
    for step in range(N_STEPS):
        var t_cur = t_span.data[step]
        var t_next = t_span.data[step + 1]
        var dt = Float32(t_next - t_cur)

        # 1. CFG-double x.
        ctx.enqueue_function[cfg3_k, cfg3_k](
            x_in2_t, x_t, 0,
            grid_dim=B2 * MEL_C, block_dim=BLOCK,
        )
        # 2. Set t_in2 = [t_cur, t_cur].
        with t_in2.map_to_host() as h:
            h[0] = t_cur
            h[1] = t_cur
        # 3. SinusoidalPosEmb → time_mlp.
        ctx.enqueue_function[emb_k, emb_k](
            t_emb_in2_t, t_in2_t, B2, Float32(1000.0),
            grid_dim=B2, block_dim=BLOCK,
        )
        ctx.enqueue_function[lin1_k, lin1_k](
            t_mlp_h2_btd_t, t_emb_btd_t, tm_w1_t, tm_b1_t, B2, 1, IN_DIM_TE, TIME_EMB_DIM,
            grid_dim=B2, block_dim=BLOCK,
        )
        ctx.enqueue_function[sw_k, sw_k](
            t_mlp_act2_flat_t, t_mlp_h2_flat_t, B2 * TIME_EMB_DIM,
            grid_dim=ceildiv(B2 * TIME_EMB_DIM, BLOCK), block_dim=BLOCK,
        )
        ctx.enqueue_function[lin2_k, lin2_k](
            t_mlp_out2_btd_t, t_mlp_act2_btd_t, tm_w2_t, tm_b2_t, B2, 1, TIME_EMB_DIM, TIME_EMB_DIM,
            grid_dim=B2, block_dim=BLOCK,
        )
        # 4. Pack [x_in2, mu_in2, spks_in2, cond_in2] → packed_in2.
        ctx.enqueue_function[pack_k, pack_k](
            packed_t, x_in2_t, mu_in2_t, spks_in2_t, cond_in2_t, B2, T,
            grid_dim=B2 * PACKED_C, block_dim=BLOCK,
        )
        # 5. Estimator forward.
        estimator_forward[B2, T, D, H, D_K, FF_INNER, TIME_EMB_DIM, MEL_C](
            ctx, packed_in2, mask_in2, t_mlp_out2, est_out,
            dn_rn, dn_tb0, dn_tb1, dn_tb2, dn_tb3, dn_ds_w, dn_ds_b,
            mid_rns, mid_tb0s, mid_tb1s, mid_tb2s, mid_tb3s,
            up_rn, up_tb0, up_tb1, up_tb2, up_tb3, up_us_w, up_us_b,
            fb_cw, fb_cb, fb_lw, fb_lb, fp_w, fp_b,
        )
        # 6. CFG combine + Euler step.
        ctx.enqueue_function[step_k, step_k](
            x_next_t, x_t, est_out_t, dt, CFG_RATE,
            grid_dim=ceildiv(n_x, BLOCK), block_dim=BLOCK,
        )
        ctx.synchronize()
        # 7. Copy x_next → x.
        with x_next_buf.map_to_host() as src:
            with x_buf.map_to_host() as dst:
                for i in range(n_x):
                    dst[i] = src[i]
        print("step", step, "done")

    # ---- Trim to mel_len2=262 (drop first 500 prompt_feat frames) and
    # compare against e2e_mel_final.bin (B, 80, 262).
    var T_TRIM = 500
    var T_OUT = T - T_TRIM   # = 262
    var max_abs: Float32 = 0.0
    var sum_abs: Float64 = 0.0
    # Save trimmed mel to disk for the HiFiGAN stage.
    from fixture import save_fp32_1d
    var trimmed = List[Float32]()
    with x_buf.map_to_host() as h:
        # x_buf is (B=1, MEL_C=80, T=762). Trim along the last axis: keep [T_TRIM:T].
        for c in range(MEL_C):
            for t in range(T_TRIM, T):
                var idx = c * T + t
                trimmed.append(h[idx])
        # Compare against exp (1, 80, 262).
        var n_trim = MEL_C * T_OUT
        for i in range(n_trim):
            var d = trimmed[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_abs += Float64(d)
            if i < 8:
                print("mel[", i, "]: mojo=", trimmed[i], "  torch=", exp.data[i], "  diff=", d)
    save_fp32_1d("tests/fixtures/real/cloned_voice_mel_mojo.bin", trimmed)
    print("FULL CFM solve (10 steps, cloned voice) — max abs:", max_abs,
          " mean:", sum_abs / Float64(MEL_C * T_OUT))
    print("Saved cloned_voice_mel_mojo.bin (", MEL_C, ",", T_OUT, ")")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
