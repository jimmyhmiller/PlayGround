"""Parity test for AudioEncoderV2 (2 strided conv + 2 ResBlock layers).

Small config (n_mels=80, n_state=128, n_head=4, n_layer=2, stride=2, T_in=24, T_out=6).
"""
from std.math import sqrt
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import block_idx, thread_idx
from layout import TileTensor, TensorLayout, row_major

from fixture import load_fp32
from conv import conv1d_kernel
from decoder_kernels import gelu_kernel
from layernorm import layernorm_kernel, linear_kernel
from fsmn_attention import (
    fsmn_depthwise_conv_kernel, fsmn_memory_kernel,
    rope_s3tokenizer_kernel, scale_4d_kernel,
    multiply_mask_3d_kernel,
    permute_bshd_to_bhsd_kernel, permute_bhsd_to_bsd_kernel,
)
from perceiver import cross_qkt_kernel, cross_softmax_kernel, cross_av_kernel, add_3d_kernel


comptime B = 1
comptime T_IN = 24
comptime T_HALF = 12        # after conv1 (stride=2)
comptime T_OUT = 6          # after conv2 (stride=2)
comptime N_MELS = 80
comptime D = 128
comptime H = 4
comptime DH = D // H
comptime HALF = DH // 2
comptime KSIZE = 31
comptime LEFT = 15
comptime RIGHT = 15
comptime BLOCK = 128
comptime MLP_INNER = D * 4
comptime N_LAYER = 2


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


# Transpose (B, C, T) → (B, T, C) for going from conv output to attention input.
def transpose_bct_to_btc_kernel[
    dtype: DType, InLayout: TensorLayout, OutLayout: TensorLayout, BLOCK: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],   # (B, T, C)
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],        # (B, C, T)
    batch: Int, c: Int, t: Int,
):
    comptime assert inp.flat_rank == 3
    comptime assert output.flat_rank == 3
    var bid = block_idx.x
    var tid = thread_idx.x
    var tt = bid % t
    var b = bid // t
    var cc = tid
    while cc < c:
        var v = rebind[Scalar[dtype]](inp[b, cc, tt]).cast[DType.float32]()
        output[b, tt, cc] = rebind[output.ElementType](v.cast[dtype]())
        cc += BLOCK


# Apply one ResBlock: x_buf (B, T, D) → out_buf (B, T, D).
def res_block_forward[T: Int](
    mut ctx: DeviceContext,
    mut x_buf: DeviceBuffer[DType.float32],
    mut out_buf: DeviceBuffer[DType.float32],
    mut mp_buf: DeviceBuffer[DType.float32],
    mut cos_buf: DeviceBuffer[DType.float32],
    mut sin_buf: DeviceBuffer[DType.float32],
    mut attn_ln_w: DeviceBuffer[DType.float32], mut attn_ln_b: DeviceBuffer[DType.float32],
    mut mlp_ln_w: DeviceBuffer[DType.float32], mut mlp_ln_b: DeviceBuffer[DType.float32],
    mut q_w: DeviceBuffer[DType.float32], mut q_b: DeviceBuffer[DType.float32],
    mut k_w: DeviceBuffer[DType.float32], mut k_b: DeviceBuffer[DType.float32],
    mut v_w: DeviceBuffer[DType.float32], mut v_b: DeviceBuffer[DType.float32],
    mut o_w: DeviceBuffer[DType.float32], mut o_b: DeviceBuffer[DType.float32],
    mut fsmn_w: DeviceBuffer[DType.float32],
    mut mlp_fc1_w: DeviceBuffer[DType.float32], mut mlp_fc1_b: DeviceBuffer[DType.float32],
    mut mlp_fc2_w: DeviceBuffer[DType.float32], mut mlp_fc2_b: DeviceBuffer[DType.float32],
) raises:
    """One FSMN residual block. See test_fsmn_resblock for details."""
    comptime x_layout = row_major[B, T, D]()
    comptime cs_layout = row_major[T, HALF]()
    comptime mp_layout = row_major[B, T, 1]()
    comptime ln_w_layout = row_major[D]()
    comptime lin_w_layout = row_major[D, D]()
    comptime x4_layout = row_major[B, T, H, DH]()
    comptime bhsd_layout = row_major[B, H, T, DH]()
    comptime logits_layout = row_major[B, H, T, T]()
    comptime fsmn_w_layout = row_major[D, 1, KSIZE]()
    comptime mlp_inner_layout = row_major[B, T, MLP_INNER]()
    comptime mlp_w1_layout = row_major[MLP_INNER, D]()
    comptime mlp_b1_layout = row_major[MLP_INNER]()
    comptime mlp_w2_layout = row_major[D, MLP_INNER]()
    comptime mlp_flat_layout = row_major[B * T * MLP_INNER]()

    # Step 1: attn_ln(x) → ln_x.
    var ln_x = ctx.enqueue_create_buffer[DType.float32](B * T * D)
    var x_tt = TileTensor(x_buf, x_layout)
    var ln_x_tt = TileTensor(ln_x, x_layout)
    var attn_ln_w_tt = TileTensor(attn_ln_w, ln_w_layout)
    var attn_ln_b_tt = TileTensor(attn_ln_b, ln_w_layout)
    comptime kln = layernorm_kernel[
        DType.float32, type_of(x_layout), type_of(ln_w_layout), type_of(x_layout), BLOCK,
    ]
    ctx.enqueue_function[kln, kln](
        ln_x_tt, x_tt, attn_ln_w_tt, attn_ln_b_tt,
        B, T, D, Float32(1.0e-5),
        grid_dim=B*T, block_dim=BLOCK,
    )

    # Step 2: FSMN attention forward (inline, no helper since helper requires fixed-size buffers).
    var q_lin = ctx.enqueue_create_buffer[DType.float32](B * T * D)
    var k_lin = ctx.enqueue_create_buffer[DType.float32](B * T * D)
    var v_lin = ctx.enqueue_create_buffer[DType.float32](B * T * D)
    var q_4d = ctx.enqueue_create_buffer[DType.float32](B * T * H * DH)
    var k_4d = ctx.enqueue_create_buffer[DType.float32](B * T * H * DH)
    var v_4d = ctx.enqueue_create_buffer[DType.float32](B * T * H * DH)
    var q_rope = ctx.enqueue_create_buffer[DType.float32](B * T * H * DH)
    var k_rope = ctx.enqueue_create_buffer[DType.float32](B * T * H * DH)
    var v_masked = ctx.enqueue_create_buffer[DType.float32](B * T * D)
    var fsmn_conv = ctx.enqueue_create_buffer[DType.float32](B * T * D)
    var fsm_mem = ctx.enqueue_create_buffer[DType.float32](B * T * D)
    var q_bhsd = ctx.enqueue_create_buffer[DType.float32](B * H * T * DH)
    var k_bhsd = ctx.enqueue_create_buffer[DType.float32](B * H * T * DH)
    var v_bhsd = ctx.enqueue_create_buffer[DType.float32](B * H * T * DH)
    var logits = ctx.enqueue_create_buffer[DType.float32](B * H * T * T)
    var probs = ctx.enqueue_create_buffer[DType.float32](B * H * T * T)
    var av = ctx.enqueue_create_buffer[DType.float32](B * H * T * DH)
    var comb = ctx.enqueue_create_buffer[DType.float32](B * T * D)
    var attn_lin_out = ctx.enqueue_create_buffer[DType.float32](B * T * D)
    var attn_out = ctx.enqueue_create_buffer[DType.float32](B * T * D)

    var cos_tt = TileTensor(cos_buf, cs_layout)
    var sin_tt = TileTensor(sin_buf, cs_layout)
    var mp_tt = TileTensor(mp_buf, mp_layout)
    var q_w_tt = TileTensor(q_w, lin_w_layout); var q_b_tt = TileTensor(q_b, ln_w_layout)
    var k_w_tt = TileTensor(k_w, lin_w_layout); var k_b_tt = TileTensor(k_b, ln_w_layout)
    var v_w_tt = TileTensor(v_w, lin_w_layout); var v_b_tt = TileTensor(v_b, ln_w_layout)
    var o_w_tt = TileTensor(o_w, lin_w_layout); var o_b_tt = TileTensor(o_b, ln_w_layout)
    var fsmn_w_tt = TileTensor(fsmn_w, fsmn_w_layout)
    var q_lin_tt = TileTensor(q_lin, x_layout)
    var k_lin_tt = TileTensor(k_lin, x_layout)
    var v_lin_tt = TileTensor(v_lin, x_layout)
    var q_4d_tt = TileTensor(q_4d, x4_layout)
    var k_4d_tt = TileTensor(k_4d, x4_layout)
    var v_4d_tt = TileTensor(v_4d, x4_layout)
    var q_rope_tt = TileTensor(q_rope, x4_layout)
    var k_rope_tt = TileTensor(k_rope, x4_layout)
    var v_masked_tt = TileTensor(v_masked, x_layout)
    var fsmn_conv_tt = TileTensor(fsmn_conv, x_layout)
    var fsm_mem_tt = TileTensor(fsm_mem, x_layout)
    var q_bhsd_tt = TileTensor(q_bhsd, bhsd_layout)
    var k_bhsd_tt = TileTensor(k_bhsd, bhsd_layout)
    var v_bhsd_tt = TileTensor(v_bhsd, bhsd_layout)
    var logits_tt = TileTensor(logits, logits_layout)
    var probs_tt = TileTensor(probs, logits_layout)
    var av_tt = TileTensor(av, bhsd_layout)
    var comb_tt = TileTensor(comb, x_layout)
    var attn_lin_out_tt = TileTensor(attn_lin_out, x_layout)
    var attn_out_tt = TileTensor(attn_out, x_layout)

    comptime klin = linear_kernel[
        DType.float32, type_of(x_layout), type_of(lin_w_layout),
        type_of(ln_w_layout), type_of(x_layout),
        True, BLOCK,
    ]
    ctx.enqueue_function[klin, klin](q_lin_tt, ln_x_tt, q_w_tt, q_b_tt, B, T, D, D, grid_dim=B*T, block_dim=BLOCK)
    ctx.enqueue_function[klin, klin](k_lin_tt, ln_x_tt, k_w_tt, k_b_tt, B, T, D, D, grid_dim=B*T, block_dim=BLOCK)
    ctx.enqueue_function[klin, klin](v_lin_tt, ln_x_tt, v_w_tt, v_b_tt, B, T, D, D, grid_dim=B*T, block_dim=BLOCK)

    ctx.enqueue_copy(q_4d, q_lin); ctx.enqueue_copy(k_4d, k_lin); ctx.enqueue_copy(v_4d, v_lin)

    comptime krope = rope_s3tokenizer_kernel[
        DType.float32, type_of(x4_layout), type_of(cs_layout),
        type_of(x4_layout), DH, HALF,
    ]
    ctx.enqueue_function[krope, krope](q_rope_tt, q_4d_tt, cos_tt, sin_tt, T, H, grid_dim=B*T*H, block_dim=DH)
    ctx.enqueue_function[krope, krope](k_rope_tt, k_4d_tt, cos_tt, sin_tt, T, H, grid_dim=B*T*H, block_dim=DH)

    comptime kmm = multiply_mask_3d_kernel[
        DType.float32, type_of(x_layout), type_of(mp_layout),
        type_of(x_layout), BLOCK,
    ]
    ctx.enqueue_function[kmm, kmm](v_masked_tt, v_lin_tt, mp_tt, B, T, D, grid_dim=B*T, block_dim=BLOCK)
    comptime kfconv = fsmn_depthwise_conv_kernel[
        DType.float32, type_of(x_layout), type_of(fsmn_w_layout),
        type_of(x_layout), KSIZE, LEFT, RIGHT, BLOCK,
    ]
    ctx.enqueue_function[kfconv, kfconv](fsmn_conv_tt, v_masked_tt, fsmn_w_tt, B, T, D, grid_dim=B*T, block_dim=BLOCK)
    comptime kfm = fsmn_memory_kernel[
        DType.float32, type_of(x_layout), type_of(x_layout),
        type_of(mp_layout), type_of(x_layout), BLOCK,
    ]
    ctx.enqueue_function[kfm, kfm](fsm_mem_tt, fsmn_conv_tt, v_masked_tt, mp_tt, B, T, D, grid_dim=B*T, block_dim=BLOCK)

    comptime kpb = permute_bshd_to_bhsd_kernel[
        DType.float32, type_of(x4_layout), type_of(bhsd_layout), H, DH,
    ]
    ctx.enqueue_function[kpb, kpb](q_bhsd_tt, q_rope_tt, B, T, grid_dim=B*T*H, block_dim=DH)
    ctx.enqueue_function[kpb, kpb](k_bhsd_tt, k_rope_tt, B, T, grid_dim=B*T*H, block_dim=DH)
    ctx.enqueue_function[kpb, kpb](v_bhsd_tt, v_4d_tt,   B, T, grid_dim=B*T*H, block_dim=DH)

    comptime scale_amt: Float32 = 1.0 / sqrt(sqrt(Float32(DH)))
    comptime ksc = scale_4d_kernel[DType.float32, type_of(bhsd_layout), BLOCK]
    ctx.enqueue_function[ksc, ksc](q_bhsd_tt, B, H, T, DH, scale_amt, grid_dim=B*H*T, block_dim=BLOCK)
    ctx.enqueue_function[ksc, ksc](k_bhsd_tt, B, H, T, DH, scale_amt, grid_dim=B*H*T, block_dim=BLOCK)

    comptime kqkt = cross_qkt_kernel[
        DType.float32, type_of(bhsd_layout), type_of(bhsd_layout),
        type_of(logits_layout), DH, T,
    ]
    ctx.enqueue_function[kqkt, kqkt](logits_tt, q_bhsd_tt, k_bhsd_tt, H, T, Float32(1.0), grid_dim=B*H*T, block_dim=T)
    comptime ksm = cross_softmax_kernel[
        DType.float32, type_of(logits_layout), type_of(logits_layout), T, BLOCK,
    ]
    ctx.enqueue_function[ksm, ksm](probs_tt, logits_tt, H, T, grid_dim=B*H*T, block_dim=BLOCK)
    comptime kav = cross_av_kernel[
        DType.float32, type_of(logits_layout), type_of(bhsd_layout),
        type_of(bhsd_layout), T, DH,
    ]
    ctx.enqueue_function[kav, kav](av_tt, probs_tt, v_bhsd_tt, H, T, grid_dim=B*H*T, block_dim=DH)

    comptime kcomb = permute_bhsd_to_bsd_kernel[
        DType.float32, type_of(bhsd_layout), type_of(x_layout), H, DH,
    ]
    ctx.enqueue_function[kcomb, kcomb](comb_tt, av_tt, B, T, grid_dim=B*T*H, block_dim=DH)
    ctx.enqueue_function[klin, klin](attn_lin_out_tt, comb_tt, o_w_tt, o_b_tt, B, T, D, D, grid_dim=B*T, block_dim=BLOCK)
    comptime kadd = add_3d_kernel[DType.float32, type_of(x_layout), BLOCK]
    ctx.enqueue_function[kadd, kadd](attn_out_tt, attn_lin_out_tt, fsm_mem_tt, B, T, D, grid_dim=B*T, block_dim=BLOCK)

    # Step 3: x + attn_out → post_attn.
    var post_attn = ctx.enqueue_create_buffer[DType.float32](B * T * D)
    var post_attn_tt = TileTensor(post_attn, x_layout)
    ctx.enqueue_function[kadd, kadd](post_attn_tt, x_tt, attn_out_tt, B, T, D, grid_dim=B*T, block_dim=BLOCK)

    # Step 4: mlp_ln(post_attn).
    var ln_post = ctx.enqueue_create_buffer[DType.float32](B * T * D)
    var ln_post_tt = TileTensor(ln_post, x_layout)
    var mlp_ln_w_tt = TileTensor(mlp_ln_w, ln_w_layout)
    var mlp_ln_b_tt = TileTensor(mlp_ln_b, ln_w_layout)
    ctx.enqueue_function[kln, kln](
        ln_post_tt, post_attn_tt, mlp_ln_w_tt, mlp_ln_b_tt,
        B, T, D, Float32(1.0e-5),
        grid_dim=B*T, block_dim=BLOCK,
    )

    # Step 5: MLP fc1 → GELU → fc2.
    var mlp_inner = ctx.enqueue_create_buffer[DType.float32](B * T * MLP_INNER)
    var mlp_inner_tt = TileTensor(mlp_inner, mlp_inner_layout)
    var mlp_fc1_w_tt = TileTensor(mlp_fc1_w, mlp_w1_layout)
    var mlp_fc1_b_tt = TileTensor(mlp_fc1_b, mlp_b1_layout)
    comptime klin_fc1 = linear_kernel[
        DType.float32, type_of(x_layout), type_of(mlp_w1_layout),
        type_of(mlp_b1_layout), type_of(mlp_inner_layout),
        True, BLOCK,
    ]
    ctx.enqueue_function[klin_fc1, klin_fc1](
        mlp_inner_tt, ln_post_tt, mlp_fc1_w_tt, mlp_fc1_b_tt,
        B, T, D, MLP_INNER,
        grid_dim=B*T, block_dim=BLOCK,
    )
    var mlp_gelu = ctx.enqueue_create_buffer[DType.float32](B * T * MLP_INNER)
    var mlp_inner_flat_tt = TileTensor(mlp_inner, mlp_flat_layout)
    var mlp_gelu_flat_tt = TileTensor(mlp_gelu, mlp_flat_layout)
    comptime kgelu = gelu_kernel[
        DType.float32, type_of(mlp_flat_layout), type_of(mlp_flat_layout), BLOCK,
    ]
    ctx.enqueue_function[kgelu, kgelu](
        mlp_gelu_flat_tt, mlp_inner_flat_tt, B * T * MLP_INNER,
        grid_dim=(B * T * MLP_INNER + BLOCK - 1) // BLOCK, block_dim=BLOCK,
    )
    var mlp_gelu_tt = TileTensor(mlp_gelu, mlp_inner_layout)
    var mlp_out = ctx.enqueue_create_buffer[DType.float32](B * T * D)
    var mlp_out_tt = TileTensor(mlp_out, x_layout)
    var mlp_fc2_w_tt = TileTensor(mlp_fc2_w, mlp_w2_layout)
    var mlp_fc2_b_tt = TileTensor(mlp_fc2_b, ln_w_layout)
    comptime klin_fc2 = linear_kernel[
        DType.float32, type_of(mlp_inner_layout), type_of(mlp_w2_layout),
        type_of(ln_w_layout), type_of(x_layout),
        True, BLOCK,
    ]
    ctx.enqueue_function[klin_fc2, klin_fc2](
        mlp_out_tt, mlp_gelu_tt, mlp_fc2_w_tt, mlp_fc2_b_tt,
        B, T, MLP_INNER, D,
        grid_dim=B*T, block_dim=BLOCK,
    )

    # Step 6: out = post_attn + mlp_out.
    var out_tt = TileTensor(out_buf, x_layout)
    ctx.enqueue_function[kadd, kadd](out_tt, post_attn_tt, mlp_out_tt, B, T, D, grid_dim=B*T, block_dim=BLOCK)


def test_audio_encoder_v2() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/audio_encoder_v2/"
    var ctx = DeviceContext()

    var x_t = load_fp32(fix + "x.bin")
    var exp = load_fp32(fix + "out.bin")
    var mp_t = load_fp32(fix + "mask_pad.bin")
    var cos_t = load_fp32(fix + "cos.bin")
    var sin_t = load_fp32(fix + "sin.bin")

    var x_buf = ctx.enqueue_create_buffer[DType.float32](B * N_MELS * T_IN)
    upload(x_buf, x_t.data, B * N_MELS * T_IN)
    var mp_buf = ctx.enqueue_create_buffer[DType.float32](B * T_OUT * 1)
    upload(mp_buf, mp_t.data, B * T_OUT * 1)
    var cos_buf = ctx.enqueue_create_buffer[DType.float32](T_OUT * HALF)
    upload(cos_buf, cos_t.data, T_OUT * HALF)
    var sin_buf = ctx.enqueue_create_buffer[DType.float32](T_OUT * HALF)
    upload(sin_buf, sin_t.data, T_OUT * HALF)

    var conv1_w = upload_w(ctx, fix, "conv1_w.bin")
    var conv1_b = upload_w(ctx, fix, "conv1_b.bin")
    var conv2_w = upload_w(ctx, fix, "conv2_w.bin")
    var conv2_b = upload_w(ctx, fix, "conv2_b.bin")

    # Conv1: (B, n_mels, T_in) → (B, D, T_HALF).
    var c1_out = ctx.enqueue_create_buffer[DType.float32](B * D * T_HALF)
    comptime x_layout = row_major[B, N_MELS, T_IN]()
    comptime c1_out_layout = row_major[B, D, T_HALF]()
    comptime c1_w_layout = row_major[D, N_MELS, 3]()
    comptime c1_b_layout = row_major[D]()
    var x_tt = TileTensor(x_buf, x_layout)
    var conv1_w_tt = TileTensor(conv1_w, c1_w_layout)
    var conv1_b_tt = TileTensor(conv1_b, c1_b_layout)
    var c1_out_tt = TileTensor(c1_out, c1_out_layout)
    comptime kconv1 = conv1d_kernel[
        DType.float32, type_of(x_layout), type_of(c1_w_layout),
        type_of(c1_b_layout), type_of(c1_out_layout), 3, True,
    ]
    ctx.enqueue_function[kconv1, kconv1](
        c1_out_tt, x_tt, conv1_w_tt, conv1_b_tt,
        B, N_MELS, D, T_IN, T_HALF, 2, 1, 1,    # stride=2, padding=1, dilation=1
        grid_dim=B * D * T_HALF, block_dim=1,
    )

    # GELU.
    var c1_gelu = ctx.enqueue_create_buffer[DType.float32](B * D * T_HALF)
    comptime c1_flat = row_major[B * D * T_HALF]()
    var c1_out_flat_tt = TileTensor(c1_out, c1_flat)
    var c1_gelu_flat_tt = TileTensor(c1_gelu, c1_flat)
    comptime kgelu1 = gelu_kernel[
        DType.float32, type_of(c1_flat), type_of(c1_flat), BLOCK,
    ]
    ctx.enqueue_function[kgelu1, kgelu1](
        c1_gelu_flat_tt, c1_out_flat_tt, B * D * T_HALF,
        grid_dim=(B * D * T_HALF + BLOCK - 1) // BLOCK, block_dim=BLOCK,
    )

    # Conv2: (B, D, T_HALF) → (B, D, T_OUT).
    var c2_out = ctx.enqueue_create_buffer[DType.float32](B * D * T_OUT)
    comptime c2_in_layout = row_major[B, D, T_HALF]()
    comptime c2_out_layout = row_major[B, D, T_OUT]()
    comptime c2_w_layout = row_major[D, D, 3]()
    var c1_gelu_tt = TileTensor(c1_gelu, c2_in_layout)
    var conv2_w_tt = TileTensor(conv2_w, c2_w_layout)
    var conv2_b_tt = TileTensor(conv2_b, c1_b_layout)
    var c2_out_tt = TileTensor(c2_out, c2_out_layout)
    comptime kconv2 = conv1d_kernel[
        DType.float32, type_of(c2_in_layout), type_of(c2_w_layout),
        type_of(c1_b_layout), type_of(c2_out_layout), 3, True,
    ]
    ctx.enqueue_function[kconv2, kconv2](
        c2_out_tt, c1_gelu_tt, conv2_w_tt, conv2_b_tt,
        B, D, D, T_HALF, T_OUT, 2, 1, 1,
        grid_dim=B * D * T_OUT, block_dim=1,
    )
    var c2_gelu = ctx.enqueue_create_buffer[DType.float32](B * D * T_OUT)
    comptime c2_flat = row_major[B * D * T_OUT]()
    var c2_out_flat_tt = TileTensor(c2_out, c2_flat)
    var c2_gelu_flat_tt = TileTensor(c2_gelu, c2_flat)
    comptime kgelu2 = gelu_kernel[
        DType.float32, type_of(c2_flat), type_of(c2_flat), BLOCK,
    ]
    ctx.enqueue_function[kgelu2, kgelu2](
        c2_gelu_flat_tt, c2_out_flat_tt, B * D * T_OUT,
        grid_dim=(B * D * T_OUT + BLOCK - 1) // BLOCK, block_dim=BLOCK,
    )

    # Transpose (B, D, T_OUT) → (B, T_OUT, D).
    var x_btc = ctx.enqueue_create_buffer[DType.float32](B * T_OUT * D)
    comptime x_btc_layout = row_major[B, T_OUT, D]()
    var c2_gelu_tt = TileTensor(c2_gelu, c2_out_layout)
    var x_btc_tt = TileTensor(x_btc, x_btc_layout)
    comptime ktrans = transpose_bct_to_btc_kernel[
        DType.float32, type_of(c2_out_layout), type_of(x_btc_layout), BLOCK,
    ]
    ctx.enqueue_function[ktrans, ktrans](
        x_btc_tt, c2_gelu_tt, B, D, T_OUT,
        grid_dim=B * T_OUT, block_dim=BLOCK,
    )

    # Load all per-layer weights and run blocks.
    var l0_attn_ln_w = upload_w(ctx, fix, "L0_attn_ln_w.bin")
    var l0_attn_ln_b = upload_w(ctx, fix, "L0_attn_ln_b.bin")
    var l0_mlp_ln_w  = upload_w(ctx, fix, "L0_mlp_ln_w.bin")
    var l0_mlp_ln_b  = upload_w(ctx, fix, "L0_mlp_ln_b.bin")
    var l0_q_w = upload_w(ctx, fix, "L0_q_w.bin"); var l0_q_b = upload_w(ctx, fix, "L0_q_b.bin")
    var l0_k_w = upload_w(ctx, fix, "L0_k_w.bin")
    var l0_k_b = ctx.enqueue_create_buffer[DType.float32](D); l0_k_b.enqueue_fill(0.0)
    var l0_v_w = upload_w(ctx, fix, "L0_v_w.bin"); var l0_v_b = upload_w(ctx, fix, "L0_v_b.bin")
    var l0_o_w = upload_w(ctx, fix, "L0_out_w.bin"); var l0_o_b = upload_w(ctx, fix, "L0_out_b.bin")
    var l0_fsmn = upload_w(ctx, fix, "L0_fsmn_w.bin")
    var l0_fc1_w = upload_w(ctx, fix, "L0_mlp_fc1_w.bin"); var l0_fc1_b = upload_w(ctx, fix, "L0_mlp_fc1_b.bin")
    var l0_fc2_w = upload_w(ctx, fix, "L0_mlp_fc2_w.bin"); var l0_fc2_b = upload_w(ctx, fix, "L0_mlp_fc2_b.bin")

    var l1_attn_ln_w = upload_w(ctx, fix, "L1_attn_ln_w.bin")
    var l1_attn_ln_b = upload_w(ctx, fix, "L1_attn_ln_b.bin")
    var l1_mlp_ln_w  = upload_w(ctx, fix, "L1_mlp_ln_w.bin")
    var l1_mlp_ln_b  = upload_w(ctx, fix, "L1_mlp_ln_b.bin")
    var l1_q_w = upload_w(ctx, fix, "L1_q_w.bin"); var l1_q_b = upload_w(ctx, fix, "L1_q_b.bin")
    var l1_k_w = upload_w(ctx, fix, "L1_k_w.bin")
    var l1_k_b = ctx.enqueue_create_buffer[DType.float32](D); l1_k_b.enqueue_fill(0.0)
    var l1_v_w = upload_w(ctx, fix, "L1_v_w.bin"); var l1_v_b = upload_w(ctx, fix, "L1_v_b.bin")
    var l1_o_w = upload_w(ctx, fix, "L1_out_w.bin"); var l1_o_b = upload_w(ctx, fix, "L1_out_b.bin")
    var l1_fsmn = upload_w(ctx, fix, "L1_fsmn_w.bin")
    var l1_fc1_w = upload_w(ctx, fix, "L1_mlp_fc1_w.bin"); var l1_fc1_b = upload_w(ctx, fix, "L1_mlp_fc1_b.bin")
    var l1_fc2_w = upload_w(ctx, fix, "L1_mlp_fc2_w.bin"); var l1_fc2_b = upload_w(ctx, fix, "L1_mlp_fc2_b.bin")

    # Block 0.
    var block0_out = ctx.enqueue_create_buffer[DType.float32](B * T_OUT * D)
    res_block_forward[T_OUT](
        ctx, x_btc, block0_out, mp_buf, cos_buf, sin_buf,
        l0_attn_ln_w, l0_attn_ln_b, l0_mlp_ln_w, l0_mlp_ln_b,
        l0_q_w, l0_q_b, l0_k_w, l0_k_b, l0_v_w, l0_v_b, l0_o_w, l0_o_b, l0_fsmn,
        l0_fc1_w, l0_fc1_b, l0_fc2_w, l0_fc2_b,
    )

    # Block 1.
    var block1_out = ctx.enqueue_create_buffer[DType.float32](B * T_OUT * D)
    res_block_forward[T_OUT](
        ctx, block0_out, block1_out, mp_buf, cos_buf, sin_buf,
        l1_attn_ln_w, l1_attn_ln_b, l1_mlp_ln_w, l1_mlp_ln_b,
        l1_q_w, l1_q_b, l1_k_w, l1_k_b, l1_v_w, l1_v_b, l1_o_w, l1_o_b, l1_fsmn,
        l1_fc1_w, l1_fc1_b, l1_fc2_w, l1_fc2_b,
    )

    ctx.synchronize()

    var n_out = B * T_OUT * D
    var max_abs: Float32 = 0.0
    var max_rel: Float32 = 0.0
    with block1_out.map_to_host() as h:
        for i in range(n_out):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            var r0 = exp.data[i]
            if r0 < 0.0: r0 = -r0
            if r0 > 1.0e-3:
                var r = d / r0
                if r > max_rel: max_rel = r
            if i < 8:
                print("ae2[", i, "]: mojo=", h[i], "  torch=", exp.data[i], "  diff=", d)
    print("AudioEncoderV2 (2 layers) — max abs:", max_abs, "  max_rel:", max_rel)
    assert_almost_equal(max_abs, 0.0, atol=2.0e-4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
