"""
Test source signal generation: f0_predictor → f0_upsample → SourceModuleHnNSF → STFT.

For the cloned voice test mel, all f0 values are below voiced_threshold=10,
so uv=0 everywhere and the source is essentially noise. The Mojo output won't
match upstream byte-exactly (different RNG), but the s_stft_cat should have
similar amplitude/spectral characteristics — when fed to HiFiGAN, it should
produce audible speech.
"""
from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import load_fp32, save_fp32_1d
from conv import conv1d_kernel_fast
from layernorm import (
    linear_kernel, transpose_btc_to_bct_kernel, transpose_bct_to_btc_kernel,
)
from decoder_kernels import elu_kernel, abs_kernel
from source_signal import (
    f0_upsample_kernel, source_signal_full_kernel,
    stft_forward_kernel, hann_window_buf,
)


comptime B = 1
comptime MEL_C = 80
comptime MEL_T = 262
comptime D = 512
comptime UPSAMP = 480
comptime T_AUDIO = MEL_T * UPSAMP   # 125760
comptime N_FFT = 16
comptime HOP = 4
comptime N_BINS = N_FFT // 2 + 1    # 9
comptime T_FRAMES = T_AUDIO // HOP + 1   # 31441
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


def test_source_signal_chain() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/real/"
    var ctx = DeviceContext()

    var mel = load_fp32(fix + "e2e_mel_final.bin")
    print("[source] mel shape:", mel.shape[0], "x", mel.shape[1], "x", mel.shape[2])

    # f0_predictor weights.
    var w0 = upload_w(ctx, fix, "weights/mel2wav__f0_predictor__condnet__0__weight.bin")
    var b0 = upload_w(ctx, fix, "weights/mel2wav__f0_predictor__condnet__0__bias.bin")
    var w2 = upload_w(ctx, fix, "weights/mel2wav__f0_predictor__condnet__2__weight.bin")
    var b2 = upload_w(ctx, fix, "weights/mel2wav__f0_predictor__condnet__2__bias.bin")
    var w4 = upload_w(ctx, fix, "weights/mel2wav__f0_predictor__condnet__4__weight.bin")
    var b4 = upload_w(ctx, fix, "weights/mel2wav__f0_predictor__condnet__4__bias.bin")
    var w6 = upload_w(ctx, fix, "weights/mel2wav__f0_predictor__condnet__6__weight.bin")
    var b6 = upload_w(ctx, fix, "weights/mel2wav__f0_predictor__condnet__6__bias.bin")
    var w8 = upload_w(ctx, fix, "weights/mel2wav__f0_predictor__condnet__8__weight.bin")
    var b8 = upload_w(ctx, fix, "weights/mel2wav__f0_predictor__condnet__8__bias.bin")
    var cls_w = upload_w(ctx, fix, "weights/mel2wav__f0_predictor__classifier__weight.bin")
    var cls_b = upload_w(ctx, fix, "weights/mel2wav__f0_predictor__classifier__bias.bin")

    # SourceModule weights.
    var ml_w = upload_w(ctx, fix, "weights/m_source_l_linear_weight.bin")
    var ml_b = upload_w(ctx, fix, "weights/m_source_l_linear_bias.bin")
    # STFT window.
    var win = upload_w(ctx, fix, "weights/stft_window.bin")

    var n_mel = B * MEL_C * MEL_T
    var n_d = B * D * MEL_T
    var n_pre_abs = B * MEL_T
    var n_f0 = B * MEL_T
    var n_f0_up = B * 1 * T_AUDIO
    var n_s = B * 1 * T_AUDIO
    var n_stft = B * 2 * N_BINS * T_FRAMES

    var mel_buf = ctx.enqueue_create_buffer[DType.float32](n_mel)
    upload(mel_buf, mel.data, n_mel)

    # ---- f0_predictor.
    var a = ctx.enqueue_create_buffer[DType.float32](n_d)
    var bx = ctx.enqueue_create_buffer[DType.float32](n_d)
    var c = ctx.enqueue_create_buffer[DType.float32](n_d)
    var d = ctx.enqueue_create_buffer[DType.float32](n_d)
    var e = ctx.enqueue_create_buffer[DType.float32](n_d)

    comptime in_mel_layout = row_major[B, MEL_C, MEL_T]()
    comptime d_layout = row_major[B, D, MEL_T]()
    comptime w_in_layout = row_major[D, MEL_C, 3]()
    comptime w_dd_layout = row_major[D, D, 3]()
    comptime p_d_layout = row_major[D]()
    comptime flat_d = row_major[B * D * MEL_T]()
    comptime btd_d_layout = row_major[B, MEL_T, D]()

    var mel_t = TileTensor(mel_buf, in_mel_layout)
    var w0_t = TileTensor(w0, w_in_layout)
    var b0_t = TileTensor(b0, p_d_layout)
    var w2_t = TileTensor(w2, w_dd_layout)
    var b2_t = TileTensor(b2, p_d_layout)
    var w4_t = TileTensor(w4, w_dd_layout)
    var b4_t = TileTensor(b4, p_d_layout)
    var w6_t = TileTensor(w6, w_dd_layout)
    var b6_t = TileTensor(b6, p_d_layout)
    var w8_t = TileTensor(w8, w_dd_layout)
    var b8_t = TileTensor(b8, p_d_layout)
    var a_t = TileTensor(a, d_layout)
    var a_flat = TileTensor(a, flat_d)
    var bx_t = TileTensor(bx, d_layout)
    var bx_flat = TileTensor(bx, flat_d)
    var c_t = TileTensor(c, d_layout)
    var c_flat = TileTensor(c, flat_d)
    var d_t = TileTensor(d, d_layout)
    var d_flat = TileTensor(d, flat_d)
    var e_t = TileTensor(e, d_layout)

    # Conv1d(80→D) + ELU. The trick: I need 2 buffers per layer for conv+ELU pre/post.
    # To save buffers I'll use them alternately. Inline each conv+ELU pair.
    var pre_buf = ctx.enqueue_create_buffer[DType.float32](n_d)
    var pre_t = TileTensor(pre_buf, d_layout)
    var pre_flat = TileTensor(pre_buf, flat_d)
    comptime conv_in_k = conv1d_kernel_fast[
        DType.float32, type_of(in_mel_layout), type_of(w_in_layout),
        type_of(p_d_layout), type_of(d_layout),
        3, True, BLOCK,
    ]
    ctx.enqueue_function[conv_in_k, conv_in_k](
        pre_t, mel_t, w0_t, b0_t, B, MEL_C, D, MEL_T, MEL_T, 1, 1, 1,
        grid_dim=B * D, block_dim=BLOCK,
    )
    comptime elu_k = elu_kernel[
        DType.float32, type_of(flat_d), type_of(flat_d), BLOCK,
    ]
    ctx.enqueue_function[elu_k, elu_k](
        a_flat, pre_flat, n_d,
        grid_dim=ceildiv(n_d, BLOCK), block_dim=BLOCK,
    )

    comptime conv_dd_k = conv1d_kernel_fast[
        DType.float32, type_of(d_layout), type_of(w_dd_layout),
        type_of(p_d_layout), type_of(d_layout),
        3, True, BLOCK,
    ]
    ctx.enqueue_function[conv_dd_k, conv_dd_k](
        pre_t, a_t, w2_t, b2_t, B, D, D, MEL_T, MEL_T, 1, 1, 1,
        grid_dim=B * D, block_dim=BLOCK,
    )
    ctx.enqueue_function[elu_k, elu_k](
        bx_flat, pre_flat, n_d,
        grid_dim=ceildiv(n_d, BLOCK), block_dim=BLOCK,
    )
    ctx.enqueue_function[conv_dd_k, conv_dd_k](
        pre_t, bx_t, w4_t, b4_t, B, D, D, MEL_T, MEL_T, 1, 1, 1,
        grid_dim=B * D, block_dim=BLOCK,
    )
    ctx.enqueue_function[elu_k, elu_k](
        c_flat, pre_flat, n_d,
        grid_dim=ceildiv(n_d, BLOCK), block_dim=BLOCK,
    )
    ctx.enqueue_function[conv_dd_k, conv_dd_k](
        pre_t, c_t, w6_t, b6_t, B, D, D, MEL_T, MEL_T, 1, 1, 1,
        grid_dim=B * D, block_dim=BLOCK,
    )
    ctx.enqueue_function[elu_k, elu_k](
        d_flat, pre_flat, n_d,
        grid_dim=ceildiv(n_d, BLOCK), block_dim=BLOCK,
    )
    ctx.enqueue_function[conv_dd_k, conv_dd_k](
        pre_t, d_t, w8_t, b8_t, B, D, D, MEL_T, MEL_T, 1, 1, 1,
        grid_dim=B * D, block_dim=BLOCK,
    )
    ctx.enqueue_function[elu_k, elu_k](
        TileTensor(e, flat_d), pre_flat, n_d,
        grid_dim=ceildiv(n_d, BLOCK), block_dim=BLOCK,
    )

    # Transpose to (B, T, D) for Linear.
    var btc_buf = ctx.enqueue_create_buffer[DType.float32](n_d)
    var btc_t = TileTensor(btc_buf, btd_d_layout)
    comptime tp_k = transpose_bct_to_btc_kernel[
        DType.float32, type_of(d_layout), type_of(btd_d_layout), BLOCK,
    ]
    ctx.enqueue_function[tp_k, tp_k](
        btc_t, e_t, B, D, MEL_T, grid_dim=B * MEL_T, block_dim=BLOCK,
    )

    # Linear(D → 1, with bias).
    comptime pre_abs_layout = row_major[B, MEL_T, 1]()
    comptime w_cls_layout = row_major[1, D]()
    comptime p1_layout = row_major[1]()
    comptime flat_pre_abs = row_major[B * MEL_T]()
    var cls_w_t = TileTensor(cls_w, w_cls_layout)
    var cls_b_t = TileTensor(cls_b, p1_layout)
    var pre_abs_buf = ctx.enqueue_create_buffer[DType.float32](n_pre_abs)
    var pre_abs_t = TileTensor(pre_abs_buf, pre_abs_layout)
    var pre_abs_flat = TileTensor(pre_abs_buf, flat_pre_abs)
    comptime lin_k = linear_kernel[
        DType.float32, type_of(btd_d_layout), type_of(w_cls_layout),
        type_of(p1_layout), type_of(pre_abs_layout),
        True, BLOCK,
    ]
    ctx.enqueue_function[lin_k, lin_k](
        pre_abs_t, btc_t, cls_w_t, cls_b_t, B, MEL_T, D, 1,
        grid_dim=B * MEL_T, block_dim=BLOCK,
    )

    # abs → f0 (B, T_mel).
    var f0_buf = ctx.enqueue_create_buffer[DType.float32](n_f0)
    comptime f0_layout_2d = row_major[B, MEL_T]()
    var f0_t = TileTensor(f0_buf, f0_layout_2d)
    var f0_flat = TileTensor(f0_buf, flat_pre_abs)
    comptime abs_k = abs_kernel[
        DType.float32, type_of(flat_pre_abs), type_of(flat_pre_abs), BLOCK,
    ]
    ctx.enqueue_function[abs_k, abs_k](
        f0_flat, pre_abs_flat, n_f0,
        grid_dim=ceildiv(n_f0, BLOCK), block_dim=BLOCK,
    )

    # ---- f0_upsample (nearest by 480) → (B, 1, T_AUDIO).
    var f0_up_buf = ctx.enqueue_create_buffer[DType.float32](n_f0_up)
    comptime f0_up_layout = row_major[B, 1, T_AUDIO]()
    var f0_up_t = TileTensor(f0_up_buf, f0_up_layout)
    comptime up_k = f0_upsample_kernel[
        DType.float32, type_of(f0_layout_2d), type_of(f0_up_layout), BLOCK,
    ]
    ctx.enqueue_function[up_k, up_k](
        f0_up_t, f0_t, B, MEL_T, T_AUDIO, UPSAMP,
        grid_dim=B, block_dim=BLOCK,
    )

    # ---- SourceModuleHnNSF (sequential per batch).
    var s_buf = ctx.enqueue_create_buffer[DType.float32](n_s)
    var s_t = TileTensor(s_buf, f0_up_layout)
    comptime w_lin_src_layout = row_major[1, 9]()
    comptime b_lin_src_layout = row_major[1]()
    var ml_w_t = TileTensor(ml_w, w_lin_src_layout)
    var ml_b_t = TileTensor(ml_b, b_lin_src_layout)
    comptime src_k = source_signal_full_kernel[
        DType.float32,
        type_of(f0_up_layout),
        type_of(w_lin_src_layout),
        type_of(b_lin_src_layout),
        type_of(f0_up_layout),
        9, 1,
    ]
    ctx.enqueue_function[src_k, src_k](
        s_t, f0_up_t, ml_w_t, ml_b_t,
        B, T_AUDIO,
        Float32(24000.0),   # sampling_rate
        Float32(0.1),       # sine_amp
        Float32(0.003),     # noise_std
        Float32(10.0),      # voiced_threshold
        42,                  # noise_seed
        grid_dim=B, block_dim=1,
    )

    # ---- STFT → s_stft_cat (B, 2*N_BINS, T_FRAMES).
    var stft_buf = ctx.enqueue_create_buffer[DType.float32](n_stft)
    comptime s_flat_layout = row_major[B, T_AUDIO]()
    comptime win_layout = row_major[N_FFT]()
    comptime stft_layout = row_major[B, 2 * N_BINS, T_FRAMES]()
    var s_flat_t = TileTensor(s_buf, s_flat_layout)
    var win_t = TileTensor(win, win_layout)
    var stft_t = TileTensor(stft_buf, stft_layout)
    comptime stft_k = stft_forward_kernel[
        DType.float32, type_of(s_flat_layout), type_of(win_layout), type_of(stft_layout),
        N_FFT, HOP, N_BINS,
        BLOCK,
    ]
    ctx.enqueue_function[stft_k, stft_k](
        stft_t, s_flat_t, win_t, B, T_AUDIO, T_FRAMES,
        grid_dim=B * T_FRAMES, block_dim=BLOCK,
    )

    ctx.synchronize()
    print("[source] computed s_stft_cat, shape: (", B, ",", 2 * N_BINS, ",", T_FRAMES, ")")

    # Save the s_stft_cat for downstream HiFiGAN consumption.
    var samples = List[Float32]()
    with stft_buf.map_to_host() as h:
        for i in range(n_stft):
            samples.append(h[i])
    save_fp32_1d("tests/fixtures/real/cloned_voice_s_stft_mojo.bin", samples)
    print("[source] saved cloned_voice_s_stft_mojo.bin (", len(samples), "values)")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
