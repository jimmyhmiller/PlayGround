"""
Pure-Mojo audio → CAMPPlus xvector pipeline. Bundles:
  ref_wav (1, 160000) at 16kHz → Kaldi fbank (998, 80) → centered fbank
  → CAMPPlus FCM (conv1+bn1+relu+layer1+layer2+conv2+bn2+relu+reshape)
    → fcm_out (1, 320, 998)
  → CAMPPlus xvector trunk (tdnn + 3 dense blocks + 3 transit + out_nonlinear + stats_pool + dense)
    → xvector (1, 192)

Inputs:
  ref_wav_buf:  DeviceBuffer with 160000 fp32 samples (16kHz mono, normalized to [-1, 1])
  fix:          path prefix for CAMPPlus weights (e.g. "tests/fixtures/campplus/")

Output: DeviceBuffer[DType.float32] of size 192 (the xvector).

All shapes and weights mirror those verified in test_voice_embed_e2e.mojo and
test_campplus_full.mojo (which produce upstream-matching xvectors).
"""
from std.math import ceildiv
from std.gpu import block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major

from fixture import load_fp32
from kaldi_fbank import (
    frame_preprocess_kernel, naive_rfft_power_kernel,
    mel_filterbank_log_kernel, subtract_per_utterance_mean_kernel,
    povey_window_fp32, mel_filterbank_fp32,
)
from conv import conv1d_kernel_fast, conv2d_kernel, batchnorm1d_kernel, batchnorm2d_kernel, relu_kernel
from campplus import (
    basic_resblock, cam_dense_tdnn_layer, transit_layer, DenseTdnnWeights,
)
from stats_kernels import stats_pool_kernel, bn_no_affine_2d_kernel


comptime N_SAMPLES = 160000
comptime WINDOW_SIZE = 400
comptime WINDOW_SHIFT = 160
comptime PADDED = 512
comptime NUM_BINS = PADDED // 2 + 1
comptime NUM_MEL = 80
comptime SAMPLE_RATE: Float32 = 16000.0
comptime LOW_FREQ: Float32 = 20.0
comptime HIGH_FREQ: Float32 = 0.0
comptime PREEMPHASIS: Float32 = 0.97
comptime EPS_FBANK: Float32 = 1.1920929e-7
comptime EPS: Float32 = 1.0e-5
comptime BLOCK = 256
comptime M = 1 + (N_SAMPLES - WINDOW_SIZE) // WINDOW_SHIFT   # 998
comptime T_TRUNK = 499
comptime GROWTH = 32
comptime BN_C = 128
comptime HALF_BN = 64
comptime KERNEL = 3
comptime SEG_LEN = 100


def _upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def _upload_w(mut ctx: DeviceContext, fix: String, name: String) raises -> DeviceBuffer[DType.float32]:
    var t = load_fp32(fix + name)
    var n = len(t.data)
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    _upload(buf, t.data, n)
    return buf^


def _load_dense(mut ctx: DeviceContext, fix: String, block_id: Int, layer_idx: Int) raises -> DenseTdnnWeights:
    var prefix = "weights/xvector__block" + String(block_id) + "__tdnnd" + String(layer_idx + 1) + "__"
    return DenseTdnnWeights(
        _upload_w(ctx, fix, prefix + "nonlinear1__batchnorm__weight.bin"),
        _upload_w(ctx, fix, prefix + "nonlinear1__batchnorm__bias.bin"),
        _upload_w(ctx, fix, prefix + "nonlinear1__batchnorm__running_mean.bin"),
        _upload_w(ctx, fix, prefix + "nonlinear1__batchnorm__running_var.bin"),
        _upload_w(ctx, fix, prefix + "linear1__weight.bin"),
        _upload_w(ctx, fix, prefix + "nonlinear2__batchnorm__weight.bin"),
        _upload_w(ctx, fix, prefix + "nonlinear2__batchnorm__bias.bin"),
        _upload_w(ctx, fix, prefix + "nonlinear2__batchnorm__running_mean.bin"),
        _upload_w(ctx, fix, prefix + "nonlinear2__batchnorm__running_var.bin"),
        _upload_w(ctx, fix, prefix + "cam_layer__linear_local__weight.bin"),
        _upload_w(ctx, fix, prefix + "cam_layer__linear1__weight.bin"),
        _upload_w(ctx, fix, prefix + "cam_layer__linear1__bias.bin"),
        _upload_w(ctx, fix, prefix + "cam_layer__linear2__weight.bin"),
        _upload_w(ctx, fix, prefix + "cam_layer__linear2__bias.bin"),
    )


def _transpose_fbank_kernel[
    dtype: DType, InLayout: TensorLayout, OutLayout: TensorLayout, BLOCK_: Int,
](
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],
    inp: TileTensor[dtype, InLayout, MutAnyOrigin],
    m: Int, c: Int,
):
    """Reshape (M, 80) → (1, 1, 80, M)."""
    comptime assert inp.flat_rank == 2
    comptime assert output.flat_rank == 4
    var b_idx = block_idx.x
    var tid = thread_idx.x
    var t = tid
    while t < m:
        var v = rebind[Scalar[dtype]](inp[t, b_idx]).cast[DType.float32]()
        output[0, 0, b_idx, t] = rebind[output.ElementType](v.cast[dtype]())
        t += BLOCK_


def audio_to_xvector(
    mut ctx: DeviceContext,
    mut ref_wav_buf: DeviceBuffer[DType.float32],   # (160000,) fp32 16kHz mono
    fix: String,                                    # weights path prefix, e.g. "tests/fixtures/campplus/"
) raises -> DeviceBuffer[DType.float32]:
    """Audio (1, 160000) → xvector (1, 192). All Mojo, mirrors upstream chatterbox CAMPPlus."""
    # ===== Phase 1: kaldi fbank =====
    var win = povey_window_fp32(WINDOW_SIZE)
    var bank = mel_filterbank_fp32(NUM_MEL, PADDED, SAMPLE_RATE, LOW_FREQ, HIGH_FREQ)
    var win_buf = ctx.enqueue_create_buffer[DType.float32](WINDOW_SIZE)
    var bank_buf = ctx.enqueue_create_buffer[DType.float32](NUM_MEL * NUM_BINS)
    var frames_buf = ctx.enqueue_create_buffer[DType.float32](M * PADDED)
    var power_buf = ctx.enqueue_create_buffer[DType.float32](M * NUM_BINS)
    var mel_buf = ctx.enqueue_create_buffer[DType.float32](M * NUM_MEL)
    var fbank_buf = ctx.enqueue_create_buffer[DType.float32](M * NUM_MEL)
    _upload(win_buf, win, WINDOW_SIZE)
    _upload(bank_buf, bank, NUM_MEL * NUM_BINS)

    comptime wav_layout = row_major[N_SAMPLES]()
    comptime win_layout = row_major[WINDOW_SIZE]()
    comptime bank_layout = row_major[NUM_MEL, NUM_BINS]()
    comptime frames_layout = row_major[M, PADDED]()
    comptime power_layout = row_major[M, NUM_BINS]()
    comptime mel_layout = row_major[M, NUM_MEL]()

    var wav_t = TileTensor(ref_wav_buf, wav_layout)
    var win_t = TileTensor(win_buf, win_layout)
    var bank_t = TileTensor(bank_buf, bank_layout)
    var frames_t = TileTensor(frames_buf, frames_layout)
    var power_t = TileTensor(power_buf, power_layout)
    var mel_t = TileTensor(mel_buf, mel_layout)
    var fbank_t = TileTensor(fbank_buf, mel_layout)

    comptime pre_k = frame_preprocess_kernel[
        DType.float32, type_of(wav_layout), type_of(frames_layout), type_of(win_layout),
        WINDOW_SIZE, PADDED, BLOCK,
    ]
    ctx.enqueue_function[pre_k, pre_k](
        frames_t, wav_t, win_t, M, WINDOW_SHIFT, PREEMPHASIS,
        grid_dim=M, block_dim=BLOCK,
    )
    comptime fft_k = naive_rfft_power_kernel[
        DType.float32, type_of(frames_layout), type_of(power_layout),
        PADDED, NUM_BINS, BLOCK,
    ]
    ctx.enqueue_function[fft_k, fft_k](power_t, frames_t, M, grid_dim=M, block_dim=BLOCK)
    comptime melf_k = mel_filterbank_log_kernel[
        DType.float32, type_of(power_layout), type_of(bank_layout), type_of(mel_layout),
        NUM_BINS, NUM_MEL, BLOCK,
    ]
    ctx.enqueue_function[melf_k, melf_k](
        mel_t, power_t, bank_t, M, EPS_FBANK,
        grid_dim=M, block_dim=BLOCK,
    )
    comptime sub_k = subtract_per_utterance_mean_kernel[
        DType.float32, type_of(mel_layout), type_of(mel_layout),
        NUM_MEL, BLOCK,
    ]
    ctx.enqueue_function[sub_k, sub_k](
        fbank_t, mel_t, M, grid_dim=NUM_MEL, block_dim=BLOCK,
    )

    # ===== Phase 2: transpose to FCM input (1, 1, 80, M) =====
    var fcm_in_buf = ctx.enqueue_create_buffer[DType.float32](1 * 1 * NUM_MEL * M)
    comptime fcm_in_layout = row_major[1, 1, NUM_MEL, M]()
    var fcm_in_t = TileTensor(fcm_in_buf, fcm_in_layout)
    comptime tp_k = _transpose_fbank_kernel[
        DType.float32, type_of(mel_layout), type_of(fcm_in_layout), BLOCK,
    ]
    ctx.enqueue_function[tp_k, tp_k](
        fcm_in_t, fbank_t, M, NUM_MEL,
        grid_dim=NUM_MEL, block_dim=BLOCK,
    )

    # ===== Phase 3: CAMPPlus FCM head =====
    var w_conv1 = _upload_w(ctx, fix, "weights/head__conv1__weight.bin")
    var bn1_w = _upload_w(ctx, fix, "weights/head__bn1__weight.bin")
    var bn1_b = _upload_w(ctx, fix, "weights/head__bn1__bias.bin")
    var bn1_m = _upload_w(ctx, fix, "weights/head__bn1__running_mean.bin")
    var bn1_v = _upload_w(ctx, fix, "weights/head__bn1__running_var.bin")
    var L10_c1 = _upload_w(ctx, fix, "weights/head__layer1__0__conv1__weight.bin")
    var L10_c2 = _upload_w(ctx, fix, "weights/head__layer1__0__conv2__weight.bin")
    var L10_bn1w = _upload_w(ctx, fix, "weights/head__layer1__0__bn1__weight.bin")
    var L10_bn1b = _upload_w(ctx, fix, "weights/head__layer1__0__bn1__bias.bin")
    var L10_bn1m = _upload_w(ctx, fix, "weights/head__layer1__0__bn1__running_mean.bin")
    var L10_bn1v = _upload_w(ctx, fix, "weights/head__layer1__0__bn1__running_var.bin")
    var L10_bn2w = _upload_w(ctx, fix, "weights/head__layer1__0__bn2__weight.bin")
    var L10_bn2b = _upload_w(ctx, fix, "weights/head__layer1__0__bn2__bias.bin")
    var L10_bn2m = _upload_w(ctx, fix, "weights/head__layer1__0__bn2__running_mean.bin")
    var L10_bn2v = _upload_w(ctx, fix, "weights/head__layer1__0__bn2__running_var.bin")
    var L10_scw = _upload_w(ctx, fix, "weights/head__layer1__0__shortcut__0__weight.bin")
    var L10_scbnw = _upload_w(ctx, fix, "weights/head__layer1__0__shortcut__1__weight.bin")
    var L10_scbnb = _upload_w(ctx, fix, "weights/head__layer1__0__shortcut__1__bias.bin")
    var L10_scbnm = _upload_w(ctx, fix, "weights/head__layer1__0__shortcut__1__running_mean.bin")
    var L10_scbnv = _upload_w(ctx, fix, "weights/head__layer1__0__shortcut__1__running_var.bin")
    var L11_c1 = _upload_w(ctx, fix, "weights/head__layer1__1__conv1__weight.bin")
    var L11_c2 = _upload_w(ctx, fix, "weights/head__layer1__1__conv2__weight.bin")
    var L11_bn1w = _upload_w(ctx, fix, "weights/head__layer1__1__bn1__weight.bin")
    var L11_bn1b = _upload_w(ctx, fix, "weights/head__layer1__1__bn1__bias.bin")
    var L11_bn1m = _upload_w(ctx, fix, "weights/head__layer1__1__bn1__running_mean.bin")
    var L11_bn1v = _upload_w(ctx, fix, "weights/head__layer1__1__bn1__running_var.bin")
    var L11_bn2w = _upload_w(ctx, fix, "weights/head__layer1__1__bn2__weight.bin")
    var L11_bn2b = _upload_w(ctx, fix, "weights/head__layer1__1__bn2__bias.bin")
    var L11_bn2m = _upload_w(ctx, fix, "weights/head__layer1__1__bn2__running_mean.bin")
    var L11_bn2v = _upload_w(ctx, fix, "weights/head__layer1__1__bn2__running_var.bin")
    var L20_c1 = _upload_w(ctx, fix, "weights/head__layer2__0__conv1__weight.bin")
    var L20_c2 = _upload_w(ctx, fix, "weights/head__layer2__0__conv2__weight.bin")
    var L20_bn1w = _upload_w(ctx, fix, "weights/head__layer2__0__bn1__weight.bin")
    var L20_bn1b = _upload_w(ctx, fix, "weights/head__layer2__0__bn1__bias.bin")
    var L20_bn1m = _upload_w(ctx, fix, "weights/head__layer2__0__bn1__running_mean.bin")
    var L20_bn1v = _upload_w(ctx, fix, "weights/head__layer2__0__bn1__running_var.bin")
    var L20_bn2w = _upload_w(ctx, fix, "weights/head__layer2__0__bn2__weight.bin")
    var L20_bn2b = _upload_w(ctx, fix, "weights/head__layer2__0__bn2__bias.bin")
    var L20_bn2m = _upload_w(ctx, fix, "weights/head__layer2__0__bn2__running_mean.bin")
    var L20_bn2v = _upload_w(ctx, fix, "weights/head__layer2__0__bn2__running_var.bin")
    var L20_scw = _upload_w(ctx, fix, "weights/head__layer2__0__shortcut__0__weight.bin")
    var L20_scbnw = _upload_w(ctx, fix, "weights/head__layer2__0__shortcut__1__weight.bin")
    var L20_scbnb = _upload_w(ctx, fix, "weights/head__layer2__0__shortcut__1__bias.bin")
    var L20_scbnm = _upload_w(ctx, fix, "weights/head__layer2__0__shortcut__1__running_mean.bin")
    var L20_scbnv = _upload_w(ctx, fix, "weights/head__layer2__0__shortcut__1__running_var.bin")
    var L21_c1 = _upload_w(ctx, fix, "weights/head__layer2__1__conv1__weight.bin")
    var L21_c2 = _upload_w(ctx, fix, "weights/head__layer2__1__conv2__weight.bin")
    var L21_bn1w = _upload_w(ctx, fix, "weights/head__layer2__1__bn1__weight.bin")
    var L21_bn1b = _upload_w(ctx, fix, "weights/head__layer2__1__bn1__bias.bin")
    var L21_bn1m = _upload_w(ctx, fix, "weights/head__layer2__1__bn1__running_mean.bin")
    var L21_bn1v = _upload_w(ctx, fix, "weights/head__layer2__1__bn1__running_var.bin")
    var L21_bn2w = _upload_w(ctx, fix, "weights/head__layer2__1__bn2__weight.bin")
    var L21_bn2b = _upload_w(ctx, fix, "weights/head__layer2__1__bn2__bias.bin")
    var L21_bn2m = _upload_w(ctx, fix, "weights/head__layer2__1__bn2__running_mean.bin")
    var L21_bn2v = _upload_w(ctx, fix, "weights/head__layer2__1__bn2__running_var.bin")
    var w_conv2 = _upload_w(ctx, fix, "weights/head__conv2__weight.bin")
    var bn2_w = _upload_w(ctx, fix, "weights/head__bn2__weight.bin")
    var bn2_b = _upload_w(ctx, fix, "weights/head__bn2__bias.bin")
    var bn2_m = _upload_w(ctx, fix, "weights/head__bn2__running_mean.bin")
    var bn2_v = _upload_w(ctx, fix, "weights/head__bn2__running_var.bin")

    var dummy1 = ctx.enqueue_create_buffer[DType.float32](32)
    var dummy2 = ctx.enqueue_create_buffer[DType.float32](1)
    var conv1_out = ctx.enqueue_create_buffer[DType.float32](1 * 32 * 80 * M)
    var bn1_out = ctx.enqueue_create_buffer[DType.float32](1 * 32 * 80 * M)
    var relu1_out = ctx.enqueue_create_buffer[DType.float32](1 * 32 * 80 * M)

    comptime mid32_layout = row_major[1, 32, 80, M]()
    comptime w31_layout = row_major[32, 1, 3, 3]()
    comptime p32_layout = row_major[32]()
    comptime p1_layout = row_major[1]()
    comptime relu1_flat = row_major[1 * 32 * 80 * M]()

    var w_t = TileTensor(w_conv1, w31_layout)
    var dummy1_t = TileTensor(dummy1, p32_layout)
    var dummy2_t = TileTensor(dummy2, p1_layout)
    var conv1_t = TileTensor(conv1_out, mid32_layout)
    var bn1_w_t = TileTensor(bn1_w, p32_layout)
    var bn1_b_t = TileTensor(bn1_b, p32_layout)
    var bn1_m_t = TileTensor(bn1_m, p32_layout)
    var bn1_v_t = TileTensor(bn1_v, p32_layout)
    var bn1_t = TileTensor(bn1_out, mid32_layout)
    var bn1_flat_t = TileTensor(bn1_out, relu1_flat)
    var relu1_flat_t = TileTensor(relu1_out, relu1_flat)
    comptime conv1_k = conv2d_kernel[
        DType.float32, type_of(fcm_in_layout), type_of(w31_layout),
        type_of(p1_layout), type_of(mid32_layout),
        3, 3, False, BLOCK,
    ]
    ctx.enqueue_function[conv1_k, conv1_k](
        conv1_t, fcm_in_t, w_t, dummy2_t,
        1, 1, 32, 80, M, 80, M, 1, 1, 1, 1,
        grid_dim=1 * 32 * 80, block_dim=BLOCK,
    )
    comptime bn_k = batchnorm2d_kernel[
        DType.float32, type_of(mid32_layout), type_of(p32_layout),
        type_of(mid32_layout), BLOCK,
    ]
    ctx.enqueue_function[bn_k, bn_k](
        bn1_t, conv1_t, bn1_w_t, bn1_b_t, bn1_m_t, bn1_v_t,
        1, 32, 80, M, EPS,
        grid_dim=1 * 32 * 80, block_dim=BLOCK,
    )
    comptime relu_k = relu_kernel[
        DType.float32, type_of(relu1_flat), type_of(relu1_flat), BLOCK,
    ]
    ctx.enqueue_function[relu_k, relu_k](
        relu1_flat_t, bn1_flat_t, 1 * 32 * 80 * M,
        grid_dim=ceildiv(1 * 32 * 80 * M, BLOCK), block_dim=BLOCK,
    )

    # FCM layer1, layer2.
    var l1_0_out = ctx.enqueue_create_buffer[DType.float32](1 * 32 * 40 * M)
    var l1_1_out = ctx.enqueue_create_buffer[DType.float32](1 * 32 * 40 * M)
    var l2_0_out = ctx.enqueue_create_buffer[DType.float32](1 * 32 * 20 * M)
    var l2_1_out = ctx.enqueue_create_buffer[DType.float32](1 * 32 * 20 * M)
    var rb_dummy = ctx.enqueue_create_buffer[DType.float32](32)
    basic_resblock[1, 32, 32, 80, 40, M, True, 2](
        ctx, relu1_out, l1_0_out,
        L10_c1, L10_c2,
        L10_bn1w, L10_bn1b, L10_bn1m, L10_bn1v,
        L10_bn2w, L10_bn2b, L10_bn2m, L10_bn2v,
        L10_scw, L10_scbnw, L10_scbnb, L10_scbnm, L10_scbnv, rb_dummy,
    )
    var u11a = ctx.enqueue_create_buffer[DType.float32](32)
    var u11b = ctx.enqueue_create_buffer[DType.float32](32)
    var u11c = ctx.enqueue_create_buffer[DType.float32](32)
    var u11d = ctx.enqueue_create_buffer[DType.float32](32)
    var u11e = ctx.enqueue_create_buffer[DType.float32](32)
    var rb_dummy_11 = ctx.enqueue_create_buffer[DType.float32](32)
    basic_resblock[1, 32, 32, 40, 40, M, False, 1](
        ctx, l1_0_out, l1_1_out,
        L11_c1, L11_c2,
        L11_bn1w, L11_bn1b, L11_bn1m, L11_bn1v,
        L11_bn2w, L11_bn2b, L11_bn2m, L11_bn2v,
        u11a, u11b, u11c, u11d, u11e, rb_dummy_11,
    )
    var rb_dummy_20 = ctx.enqueue_create_buffer[DType.float32](32)
    basic_resblock[1, 32, 32, 40, 20, M, True, 2](
        ctx, l1_1_out, l2_0_out,
        L20_c1, L20_c2,
        L20_bn1w, L20_bn1b, L20_bn1m, L20_bn1v,
        L20_bn2w, L20_bn2b, L20_bn2m, L20_bn2v,
        L20_scw, L20_scbnw, L20_scbnb, L20_scbnm, L20_scbnv, rb_dummy_20,
    )
    var u21a = ctx.enqueue_create_buffer[DType.float32](32)
    var u21b = ctx.enqueue_create_buffer[DType.float32](32)
    var u21c = ctx.enqueue_create_buffer[DType.float32](32)
    var u21d = ctx.enqueue_create_buffer[DType.float32](32)
    var u21e = ctx.enqueue_create_buffer[DType.float32](32)
    var rb_dummy_21 = ctx.enqueue_create_buffer[DType.float32](32)
    basic_resblock[1, 32, 32, 20, 20, M, False, 1](
        ctx, l2_0_out, l2_1_out,
        L21_c1, L21_c2,
        L21_bn1w, L21_bn1b, L21_bn1m, L21_bn1v,
        L21_bn2w, L21_bn2b, L21_bn2m, L21_bn2v,
        u21a, u21b, u21c, u21d, u21e, rb_dummy_21,
    )

    var n_post2 = 1 * 32 * 10 * M
    var c2_out = ctx.enqueue_create_buffer[DType.float32](n_post2)
    var bn2_out = ctx.enqueue_create_buffer[DType.float32](n_post2)
    var final_out = ctx.enqueue_create_buffer[DType.float32](n_post2)
    comptime l21_layout = row_major[1, 32, 20, M]()
    comptime final_layout = row_major[1, 32, 10, M]()
    comptime w_3232 = row_major[32, 32, 3, 3]()
    comptime final_flat = row_major[1 * 32 * 10 * M]()
    var l21_t = TileTensor(l2_1_out, l21_layout)
    var w_conv2_t = TileTensor(w_conv2, w_3232)
    var c2_out_t = TileTensor(c2_out, final_layout)
    var bn2_w_t = TileTensor(bn2_w, p32_layout)
    var bn2_b_t = TileTensor(bn2_b, p32_layout)
    var bn2_m_t = TileTensor(bn2_m, p32_layout)
    var bn2_v_t = TileTensor(bn2_v, p32_layout)
    var bn2_out_t = TileTensor(bn2_out, final_layout)
    var bn2_flat_t = TileTensor(bn2_out, final_flat)
    var final_flat_t = TileTensor(final_out, final_flat)
    comptime conv2_k = conv2d_kernel[
        DType.float32, type_of(l21_layout), type_of(w_3232),
        type_of(p32_layout), type_of(final_layout),
        3, 3, False, BLOCK,
    ]
    ctx.enqueue_function[conv2_k, conv2_k](
        c2_out_t, l21_t, w_conv2_t, dummy1_t,
        1, 32, 32, 20, M, 10, M, 2, 1, 1, 1,
        grid_dim=1 * 32 * 10, block_dim=BLOCK,
    )
    comptime bn2_k = batchnorm2d_kernel[
        DType.float32, type_of(final_layout), type_of(p32_layout),
        type_of(final_layout), BLOCK,
    ]
    ctx.enqueue_function[bn2_k, bn2_k](
        bn2_out_t, c2_out_t, bn2_w_t, bn2_b_t, bn2_m_t, bn2_v_t,
        1, 32, 10, M, EPS,
        grid_dim=1 * 32 * 10, block_dim=BLOCK,
    )
    comptime final_relu = relu_kernel[
        DType.float32, type_of(final_flat), type_of(final_flat), BLOCK,
    ]
    ctx.enqueue_function[final_relu, final_relu](
        final_flat_t, bn2_flat_t, n_post2,
        grid_dim=ceildiv(n_post2, BLOCK), block_dim=BLOCK,
    )

    # ===== Phase 4: TDNN + 3 dense blocks + 3 transits + out_nonlinear + stats + dense =====
    var w_tdnn = _upload_w(ctx, fix, "weights/xvector__tdnn__linear__weight.bin")
    var tdnn_bn_w = _upload_w(ctx, fix, "weights/xvector__tdnn__nonlinear__batchnorm__weight.bin")
    var tdnn_bn_b = _upload_w(ctx, fix, "weights/xvector__tdnn__nonlinear__batchnorm__bias.bin")
    var tdnn_bn_m = _upload_w(ctx, fix, "weights/xvector__tdnn__nonlinear__batchnorm__running_mean.bin")
    var tdnn_bn_v = _upload_w(ctx, fix, "weights/xvector__tdnn__nonlinear__batchnorm__running_var.bin")
    var tdnn_bias_dummy = ctx.enqueue_create_buffer[DType.float32](128)

    var n_tdnn = 1 * 128 * T_TRUNK
    var tdnn_pre = ctx.enqueue_create_buffer[DType.float32](n_tdnn)
    var tdnn_bn = ctx.enqueue_create_buffer[DType.float32](n_tdnn)
    var tdnn_out = ctx.enqueue_create_buffer[DType.float32](n_tdnn)
    comptime fcm_out_layout = row_major[1, 320, M]()
    comptime tdnn_w_layout = row_major[128, 320, 5]()
    comptime tdnn_layout = row_major[1, 128, T_TRUNK]()
    comptime tdnn_p_layout = row_major[128]()
    comptime tdnn_flat = row_major[1 * 128 * T_TRUNK]()
    var fcm_out_t = TileTensor(final_out, fcm_out_layout)
    var w_tdnn_t = TileTensor(w_tdnn, tdnn_w_layout)
    var tdnn_bias_t = TileTensor(tdnn_bias_dummy, tdnn_p_layout)
    var tdnn_pre_t = TileTensor(tdnn_pre, tdnn_layout)
    var tdnn_bn_w_t = TileTensor(tdnn_bn_w, tdnn_p_layout)
    var tdnn_bn_b_t = TileTensor(tdnn_bn_b, tdnn_p_layout)
    var tdnn_bn_m_t = TileTensor(tdnn_bn_m, tdnn_p_layout)
    var tdnn_bn_v_t = TileTensor(tdnn_bn_v, tdnn_p_layout)
    var tdnn_bn_t = TileTensor(tdnn_bn, tdnn_layout)
    var tdnn_bn_flat = TileTensor(tdnn_bn, tdnn_flat)
    var tdnn_out_flat = TileTensor(tdnn_out, tdnn_flat)
    comptime tdnn_conv_k = conv1d_kernel_fast[
        DType.float32, type_of(fcm_out_layout), type_of(tdnn_w_layout),
        type_of(tdnn_p_layout), type_of(tdnn_layout),
        5, False, BLOCK,
    ]
    ctx.enqueue_function[tdnn_conv_k, tdnn_conv_k](
        tdnn_pre_t, fcm_out_t, w_tdnn_t, tdnn_bias_t,
        1, 320, 128, M, T_TRUNK, 2, 2, 1,
        grid_dim=1 * 128, block_dim=BLOCK,
    )
    comptime tdnn_bn_k = batchnorm1d_kernel[
        DType.float32, type_of(tdnn_layout), type_of(tdnn_p_layout),
        type_of(tdnn_layout), BLOCK,
    ]
    ctx.enqueue_function[tdnn_bn_k, tdnn_bn_k](
        tdnn_bn_t, tdnn_pre_t, tdnn_bn_w_t, tdnn_bn_b_t, tdnn_bn_m_t, tdnn_bn_v_t,
        1, 128, T_TRUNK, EPS,
        grid_dim=1 * 128, block_dim=BLOCK,
    )
    comptime tdnn_relu_k = relu_kernel[
        DType.float32, type_of(tdnn_flat), type_of(tdnn_flat), BLOCK,
    ]
    ctx.enqueue_function[tdnn_relu_k, tdnn_relu_k](
        tdnn_out_flat, tdnn_bn_flat, n_tdnn,
        grid_dim=ceildiv(n_tdnn, BLOCK), block_dim=BLOCK,
    )

    # block1: 12 dense layers.
    var b1_w = List[DenseTdnnWeights]()
    for i in range(12):
        b1_w.append(_load_dense(ctx, fix, 1, i))
    var b1_dummy = List[DeviceBuffer[DType.float32]]()
    for i in range(12):
        b1_dummy.append(ctx.enqueue_create_buffer[DType.float32](BN_C))
    var b1_out0 = ctx.enqueue_create_buffer[DType.float32](1 * 160 * T_TRUNK)
    var b1_out1 = ctx.enqueue_create_buffer[DType.float32](1 * 192 * T_TRUNK)
    var b1_out2 = ctx.enqueue_create_buffer[DType.float32](1 * 224 * T_TRUNK)
    var b1_out3 = ctx.enqueue_create_buffer[DType.float32](1 * 256 * T_TRUNK)
    var b1_out4 = ctx.enqueue_create_buffer[DType.float32](1 * 288 * T_TRUNK)
    var b1_out5 = ctx.enqueue_create_buffer[DType.float32](1 * 320 * T_TRUNK)
    var b1_out6 = ctx.enqueue_create_buffer[DType.float32](1 * 352 * T_TRUNK)
    var b1_out7 = ctx.enqueue_create_buffer[DType.float32](1 * 384 * T_TRUNK)
    var b1_out8 = ctx.enqueue_create_buffer[DType.float32](1 * 416 * T_TRUNK)
    var b1_out9 = ctx.enqueue_create_buffer[DType.float32](1 * 448 * T_TRUNK)
    var b1_out10 = ctx.enqueue_create_buffer[DType.float32](1 * 480 * T_TRUNK)
    var b1_out11 = ctx.enqueue_create_buffer[DType.float32](1 * 512 * T_TRUNK)
    cam_dense_tdnn_layer[1, 128, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 1, SEG_LEN](ctx, tdnn_out, b1_out0, b1_w[0], b1_dummy[0])
    cam_dense_tdnn_layer[1, 160, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 1, SEG_LEN](ctx, b1_out0, b1_out1, b1_w[1], b1_dummy[1])
    cam_dense_tdnn_layer[1, 192, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 1, SEG_LEN](ctx, b1_out1, b1_out2, b1_w[2], b1_dummy[2])
    cam_dense_tdnn_layer[1, 224, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 1, SEG_LEN](ctx, b1_out2, b1_out3, b1_w[3], b1_dummy[3])
    cam_dense_tdnn_layer[1, 256, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 1, SEG_LEN](ctx, b1_out3, b1_out4, b1_w[4], b1_dummy[4])
    cam_dense_tdnn_layer[1, 288, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 1, SEG_LEN](ctx, b1_out4, b1_out5, b1_w[5], b1_dummy[5])
    cam_dense_tdnn_layer[1, 320, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 1, SEG_LEN](ctx, b1_out5, b1_out6, b1_w[6], b1_dummy[6])
    cam_dense_tdnn_layer[1, 352, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 1, SEG_LEN](ctx, b1_out6, b1_out7, b1_w[7], b1_dummy[7])
    cam_dense_tdnn_layer[1, 384, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 1, SEG_LEN](ctx, b1_out7, b1_out8, b1_w[8], b1_dummy[8])
    cam_dense_tdnn_layer[1, 416, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 1, SEG_LEN](ctx, b1_out8, b1_out9, b1_w[9], b1_dummy[9])
    cam_dense_tdnn_layer[1, 448, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 1, SEG_LEN](ctx, b1_out9, b1_out10, b1_w[10], b1_dummy[10])
    cam_dense_tdnn_layer[1, 480, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 1, SEG_LEN](ctx, b1_out10, b1_out11, b1_w[11], b1_dummy[11])

    # transit1.
    var t1_bn_w = _upload_w(ctx, fix, "weights/xvector__transit1__nonlinear__batchnorm__weight.bin")
    var t1_bn_b = _upload_w(ctx, fix, "weights/xvector__transit1__nonlinear__batchnorm__bias.bin")
    var t1_bn_m = _upload_w(ctx, fix, "weights/xvector__transit1__nonlinear__batchnorm__running_mean.bin")
    var t1_bn_v = _upload_w(ctx, fix, "weights/xvector__transit1__nonlinear__batchnorm__running_var.bin")
    var t1_lin_w = _upload_w(ctx, fix, "weights/xvector__transit1__linear__weight.bin")
    var t1_lin_b_dummy = ctx.enqueue_create_buffer[DType.float32](256)
    var t1_dummy = ctx.enqueue_create_buffer[DType.float32](256)
    var t1_out = ctx.enqueue_create_buffer[DType.float32](1 * 256 * T_TRUNK)
    transit_layer[1, 512, 256, T_TRUNK, False](
        ctx, b1_out11, t1_out,
        t1_bn_w, t1_bn_b, t1_bn_m, t1_bn_v,
        t1_lin_w, t1_lin_b_dummy, t1_dummy)

    # block2: 24 dense layers.
    var b2_w = List[DenseTdnnWeights]()
    for i in range(24):
        b2_w.append(_load_dense(ctx, fix, 2, i))
    var b2_dummy = List[DeviceBuffer[DType.float32]]()
    for i in range(24):
        b2_dummy.append(ctx.enqueue_create_buffer[DType.float32](BN_C))
    var b2_out0 = ctx.enqueue_create_buffer[DType.float32](1 * 288 * T_TRUNK)
    var b2_out1 = ctx.enqueue_create_buffer[DType.float32](1 * 320 * T_TRUNK)
    var b2_out2 = ctx.enqueue_create_buffer[DType.float32](1 * 352 * T_TRUNK)
    var b2_out3 = ctx.enqueue_create_buffer[DType.float32](1 * 384 * T_TRUNK)
    var b2_out4 = ctx.enqueue_create_buffer[DType.float32](1 * 416 * T_TRUNK)
    var b2_out5 = ctx.enqueue_create_buffer[DType.float32](1 * 448 * T_TRUNK)
    var b2_out6 = ctx.enqueue_create_buffer[DType.float32](1 * 480 * T_TRUNK)
    var b2_out7 = ctx.enqueue_create_buffer[DType.float32](1 * 512 * T_TRUNK)
    var b2_out8 = ctx.enqueue_create_buffer[DType.float32](1 * 544 * T_TRUNK)
    var b2_out9 = ctx.enqueue_create_buffer[DType.float32](1 * 576 * T_TRUNK)
    var b2_out10 = ctx.enqueue_create_buffer[DType.float32](1 * 608 * T_TRUNK)
    var b2_out11 = ctx.enqueue_create_buffer[DType.float32](1 * 640 * T_TRUNK)
    var b2_out12 = ctx.enqueue_create_buffer[DType.float32](1 * 672 * T_TRUNK)
    var b2_out13 = ctx.enqueue_create_buffer[DType.float32](1 * 704 * T_TRUNK)
    var b2_out14 = ctx.enqueue_create_buffer[DType.float32](1 * 736 * T_TRUNK)
    var b2_out15 = ctx.enqueue_create_buffer[DType.float32](1 * 768 * T_TRUNK)
    var b2_out16 = ctx.enqueue_create_buffer[DType.float32](1 * 800 * T_TRUNK)
    var b2_out17 = ctx.enqueue_create_buffer[DType.float32](1 * 832 * T_TRUNK)
    var b2_out18 = ctx.enqueue_create_buffer[DType.float32](1 * 864 * T_TRUNK)
    var b2_out19 = ctx.enqueue_create_buffer[DType.float32](1 * 896 * T_TRUNK)
    var b2_out20 = ctx.enqueue_create_buffer[DType.float32](1 * 928 * T_TRUNK)
    var b2_out21 = ctx.enqueue_create_buffer[DType.float32](1 * 960 * T_TRUNK)
    var b2_out22 = ctx.enqueue_create_buffer[DType.float32](1 * 992 * T_TRUNK)
    var b2_out23 = ctx.enqueue_create_buffer[DType.float32](1 * 1024 * T_TRUNK)
    cam_dense_tdnn_layer[1, 256, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, t1_out, b2_out0, b2_w[0], b2_dummy[0])
    cam_dense_tdnn_layer[1, 288, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b2_out0, b2_out1, b2_w[1], b2_dummy[1])
    cam_dense_tdnn_layer[1, 320, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b2_out1, b2_out2, b2_w[2], b2_dummy[2])
    cam_dense_tdnn_layer[1, 352, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b2_out2, b2_out3, b2_w[3], b2_dummy[3])
    cam_dense_tdnn_layer[1, 384, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b2_out3, b2_out4, b2_w[4], b2_dummy[4])
    cam_dense_tdnn_layer[1, 416, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b2_out4, b2_out5, b2_w[5], b2_dummy[5])
    cam_dense_tdnn_layer[1, 448, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b2_out5, b2_out6, b2_w[6], b2_dummy[6])
    cam_dense_tdnn_layer[1, 480, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b2_out6, b2_out7, b2_w[7], b2_dummy[7])
    cam_dense_tdnn_layer[1, 512, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b2_out7, b2_out8, b2_w[8], b2_dummy[8])
    cam_dense_tdnn_layer[1, 544, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b2_out8, b2_out9, b2_w[9], b2_dummy[9])
    cam_dense_tdnn_layer[1, 576, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b2_out9, b2_out10, b2_w[10], b2_dummy[10])
    cam_dense_tdnn_layer[1, 608, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b2_out10, b2_out11, b2_w[11], b2_dummy[11])
    cam_dense_tdnn_layer[1, 640, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b2_out11, b2_out12, b2_w[12], b2_dummy[12])
    cam_dense_tdnn_layer[1, 672, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b2_out12, b2_out13, b2_w[13], b2_dummy[13])
    cam_dense_tdnn_layer[1, 704, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b2_out13, b2_out14, b2_w[14], b2_dummy[14])
    cam_dense_tdnn_layer[1, 736, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b2_out14, b2_out15, b2_w[15], b2_dummy[15])
    cam_dense_tdnn_layer[1, 768, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b2_out15, b2_out16, b2_w[16], b2_dummy[16])
    cam_dense_tdnn_layer[1, 800, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b2_out16, b2_out17, b2_w[17], b2_dummy[17])
    cam_dense_tdnn_layer[1, 832, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b2_out17, b2_out18, b2_w[18], b2_dummy[18])
    cam_dense_tdnn_layer[1, 864, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b2_out18, b2_out19, b2_w[19], b2_dummy[19])
    cam_dense_tdnn_layer[1, 896, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b2_out19, b2_out20, b2_w[20], b2_dummy[20])
    cam_dense_tdnn_layer[1, 928, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b2_out20, b2_out21, b2_w[21], b2_dummy[21])
    cam_dense_tdnn_layer[1, 960, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b2_out21, b2_out22, b2_w[22], b2_dummy[22])
    cam_dense_tdnn_layer[1, 992, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b2_out22, b2_out23, b2_w[23], b2_dummy[23])

    # transit2.
    var t2_bn_w = _upload_w(ctx, fix, "weights/xvector__transit2__nonlinear__batchnorm__weight.bin")
    var t2_bn_b = _upload_w(ctx, fix, "weights/xvector__transit2__nonlinear__batchnorm__bias.bin")
    var t2_bn_m = _upload_w(ctx, fix, "weights/xvector__transit2__nonlinear__batchnorm__running_mean.bin")
    var t2_bn_v = _upload_w(ctx, fix, "weights/xvector__transit2__nonlinear__batchnorm__running_var.bin")
    var t2_lin_w = _upload_w(ctx, fix, "weights/xvector__transit2__linear__weight.bin")
    var t2_lin_b_dummy = ctx.enqueue_create_buffer[DType.float32](512)
    var t2_dummy = ctx.enqueue_create_buffer[DType.float32](512)
    var t2_out = ctx.enqueue_create_buffer[DType.float32](1 * 512 * T_TRUNK)
    transit_layer[1, 1024, 512, T_TRUNK, False](
        ctx, b2_out23, t2_out,
        t2_bn_w, t2_bn_b, t2_bn_m, t2_bn_v,
        t2_lin_w, t2_lin_b_dummy, t2_dummy)

    # block3: 16 dense layers.
    var b3_w = List[DenseTdnnWeights]()
    for i in range(16):
        b3_w.append(_load_dense(ctx, fix, 3, i))
    var b3_dummy = List[DeviceBuffer[DType.float32]]()
    for i in range(16):
        b3_dummy.append(ctx.enqueue_create_buffer[DType.float32](BN_C))
    var b3_out0 = ctx.enqueue_create_buffer[DType.float32](1 * 544 * T_TRUNK)
    var b3_out1 = ctx.enqueue_create_buffer[DType.float32](1 * 576 * T_TRUNK)
    var b3_out2 = ctx.enqueue_create_buffer[DType.float32](1 * 608 * T_TRUNK)
    var b3_out3 = ctx.enqueue_create_buffer[DType.float32](1 * 640 * T_TRUNK)
    var b3_out4 = ctx.enqueue_create_buffer[DType.float32](1 * 672 * T_TRUNK)
    var b3_out5 = ctx.enqueue_create_buffer[DType.float32](1 * 704 * T_TRUNK)
    var b3_out6 = ctx.enqueue_create_buffer[DType.float32](1 * 736 * T_TRUNK)
    var b3_out7 = ctx.enqueue_create_buffer[DType.float32](1 * 768 * T_TRUNK)
    var b3_out8 = ctx.enqueue_create_buffer[DType.float32](1 * 800 * T_TRUNK)
    var b3_out9 = ctx.enqueue_create_buffer[DType.float32](1 * 832 * T_TRUNK)
    var b3_out10 = ctx.enqueue_create_buffer[DType.float32](1 * 864 * T_TRUNK)
    var b3_out11 = ctx.enqueue_create_buffer[DType.float32](1 * 896 * T_TRUNK)
    var b3_out12 = ctx.enqueue_create_buffer[DType.float32](1 * 928 * T_TRUNK)
    var b3_out13 = ctx.enqueue_create_buffer[DType.float32](1 * 960 * T_TRUNK)
    var b3_out14 = ctx.enqueue_create_buffer[DType.float32](1 * 992 * T_TRUNK)
    var b3_out15 = ctx.enqueue_create_buffer[DType.float32](1 * 1024 * T_TRUNK)
    cam_dense_tdnn_layer[1, 512, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, t2_out, b3_out0, b3_w[0], b3_dummy[0])
    cam_dense_tdnn_layer[1, 544, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b3_out0, b3_out1, b3_w[1], b3_dummy[1])
    cam_dense_tdnn_layer[1, 576, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b3_out1, b3_out2, b3_w[2], b3_dummy[2])
    cam_dense_tdnn_layer[1, 608, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b3_out2, b3_out3, b3_w[3], b3_dummy[3])
    cam_dense_tdnn_layer[1, 640, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b3_out3, b3_out4, b3_w[4], b3_dummy[4])
    cam_dense_tdnn_layer[1, 672, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b3_out4, b3_out5, b3_w[5], b3_dummy[5])
    cam_dense_tdnn_layer[1, 704, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b3_out5, b3_out6, b3_w[6], b3_dummy[6])
    cam_dense_tdnn_layer[1, 736, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b3_out6, b3_out7, b3_w[7], b3_dummy[7])
    cam_dense_tdnn_layer[1, 768, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b3_out7, b3_out8, b3_w[8], b3_dummy[8])
    cam_dense_tdnn_layer[1, 800, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b3_out8, b3_out9, b3_w[9], b3_dummy[9])
    cam_dense_tdnn_layer[1, 832, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b3_out9, b3_out10, b3_w[10], b3_dummy[10])
    cam_dense_tdnn_layer[1, 864, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b3_out10, b3_out11, b3_w[11], b3_dummy[11])
    cam_dense_tdnn_layer[1, 896, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b3_out11, b3_out12, b3_w[12], b3_dummy[12])
    cam_dense_tdnn_layer[1, 928, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b3_out12, b3_out13, b3_w[13], b3_dummy[13])
    cam_dense_tdnn_layer[1, 960, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b3_out13, b3_out14, b3_w[14], b3_dummy[14])
    cam_dense_tdnn_layer[1, 992, GROWTH, BN_C, HALF_BN, T_TRUNK, KERNEL, 2, SEG_LEN](ctx, b3_out14, b3_out15, b3_w[15], b3_dummy[15])

    # transit3.
    var t3_bn_w = _upload_w(ctx, fix, "weights/xvector__transit3__nonlinear__batchnorm__weight.bin")
    var t3_bn_b = _upload_w(ctx, fix, "weights/xvector__transit3__nonlinear__batchnorm__bias.bin")
    var t3_bn_m = _upload_w(ctx, fix, "weights/xvector__transit3__nonlinear__batchnorm__running_mean.bin")
    var t3_bn_v = _upload_w(ctx, fix, "weights/xvector__transit3__nonlinear__batchnorm__running_var.bin")
    var t3_lin_w = _upload_w(ctx, fix, "weights/xvector__transit3__linear__weight.bin")
    var t3_lin_b_dummy = ctx.enqueue_create_buffer[DType.float32](512)
    var t3_dummy = ctx.enqueue_create_buffer[DType.float32](512)
    var t3_out = ctx.enqueue_create_buffer[DType.float32](1 * 512 * T_TRUNK)
    transit_layer[1, 1024, 512, T_TRUNK, False](
        ctx, b3_out15, t3_out,
        t3_bn_w, t3_bn_b, t3_bn_m, t3_bn_v,
        t3_lin_w, t3_lin_b_dummy, t3_dummy)

    # out_nonlinear: BN1d(512) + ReLU.
    var on_bn_w = _upload_w(ctx, fix, "weights/xvector__out_nonlinear__batchnorm__weight.bin")
    var on_bn_b = _upload_w(ctx, fix, "weights/xvector__out_nonlinear__batchnorm__bias.bin")
    var on_bn_m = _upload_w(ctx, fix, "weights/xvector__out_nonlinear__batchnorm__running_mean.bin")
    var on_bn_v = _upload_w(ctx, fix, "weights/xvector__out_nonlinear__batchnorm__running_var.bin")
    var on_bn_buf = ctx.enqueue_create_buffer[DType.float32](1 * 512 * T_TRUNK)
    var on_out_buf = ctx.enqueue_create_buffer[DType.float32](1 * 512 * T_TRUNK)
    comptime t3_layout = row_major[1, 512, T_TRUNK]()
    comptime t3_p_layout = row_major[512]()
    comptime t3_flat = row_major[1 * 512 * T_TRUNK]()
    var t3_out_t = TileTensor(t3_out, t3_layout)
    var on_bn_w_t = TileTensor(on_bn_w, t3_p_layout)
    var on_bn_b_t = TileTensor(on_bn_b, t3_p_layout)
    var on_bn_m_t = TileTensor(on_bn_m, t3_p_layout)
    var on_bn_v_t = TileTensor(on_bn_v, t3_p_layout)
    var on_bn_t = TileTensor(on_bn_buf, t3_layout)
    var on_bn_flat = TileTensor(on_bn_buf, t3_flat)
    var on_out_flat = TileTensor(on_out_buf, t3_flat)

    comptime on_bn_k = batchnorm1d_kernel[
        DType.float32, type_of(t3_layout), type_of(t3_p_layout),
        type_of(t3_layout), BLOCK,
    ]
    ctx.enqueue_function[on_bn_k, on_bn_k](
        on_bn_t, t3_out_t, on_bn_w_t, on_bn_b_t, on_bn_m_t, on_bn_v_t,
        1, 512, T_TRUNK, EPS,
        grid_dim=1 * 512, block_dim=BLOCK,
    )
    comptime on_relu_k = relu_kernel[
        DType.float32, type_of(t3_flat), type_of(t3_flat), BLOCK,
    ]
    ctx.enqueue_function[on_relu_k, on_relu_k](
        on_out_flat, on_bn_flat, 1 * 512 * T_TRUNK,
        grid_dim=ceildiv(1 * 512 * T_TRUNK, BLOCK), block_dim=BLOCK,
    )

    # stats: mean+std along T -> (1, 1024).
    var stats_buf = ctx.enqueue_create_buffer[DType.float32](1 * 1024)
    var on_out_t = TileTensor(on_out_buf, t3_layout)
    comptime stats_layout = row_major[1, 1024]()
    var stats_t = TileTensor(stats_buf, stats_layout)
    comptime stats_k = stats_pool_kernel[
        DType.float32, type_of(t3_layout), type_of(stats_layout), BLOCK,
    ]
    ctx.enqueue_function[stats_k, stats_k](
        stats_t, on_out_t, 1, 512, T_TRUNK,
        grid_dim=1 * 512, block_dim=BLOCK,
    )

    # dense: Conv1d 1x1 (1024 → 192) + BN no-affine.
    var d_lin_w = _upload_w(ctx, fix, "weights/xvector__dense__linear__weight.bin")
    var d_bn_m = _upload_w(ctx, fix, "weights/xvector__dense__nonlinear__batchnorm__running_mean.bin")
    var d_bn_v = _upload_w(ctx, fix, "weights/xvector__dense__nonlinear__batchnorm__running_var.bin")
    var d_dummy = ctx.enqueue_create_buffer[DType.float32](192)
    var d_lin_out = ctx.enqueue_create_buffer[DType.float32](1 * 192 * 1)
    var d_out = ctx.enqueue_create_buffer[DType.float32](1 * 192)

    comptime stats_3d_layout = row_major[1, 1024, 1]()
    comptime d_w_layout = row_major[192, 1024, 1]()
    comptime d_out_3d_layout = row_major[1, 192, 1]()
    comptime d_2d_layout = row_major[1, 192]()
    comptime d_p_layout = row_major[192]()

    var stats_3d_t = TileTensor(stats_buf, stats_3d_layout)
    var d_lin_w_t = TileTensor(d_lin_w, d_w_layout)
    var d_dummy_t = TileTensor(d_dummy, d_p_layout)
    var d_lin_out_t = TileTensor(d_lin_out, d_out_3d_layout)

    comptime d_conv_k = conv1d_kernel_fast[
        DType.float32, type_of(stats_3d_layout), type_of(d_w_layout),
        type_of(d_p_layout), type_of(d_out_3d_layout),
        1, False, BLOCK,
    ]
    ctx.enqueue_function[d_conv_k, d_conv_k](
        d_lin_out_t, stats_3d_t, d_lin_w_t, d_dummy_t,
        1, 1024, 192, 1, 1, 1, 0, 1,
        grid_dim=1 * 192, block_dim=BLOCK,
    )

    var d_lin_2d_t = TileTensor(d_lin_out, d_2d_layout)
    var d_bn_m_t = TileTensor(d_bn_m, d_p_layout)
    var d_bn_v_t = TileTensor(d_bn_v, d_p_layout)
    var d_out_t = TileTensor(d_out, d_2d_layout)
    comptime bn_na_k = bn_no_affine_2d_kernel[
        DType.float32, type_of(d_2d_layout), type_of(d_p_layout),
        type_of(d_2d_layout), BLOCK,
    ]
    ctx.enqueue_function[bn_na_k, bn_na_k](
        d_out_t, d_lin_2d_t, d_bn_m_t, d_bn_v_t,
        1, 192, EPS,
        grid_dim=ceildiv(1 * 192, BLOCK), block_dim=BLOCK,
    )

    return d_out^
