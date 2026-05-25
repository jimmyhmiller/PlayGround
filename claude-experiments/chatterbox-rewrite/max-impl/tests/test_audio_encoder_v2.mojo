"""AudioEncoderV2 parity test reusing existing mojo-t3 fixture.

Config: B=1, T_in=24, n_mels=80, n_state=128, n_head=4, n_layer=2, stride=2.

This validates the S3Tokenizer's encoder path (sans final FSQ codebook).
All ops route through MAX abstractions via our wrapper modules.
"""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from fixture import load_fp32
from modules import Linear, LayerNorm, gelu
from conv1d import Conv1d, conv1d_forward
from s3tokenizer_block import FSMNAttention, S3TokenizerBlock, s3tokenizer_block_forward
from transformer_blocks import MLP


comptime B = 1
comptime T_IN = 24
comptime T_OUT = 6   # T_in / (stride * 2) = 6 after two stride-2 convs
comptime N_MELS = 80
comptime D = 128
comptime H = 4
comptime DH = D // H
comptime HALF = DH // 2
comptime KSIZE = 31
comptime LEFT_PAD = 15
comptime RIGHT_PAD = 15
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


def test_audio_encoder_v2() raises:
    """Run AudioEncoderV2 (conv1+gelu+conv2+gelu+transpose+2*FSMNblock).

    Compares against torch output exported as `out.bin`.
    """
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "../mojo-t3/tests/fixtures/audio_encoder_v2/"
    var ctx = DeviceContext()

    var x_t = load_fp32(fix + "x.bin")
    var exp_t = load_fp32(fix + "out.bin")
    var mp_t = load_fp32(fix + "mask_pad.bin")
    var cos_t = load_fp32(fix + "cos.bin")
    var sin_t = load_fp32(fix + "sin.bin")

    var conv1_w = upload_w(ctx, fix, "conv1_w.bin")
    var conv1_b = upload_w(ctx, fix, "conv1_b.bin")
    var conv2_w = upload_w(ctx, fix, "conv2_w.bin")
    var conv2_b = upload_w(ctx, fix, "conv2_b.bin")

    var x_buf = ctx.enqueue_create_buffer[DType.float32](B * N_MELS * T_IN)
    upload(x_buf, x_t.data, B * N_MELS * T_IN)
    var mp_buf = ctx.enqueue_create_buffer[DType.float32](B * T_OUT * 1)
    upload(mp_buf, mp_t.data, B * T_OUT * 1)
    var cos_buf = ctx.enqueue_create_buffer[DType.float32](T_OUT * HALF)
    upload(cos_buf, cos_t.data, T_OUT * HALF)
    var sin_buf = ctx.enqueue_create_buffer[DType.float32](T_OUT * HALF)
    upload(sin_buf, sin_t.data, T_OUT * HALF)
    var attn_mask_buf = ctx.enqueue_create_buffer[DType.float32](T_OUT * T_OUT)
    attn_mask_buf.enqueue_fill(0.0)

    # --- Conv1 + GELU ---
    var T_HALF = 12   # after stride-2 conv1
    var c1_out = ctx.enqueue_create_buffer[DType.float32](B * D * T_HALF)
    var conv1 = Conv1d(conv1_w^, conv1_b^, N_MELS, D, 3, 2, 1, 1, 1, True)
    conv1d_forward(ctx, conv1, x_buf, c1_out, B, T_IN, T_HALF)
    var c1_act = ctx.enqueue_create_buffer[DType.float32](B * D * T_HALF)
    gelu(ctx, c1_out, c1_act, B * D * T_HALF)

    # --- Conv2 + GELU ---
    var c2_out = ctx.enqueue_create_buffer[DType.float32](B * D * T_OUT)
    var conv2 = Conv1d(conv2_w^, conv2_b^, D, D, 3, 2, 1, 1, 1, True)
    conv1d_forward(ctx, conv2, c1_act, c2_out, B, T_HALF, T_OUT)
    var c2_act = ctx.enqueue_create_buffer[DType.float32](B * D * T_OUT)
    gelu(ctx, c2_out, c2_act, B * D * T_OUT)

    # --- Transpose (B, D, T) → (B, T, D) ---
    var x_seq = ctx.enqueue_create_buffer[DType.float32](B * T_OUT * D)
    var cp = c2_act.unsafe_ptr()
    var sp = x_seq.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(cp, sp)
    def trans_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (T_OUT * D)
        var rem = i - bi * T_OUT * D
        var ti = rem // D
        var di = rem - ti * D
        sp[i] = cp[bi * D * T_OUT + di * T_OUT + ti]
    elementwise[trans_func, simd_width=1, target="gpu"](
        IndexList[1](B * T_OUT * D), DeviceContextPtr(ctx),
    )

    # --- Build the 2 FSMN blocks ---
    var blocks = List[S3TokenizerBlock]()
    var i_layer = 0
    while i_layer < N_LAYER:
        var pre = "L" + String(i_layer) + "_"
        var attn_ln_w = upload_w(ctx, fix, pre + "attn_ln_w.bin")
        var attn_ln_b = upload_w(ctx, fix, pre + "attn_ln_b.bin")
        var mlp_ln_w = upload_w(ctx, fix, pre + "mlp_ln_w.bin")
        var mlp_ln_b = upload_w(ctx, fix, pre + "mlp_ln_b.bin")
        var q_w = upload_w(ctx, fix, pre + "q_w.bin")
        var q_b = upload_w(ctx, fix, pre + "q_b.bin")
        var k_w = upload_w(ctx, fix, pre + "k_w.bin")
        var k_b = ctx.enqueue_create_buffer[DType.float32](D)
        k_b.enqueue_fill(0.0)
        var v_w = upload_w(ctx, fix, pre + "v_w.bin")
        var v_b = upload_w(ctx, fix, pre + "v_b.bin")
        var out_w = upload_w(ctx, fix, pre + "out_w.bin")
        var out_b = upload_w(ctx, fix, pre + "out_b.bin")
        var fsmn_w = upload_w(ctx, fix, pre + "fsmn_w.bin")
        var mlp_fc1_w = upload_w(ctx, fix, pre + "mlp_fc1_w.bin")
        var mlp_fc1_b = upload_w(ctx, fix, pre + "mlp_fc1_b.bin")
        var mlp_fc2_w = upload_w(ctx, fix, pre + "mlp_fc2_w.bin")
        var mlp_fc2_b = upload_w(ctx, fix, pre + "mlp_fc2_b.bin")

        var to_q = Linear(q_w^, q_b^, D, D, True)
        var to_k = Linear(k_w^, k_b^, D, D, False)
        var to_v = Linear(v_w^, v_b^, D, D, True)
        var to_out = Linear(out_w^, out_b^, D, D, True)
        var fsmn_conv = Conv1d(fsmn_w^, ctx.enqueue_create_buffer[DType.float32](0),
                                D, D, KSIZE, 1, LEFT_PAD, 1, D, False)
        var fsmn_attn = FSMNAttention(to_q^, to_k^, to_v^, to_out^, fsmn_conv^, H, DH)
        var attn_ln = LayerNorm(attn_ln_w^, attn_ln_b^, D, Float32(1.0e-5))
        var mlp_ln = LayerNorm(mlp_ln_w^, mlp_ln_b^, D, Float32(1.0e-5))
        var mlp_fc1 = Linear(mlp_fc1_w^, mlp_fc1_b^, D, D * 4, True)
        var mlp_fc2 = Linear(mlp_fc2_w^, mlp_fc2_b^, D * 4, D, True)
        var mlp = MLP(mlp_fc1^, mlp_fc2^, D, D * 4)
        var block = S3TokenizerBlock(attn_ln^, mlp_ln^, fsmn_attn^, mlp^)
        blocks.append(block^)
        i_layer += 1

    # --- Run blocks ---
    for i in range(len(blocks)):
        s3tokenizer_block_forward(
            ctx, blocks[i], x_seq, cos_buf, sin_buf,
            mp_buf, attn_mask_buf, B, T_OUT, False,
        )
    ctx.synchronize()

    var n_out = B * T_OUT * D
    var max_abs: Float32 = 0.0
    with x_seq.map_to_host() as h:
        for i in range(n_out):
            var d = h[i] - exp_t.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            if i < 8:
                print("ae2[", i, "]: mojo=", h[i], "  torch=", exp_t.data[i], "  diff=", d)
            assert_almost_equal(h[i], exp_t.data[i], atol=2.0e-4)
    print("AudioEncoderV2 (B=1, T=24→6, n_state=128, 2 FSMN layers) — max abs:", max_abs)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
