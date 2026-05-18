"""VoiceEncoder parity test. Reuses existing fixture from mojo-t3."""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer

from fixture import load_fp32
from modules import Linear
from lstm import LSTMLayer
from voice_encoder import VoiceEncoder, voice_encoder_forward


comptime B = 4
comptime T = 160
comptime M = 40
comptime H = 256
comptime EMBED = 256


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


def test_voice_encoder_pure_max() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "../mojo-t3/tests/fixtures/ve_forward/"
    var ctx = DeviceContext()

    var mels = load_fp32(fix + "mels.bin")
    var exp = load_fp32(fix + "embed.bin")

    var mels_buf = ctx.enqueue_create_buffer[DType.float32](B * T * M)
    upload(mels_buf, mels.data, B * T * M)
    var embed_buf = ctx.enqueue_create_buffer[DType.float32](B * EMBED)

    var w_ih_0 = upload_w(ctx, fix, "weight_ih_l0.bin")
    var w_hh_0 = upload_w(ctx, fix, "weight_hh_l0.bin")
    var b_ih_0 = upload_w(ctx, fix, "bias_ih_l0.bin")
    var b_hh_0 = upload_w(ctx, fix, "bias_hh_l0.bin")
    var w_ih_1 = upload_w(ctx, fix, "weight_ih_l1.bin")
    var w_hh_1 = upload_w(ctx, fix, "weight_hh_l1.bin")
    var b_ih_1 = upload_w(ctx, fix, "bias_ih_l1.bin")
    var b_hh_1 = upload_w(ctx, fix, "bias_hh_l1.bin")
    var w_ih_2 = upload_w(ctx, fix, "weight_ih_l2.bin")
    var w_hh_2 = upload_w(ctx, fix, "weight_hh_l2.bin")
    var b_ih_2 = upload_w(ctx, fix, "bias_ih_l2.bin")
    var b_hh_2 = upload_w(ctx, fix, "bias_hh_l2.bin")
    var proj_w = upload_w(ctx, fix, "proj_weight.bin")
    var proj_b = upload_w(ctx, fix, "proj_bias.bin")

    var lstm_l0 = LSTMLayer(w_ih_0, w_hh_0, b_ih_0, b_hh_0, M, H)
    var lstm_l1 = LSTMLayer(w_ih_1, w_hh_1, b_ih_1, b_hh_1, H, H)
    var lstm_l2 = LSTMLayer(w_ih_2, w_hh_2, b_ih_2, b_hh_2, H, H)
    var proj   = Linear(proj_w, proj_b, H, EMBED, True)
    var ve     = VoiceEncoder(lstm_l0^, lstm_l1^, lstm_l2^, proj^, H)

    # Our LSTM helper supports B=1 only currently. Run B=4 by iterating.
    # Each row's embedding is computed by isolating its mels slice.
    var single_mels_buf = ctx.enqueue_create_buffer[DType.float32](1 * T * M)
    var single_embed_buf = ctx.enqueue_create_buffer[DType.float32](1 * EMBED)
    var collected = List[Float32]()
    for _ in range(B * EMBED): collected.append(Float32(0.0))

    for bi in range(B):
        # Copy mels[bi, :, :] into single_mels_buf.
        with mels_buf.map_to_host() as src:
            with single_mels_buf.map_to_host() as dst:
                var off = bi * T * M
                for i in range(T * M):
                    dst[i] = src[off + i]
        voice_encoder_forward(ctx, ve, single_mels_buf, single_embed_buf, 1, T)
        ctx.synchronize()
        with single_embed_buf.map_to_host() as eh:
            for i in range(EMBED):
                collected[bi * EMBED + i] = eh[i]

    var n_out = B * EMBED
    var max_abs: Float32 = 0.0
    for i in range(n_out):
        var d = collected[i] - exp.data[i]
        if d < 0.0: d = -d
        if d > max_abs: max_abs = d
        if i < 8:
            print("ve[", i, "]: mojo=", collected[i], "  torch=", exp.data[i], "  diff=", d)
        assert_almost_equal(collected[i], exp.data[i], atol=2.0e-5)
    print("VoiceEncoder (B=4, T=160) — max abs:", max_abs)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
