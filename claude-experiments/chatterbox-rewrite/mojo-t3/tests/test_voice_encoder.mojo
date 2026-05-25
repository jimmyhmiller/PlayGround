"""Parity test for full VoiceEncoder forward (B=4, T=160, M=40, embed=256)."""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import load_fp32
from lstm import lstm_layer_first_kernel
from layernorm import linear_kernel
from voice_encoder import extract_last_step_kernel, relu_l2_norm_row_kernel


comptime B = 4
comptime T = 160
comptime M = 40
comptime H = 256
comptime EMBED = 256
comptime BLOCK = 128


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


def test_voice_encoder() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/ve_forward/"
    var ctx = DeviceContext()

    var mels = load_fp32(fix + "mels.bin")
    var exp = load_fp32(fix + "embed.bin")

    var mels_buf = ctx.enqueue_create_buffer[DType.float32](B * T * M)
    upload(mels_buf, mels.data, B * T * M)

    var l0_buf = ctx.enqueue_create_buffer[DType.float32](B * T * H)
    var l1_buf = ctx.enqueue_create_buffer[DType.float32](B * T * H)
    var l2_buf = ctx.enqueue_create_buffer[DType.float32](B * T * H)
    var last_buf = ctx.enqueue_create_buffer[DType.float32](B * 1 * H)
    var lin_buf = ctx.enqueue_create_buffer[DType.float32](B * 1 * EMBED)
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

    comptime mels_layout = row_major[B, T, M]()
    comptime hseq_layout = row_major[B, T, H]()
    comptime last_layout = row_major[B, 1, H]()
    comptime lin_layout  = row_major[B, 1, EMBED]()
    comptime embed_layout = row_major[B, EMBED]()
    comptime w_ih_0_layout = row_major[4 * H, M]()
    comptime w_hh_layout = row_major[4 * H, H]()
    comptime b_layout = row_major[4 * H]()
    comptime proj_w_layout = row_major[EMBED, H]()
    comptime proj_b_layout = row_major[EMBED]()

    var mels_t = TileTensor(mels_buf, mels_layout)
    var l0_t = TileTensor(l0_buf, hseq_layout)
    var l1_t = TileTensor(l1_buf, hseq_layout)
    var l2_t = TileTensor(l2_buf, hseq_layout)
    var last_t = TileTensor(last_buf, last_layout)
    var lin_t = TileTensor(lin_buf, lin_layout)
    var embed_t = TileTensor(embed_buf, embed_layout)
    var w_ih_0_t = TileTensor(w_ih_0, w_ih_0_layout)
    var w_hh_0_t = TileTensor(w_hh_0, w_hh_layout)
    var b_ih_0_t = TileTensor(b_ih_0, b_layout)
    var b_hh_0_t = TileTensor(b_hh_0, b_layout)
    var w_ih_1_t = TileTensor(w_ih_1, w_hh_layout)
    var w_hh_1_t = TileTensor(w_hh_1, w_hh_layout)
    var b_ih_1_t = TileTensor(b_ih_1, b_layout)
    var b_hh_1_t = TileTensor(b_hh_1, b_layout)
    var w_ih_2_t = TileTensor(w_ih_2, w_hh_layout)
    var w_hh_2_t = TileTensor(w_hh_2, w_hh_layout)
    var b_ih_2_t = TileTensor(b_ih_2, b_layout)
    var b_hh_2_t = TileTensor(b_hh_2, b_layout)
    var proj_w_t = TileTensor(proj_w, proj_w_layout)
    var proj_b_t = TileTensor(proj_b, proj_b_layout)

    # LSTM Layer 0 (M=40 → H=256).
    comptime k0 = lstm_layer_first_kernel[
        DType.float32, type_of(mels_layout), type_of(hseq_layout),
        type_of(w_ih_0_layout), type_of(w_hh_layout), type_of(b_layout),
        H, M, BLOCK,
    ]
    ctx.enqueue_function[k0, k0](
        l0_t, mels_t, w_ih_0_t, w_hh_0_t, b_ih_0_t, b_hh_0_t,
        B, T, grid_dim=B, block_dim=BLOCK,
    )
    # Layers 1 & 2 (H=256 → H=256).
    comptime k1 = lstm_layer_first_kernel[
        DType.float32, type_of(hseq_layout), type_of(hseq_layout),
        type_of(w_hh_layout), type_of(w_hh_layout), type_of(b_layout),
        H, H, BLOCK,
    ]
    ctx.enqueue_function[k1, k1](
        l1_t, l0_t, w_ih_1_t, w_hh_1_t, b_ih_1_t, b_hh_1_t,
        B, T, grid_dim=B, block_dim=BLOCK,
    )
    ctx.enqueue_function[k1, k1](
        l2_t, l1_t, w_ih_2_t, w_hh_2_t, b_ih_2_t, b_hh_2_t,
        B, T, grid_dim=B, block_dim=BLOCK,
    )

    # Extract last hidden state (B, T, H) → (B, 1, H).
    comptime ke = extract_last_step_kernel[
        DType.float32, type_of(hseq_layout), type_of(last_layout),
        H, BLOCK,
    ]
    ctx.enqueue_function[ke, ke](
        last_t, l2_t, B, T, grid_dim=B, block_dim=BLOCK,
    )

    # proj Linear: (B, 1, H) → (B, 1, EMBED).
    comptime kl = linear_kernel[
        DType.float32, type_of(last_layout), type_of(proj_w_layout),
        type_of(proj_b_layout), type_of(lin_layout),
        True, BLOCK,
    ]
    ctx.enqueue_function[kl, kl](
        lin_t, last_t, proj_w_t, proj_b_t,
        B, 1, H, EMBED,
        grid_dim=B * 1, block_dim=BLOCK,
    )

    # ReLU + L2 normalize.
    comptime kn = relu_l2_norm_row_kernel[
        DType.float32, type_of(lin_layout), type_of(embed_layout),
        EMBED, BLOCK,
    ]
    ctx.enqueue_function[kn, kn](
        embed_t, lin_t, B, grid_dim=B, block_dim=BLOCK,
    )
    ctx.synchronize()

    var n_out = B * EMBED
    var max_abs: Float32 = 0.0
    with embed_buf.map_to_host() as host:
        for i in range(n_out):
            var d = host[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            if i < 8:
                print("ve[", i, "]: mojo=", host[i], "  torch=", exp.data[i], "  diff=", d)
            assert_almost_equal(host[i], exp.data[i], atol=1.0e-5)
    print("VoiceEncoder (B=4, T=160) — max abs:", max_abs)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
