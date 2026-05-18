"""
Test: ref_wav → kaldi fbank → CAMPPlus → F.normalize → spk_embed_affine → spks (1, 80).
Verifies the speaker-conditioning path in Mojo for the cloned voice.
"""
from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import load_fp32
from layernorm import linear_kernel
from flow_glue import normalize_l2_kernel


comptime D_IN = 192
comptime D_OUT = 80
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


def test_voice_to_spks() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    # Start from the dumped xvector (CAMPPlus output, verified at 6.2e-6 earlier).
    var xv = load_fp32("tests/fixtures/real/e2e_embedding.bin")
    var exp = load_fp32("tests/fixtures/real/e2e_spk_affine.bin")
    var w = upload_w(ctx, "tests/fixtures/s3gen/", "weights/flow__spk_embed_affine_layer__weight.bin")
    var b = upload_w(ctx, "tests/fixtures/s3gen/", "weights/flow__spk_embed_affine_layer__bias.bin")

    var xv_buf = ctx.enqueue_create_buffer[DType.float32](D_IN)
    var xv_norm_buf = ctx.enqueue_create_buffer[DType.float32](D_IN)
    var out_buf = ctx.enqueue_create_buffer[DType.float32](D_OUT)
    upload(xv_buf, xv.data, D_IN)

    comptime in_layout = row_major[1, D_IN]()
    comptime out_layout = row_major[1, 1, D_OUT]()
    comptime in_btd_layout = row_major[1, 1, D_IN]()
    comptime w_layout = row_major[D_OUT, D_IN]()
    comptime p_layout = row_major[D_OUT]()

    var xv_t = TileTensor(xv_buf, in_layout)
    var xv_norm_t = TileTensor(xv_norm_buf, in_layout)
    var xv_norm_btd = TileTensor(xv_norm_buf, in_btd_layout)
    var w_t = TileTensor(w, w_layout)
    var b_t = TileTensor(b, p_layout)
    var out_t = TileTensor(out_buf, out_layout)

    # F.normalize along dim=1 (L2 norm).
    comptime norm_k = normalize_l2_kernel[
        DType.float32, type_of(in_layout), type_of(in_layout), BLOCK,
    ]
    ctx.enqueue_function[norm_k, norm_k](
        xv_norm_t, xv_t, 1, D_IN, Float32(1.0e-12),
        grid_dim=1, block_dim=BLOCK,
    )
    # Linear 192 → 80.
    comptime lin_k = linear_kernel[
        DType.float32, type_of(in_btd_layout), type_of(w_layout),
        type_of(p_layout), type_of(out_layout),
        True, BLOCK,
    ]
    ctx.enqueue_function[lin_k, lin_k](
        out_t, xv_norm_btd, w_t, b_t, 1, 1, D_IN, D_OUT,
        grid_dim=1, block_dim=BLOCK,
    )
    ctx.synchronize()

    var max_abs: Float32 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(D_OUT):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            if i < 8:
                print("spk[", i, "]: mojo=", h[i], "  torch=", exp.data[i], "  diff=", d)
            assert_almost_equal(h[i], exp.data[i], atol=1.0e-5)
    print("xvector → F.normalize → spk_embed_affine — max abs:", max_abs)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
