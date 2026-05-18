"""
Parity test for the full CAMPPlus FCM head:
  fcm_input_4d (1, 1, 80, 998)
    -> conv1 (3x3 pad=1) -> bn1 -> relu1                   (1,32,80,998)
    -> layer1[0] BasicResBlock(stride=2,shortcut)          (1,32,40,998)
    -> layer1[1] BasicResBlock(stride=1,no-shortcut)       (1,32,40,998)
    -> layer2[0] BasicResBlock(stride=2,shortcut)          (1,32,20,998)
    -> layer2[1] BasicResBlock(stride=1,no-shortcut)       (1,32,20,998)
    -> conv2 (3x3 stride=(2,1) pad=1) -> bn2 -> relu       (1,32,10,998)
    -> reshape to (1, 320, 998)

Compared against fcm_out.bin.
"""
from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import load_fp32
from conv import conv2d_kernel, batchnorm2d_kernel, relu_kernel
from campplus import basic_resblock


comptime B = 1
comptime H_INPUT = 80
comptime W = 998
comptime EPS_BN: Float32 = 1.0e-5
comptime BLOCK_PW: Int = 256


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


def test_fcm_full() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/campplus/"
    var ctx = DeviceContext()

    # ---- Load all weights up front.
    var w_conv1 = upload_w(ctx, fix, "weights/head__conv1__weight.bin")            # (32,1,3,3)
    var bn1_w = upload_w(ctx, fix, "weights/head__bn1__weight.bin")
    var bn1_b = upload_w(ctx, fix, "weights/head__bn1__bias.bin")
    var bn1_m = upload_w(ctx, fix, "weights/head__bn1__running_mean.bin")
    var bn1_v = upload_w(ctx, fix, "weights/head__bn1__running_var.bin")

    # layer1[0], [1]
    var L10_c1 = upload_w(ctx, fix, "weights/head__layer1__0__conv1__weight.bin")
    var L10_c2 = upload_w(ctx, fix, "weights/head__layer1__0__conv2__weight.bin")
    var L10_bn1w = upload_w(ctx, fix, "weights/head__layer1__0__bn1__weight.bin")
    var L10_bn1b = upload_w(ctx, fix, "weights/head__layer1__0__bn1__bias.bin")
    var L10_bn1m = upload_w(ctx, fix, "weights/head__layer1__0__bn1__running_mean.bin")
    var L10_bn1v = upload_w(ctx, fix, "weights/head__layer1__0__bn1__running_var.bin")
    var L10_bn2w = upload_w(ctx, fix, "weights/head__layer1__0__bn2__weight.bin")
    var L10_bn2b = upload_w(ctx, fix, "weights/head__layer1__0__bn2__bias.bin")
    var L10_bn2m = upload_w(ctx, fix, "weights/head__layer1__0__bn2__running_mean.bin")
    var L10_bn2v = upload_w(ctx, fix, "weights/head__layer1__0__bn2__running_var.bin")
    var L10_scw = upload_w(ctx, fix, "weights/head__layer1__0__shortcut__0__weight.bin")
    var L10_scbnw = upload_w(ctx, fix, "weights/head__layer1__0__shortcut__1__weight.bin")
    var L10_scbnb = upload_w(ctx, fix, "weights/head__layer1__0__shortcut__1__bias.bin")
    var L10_scbnm = upload_w(ctx, fix, "weights/head__layer1__0__shortcut__1__running_mean.bin")
    var L10_scbnv = upload_w(ctx, fix, "weights/head__layer1__0__shortcut__1__running_var.bin")

    var L11_c1 = upload_w(ctx, fix, "weights/head__layer1__1__conv1__weight.bin")
    var L11_c2 = upload_w(ctx, fix, "weights/head__layer1__1__conv2__weight.bin")
    var L11_bn1w = upload_w(ctx, fix, "weights/head__layer1__1__bn1__weight.bin")
    var L11_bn1b = upload_w(ctx, fix, "weights/head__layer1__1__bn1__bias.bin")
    var L11_bn1m = upload_w(ctx, fix, "weights/head__layer1__1__bn1__running_mean.bin")
    var L11_bn1v = upload_w(ctx, fix, "weights/head__layer1__1__bn1__running_var.bin")
    var L11_bn2w = upload_w(ctx, fix, "weights/head__layer1__1__bn2__weight.bin")
    var L11_bn2b = upload_w(ctx, fix, "weights/head__layer1__1__bn2__bias.bin")
    var L11_bn2m = upload_w(ctx, fix, "weights/head__layer1__1__bn2__running_mean.bin")
    var L11_bn2v = upload_w(ctx, fix, "weights/head__layer1__1__bn2__running_var.bin")

    # layer2[0], [1]
    var L20_c1 = upload_w(ctx, fix, "weights/head__layer2__0__conv1__weight.bin")
    var L20_c2 = upload_w(ctx, fix, "weights/head__layer2__0__conv2__weight.bin")
    var L20_bn1w = upload_w(ctx, fix, "weights/head__layer2__0__bn1__weight.bin")
    var L20_bn1b = upload_w(ctx, fix, "weights/head__layer2__0__bn1__bias.bin")
    var L20_bn1m = upload_w(ctx, fix, "weights/head__layer2__0__bn1__running_mean.bin")
    var L20_bn1v = upload_w(ctx, fix, "weights/head__layer2__0__bn1__running_var.bin")
    var L20_bn2w = upload_w(ctx, fix, "weights/head__layer2__0__bn2__weight.bin")
    var L20_bn2b = upload_w(ctx, fix, "weights/head__layer2__0__bn2__bias.bin")
    var L20_bn2m = upload_w(ctx, fix, "weights/head__layer2__0__bn2__running_mean.bin")
    var L20_bn2v = upload_w(ctx, fix, "weights/head__layer2__0__bn2__running_var.bin")
    var L20_scw = upload_w(ctx, fix, "weights/head__layer2__0__shortcut__0__weight.bin")
    var L20_scbnw = upload_w(ctx, fix, "weights/head__layer2__0__shortcut__1__weight.bin")
    var L20_scbnb = upload_w(ctx, fix, "weights/head__layer2__0__shortcut__1__bias.bin")
    var L20_scbnm = upload_w(ctx, fix, "weights/head__layer2__0__shortcut__1__running_mean.bin")
    var L20_scbnv = upload_w(ctx, fix, "weights/head__layer2__0__shortcut__1__running_var.bin")

    var L21_c1 = upload_w(ctx, fix, "weights/head__layer2__1__conv1__weight.bin")
    var L21_c2 = upload_w(ctx, fix, "weights/head__layer2__1__conv2__weight.bin")
    var L21_bn1w = upload_w(ctx, fix, "weights/head__layer2__1__bn1__weight.bin")
    var L21_bn1b = upload_w(ctx, fix, "weights/head__layer2__1__bn1__bias.bin")
    var L21_bn1m = upload_w(ctx, fix, "weights/head__layer2__1__bn1__running_mean.bin")
    var L21_bn1v = upload_w(ctx, fix, "weights/head__layer2__1__bn1__running_var.bin")
    var L21_bn2w = upload_w(ctx, fix, "weights/head__layer2__1__bn2__weight.bin")
    var L21_bn2b = upload_w(ctx, fix, "weights/head__layer2__1__bn2__bias.bin")
    var L21_bn2m = upload_w(ctx, fix, "weights/head__layer2__1__bn2__running_mean.bin")
    var L21_bn2v = upload_w(ctx, fix, "weights/head__layer2__1__bn2__running_var.bin")

    var w_conv2 = upload_w(ctx, fix, "weights/head__conv2__weight.bin")            # (32,32,3,3)
    var bn2_w = upload_w(ctx, fix, "weights/head__bn2__weight.bin")
    var bn2_b = upload_w(ctx, fix, "weights/head__bn2__bias.bin")
    var bn2_m = upload_w(ctx, fix, "weights/head__bn2__running_mean.bin")
    var bn2_v = upload_w(ctx, fix, "weights/head__bn2__running_var.bin")

    # ---- Load input + run conv1/bn1/relu1 inline (single-IN-channel, can't go
    # through basic_resblock).
    var x_in = load_fp32(fix + "fcm_input_4d.bin")        # (1,1,80,998)
    var n_in = B * 1 * H_INPUT * W
    var n_relu1 = B * 32 * H_INPUT * W

    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_in)
    upload(x_buf, x_in.data, n_in)
    var dummy1 = ctx.enqueue_create_buffer[DType.float32](32)
    var dummy2 = ctx.enqueue_create_buffer[DType.float32](1)
    var conv1_out = ctx.enqueue_create_buffer[DType.float32](n_relu1)
    var bn1_out = ctx.enqueue_create_buffer[DType.float32](n_relu1)
    var relu1_out = ctx.enqueue_create_buffer[DType.float32](n_relu1)

    comptime in_layout = row_major[B, 1, H_INPUT, W]()
    comptime mid32_layout = row_major[B, 32, H_INPUT, W]()
    comptime w31_layout = row_major[32, 1, 3, 3]()
    comptime p32_layout = row_major[32]()
    comptime p1_layout = row_major[1]()
    comptime relu1_flat = row_major[B * 32 * H_INPUT * W]()

    var x_t = TileTensor(x_buf, in_layout)
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
        DType.float32, type_of(in_layout), type_of(w31_layout),
        type_of(p1_layout), type_of(mid32_layout),
        3, 3, False, 256,
    ]
    ctx.enqueue_function[conv1_k, conv1_k](
        conv1_t, x_t, w_t, dummy2_t,
        B, 1, 32, H_INPUT, W, H_INPUT, W, 1, 1, 1, 1,
        grid_dim=B * 32 * H_INPUT, block_dim=256,
    )
    comptime bn_k = batchnorm2d_kernel[
        DType.float32, type_of(mid32_layout), type_of(p32_layout),
        type_of(mid32_layout), 256,
    ]
    ctx.enqueue_function[bn_k, bn_k](
        bn1_t, conv1_t, bn1_w_t, bn1_b_t, bn1_m_t, bn1_v_t,
        B, 32, H_INPUT, W, EPS_BN,
        grid_dim=B * 32 * H_INPUT, block_dim=256,
    )
    comptime relu_k = relu_kernel[
        DType.float32, type_of(relu1_flat), type_of(relu1_flat), BLOCK_PW,
    ]
    ctx.enqueue_function[relu_k, relu_k](
        relu1_flat_t, bn1_flat_t, n_relu1,
        grid_dim=ceildiv(n_relu1, BLOCK_PW), block_dim=BLOCK_PW,
    )

    # ---- Now run the 4 BasicResBlocks via the helper.
    var l1_0_out = ctx.enqueue_create_buffer[DType.float32](B * 32 * 40 * W)
    var l1_1_out = ctx.enqueue_create_buffer[DType.float32](B * 32 * 40 * W)
    var l2_0_out = ctx.enqueue_create_buffer[DType.float32](B * 32 * 20 * W)
    var l2_1_out = ctx.enqueue_create_buffer[DType.float32](B * 32 * 20 * W)
    var rb_dummy = ctx.enqueue_create_buffer[DType.float32](32)

    basic_resblock[B, 32, 32, 80, 40, W, True, 2](
        ctx, relu1_out, l1_0_out,
        L10_c1, L10_c2,
        L10_bn1w, L10_bn1b, L10_bn1m, L10_bn1v,
        L10_bn2w, L10_bn2b, L10_bn2m, L10_bn2v,
        L10_scw, L10_scbnw, L10_scbnb, L10_scbnm, L10_scbnv,
        rb_dummy,
    )
    # Unused-slot placeholders (distinct mut buffers).
    var u11a = ctx.enqueue_create_buffer[DType.float32](32)
    var u11b = ctx.enqueue_create_buffer[DType.float32](32)
    var u11c = ctx.enqueue_create_buffer[DType.float32](32)
    var u11d = ctx.enqueue_create_buffer[DType.float32](32)
    var u11e = ctx.enqueue_create_buffer[DType.float32](32)
    var rb_dummy_11 = ctx.enqueue_create_buffer[DType.float32](32)
    basic_resblock[B, 32, 32, 40, 40, W, False, 1](
        ctx, l1_0_out, l1_1_out,
        L11_c1, L11_c2,
        L11_bn1w, L11_bn1b, L11_bn1m, L11_bn1v,
        L11_bn2w, L11_bn2b, L11_bn2m, L11_bn2v,
        u11a, u11b, u11c, u11d, u11e,
        rb_dummy_11,
    )
    var rb_dummy_20 = ctx.enqueue_create_buffer[DType.float32](32)
    basic_resblock[B, 32, 32, 40, 20, W, True, 2](
        ctx, l1_1_out, l2_0_out,
        L20_c1, L20_c2,
        L20_bn1w, L20_bn1b, L20_bn1m, L20_bn1v,
        L20_bn2w, L20_bn2b, L20_bn2m, L20_bn2v,
        L20_scw, L20_scbnw, L20_scbnb, L20_scbnm, L20_scbnv,
        rb_dummy_20,
    )
    var u21a = ctx.enqueue_create_buffer[DType.float32](32)
    var u21b = ctx.enqueue_create_buffer[DType.float32](32)
    var u21c = ctx.enqueue_create_buffer[DType.float32](32)
    var u21d = ctx.enqueue_create_buffer[DType.float32](32)
    var u21e = ctx.enqueue_create_buffer[DType.float32](32)
    var rb_dummy_21 = ctx.enqueue_create_buffer[DType.float32](32)
    basic_resblock[B, 32, 32, 20, 20, W, False, 1](
        ctx, l2_0_out, l2_1_out,
        L21_c1, L21_c2,
        L21_bn1w, L21_bn1b, L21_bn1m, L21_bn1v,
        L21_bn2w, L21_bn2b, L21_bn2m, L21_bn2v,
        u21a, u21b, u21c, u21d, u21e,
        rb_dummy_21,
    )

    # ---- Final conv2 (32->32, 3x3, stride=(2,1), pad=1) + bn2 + relu.
    var n_pre2 = B * 32 * 20 * W
    var n_post2 = B * 32 * 10 * W
    var c2_out = ctx.enqueue_create_buffer[DType.float32](n_post2)
    var bn2_out = ctx.enqueue_create_buffer[DType.float32](n_post2)
    var final_out = ctx.enqueue_create_buffer[DType.float32](n_post2)

    comptime l21_layout = row_major[B, 32, 20, W]()
    comptime final_layout = row_major[B, 32, 10, W]()
    comptime w_3232 = row_major[32, 32, 3, 3]()
    comptime final_flat = row_major[B * 32 * 10 * W]()
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
        3, 3, False, 256,
    ]
    ctx.enqueue_function[conv2_k, conv2_k](
        c2_out_t, l21_t, w_conv2_t, dummy1_t,
        B, 32, 32, 20, W, 10, W, 2, 1, 1, 1,
        grid_dim=B * 32 * 10, block_dim=256,
    )
    comptime bn2_k = batchnorm2d_kernel[
        DType.float32, type_of(final_layout), type_of(p32_layout),
        type_of(final_layout), 256,
    ]
    ctx.enqueue_function[bn2_k, bn2_k](
        bn2_out_t, c2_out_t, bn2_w_t, bn2_b_t, bn2_m_t, bn2_v_t,
        B, 32, 10, W, EPS_BN,
        grid_dim=B * 32 * 10, block_dim=256,
    )
    comptime final_relu = relu_kernel[
        DType.float32, type_of(final_flat), type_of(final_flat), BLOCK_PW,
    ]
    ctx.enqueue_function[final_relu, final_relu](
        final_flat_t, bn2_flat_t, n_post2,
        grid_dim=ceildiv(n_post2, BLOCK_PW), block_dim=BLOCK_PW,
    )
    ctx.synchronize()

    # ---- The reshape (B,32,10,T) -> (B,320,T) is a no-op in row-major memory.
    # fcm_out.bin has shape (B, 320, T) = (1, 320, 998). Same element count
    # and same memory order as (B, 32, 10, T) row-major.
    var exp = load_fp32(fix + "fcm_out.bin")
    var max_abs: Float32 = 0.0
    var sum_abs: Float64 = 0.0
    with final_out.map_to_host() as h:
        for i in range(n_post2):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_abs += Float64(d)
            assert_almost_equal(h[i], exp.data[i], atol=3.0e-4)
    print("CAMPPlus FCM full — max abs:", max_abs,
          " mean:", sum_abs / Float64(n_post2),
          " (output (1,320,", W, "))")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
