"""
Parity test for the full FCM head.layer1 (two BasicResBlocks):
  block0: stride=2, shortcut    (1,32,80,998) -> (1,32,40,998)
  block1: stride=1, no shortcut (1,32,40,998) -> (1,32,40,998)

Inputs : fcm_relu1_out.bin     (1, 32, 80, 998)
Target : fcm_layer1_out.bin     (1, 32, 40, 998)
"""
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer

from fixture import load_fp32
from campplus import basic_resblock


comptime B = 1
comptime C = 32
comptime H_IN = 80
comptime H_MID = 40
comptime W = 998


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def upload_w(ctx: DeviceContext, fix: String, name: String) raises -> DeviceBuffer[DType.float32]:
    var t = load_fp32(fix + name)
    var n = len(t.data)
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    upload(buf, t.data, n)
    return buf


def test_layer1_full() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/campplus/"
    var ctx = DeviceContext()

    # Input.
    var x_in = load_fp32(fix + "fcm_relu1_out.bin")
    var n_in = B * C * H_IN * W
    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_in)
    upload(x_buf, x_in.data, n_in)

    # Block0 weights.
    var b0_c1 = upload_w(ctx, fix, "weights/head__layer1__0__conv1__weight.bin")
    var b0_c2 = upload_w(ctx, fix, "weights/head__layer1__0__conv2__weight.bin")
    var b0_bn1_w = upload_w(ctx, fix, "weights/head__layer1__0__bn1__weight.bin")
    var b0_bn1_b = upload_w(ctx, fix, "weights/head__layer1__0__bn1__bias.bin")
    var b0_bn1_m = upload_w(ctx, fix, "weights/head__layer1__0__bn1__running_mean.bin")
    var b0_bn1_v = upload_w(ctx, fix, "weights/head__layer1__0__bn1__running_var.bin")
    var b0_bn2_w = upload_w(ctx, fix, "weights/head__layer1__0__bn2__weight.bin")
    var b0_bn2_b = upload_w(ctx, fix, "weights/head__layer1__0__bn2__bias.bin")
    var b0_bn2_m = upload_w(ctx, fix, "weights/head__layer1__0__bn2__running_mean.bin")
    var b0_bn2_v = upload_w(ctx, fix, "weights/head__layer1__0__bn2__running_var.bin")
    var b0_sc_w = upload_w(ctx, fix, "weights/head__layer1__0__shortcut__0__weight.bin")
    var b0_sc_bnw = upload_w(ctx, fix, "weights/head__layer1__0__shortcut__1__weight.bin")
    var b0_sc_bnb = upload_w(ctx, fix, "weights/head__layer1__0__shortcut__1__bias.bin")
    var b0_sc_bnm = upload_w(ctx, fix, "weights/head__layer1__0__shortcut__1__running_mean.bin")
    var b0_sc_bnv = upload_w(ctx, fix, "weights/head__layer1__0__shortcut__1__running_var.bin")
    # Block1 weights.
    var b1_c1 = upload_w(ctx, fix, "weights/head__layer1__1__conv1__weight.bin")
    var b1_c2 = upload_w(ctx, fix, "weights/head__layer1__1__conv2__weight.bin")
    var b1_bn1_w = upload_w(ctx, fix, "weights/head__layer1__1__bn1__weight.bin")
    var b1_bn1_b = upload_w(ctx, fix, "weights/head__layer1__1__bn1__bias.bin")
    var b1_bn1_m = upload_w(ctx, fix, "weights/head__layer1__1__bn1__running_mean.bin")
    var b1_bn1_v = upload_w(ctx, fix, "weights/head__layer1__1__bn1__running_var.bin")
    var b1_bn2_w = upload_w(ctx, fix, "weights/head__layer1__1__bn2__weight.bin")
    var b1_bn2_b = upload_w(ctx, fix, "weights/head__layer1__1__bn2__bias.bin")
    var b1_bn2_m = upload_w(ctx, fix, "weights/head__layer1__1__bn2__running_mean.bin")
    var b1_bn2_v = upload_w(ctx, fix, "weights/head__layer1__1__bn2__running_var.bin")

    var dummy = ctx.enqueue_create_buffer[DType.float32](C)

    # block0 → mid_buf
    var n_mid = B * C * H_MID * W
    var mid_buf = ctx.enqueue_create_buffer[DType.float32](n_mid)
    basic_resblock[B, C, C, H_IN, H_MID, W, True, 2](
        ctx, x_buf, mid_buf,
        b0_c1, b0_c2,
        b0_bn1_w, b0_bn1_b, b0_bn1_m, b0_bn1_v,
        b0_bn2_w, b0_bn2_b, b0_bn2_m, b0_bn2_v,
        b0_sc_w, b0_sc_bnw, b0_sc_bnb, b0_sc_bnm, b0_sc_bnv,
        dummy,
    )
    # block1 → out_buf
    var out_buf = ctx.enqueue_create_buffer[DType.float32](n_mid)
    # Unused-shortcut placeholder buffers (mut aliasing forbids reusing one).
    var u1 = ctx.enqueue_create_buffer[DType.float32](C)
    var u2 = ctx.enqueue_create_buffer[DType.float32](C)
    var u3 = ctx.enqueue_create_buffer[DType.float32](C)
    var u4 = ctx.enqueue_create_buffer[DType.float32](C)
    var u5 = ctx.enqueue_create_buffer[DType.float32](C)
    var dummy1 = ctx.enqueue_create_buffer[DType.float32](C)
    basic_resblock[B, C, C, H_MID, H_MID, W, False, 1](
        ctx, mid_buf, out_buf,
        b1_c1, b1_c2,
        b1_bn1_w, b1_bn1_b, b1_bn1_m, b1_bn1_v,
        b1_bn2_w, b1_bn2_b, b1_bn2_m, b1_bn2_v,
        u1, u2, u3, u4, u5,
        dummy1,
    )
    ctx.synchronize()

    var exp = load_fp32(fix + "fcm_layer1_out.bin")
    var max_abs: Float32 = 0.0
    var sum_abs: Float64 = 0.0
    with out_buf.map_to_host() as h:
        for i in range(n_mid):
            var d = h[i] - exp.data[i]
            if d < 0.0: d = -d
            if d > max_abs: max_abs = d
            sum_abs += Float64(d)
            assert_almost_equal(h[i], exp.data[i], atol=2.0e-4)
    print("CAMPPlus layer1 (2 BasicResBlocks) — max abs:", max_abs,
          " mean:", sum_abs / Float64(n_mid))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
