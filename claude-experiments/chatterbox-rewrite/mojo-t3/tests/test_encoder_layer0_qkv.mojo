"""
Parity test for encoder.encoders[0].self_attn.forward_qkv.

Input:  enc_layer_0_norm_mha_out.bin   (1, 376, 512)
Target: enc_layer_0_{q,k,v}.bin         (1, 8, 376, 64) each

Sequence:
  q = Linear(linear_q)(x).view(B, T, H, D_k).transpose(1, 2)   # (B, H, T, D_k)
  k = same with linear_k
  v = same with linear_v
"""
from std.math import ceildiv
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from fixture import load_fp32
from attention import qkv_proj_reshape_kernel


comptime B = 1
comptime T = 376
comptime D = 512
comptime H = 8
comptime D_K = 64
comptime BLOCK = 64   # block over d_k=64


def upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def test_qkv_proj() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var fix = "tests/fixtures/s3gen/"
    var ctx = DeviceContext()

    var x_in = load_fp32(fix + "enc_layer_0_norm_mha_out.bin")
    var w_q = load_fp32(fix + "weights/flow__encoder__encoders__0__self_attn__linear_q__weight.bin")
    var b_q = load_fp32(fix + "weights/flow__encoder__encoders__0__self_attn__linear_q__bias.bin")
    var w_k = load_fp32(fix + "weights/flow__encoder__encoders__0__self_attn__linear_k__weight.bin")
    var b_k = load_fp32(fix + "weights/flow__encoder__encoders__0__self_attn__linear_k__bias.bin")
    var w_v = load_fp32(fix + "weights/flow__encoder__encoders__0__self_attn__linear_v__weight.bin")
    var b_v = load_fp32(fix + "weights/flow__encoder__encoders__0__self_attn__linear_v__bias.bin")
    var exp_q = load_fp32(fix + "enc_layer_0_q.bin")
    var exp_k = load_fp32(fix + "enc_layer_0_k.bin")
    var exp_v = load_fp32(fix + "enc_layer_0_v.bin")

    var n_x = B * T * D
    var n_w = D * D
    var n_qkv = B * H * T * D_K

    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_x)
    var w_q_buf = ctx.enqueue_create_buffer[DType.float32](n_w)
    var b_q_buf = ctx.enqueue_create_buffer[DType.float32](D)
    var w_k_buf = ctx.enqueue_create_buffer[DType.float32](n_w)
    var b_k_buf = ctx.enqueue_create_buffer[DType.float32](D)
    var w_v_buf = ctx.enqueue_create_buffer[DType.float32](n_w)
    var b_v_buf = ctx.enqueue_create_buffer[DType.float32](D)
    var q_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv)
    var k_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv)
    var v_buf = ctx.enqueue_create_buffer[DType.float32](n_qkv)

    upload(x_buf, x_in.data, n_x)
    upload(w_q_buf, w_q.data, n_w)
    upload(b_q_buf, b_q.data, D)
    upload(w_k_buf, w_k.data, n_w)
    upload(b_k_buf, b_k.data, D)
    upload(w_v_buf, w_v.data, n_w)
    upload(b_v_buf, b_v.data, D)

    comptime in_layout = row_major[B, T, D]()
    comptime w_layout = row_major[D, D]()
    comptime p_layout = row_major[D]()
    comptime qkv_layout = row_major[B, H, T, D_K]()
    var x_t = TileTensor(x_buf, in_layout)
    var w_q_t = TileTensor(w_q_buf, w_layout)
    var b_q_t = TileTensor(b_q_buf, p_layout)
    var w_k_t = TileTensor(w_k_buf, w_layout)
    var b_k_t = TileTensor(b_k_buf, p_layout)
    var w_v_t = TileTensor(w_v_buf, w_layout)
    var b_v_t = TileTensor(b_v_buf, p_layout)
    var q_t = TileTensor(q_buf, qkv_layout)
    var k_t = TileTensor(k_buf, qkv_layout)
    var v_t = TileTensor(v_buf, qkv_layout)

    comptime qkv_k = qkv_proj_reshape_kernel[
        DType.float32, type_of(in_layout), type_of(w_layout),
        type_of(p_layout), type_of(qkv_layout),
        True, BLOCK,
    ]
    ctx.enqueue_function[qkv_k, qkv_k](
        q_t, x_t, w_q_t, b_q_t,
        B, T, H, D_K, D,
        grid_dim=B * H * T, block_dim=BLOCK,
    )
    ctx.enqueue_function[qkv_k, qkv_k](
        k_t, x_t, w_k_t, b_k_t,
        B, T, H, D_K, D,
        grid_dim=B * H * T, block_dim=BLOCK,
    )
    ctx.enqueue_function[qkv_k, qkv_k](
        v_t, x_t, w_v_t, b_v_t,
        B, T, H, D_K, D,
        grid_dim=B * H * T, block_dim=BLOCK,
    )
    ctx.synchronize()

    var max_q: Float32 = 0.0
    var max_k: Float32 = 0.0
    var max_v: Float32 = 0.0
    with q_buf.map_to_host() as h:
        for i in range(n_qkv):
            var d = h[i] - exp_q.data[i]
            if d < 0.0: d = -d
            if d > max_q: max_q = d
            assert_almost_equal(h[i], exp_q.data[i], atol=1.0e-3)
    with k_buf.map_to_host() as h:
        for i in range(n_qkv):
            var d = h[i] - exp_k.data[i]
            if d < 0.0: d = -d
            if d > max_k: max_k = d
            assert_almost_equal(h[i], exp_k.data[i], atol=1.0e-3)
    with v_buf.map_to_host() as h:
        for i in range(n_qkv):
            var d = h[i] - exp_v.data[i]
            if d < 0.0: d = -d
            if d > max_v: max_v = d
            assert_almost_equal(h[i], exp_v.data[i], atol=1.0e-3)
    print("encoder layer0 qkv_proj — max abs: q=", max_q, " k=", max_k, " v=", max_v)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
