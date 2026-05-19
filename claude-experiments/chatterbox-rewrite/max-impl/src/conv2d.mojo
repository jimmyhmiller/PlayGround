"""2D convolution + BatchNorm2D for FCM (CAMPPlus voice encoder front-end).

Conv2D semantics: input (B, C_in, H_in, W_in), weight (C_out, C_in, kH, kW),
optional bias (C_out,), with strides=(sH, sW), pads=(pH, pW). No groups, no dilation.
Output (B, C_out, H_out, W_out) where
    H_out = (H_in + 2*pH - kH) // sH + 1
    W_out = (W_in + 2*pW - kW) // sW + 1
"""
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList


@fieldwise_init
struct Conv2d(Copyable, Movable):
    var weight: DeviceBuffer[DType.float32]   # (C_out, C_in, kH, kW)
    var bias:   DeviceBuffer[DType.float32]   # (C_out,) — may be zero-len if no bias
    var c_in:   Int
    var c_out:  Int
    var kh:     Int
    var kw:     Int
    var sh:     Int
    var sw:     Int
    var ph:     Int
    var pw:     Int
    var has_bias: Bool


@fieldwise_init
struct BatchNorm2d(Copyable, Movable):
    """y[c, h, w] = (x[c, h, w] - running_mean[c]) / sqrt(running_var[c] + eps) * weight[c] + bias[c]."""
    var weight: DeviceBuffer[DType.float32]        # (C,) — gamma
    var bias:   DeviceBuffer[DType.float32]        # (C,) — beta
    var running_mean: DeviceBuffer[DType.float32]  # (C,)
    var running_var:  DeviceBuffer[DType.float32]  # (C,)
    var channels: Int
    var eps: Float32


def conv2d_forward(
    mut ctx: DeviceContext,
    mut module: Conv2d,
    mut x_buf: DeviceBuffer[DType.float32],     # (B, C_in, H_in, W_in)
    mut out_buf: DeviceBuffer[DType.float32],   # (B, C_out, H_out, W_out)
    batch: Int, h_in: Int, w_in: Int, h_out: Int, w_out: Int,
) raises:
    """Apply 2D convolution (NCHW). Caller pre-computes h_out, w_out."""
    var x_ptr = x_buf.unsafe_ptr()
    var w_ptr = module.weight.unsafe_ptr()
    var b_ptr = module.bias.unsafe_ptr()
    var o_ptr = out_buf.unsafe_ptr()

    var c_in = module.c_in
    var c_out = module.c_out
    var kh = module.kh
    var kw = module.kw
    var sh = module.sh
    var sw = module.sw
    var ph = module.ph
    var pw = module.pw
    var has_bias = module.has_bias

    var total = batch * c_out * h_out * w_out

    @always_inline
    @parameter
    @__copy_capture(
        x_ptr, w_ptr, b_ptr, o_ptr,
        c_in, c_out, kh, kw, sh, sw, ph, pw,
        h_in, w_in, h_out, w_out, has_bias,
    )
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        # i indexes (B, C_out, H_out, W_out).
        var b = i // (c_out * h_out * w_out)
        var rem = i - b * c_out * h_out * w_out
        var oc = rem // (h_out * w_out)
        var rem2 = rem - oc * h_out * w_out
        var oh = rem2 // w_out
        var ow = rem2 - oh * w_out

        var acc: Float32 = 0.0
        if has_bias:
            acc = b_ptr[oc]

        for ic in range(c_in):
            for kr in range(kh):
                var ih = oh * sh + kr - ph
                if ih < 0 or ih >= h_in: continue
                for kc in range(kw):
                    var iw = ow * sw + kc - pw
                    if iw < 0 or iw >= w_in: continue
                    var x_v = x_ptr[
                        b * c_in * h_in * w_in
                        + ic * h_in * w_in
                        + ih * w_in
                        + iw
                    ]
                    var w_v = w_ptr[
                        oc * c_in * kh * kw
                        + ic * kh * kw
                        + kr * kw
                        + kc
                    ]
                    acc += x_v * w_v
        o_ptr[i] = acc
    elementwise[func, simd_width=1, target="gpu"](
        IndexList[1](total), DeviceContextPtr(ctx),
    )


def batchnorm2d_forward(
    mut ctx: DeviceContext,
    mut module: BatchNorm2d,
    mut x_buf: DeviceBuffer[DType.float32],     # (B, C, H, W)
    mut out_buf: DeviceBuffer[DType.float32],   # (B, C, H, W)
    batch: Int, h: Int, w: Int,
) raises:
    """Apply BN2D: y = (x - mu)/sqrt(var+eps) * gamma + beta, per-channel."""
    var x_ptr = x_buf.unsafe_ptr()
    var w_ptr = module.weight.unsafe_ptr()
    var b_ptr = module.bias.unsafe_ptr()
    var mu_ptr = module.running_mean.unsafe_ptr()
    var var_ptr = module.running_var.unsafe_ptr()
    var o_ptr = out_buf.unsafe_ptr()
    var c = module.channels
    var eps = module.eps

    from std.math import sqrt as msqrt

    var total = batch * c * h * w

    @always_inline
    @parameter
    @__copy_capture(x_ptr, w_ptr, b_ptr, mu_ptr, var_ptr, o_ptr, c, h, w, eps)
    def bn_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (c * h * w)
        var rem = i - bi * c * h * w
        var ci = rem // (h * w)
        var x_v = x_ptr[i]
        var mu = mu_ptr[ci]
        var vr = var_ptr[ci]
        var gamma = w_ptr[ci]
        var beta = b_ptr[ci]
        var inv_std = Float32(1.0) / msqrt(vr + eps)
        o_ptr[i] = (x_v - mu) * inv_std * gamma + beta
    elementwise[bn_fn, simd_width=1, target="gpu"](
        IndexList[1](total), DeviceContextPtr(ctx),
    )


def relu_inplace_2d(
    mut ctx: DeviceContext,
    mut buf: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    """In-place ReLU."""
    var p = buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(p)
    def relu_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        if p[i] < 0.0: p[i] = 0.0
    elementwise[relu_fn, simd_width=1, target="gpu"](
        IndexList[1](n), DeviceContextPtr(ctx),
    )


def add_inplace_2d(
    mut ctx: DeviceContext,
    mut dst: DeviceBuffer[DType.float32],
    mut src: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    """dst += src element-wise."""
    var dp = dst.unsafe_ptr()
    var sp = src.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(dp, sp)
    def add_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        dp[i] = dp[i] + sp[i]
    elementwise[add_fn, simd_width=1, target="gpu"](
        IndexList[1](n), DeviceContextPtr(ctx),
    )
