"""CAMPPlus speaker encoder — struct definitions matching real upstream
Chatterbox `speaker_encoder` weights.

Architecture (from chatterbox/models/s3gen/CAMPPlus):

  xvector
    tdnn      : 1×1 Conv1d 320→128 (k=5) + BN + nonlinear
    block1    : CAMDenseTDNNBlock (12 layers tdnnd1..tdnnd12), growth=32, in=128
    transit1  : 1×1 Conv1d 128 + 12*32 → 256 + BN
    block2    : CAMDenseTDNNBlock (24 layers), in=256, growth=32
    transit2  : Conv1d 256 + 24*32 → 512 + BN
    block3    : CAMDenseTDNNBlock (16 layers), in=512, growth=32
    transit3  : Conv1d 512 + 16*32 → 1024 + BN
    out_nonlinear  : BN(1024)
    dense          : Linear (192, 1024) + BN

  head : ResNetHead with bn1+conv1+bn2+conv2 + layer1 (2 ResBlocks) + layer2 (2 ResBlocks)

Each `tdnnd*` (CAMDenseTDNNLayer) is:
  nonlinear1 (BN) → linear1 (Conv1d 1x1) → nonlinear2 (BN) → cam_layer
  where cam_layer = CAMLayer:
    linear_local : Conv1d (32, 128, 3)         — context conv
    linear1      : Conv1d (64, 128, 1)         — point conv
    linear2      : Conv1d (32, 64, 1)          — output conv

Loader populates buffers; forward implementation is a follow-up.
"""
from std.math import sqrt
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from modules import Linear, linear_forward, relu, residual_add
from conv1d import Conv1d, conv1d_forward


@fieldwise_init
struct BatchNorm1d(Copyable, Movable):
    """PyTorch BatchNorm1d: weight (γ), bias (β), running_mean, running_var.
    forward = (x - mean) / sqrt(var + eps) * γ + β
    """
    var weight: DeviceBuffer[DType.float32]
    var bias: DeviceBuffer[DType.float32]
    var running_mean: DeviceBuffer[DType.float32]
    var running_var: DeviceBuffer[DType.float32]
    var channels: Int
    var eps: Float32


@fieldwise_init
struct CAMLayer(Copyable, Movable):
    """Context-Aware Masking layer used inside CAMDenseTDNNLayer."""
    var linear_local: Conv1d  # (32, 128, 3) — depthwise-like context conv
    var linear1: Conv1d        # (64, 128, 1) — point conv expanding
    var linear2: Conv1d        # (32, 64, 1)  — point conv contracting


@fieldwise_init
struct CAMDenseTDNNLayer(Copyable, Movable):
    """One tdnnd* layer = BN + Conv1d (1x1 128→128) + BN + CAMLayer."""
    var nonlinear1: BatchNorm1d
    var linear1: Conv1d
    var nonlinear2: BatchNorm1d
    var cam_layer: CAMLayer


@fieldwise_init
struct CAMDenseTDNNBlock(Copyable, Movable):
    """A block of N CAMDenseTDNNLayers with growth channels concatenated."""
    var layers: List[CAMDenseTDNNLayer]


@fieldwise_init
struct TransitLayer(Copyable, Movable):
    """Transit layer between dense TDNN blocks: BN + 1x1 Conv1d to expand channels."""
    var nonlinear: BatchNorm1d
    var linear: Conv1d


@fieldwise_init
struct TDNN(Copyable, Movable):
    """First TDNN layer: BN + 1x1 Conv1d (320→128) with kernel 5."""
    var linear: Conv1d
    var nonlinear: BatchNorm1d


@fieldwise_init
struct DenseLayer(Copyable, Movable):
    """Final dense head: Linear(1024 → 192) + BN(192)."""
    var linear: Conv1d   # implemented as 1x1 Conv1d in upstream
    var nonlinear: BatchNorm1d


@fieldwise_init
struct XVectorBackbone(Copyable, Movable):
    """xvector backbone:
       tdnn → block1 → transit1 → block2 → transit2 → block3 → transit3
       → out_nonlinear → dense
    """
    var tdnn: TDNN
    var block1: CAMDenseTDNNBlock
    var transit1: TransitLayer
    var block2: CAMDenseTDNNBlock
    var transit2: TransitLayer
    var block3: CAMDenseTDNNBlock
    var transit3: TransitLayer
    var out_nonlinear: BatchNorm1d
    var dense: DenseLayer


@fieldwise_init
struct ResNetBasicBlock(Copyable, Movable):
    """One residual block in ResNet head: bn1+conv1+bn2+conv2 (+ optional downsample)."""
    var bn1: BatchNorm1d
    var conv1: Conv1d
    var bn2: BatchNorm1d
    var conv2: Conv1d


@fieldwise_init
struct ResNetHead(Copyable, Movable):
    """Top of CAMPPlus: stem (bn1+conv1+bn2+conv2) + layer1 (2 blocks) + layer2 (2 blocks).

    Note: the head receives the dense feature and refines speaker embedding.
    """
    var bn1: BatchNorm1d
    var conv1: Conv1d
    var bn2: BatchNorm1d
    var conv2: Conv1d
    var layer1: List[ResNetBasicBlock]   # 2 blocks
    var layer2: List[ResNetBasicBlock]   # 2 blocks


@fieldwise_init
struct CAMPPlus(Copyable, Movable):
    var xvector: XVectorBackbone
    var head: ResNetHead


# ============================================================================
# Forward helpers
# ============================================================================


def batchnorm1d_forward(
    mut ctx: DeviceContext,
    mut bn: BatchNorm1d,
    mut in_buf: DeviceBuffer[DType.float32],     # (B, C, T)
    mut out_buf: DeviceBuffer[DType.float32],    # (B, C, T)
    b: Int, c: Int, t: Int,
) raises:
    """PyTorch BatchNorm1d inference forward:
       y = (x - running_mean) / sqrt(running_var + eps) * weight + bias
    """
    var x_ptr = in_buf.unsafe_ptr()
    var o_ptr = out_buf.unsafe_ptr()
    var w_ptr = bn.weight.unsafe_ptr()
    var b_ptr = bn.bias.unsafe_ptr()
    var rm_ptr = bn.running_mean.unsafe_ptr()
    var rv_ptr = bn.running_var.unsafe_ptr()
    var eps = bn.eps

    @always_inline
    @parameter
    @__copy_capture(x_ptr, o_ptr, w_ptr, b_ptr, rm_ptr, rv_ptr, eps, c, t)
    def bn_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (c * t)
        var rem = i - bi * c * t
        var ci = rem // t
        var inv_std = 1.0 / sqrt(rv_ptr[ci] + eps)
        o_ptr[i] = (x_ptr[i] - rm_ptr[ci]) * inv_std * w_ptr[ci] + b_ptr[ci]
    elementwise[bn_fn, simd_width=1, target="gpu"](
        IndexList[1](b * c * t), DeviceContextPtr(ctx),
    )


def relu_inplace_bct(
    mut ctx: DeviceContext,
    mut x_buf: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    var x_ptr = x_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(x_ptr)
    def r_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        if x_ptr[i] < 0.0:
            x_ptr[i] = 0.0
    elementwise[r_fn, simd_width=1, target="gpu"](
        IndexList[1](n), DeviceContextPtr(ctx),
    )


def tdnn_first_forward(
    mut ctx: DeviceContext,
    mut tdnn: TDNN,
    mut in_buf: DeviceBuffer[DType.float32],     # (B, 320, T_in)
    mut out_buf: DeviceBuffer[DType.float32],    # (B, 128, T_out)
    b: Int, t_in: Int, t_out: Int,
) raises:
    """First TDNN: Conv1d (320→128, k=5, stride=2, padding=2) + BN + ReLU.

    Loader passed padding=0 (since padding=-1 in upstream means "compute as
    (k-1)//2 = 2"). We need to pad the input by 2 each side externally OR
    call conv1d with padding=2 — but our loader fixed padding=0. So we'll
    pre-pad on each side.
    """
    var c_in = 320
    var c_out = 128
    var k = 5
    var stride = 2
    var pad = 2

    # Pad input by `pad` on each side.
    var t_padded = t_in + 2 * pad
    var x_padded = ctx.enqueue_create_buffer[DType.float32](b * c_in * t_padded)
    var src_ptr = in_buf.unsafe_ptr()
    var dst_ptr = x_padded.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(src_ptr, dst_ptr, c_in, t_in, t_padded, pad)
    def pad_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (c_in * t_padded)
        var rem = i - bi * c_in * t_padded
        var ci = rem // t_padded
        var ti = rem - ci * t_padded
        if ti < pad or ti >= pad + t_in:
            dst_ptr[i] = 0.0
        else:
            dst_ptr[i] = src_ptr[bi * c_in * t_in + ci * t_in + (ti - pad)]
    elementwise[pad_fn, simd_width=1, target="gpu"](
        IndexList[1](b * c_in * t_padded), DeviceContextPtr(ctx),
    )

    # Conv1d.
    var conv_out = ctx.enqueue_create_buffer[DType.float32](b * c_out * t_out)
    conv1d_forward(ctx, tdnn.linear, x_padded, conv_out, b, t_padded, t_out)

    # BN + ReLU.
    batchnorm1d_forward(ctx, tdnn.nonlinear, conv_out, out_buf, b, c_out, t_out)
    relu_inplace_bct(ctx, out_buf, b * c_out * t_out)


def seg_pooling_avg(
    mut ctx: DeviceContext,
    mut in_buf: DeviceBuffer[DType.float32],     # (B, C, T)
    mut out_buf: DeviceBuffer[DType.float32],    # (B, C, T) — same shape (expanded back)
    b: Int, c: Int, t: Int, seg_len: Int,
) raises:
    """avg_pool1d (kernel_size=seg_len, stride=seg_len, ceil_mode=True) then
    expand back to original length by repeating each segment value seg_len times,
    truncated to T.

    Each output element out[b, c, t] = avg_{ti in seg starting at floor(t/seg_len)*seg_len} in[b, c, ti].
    """
    var in_ptr = in_buf.unsafe_ptr()
    var out_ptr = out_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(in_ptr, out_ptr, c, t, seg_len)
    def sp_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (c * t)
        var rem = i - bi * c * t
        var ci = rem // t
        var ti = rem - ci * t
        var seg_idx = ti // seg_len
        var seg_start = seg_idx * seg_len
        var seg_end = seg_start + seg_len
        if seg_end > t: seg_end = t
        var acc: Float32 = 0.0
        var cnt: Float32 = 0.0
        for s in range(seg_start, seg_end):
            acc += in_ptr[bi * c * t + ci * t + s]
            cnt += 1.0
        if cnt > 0.0:
            out_ptr[i] = acc / cnt
        else:
            out_ptr[i] = 0.0
    elementwise[sp_fn, simd_width=1, target="gpu"](
        IndexList[1](b * c * t), DeviceContextPtr(ctx),
    )


def mean_global_bct(
    mut ctx: DeviceContext,
    mut in_buf: DeviceBuffer[DType.float32],     # (B, C, T)
    mut out_buf: DeviceBuffer[DType.float32],    # (B, C) — mean over T
    b: Int, c: Int, t: Int,
) raises:
    """Compute mean over T axis: out[b, c] = mean_{ti} in[b, c, ti]."""
    var in_ptr = in_buf.unsafe_ptr()
    var o_ptr = out_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(in_ptr, o_ptr, c, t)
    def m_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // c
        var ci = i - bi * c
        var acc: Float32 = 0.0
        for ti in range(t):
            acc += in_ptr[bi * c * t + ci * t + ti]
        o_ptr[i] = acc / Float32(t)
    elementwise[m_fn, simd_width=1, target="gpu"](
        IndexList[1](b * c), DeviceContextPtr(ctx),
    )


def cam_layer_forward(
    mut ctx: DeviceContext,
    mut cam: CAMLayer,
    mut in_buf: DeviceBuffer[DType.float32],     # (B, bn_channels, T)
    mut out_buf: DeviceBuffer[DType.float32],    # (B, out_channels, T)
    b: Int, bn_channels: Int, out_channels: Int, t: Int,
    kernel_size: Int, padding: Int, dilation: Int,
) raises:
    """CAMLayer forward:
       y = linear_local(x)                          # (B, out, T)
       global_mean = x.mean(-1, keepdim=True)        # (B, bn, 1)
       seg = seg_pooling(x, 100, avg, ceil)          # expanded to (B, bn, T)
       context = global_mean + seg                   # (B, bn, T)
       context = relu(linear1(context))              # (B, bn//2, T)
       m = sigmoid(linear2(context))                 # (B, out, T)
       return y * m
    """
    # 1. linear_local — depthwise-like Conv1d with dilation. Need to pad input.
    var pad_total = 2 * padding
    var t_pad = t + pad_total
    var x_padded = ctx.enqueue_create_buffer[DType.float32](b * bn_channels * t_pad)
    var src_ptr = in_buf.unsafe_ptr()
    var dst_ptr = x_padded.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(src_ptr, dst_ptr, bn_channels, t, t_pad, padding)
    def pad_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (bn_channels * t_pad)
        var rem = i - bi * bn_channels * t_pad
        var ci = rem // t_pad
        var ti = rem - ci * t_pad
        if ti < padding or ti >= padding + t:
            dst_ptr[i] = 0.0
        else:
            dst_ptr[i] = src_ptr[bi * bn_channels * t + ci * t + (ti - padding)]
    elementwise[pad_fn, simd_width=1, target="gpu"](
        IndexList[1](b * bn_channels * t_pad), DeviceContextPtr(ctx),
    )

    var y = ctx.enqueue_create_buffer[DType.float32](b * out_channels * t)
    conv1d_forward(ctx, cam.linear_local, x_padded, y, b, t_pad, t)

    # 2. Compute context: global mean + seg-pool, both (B, bn, T)-shaped.
    var gmean = ctx.enqueue_create_buffer[DType.float32](b * bn_channels)
    mean_global_bct(ctx, in_buf, gmean, b, bn_channels, t)
    var seg = ctx.enqueue_create_buffer[DType.float32](b * bn_channels * t)
    seg_pooling_avg(ctx, in_buf, seg, b, bn_channels, t, 100)

    # context = seg + broadcast(gmean over T).
    var ctx_buf = ctx.enqueue_create_buffer[DType.float32](b * bn_channels * t)
    var s_ptr = seg.unsafe_ptr()
    var g_ptr = gmean.unsafe_ptr()
    var c_ptr = ctx_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(s_ptr, g_ptr, c_ptr, bn_channels, t)
    def ctx_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (bn_channels * t)
        var rem = i - bi * bn_channels * t
        var ci = rem // t
        c_ptr[i] = s_ptr[i] + g_ptr[bi * bn_channels + ci]
    elementwise[ctx_fn, simd_width=1, target="gpu"](
        IndexList[1](b * bn_channels * t), DeviceContextPtr(ctx),
    )

    # 3. linear1 (bn → bn//2) + ReLU.
    var bn_red = bn_channels // 2
    var ctx_l1 = ctx.enqueue_create_buffer[DType.float32](b * bn_red * t)
    conv1d_forward(ctx, cam.linear1, ctx_buf, ctx_l1, b, t, t)
    relu_inplace_bct(ctx, ctx_l1, b * bn_red * t)

    # 4. linear2 (bn//2 → out) + Sigmoid.
    var ctx_l2 = ctx.enqueue_create_buffer[DType.float32](b * out_channels * t)
    conv1d_forward(ctx, cam.linear2, ctx_l1, ctx_l2, b, t, t)
    # Sigmoid in-place.
    from std.math import exp as mexp
    var l2_ptr = ctx_l2.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(l2_ptr)
    def sig_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        l2_ptr[i] = 1.0 / (1.0 + mexp(-l2_ptr[i]))
    elementwise[sig_fn, simd_width=1, target="gpu"](
        IndexList[1](b * out_channels * t), DeviceContextPtr(ctx),
    )

    # 5. out = y * m.
    var y_ptr = y.unsafe_ptr()
    var o_ptr = out_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(y_ptr, l2_ptr, o_ptr)
    def mul_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        o_ptr[i] = y_ptr[i] * l2_ptr[i]
    elementwise[mul_fn, simd_width=1, target="gpu"](
        IndexList[1](b * out_channels * t), DeviceContextPtr(ctx),
    )


def camdense_tdnn_layer_forward(
    mut ctx: DeviceContext,
    mut layer: CAMDenseTDNNLayer,
    mut in_buf: DeviceBuffer[DType.float32],     # (B, in_channels, T)
    mut out_buf: DeviceBuffer[DType.float32],    # (B, growth=32, T)
    b: Int, in_channels: Int, t: Int,
    kernel_size: Int, padding: Int, dilation: Int,
) raises:
    """One tdnnd layer:
       x = nonlinear1(x)        # BN + ReLU over in_channels
       x = linear1(x)            # 1x1 Conv in→128 (bn_channels)
       x = nonlinear2(x)         # BN + ReLU over 128
       x = cam_layer(x)          # → 32 (growth)
    """
    var bn_channels = 128
    var growth = 32

    var h_bn = ctx.enqueue_create_buffer[DType.float32](b * in_channels * t)
    batchnorm1d_forward(ctx, layer.nonlinear1, in_buf, h_bn, b, in_channels, t)
    relu_inplace_bct(ctx, h_bn, b * in_channels * t)

    var h_lin = ctx.enqueue_create_buffer[DType.float32](b * bn_channels * t)
    conv1d_forward(ctx, layer.linear1, h_bn, h_lin, b, t, t)

    var h_bn2 = ctx.enqueue_create_buffer[DType.float32](b * bn_channels * t)
    batchnorm1d_forward(ctx, layer.nonlinear2, h_lin, h_bn2, b, bn_channels, t)
    relu_inplace_bct(ctx, h_bn2, b * bn_channels * t)

    cam_layer_forward(
        ctx, layer.cam_layer, h_bn2, out_buf, b, bn_channels, growth, t,
        kernel_size, padding, dilation,
    )


def channel_concat_bct_cp(
    mut ctx: DeviceContext,
    mut a_buf: DeviceBuffer[DType.float32],     # (B, Ca, T)
    mut b_buf_in: DeviceBuffer[DType.float32],  # (B, Cb, T)
    mut out_buf: DeviceBuffer[DType.float32],   # (B, Ca + Cb, T)
    b: Int, ca: Int, cb: Int, t: Int,
) raises:
    """Concat two (B, C, T) tensors along the channel axis. (Renamed to avoid
    name collision with cfm_estimator_new's version.)
    """
    var a_ptr = a_buf.unsafe_ptr()
    var b_ptr = b_buf_in.unsafe_ptr()
    var o_ptr = out_buf.unsafe_ptr()
    var c_total = ca + cb

    @always_inline
    @parameter
    @__copy_capture(a_ptr, b_ptr, o_ptr, ca, cb, t, c_total)
    def cat_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (c_total * t)
        var rem = i - bi * c_total * t
        var ci = rem // t
        var ti = rem - ci * t
        if ci < ca:
            o_ptr[i] = a_ptr[bi * ca * t + ci * t + ti]
        else:
            o_ptr[i] = b_ptr[bi * cb * t + (ci - ca) * t + ti]
    elementwise[cat_func, simd_width=1, target="gpu"](
        IndexList[1](b * c_total * t), DeviceContextPtr(ctx),
    )


def camdense_tdnn_block_forward(
    mut ctx: DeviceContext,
    mut block: CAMDenseTDNNBlock,
    mut in_buf: DeviceBuffer[DType.float32],     # (B, base_in_channels, T)
    mut out_buf: DeviceBuffer[DType.float32],    # (B, base_in_channels + N*growth, T)
    b: Int, base_in_channels: Int, t: Int,
    kernel_size: Int, padding: Int, dilation: Int,
) raises:
    """DenseNet-style block: for each tdnnd layer, compute y = layer(x), then
    x = cat([x, y]). Final output channels = base + N*growth.
    """
    var growth = 32
    var n_layers = len(block.layers)

    # We need to maintain a growing buffer. Allocate two scratch buffers and
    # alternate to avoid allocating O(N) buffers.
    var max_channels = base_in_channels + n_layers * growth
    var buf_a = ctx.enqueue_create_buffer[DType.float32](b * max_channels * t)
    var buf_b = ctx.enqueue_create_buffer[DType.float32](b * max_channels * t)
    buf_a.enqueue_fill(0.0)
    # Copy (B, base_in_channels, T) of `in_buf` into the leading slots of buf_a's
    # (B, max_channels, T) layout. enqueue_copy can't help here because of the
    # stride mismatch on the channel axis — use an elementwise copy kernel.
    var src_ptr = in_buf.unsafe_ptr()
    var dst_ptr = buf_a.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(src_ptr, dst_ptr, base_in_channels, t, max_channels)
    def init_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i_idx = idx[0]
        var bi = i_idx // (base_in_channels * t)
        var rem = i_idx - bi * base_in_channels * t
        var ci = rem // t
        var ti = rem - ci * t
        dst_ptr[bi * max_channels * t + ci * t + ti] = src_ptr[i_idx]
    elementwise[init_fn, simd_width=1, target="gpu"](
        IndexList[1](b * base_in_channels * t), DeviceContextPtr(ctx),
    )

    var cur_channels = base_in_channels
    for i in range(n_layers):
        var y = ctx.enqueue_create_buffer[DType.float32](b * growth * t)
        # Compute layer(x) where x is the first (cur_channels * T) elements of buf_a.
        # Since buf_a has the data contiguously (B, cur, T) packed at the start...
        # Actually we packed it as (B, max_channels, T) but only cur is populated.
        # Need a clean (B, cur_channels, T) view. Allocate fresh.
        var x_view = ctx.enqueue_create_buffer[DType.float32](b * cur_channels * t)
        # Copy (B, cur, T) from buf_a into x_view (since memory is packed (B, max, T)
        # but our writes have been (B, cur, T), the slice is the first `cur*T` of each batch).
        var bua_ptr = buf_a.unsafe_ptr()
        var xv_ptr = x_view.unsafe_ptr()

        @always_inline
        @parameter
        @__copy_capture(bua_ptr, xv_ptr, cur_channels, t, max_channels)
        def copy_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
            var i_idx = idx[0]
            var bi = i_idx // (cur_channels * t)
            var rem = i_idx - bi * cur_channels * t
            xv_ptr[i_idx] = bua_ptr[bi * max_channels * t + rem]
        elementwise[copy_fn, simd_width=1, target="gpu"](
            IndexList[1](b * cur_channels * t), DeviceContextPtr(ctx),
        )

        camdense_tdnn_layer_forward(
            ctx, block.layers[i], x_view, y, b, cur_channels, t,
            kernel_size, padding, dilation,
        )

        # Concat x_view (B, cur, T) and y (B, growth, T) → next state (B, cur+growth, T)
        # Write into buf_b's (B, max_channels, T) layout — only first (cur+growth)*T per batch is used.
        var nxt = cur_channels + growth
        var bub_ptr = buf_b.unsafe_ptr()
        var y_ptr = y.unsafe_ptr()

        @always_inline
        @parameter
        @__copy_capture(bua_ptr, y_ptr, bub_ptr, cur_channels, growth, nxt, t, max_channels)
        def cat_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
            var i_idx = idx[0]
            var bi = i_idx // (nxt * t)
            var rem = i_idx - bi * nxt * t
            var ci = rem // t
            var ti = rem - ci * t
            if ci < cur_channels:
                bub_ptr[bi * max_channels * t + ci * t + ti] = bua_ptr[bi * max_channels * t + ci * t + ti]
            else:
                var gci = ci - cur_channels
                bub_ptr[bi * max_channels * t + ci * t + ti] = y_ptr[bi * growth * t + gci * t + ti]
        elementwise[cat_fn, simd_width=1, target="gpu"](
            IndexList[1](b * nxt * t), DeviceContextPtr(ctx),
        )

        # Swap buf_a and buf_b for next iteration.
        ctx.enqueue_copy(buf_a, buf_b)
        cur_channels = nxt

    # Copy final state from buf_a into out_buf.
    var ba_ptr = buf_a.unsafe_ptr()
    var op = out_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(ba_ptr, op, max_channels, cur_channels, t)
    def final_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (cur_channels * t)
        var rem = i - bi * cur_channels * t
        op[i] = ba_ptr[bi * max_channels * t + rem]
    elementwise[final_fn, simd_width=1, target="gpu"](
        IndexList[1](b * cur_channels * t), DeviceContextPtr(ctx),
    )


def transit_layer_forward(
    mut ctx: DeviceContext,
    mut tr: TransitLayer,
    mut in_buf: DeviceBuffer[DType.float32],     # (B, C_in, T)
    mut out_buf: DeviceBuffer[DType.float32],    # (B, C_out, T)
    b: Int, c_in: Int, c_out: Int, t: Int,
) raises:
    """Transit: BN + ReLU + 1x1 Conv1d."""
    var h = ctx.enqueue_create_buffer[DType.float32](b * c_in * t)
    batchnorm1d_forward(ctx, tr.nonlinear, in_buf, h, b, c_in, t)
    relu_inplace_bct(ctx, h, b * c_in * t)
    conv1d_forward(ctx, tr.linear, h, out_buf, b, t, t)


def stats_pool_forward(
    mut ctx: DeviceContext,
    mut in_buf: DeviceBuffer[DType.float32],     # (B, C, T)
    mut out_buf: DeviceBuffer[DType.float32],    # (B, 2*C) — cat([mean, std])
    b: Int, c: Int, t: Int,
) raises:
    """Statistics pool: out = cat([mean(x, dim=-1), std(x, dim=-1, unbiased=True)], dim=-1)."""
    var in_ptr = in_buf.unsafe_ptr()
    var o_ptr = out_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(in_ptr, o_ptr, c, t)
    def sp_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (2 * c)
        var ci = i - bi * 2 * c
        # First half: mean. Second half: std.
        if ci < c:
            var acc: Float32 = 0.0
            for ti in range(t):
                acc += in_ptr[bi * c * t + ci * t + ti]
            o_ptr[i] = acc / Float32(t)
        else:
            var ci_real = ci - c
            var m: Float32 = 0.0
            for ti in range(t):
                m += in_ptr[bi * c * t + ci_real * t + ti]
            m = m / Float32(t)
            var var_acc: Float32 = 0.0
            for ti in range(t):
                var d = in_ptr[bi * c * t + ci_real * t + ti] - m
                var_acc += d * d
            # Unbiased: divide by (T - 1) when T > 1.
            # Upstream `torch.std(unbiased=True)` doesn't add eps inside sqrt.
            var denom: Float32 = Float32(t - 1) if t > 1 else 1.0
            var v = var_acc / denom
            o_ptr[i] = sqrt(v)
    elementwise[sp_fn, simd_width=1, target="gpu"](
        IndexList[1](b * 2 * c), DeviceContextPtr(ctx),
    )


def dense_layer_forward(
    mut ctx: DeviceContext,
    mut dense: DenseLayer,
    mut in_buf: DeviceBuffer[DType.float32],     # (B, c_in)
    mut out_buf: DeviceBuffer[DType.float32],    # (B, c_out)
    b: Int, c_in: Int, c_out: Int,
) raises:
    """Dense: 1x1 Conv1d (effectively Linear) + BN (affine=False). For (B, C)
    input, we treat T=1.

    BN here is built with `batchnorm_` (affine=False) — weight=1, bias=0 (filled
    by the loader when affine=False).
    """
    var h = ctx.enqueue_create_buffer[DType.float32](b * c_out * 1)
    conv1d_forward(ctx, dense.linear, in_buf, h, b, 1, 1)
    batchnorm1d_forward(ctx, dense.nonlinear, h, out_buf, b, c_out, 1)


def xvector_forward(
    mut ctx: DeviceContext,
    mut backbone: XVectorBackbone,
    mut in_buf: DeviceBuffer[DType.float32],     # (B, 320, T_in)  — output of FCM
    mut out_buf: DeviceBuffer[DType.float32],    # (B, 192) — speaker embedding
    b: Int, t_in: Int,
) raises:
    """Xvector backbone forward (post-FCM):
      tdnn(320→128, k=5, s=2)
      block1 (12 layers, growth=32, k=3, d=1) → 128 + 384 = 512
      transit1 → 256
      block2 (24 layers, k=3, d=2) → 256 + 768 = 1024
      transit2 → 512
      block3 (16 layers, k=3, d=2) → 512 + 512 = 1024
      transit3 → 512
      out_nonlinear (BN over 512)
      stats_pool → 1024
      dense (1024 → 192)
    """
    # tdnn: PyTorch Conv1d formula: L_out = (L_in + 2*pad - dilation*(K-1) - 1) // stride + 1
    # With K=5, stride=2, pad=2, dilation=1: (T_in + 4 - 4 - 1) // 2 + 1.
    var t_tdnn = (t_in + 2 * 2 - 4 - 1) // 2 + 1
    var h_tdnn = ctx.enqueue_create_buffer[DType.float32](b * 128 * t_tdnn)
    tdnn_first_forward(ctx, backbone.tdnn, in_buf, h_tdnn, b, t_in, t_tdnn)

    # block1: 128 → 512, k=3, d=1, pad=1
    var c_after_b1 = 128 + 12 * 32
    var h_b1 = ctx.enqueue_create_buffer[DType.float32](b * c_after_b1 * t_tdnn)
    camdense_tdnn_block_forward(
        ctx, backbone.block1, h_tdnn, h_b1, b, 128, t_tdnn,
        3, 1, 1,
    )
    var h_t1 = ctx.enqueue_create_buffer[DType.float32](b * 256 * t_tdnn)
    transit_layer_forward(ctx, backbone.transit1, h_b1, h_t1, b, c_after_b1, 256, t_tdnn)

    # block2: 256 → 1024, k=3, d=2, pad=2
    var c_after_b2 = 256 + 24 * 32
    var h_b2 = ctx.enqueue_create_buffer[DType.float32](b * c_after_b2 * t_tdnn)
    camdense_tdnn_block_forward(
        ctx, backbone.block2, h_t1, h_b2, b, 256, t_tdnn,
        3, 2, 2,
    )
    var h_t2 = ctx.enqueue_create_buffer[DType.float32](b * 512 * t_tdnn)
    transit_layer_forward(ctx, backbone.transit2, h_b2, h_t2, b, c_after_b2, 512, t_tdnn)

    # block3: 512 → 1024, k=3, d=2, pad=2
    var c_after_b3 = 512 + 16 * 32
    var h_b3 = ctx.enqueue_create_buffer[DType.float32](b * c_after_b3 * t_tdnn)
    camdense_tdnn_block_forward(
        ctx, backbone.block3, h_t2, h_b3, b, 512, t_tdnn,
        3, 2, 2,
    )
    var h_t3 = ctx.enqueue_create_buffer[DType.float32](b * 512 * t_tdnn)
    transit_layer_forward(ctx, backbone.transit3, h_b3, h_t3, b, c_after_b3, 512, t_tdnn)

    # out_nonlinear: BN(512) + ReLU
    var h_on = ctx.enqueue_create_buffer[DType.float32](b * 512 * t_tdnn)
    batchnorm1d_forward(ctx, backbone.out_nonlinear, h_t3, h_on, b, 512, t_tdnn)
    relu_inplace_bct(ctx, h_on, b * 512 * t_tdnn)

    # stats_pool → (B, 1024)
    var stats = ctx.enqueue_create_buffer[DType.float32](b * 1024)
    stats_pool_forward(ctx, h_on, stats, b, 512, t_tdnn)

    # dense → (B, 192)
    dense_layer_forward(ctx, backbone.dense, stats, out_buf, b, 1024, 192)
