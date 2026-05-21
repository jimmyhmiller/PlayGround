"""CAMPPlus-specific weight loader.

Extracted from src/weights.mojo, scoped to just what op_campplus needs so the
op compiles without pulling in 21 other unrelated source files.
"""
from std.gpu.host import DeviceContext, DeviceBuffer

from fixture import load_fp32
from conv1d import Conv1d
from campplus import (
    BatchNorm1d, CAMLayer, CAMDenseTDNNLayer, CAMDenseTDNNBlock,
    TransitLayer, TDNN, DenseLayer, XVectorBackbone,
)


def _upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def upload_fp32(mut ctx: DeviceContext, path: String) raises -> DeviceBuffer[DType.float32]:
    var t = load_fp32(path)
    var n = len(t.data)
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    _upload(buf, t.data, n)
    return buf^


def _zero_buf(mut ctx: DeviceContext, n: Int) raises -> DeviceBuffer[DType.float32]:
    var b = ctx.enqueue_create_buffer[DType.float32](n)
    b.enqueue_fill(0.0)
    return b^


def _load_batchnorm(
    mut ctx: DeviceContext, base: String, channels: Int,
    affine: Bool = True,
    nested: Bool = True,
) raises -> BatchNorm1d:
    var sub: String
    if nested:
        sub = "/batchnorm/"
    else:
        sub = "/"

    var w: DeviceBuffer[DType.float32]
    var b: DeviceBuffer[DType.float32]
    if affine:
        w = upload_fp32(ctx, base + sub + "weight.bin")
        b = upload_fp32(ctx, base + sub + "bias.bin")
    else:
        w = ctx.enqueue_create_buffer[DType.float32](channels)
        w.enqueue_fill(1.0)
        b = _zero_buf(ctx, channels)
    var rm = upload_fp32(ctx, base + sub + "running_mean.bin")
    var rv = upload_fp32(ctx, base + sub + "running_var.bin")
    return BatchNorm1d(w^, b^, rm^, rv^, channels, Float32(1.0e-5))


def _load_conv1d_no_bias(
    mut ctx: DeviceContext, base: String,
    c_in: Int, c_out: Int, k: Int, stride: Int, pad: Int, dilation: Int,
    groups: Int = 1,
) raises -> Conv1d:
    var w = upload_fp32(ctx, base + "/weight.bin")
    var zb = _zero_buf(ctx, c_out)
    return Conv1d(w^, zb^, c_in, c_out, k, stride, pad, dilation, groups, False)


def _load_cam_layer(
    mut ctx: DeviceContext, base: String, dilation: Int = 1,
) raises -> CAMLayer:
    var linear_local = _load_conv1d_no_bias(
        ctx, base + "/linear_local", 128, 32, 3, 1, 0, dilation, 1,
    )
    var w1 = upload_fp32(ctx, base + "/linear1/weight.bin")
    var b1 = upload_fp32(ctx, base + "/linear1/bias.bin")
    var linear1 = Conv1d(w1^, b1^, 128, 64, 1, 1, 0, 1, 1, True)
    var w2 = upload_fp32(ctx, base + "/linear2/weight.bin")
    var b2 = upload_fp32(ctx, base + "/linear2/bias.bin")
    var linear2 = Conv1d(w2^, b2^, 64, 32, 1, 1, 0, 1, 1, True)
    return CAMLayer(linear_local^, linear1^, linear2^)


def _load_camdense_tdnn_layer(
    mut ctx: DeviceContext, base: String, in_channels: Int, dilation: Int = 1,
) raises -> CAMDenseTDNNLayer:
    var nl1 = _load_batchnorm(ctx, base + "/nonlinear1", in_channels)
    var linear1 = _load_conv1d_no_bias(
        ctx, base + "/linear1", in_channels, 128, 1, 1, 0, 1, 1,
    )
    var nl2 = _load_batchnorm(ctx, base + "/nonlinear2", 128)
    var cam = _load_cam_layer(ctx, base + "/cam_layer", dilation)
    return CAMDenseTDNNLayer(nl1^, linear1^, nl2^, cam^)


def _load_camdense_tdnn_block(
    mut ctx: DeviceContext, base: String, n_layers: Int,
    base_in_channels: Int, growth: Int, dilation: Int = 1,
) raises -> CAMDenseTDNNBlock:
    var layers = List[CAMDenseTDNNLayer]()
    for i in range(n_layers):
        var lyr_base = base + "/tdnnd" + String(i + 1)
        var in_ch = base_in_channels + i * growth
        var lyr = _load_camdense_tdnn_layer(ctx, lyr_base, in_ch, dilation)
        layers.append(lyr^)
    return CAMDenseTDNNBlock(layers^)


def _load_transit_layer(
    mut ctx: DeviceContext, base: String, c_in: Int, c_out: Int,
) raises -> TransitLayer:
    var nonlin = _load_batchnorm(ctx, base + "/nonlinear", c_in)
    var linear = _load_conv1d_no_bias(ctx, base + "/linear", c_in, c_out, 1, 1, 0, 1, 1)
    return TransitLayer(nonlin^, linear^)


def _load_tdnn_first(mut ctx: DeviceContext, base: String) raises -> TDNN:
    var linear = _load_conv1d_no_bias(ctx, base + "/linear", 320, 128, 5, 2, 0, 1, 1)
    var nonlin = _load_batchnorm(ctx, base + "/nonlinear", 128)
    return TDNN(linear^, nonlin^)


def _load_dense_layer(
    mut ctx: DeviceContext, base: String, c_in: Int, c_out: Int,
) raises -> DenseLayer:
    var linear = _load_conv1d_no_bias(ctx, base + "/linear", c_in, c_out, 1, 1, 0, 1, 1)
    var nonlin = _load_batchnorm(ctx, base + "/nonlinear", c_out, affine=False)
    return DenseLayer(linear^, nonlin^)


def load_xvector_backbone(mut ctx: DeviceContext, base: String) raises -> XVectorBackbone:
    """Load the xvector backbone (the post-FCM portion) from
    weights/s3gen/speaker_encoder.

    Returns just the xvector backbone — Phase B op exposes only xvector_forward.
    """
    var xv = base + "/xvector"

    var tdnn = _load_tdnn_first(ctx, xv + "/tdnn")

    var block1 = _load_camdense_tdnn_block(ctx, xv + "/block1", 12, 128, 32, 1)
    var transit1 = _load_transit_layer(ctx, xv + "/transit1", 128 + 12 * 32, 256)
    var block2 = _load_camdense_tdnn_block(ctx, xv + "/block2", 24, 256, 32, 2)
    var transit2 = _load_transit_layer(ctx, xv + "/transit2", 256 + 24 * 32, 512)
    var block3 = _load_camdense_tdnn_block(ctx, xv + "/block3", 16, 512, 32, 2)
    var transit3 = _load_transit_layer(ctx, xv + "/transit3", 512 + 16 * 32, 512)

    var out_nonlin = _load_batchnorm(ctx, xv + "/out_nonlinear", 512)
    var dense = _load_dense_layer(ctx, xv + "/dense", 1024, 192)

    return XVectorBackbone(
        tdnn^, block1^, transit1^, block2^, transit2^, block3^, transit3^,
        out_nonlin^, dense^,
    )
