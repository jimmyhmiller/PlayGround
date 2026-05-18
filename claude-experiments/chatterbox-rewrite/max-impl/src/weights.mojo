"""Load module weights from the fixture-format `.bin` files produced by
`scripts/convert_weights.py`.

Each loader takes a base directory and constructs a populated module struct
ready for `*_forward()` calls. All files are read via `fixture.load_fp32`
and uploaded to GPU via `DeviceContext.enqueue_create_buffer`.
"""
from std.gpu.host import DeviceContext, DeviceBuffer

from fixture import load_fp32
from modules import Linear, LayerNorm, RMSNorm, Embedding
from lstm import LSTMLayer
from voice_encoder import VoiceEncoder


def _upload(buf: DeviceBuffer[DType.float32], data: List[Float32], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = data[i]


def upload_fp32(mut ctx: DeviceContext, path: String) raises -> DeviceBuffer[DType.float32]:
    """Read a `.bin` file in our fixture format and upload to GPU."""
    var t = load_fp32(path)
    var n = len(t.data)
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    _upload(buf, t.data, n)
    return buf^


# ============================================================================
# VoiceEncoder loader
# ============================================================================

def load_voice_encoder(mut ctx: DeviceContext, base: String) raises -> VoiceEncoder:
    """Load the 3-layer-LSTM + Linear VoiceEncoder.

    Expects:
      {base}/weight_ih_l{0,1,2}.bin    (4H, IN)
      {base}/weight_hh_l{0,1,2}.bin    (4H, H)
      {base}/bias_ih_l{0,1,2}.bin      (4H,)
      {base}/bias_hh_l{0,1,2}.bin      (4H,)
      {base}/proj_w.bin                (256, 256)
      {base}/proj_b.bin                (256,)
    """
    var H = 256
    var IN_0 = 40    # mel bins
    var IN_HH = H

    var w_ih_0 = upload_fp32(ctx, base + "/weight_ih_l0.bin")
    var w_hh_0 = upload_fp32(ctx, base + "/weight_hh_l0.bin")
    var b_ih_0 = upload_fp32(ctx, base + "/bias_ih_l0.bin")
    var b_hh_0 = upload_fp32(ctx, base + "/bias_hh_l0.bin")
    var l0 = LSTMLayer(w_ih_0^, w_hh_0^, b_ih_0^, b_hh_0^, IN_0, H)

    var w_ih_1 = upload_fp32(ctx, base + "/weight_ih_l1.bin")
    var w_hh_1 = upload_fp32(ctx, base + "/weight_hh_l1.bin")
    var b_ih_1 = upload_fp32(ctx, base + "/bias_ih_l1.bin")
    var b_hh_1 = upload_fp32(ctx, base + "/bias_hh_l1.bin")
    var l1 = LSTMLayer(w_ih_1^, w_hh_1^, b_ih_1^, b_hh_1^, IN_HH, H)

    var w_ih_2 = upload_fp32(ctx, base + "/weight_ih_l2.bin")
    var w_hh_2 = upload_fp32(ctx, base + "/weight_hh_l2.bin")
    var b_ih_2 = upload_fp32(ctx, base + "/bias_ih_l2.bin")
    var b_hh_2 = upload_fp32(ctx, base + "/bias_hh_l2.bin")
    var l2 = LSTMLayer(w_ih_2^, w_hh_2^, b_ih_2^, b_hh_2^, IN_HH, H)

    var proj_w = upload_fp32(ctx, base + "/proj_w.bin")
    var proj_b = upload_fp32(ctx, base + "/proj_b.bin")
    var proj = Linear(proj_w^, proj_b^, H, H, True)

    return VoiceEncoder(l0^, l1^, l2^, proj^, H)
