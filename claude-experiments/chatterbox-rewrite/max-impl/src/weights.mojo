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
from transformer_blocks import LlamaMLP
from t3_block import T3Block
from t3 import T3


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


# ============================================================================
# T3 loader — 30-layer Llama backbone
# ============================================================================

def load_t3_block(mut ctx: DeviceContext, layer_base: String,
                   hidden: Int, intermediate: Int,
                   n_heads: Int, head_dim: Int) raises -> T3Block:
    """Load one T3 transformer block from {layer_base}/{in_norm,post_norm,qw,kw,vw,ow,gate_w,up_w,down_w}.bin.

    Llama biases are all zero. We pre-transpose all Linear weights to (OUT, IN)
    since the upstream HF Llama stores them as (OUT, IN) natively (NOT pre-transposed
    like the mojo-t3 fixtures did). Check by inspecting shapes.
    """
    var in_norm_w = upload_fp32(ctx, layer_base + "/in_norm.bin")
    var post_norm_w = upload_fp32(ctx, layer_base + "/post_norm.bin")
    var qw = upload_fp32(ctx, layer_base + "/qw.bin")
    var kw = upload_fp32(ctx, layer_base + "/kw.bin")
    var vw = upload_fp32(ctx, layer_base + "/vw.bin")
    var ow = upload_fp32(ctx, layer_base + "/ow.bin")
    var gate_w = upload_fp32(ctx, layer_base + "/gate_w.bin")
    var up_w = upload_fp32(ctx, layer_base + "/up_w.bin")
    var down_w = upload_fp32(ctx, layer_base + "/down_w.bin")

    var zero_d = ctx.enqueue_create_buffer[DType.float32](hidden)
    zero_d.enqueue_fill(0.0)
    var zero_inter = ctx.enqueue_create_buffer[DType.float32](intermediate)
    zero_inter.enqueue_fill(0.0)

    var in_norm = RMSNorm(in_norm_w^, hidden, Float32(1.0e-5))
    var post_norm = RMSNorm(post_norm_w^, hidden, Float32(1.0e-5))
    var to_q = Linear(qw^, zero_d, hidden, hidden, False)
    var to_k = Linear(kw^, zero_d.copy(), hidden, hidden, False)
    var to_v = Linear(vw^, zero_d.copy(), hidden, hidden, False)
    var to_out = Linear(ow^, zero_d.copy(), hidden, hidden, False)
    var gate = Linear(gate_w^, zero_inter, hidden, intermediate, False)
    var up = Linear(up_w^, zero_inter.copy(), hidden, intermediate, False)
    var down = Linear(down_w^, zero_d.copy(), intermediate, hidden, False)
    var mlp = LlamaMLP(gate^, up^, down^, hidden, intermediate)

    return T3Block(in_norm^, post_norm^, to_q^, to_k^, to_v^, to_out^, mlp^,
                    n_heads, head_dim)


def load_t3(mut ctx: DeviceContext, base: String) raises -> T3:
    """Load full T3 Llama-30L from {base}/layer{0..29}/ + {base}/final_norm_w.bin
    + {base}/speech_emb_w.bin + {base}/speech_head_w.bin.
    """
    var N_LAYERS = 30
    var HIDDEN = 1024
    var INTERMEDIATE = 4096
    var N_HEADS = 16
    var HEAD_DIM = 64
    var V_SPEECH = 8194

    var blocks = List[T3Block]()
    for L in range(N_LAYERS):
        var layer_base = base + "/layer" + String(L)
        var b = load_t3_block(ctx, layer_base, HIDDEN, INTERMEDIATE, N_HEADS, HEAD_DIM)
        blocks.append(b^)

    var final_norm_w = upload_fp32(ctx, base + "/final_norm_w.bin")
    var final_norm = RMSNorm(final_norm_w^, HIDDEN, Float32(1.0e-5))

    var speech_emb_w = upload_fp32(ctx, base + "/speech_emb_w.bin")
    var speech_emb = Embedding(speech_emb_w^, V_SPEECH, HIDDEN)

    var speech_head_w = upload_fp32(ctx, base + "/speech_head_w.bin")
    var zero_d = ctx.enqueue_create_buffer[DType.float32](V_SPEECH)
    zero_d.enqueue_fill(0.0)
    var speech_head = Linear(speech_head_w^, zero_d^, HIDDEN, V_SPEECH, False)

    return T3(blocks^, final_norm^, speech_emb^, speech_head^,
                N_LAYERS, N_HEADS, HEAD_DIM, HIDDEN, V_SPEECH)
