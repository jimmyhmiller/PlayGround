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
from transformer_blocks import LlamaMLP, MLP
from t3_block import T3Block
from t3 import T3
from conv1d import Conv1d
from s3tokenizer_block import FSMNAttention, S3TokenizerBlock
from s3tokenizer import S3Tokenizer
from conformer import RelPosMHA, TransformerEncoderLayer


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


# ============================================================================
# S3Tokenizer loader — 2 conv1d + 6 FSMN blocks + FSQ project_down
# ============================================================================

def _zero_buf(mut ctx: DeviceContext, n: Int) raises -> DeviceBuffer[DType.float32]:
    var b = ctx.enqueue_create_buffer[DType.float32](n)
    b.enqueue_fill(0.0)
    return b^


def load_s3tokenizer_block(
    mut ctx: DeviceContext, block_base: String,
    hidden: Int, intermediate: Int, n_heads: Int, head_dim: Int,
    fsmn_kernel: Int,
) raises -> S3TokenizerBlock:
    """Load one s3tokenizer ResidualAttentionBlock."""
    var attn_ln_w = upload_fp32(ctx, block_base + "/attn_ln_w.bin")
    var attn_ln_b = upload_fp32(ctx, block_base + "/attn_ln_b.bin")
    var mlp_ln_w = upload_fp32(ctx, block_base + "/mlp_ln_w.bin")
    var mlp_ln_b = upload_fp32(ctx, block_base + "/mlp_ln_b.bin")

    var qw = upload_fp32(ctx, block_base + "/qw.bin")
    var qb = upload_fp32(ctx, block_base + "/qb.bin")
    var kw = upload_fp32(ctx, block_base + "/kw.bin")
    var vw = upload_fp32(ctx, block_base + "/vw.bin")
    var vb = upload_fp32(ctx, block_base + "/vb.bin")
    var ow = upload_fp32(ctx, block_base + "/ow.bin")
    var ob = upload_fp32(ctx, block_base + "/ob.bin")
    var fsmn_w = upload_fp32(ctx, block_base + "/fsmn_w.bin")

    var fc1_w = upload_fp32(ctx, block_base + "/mlp_fc1_w.bin")
    var fc1_b = upload_fp32(ctx, block_base + "/mlp_fc1_b.bin")
    var fc2_w = upload_fp32(ctx, block_base + "/mlp_fc2_w.bin")
    var fc2_b = upload_fp32(ctx, block_base + "/mlp_fc2_b.bin")

    var attn_ln = LayerNorm(attn_ln_w^, attn_ln_b^, hidden, Float32(1.0e-5))
    var mlp_ln = LayerNorm(mlp_ln_w^, mlp_ln_b^, hidden, Float32(1.0e-5))

    var to_q = Linear(qw^, qb^, hidden, hidden, True)
    var zero_k = _zero_buf(ctx, hidden)
    var to_k = Linear(kw^, zero_k^, hidden, hidden, False)
    var to_v = Linear(vw^, vb^, hidden, hidden, True)
    var to_out = Linear(ow^, ob^, hidden, hidden, True)

    # FSMN block: depthwise conv1d, groups = hidden, kernel = fsmn_kernel.
    # Upstream weight shape is (hidden, 1, kernel) which matches depthwise layout.
    var zero_fsmn_b = _zero_buf(ctx, hidden)
    var pad = (fsmn_kernel - 1) // 2
    var fsmn_conv = Conv1d(
        fsmn_w^, zero_fsmn_b^,
        1, hidden, fsmn_kernel, 1, pad, 1, hidden, False,
    )

    var attn = FSMNAttention(to_q^, to_k^, to_v^, to_out^, fsmn_conv^,
                              n_heads, head_dim)

    var fc1 = Linear(fc1_w^, fc1_b^, hidden, intermediate, True)
    var fc2 = Linear(fc2_w^, fc2_b^, intermediate, hidden, True)
    var mlp = MLP(fc1^, fc2^, hidden, intermediate)

    return S3TokenizerBlock(attn_ln^, mlp_ln^, attn^, mlp^)


def load_s3tokenizer(mut ctx: DeviceContext, base: String) raises -> S3Tokenizer:
    """Load S3TokenizerV2 (128-mel → 1280-state → 6 blocks → FSQ).

    Expects layout produced by `convert_weights.py --s3t`:
      {base}/{conv1_w,conv1_b,conv2_w,conv2_b}.bin
      {base}/block{0..5}/{attn_ln_*,mlp_ln_*,qw,qb,kw,vw,vb,ow,ob,fsmn_w,mlp_*}.bin
      {base}/{project_down_w,project_down_b}.bin
    """
    var N_MELS = 128
    var N_STATE = 1280
    var N_HEADS = 1            # FSMN multi-head uses single head in this checkpoint
    var HEAD_DIM = 1280
    var INTERMEDIATE = 5120
    var N_LAYERS = 6
    var FSMN_K = 31

    var conv1_w = upload_fp32(ctx, base + "/conv1_w.bin")
    var conv1_b = upload_fp32(ctx, base + "/conv1_b.bin")
    var conv2_w = upload_fp32(ctx, base + "/conv2_w.bin")
    var conv2_b = upload_fp32(ctx, base + "/conv2_b.bin")

    # Upstream Whisper-style: conv1 in=mel out=state K=3 stride=1 pad=1 ;
    # conv2 in=state out=state K=3 stride=2 pad=1.
    var conv1 = Conv1d(conv1_w^, conv1_b^, N_MELS, N_STATE, 3, 1, 1, 1, 1, True)
    var conv2 = Conv1d(conv2_w^, conv2_b^, N_STATE, N_STATE, 3, 2, 1, 1, 1, True)

    var blocks = List[S3TokenizerBlock]()
    for L in range(N_LAYERS):
        var block_base = base + "/block" + String(L)
        var blk = load_s3tokenizer_block(
            ctx, block_base, N_STATE, INTERMEDIATE, N_HEADS, HEAD_DIM, FSMN_K,
        )
        blocks.append(blk^)

    var proj_w = upload_fp32(ctx, base + "/project_down_w.bin")
    var proj_b = upload_fp32(ctx, base + "/project_down_b.bin")
    var project_down = Linear(proj_w^, proj_b^, N_STATE, 8, True)

    return S3Tokenizer(
        conv1^, conv2^, blocks^, project_down^,
        N_MELS, N_STATE, N_HEADS, HEAD_DIM, N_LAYERS,
    )


# ============================================================================
# S3Gen flow encoder layer loader — real upstream TransformerEncoderLayer
# ============================================================================

def load_transformer_encoder_layer(
    mut ctx: DeviceContext, layer_base: String,
    d_model: Int, intermediate: Int, n_heads: Int, head_dim: Int,
) raises -> TransformerEncoderLayer:
    """Load one upstream s3gen flow-encoder layer from disk.

    Expects {layer_base}/{norm_mha,norm_ff,self_attn,feed_forward}/... layout
    produced by `convert_weights.py --s3gen` (pass-through nested keys).
    """
    var nm_w = upload_fp32(ctx, layer_base + "/norm_mha/weight.bin")
    var nm_b = upload_fp32(ctx, layer_base + "/norm_mha/bias.bin")
    var nf_w = upload_fp32(ctx, layer_base + "/norm_ff/weight.bin")
    var nf_b = upload_fp32(ctx, layer_base + "/norm_ff/bias.bin")

    var qw = upload_fp32(ctx, layer_base + "/self_attn/linear_q/weight.bin")
    var qb = upload_fp32(ctx, layer_base + "/self_attn/linear_q/bias.bin")
    var kw = upload_fp32(ctx, layer_base + "/self_attn/linear_k/weight.bin")
    var kb = upload_fp32(ctx, layer_base + "/self_attn/linear_k/bias.bin")
    var vw = upload_fp32(ctx, layer_base + "/self_attn/linear_v/weight.bin")
    var vb = upload_fp32(ctx, layer_base + "/self_attn/linear_v/bias.bin")
    var ow = upload_fp32(ctx, layer_base + "/self_attn/linear_out/weight.bin")
    var ob = upload_fp32(ctx, layer_base + "/self_attn/linear_out/bias.bin")
    var pw = upload_fp32(ctx, layer_base + "/self_attn/linear_pos/weight.bin")
    var pos_u = upload_fp32(ctx, layer_base + "/self_attn/pos_bias_u.bin")
    var pos_v = upload_fp32(ctx, layer_base + "/self_attn/pos_bias_v.bin")

    var w1_w = upload_fp32(ctx, layer_base + "/feed_forward/w_1/weight.bin")
    var w1_b = upload_fp32(ctx, layer_base + "/feed_forward/w_1/bias.bin")
    var w2_w = upload_fp32(ctx, layer_base + "/feed_forward/w_2/weight.bin")
    var w2_b = upload_fp32(ctx, layer_base + "/feed_forward/w_2/bias.bin")

    var norm_mha = LayerNorm(nm_w^, nm_b^, d_model, Float32(1.0e-5))
    var norm_ff = LayerNorm(nf_w^, nf_b^, d_model, Float32(1.0e-5))

    var to_q = Linear(qw^, qb^, d_model, d_model, True)
    var to_k = Linear(kw^, kb^, d_model, d_model, True)
    var to_v = Linear(vw^, vb^, d_model, d_model, True)
    var to_out = Linear(ow^, ob^, d_model, d_model, True)
    var zero_pos_b = _zero_buf(ctx, d_model)
    var linear_pos = Linear(pw^, zero_pos_b^, d_model, d_model, False)

    var self_attn = RelPosMHA(to_q^, to_k^, to_v^, to_out^, linear_pos^,
                                pos_u^, pos_v^, n_heads, head_dim)

    var w1 = Linear(w1_w^, w1_b^, d_model, intermediate, True)
    var w2 = Linear(w2_w^, w2_b^, intermediate, d_model, True)

    return TransformerEncoderLayer(
        norm_mha^, norm_ff^, self_attn^, w1^, w2^, d_model, intermediate,
    )


def load_s3gen_flow_encoder_layers(
    mut ctx: DeviceContext, encoders_base: String, n_layers: Int,
    d_model: Int, intermediate: Int, n_heads: Int, head_dim: Int,
) raises -> List[TransformerEncoderLayer]:
    """Load a stack of N encoder layers from `encoders_base/{0..N-1}/`.

    Use for both `encoder/encoders/` (6 layers) and `encoder/up_encoders/` (4
    layers) — same per-layer structure.
    """
    var layers = List[TransformerEncoderLayer]()
    for L in range(n_layers):
        var layer_base = encoders_base + "/" + String(L)
        var lyr = load_transformer_encoder_layer(
            ctx, layer_base, d_model, intermediate, n_heads, head_dim,
        )
        layers.append(lyr^)
    return layers^
