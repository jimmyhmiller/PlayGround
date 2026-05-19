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
from conv2d import Conv2d, BatchNorm2d
from fcm import FCM, BasicResBlock2d
from s3tokenizer_block import FSMNAttention, S3TokenizerBlock
from s3tokenizer import S3Tokenizer
from conformer import RelPosMHA, TransformerEncoderLayer
from hift_generator import (
    SnakeActivation, HiFTResBlock, F0Predictor, MSource, HiFTGenerator,
)
from campplus import (
    BatchNorm1d, CAMLayer, CAMDenseTDNNLayer, CAMDenseTDNNBlock,
    TransitLayer, TDNN, DenseLayer, XVectorBackbone,
    ResNetBasicBlock, ResNetHead, CAMPPlus,
)
from cfm_estimator_new import (
    GroupNorm1d, Block1D, Resnet1D,
    CFMAttention, CFMFeedForward, BasicTransformerBlock,
    CFMDownStage, CFMMidStage, CFMUpStage, CFMEstimatorReal,
)
from perceiver import Perceiver, PerceiverBlock
from cond_enc import T3CondEnc
from upsample_encoder import (
    EmbedOut, PreLookaheadLayer, UpLayerConv, UpsampleConformerEncoderReal,
)


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

    var V_TEXT = 704
    var MAX_TEXT_POS = 2050
    var MAX_SPEECH_POS = 4100
    var text_emb_w = upload_fp32(ctx, base + "/text_emb_w.bin")
    var text_emb = Embedding(text_emb_w^, V_TEXT, HIDDEN)
    var text_pos_w = upload_fp32(ctx, base + "/text_pos_w.bin")
    var text_pos_emb = Embedding(text_pos_w^, MAX_TEXT_POS, HIDDEN)
    var speech_pos_w = upload_fp32(ctx, base + "/speech_pos_w.bin")
    var speech_pos_emb = Embedding(speech_pos_w^, MAX_SPEECH_POS, HIDDEN)

    return T3(blocks^, final_norm^, speech_emb^, speech_head^, text_emb^, text_pos_emb^,
                speech_pos_emb^, N_LAYERS, N_HEADS, HEAD_DIM, HIDDEN, V_SPEECH, V_TEXT)


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


# ============================================================================
# HiFTGenerator (NSF-HiFiGAN) loader
# ============================================================================

def _load_conv1d_wn(
    mut ctx: DeviceContext, base: String,
    c_in: Int, c_out: Int, k: Int, stride: Int, pad: Int, dilation: Int,
    groups: Int = 1,
) raises -> Conv1d:
    """Load a weight-norm-collapsed Conv1d: expects {base}/weight.bin + {base}/bias.bin."""
    var w = upload_fp32(ctx, base + "/weight.bin")
    var b = upload_fp32(ctx, base + "/bias.bin")
    return Conv1d(w^, b^, c_in, c_out, k, stride, pad, dilation, groups, True)


def _load_conv1d_plain(
    mut ctx: DeviceContext, base: String,
    c_in: Int, c_out: Int, k: Int, stride: Int, pad: Int, dilation: Int,
    groups: Int = 1,
) raises -> Conv1d:
    """Same as _load_conv1d_wn — both layouts produce weight.bin+bias.bin after
    weight_norm collapse."""
    return _load_conv1d_wn(ctx, base, c_in, c_out, k, stride, pad, dilation, groups)


def _load_hift_resblock(
    mut ctx: DeviceContext, base: String, channels: Int,
    kernel: Int, dilations: List[Int],
) raises -> HiFTResBlock:
    """Load one MRF resblock: convs1 (dilated, kernel `kernel`) + convs2 (1x1)
    + snake activations.
    """
    var convs1 = List[Conv1d]()
    var convs2 = List[Conv1d]()
    var acts1 = List[SnakeActivation]()
    var acts2 = List[SnakeActivation]()

    for j in range(3):
        var dil = dilations[j]
        var pad1 = ((kernel - 1) * dil) // 2
        var c1 = _load_conv1d_wn(
            ctx, base + "/convs1/" + String(j),
            channels, channels, kernel, 1, pad1, dil, 1,
        )
        convs1.append(c1^)

        var pad2 = (kernel - 1) // 2
        var c2 = _load_conv1d_wn(
            ctx, base + "/convs2/" + String(j),
            channels, channels, kernel, 1, pad2, 1, 1,
        )
        convs2.append(c2^)

        var a1 = upload_fp32(ctx, base + "/activations1/" + String(j) + "/alpha.bin")
        acts1.append(SnakeActivation(a1^, channels))
        var a2 = upload_fp32(ctx, base + "/activations2/" + String(j) + "/alpha.bin")
        acts2.append(SnakeActivation(a2^, channels))

    return HiFTResBlock(convs1^, convs2^, acts1^, acts2^, channels)


def load_hift_generator(mut ctx: DeviceContext, base: String) raises -> HiFTGenerator:
    """Load full NSF-HiFiGAN from weights/s3gen/mel2wav/.

    Architectural constants are hardcoded to match upstream Chatterbox config:
      base_channels=512, upsample_rates=[8,8,4] (3 stages),
      upsample_kernel_sizes=[16,16,8],
      resblock_kernel_sizes=[3,7,11], dilations=[[1,3,5],[1,3,5],[1,3,5]],
      n_fft=16 (so conv_post outputs n_fft+2=18 channels).
    """
    comptime BASE = 512
    comptime N_FFT = 16
    comptime N_OUT = 18    # n_fft + 2

    # conv_pre: 80 → 512, kernel 7, pad 3.
    var conv_pre = _load_conv1d_wn(ctx, base + "/conv_pre", 80, BASE, 7, 1, 3, 1, 1)

    # ups: 3 transposed convs. Channel widths halve each stage.
    var ups = List[Conv1d]()
    var ups_rates = [8, 5, 3]
    var ups_kernels = [16, 11, 7]
    for i in range(3):
        var c_in = BASE // (1 << i)
        var c_out = BASE // (1 << (i + 1))
        var u = ups_rates[i]
        var k = ups_kernels[i]
        var pad = (k - u) // 2
        # NOTE: ConvTranspose1d stored as Conv1d here — caller wires transpose forward.
        var up = _load_conv1d_wn(ctx, base + "/ups/" + String(i), c_in, c_out, k, u, pad, 1, 1)
        ups.append(up^)

    # resblocks: 9 (3 per ups stage). MRF kernel sizes [3, 7, 11], dilations all [1,3,5].
    var resblock_kernels = [3, 7, 11]
    var dilations: List[Int] = [1, 3, 5]
    var resblocks = List[HiFTResBlock]()
    for stage in range(3):
        var ch = BASE // (1 << (stage + 1))
        for j in range(3):
            var rb_idx = stage * 3 + j
            var rb = _load_hift_resblock(
                ctx, base + "/resblocks/" + String(rb_idx),
                ch, resblock_kernels[j], dilations.copy(),
            )
            resblocks.append(rb^)

    # source_downs: 3 stages, each 18 → channel-at-stage. Kernel varies.
    # Empirically from weight shapes:
    #   source_downs.0: (256, 18, 30) stride large
    #   source_downs.1: (128, 18, 6)
    #   source_downs.2: (64, 18, 1)
    var source_downs = List[Conv1d]()
    var sd_kernels = [30, 6, 1]
    var sd_strides = [15, 3, 1]
    var sd_pads = [7, 1, 0]
    for i in range(3):
        var ch = BASE // (1 << (i + 1))
        var sd = _load_conv1d_wn(
            ctx, base + "/source_downs/" + String(i),
            N_OUT, ch, sd_kernels[i], sd_strides[i], sd_pads[i], 1, 1,
        )
        source_downs.append(sd^)

    # source_resblocks: 3 MRF resblocks; kernels [7, 11, 11] per stage (varies)
    var source_resblocks = List[HiFTResBlock]()
    var src_rb_kernels = [7, 7, 11]
    for i in range(3):
        var ch = BASE // (1 << (i + 1))
        var srb = _load_hift_resblock(
            ctx, base + "/source_resblocks/" + String(i),
            ch, src_rb_kernels[i], dilations.copy(),
        )
        source_resblocks.append(srb^)

    # conv_post: 64 → 18, kernel 7, pad 3.
    var conv_post = _load_conv1d_wn(ctx, base + "/conv_post", 64, N_OUT, 7, 1, 3, 1, 1)

    # m_source.l_linear: (1, 9) + bias (1,)
    var ms_w = upload_fp32(ctx, base + "/m_source/l_linear/weight.bin")
    var ms_b = upload_fp32(ctx, base + "/m_source/l_linear/bias.bin")
    var m_source = MSource(Linear(ms_w^, ms_b^, 9, 1, True))

    # f0_predictor: 5 Conv1d (k=3, pad=1) + ELU, then Linear classifier → F0.
    # Upstream `ConvRNNF0Predictor` takes mel (B, 80, T) and outputs F0 (B, T).
    # Layers at condnet indices [0, 2, 4, 6, 8] (odd indices are ELU activations).
    # First layer: 80 → 512. Subsequent layers: 512 → 512.
    var condnet = List[Conv1d]()
    var condnet_indices = [0, 2, 4, 6, 8]
    for i in range(5):
        var idx = condnet_indices[i]
        var c_in = 80 if i == 0 else 512
        var c = _load_conv1d_wn(
            ctx, base + "/f0_predictor/condnet/" + String(idx),
            c_in, 512, 3, 1, 1, 1, 1,
        )
        condnet.append(c^)
    var cls_w = upload_fp32(ctx, base + "/f0_predictor/classifier/weight.bin")
    var cls_b = upload_fp32(ctx, base + "/f0_predictor/classifier/bias.bin")
    var classifier = Linear(cls_w^, cls_b^, 512, 1, True)
    var f0_predictor = F0Predictor(condnet^, classifier^)

    return HiFTGenerator(
        conv_pre^, ups^, resblocks^, source_downs^, source_resblocks^,
        conv_post^, m_source^, f0_predictor^,
        N_FFT, 4, Float32(0.1),
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


# ============================================================================
# CAMPPlus speaker encoder loader
# ============================================================================

def _load_batchnorm(
    mut ctx: DeviceContext, base: String, channels: Int,
    affine: Bool = True,
    nested: Bool = True,
) raises -> BatchNorm1d:
    """Load PyTorch BatchNorm1d: weight/bias/running_mean/running_var.

    Two on-disk layouts coexist:
      nested=True:  {base}/batchnorm/{weight,bias,running_mean,running_var}.bin
                    (used by xvector wrappers that nest BN inside a
                    `nonlinear.batchnorm` submodule)
      nested=False: {base}/{weight,bias,running_mean,running_var}.bin
                    (used by the head's bn1/bn2 directly)

    `affine=False` means upstream BN had no learnable gamma/beta — use 1/0
    buffers so the downstream multiply/add can stay unconditional.
    """
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
    """Conv1d that may not have a bias file; uses a zero-buffer when absent.

    Used by `linear*` modules in CAMPPlus that are stored bias-less.
    """
    var w = upload_fp32(ctx, base + "/weight.bin")
    var zb = _zero_buf(ctx, c_out)
    return Conv1d(w^, zb^, c_in, c_out, k, stride, pad, dilation, groups, False)


def _load_cam_layer(
    mut ctx: DeviceContext, base: String, dilation: Int = 1,
) raises -> CAMLayer:
    """linear_local: Conv1d(128, 32, K=3, dilation=`dilation`) with padding
    handled externally by cam_layer_forward (pre-pads input, then conv with
    internal pad=0). dilation is 1 in block1, 2 in block2/block3.
    """
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
    """Each tdnnd block: nonlinear1 (BN over `in_channels`) + linear1 (1×1
    Conv in→128) + nonlinear2 (BN 128) + cam_layer (CAM with `dilation`).
    """
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
    """Block of N dense TDNN layers. Each layer's input grows by `growth`
    channels (DenseNet-style concatenation): layer i gets
    `base_in_channels + i * growth` input channels. `dilation` is the
    CAMLayer's linear_local dilation (1 in block1, 2 in block2/block3).
    """
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
    """First TDNN: Conv1d (in=320, out=128, k=5, stride=2) + BN(128). Bias-less
    linear. Forward pre-pads input by 2 each side so internal pad=0.
    """
    var linear = _load_conv1d_no_bias(ctx, base + "/linear", 320, 128, 5, 2, 0, 1, 1)
    var nonlin = _load_batchnorm(ctx, base + "/nonlinear", 128)
    return TDNN(linear^, nonlin^)


def _load_dense_layer(
    mut ctx: DeviceContext, base: String, c_in: Int, c_out: Int,
) raises -> DenseLayer:
    """Final dense layer's BN is created with `affine=False` upstream — only
    running_mean / running_var on disk, no weight/bias.
    """
    var linear = _load_conv1d_no_bias(ctx, base + "/linear", c_in, c_out, 1, 1, 0, 1, 1)
    var nonlin = _load_batchnorm(ctx, base + "/nonlinear", c_out, affine=False)
    return DenseLayer(linear^, nonlin^)


def _load_resnet_basic_block(
    mut ctx: DeviceContext, base: String, channels: Int,
) raises -> ResNetBasicBlock:
    var bn1 = _load_batchnorm(ctx, base + "/bn1", channels, nested=False)
    var c1_w = upload_fp32(ctx, base + "/conv1/weight.bin")
    var conv1 = Conv1d(c1_w^, _zero_buf(ctx, channels)^,
                        channels, channels, 3, 1, 1, 1, 1, False)
    var bn2 = _load_batchnorm(ctx, base + "/bn2", channels, nested=False)
    var c2_w = upload_fp32(ctx, base + "/conv2/weight.bin")
    var conv2 = Conv1d(c2_w^, _zero_buf(ctx, channels)^,
                        channels, channels, 3, 1, 1, 1, 1, False)
    return ResNetBasicBlock(bn1^, conv1^, bn2^, conv2^)


def _load_resnet_layer(
    mut ctx: DeviceContext, base: String, channels: Int,
) raises -> List[ResNetBasicBlock]:
    var blocks = List[ResNetBasicBlock]()
    for i in range(2):
        var b = _load_resnet_basic_block(ctx, base + "/" + String(i), channels)
        blocks.append(b^)
    return blocks^


def load_campplus(mut ctx: DeviceContext, base: String) raises -> CAMPPlus:
    """Load full CAMPPlus from weights/s3gen/speaker_encoder/.

    Block layer counts derived from upstream weight tree:
      block1: 12 tdnnd, block2: 24 tdnnd, block3: 16 tdnnd.
      growth=32 channels per layer (from CAM cam_layer.linear2 out=32).
      Transit expansions: 128+12*32=512 → ... etc. — but matches upstream
      weights when block channels are correctly inferred from BN shapes.

    NOTE: Exact channel widths are inferred at load time from BN tensor sizes;
    if these constants drift in future Chatterbox releases, the loader will
    raise on missing files rather than producing wrong-shaped buffers.
    """
    var xv = base + "/xvector"

    # First tdnn (input mel=80 frames concatenated, in_channels=320 from
    # frame-context window of 5).
    var tdnn = _load_tdnn_first(ctx, xv + "/tdnn")

    # CAMDense blocks with growth=32. Channel widths from upstream:
    #   block1 in=128, 12 layers → 128 + 12*32 = 512 → transit1 (512 → 256)
    #   block2 in=256, 24 layers → 256 + 24*32 = 1024 → transit2 (1024 → 512)
    #   block3 in=512, 16 layers → 512 + 16*32 = 1024 → transit3 (1024 → 512)
    # Upstream CAMPPlus block dilations: (1, 2, 2) per (block1, block2, block3).
    var block1 = _load_camdense_tdnn_block(ctx, xv + "/block1", 12, 128, 32, 1)
    var transit1 = _load_transit_layer(ctx, xv + "/transit1", 128 + 12 * 32, 256)
    var block2 = _load_camdense_tdnn_block(ctx, xv + "/block2", 24, 256, 32, 2)
    var transit2 = _load_transit_layer(ctx, xv + "/transit2", 256 + 24 * 32, 512)
    var block3 = _load_camdense_tdnn_block(ctx, xv + "/block3", 16, 512, 32, 2)
    var transit3 = _load_transit_layer(ctx, xv + "/transit3", 512 + 16 * 32, 512)

    # out_nonlinear is BN over 512 (output of transit3).
    var out_nonlin = _load_batchnorm(ctx, xv + "/out_nonlinear", 512)
    # dense: Linear(1024, 192) — input is 1024 = 512 (channels) * 2 (mean+std from StatsPool).
    var dense = _load_dense_layer(ctx, xv + "/dense", 1024, 192)

    var xvector = XVectorBackbone(
        tdnn^, block1^, transit1^, block2^, transit2^, block3^, transit3^,
        out_nonlin^, dense^,
    )

    # Head: bn1+conv1+bn2+conv2 + layer1 (2 blocks) + layer2 (2 blocks).
    var hd = base + "/head"
    var head_bn1 = _load_batchnorm(ctx, hd + "/bn1", 192, nested=False)
    var hc1_w = upload_fp32(ctx, hd + "/conv1/weight.bin")
    var head_conv1 = Conv1d(hc1_w^, _zero_buf(ctx, 192)^, 192, 192, 3, 1, 1, 1, 1, False)
    var head_bn2 = _load_batchnorm(ctx, hd + "/bn2", 192, nested=False)
    var hc2_w = upload_fp32(ctx, hd + "/conv2/weight.bin")
    var head_conv2 = Conv1d(hc2_w^, _zero_buf(ctx, 192)^, 192, 192, 3, 1, 1, 1, 1, False)
    var layer1 = _load_resnet_layer(ctx, hd + "/layer1", 192)
    var layer2 = _load_resnet_layer(ctx, hd + "/layer2", 192)
    var head = ResNetHead(
        head_bn1^, head_conv1^, head_bn2^, head_conv2^, layer1^, layer2^,
    )

    return CAMPPlus(xvector^, head^)


# ============================================================================
# CFM estimator loader (real upstream shape)
# ============================================================================

def _load_block1d(
    mut ctx: DeviceContext, base: String, c_in: Int, c_out: Int,
) raises -> Block1D:
    """CausalBlock1D = Sequential(CausalConv1d, Transpose, LayerNorm, Transpose, Mish).

    On disk:
      {base}/block/0/{weight,bias}.bin   — CausalConv1d (c_out, c_in, 3)
      {base}/block/2/{weight,bias}.bin   — LayerNorm (c_out,)
    (Transpose ops and Mish have no weights.)

    Loader treats the conv as a normal 1D conv; the forward path pre-pads the
    input by (k-1) on the left to realize the causal behavior, then no
    additional pad is applied inside the conv kernel.
    """
    var cw = upload_fp32(ctx, base + "/block/0/weight.bin")
    var cb = upload_fp32(ctx, base + "/block/0/bias.bin")
    # Causal conv: padding=0 here, caller pads input by (k-1)=2 on the left.
    var conv = Conv1d(cw^, cb^, c_in, c_out, 3, 1, 0, 1, 1, True)
    var lw = upload_fp32(ctx, base + "/block/2/weight.bin")
    var lb = upload_fp32(ctx, base + "/block/2/bias.bin")
    var ln = LayerNorm(lw^, lb^, c_out, Float32(1.0e-5))
    return Block1D(conv^, ln^)


def _load_resnet1d(
    mut ctx: DeviceContext, base: String, c_in: Int, c_out: Int,
    time_emb_dim: Int,
) raises -> Resnet1D:
    """Resnet1D: block1 + block2 + mlp + res_conv (1x1, only when c_in ≠ c_out)."""
    var block1 = _load_block1d(ctx, base + "/block1", c_in, c_out)
    var block2 = _load_block1d(ctx, base + "/block2", c_out, c_out)
    # mlp is stored as `mlp.1.{weight,bias}.bin` (index 0 is activation).
    var mw = upload_fp32(ctx, base + "/mlp/1/weight.bin")
    var mb = upload_fp32(ctx, base + "/mlp/1/bias.bin")
    var mlp = Linear(mw^, mb^, time_emb_dim, c_out, True)
    # res_conv is always present in upstream weights (even when c_in == c_out — uses identity-ish 1x1).
    var rw = upload_fp32(ctx, base + "/res_conv/weight.bin")
    var rb = upload_fp32(ctx, base + "/res_conv/bias.bin")
    var res_conv = Conv1d(rw^, rb^, c_in, c_out, 1, 1, 0, 1, 1, True)
    return Resnet1D(block1^, block2^, mlp^, res_conv^)


def _load_cfm_attention(
    mut ctx: DeviceContext, base: String,
    d_model: Int, n_heads: Int, head_dim: Int,
) raises -> CFMAttention:
    """Attention with Q/K/V dim=512, output dim=256, stored as plain Linear
    layers (matches upstream `nn.Linear`).

    Shapes:
      attn1.to_q.weight (512, 256)      — no bias
      attn1.to_k.weight (512, 256)      — no bias
      attn1.to_v.weight (512, 256)      — no bias
      attn1.to_out.0.weight (256, 512), .bias (256,)
    """
    var inner = n_heads * head_dim   # 512
    var qw = upload_fp32(ctx, base + "/to_q/weight.bin")
    var kw = upload_fp32(ctx, base + "/to_k/weight.bin")
    var vw = upload_fp32(ctx, base + "/to_v/weight.bin")
    var ow = upload_fp32(ctx, base + "/to_out/0/weight.bin")
    var ob = upload_fp32(ctx, base + "/to_out/0/bias.bin")

    var zero = _zero_buf(ctx, inner)
    var to_q = Linear(qw^, zero^, d_model, inner, False)
    var zero_k = _zero_buf(ctx, inner)
    var to_k = Linear(kw^, zero_k^, d_model, inner, False)
    var zero_v = _zero_buf(ctx, inner)
    var to_v = Linear(vw^, zero_v^, d_model, inner, False)
    var to_out = Linear(ow^, ob^, inner, d_model, True)
    return CFMAttention(to_q^, to_k^, to_v^, to_out^, n_heads, head_dim)


def _load_cfm_ff(
    mut ctx: DeviceContext, base: String, d_model: Int, intermediate: Int,
) raises -> CFMFeedForward:
    """GEGLU FF — net.0.proj (intermediate, d_model), net.2 (d_model, intermediate)."""
    var p0w = upload_fp32(ctx, base + "/net/0/proj/weight.bin")
    var p0b = upload_fp32(ctx, base + "/net/0/proj/bias.bin")
    var net0 = Linear(p0w^, p0b^, d_model, intermediate, True)
    var p2w = upload_fp32(ctx, base + "/net/2/weight.bin")
    var p2b = upload_fp32(ctx, base + "/net/2/bias.bin")
    var net2 = Linear(p2w^, p2b^, intermediate, d_model, True)
    return CFMFeedForward(net0^, net2^)


def _load_basic_transformer_block(
    mut ctx: DeviceContext, base: String,
    d_model: Int, n_heads: Int, head_dim: Int, intermediate: Int,
) raises -> BasicTransformerBlock:
    var n1_w = upload_fp32(ctx, base + "/norm1/weight.bin")
    var n1_b = upload_fp32(ctx, base + "/norm1/bias.bin")
    var norm1 = LayerNorm(n1_w^, n1_b^, d_model, Float32(1.0e-5))
    var attn1 = _load_cfm_attention(ctx, base + "/attn1", d_model, n_heads, head_dim)
    var n3_w = upload_fp32(ctx, base + "/norm3/weight.bin")
    var n3_b = upload_fp32(ctx, base + "/norm3/bias.bin")
    var norm3 = LayerNorm(n3_w^, n3_b^, d_model, Float32(1.0e-5))
    var ff = _load_cfm_ff(ctx, base + "/ff", d_model, intermediate)
    return BasicTransformerBlock(norm1^, attn1^, norm3^, ff^)


def _load_transformer_stack(
    mut ctx: DeviceContext, base: String, n_blocks: Int,
    d_model: Int, n_heads: Int, head_dim: Int, intermediate: Int,
) raises -> List[BasicTransformerBlock]:
    var blocks = List[BasicTransformerBlock]()
    for i in range(n_blocks):
        var b = _load_basic_transformer_block(
            ctx, base + "/" + String(i),
            d_model, n_heads, head_dim, intermediate,
        )
        blocks.append(b^)
    return blocks^


def load_cfm_estimator_real(mut ctx: DeviceContext, base: String) raises -> CFMEstimatorReal:
    """Load CFM estimator (Matcha-TTS / CosyVoice style) from
    weights/s3gen/flow/decoder/estimator/.

    Hardcoded architectural constants matching the upstream weight shapes:
      d_model = 256, time_emb_dim = 1024,
      attn inner = 512 (n_heads = 8, head_dim = 64),
      ff intermediate = 1024,
      down/up have 1 stage each, mid has 12 stages,
      each stage has 4 transformer blocks after the resnet,
      first down stage Conv1d input = 320 (mel + cond + spk emb concat).
    """
    comptime D = 256
    comptime TIME_DIM = 1024
    comptime N_HEADS = 8
    comptime HEAD_DIM = 64
    comptime FF_INTER = 1024
    comptime N_TRANSFORMER_PER_STAGE = 4
    comptime FIRST_IN = 320   # mel + cond + spk concat
    comptime MEL = 80

    # time_mlp: Linear (TIME_DIM, time_input=320) → silu (no params) → Linear (TIME_DIM, TIME_DIM)
    var t1_w = upload_fp32(ctx, base + "/time_mlp/linear_1/weight.bin")
    var t1_b = upload_fp32(ctx, base + "/time_mlp/linear_1/bias.bin")
    var time_mlp1 = Linear(t1_w^, t1_b^, 320, TIME_DIM, True)
    var t2_w = upload_fp32(ctx, base + "/time_mlp/linear_2/weight.bin")
    var t2_b = upload_fp32(ctx, base + "/time_mlp/linear_2/bias.bin")
    var time_mlp2 = Linear(t2_w^, t2_b^, TIME_DIM, TIME_DIM, True)

    # down_blocks: 1 stage.
    var down_blocks = List[CFMDownStage]()
    var d0_resnet = _load_resnet1d(ctx, base + "/down_blocks/0/0", FIRST_IN, D, TIME_DIM)
    var d0_trans = _load_transformer_stack(
        ctx, base + "/down_blocks/0/1", N_TRANSFORMER_PER_STAGE,
        D, N_HEADS, HEAD_DIM, FF_INTER,
    )
    # is_last=True (only stage) → downsampler is CausalConv1d(k=3) (stride 1, left-pad-only).
    var d0_dn_w = upload_fp32(ctx, base + "/down_blocks/0/2/weight.bin")
    var d0_dn_b = upload_fp32(ctx, base + "/down_blocks/0/2/bias.bin")
    var d0_downsampler = Conv1d(d0_dn_w^, d0_dn_b^, D, D, 3, 1, 0, 1, 1, True)
    down_blocks.append(CFMDownStage(d0_resnet^, d0_trans^, d0_downsampler^))

    # mid_blocks: 12 stages.
    var mid_blocks = List[CFMMidStage]()
    for i in range(12):
        var m_base = base + "/mid_blocks/" + String(i)
        var m_resnet = _load_resnet1d(ctx, m_base + "/0", D, D, TIME_DIM)
        var m_trans = _load_transformer_stack(
            ctx, m_base + "/1", N_TRANSFORMER_PER_STAGE,
            D, N_HEADS, HEAD_DIM, FF_INTER,
        )
        mid_blocks.append(CFMMidStage(m_resnet^, m_trans^))

    # up_blocks: 1 stage. Resnet takes concat (skip + x) so input is 2*D = 512.
    var up_blocks = List[CFMUpStage]()
    var u0_resnet = _load_resnet1d(ctx, base + "/up_blocks/0/0", D * 2, D, TIME_DIM)
    var u0_trans = _load_transformer_stack(
        ctx, base + "/up_blocks/0/1", N_TRANSFORMER_PER_STAGE,
        D, N_HEADS, HEAD_DIM, FF_INTER,
    )
    # is_last=True → upsampler is CausalConv1d(k=3) (left-pad-only).
    var u0_up_w = upload_fp32(ctx, base + "/up_blocks/0/2/weight.bin")
    var u0_up_b = upload_fp32(ctx, base + "/up_blocks/0/2/bias.bin")
    var u0_upsampler = Conv1d(u0_up_w^, u0_up_b^, D, D, 3, 1, 0, 1, 1, True)
    up_blocks.append(CFMUpStage(u0_resnet^, u0_trans^, u0_upsampler^))

    # final_block: just a Block1D — Conv1d + GroupNorm.
    var final_block = _load_block1d(ctx, base + "/final_block", D, D)

    # final_proj: Conv1d 256 → 80 (k=1).
    var fp_w = upload_fp32(ctx, base + "/final_proj/weight.bin")
    var fp_b = upload_fp32(ctx, base + "/final_proj/bias.bin")
    var final_proj = Conv1d(fp_w^, fp_b^, D, MEL, 1, 1, 0, 1, 1, True)

    return CFMEstimatorReal(
        time_mlp1^, time_mlp2^,
        down_blocks^, mid_blocks^, up_blocks^,
        final_block^, final_proj^,
    )


# ============================================================================
# T3CondEnc loader — Perceiver resampler + speaker projection + emotion FC
# ============================================================================

def load_t3_cond_enc(
    mut ctx: DeviceContext, base: String,
    speech_emb: Embedding,
    speech_pos_emb: Embedding,
    n_queries: Int = 32,
    n_perc_heads: Int = 4,
    perc_head_dim: Int = 256,
    speaker_embed_size: Int = 256,
    d_model: Int = 1024,
    cond_prompt_len: Int = 150,
) raises -> T3CondEnc:
    """Load T3CondEnc from {base}/cond_enc/. Reuses `speech_emb` from the
    parent T3 (cond_prompt tokens use the same embedding table).

    Hyperparameters default to the upstream Chatterbox values; override at call
    site if needed.
    """
    var ce = base + "/cond_enc"

    # Speaker projection: 256 → 1024.
    var sw = upload_fp32(ctx, ce + "/spkr_w.bin")
    var sb = upload_fp32(ctx, ce + "/spkr_b.bin")
    var spkr_enc = Linear(sw^, sb^, speaker_embed_size, d_model, True)

    # Emotion FC: 1 → 1024.
    var ew = upload_fp32(ctx, ce + "/emo_w.bin")
    var zero_eb = _zero_buf(ctx, d_model)
    var emotion_fc = Linear(ew^, zero_eb^, 1, d_model, False)

    # Perceiver: pre_attention_query + one block (norm, q/k/v, proj_out).
    var pre_q = upload_fp32(ctx, ce + "/perceiver/pre_q.bin")
    var pn_w = upload_fp32(ctx, ce + "/perceiver/perc_norm_w.bin")
    var pn_b = upload_fp32(ctx, ce + "/perceiver/perc_norm_b.bin")
    var perc_ln = LayerNorm(pn_w^, pn_b^, d_model, Float32(1.0e-5))

    var qw = upload_fp32(ctx, ce + "/perceiver/perc_q_w.bin")
    var qb = upload_fp32(ctx, ce + "/perceiver/perc_q_b.bin")
    var to_q = Linear(qw^, qb^, d_model, d_model, True)
    var kw = upload_fp32(ctx, ce + "/perceiver/perc_k_w.bin")
    var kb = upload_fp32(ctx, ce + "/perceiver/perc_k_b.bin")
    var to_k = Linear(kw^, kb^, d_model, d_model, True)
    var vw = upload_fp32(ctx, ce + "/perceiver/perc_v_w.bin")
    var vb = upload_fp32(ctx, ce + "/perceiver/perc_v_b.bin")
    var to_v = Linear(vw^, vb^, d_model, d_model, True)
    var ow = upload_fp32(ctx, ce + "/perceiver/perc_o_w.bin")
    var ob = upload_fp32(ctx, ce + "/perceiver/perc_o_b.bin")
    var proj_out = Linear(ow^, ob^, d_model, d_model, True)

    var block = PerceiverBlock(perc_ln^, to_q^, to_k^, to_v^, proj_out^,
                                n_perc_heads, perc_head_dim)
    var perceiver = Perceiver(pre_q^, block^, n_queries, d_model)

    return T3CondEnc(spkr_enc^, emotion_fc^, speech_emb.copy(), speech_pos_emb.copy(),
                      perceiver^, speaker_embed_size, d_model, cond_prompt_len)


# ============================================================================
# UpsampleConformerEncoder loader (flow encoder wrapper)
# ============================================================================

def _load_embed_out(
    mut ctx: DeviceContext, base: String, idim: Int, odim: Int,
) raises -> EmbedOut:
    """Load embed.out = Sequential(Linear(idim, odim), LayerNorm(odim), Dropout).

    On disk:
      {base}/0/{weight,bias}.bin   — Linear (odim, idim)
      {base}/1/{weight,bias}.bin   — LayerNorm (odim,)
    """
    var lw = upload_fp32(ctx, base + "/0/weight.bin")
    var lb = upload_fp32(ctx, base + "/0/bias.bin")
    var lin = Linear(lw^, lb^, idim, odim, True)
    var nw = upload_fp32(ctx, base + "/1/weight.bin")
    var nb = upload_fp32(ctx, base + "/1/bias.bin")
    var ln = LayerNorm(nw^, nb^, odim, Float32(1.0e-5))
    return EmbedOut(lin^, ln^)


def _load_pre_lookahead(
    mut ctx: DeviceContext, base: String, channels: Int, pre_la_len: Int = 3,
) raises -> PreLookaheadLayer:
    var c1_w = upload_fp32(ctx, base + "/conv1/weight.bin")
    var c1_b = upload_fp32(ctx, base + "/conv1/bias.bin")
    # kernel = pre_lookahead_len + 1 = 4, padding=0 (caller pads input).
    var conv1 = Conv1d(c1_w^, c1_b^, channels, channels, pre_la_len + 1, 1, 0, 1, 1, True)
    var c2_w = upload_fp32(ctx, base + "/conv2/weight.bin")
    var c2_b = upload_fp32(ctx, base + "/conv2/bias.bin")
    # kernel = 3, padding=0.
    var conv2 = Conv1d(c2_w^, c2_b^, channels, channels, 3, 1, 0, 1, 1, True)
    return PreLookaheadLayer(conv1^, conv2^)


def _load_up_layer(
    mut ctx: DeviceContext, base: String, channels: Int, stride: Int = 2,
) raises -> UpLayerConv:
    var cw = upload_fp32(ctx, base + "/conv/weight.bin")
    var cb = upload_fp32(ctx, base + "/conv/bias.bin")
    # kernel = stride*2 + 1 = 5, padding=0 (caller pads input).
    var conv = Conv1d(cw^, cb^, channels, channels, stride * 2 + 1, 1, 0, 1, 1, True)
    return UpLayerConv(conv^)


def load_upsample_conformer_encoder(
    mut ctx: DeviceContext, base: String,
) raises -> UpsampleConformerEncoderReal:
    """Load full UpsampleConformerEncoder from weights/s3gen/flow/.

    Architectural constants from upstream:
      vocab_size = 6561 (speech tokens), d_model = 512, n_heads = 8, head_dim = 64,
      intermediate (FF) = 2048, mel_dim = 80,
      6 pre-upsample encoder layers, 4 post-upsample encoder layers.
    """
    comptime VOCAB = 6561   # speech token vocab from input_embedding shape
    # flow.input_embedding only embeds speech tokens.
    comptime D = 512
    comptime INT = 2048
    comptime H = 8
    comptime DH = 64
    comptime MEL = 80

    # input_embedding: vocab → D.
    var iew = upload_fp32(ctx, base + "/input_embedding/weight.bin")
    var input_embedding = Embedding(iew^, VOCAB, D)

    var embed = _load_embed_out(ctx, base + "/encoder/embed/out", D, D)
    var up_embed = _load_embed_out(ctx, base + "/encoder/up_embed/out", D, D)

    var pre_la = _load_pre_lookahead(ctx, base + "/encoder/pre_lookahead_layer", D, 3)
    var up_layer = _load_up_layer(ctx, base + "/encoder/up_layer", D, 2)

    var encoders = load_s3gen_flow_encoder_layers(
        ctx, base + "/encoder/encoders", 6, D, INT, H, DH,
    )
    var up_encoders = load_s3gen_flow_encoder_layers(
        ctx, base + "/encoder/up_encoders", 4, D, INT, H, DH,
    )

    var an_w = upload_fp32(ctx, base + "/encoder/after_norm/weight.bin")
    var an_b = upload_fp32(ctx, base + "/encoder/after_norm/bias.bin")
    var after_norm = LayerNorm(an_w^, an_b^, D, Float32(1.0e-5))

    var ep_w = upload_fp32(ctx, base + "/encoder_proj/weight.bin")
    var ep_b = upload_fp32(ctx, base + "/encoder_proj/bias.bin")
    var encoder_proj = Linear(ep_w^, ep_b^, D, MEL, True)

    return UpsampleConformerEncoderReal(
        input_embedding^, embed^, up_embed^, pre_la^, up_layer^,
        encoders^, up_encoders^, after_norm^, encoder_proj^,
        D, MEL,
    )


# ============================================================================
# FCM loader (CAMPPlus head — feature compression module)
# ============================================================================


def _load_conv2d(mut ctx: DeviceContext, base: String,
                  c_out: Int, c_in: Int, kh: Int, kw: Int,
                  sh: Int, sw: Int, ph: Int, pw: Int) raises -> Conv2d:
    var w = upload_fp32(ctx, base + "/weight.bin")
    var zero_bias = ctx.enqueue_create_buffer[DType.float32](c_out)
    zero_bias.enqueue_fill(0.0)
    return Conv2d(w^, zero_bias^, c_in, c_out, kh, kw, sh, sw, ph, pw, False)


def _load_bn2d(mut ctx: DeviceContext, base: String, channels: Int) raises -> BatchNorm2d:
    var w = upload_fp32(ctx, base + "/weight.bin")
    var b = upload_fp32(ctx, base + "/bias.bin")
    var rm = upload_fp32(ctx, base + "/running_mean.bin")
    var rv = upload_fp32(ctx, base + "/running_var.bin")
    return BatchNorm2d(w^, b^, rm^, rv^, channels, Float32(1.0e-5))


def _load_basic_resblock_2d(mut ctx: DeviceContext, base: String,
                              in_planes: Int, planes: Int, stride: Int) raises -> BasicResBlock2d:
    var conv1 = _load_conv2d(ctx, base + "/conv1", planes, in_planes, 3, 3, stride, 1, 1, 1)
    var bn1 = _load_bn2d(ctx, base + "/bn1", planes)
    var conv2 = _load_conv2d(ctx, base + "/conv2", planes, planes, 3, 3, 1, 1, 1, 1)
    var bn2 = _load_bn2d(ctx, base + "/bn2", planes)
    var has_shortcut = (stride != 1) or (in_planes != planes)
    # Dummy shortcut (only used if has_shortcut=True).
    if has_shortcut:
        var sc_conv = _load_conv2d(ctx, base + "/shortcut/0", planes, in_planes, 1, 1, stride, 1, 0, 0)
        var sc_bn = _load_bn2d(ctx, base + "/shortcut/1", planes)
        return BasicResBlock2d(conv1^, bn1^, conv2^, bn2^, True, sc_conv^, sc_bn^, stride, in_planes, planes)
    else:
        # Build dummy buffers for unused shortcut. Each field needs its own buffer.
        var d1 = ctx.enqueue_create_buffer[DType.float32](1)
        var d2 = ctx.enqueue_create_buffer[DType.float32](1)
        var d3 = ctx.enqueue_create_buffer[DType.float32](1)
        var d4 = ctx.enqueue_create_buffer[DType.float32](1)
        var d5 = ctx.enqueue_create_buffer[DType.float32](1)
        var d6 = ctx.enqueue_create_buffer[DType.float32](1)
        var sc_conv = Conv2d(d1^, d2^, 1, 1, 1, 1, 1, 1, 0, 0, False)
        var sc_bn = BatchNorm2d(d3^, d4^, d5^, d6^, 1, Float32(1.0e-5))
        return BasicResBlock2d(conv1^, bn1^, conv2^, bn2^, False, sc_conv^, sc_bn^, stride, in_planes, planes)


def load_fcm(mut ctx: DeviceContext, base: String) raises -> FCM:
    """Load FCM from weights/s3gen/speaker_encoder/head/."""
    var M = 32
    var FEAT = 80

    var conv1 = _load_conv2d(ctx, base + "/conv1", M, 1, 3, 3, 1, 1, 1, 1)
    var bn1 = _load_bn2d(ctx, base + "/bn1", M)

    var l1_b0 = _load_basic_resblock_2d(ctx, base + "/layer1/0", M, M, 2)
    var l1_b1 = _load_basic_resblock_2d(ctx, base + "/layer1/1", M, M, 1)
    var l2_b0 = _load_basic_resblock_2d(ctx, base + "/layer2/0", M, M, 2)
    var l2_b1 = _load_basic_resblock_2d(ctx, base + "/layer2/1", M, M, 1)

    var conv2 = _load_conv2d(ctx, base + "/conv2", M, M, 3, 3, 2, 1, 1, 1)
    var bn2 = _load_bn2d(ctx, base + "/bn2", M)

    return FCM(conv1^, bn1^, l1_b0^, l1_b1^, l2_b0^, l2_b1^, conv2^, bn2^, FEAT, M)
