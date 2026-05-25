"""Weight loaders for op_audio_in: VoiceEncoder + S3Tokenizer.

Extracted from src/weights.mojo so this op compiles without pulling in
unrelated model loaders.
"""
from std.gpu.host import DeviceContext, DeviceBuffer

from fixture import load_fp32
from modules import Linear, LayerNorm, RMSNorm, Embedding
from conv1d import Conv1d
from lstm import LSTMLayer
from voice_encoder import VoiceEncoder
from transformer_blocks import MLP
from s3tokenizer_block import FSMNAttention, S3TokenizerBlock
from s3tokenizer import S3Tokenizer


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


def load_voice_encoder(mut ctx: DeviceContext, base: String) raises -> VoiceEncoder:
    var H = 256
    var IN_0 = 40
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


def load_s3tokenizer_block(
    mut ctx: DeviceContext, block_base: String,
    hidden: Int, intermediate: Int, n_heads: Int, head_dim: Int,
    fsmn_kernel: Int,
) raises -> S3TokenizerBlock:
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

    var zero_fsmn_b = _zero_buf(ctx, hidden)
    var pad = (fsmn_kernel - 1) // 2
    var fsmn_conv = Conv1d(
        fsmn_w^, zero_fsmn_b^,
        hidden, hidden, fsmn_kernel, 1, pad, 1, hidden, False,
    )

    var attn = FSMNAttention(to_q^, to_k^, to_v^, to_out^, fsmn_conv^,
                              n_heads, head_dim)

    var fc1 = Linear(fc1_w^, fc1_b^, hidden, intermediate, True)
    var fc2 = Linear(fc2_w^, fc2_b^, intermediate, hidden, True)
    var mlp = MLP(fc1^, fc2^, hidden, intermediate)

    return S3TokenizerBlock(attn_ln^, mlp_ln^, attn^, mlp^)


def load_s3tokenizer(mut ctx: DeviceContext, base: String) raises -> S3Tokenizer:
    var N_MELS = 128
    var N_STATE = 1280
    var N_HEADS = 20
    var HEAD_DIM = 64
    var INTERMEDIATE = 5120
    var N_LAYERS = 6
    var FSMN_K = 31

    var conv1_w = upload_fp32(ctx, base + "/conv1_w.bin")
    var conv1_b = upload_fp32(ctx, base + "/conv1_b.bin")
    var conv2_w = upload_fp32(ctx, base + "/conv2_w.bin")
    var conv2_b = upload_fp32(ctx, base + "/conv2_b.bin")

    var conv1 = Conv1d(conv1_w^, conv1_b^, N_MELS, N_STATE, 3, 2, 1, 1, 1, True)
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
