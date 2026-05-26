"""T3 + T3CondEnc weight loaders."""
from std.gpu.host import DeviceContext, DeviceBuffer

from fixture import load_fp32, load_bf16, Tensor, TensorBF16
from std.os import getenv
from modules import Linear, LayerNorm, RMSNorm, Embedding
from t3_block import T3Block
from t3 import T3
from cond_enc import T3CondEnc
from perceiver import Perceiver, PerceiverBlock
from transformer_blocks import LlamaMLP


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


def upload_bf16(mut ctx: DeviceContext, path: String) raises -> DeviceBuffer[DType.bfloat16]:
    """Read a `.bf16.bin` file and upload to GPU."""
    var t = load_bf16(path)
    var n = len(t.data)
    var buf = ctx.enqueue_create_buffer[DType.bfloat16](n)
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = t.data[i]
    return buf^


def upload_bf16_concat_rows(
    mut ctx: DeviceContext, paths: List[String]
) raises -> DeviceBuffer[DType.bfloat16]:
    """Concatenate row-major bf16 weights along OUT dim → (sum(OUT_i), IN)."""
    var total = 0
    var tensors = List[TensorBF16]()
    for i in range(len(paths)):
        var t = load_bf16(paths[i])
        total += len(t.data)
        tensors.append(t^)
    var buf = ctx.enqueue_create_buffer[DType.bfloat16](total)
    with buf.map_to_host() as h:
        var off = 0
        for ti in range(len(tensors)):
            ref data = tensors[ti].data
            var n = len(data)
            for k in range(n):
                h[off + k] = data[k]
            off += n
    return buf^


def _use_bf16() -> Bool:
    """Returns True iff CHATTERBOX_BF16=1 is set in the environment."""
    try:
        var v = getenv("CHATTERBOX_BF16")
        return v == "1"
    except:
        return False


def upload_concat_rows_fp32(
    mut ctx: DeviceContext, paths: List[String]
) raises -> DeviceBuffer[DType.float32]:
    """Concatenate row-major (OUT_i, IN) weights along OUT dim → (sum(OUT_i), IN)."""
    var total = 0
    var tensors = List[Tensor]()
    for i in range(len(paths)):
        var t = load_fp32(paths[i])
        total += len(t.data)
        tensors.append(t^)
    var buf = ctx.enqueue_create_buffer[DType.float32](total)
    with buf.map_to_host() as h:
        var off = 0
        for ti in range(len(tensors)):
            ref data = tensors[ti].data
            var n = len(data)
            for k in range(n):
                h[off + k] = data[k]
            off += n
    return buf^


def _zero_buf(mut ctx: DeviceContext, n: Int) raises -> DeviceBuffer[DType.float32]:
    var b = ctx.enqueue_create_buffer[DType.float32](n)
    b.enqueue_fill(0.0)
    return b^


def load_t3_block(mut ctx: DeviceContext, layer_base: String,
                   hidden: Int, intermediate: Int,
                   n_heads: Int, head_dim: Int) raises -> T3Block:
    var in_norm_w = upload_fp32(ctx, layer_base + "/in_norm.bin")
    var post_norm_w = upload_fp32(ctx, layer_base + "/post_norm.bin")
    var qw = upload_fp32(ctx, layer_base + "/qw.bin")
    var kw = upload_fp32(ctx, layer_base + "/kw.bin")
    var vw = upload_fp32(ctx, layer_base + "/vw.bin")
    var ow = upload_fp32(ctx, layer_base + "/ow.bin")
    var gate_w = upload_fp32(ctx, layer_base + "/gate_w.bin")
    var up_w = upload_fp32(ctx, layer_base + "/up_w.bin")
    var down_w = upload_fp32(ctx, layer_base + "/down_w.bin")

    var qkv_paths = List[String]()
    qkv_paths.append(layer_base + "/qw.bin")
    qkv_paths.append(layer_base + "/kw.bin")
    qkv_paths.append(layer_base + "/vw.bin")
    var qkv_w = upload_concat_rows_fp32(ctx, qkv_paths)

    var gu_paths = List[String]()
    gu_paths.append(layer_base + "/gate_w.bin")
    gu_paths.append(layer_base + "/up_w.bin")
    var gate_up_w = upload_concat_rows_fp32(ctx, gu_paths)

    var zero_d = ctx.enqueue_create_buffer[DType.float32](hidden)
    zero_d.enqueue_fill(0.0)
    var zero_inter = ctx.enqueue_create_buffer[DType.float32](intermediate)
    zero_inter.enqueue_fill(0.0)
    var zero_3d = ctx.enqueue_create_buffer[DType.float32](3 * hidden)
    zero_3d.enqueue_fill(0.0)
    var zero_2inter = ctx.enqueue_create_buffer[DType.float32](2 * intermediate)
    zero_2inter.enqueue_fill(0.0)

    var in_norm = RMSNorm(in_norm_w^, hidden, Float32(1.0e-5))
    var post_norm = RMSNorm(post_norm_w^, hidden, Float32(1.0e-5))

    if _use_bf16():
        # Load pre-cast bf16 copies alongside the f32 weights.
        var qw_b = upload_bf16(ctx, layer_base + "/qw.bf16.bin")
        var kw_b = upload_bf16(ctx, layer_base + "/kw.bf16.bin")
        var vw_b = upload_bf16(ctx, layer_base + "/vw.bf16.bin")
        var ow_b = upload_bf16(ctx, layer_base + "/ow.bf16.bin")
        var gate_b = upload_bf16(ctx, layer_base + "/gate_w.bf16.bin")
        var up_b = upload_bf16(ctx, layer_base + "/up_w.bf16.bin")
        var down_b = upload_bf16(ctx, layer_base + "/down_w.bf16.bin")

        var qkv_b_paths = List[String]()
        qkv_b_paths.append(layer_base + "/qw.bf16.bin")
        qkv_b_paths.append(layer_base + "/kw.bf16.bin")
        qkv_b_paths.append(layer_base + "/vw.bf16.bin")
        var qkv_b = upload_bf16_concat_rows(ctx, qkv_b_paths)

        var gu_b_paths = List[String]()
        gu_b_paths.append(layer_base + "/gate_w.bf16.bin")
        gu_b_paths.append(layer_base + "/up_w.bf16.bin")
        var gate_up_b = upload_bf16_concat_rows(ctx, gu_b_paths)

        var to_q = Linear(qw^, zero_d, hidden, hidden, False, qw_b^)
        var to_k = Linear(kw^, zero_d.copy(), hidden, hidden, False, kw_b^)
        var to_v = Linear(vw^, zero_d.copy(), hidden, hidden, False, vw_b^)
        var to_out = Linear(ow^, zero_d.copy(), hidden, hidden, False, ow_b^)
        var gate = Linear(gate_w^, zero_inter, hidden, intermediate, False, gate_b^)
        var up = Linear(up_w^, zero_inter.copy(), hidden, intermediate, False, up_b^)
        var down = Linear(down_w^, zero_d.copy(), intermediate, hidden, False, down_b^)
        var mlp = LlamaMLP(gate^, up^, down^, hidden, intermediate)
        var qkv = Linear(qkv_w^, zero_3d, hidden, 3 * hidden, False, qkv_b^)
        var gate_up = Linear(gate_up_w^, zero_2inter, hidden, 2 * intermediate, False, gate_up_b^)

        return T3Block(in_norm^, post_norm^, to_q^, to_k^, to_v^, to_out^, mlp^,
                        qkv^, gate_up^, n_heads, head_dim)
    else:
        var to_q = Linear(qw^, zero_d, hidden, hidden, False)
        var to_k = Linear(kw^, zero_d.copy(), hidden, hidden, False)
        var to_v = Linear(vw^, zero_d.copy(), hidden, hidden, False)
        var to_out = Linear(ow^, zero_d.copy(), hidden, hidden, False)
        var gate = Linear(gate_w^, zero_inter, hidden, intermediate, False)
        var up = Linear(up_w^, zero_inter.copy(), hidden, intermediate, False)
        var down = Linear(down_w^, zero_d.copy(), intermediate, hidden, False)
        var mlp = LlamaMLP(gate^, up^, down^, hidden, intermediate)
        var qkv = Linear(qkv_w^, zero_3d, hidden, 3 * hidden, False)
        var gate_up = Linear(gate_up_w^, zero_2inter, hidden, 2 * intermediate, False)

        return T3Block(in_norm^, post_norm^, to_q^, to_k^, to_v^, to_out^, mlp^,
                        qkv^, gate_up^, n_heads, head_dim)


def load_t3(mut ctx: DeviceContext, base: String) raises -> T3:
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
    var speech_head: Linear
    if _use_bf16():
        var speech_head_b = upload_bf16(ctx, base + "/speech_head_w.bf16.bin")
        speech_head = Linear(speech_head_w^, zero_d^, HIDDEN, V_SPEECH, False, speech_head_b^)
    else:
        speech_head = Linear(speech_head_w^, zero_d^, HIDDEN, V_SPEECH, False)

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


def load_t3_cond_enc(
    mut ctx: DeviceContext, base: String,
) raises -> T3CondEnc:
    """Load T3CondEnc. Caller passes the base path to the cond_enc weights dir.

    Uses its own speech_emb from {base}/cond_enc — we re-load it here rather
    than referencing T3's, since op_t3 builds them independently.
    """
    var n_queries = 32
    var n_perc_heads = 4
    var perc_head_dim = 256
    var speaker_embed_size = 256
    var d_model = 1024
    var cond_prompt_len = 150
    var V_SPEECH = 8194
    var MAX_SPEECH_POS = 4100

    var ce = base + "/cond_enc"

    var sw = upload_fp32(ctx, ce + "/spkr_w.bin")
    var sb = upload_fp32(ctx, ce + "/spkr_b.bin")
    var spkr_enc = Linear(sw^, sb^, speaker_embed_size, d_model, True)

    var ew = upload_fp32(ctx, ce + "/emo_w.bin")
    var zero_eb = _zero_buf(ctx, d_model)
    var emotion_fc = Linear(ew^, zero_eb^, 1, d_model, False)

    # speech_emb / speech_pos_emb reused from T3 path.
    var speech_emb_w = upload_fp32(ctx, base + "/speech_emb_w.bin")
    var speech_emb = Embedding(speech_emb_w^, V_SPEECH, d_model)
    var speech_pos_w = upload_fp32(ctx, base + "/speech_pos_w.bin")
    var speech_pos_emb = Embedding(speech_pos_w^, MAX_SPEECH_POS, d_model)

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

    return T3CondEnc(spkr_enc^, emotion_fc^, speech_emb^, speech_pos_emb^,
                      perceiver^, speaker_embed_size, d_model, cond_prompt_len)
