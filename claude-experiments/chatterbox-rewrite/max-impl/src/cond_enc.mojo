"""T3CondEnc: builds the cond_emb prefix from speaker_emb + cond_prompt_tokens +
emotion_adv, using Perceiver + Linear + Embedding + concat — all via MAX ops."""
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from modules import Linear, linear_forward, Embedding, embedding_forward
from perceiver import Perceiver, perceiver_forward


@fieldwise_init
struct T3CondEnc(Copyable, Movable):
    var spkr_enc: Linear
    var emotion_fc: Linear
    var speech_emb: Embedding   # shared with T3's speech token embedding
    var speech_pos_emb: Embedding  # shared with T3's speech pos embedding
    var perceiver: Perceiver
    var speaker_embed_size: Int
    var d_model: Int
    var cond_prompt_len: Int


def cond_emb_concat(
    mut ctx: DeviceContext,
    mut spkr_buf: DeviceBuffer[DType.float32],   # (B, 1, D)
    mut perc_buf: DeviceBuffer[DType.float32],   # (B, 32, D)
    mut emo_buf: DeviceBuffer[DType.float32],    # (B, 1, D)
    mut out_buf: DeviceBuffer[DType.float32],    # (B, 1+32+1, D)
    b: Int, sq: Int, d: Int,
) raises:
    """Concat spkr | perceiver | emotion along the time axis. T_cond = 2 + sq."""
    var t_cond = 2 + sq
    var s_ptr = spkr_buf.unsafe_ptr()
    var p_ptr = perc_buf.unsafe_ptr()
    var e_ptr = emo_buf.unsafe_ptr()
    var o_ptr = out_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(s_ptr, p_ptr, e_ptr, o_ptr, t_cond, sq, d)
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (t_cond * d)
        var rem = i - bi * t_cond * d
        var ti = rem // d
        var di = rem - ti * d
        if ti == 0:
            o_ptr[i] = s_ptr[bi * d + di]
        elif ti < 1 + sq:
            var src = ti - 1
            o_ptr[i] = p_ptr[bi * sq * d + src * d + di]
        else:
            o_ptr[i] = e_ptr[bi * d + di]
    elementwise[func, simd_width=1, target="gpu"](
        IndexList[1](b * t_cond * d), DeviceContextPtr(ctx),
    )


def t3_cond_enc_forward(
    mut ctx: DeviceContext,
    mut model: T3CondEnc,
    mut speaker_emb_buf: DeviceBuffer[DType.float32],     # (B, speaker_embed_size)
    mut cond_tokens_buf: DeviceBuffer[DType.int64],       # (B, cond_prompt_len)
    mut emotion_adv_buf: DeviceBuffer[DType.float32],     # (B, 1, 1)
    mut cond_emb_out_buf: DeviceBuffer[DType.float32],    # (B, T_cond=34, D)
    mut zero_mask_q: DeviceBuffer[DType.float32],
    mut zero_mask_qq: DeviceBuffer[DType.float32],
    b: Int,
) raises:
    var D = model.d_model
    var SE = model.speaker_embed_size
    var CL = model.cond_prompt_len
    var SQ = model.perceiver.n_queries   # 32

    # spkr_enc(speaker_emb): (B, SE) → (B, 1, D). linear_forward returns (B, D) flattened;
    # we treat it as (B*1, D) — same memory.
    var spkr_proj = ctx.enqueue_create_buffer[DType.float32](b * D)
    linear_forward(ctx, model.spkr_enc, speaker_emb_buf, spkr_proj, b)

    # speech_emb(cond_tokens) + speech_pos_emb[arange(CL)]: (B, CL) → (B, CL, D).
    var cs_emb = ctx.enqueue_create_buffer[DType.float32](b * CL * D)
    embedding_forward(ctx, model.speech_emb, cond_tokens_buf, cs_emb, b, CL)

    var ce_ptr = cs_emb.unsafe_ptr()
    var pp_ptr = model.speech_pos_emb.table.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(ce_ptr, pp_ptr, b, CL, D)
    def add_pos_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (CL * D)
        var rem = i - bi * CL * D
        var ti = rem // D
        var di = rem - ti * D
        ce_ptr[i] = ce_ptr[i] + pp_ptr[ti * D + di]
    elementwise[add_pos_fn, simd_width=1, target="gpu"](
        IndexList[1](b * CL * D), DeviceContextPtr(ctx),
    )

    # perceiver(cs_emb): (B, CL, D) → (B, SQ, D).
    var perc_out = ctx.enqueue_create_buffer[DType.float32](b * SQ * D)
    perceiver_forward(ctx, model.perceiver, cs_emb, perc_out,
                       zero_mask_q, zero_mask_qq, b, CL)

    # emotion_fc(emotion_adv): (B, 1, 1) → (B, 1, D). emotion_adv has D-element flatten.
    var emo_out = ctx.enqueue_create_buffer[DType.float32](b * D)
    linear_forward(ctx, model.emotion_fc, emotion_adv_buf, emo_out, b)

    # Concat spkr_proj | perc_out | emo_out → cond_emb_out (B, T_cond=2+SQ, D).
    cond_emb_concat(ctx, spkr_proj, perc_out, emo_out, cond_emb_out_buf,
                     b, SQ, D)
