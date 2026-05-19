"""Build text-side embeddings for T3 from a string.

Replicates upstream:
    text_tokens = tokenizer.text_to_tokens(text)
    full = [start_text_token, *text_tokens, stop_text_token]
    text_emb = t3.text_emb(full) + t3.text_pos_emb(arange(len(full)))
    bos_emb = t3.speech_emb(start_speech_token) + t3.speech_pos_emb(0)

Also builds the RoPE cos/sin tables (head_dim=64, base=10000) for any context
length, since those are deterministic functions of position not actual data.
"""
from std.math import sin as msin, cos as mcos
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from bpe_tokenizer import Tokenizer, tokenize
from t3 import T3
from modules import Embedding, embedding_forward


comptime START_TEXT_TOKEN: Int = 255
comptime STOP_TEXT_TOKEN: Int = 0
comptime START_SPEECH_TOKEN: Int = 6561


def text_to_input_ids(text: String, tok: Tokenizer) raises -> List[Int64]:
    """Tokenize text and wrap with [START_TEXT, ..., STOP_TEXT]."""
    var raw_ids = tokenize(text, tok)
    var out = List[Int64]()
    out.append(Int64(START_TEXT_TOKEN))
    for i in range(len(raw_ids)):
        out.append(Int64(raw_ids[i]))
    out.append(Int64(STOP_TEXT_TOKEN))
    return out^


def build_text_emb(
    mut ctx: DeviceContext,
    mut model: T3,
    ids: List[Int64],
    mut out_buf: DeviceBuffer[DType.float32],   # (1, T_TEXT, D)
) raises:
    """text_emb(ids) + text_pos_emb(arange(len(ids))) → (1, T_TEXT, D)."""
    var T = len(ids)
    var D = model.d_model

    # Token embedding.
    var ids_buf = ctx.enqueue_create_buffer[DType.int64](T)
    with ids_buf.map_to_host() as h:
        for i in range(T):
            h[i] = ids[i]
    embedding_forward(ctx, model.text_emb, ids_buf, out_buf, 1, T)
    ctx.synchronize()

    # Add text_pos_emb[0..T) — positions are 0,1,2,...,T-1.
    var op = out_buf.unsafe_ptr()
    var pp = model.text_pos_emb.table.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(op, pp, T, D)
    def add_pos_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var ti = i // D
        var di = i - ti * D
        op[i] = op[i] + pp[ti * D + di]
    elementwise[add_pos_fn, simd_width=1, target="gpu"](
        IndexList[1](T * D), DeviceContextPtr(ctx),
    )


def build_bos_emb(
    mut ctx: DeviceContext,
    mut model: T3,
    mut out_buf: DeviceBuffer[DType.float32],   # (1, 1, D)
) raises:
    """speech_emb(START_SPEECH_TOKEN) + speech_pos_emb(0) → (1, 1, D)."""
    var D = model.d_model
    var ids_buf = ctx.enqueue_create_buffer[DType.int64](1)
    with ids_buf.map_to_host() as h:
        h[0] = Int64(START_SPEECH_TOKEN)
    embedding_forward(ctx, model.speech_emb, ids_buf, out_buf, 1, 1)
    ctx.synchronize()

    # Add speech_pos_emb[0].
    var op = out_buf.unsafe_ptr()
    var sp = model.speech_pos_emb.table.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(op, sp, D)
    def add_spk0[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        op[i] = op[i] + sp[i]
    elementwise[add_spk0, simd_width=1, target="gpu"](
        IndexList[1](D), DeviceContextPtr(ctx),
    )


def build_rope_tables(
    mut ctx: DeviceContext,
    max_ctx: Int, head_dim: Int,
    mut cos_buf: DeviceBuffer[DType.float32],   # (max_ctx, head_dim)
    mut sin_buf: DeviceBuffer[DType.float32],   # (max_ctx, head_dim)
) raises:
    """HF Llama RoPE: cos/sin tables for positions [0, max_ctx).

        inv_freq[k] = 1 / 10000^(2k / D)       k in [0, D/2)
        pos[t, k]   = t * inv_freq[k]
        cos/sin tables: cat([cos(pos), cos(pos)], dim=-1) → (max_ctx, D)
    """
    var cp = cos_buf.unsafe_ptr()
    var sp = sin_buf.unsafe_ptr()
    var d_half = head_dim // 2

    @always_inline
    @parameter
    @__copy_capture(cp, sp, max_ctx, head_dim, d_half)
    def rope_fn[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var t = i // head_dim
        var k = i - t * head_dim
        # k in [0, D) — but cos/sin are duplicated halves: position uses k_half = k % d_half.
        var k_half: Int
        if k < d_half:
            k_half = k
        else:
            k_half = k - d_half
        # inv_freq[k_half] = 1 / 10000^(2*k_half / D).
        var exponent: Float32 = (2.0 * Float32(k_half)) / Float32(head_dim)
        # 10000^exponent = exp(exponent * ln(10000)).
        # ln(10000) ≈ 9.210340371976184
        var inv_freq: Float32 = 1.0
        # Use repeated squaring/Taylor — but we have std.math.exp/log via from import.
        # Just inline pow as exp(exponent * 9.210340372).
        from std.math import exp as mexp
        inv_freq = mexp(-exponent * 9.210340371976184)
        var pos: Float32 = Float32(t) * inv_freq
        cp[i] = mcos(pos)
        sp[i] = msin(pos)
    elementwise[rope_fn, simd_width=1, target="gpu"](
        IndexList[1](max_ctx * head_dim), DeviceContextPtr(ctx),
    )
