"""T3 (Llama-30L) backbone forward + multi-step decode loop.

Composed strictly from `nn.normalization.rms_norm`, `linalg.matmul`,
`linalg.bmm.batched_matmul`, `nn.softmax.softmax`, `nn.gather_scatter.gather`,
`nn.argmaxmin.argmax`, and `elementwise[..., target="gpu"]`.

KV cache is plain GPU buffers; we keep K and V caches per-layer and append
to them per timestep.
"""
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from modules import (
    Linear, linear_forward,
    RMSNorm, rms_norm_forward,
    Embedding, embedding_forward,
)
from t3_block import T3Block, t3_block_forward


@fieldwise_init
struct T3(Copyable, Movable):
    """T3 30-layer Llama backbone with separate text & speech embeddings."""
    var blocks: List[T3Block]
    var final_norm: RMSNorm
    var speech_emb: Embedding
    var speech_head: Linear        # (V_speech, D) → logits matmul (we use x @ W.T)
    var text_emb: Embedding        # (V_text=704, D) text token embedding
    var text_pos_emb: Embedding    # (max_text_seq_len, D) positional embedding
    var speech_pos_emb: Embedding  # (max_speech_seq_len, D) speech positional embedding
    var n_layers: Int
    var n_heads: Int
    var head_dim: Int
    var d_model: Int               # = n_heads * head_dim
    var v_speech: Int
    var v_text: Int


def t3_prefill_forward(
    mut ctx: DeviceContext,
    mut model: T3,
    mut x_buf: DeviceBuffer[DType.float32],     # (B, T, D) input embedding (already produced)
    mut cos_buf: DeviceBuffer[DType.float32],
    mut sin_buf: DeviceBuffer[DType.float32],
    mut mask_buf: DeviceBuffer[DType.float32],  # (T, T) causal bias
    b: Int, t: Int,
) raises:
    """Run all 30 layers as a single-shot prefill. Updates x_buf in-place.

    KV cache population is NOT done here yet — see `t3_prefill_with_kv` for the
    cache variant. This version is suitable for verifying block correctness.
    """
    for i in range(model.n_layers):
        t3_block_forward(
            ctx, model.blocks[i], x_buf, cos_buf, sin_buf, mask_buf,
            b, t, has_mask=True,
        )

    # Final RMSNorm in-place: alloc temp, run, copy back.
    var D = model.d_model
    var tmp = ctx.enqueue_create_buffer[DType.float32](b * t * D)
    rms_norm_forward(ctx, model.final_norm, x_buf, tmp, b * t)
    ctx.enqueue_copy(x_buf, tmp)
