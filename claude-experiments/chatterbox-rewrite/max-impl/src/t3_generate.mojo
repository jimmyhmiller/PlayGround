"""T3 autoregressive generation: prefill input embedding through 30 layers
(populating KV cache) → loop {decode-step + LM-head + argmax} until EOS.

Uses only validated MAX-abstraction primitives:
  - `t3_block_forward` (prefill prefix)
  - `t3_decode_step` (single-token decode with KV cache)
  - `linear_forward` (LM head)
  - argmax via custom elementwise reduce (no `nn.argmaxmin` direct yet)
  - Embedding via `embedding_forward` for new tokens

The caller supplies a populated `T3` model and an input embedding `(B, T_prefill, D)`.
We return the list of generated speech token IDs.
"""
from std.math import sqrt, exp
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList
from layout import Idx, TileTensor, row_major

from modules import (
    Linear, linear_forward, RMSNorm, rms_norm_forward, Embedding, embedding_forward,
)
from t3_block import T3Block, t3_block_forward, t3_block_prefill
from t3 import T3
from t3_decode import t3_decode_step, cache_write_step
from transformer_blocks import reshape_bsd_to_bhsd, apply_rope_hf_style


def argmax_lastdim(
    mut ctx: DeviceContext,
    mut logits_buf: DeviceBuffer[DType.float32],   # (B, V)
    mut argmax_buf:  DeviceBuffer[DType.int64],     # (B,)
    b: Int, v: Int,
) raises:
    """Compute argmax along the last (V) dim. One thread per row scans V values."""
    var lp = logits_buf.unsafe_ptr()
    var ap = argmax_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(lp, ap, v)
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var bi = idx[0]
        var off = bi * v
        var best_idx: Int = 0
        var best_val: Float32 = lp[off]
        for i in range(1, v):
            var x = lp[off + i]
            if x > best_val:
                best_val = x
                best_idx = i
        ap[bi] = Int64(best_idx)
    elementwise[func, simd_width=1, target="gpu"](
        IndexList[1](b), DeviceContextPtr(ctx),
    )


def gather_cos_sin_at_pos(
    mut ctx: DeviceContext,
    mut cos_full: DeviceBuffer[DType.float32],   # (MAX_CTX, Dh)
    mut sin_full: DeviceBuffer[DType.float32],
    mut cos_step: DeviceBuffer[DType.float32],   # (B, 1, Dh) — output
    mut sin_step: DeviceBuffer[DType.float32],
    b: Int, dh: Int, pos: Int,
) raises:
    """Gather cos/sin at position `pos` from a (MAX_CTX, Dh) table into a
    (B, 1, Dh) per-step buffer (broadcast across batch).
    """
    var csp = cos_full.unsafe_ptr()
    var ssp = sin_full.unsafe_ptr()
    var cop = cos_step.unsafe_ptr()
    var sop = sin_step.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(csp, ssp, cop, sop, dh, pos)
    def func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var di = i % dh
        cop[i] = csp[pos * dh + di]
        sop[i] = ssp[pos * dh + di]
    elementwise[func, simd_width=1, target="gpu"](
        IndexList[1](b * dh), DeviceContextPtr(ctx),
    )


def t3_generate(
    mut ctx: DeviceContext,
    mut model: T3,
    mut input_embed_buf: DeviceBuffer[DType.float32],   # (B, T_prefill, D) — already-built prefix
    mut cos_full: DeviceBuffer[DType.float32],          # (MAX_CTX, Dh) — full RoPE table
    mut sin_full: DeviceBuffer[DType.float32],
    mut prefill_mask: DeviceBuffer[DType.float32],      # (T_prefill, T_prefill) causal bias
    mut speech_pos_emb_table: DeviceBuffer[DType.float32],  # (P_speech, D)
    b: Int, t_prefill: Int, max_ctx: Int, n_steps: Int,
    speech_pos_offset: Int,    # starting position index for speech tokens
    eos_token: Int,
) raises -> List[Int64]:
    """Generate up to `n_steps` speech tokens.

    Approach:
      1. Prefill: run `t3_block_forward` for each of N_LAYERS, populating
         per-layer K/V caches via cache_write_step calls inside.
         (Our current `t3_block_forward` doesn't write to a KV cache directly.
         Instead we rerun the same Q/K/V compute inside the decode step. To
         keep this orchestrator simple and correct, we treat the prefill as
         a one-shot block forward AND a parallel population of the caches
         via the K/V projections from the prefill RoPE'd outputs.)
      2. Sample first token from final hidden state's last position.
      3. Loop decode steps until n_steps or EOS.

    A cleaner architecture would have `t3_block_forward` accept optional
    K/V cache buffers. We add a thin wrapper here that captures Q/K/V from
    each prefill layer; for an MVP we use the existing prefill block and
    rebuild K/V from a second pass through each layer's projections.

    For now this function is **a structural skeleton** that establishes the
    full generation loop wiring. The first-cut implementation rebuilds the
    K/V caches by recomputing projections at each prefill step (this is
    correct but redundant work — fine for an MVP). A follow-up commit
    refactors `t3_block_forward` to emit K/V cache snapshots directly.
    """
    var D = model.d_model
    var H = model.n_heads
    var Dh = model.head_dim
    var V = model.v_speech

    # 1. Prefill via t3_block_prefill — populates x_buf AND per-layer KV caches.
    var x_buf = ctx.enqueue_create_buffer[DType.float32](b * t_prefill * D)
    ctx.enqueue_copy(x_buf, input_embed_buf)

    var k_caches = List[DeviceBuffer[DType.float32]]()
    var v_caches = List[DeviceBuffer[DType.float32]]()
    for _ in range(model.n_layers):
        var kc = ctx.enqueue_create_buffer[DType.float32](b * H * max_ctx * Dh)
        kc.enqueue_fill(0.0)
        var vc = ctx.enqueue_create_buffer[DType.float32](b * H * max_ctx * Dh)
        vc.enqueue_fill(0.0)
        k_caches.append(kc^)
        v_caches.append(vc^)

    # cos_full / sin_full are (MAX_CTX, Dh); for prefill we use slots [0..t_prefill).
    # The t3_block_prefill expects cos/sin matching the seq layout (B, S, Dh).
    # We assume the caller passes cos_full with prefill positions in [0..T) and
    # build a (B, t_prefill, Dh) view by gathering, or trust the caller already
    # supplied (B, T, Dh) shaped buffers. For now we assume `cos_full` is the
    # full table (MAX_CTX, Dh) and `prefill_mask` plus (B, T) layout works.
    # Construct (B, T, Dh) cos/sin for prefill:
    var cos_pre = ctx.enqueue_create_buffer[DType.float32](b * t_prefill * Dh)
    var sin_pre = ctx.enqueue_create_buffer[DType.float32](b * t_prefill * Dh)
    var cfp = cos_full.unsafe_ptr()
    var sfp = sin_full.unsafe_ptr()
    var cpp = cos_pre.unsafe_ptr()
    var spp = sin_pre.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(cfp, sfp, cpp, spp, t_prefill, Dh)
    def gather_pre[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (t_prefill * Dh)
        var rem = i - bi * t_prefill * Dh
        var si = rem // Dh
        var di = rem - si * Dh
        cpp[i] = cfp[si * Dh + di]
        spp[i] = sfp[si * Dh + di]
    elementwise[gather_pre, simd_width=1, target="gpu"](
        IndexList[1](b * t_prefill * Dh), DeviceContextPtr(ctx),
    )

    for L in range(model.n_layers):
        t3_block_prefill(
            ctx, model.blocks[L], x_buf, cos_pre, sin_pre, prefill_mask,
            k_caches[L], v_caches[L], b, t_prefill, max_ctx,
        )

    # 2. Final RMSNorm + LM head + argmax → first generated token.
    var tmp = ctx.enqueue_create_buffer[DType.float32](b * t_prefill * D)
    rms_norm_forward(ctx, model.final_norm, x_buf, tmp, b * t_prefill)
    ctx.enqueue_copy(x_buf, tmp)

    # Take last position only: (B, D). We index by computing the offset.
    var last_hidden = ctx.enqueue_create_buffer[DType.float32](b * D)
    var xp = x_buf.unsafe_ptr()
    var lhp = last_hidden.unsafe_ptr()
    var tpf = t_prefill

    @always_inline
    @parameter
    @__copy_capture(xp, lhp, D, tpf)
    def take_last[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // D
        var di = i - bi * D
        lhp[i] = xp[bi * tpf * D + (tpf - 1) * D + di]
    elementwise[take_last, simd_width=1, target="gpu"](
        IndexList[1](b * D), DeviceContextPtr(ctx),
    )

    var logits_buf = ctx.enqueue_create_buffer[DType.float32](b * V)
    linear_forward(ctx, model.speech_head, last_hidden, logits_buf, b)
    var argmax_buf = ctx.enqueue_create_buffer[DType.int64](b)
    argmax_lastdim(ctx, logits_buf, argmax_buf, b, V)
    ctx.synchronize()

    var generated = List[Int64]()
    var first_tok: Int64 = 0
    with argmax_buf.map_to_host() as h:
        first_tok = h[0]
    generated.append(first_tok)

    if first_tok == Int64(eos_token):
        return generated^

    # 3. Decode loop.
    var cur_len = t_prefill
    var cur_tok_buf = ctx.enqueue_create_buffer[DType.int64](b)
    with cur_tok_buf.map_to_host() as h:
        h[0] = first_tok
    var step_emb_buf = ctx.enqueue_create_buffer[DType.float32](b * D)
    var step_emb_3d_buf = ctx.enqueue_create_buffer[DType.float32](b * 1 * D)
    var cos_step = ctx.enqueue_create_buffer[DType.float32](b * Dh)
    var sin_step = ctx.enqueue_create_buffer[DType.float32](b * Dh)

    var step = 1
    while step < n_steps:
        embedding_forward(ctx, model.speech_emb, cur_tok_buf, step_emb_3d_buf, b, 1)
        ctx.enqueue_copy(step_emb_buf, step_emb_3d_buf)

        var pos_idx = speech_pos_offset + step - 1
        var sep = step_emb_buf.unsafe_ptr()
        var spt = speech_pos_emb_table.unsafe_ptr()

        @always_inline
        @parameter
        @__copy_capture(sep, spt, D, pos_idx)
        def add_pos[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
            var i = idx[0]
            var bi = i // D
            var di = i - bi * D
            sep[i] = sep[i] + spt[pos_idx * D + di]
        elementwise[add_pos, simd_width=1, target="gpu"](
            IndexList[1](b * D), DeviceContextPtr(ctx),
        )

        gather_cos_sin_at_pos(ctx, cos_full, sin_full, cos_step, sin_step,
                              b, Dh, cur_len)

        for L in range(model.n_layers):
            t3_decode_step(
                ctx, model.blocks[L], step_emb_buf,
                k_caches[L], v_caches[L], cos_step, sin_step,
                b, max_ctx, cur_len,
            )

        var tmp2 = ctx.enqueue_create_buffer[DType.float32](b * D)
        rms_norm_forward(ctx, model.final_norm, step_emb_buf, tmp2, b)
        ctx.enqueue_copy(step_emb_buf, tmp2)

        linear_forward(ctx, model.speech_head, step_emb_buf, logits_buf, b)
        argmax_lastdim(ctx, logits_buf, argmax_buf, b, V)
        ctx.synchronize()

        var next_tok: Int64 = 0
        with argmax_buf.map_to_host() as h:
            next_tok = h[0]
        generated.append(next_tok)
        if next_tok == Int64(eos_token):
            break

        with cur_tok_buf.map_to_host() as h:
            h[0] = next_tok
        cur_len += 1
        step += 1

    return generated^


def cfg_combine_sample(
    mut ctx: DeviceContext,
    mut logits_buf: DeviceBuffer[DType.float32],   # (2*b, V)
    history: List[Int64],                          # all previously generated tokens
    mut rng_state: UInt64,
    b: Int, v: Int, cfg: Float32,
    temperature: Float32,
    top_p: Float32,
    rep_penalty: Float32,
) raises -> Int64:
    """CPU-side: pull logits, CFG-combine, apply repetition penalty + temperature +
    top-p, sample multinomially via LCG. Returns (token, next_rng_state).

    Matches upstream HF generate sampling pipeline:
        cond, uncond = split(logits)
        logits = cond + cfg * (cond - uncond)
        for tok in history: logits[tok] /= rep_penalty if logits[tok]>0 else logits[tok] *= rep_penalty
        logits = logits / temperature
        sorted_idx, sorted_logits = sort_desc(logits)
        cumprob = cumsum(softmax(sorted_logits))
        mask = cumprob > top_p; keep first index
        masked_logits = sorted_logits with mask applied (set to -inf)
        probs = softmax(masked_logits)
        sample from categorical
    """
    var lp = logits_buf.unsafe_ptr()

    # Pull only the first row of cond and uncond — b=1 in our path.
    if b != 1:
        raise Error("cfg_combine_sample: only b=1 supported")

    var logits = List[Float32](capacity=v)
    with logits_buf.map_to_host() as host:
        for k in range(v):
            var cond_v = host[k]
            var unc_v = host[v + k]
            logits.append(cond_v + cfg * (cond_v - unc_v))

    # Repetition penalty: for each token in history, if logit>0 divide else multiply.
    for k in range(len(history)):
        var idx = Int(history[k])
        if idx < 0 or idx >= v: continue
        var l = logits[idx]
        if l > 0.0:
            logits[idx] = l / rep_penalty
        else:
            logits[idx] = l * rep_penalty

    # Temperature scaling.
    if temperature != 1.0:
        for k in range(v):
            logits[k] = logits[k] / temperature

    # Top-p filtering: sort logits desc, compute softmax cumsum, keep first idx
    # where cumsum > top_p (i.e., include everything up to and including the
    # first token that pushes cumprob past top_p).
    var idx_arr = List[Int](capacity=v)
    for k in range(v):
        idx_arr.append(k)

    # Simple sort (since V=6562 is small enough): use selection-sort O(V*K) where
    # K is the typical top-p size (~10-50). Find top K such that sum-prob >= top_p.
    # Faster path: just sort everything via insertion-sort? V=6562 is too big.
    # Use partial sort: repeatedly find max until cumsum reaches top_p.
    var max_logit: Float32 = -1.0e30
    for k in range(v):
        if logits[k] > max_logit: max_logit = logits[k]

    # Compute softmax with max-subtract for numerical stability.
    var exp_logits = List[Float32](capacity=v)
    var exp_sum: Float32 = 0.0
    for k in range(v):
        var e = Float32(0.0)
        var diff = logits[k] - max_logit
        if diff > -50.0:
            e = exp(diff)
        exp_logits.append(e)
        exp_sum += e
    if exp_sum <= 0.0: exp_sum = 1.0

    # Partial sort: find tokens in descending probability order until cumprob >= top_p.
    var kept_idx = List[Int]()
    var kept_logit = List[Float32]()
    var seen = List[Bool](capacity=v)
    for k in range(v):
        seen.append(False)
    var cumprob: Float32 = 0.0
    var safety = 0
    while cumprob < top_p and safety < 500:
        var best_p: Float32 = -1.0
        var best_k: Int = -1
        for k in range(v):
            if seen[k]: continue
            var p = exp_logits[k] / exp_sum
            if p > best_p:
                best_p = p
                best_k = k
        if best_k < 0: break
        seen[best_k] = True
        kept_idx.append(best_k)
        kept_logit.append(logits[best_k])
        cumprob += best_p
        safety += 1

    # Renormalize over kept set: softmax(kept_logit) with new max.
    var k_max: Float32 = -1.0e30
    for k in range(len(kept_logit)):
        if kept_logit[k] > k_max: k_max = kept_logit[k]
    var k_exp = List[Float32](capacity=len(kept_logit))
    var k_sum: Float32 = 0.0
    for k in range(len(kept_logit)):
        var e = exp(kept_logit[k] - k_max)
        k_exp.append(e)
        k_sum += e

    # LCG-uniform sample.
    rng_state = rng_state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
    var bits: UInt64 = (rng_state >> UInt64(40)) & UInt64(0xFFFFFF)
    var u: Float32 = (Float32(Int(bits)) + 0.5) / Float32(16777216.0)
    var target = u * k_sum

    var pick: Int = len(kept_idx) - 1
    var run: Float32 = 0.0
    for k in range(len(kept_idx)):
        run += k_exp[k]
        if run >= target:
            pick = k
            break

    return Int64(kept_idx[pick])


def cfg_combine_argmax(
    mut ctx: DeviceContext,
    mut logits_buf: DeviceBuffer[DType.float32],   # (2*b, V)
    mut argmax_buf: DeviceBuffer[DType.int64],     # (b,)
    b: Int, v: Int, cfg: Float32,
) raises:
    """Combine doubled-batch logits via CFG:
       cond   = logits[0:b]
       uncond = logits[b:2b]
       out    = cond + cfg * (cond - uncond)
    Then argmax along V.
    """
    var lp = logits_buf.unsafe_ptr()
    var ap = argmax_buf.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(lp, ap, b, v, cfg)
    def cf[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var bi = idx[0]
        var best_val: Float32 = -1.0e30
        var best_idx: Int64 = 0
        for k in range(v):
            var cond_v = lp[bi * v + k]
            var unc_v = lp[(b + bi) * v + k]
            var combined = cond_v + cfg * (cond_v - unc_v)
            if combined > best_val:
                best_val = combined
                best_idx = Int64(k)
        ap[bi] = best_idx
    elementwise[cf, simd_width=1, target="gpu"](
        IndexList[1](b), DeviceContextPtr(ctx),
    )


def t3_generate_cfg(
    mut ctx: DeviceContext,
    mut model: T3,
    mut input_embed_buf: DeviceBuffer[DType.float32],   # (2*B, T_prefill, D) — already CFG-doubled
    mut cos_full: DeviceBuffer[DType.float32],
    mut sin_full: DeviceBuffer[DType.float32],
    mut prefill_mask: DeviceBuffer[DType.float32],      # (T_prefill, T_prefill) causal bias
    mut speech_pos_emb_table: DeviceBuffer[DType.float32],
    b: Int, t_prefill: Int, max_ctx: Int, n_steps: Int,
    speech_pos_offset: Int,
    eos_token: Int,
    cfg_weight: Float32 = 0.5,
) raises -> List[Int64]:
    """Generate speech tokens with classifier-free guidance.

    Caller passes `input_embed_buf` of shape (2*B, T_prefill, D) where:
      rows [0..B):   conditional prefix (cond_emb + text_emb + bos_emb)
      rows [B..2B):  unconditional prefix (cond_emb + zero_text + bos_emb)

    At each step the model runs at effective batch 2*B. Logits are CFG-combined
    (cond + cfg * (cond - uncond)) before argmax. The same chosen token is fed
    back to both branches for the next step.
    """
    var D = model.d_model
    var H = model.n_heads
    var Dh = model.head_dim
    var V = model.v_speech
    var B2 = 2 * b

    # 1. Prefill at 2*B.
    var x_buf = ctx.enqueue_create_buffer[DType.float32](B2 * t_prefill * D)
    ctx.enqueue_copy(x_buf, input_embed_buf)

    var k_caches = List[DeviceBuffer[DType.float32]]()
    var v_caches = List[DeviceBuffer[DType.float32]]()
    for _ in range(model.n_layers):
        var kc = ctx.enqueue_create_buffer[DType.float32](B2 * H * max_ctx * Dh)
        kc.enqueue_fill(0.0)
        var vc = ctx.enqueue_create_buffer[DType.float32](B2 * H * max_ctx * Dh)
        vc.enqueue_fill(0.0)
        k_caches.append(kc^)
        v_caches.append(vc^)

    # Gather (B2, T, Dh) cos/sin from full table.
    var cos_pre = ctx.enqueue_create_buffer[DType.float32](B2 * t_prefill * Dh)
    var sin_pre = ctx.enqueue_create_buffer[DType.float32](B2 * t_prefill * Dh)
    var cfp = cos_full.unsafe_ptr()
    var sfp = sin_full.unsafe_ptr()
    var cpp = cos_pre.unsafe_ptr()
    var spp = sin_pre.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(cfp, sfp, cpp, spp, t_prefill, Dh)
    def gather_pre[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (t_prefill * Dh)
        var rem = i - bi * t_prefill * Dh
        var si = rem // Dh
        var di = rem - si * Dh
        cpp[i] = cfp[si * Dh + di]
        spp[i] = sfp[si * Dh + di]
    elementwise[gather_pre, simd_width=1, target="gpu"](
        IndexList[1](B2 * t_prefill * Dh), DeviceContextPtr(ctx),
    )

    for L in range(model.n_layers):
        t3_block_prefill(
            ctx, model.blocks[L], x_buf, cos_pre, sin_pre, prefill_mask,
            k_caches[L], v_caches[L], B2, t_prefill, max_ctx,
        )

    # 2. Final RMSNorm + LM head on last position of each row → (2B, V).
    var tmp = ctx.enqueue_create_buffer[DType.float32](B2 * t_prefill * D)
    rms_norm_forward(ctx, model.final_norm, x_buf, tmp, B2 * t_prefill)
    ctx.enqueue_copy(x_buf, tmp)

    var last_hidden = ctx.enqueue_create_buffer[DType.float32](B2 * D)
    var xp = x_buf.unsafe_ptr()
    var lhp = last_hidden.unsafe_ptr()
    var tpf = t_prefill

    @always_inline
    @parameter
    @__copy_capture(xp, lhp, D, tpf)
    def take_last[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // D
        var di = i - bi * D
        lhp[i] = xp[bi * tpf * D + (tpf - 1) * D + di]
    elementwise[take_last, simd_width=1, target="gpu"](
        IndexList[1](B2 * D), DeviceContextPtr(ctx),
    )

    var logits_buf = ctx.enqueue_create_buffer[DType.float32](B2 * V)
    linear_forward(ctx, model.speech_head, last_hidden, logits_buf, B2)
    var argmax_buf = ctx.enqueue_create_buffer[DType.int64](b)
    cfg_combine_argmax(ctx, logits_buf, argmax_buf, b, V, cfg_weight)
    ctx.synchronize()

    var generated = List[Int64]()
    var first_tok: Int64 = 0
    with argmax_buf.map_to_host() as h:
        first_tok = h[0]
    generated.append(first_tok)
    if first_tok == Int64(eos_token):
        return generated^

    # 3. Decode loop at 2*B.
    var cur_len = t_prefill
    var cur_tok_buf = ctx.enqueue_create_buffer[DType.int64](B2)
    with cur_tok_buf.map_to_host() as h:
        for bi in range(B2):
            h[bi] = first_tok       # both branches embed same token
    var step_emb_buf = ctx.enqueue_create_buffer[DType.float32](B2 * D)
    var step_emb_3d_buf = ctx.enqueue_create_buffer[DType.float32](B2 * 1 * D)
    var cos_step = ctx.enqueue_create_buffer[DType.float32](B2 * Dh)
    var sin_step = ctx.enqueue_create_buffer[DType.float32](B2 * Dh)

    var step = 1
    while step < n_steps:
        embedding_forward(ctx, model.speech_emb, cur_tok_buf, step_emb_3d_buf, B2, 1)
        ctx.enqueue_copy(step_emb_buf, step_emb_3d_buf)

        var pos_idx = speech_pos_offset + step - 1
        var sep = step_emb_buf.unsafe_ptr()
        var spt = speech_pos_emb_table.unsafe_ptr()

        @always_inline
        @parameter
        @__copy_capture(sep, spt, D, pos_idx)
        def add_pos[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
            var i = idx[0]
            var bi = i // D
            var di = i - bi * D
            sep[i] = sep[i] + spt[pos_idx * D + di]
        elementwise[add_pos, simd_width=1, target="gpu"](
            IndexList[1](B2 * D), DeviceContextPtr(ctx),
        )

        gather_cos_sin_at_pos(ctx, cos_full, sin_full, cos_step, sin_step,
                              B2, Dh, cur_len)

        for L in range(model.n_layers):
            t3_decode_step(
                ctx, model.blocks[L], step_emb_buf,
                k_caches[L], v_caches[L], cos_step, sin_step,
                B2, max_ctx, cur_len,
            )

        var tmp2 = ctx.enqueue_create_buffer[DType.float32](B2 * D)
        rms_norm_forward(ctx, model.final_norm, step_emb_buf, tmp2, B2)
        ctx.enqueue_copy(step_emb_buf, tmp2)

        linear_forward(ctx, model.speech_head, step_emb_buf, logits_buf, B2)
        cfg_combine_argmax(ctx, logits_buf, argmax_buf, b, V, cfg_weight)
        ctx.synchronize()

        var next_tok: Int64 = 0
        with argmax_buf.map_to_host() as h:
            next_tok = h[0]
        generated.append(next_tok)
        if next_tok == Int64(eos_token):
            break

        with cur_tok_buf.map_to_host() as h:
            for bi in range(B2):
                h[bi] = next_tok
        cur_len += 1
        step += 1

    return generated^


def t3_generate_cfg_sample(
    mut ctx: DeviceContext,
    mut model: T3,
    mut input_embed_buf: DeviceBuffer[DType.float32],
    mut cos_full: DeviceBuffer[DType.float32],
    mut sin_full: DeviceBuffer[DType.float32],
    mut prefill_mask: DeviceBuffer[DType.float32],
    mut speech_pos_emb_table: DeviceBuffer[DType.float32],
    b: Int, t_prefill: Int, max_ctx: Int, n_steps: Int,
    speech_pos_offset: Int,
    eos_token: Int,
    cfg_weight: Float32 = 0.5,
    temperature: Float32 = 0.8,
    top_p: Float32 = 0.95,
    rep_penalty: Float32 = 1.2,
    rng_seed: UInt64 = UInt64(0xDEADBEEF),
) raises -> List[Int64]:
    """Like t3_generate_cfg but uses temperature + top-p sampling instead of argmax.
    Matches upstream HF generate sampling chain.
    """
    var D = model.d_model
    var H = model.n_heads
    var Dh = model.head_dim
    var V = model.v_speech
    var B2 = 2 * b

    if b != 1:
        raise Error("t3_generate_cfg_sample: only b=1 supported")

    var x_buf = ctx.enqueue_create_buffer[DType.float32](B2 * t_prefill * D)
    ctx.enqueue_copy(x_buf, input_embed_buf)

    var k_caches = List[DeviceBuffer[DType.float32]]()
    var v_caches = List[DeviceBuffer[DType.float32]]()
    for _ in range(model.n_layers):
        var kc = ctx.enqueue_create_buffer[DType.float32](B2 * H * max_ctx * Dh)
        kc.enqueue_fill(0.0)
        var vc = ctx.enqueue_create_buffer[DType.float32](B2 * H * max_ctx * Dh)
        vc.enqueue_fill(0.0)
        k_caches.append(kc^)
        v_caches.append(vc^)

    var cos_pre = ctx.enqueue_create_buffer[DType.float32](B2 * t_prefill * Dh)
    var sin_pre = ctx.enqueue_create_buffer[DType.float32](B2 * t_prefill * Dh)
    var cfp = cos_full.unsafe_ptr()
    var sfp = sin_full.unsafe_ptr()
    var cpp = cos_pre.unsafe_ptr()
    var spp = sin_pre.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(cfp, sfp, cpp, spp, t_prefill, Dh)
    def gather_pre[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (t_prefill * Dh)
        var rem = i - bi * t_prefill * Dh
        var si = rem // Dh
        var di = rem - si * Dh
        cpp[i] = cfp[si * Dh + di]
        spp[i] = sfp[si * Dh + di]
    elementwise[gather_pre, simd_width=1, target="gpu"](
        IndexList[1](B2 * t_prefill * Dh), DeviceContextPtr(ctx),
    )

    for L in range(model.n_layers):
        t3_block_prefill(
            ctx, model.blocks[L], x_buf, cos_pre, sin_pre, prefill_mask,
            k_caches[L], v_caches[L], B2, t_prefill, max_ctx,
        )

    var tmp = ctx.enqueue_create_buffer[DType.float32](B2 * t_prefill * D)
    rms_norm_forward(ctx, model.final_norm, x_buf, tmp, B2 * t_prefill)
    ctx.enqueue_copy(x_buf, tmp)

    var last_hidden = ctx.enqueue_create_buffer[DType.float32](B2 * D)
    var xp = x_buf.unsafe_ptr()
    var lhp = last_hidden.unsafe_ptr()
    var tpf = t_prefill

    @always_inline
    @parameter
    @__copy_capture(xp, lhp, D, tpf)
    def take_last[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // D
        var di = i - bi * D
        lhp[i] = xp[bi * tpf * D + (tpf - 1) * D + di]
    elementwise[take_last, simd_width=1, target="gpu"](
        IndexList[1](B2 * D), DeviceContextPtr(ctx),
    )

    var logits_buf = ctx.enqueue_create_buffer[DType.float32](B2 * V)
    linear_forward(ctx, model.speech_head, last_hidden, logits_buf, B2)
    ctx.synchronize()

    var generated = List[Int64]()
    var rng_state: UInt64 = rng_seed
    var first_tok = cfg_combine_sample(
        ctx, logits_buf, generated, rng_state, b, V, cfg_weight,
        temperature, top_p, rep_penalty,
    )
    generated.append(first_tok)
    if first_tok == Int64(eos_token):
        return generated^

    var cur_len = t_prefill
    var cur_tok_buf = ctx.enqueue_create_buffer[DType.int64](B2)
    with cur_tok_buf.map_to_host() as h:
        for bi in range(B2):
            h[bi] = first_tok
    var step_emb_buf = ctx.enqueue_create_buffer[DType.float32](B2 * D)
    var step_emb_3d_buf = ctx.enqueue_create_buffer[DType.float32](B2 * 1 * D)
    var cos_step = ctx.enqueue_create_buffer[DType.float32](B2 * Dh)
    var sin_step = ctx.enqueue_create_buffer[DType.float32](B2 * Dh)

    var step = 1
    while step < n_steps:
        embedding_forward(ctx, model.speech_emb, cur_tok_buf, step_emb_3d_buf, B2, 1)
        ctx.enqueue_copy(step_emb_buf, step_emb_3d_buf)

        var pos_idx = speech_pos_offset + step - 1
        var sep = step_emb_buf.unsafe_ptr()
        var spt = speech_pos_emb_table.unsafe_ptr()

        @always_inline
        @parameter
        @__copy_capture(sep, spt, D, pos_idx)
        def add_pos[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
            var i = idx[0]
            var bi = i // D
            var di = i - bi * D
            sep[i] = sep[i] + spt[pos_idx * D + di]
        elementwise[add_pos, simd_width=1, target="gpu"](
            IndexList[1](B2 * D), DeviceContextPtr(ctx),
        )

        gather_cos_sin_at_pos(ctx, cos_full, sin_full, cos_step, sin_step,
                              B2, Dh, cur_len)

        for L in range(model.n_layers):
            t3_decode_step(
                ctx, model.blocks[L], step_emb_buf,
                k_caches[L], v_caches[L], cos_step, sin_step,
                B2, max_ctx, cur_len,
            )

        var tmp2 = ctx.enqueue_create_buffer[DType.float32](B2 * D)
        rms_norm_forward(ctx, model.final_norm, step_emb_buf, tmp2, B2)
        ctx.enqueue_copy(step_emb_buf, tmp2)

        linear_forward(ctx, model.speech_head, step_emb_buf, logits_buf, B2)
        ctx.synchronize()

        var next_tok = cfg_combine_sample(
            ctx, logits_buf, generated, rng_state, b, V, cfg_weight,
            temperature, top_p, rep_penalty,
        )
        generated.append(next_tok)
        if next_tok == Int64(eos_token):
            break

        with cur_tok_buf.map_to_host() as h:
            for bi in range(B2):
                h[bi] = next_tok
        cur_len += 1
        step += 1

    return generated^
