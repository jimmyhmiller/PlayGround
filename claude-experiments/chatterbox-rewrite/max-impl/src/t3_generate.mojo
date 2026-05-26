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
from t3_block import T3Block, t3_block_forward, t3_block_prefill, t3_block_prefill_with_attn
from t3 import T3
from t3_decode import t3_decode_step, t3_decode_step_with_attn, cache_write_step
from transformer_blocks import reshape_bsd_to_bhsd, apply_rope_hf_style
from alignment_analyzer import AlignmentAnalyzer, make_alignment_analyzer, aa_step
from std.io.file import open


def _dump_x_buf_cond(path: String, ctx: DeviceContext,
                     x_buf: DeviceBuffer[DType.float32],
                     b2: Int, s: Int, d: Int):
    """Dump only the cond batch (bi=0) of x_buf: (b2, s, d) → (s, d)."""
    try:
        var f = open(path, "w")
        var n = s * d
        with x_buf.map_to_host() as h:
            var buf = List[UInt8](capacity=n * 4)
            for k in range(n):
                var v = h[k]   # bi=0 contiguous prefix
                var p = UnsafePointer(to=v).bitcast[UInt32]()
                var bits = p[0]
                for bb in range(4):
                    buf.append(UInt8(Int((bits >> UInt32(8 * bb)) & 0xFF)))
            f.write_bytes(Span(buf))
        f.close()
    except e:
        print("dump x_buf failed:", e)


def _dump_logits_f32(path: String, data: List[Float32], n: Int):
    """Write n Float32 values as raw little-endian bytes."""
    try:
        var f = open(path, "w")
        var i = 0
        while i < n:
            var chunk_end = min(i + 1024, n)
            var buf = List[UInt8](capacity=(chunk_end - i) * 4)
            for j in range(i, chunk_end):
                var v = data[j]
                var p = UnsafePointer(to=v).bitcast[UInt32]()
                var bits = p[0]
                for k in range(4):
                    buf.append(UInt8(Int((bits >> UInt32(8 * k)) & 0xFF)))
            f.write_bytes(Span(buf))
            i = chunk_end
        f.close()
    except e:
        print("dump failed:", e)


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
    min_p: Float32 = 0.0,
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

    # Match upstream HF order: min_p first (on full vocab), then top_p (on full vocab).
    # Both prune by setting logits to -inf, then softmax+multinomial samples from
    # whatever survives.

    # Compute full-vocab softmax (for min_p threshold comparison and top_p ordering).
    var max_logit: Float32 = -1.0e30
    for k in range(v):
        if logits[k] > max_logit: max_logit = logits[k]
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

    # min_p filter on FULL vocab: HF MinPLogitsWarper keeps tokens where
    # prob >= min_p * top_prob. Always keep top-1.
    var min_p_thresh: Float32 = 0.0
    if min_p > 0.0:
        # Top prob over full vocab = exp(max-max)/exp_sum = 1/exp_sum
        min_p_thresh = min_p * (Float32(1.0) / exp_sum)

    # Walk all V tokens in descending probability order, stop when cumprob > top_p
    # (matches HF TopPLogitsWarper which keeps the first token that exceeds top_p too).
    var kept_idx = List[Int]()
    var kept_logit = List[Float32]()
    var seen = List[Bool](capacity=v)
    for k in range(v):
        seen.append(False)
    var cumprob: Float32 = 0.0
    var top_p_done = False
    # Safety cap: full vocab — but we break early under top_p.
    for _i in range(v):
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

        # If top_p threshold already exceeded, skip remaining unless min_p forces keep.
        # Actually HF order: min_p first, then top_p. We're doing them jointly here:
        # always keep top-1; otherwise keep iff (NOT top_p_done) AND (p >= min_p_thresh).
        var should_keep = False
        if len(kept_idx) == 0:
            should_keep = True   # always keep top-1
        else:
            # min_p check: must be >= threshold
            var meets_min_p = (min_p_thresh == 0.0) or (best_p >= min_p_thresh)
            # top_p: keep while cumprob hadn't yet exceeded (and include the first one that does)
            should_keep = meets_min_p and (not top_p_done)

        if should_keep:
            kept_idx.append(best_k)
            kept_logit.append(logits[best_k])
            cumprob += best_p
            if cumprob >= top_p:
                top_p_done = True
        else:
            # Could keep going for min_p but they're already pruned by top_p — stop.
            if top_p_done:
                break

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


def cfg_combine_sample_with_analyzer(
    mut ctx: DeviceContext,
    mut logits_buf: DeviceBuffer[DType.float32],
    history: List[Int64],
    mut rng_state: UInt64,
    mut analyzer: AlignmentAnalyzer,
    aligned_row: List[Float32],
    last_token: Int64,
    b: Int, v: Int, cfg: Float32,
    temperature: Float32,
    top_p: Float32,
    rep_penalty: Float32,
    min_p: Float32 = 0.0,
) raises -> Int64:
    """Same as cfg_combine_sample but runs the AlignmentStreamAnalyzer between
    the CFG combine and the warpers (matching upstream order in
    chatterbox/models/t3/t3.py)."""
    var lp = logits_buf.unsafe_ptr()
    if b != 1:
        raise Error("cfg_combine_sample_with_analyzer: only b=1 supported")

    # 1. CFG combine.
    var logits = List[Float32](capacity=v)
    with logits_buf.map_to_host() as host:
        for k in range(v):
            var cond_v = host[k]
            var unc_v = host[v + k]
            logits.append(cond_v + cfg * (cond_v - unc_v))

    # 2. Alignment analyzer (may zero out EOS or force-emit EOS).
    aa_step(analyzer, aligned_row, logits, last_token)

    # 3. Repetition penalty.
    for k in range(len(history)):
        var idx = Int(history[k])
        if idx < 0 or idx >= v: continue
        var l = logits[idx]
        if l > 0.0:
            logits[idx] = l / rep_penalty
        else:
            logits[idx] = l * rep_penalty

    # 4. Temperature.
    if temperature != 1.0:
        for k in range(v):
            logits[k] = logits[k] / temperature

    # 5. top_p / min_p filtering — partial-sort + softmax + filter, then sample.
    var max_logit: Float32 = -1.0e30
    for k in range(v):
        if logits[k] > max_logit: max_logit = logits[k]
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

    # min_p
    if min_p > 0.0 and len(kept_idx) > 1:
        var top_p_full = exp_logits[kept_idx[0]] / exp_sum
        var threshold = min_p * top_p_full
        var filtered_idx = List[Int]()
        var filtered_logit = List[Float32]()
        filtered_idx.append(kept_idx[0])
        filtered_logit.append(kept_logit[0])
        for k in range(1, len(kept_idx)):
            var p_k = exp_logits[kept_idx[k]] / exp_sum
            if p_k >= threshold:
                filtered_idx.append(kept_idx[k])
                filtered_logit.append(kept_logit[k])
        kept_idx = filtered_idx^
        kept_logit = filtered_logit^

    # Renormalize and sample.
    var k_max: Float32 = -1.0e30
    for k in range(len(kept_logit)):
        if kept_logit[k] > k_max: k_max = kept_logit[k]
    var k_exp = List[Float32](capacity=len(kept_logit))
    var k_sum: Float32 = 0.0
    for k in range(len(kept_logit)):
        var e = exp(kept_logit[k] - k_max)
        k_exp.append(e)
        k_sum += e

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
    min_p: Float32 = 0.0,
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
        temperature, top_p, rep_penalty, min_p,
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
            temperature, top_p, rep_penalty, min_p,
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


# ── Aligned variant with AlignmentStreamAnalyzer ───────────────────────────

# Hardcoded from upstream chatterbox: LLAMA_ALIGNED_HEADS = [(12, 15), (13, 11), (9, 2)]
comptime _ALIGN_LAYER_0 = 9
comptime _ALIGN_HEAD_0  = 2
comptime _ALIGN_LAYER_1 = 12
comptime _ALIGN_HEAD_1  = 15
comptime _ALIGN_LAYER_2 = 13
comptime _ALIGN_HEAD_2  = 11


def _extract_aligned_row_prefill(
    ctx: DeviceContext,
    attn0: DeviceBuffer[DType.float32],   # (B2, H, S, S)
    attn1: DeviceBuffer[DType.float32],
    attn2: DeviceBuffer[DType.float32],
    b2: Int, H: Int, s: Int,
    i: Int, j: Int,
) raises -> List[Float32]:
    var t_text = j - i
    var out = List[Float32](capacity=t_text)
    # cond batch (bi=0); off = hi*S*S + (s-1)*S + ki
    with attn0.map_to_host() as a0:
        with attn1.map_to_host() as a1:
            with attn2.map_to_host() as a2:
                for ki in range(i, j):
                    var off0 = _ALIGN_HEAD_0 * s * s + (s - 1) * s + ki
                    var off1 = _ALIGN_HEAD_1 * s * s + (s - 1) * s + ki
                    var off2 = _ALIGN_HEAD_2 * s * s + (s - 1) * s + ki
                    var v = (a0[off0] + a1[off1] + a2[off2]) / Float32(3.0)
                    out.append(v)
    return out^


def _extract_aligned_row_decode(
    ctx: DeviceContext,
    attn0: DeviceBuffer[DType.float32],   # (B2, H, 1, s_k)
    attn1: DeviceBuffer[DType.float32],
    attn2: DeviceBuffer[DType.float32],
    b2: Int, H: Int, s_k: Int,
    i: Int, j: Int,
) raises -> List[Float32]:
    var t_text = j - i
    var out = List[Float32](capacity=t_text)
    with attn0.map_to_host() as a0:
        with attn1.map_to_host() as a1:
            with attn2.map_to_host() as a2:
                for ki in range(i, j):
                    var off0 = _ALIGN_HEAD_0 * s_k + ki
                    var off1 = _ALIGN_HEAD_1 * s_k + ki
                    var off2 = _ALIGN_HEAD_2 * s_k + ki
                    var v = (a0[off0] + a1[off1] + a2[off2]) / Float32(3.0)
                    out.append(v)
    return out^


def t3_generate_cfg_sample_aligned(
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
    text_slice_i: Int,
    text_slice_j: Int,
    cfg_weight: Float32 = 0.5,
    temperature: Float32 = 0.8,
    top_p: Float32 = 0.95,
    rep_penalty: Float32 = 1.2,
    rng_seed: UInt64 = UInt64(0xDEADBEEF),
    min_p: Float32 = 0.0,
) raises -> List[Int64]:
    """t3_generate_cfg_sample + AlignmentStreamAnalyzer."""
    var D = model.d_model
    var H = model.n_heads
    var Dh = model.head_dim
    var V = model.v_speech
    var B2 = 2 * b

    if b != 1:
        raise Error("t3_generate_cfg_sample_aligned: only b=1 supported")

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
    def gather_pre_a[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (t_prefill * Dh)
        var rem = i - bi * t_prefill * Dh
        var si = rem // Dh
        var di = rem - si * Dh
        cpp[i] = cfp[si * Dh + di]
        spp[i] = sfp[si * Dh + di]
    elementwise[gather_pre_a, simd_width=1, target="gpu"](
        IndexList[1](B2 * t_prefill * Dh), DeviceContextPtr(ctx),
    )

    var attn_pre_0 = ctx.enqueue_create_buffer[DType.float32](B2 * H * t_prefill * t_prefill)
    var attn_pre_1 = ctx.enqueue_create_buffer[DType.float32](B2 * H * t_prefill * t_prefill)
    var attn_pre_2 = ctx.enqueue_create_buffer[DType.float32](B2 * H * t_prefill * t_prefill)

    for L in range(model.n_layers):
        if L == _ALIGN_LAYER_0:
            t3_block_prefill_with_attn(
                ctx, model.blocks[L], x_buf, cos_pre, sin_pre, prefill_mask,
                k_caches[L], v_caches[L], attn_pre_0, B2, t_prefill, max_ctx,
            )
        elif L == _ALIGN_LAYER_1:
            t3_block_prefill_with_attn(
                ctx, model.blocks[L], x_buf, cos_pre, sin_pre, prefill_mask,
                k_caches[L], v_caches[L], attn_pre_1, B2, t_prefill, max_ctx,
            )
        elif L == _ALIGN_LAYER_2:
            t3_block_prefill_with_attn(
                ctx, model.blocks[L], x_buf, cos_pre, sin_pre, prefill_mask,
                k_caches[L], v_caches[L], attn_pre_2, B2, t_prefill, max_ctx,
            )
        else:
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
    def take_last_a[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // D
        var di = i - bi * D
        lhp[i] = xp[bi * tpf * D + (tpf - 1) * D + di]
    elementwise[take_last_a, simd_width=1, target="gpu"](
        IndexList[1](B2 * D), DeviceContextPtr(ctx),
    )

    var logits_buf = ctx.enqueue_create_buffer[DType.float32](B2 * V)
    linear_forward(ctx, model.speech_head, last_hidden, logits_buf, B2)
    ctx.synchronize()

    var analyzer = make_alignment_analyzer(text_slice_j - text_slice_i, eos_token)

    var aligned_row_pre = _extract_aligned_row_prefill(
        ctx, attn_pre_0, attn_pre_1, attn_pre_2,
        B2, H, t_prefill, text_slice_i, text_slice_j,
    )

    var generated = List[Int64]()
    var rng_state: UInt64 = rng_seed
    var first_tok = cfg_combine_sample_with_analyzer(
        ctx, logits_buf, generated, rng_state, analyzer,
        aligned_row_pre, Int64(-1),
        b, V, cfg_weight, temperature, top_p, rep_penalty, min_p,
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
        def add_pos_a[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
            var i = idx[0]
            var bi = i // D
            var di = i - bi * D
            sep[i] = sep[i] + spt[pos_idx * D + di]
        elementwise[add_pos_a, simd_width=1, target="gpu"](
            IndexList[1](B2 * D), DeviceContextPtr(ctx),
        )

        gather_cos_sin_at_pos(ctx, cos_full, sin_full, cos_step, sin_step,
                              B2, Dh, cur_len)

        var s_k = cur_len + 1
        var attn_step_0 = ctx.enqueue_create_buffer[DType.float32](B2 * H * 1 * s_k)
        var attn_step_1 = ctx.enqueue_create_buffer[DType.float32](B2 * H * 1 * s_k)
        var attn_step_2 = ctx.enqueue_create_buffer[DType.float32](B2 * H * 1 * s_k)

        for L in range(model.n_layers):
            if L == _ALIGN_LAYER_0:
                t3_decode_step_with_attn(
                    ctx, model.blocks[L], step_emb_buf,
                    k_caches[L], v_caches[L], cos_step, sin_step,
                    attn_step_0, B2, max_ctx, cur_len,
                )
            elif L == _ALIGN_LAYER_1:
                t3_decode_step_with_attn(
                    ctx, model.blocks[L], step_emb_buf,
                    k_caches[L], v_caches[L], cos_step, sin_step,
                    attn_step_1, B2, max_ctx, cur_len,
                )
            elif L == _ALIGN_LAYER_2:
                t3_decode_step_with_attn(
                    ctx, model.blocks[L], step_emb_buf,
                    k_caches[L], v_caches[L], cos_step, sin_step,
                    attn_step_2, B2, max_ctx, cur_len,
                )
            else:
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

        var aligned_row = _extract_aligned_row_decode(
            ctx, attn_step_0, attn_step_1, attn_step_2,
            B2, H, s_k, text_slice_i, text_slice_j,
        )

        var last_emitted = generated[len(generated) - 1]
        var next_tok = cfg_combine_sample_with_analyzer(
            ctx, logits_buf, generated, rng_state, analyzer,
            aligned_row, last_emitted,
            b, V, cfg_weight, temperature, top_p, rep_penalty, min_p,
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
