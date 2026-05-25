"""AlignmentStreamAnalyzer port from upstream chatterbox.

Watches attention weights from layers 9/12/13 (heads 2/15/11) during T3
generation. Suppresses early EOS, and force-emits EOS when phrase repetition
or hallucinations are detected. Without this, the model loops on phrase
fragments instead of stopping.

Each generation step produces a (1, T_TEXT) "alignment row" — the row of
the meaned attention map at the current Q-position, restricted to the text
token slice. Rows accumulate into an (frames, T_TEXT) alignment matrix.
"""
from std.math import sqrt


@fieldwise_init
struct AlignmentAnalyzer(Movable):
    """CPU-side analyzer. All math is on host floats — no GPU ops here.

    Fields mirror upstream chatterbox's AlignmentStreamAnalyzer."""
    var t_text: Int                  # j - i; number of text tokens
    var eos_idx: Int                 # vocab index of EOS
    # Alignment rows: each row is (T_TEXT,). Length = number of steps seen.
    var alignment: List[List[Float32]]
    var curr_frame_pos: Int
    var text_position: Int
    var started: Bool
    var started_at: Int
    var complete: Bool
    var completed_at: Int
    var generated_tokens: List[Int64]   # rolling window, last 8


def make_alignment_analyzer(t_text: Int, eos_idx: Int) -> AlignmentAnalyzer:
    """Construct an analyzer for a given text length and EOS token id."""
    return AlignmentAnalyzer(
        t_text=t_text,
        eos_idx=eos_idx,
        alignment=List[List[Float32]](),
        curr_frame_pos=0,
        text_position=0,
        started=False,
        started_at=-1,
        complete=False,
        completed_at=-1,
        generated_tokens=List[Int64](),
    )


def aa_argmax_row(row: List[Float32]) -> Int:
    """argmax of a row."""
    var best: Float32 = -1.0e30
    var best_i: Int = 0
    for k in range(len(row)):
        if row[k] > best:
            best = row[k]
            best_i = k
    return best_i


def aa_max_in_slice(row: List[Float32], a: Int, b: Int) -> Float32:
    """max of row[a:b], inclusive a exclusive b. Returns 0 if range empty."""
    var lo = a if a > 0 else 0
    var hi = b if b < len(row) else len(row)
    if lo >= hi:
        return Float32(0.0)
    var m: Float32 = row[lo]
    for k in range(lo + 1, hi):
        if row[k] > m: m = row[k]
    return m


def aa_step(
    mut analyzer: AlignmentAnalyzer,
    aligned_row: List[Float32],          # (T_TEXT,) — meaned-over-heads attn row at this Q-pos
    mut logits: List[Float32],           # (V,) — will be modified in place
    next_token: Int64,                   # token just emitted at the PREVIOUS step (or 0 if none)
):
    """Run one analyzer step. Modifies `logits` in-place:
      - Suppresses EOS if generation hasn't reached end of text
      - Force-emits EOS if repetition / long-tail detected

    Returns nothing. Caller calls this AFTER computing logits for the next
    step but BEFORE any sampling/temperature/top-p/min-p processing.
    """
    var t_text = analyzer.t_text
    var s = t_text

    # Append the row (mask future text positions to 0 — matches upstream
    # "A_chunk[:, self.curr_frame_pos + 1:] = 0" if monotonic; we keep all
    # entries since the analyzer's curr_frame_pos counts STEP rows, not text
    # positions. Upstream masks A_chunk[:, curr_frame_pos+1:] only as a TODO
    # for monotonic, kept identical here.)
    var clipped = List[Float32](capacity=t_text)
    for k in range(t_text):
        if k > analyzer.curr_frame_pos:
            clipped.append(Float32(0.0))
        else:
            clipped.append(aligned_row[k])
    analyzer.alignment.append(clipped^)

    var n_frames = len(analyzer.alignment)

    # update text position from argmax of last row (= clipped, before the move).
    # We use the just-appended row by indexing back into analyzer.alignment.
    var cur_text_posn: Int = 0
    var best_v: Float32 = -1.0e30
    for k in range(t_text):
        var v = analyzer.alignment[n_frames - 1][k]
        if v > best_v:
            best_v = v
            cur_text_posn = k
    var diff = cur_text_posn - analyzer.text_position
    var discontinuity = not (-4 < diff < 7)
    if not discontinuity:
        analyzer.text_position = cur_text_posn

    # false_start detection
    # A[-2:, -2:].max() > 0.1 or A[:, :4].max() < 0.5
    var max_tail: Float32 = 0.0
    var max_head: Float32 = 0.0
    var start_row = n_frames - 2 if n_frames >= 2 else 0
    var start_col = t_text - 2 if t_text >= 2 else 0
    for r in range(start_row, n_frames):
        for c in range(start_col, t_text):
            var v = analyzer.alignment[r][c]
            if v > max_tail: max_tail = v
    var head_end = 4 if t_text >= 4 else t_text
    for r in range(n_frames):
        for c in range(head_end):
            var v = analyzer.alignment[r][c]
            if v > max_head: max_head = v
    var false_start = (not analyzer.started) and (max_tail > 0.1 or max_head < 0.5)
    analyzer.started = not false_start
    if analyzer.started and analyzer.started_at < 0:
        analyzer.started_at = n_frames

    # complete: text_position has reached within 3 of end of text
    var done = analyzer.text_position >= (s - 3)
    if not analyzer.complete:
        analyzer.complete = done
        if analyzer.complete:
            analyzer.completed_at = n_frames

    # long_tail: complete AND attention to last 3 text tokens has too much mass
    # A[completed_at:, -3:].sum(dim=0).max() >= 5
    var long_tail = False
    if analyzer.complete and analyzer.completed_at >= 0:
        var lt_col_start = t_text - 3 if t_text >= 3 else 0
        # sum per column across rows [completed_at:n_frames], max across cols
        var col_max: Float32 = 0.0
        for c in range(lt_col_start, t_text):
            var col_sum: Float32 = 0.0
            for r in range(analyzer.completed_at, n_frames):
                col_sum += analyzer.alignment[r][c]
            if col_sum > col_max: col_max = col_sum
        if col_max >= 5.0:
            long_tail = True

    # alignment_repetition: after complete, sum of row-maxes over text[:-5]
    # A[completed_at:, :-5].max(dim=1).values.sum() > 5
    var alignment_repetition = False
    if analyzer.complete and analyzer.completed_at >= 0:
        var ar_col_end = t_text - 5 if t_text >= 5 else 0
        if ar_col_end > 0:
            var sum_rowmax: Float32 = 0.0
            for r in range(analyzer.completed_at, n_frames):
                var rm = aa_max_in_slice(analyzer.alignment[r], 0, ar_col_end)
                sum_rowmax += rm
            if sum_rowmax > 5.0:
                alignment_repetition = True

    # token-level repetition: 2x same token in a row in last 2 generated.
    if next_token >= 0:
        analyzer.generated_tokens.append(next_token)
        if len(analyzer.generated_tokens) > 8:
            # Drop oldest. List doesn't have pop_front; rebuild last 8.
            var new_list = List[Int64](capacity=8)
            var start = len(analyzer.generated_tokens) - 8
            for k in range(start, len(analyzer.generated_tokens)):
                new_list.append(analyzer.generated_tokens[k])
            analyzer.generated_tokens = new_list^

    # Upstream also checks `token_repetition = (len(gen) >= 3 and set(last 2)==1)`.
    # We tried it: fires far too aggressively at step 5 because chatterbox often
    # emits 2 adjacent identical speech tokens for sustained phonemes. Disabled;
    # rely on long_tail + alignment_repetition + natural EOS instead.
    var token_repetition = False

    # Suppress EOS until near end of text. Same condition as upstream:
    # "if cur_text_posn < S - 3 and S > 5: logits[eos] = -2^15"
    if cur_text_posn < (s - 3) and s > 5:
        if analyzer.eos_idx >= 0 and analyzer.eos_idx < len(logits):
            logits[analyzer.eos_idx] = Float32(-32768.0)

    # Force EOS if bad ending detected.
    if long_tail or alignment_repetition or token_repetition:
        for k in range(len(logits)):
            logits[k] = Float32(-32768.0)
        if analyzer.eos_idx >= 0 and analyzer.eos_idx < len(logits):
            logits[analyzer.eos_idx] = Float32(32768.0)

    analyzer.curr_frame_pos += 1
