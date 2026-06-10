//! Soft word-wrap layout for the editor.
//!
//! The editor renders on a strict monospace grid: every visual row is
//! `LINE_HEIGHT` tall and column `c` sits at `c * cell_width`. Without
//! wrapping, one logical line == one visual row. With wrapping, a single
//! logical line spans one *or more* visual rows, each holding a
//! contiguous slice of the line.
//!
//! [`wrap_segments`] splits one logical line into those slices (half-open
//! `[start_col, end_col)` char ranges that partition `[0, len]`), and
//! [`WrapLayout`] stacks every line's segments into a document-wide map
//! between logical `(line, col)` positions and global visual rows. All
//! editor geometry — rendering, caret, selection, click hit-testing,
//! vertical caret movement, scroll — reads from this one map so they
//! never disagree.
//!
//! Wrapping is word-granular (break at spaces, trailing spaces stay on
//! the upper row) with a character-break fallback for a single token
//! longer than the wrap width. Because the font is monospace, counting
//! columns is exact, so this matches what's actually drawn.

use ropey::Rope;

/// Split one logical line (no trailing `\n`) into visual-row segments,
/// each a half-open `[start_col, end_col)` char range. The segments are
/// contiguous and cover `[0, len]`; the result always has at least one
/// segment (an empty line yields `[(0, 0)]`). `cols` is the wrap width in
/// monospace columns.
pub fn wrap_segments(line: &str, cols: usize) -> Vec<(usize, usize)> {
    let len = line.chars().count();
    if cols == 0 || len == 0 {
        return vec![(0, len)];
    }
    let chars: Vec<char> = line.chars().collect();
    let mut segs: Vec<(usize, usize)> = Vec::new();
    let mut seg_start = 0usize; // char index where the current row starts
    let mut i = 0usize;

    while i < len {
        // A token is a maximal run of spaces or a maximal run of
        // non-spaces. Spaces never force a wrap (a space at the right
        // edge is invisible, so we let it overflow); a non-space token
        // that doesn't fit wraps before itself, char-breaking if it is
        // wider than a whole row.
        let tok_start = i;
        let is_space = chars[i] == ' ';
        if is_space {
            while i < len && chars[i] == ' ' {
                i += 1;
            }
            continue;
        }
        while i < len && chars[i] != ' ' {
            i += 1;
        }
        let tok_end = i;
        let tlen = tok_end - tok_start;
        let col = tok_start - seg_start; // columns already used on this row

        if col + tlen <= cols {
            continue; // fits on the current row
        }
        if tlen <= cols {
            // Word fits on a row of its own — wrap before it. Any spaces
            // preceding it stay on the upper row (they're inside
            // `[seg_start, tok_start)`).
            if col > 0 {
                segs.push((seg_start, tok_start));
                seg_start = tok_start;
            }
            continue;
        }
        // Token is wider than a full row — char-break it.
        if col > 0 {
            segs.push((seg_start, tok_start));
            seg_start = tok_start;
        }
        let mut p = tok_start;
        while tok_end - p > cols {
            segs.push((p, p + cols));
            p += cols;
        }
        seg_start = p; // remainder stays on the current row
    }
    segs.push((seg_start, len));
    segs
}

/// Document-wide wrap map: every logical line's segments plus a prefix
/// sum of visual-row counts so positions and rows convert in O(log n).
#[derive(Clone, Debug, Default)]
pub struct WrapLayout {
    pub cols: usize,
    /// `line_segs[line]` = that line's visual-row segments.
    pub line_segs: Vec<Vec<(usize, usize)>>,
    /// `rows_before[line]` = total visual rows in lines `0..line`.
    /// Length is `line_segs.len() + 1`; the last entry is the total.
    pub rows_before: Vec<usize>,
}

impl WrapLayout {
    /// Build the map for `rope` wrapped at `cols` columns. Iterates
    /// `effective` logical lines (the caller passes the editor's
    /// effective line count so a trailing empty line is treated the way
    /// the rest of the editor treats it).
    pub fn build(rope: &Rope, cols: usize, effective: usize) -> Self {
        let mut line_segs = Vec::with_capacity(effective);
        let mut rows_before = Vec::with_capacity(effective + 1);
        let mut acc = 0usize;
        for idx in 0..effective {
            rows_before.push(acc);
            let text = line_text(rope, idx);
            let segs = wrap_segments(&text, cols);
            acc += segs.len();
            line_segs.push(segs);
        }
        rows_before.push(acc);
        Self {
            cols,
            line_segs,
            rows_before,
        }
    }

    pub fn total_rows(&self) -> usize {
        self.rows_before.last().copied().unwrap_or(0)
    }

    /// Visual rows occupied by one logical line (>= 1).
    pub fn line_visual_rows(&self, line: usize) -> usize {
        self.line_segs.get(line).map(|s| s.len()).unwrap_or(1)
    }

    /// Map a logical `(line, col)` to a global visual row and the column
    /// offset within that row. A position exactly on a wrap boundary maps
    /// to the *start* of the lower row.
    pub fn pos_to_visual(&self, line: usize, col: usize) -> (usize, usize) {
        let Some(segs) = self.line_segs.get(line) else {
            return (self.rows_before.get(line).copied().unwrap_or(0), col);
        };
        let base = self.rows_before[line];
        // Last segment whose start <= col (so a boundary lands on the
        // lower row). Segments are contiguous and ordered.
        let mut seg_idx = 0usize;
        for (k, &(s, _e)) in segs.iter().enumerate() {
            if s <= col {
                seg_idx = k;
            } else {
                break;
            }
        }
        let (s, _e) = segs[seg_idx];
        (base + seg_idx, col - s)
    }

    /// Map a global visual `row` to the logical line and the segment
    /// index within that line. Clamps to the last row when out of range.
    pub fn row_to_line_seg(&self, row: usize) -> Option<(usize, usize)> {
        if self.line_segs.is_empty() {
            return None;
        }
        let total = self.total_rows();
        let row = row.min(total.saturating_sub(1));
        // Largest line with `rows_before[line] <= row`. Since every line
        // has at least one row, `rows_before` is strictly increasing.
        let line = match self.rows_before.binary_search(&row) {
            Ok(i) => i.min(self.line_segs.len() - 1),
            Err(i) => i.saturating_sub(1).min(self.line_segs.len() - 1),
        };
        let seg_idx = (row - self.rows_before[line]).min(self.line_segs[line].len() - 1);
        Some((line, seg_idx))
    }

    /// Segment (char range) shown on a global visual `row`.
    pub fn segment_at_row(&self, row: usize) -> Option<(usize, usize, usize)> {
        let (line, seg_idx) = self.row_to_line_seg(row)?;
        let (s, e) = self.line_segs[line][seg_idx];
        Some((line, s, e))
    }

    /// Map a global visual `row` plus a column offset `x_col` to a logical
    /// `(line, col)`. `x_col` is clamped into the row's segment.
    pub fn visual_to_pos(&self, row: usize, x_col: usize) -> (usize, usize) {
        let Some((line, seg_idx)) = self.row_to_line_seg(row) else {
            return (0, 0);
        };
        let (s, e) = self.line_segs[line][seg_idx];
        let col = (s + x_col).min(e);
        (line, col)
    }
}

/// One logical line's text with any trailing newline stripped. Mirrors
/// the helper in `lib.rs`; duplicated here to keep `wrap` self-contained
/// and unit-testable.
fn line_text(rope: &Rope, idx: usize) -> String {
    let s = rope.line(idx).to_string();
    s.strip_suffix('\n').map(str::to_string).unwrap_or(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cols_of(line: &str, cols: usize) -> Vec<String> {
        wrap_segments(line, cols)
            .into_iter()
            .map(|(s, e)| line.chars().skip(s).take(e - s).collect())
            .collect()
    }

    #[test]
    fn empty_and_short_lines_are_one_segment() {
        assert_eq!(wrap_segments("", 10), vec![(0, 0)]);
        assert_eq!(wrap_segments("abc", 10), vec![(0, 3)]);
    }

    #[test]
    fn wraps_at_word_boundary_keeping_space_on_upper_row() {
        // "hello world" at width 7: "hello " then "world".
        assert_eq!(cols_of("hello world", 7), vec!["hello ", "world"]);
    }

    #[test]
    fn segments_are_contiguous_and_cover_the_line() {
        for line in ["", "a", "one two three four", "xx yyyy zzzzzzzzzzzz q"] {
            for cols in [1usize, 3, 5, 8] {
                let segs = wrap_segments(line, cols);
                assert_eq!(segs.first().unwrap().0, 0, "starts at 0: {line:?}/{cols}");
                assert_eq!(
                    segs.last().unwrap().1,
                    line.chars().count(),
                    "ends at len: {line:?}/{cols}"
                );
                for w in segs.windows(2) {
                    assert_eq!(w[0].1, w[1].0, "contiguous: {line:?}/{cols}");
                }
            }
        }
    }

    #[test]
    fn char_breaks_a_token_wider_than_the_row() {
        // 12-char token, width 5 → 5 + 5 + 2.
        assert_eq!(cols_of("abcdefghijkl", 5), vec!["abcde", "fghij", "kl"]);
    }

    #[test]
    fn char_break_then_wraps_following_word_that_doesnt_fit() {
        // "zzzzzz" (6) at width 4 char-breaks to "zzzz","zz"; the trailing
        // space sits on the "zz" row ("zz "=3 cols), and "ab" (2 cols)
        // overflows 4, so it wraps to its own row.
        assert_eq!(cols_of("zzzzzz ab", 4), vec!["zzzz", "zz ", "ab"]);
    }

    #[test]
    fn char_break_remainder_packs_a_fitting_word() {
        // Remainder "zz" (2) + " a" (2) == 4 exactly → packs.
        assert_eq!(cols_of("zzzzzz a", 4), vec!["zzzz", "zz a"]);
    }

    #[test]
    fn layout_round_trips_position_and_row() {
        let rope = Rope::from_str("hello world\nshort\n");
        // effective lines: 2 ("hello world", "short").
        let layout = WrapLayout::build(&rope, 7, 2);
        // "hello world" → rows 0,1 ; "short" → row 2.
        assert_eq!(layout.total_rows(), 3);
        assert_eq!(layout.line_visual_rows(0), 2);
        assert_eq!(layout.line_visual_rows(1), 1);

        // col 0 of line 0 → row 0, x 0.
        assert_eq!(layout.pos_to_visual(0, 0), (0, 0));
        // col 6 ("w" of "world") is the start of the 2nd segment (seg
        // ["hello "]=[0,6), ["world"]=[6,11]) → row 1, x 0.
        assert_eq!(layout.pos_to_visual(0, 6), (1, 0));
        // col 8 → row 1, x 2.
        assert_eq!(layout.pos_to_visual(0, 8), (1, 2));
        // line 1 col 3 → row 2, x 3.
        assert_eq!(layout.pos_to_visual(1, 3), (2, 3));

        // Inverse: row 1 x 2 → line 0 col 8.
        assert_eq!(layout.visual_to_pos(1, 2), (0, 8));
        // row 2 x 10 (past end) → clamps to line 1 col 5 (len "short").
        assert_eq!(layout.visual_to_pos(2, 10), (1, 5));
        // row 0 x 3 → line 0 col 3.
        assert_eq!(layout.visual_to_pos(0, 3), (0, 3));
    }

    #[test]
    fn boundary_position_maps_to_lower_row() {
        let rope = Rope::from_str("aaaabbbb\n");
        // width 4: "aaaa"|"bbbb" → 2 rows. col 4 is the boundary.
        let layout = WrapLayout::build(&rope, 4, 1);
        assert_eq!(layout.pos_to_visual(0, 4), (1, 0));
        // end of line (col 8) → last row, x 4.
        assert_eq!(layout.pos_to_visual(0, 8), (1, 4));
    }
}
