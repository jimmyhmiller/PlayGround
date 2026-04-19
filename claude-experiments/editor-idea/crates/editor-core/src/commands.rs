use ropey::Rope;

use crate::selection::{Range, Selection};
use crate::state::EditorState;
use crate::transaction::{Change, Transaction};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CharClass {
    Newline,
    Space,
    Word,
    Punct,
}

fn classify(c: char) -> CharClass {
    if c == '\n' {
        CharClass::Newline
    } else if c == ' ' || c == '\t' {
        CharClass::Space
    } else if c.is_alphanumeric() || c == '_' {
        CharClass::Word
    } else {
        CharClass::Punct
    }
}

/// A command takes a state and (optionally) returns a transaction. Mirrors
/// CodeMirror 6's `StateCommand`: returning `None` means "didn't apply".
pub type Command = fn(&EditorState) -> Option<Transaction>;

fn line_starts_in_range(state: &EditorState, range: Range) -> Vec<usize> {
    let doc = &state.doc;
    let from_line = doc.char_to_line(range.from());
    // If the range ends right at a line boundary and isn't empty, exclude that
    // trailing line — it's the convention in CM6 (a selection ending at the
    // start of the next line shouldn't indent that next line).
    let to = range.to();
    let to_line = if range.is_empty() || to == doc.line_to_char(doc.char_to_line(to)) && to > range.from() {
        let l = doc.char_to_line(to);
        if l > from_line && to == doc.line_to_char(l) {
            l - 1
        } else {
            l
        }
    } else {
        doc.char_to_line(to)
    };
    (from_line..=to_line).map(|l| doc.line_to_char(l)).collect()
}

fn unique_line_starts(state: &EditorState) -> Vec<usize> {
    let mut starts: Vec<usize> = state
        .selection
        .ranges
        .iter()
        .flat_map(|r| line_starts_in_range(state, *r))
        .collect();
    starts.sort_unstable();
    starts.dedup();
    starts
}

/// Shift `pos` by the cumulative effect of `shifts` (sorted `(at, delta)` pairs).
/// Insertions at exactly `pos` push the position forward — matches CodeMirror's
/// `changeByRange` behavior for indent commands.
fn shift_pos(shifts: &[(usize, isize)], pos: usize) -> usize {
    let mut total: isize = 0;
    for &(at, delta) in shifts {
        if at <= pos {
            total += delta;
        }
    }
    (pos as isize + total).max(0) as usize
}

fn rebuild_selection(state: &EditorState, shifts: &[(usize, isize)]) -> Selection {
    let ranges = state
        .selection
        .ranges
        .iter()
        .map(|r| Range::new(shift_pos(shifts, r.anchor), shift_pos(shifts, r.head)))
        .collect();
    Selection::new(ranges, state.selection.primary)
}

pub fn indent_more(state: &EditorState) -> Option<Transaction> {
    let starts = unique_line_starts(state);
    if starts.is_empty() {
        return None;
    }
    let unit = &state.indent_unit;
    let unit_len = unit.chars().count() as isize;
    let changes: Vec<Change> = starts.iter().map(|&pos| Change::insert(pos, unit)).collect();
    let shifts: Vec<(usize, isize)> = starts.iter().map(|&pos| (pos, unit_len)).collect();
    let selection = rebuild_selection(state, &shifts);
    Some(Transaction { changes, selection: Some(selection) })
}

/// Returns the (first_line, last_line) pair covered by `range`. Lines are
/// inclusive. A non-empty range that ends exactly at a line start does not
/// include that trailing line — matches CodeMirror's `selectedLineBlocks`.
fn range_lines(state: &EditorState, range: Range) -> (usize, usize) {
    let doc = &state.doc;
    let from_line = doc.char_to_line(range.from());
    let to = range.to();
    let to_line = doc.char_to_line(to);
    let last = if !range.is_empty() && to_line > from_line && to == doc.line_to_char(to_line) {
        to_line - 1
    } else {
        to_line
    };
    (from_line, last)
}

/// Group all selection ranges into merged inclusive line blocks.
/// Adjacent (`next.0 == prev.1 + 1`) and overlapping blocks coalesce.
fn line_blocks(state: &EditorState) -> Vec<(usize, usize)> {
    let mut blocks: Vec<(usize, usize)> = state
        .selection
        .ranges
        .iter()
        .map(|r| range_lines(state, *r))
        .collect();
    blocks.sort_by_key(|b| b.0);
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(blocks.len());
    for b in blocks {
        match merged.last_mut() {
            Some(last) if b.0 <= last.1 + 1 => last.1 = last.1.max(b.1),
            _ => merged.push(b),
        }
    }
    merged
}

/// Char index just past the end of `line_idx` (start of next line, or doc end).
fn line_end_char(doc: &ropey::Rope, line_idx: usize) -> usize {
    if line_idx + 1 < doc.len_lines() {
        doc.line_to_char(line_idx + 1)
    } else {
        doc.len_chars()
    }
}

/// Last "real" line index (skipping the phantom trailing empty line ropey
/// reports for docs ending in `\n`).
fn last_real_line(doc: &ropey::Rope) -> usize {
    let lines = doc.len_lines();
    if lines == 0 {
        0
    } else if doc.len_chars() > 0
        && doc.char(doc.len_chars() - 1) == '\n'
        && lines >= 2
    {
        lines - 2
    } else {
        lines - 1
    }
}

/// Position mapping for `move_line_up`: above/block swap.
/// Boundary `p == block_start` and `p == block_end` are treated as "in block"
/// so range endpoints anchored at line boundaries track the moved block.
fn map_swap_up(
    p: usize,
    above_start: usize,
    block_start: usize,
    block_end: usize,
    new_block_len: usize,
    new_above_len: usize,
) -> usize {
    if p < above_start || p > block_end {
        return p;
    }
    if p < block_start {
        // In above (originally first, now last).
        let off = (p - above_start).min(new_above_len);
        above_start + new_block_len + off
    } else {
        // In block (block_start..=block_end). Originally last, now first.
        let off = (p - block_start).min(new_block_len);
        above_start + off
    }
}

/// Position mapping for `move_line_down`: block/below swap.
fn map_swap_down(
    p: usize,
    block_start: usize,
    block_end: usize,
    below_end: usize,
    new_below_len: usize,
    new_block_len: usize,
) -> usize {
    if p < block_start || p > below_end {
        return p;
    }
    if p <= block_end {
        // In block (originally first, now last).
        let off = (p - block_start).min(new_block_len);
        block_start + new_below_len + off
    } else {
        // In below (block_end < p <= below_end). Originally last, now first.
        let off = (p - block_end).min(new_below_len);
        block_start + off
    }
}

struct SwapUp {
    above_start: usize,
    block_start: usize,
    block_end: usize,
    new_block_len: usize,
    new_above_len: usize,
}

struct SwapDown {
    block_start: usize,
    block_end: usize,
    below_end: usize,
    new_below_len: usize,
    new_block_len: usize,
}

pub fn move_line_up(state: &EditorState) -> Option<Transaction> {
    let doc = &state.doc;
    let mut changes = Vec::new();
    let mut swaps: Vec<SwapUp> = Vec::new();

    for (a, b) in line_blocks(state) {
        if a == 0 {
            continue;
        }
        let above_start = doc.line_to_char(a - 1);
        let block_start = doc.line_to_char(a);
        let block_end = line_end_char(doc, b);

        let above_text: String = doc.slice(above_start..block_start).to_string();
        let block_text: String = doc.slice(block_start..block_end).to_string();

        // Special case: block was at end of doc without trailing newline; the
        // line above contributes its newline.
        let (new_block, new_above) =
            if !block_text.ends_with('\n') && above_text.ends_with('\n') {
                let nb = format!("{}\n", block_text);
                let na = above_text[..above_text.len() - 1].to_string();
                (nb, na)
            } else {
                (block_text.clone(), above_text.clone())
            };

        let new_block_len = new_block.chars().count();
        let new_above_len = new_above.chars().count();

        let mut combined = new_block;
        combined.push_str(&new_above);
        changes.push(Change::new(above_start, block_end, combined));
        swaps.push(SwapUp {
            above_start,
            block_start,
            block_end,
            new_block_len,
            new_above_len,
        });
    }

    if changes.is_empty() {
        return None;
    }

    let new_ranges: Vec<Range> = state
        .selection
        .ranges
        .iter()
        .map(|r| {
            let mut anchor = r.anchor;
            let mut head = r.head;
            for s in &swaps {
                anchor = map_swap_up(
                    anchor,
                    s.above_start,
                    s.block_start,
                    s.block_end,
                    s.new_block_len,
                    s.new_above_len,
                );
                head = map_swap_up(
                    head,
                    s.above_start,
                    s.block_start,
                    s.block_end,
                    s.new_block_len,
                    s.new_above_len,
                );
            }
            Range::new(anchor, head)
        })
        .collect();

    Some(Transaction {
        changes,
        selection: Some(Selection::new(new_ranges, state.selection.primary)),
    })
}

pub fn move_line_down(state: &EditorState) -> Option<Transaction> {
    let doc = &state.doc;
    let last = last_real_line(doc);
    let mut changes = Vec::new();
    let mut swaps: Vec<SwapDown> = Vec::new();

    for (a, b) in line_blocks(state) {
        if b >= last {
            continue;
        }
        let block_start = doc.line_to_char(a);
        let block_end = doc.line_to_char(b + 1);
        let below_end = line_end_char(doc, b + 1);

        let block_text: String = doc.slice(block_start..block_end).to_string();
        let below_text: String = doc.slice(block_end..below_end).to_string();

        // Special case: below is the doc-final line lacking a trailing newline.
        // Transfer the newline from the moving block to keep line structure.
        let (new_below, new_block) =
            if !below_text.ends_with('\n') && block_text.ends_with('\n') {
                let nb = format!("{}\n", below_text);
                let nbk = block_text[..block_text.len() - 1].to_string();
                (nb, nbk)
            } else {
                (below_text.clone(), block_text.clone())
            };

        let new_below_len = new_below.chars().count();
        let new_block_len = new_block.chars().count();

        let mut combined = new_below;
        combined.push_str(&new_block);
        changes.push(Change::new(block_start, below_end, combined));
        swaps.push(SwapDown {
            block_start,
            block_end,
            below_end,
            new_below_len,
            new_block_len,
        });
    }

    if changes.is_empty() {
        return None;
    }

    let new_ranges: Vec<Range> = state
        .selection
        .ranges
        .iter()
        .map(|r| {
            let mut anchor = r.anchor;
            let mut head = r.head;
            for s in &swaps {
                anchor = map_swap_down(
                    anchor,
                    s.block_start,
                    s.block_end,
                    s.below_end,
                    s.new_below_len,
                    s.new_block_len,
                );
                head = map_swap_down(
                    head,
                    s.block_start,
                    s.block_end,
                    s.below_end,
                    s.new_below_len,
                    s.new_block_len,
                );
            }
            Range::new(anchor, head)
        })
        .collect();

    Some(Transaction {
        changes,
        selection: Some(Selection::new(new_ranges, state.selection.primary)),
    })
}

/// Find where forward group-deletion from `p` should end. Mirrors CM6 rules:
///  * a `\n` is its own group (delete one char)
///  * a run of >1 whitespace deletes just the whitespace
///  * a run of exactly 1 whitespace followed by a non-newline group also
///    eats that following group (so a single trailing space gets deleted
///    along with the next word — `"one| two"` → `"one|"`)
fn find_group_end_forward(doc: &Rope, p: usize) -> usize {
    let len = doc.len_chars();
    if p >= len {
        return p;
    }
    let c = doc.char(p);
    match classify(c) {
        CharClass::Newline => p + 1,
        CharClass::Space => {
            let mut q = p;
            while q < len && classify(doc.char(q)) == CharClass::Space {
                q += 1;
            }
            let ws_count = q - p;
            if ws_count > 1 || q == len || classify(doc.char(q)) == CharClass::Newline {
                q
            } else {
                let next_class = classify(doc.char(q));
                while q < len && classify(doc.char(q)) == next_class {
                    q += 1;
                }
                q
            }
        }
        cls => {
            let mut q = p;
            while q < len && classify(doc.char(q)) == cls {
                q += 1;
            }
            q
        }
    }
}

/// Symmetric backward variant. Returns the position to delete *back to* —
/// the resulting deletion range is `[returned, p)`.
fn find_group_end_backward(doc: &Rope, p: usize) -> usize {
    if p == 0 {
        return p;
    }
    let c = doc.char(p - 1);
    match classify(c) {
        CharClass::Newline => p - 1,
        CharClass::Space => {
            let mut q = p;
            while q > 0 && classify(doc.char(q - 1)) == CharClass::Space {
                q -= 1;
            }
            let ws_count = p - q;
            if ws_count > 1 || q == 0 || classify(doc.char(q - 1)) == CharClass::Newline {
                q
            } else {
                let prev_class = classify(doc.char(q - 1));
                while q > 0 && classify(doc.char(q - 1)) == prev_class {
                    q -= 1;
                }
                q
            }
        }
        cls => {
            let mut q = p;
            while q > 0 && classify(doc.char(q - 1)) == cls {
                q -= 1;
            }
            q
        }
    }
}

fn delete_by_group(state: &EditorState, forward: bool) -> Option<Transaction> {
    let doc = &state.doc;
    let mut indexed: Vec<(usize, Range)> = state
        .selection
        .ranges
        .iter()
        .copied()
        .enumerate()
        .collect();
    indexed.sort_by_key(|(_, r)| r.from());

    let mut changes = Vec::new();
    let mut new_positions: Vec<(usize, usize)> = Vec::new();
    let mut shift: isize = 0;

    for (orig_idx, range) in indexed {
        let (del_from, del_to) = if !range.is_empty() {
            (range.from(), range.to())
        } else if forward {
            let p = range.from();
            (p, find_group_end_forward(doc, p))
        } else {
            let p = range.from();
            (find_group_end_backward(doc, p), p)
        };

        if del_from == del_to {
            let new_pos = (range.from() as isize + shift) as usize;
            new_positions.push((orig_idx, new_pos));
            continue;
        }

        changes.push(Change::delete(del_from, del_to));
        let new_pos = (del_from as isize + shift) as usize;
        new_positions.push((orig_idx, new_pos));
        shift -= (del_to - del_from) as isize;
    }

    if changes.is_empty() {
        return None;
    }

    new_positions.sort_by_key(|(idx, _)| *idx);
    let new_ranges: Vec<Range> = new_positions
        .iter()
        .map(|(_, p)| Range::cursor(*p))
        .collect();

    Some(Transaction {
        changes,
        selection: Some(Selection::new(new_ranges, state.selection.primary)),
    })
}

pub fn delete_group_forward(state: &EditorState) -> Option<Transaction> {
    delete_by_group(state, true)
}

pub fn delete_group_backward(state: &EditorState) -> Option<Transaction> {
    delete_by_group(state, false)
}

/// Backspace: delete the character before each cursor; for non-empty
/// selections, delete the selection.
pub fn delete_char_backward(state: &EditorState) -> Option<Transaction> {
    delete_by(state, false, |_doc, p| p.saturating_sub(1))
}

/// Delete: delete the character after each cursor; for non-empty selections,
/// delete the selection.
pub fn delete_char_forward(state: &EditorState) -> Option<Transaction> {
    delete_by(state, true, |doc, p| (p + 1).min(doc.len_chars()))
}

fn delete_by(
    state: &EditorState,
    forward: bool,
    find_other_end: impl Fn(&Rope, usize) -> usize,
) -> Option<Transaction> {
    let doc = &state.doc;
    let mut indexed: Vec<(usize, Range)> = state
        .selection
        .ranges
        .iter()
        .copied()
        .enumerate()
        .collect();
    indexed.sort_by_key(|(_, r)| r.from());

    let mut changes = Vec::new();
    let mut new_positions: Vec<(usize, usize)> = Vec::new();
    let mut shift: isize = 0;

    for (orig_idx, range) in indexed {
        let (del_from, del_to) = if !range.is_empty() {
            (range.from(), range.to())
        } else if forward {
            let p = range.from();
            (p, find_other_end(doc, p))
        } else {
            let p = range.from();
            (find_other_end(doc, p), p)
        };
        if del_from == del_to {
            let new_pos = (range.from() as isize + shift) as usize;
            new_positions.push((orig_idx, new_pos));
            continue;
        }
        changes.push(Change::delete(del_from, del_to));
        new_positions.push((orig_idx, (del_from as isize + shift) as usize));
        shift -= (del_to - del_from) as isize;
    }

    if changes.is_empty() {
        return None;
    }
    new_positions.sort_by_key(|(idx, _)| *idx);
    let new_ranges = new_positions
        .iter()
        .map(|(_, p)| Range::cursor(*p))
        .collect();
    Some(Transaction {
        changes,
        selection: Some(Selection::new(new_ranges, state.selection.primary)),
    })
}

/// Delete from each cursor to the end of its line. If a selection is
/// non-empty it's deleted instead.
pub fn delete_to_line_end(state: &EditorState) -> Option<Transaction> {
    delete_by(state, true, |doc, p| {
        let line = doc.char_to_line(p);
        if line + 1 < doc.len_lines() {
            let next = doc.line_to_char(line + 1);
            if next > 0 && doc.char(next - 1) == '\n' {
                next - 1
            } else {
                next
            }
        } else {
            doc.len_chars()
        }
    })
}

/// Delete from each cursor back to the start of its line.
pub fn delete_to_line_start(state: &EditorState) -> Option<Transaction> {
    delete_by(state, false, |doc, p| {
        doc.line_to_char(doc.char_to_line(p))
    })
}

/// Duplicate each line containing a selection (or a cursor) and place the
/// copy *above* the original.
pub fn copy_line_up(state: &EditorState) -> Option<Transaction> {
    copy_lines(state, true)
}

/// Duplicate each line containing a selection (or a cursor) and place the
/// copy *below* the original.
pub fn copy_line_down(state: &EditorState) -> Option<Transaction> {
    copy_lines(state, false)
}

fn copy_lines(state: &EditorState, up: bool) -> Option<Transaction> {
    let doc = &state.doc;
    let blocks = line_blocks(state);
    if blocks.is_empty() {
        return None;
    }
    let mut changes = Vec::new();
    let mut shifts: Vec<(usize, isize)> = Vec::new();
    for (a, b) in blocks {
        let block_start = doc.line_to_char(a);
        let block_end = if b + 1 < doc.len_lines() {
            doc.line_to_char(b + 1)
        } else {
            doc.len_chars()
        };
        let block_text: String = doc.slice(block_start..block_end).to_string();
        let needs_nl = !block_text.ends_with('\n');
        let (insert_at, insert_text) = if up {
            // Place the duplicate before the original. Ensure it ends with a
            // newline so the original starts on a fresh line.
            let mut t = block_text.clone();
            if needs_nl {
                t.push('\n');
            }
            (block_start, t)
        } else if !needs_nl {
            // Down-copy of a normal line — append after the trailing \n.
            (block_end, block_text.clone())
        } else {
            // Down-copy of the doc-final block lacking \n: emit "\nORIG"
            // after the original so we get `original\nduplicate`.
            (block_end, format!("\n{}", block_text))
        };
        let insert_chars = insert_text.chars().count() as isize;
        changes.push(Change::insert(insert_at, insert_text));
        // Both up and down move the user's cursor onto the *new* copy:
        //  * up   — original moves down by `insert_chars`, so positions in
        //           the block shift by +insert_chars.
        //  * down — duplicate sits below the original, so positions in the
        //           block shift by +insert_chars to land on the copy.
        shifts.push((block_start, insert_chars));
    }
    let selection = rebuild_selection(state, &shifts);
    Some(Transaction { changes, selection: Some(selection) })
}

/// Swap the character before each cursor with the one after. Mirrors emacs
/// `transpose-chars`. No-op at doc bounds. If the cursor is at end-of-line
/// (and there's a previous char on the line), swaps the two chars on the
/// preceding line — same as emacs.
pub fn transpose_chars(state: &EditorState) -> Option<Transaction> {
    let doc = &state.doc;
    let mut changes = Vec::new();
    let mut indexed: Vec<(usize, Range)> = state
        .selection
        .ranges
        .iter()
        .copied()
        .enumerate()
        .collect();
    indexed.sort_by_key(|(_, r)| r.from());
    let mut new_positions: Vec<(usize, usize)> = Vec::new();
    let shift: isize = 0;
    for (orig_idx, r) in indexed {
        if !r.is_empty() {
            new_positions.push((orig_idx, (r.head as isize + shift) as usize));
            continue;
        }
        let p = r.from();
        if p == 0 || p == doc.len_chars() || doc.char(p - 1) == '\n' || doc.char(p) == '\n' {
            new_positions.push((orig_idx, (p as isize + shift) as usize));
            continue;
        }
        let a = doc.char(p - 1);
        let b = doc.char(p);
        let swapped: String = format!("{}{}", b, a);
        changes.push(Change::new(p - 1, p + 1, swapped));
        new_positions.push((orig_idx, (p as isize + shift + 1) as usize));
    }
    if changes.is_empty() {
        return None;
    }
    new_positions.sort_by_key(|(idx, _)| *idx);
    let new_ranges = new_positions
        .iter()
        .map(|(_, p)| Range::cursor(*p))
        .collect();
    Some(Transaction {
        changes,
        selection: Some(Selection::new(new_ranges, state.selection.primary)),
    })
}

/// Delete each line containing a selection (or a cursor). Multiple cursors
/// on the same line collapse to one delete; line + 1 chars deleted (the line
/// content + its trailing newline). Mirrors `deleteLine` in CM6.
pub fn delete_line(state: &EditorState) -> Option<Transaction> {
    use std::collections::BTreeSet;
    let doc = &state.doc;
    let mut lines: BTreeSet<usize> = BTreeSet::new();
    for r in &state.selection.ranges {
        let from_l = doc.char_to_line(r.from());
        let to_l = doc.char_to_line(r.to());
        for l in from_l..=to_l {
            lines.insert(l);
        }
    }
    if lines.is_empty() {
        return None;
    }
    // Merge consecutive line indices into ranges so we delete each block as
    // one change.
    let mut blocks: Vec<(usize, usize)> = Vec::new();
    for l in &lines {
        match blocks.last_mut() {
            Some(b) if b.1 + 1 == *l => b.1 = *l,
            _ => blocks.push((*l, *l)),
        }
    }
    let mut changes = Vec::new();
    for (a, b) in &blocks {
        let from = doc.line_to_char(*a);
        let to = if b + 1 < doc.len_lines() {
            doc.line_to_char(b + 1)
        } else {
            // Last line: also remove the leading newline, if any, so the
            // previous line doesn't end up with an unwanted trailing \n.
            doc.len_chars()
        };
        if from < to {
            changes.push(Change::delete(from, to));
        }
    }
    if changes.is_empty() {
        return None;
    }
    Some(Transaction { changes, selection: None })
}

// ---- Bracket matching --------------------------------------------------

fn matching_open_for(state: &EditorState, close: char) -> Option<char> {
    state
        .indent_rules
        .pairs
        .iter()
        .find(|(_, c)| *c == close)
        .map(|p| p.0)
}

fn scan_open_to_close(doc: &Rope, start: usize, open: char, close: char) -> Option<usize> {
    let mut depth = 1isize;
    let mut i = start;
    while i < doc.len_chars() {
        let c = doc.char(i);
        if c == open {
            depth += 1;
        } else if c == close {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}

fn scan_close_to_open(doc: &Rope, start_exclusive: usize, open: char, close: char) -> Option<usize> {
    let mut depth = 1isize;
    let mut i = start_exclusive;
    while i > 0 {
        i -= 1;
        let c = doc.char(i);
        if c == close {
            depth += 1;
        } else if c == open {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
    }
    None
}

/// Find the matching bracket position for the bracket at or just before `pos`.
fn find_matching_bracket(state: &EditorState, pos: usize) -> Option<usize> {
    let doc = &state.doc;
    let len = doc.len_chars();
    let pick = |i: usize| -> Option<(usize, char)> {
        if i < len {
            let c = doc.char(i);
            if state.indent_rules.open.contains(&c) || state.indent_rules.close.contains(&c) {
                return Some((i, c));
            }
        }
        None
    };
    let (bracket_pos, c) = pick(pos)
        .or_else(|| if pos > 0 { pick(pos - 1) } else { None })?;
    if let Some(close) = state.indent_rules.matching_close(c) {
        scan_open_to_close(doc, bracket_pos + 1, c, close)
    } else if let Some(open) = matching_open_for(state, c) {
        scan_close_to_open(doc, bracket_pos, open, c)
    } else {
        None
    }
}

pub fn cursor_matching_bracket(state: &EditorState) -> Option<Transaction> {
    let new: Vec<Range> = state
        .selection
        .ranges
        .iter()
        .map(|r| match find_matching_bracket(state, r.head) {
            Some(p) => Range::cursor(p),
            None => *r,
        })
        .collect();
    let new_sel = Selection::new(new, state.selection.primary);
    if new_sel == state.selection {
        return None;
    }
    Some(Transaction::new().select(new_sel))
}

pub fn select_matching_bracket(state: &EditorState) -> Option<Transaction> {
    let new: Vec<Range> = state
        .selection
        .ranges
        .iter()
        .map(|r| match find_matching_bracket(state, r.head) {
            Some(p) => Range::new(r.anchor, p),
            None => *r,
        })
        .collect();
    let new_sel = Selection::new(new, state.selection.primary);
    if new_sel == state.selection {
        return None;
    }
    Some(Transaction::new().select(new_sel))
}

// ---- Misc edit commands -----------------------------------------------

/// Insert a `\n` at each cursor; the cursor stays *before* the newline so the
/// effect is "an empty line opens up below me" — Ctrl-O / Cmd-Enter style.
pub fn insert_blank_line(state: &EditorState) -> Option<Transaction> {
    let mut indexed: Vec<(usize, Range)> = state
        .selection
        .ranges
        .iter()
        .copied()
        .enumerate()
        .collect();
    indexed.sort_by_key(|(_, r)| r.from());

    let mut changes = Vec::new();
    let mut new_positions: Vec<(usize, usize)> = Vec::new();
    let mut shift: isize = 0;

    for (orig_idx, r) in indexed {
        let p = r.from();
        changes.push(Change::insert(p, "\n"));
        new_positions.push((orig_idx, (p as isize + shift) as usize));
        shift += 1;
    }
    new_positions.sort_by_key(|(idx, _)| *idx);
    let new_ranges = new_positions
        .iter()
        .map(|(_, p)| Range::cursor(*p))
        .collect();
    Some(Transaction {
        changes,
        selection: Some(Selection::new(new_ranges, state.selection.primary)),
    })
}

/// Dispatch to `toggle_line_comment` if a line token is configured, else
/// `toggle_block_comment`. CM6 calls this `toggleComment`.
pub fn toggle_comment(state: &EditorState) -> Option<Transaction> {
    if state.comment_tokens.line.is_some() {
        crate::comment::toggle_line_comment(state)
    } else if state.comment_tokens.block.is_some() {
        crate::comment::toggle_block_comment(state)
    } else {
        None
    }
}

// ---- Subword motion ---------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SubwordKind {
    Lower,
    Upper,
    Digit,
    Underscore,
    Other,
}

fn subword_kind(c: char) -> SubwordKind {
    if c == '_' {
        SubwordKind::Underscore
    } else if c.is_ascii_digit() {
        SubwordKind::Digit
    } else if c.is_alphabetic() && c.is_uppercase() {
        SubwordKind::Upper
    } else if c.is_alphabetic() && c.is_lowercase() {
        SubwordKind::Lower
    } else {
        SubwordKind::Other
    }
}

/// Find the next subword boundary going forward from `p`.
/// Recognises camelCase, snake_case, ALLCAPS-prefix runs (XMLHttp), and
/// digit/letter transitions.
fn next_subword_boundary_forward(doc: &Rope, p: usize) -> usize {
    let len = doc.len_chars();
    let mut q = p;
    while q < len && matches!(subword_kind(doc.char(q)), SubwordKind::Other) {
        q += 1;
    }
    if q >= len {
        return q;
    }
    let first = subword_kind(doc.char(q));
    if first == SubwordKind::Underscore {
        return q + 1;
    }
    q += 1;
    match first {
        SubwordKind::Upper => {
            // Could be Camel (Upper+Lowers) or ALLCAPS (run of Uppers).
            if q < len && subword_kind(doc.char(q)) == SubwordKind::Upper {
                while q < len && subword_kind(doc.char(q)) == SubwordKind::Upper {
                    q += 1;
                }
                // Trailing Lower means the last Upper actually starts the
                // next Camel subword (XMLHttp → boundary before H).
                if q < len && subword_kind(doc.char(q)) == SubwordKind::Lower {
                    return q - 1;
                }
                q
            } else {
                while q < len {
                    let k = subword_kind(doc.char(q));
                    if matches!(k, SubwordKind::Lower | SubwordKind::Digit) {
                        q += 1;
                    } else {
                        break;
                    }
                }
                q
            }
        }
        SubwordKind::Lower => {
            while q < len {
                let k = subword_kind(doc.char(q));
                if matches!(k, SubwordKind::Lower | SubwordKind::Digit) {
                    q += 1;
                } else {
                    break;
                }
            }
            q
        }
        SubwordKind::Digit => {
            while q < len && subword_kind(doc.char(q)) == SubwordKind::Digit {
                q += 1;
            }
            q
        }
        _ => q,
    }
}

fn next_subword_boundary_backward(doc: &Rope, p: usize) -> usize {
    let mut q = p;
    while q > 0 && matches!(subword_kind(doc.char(q - 1)), SubwordKind::Other) {
        q -= 1;
    }
    if q == 0 {
        return 0;
    }
    let last = subword_kind(doc.char(q - 1));
    if last == SubwordKind::Underscore {
        return q - 1;
    }
    q -= 1;
    match last {
        SubwordKind::Lower => {
            // Walk back through Lower/Digit; absorb a leading Upper if one
            // is present (Camel-case prefix).
            while q > 0 {
                let k = subword_kind(doc.char(q - 1));
                if matches!(k, SubwordKind::Lower | SubwordKind::Digit) {
                    q -= 1;
                } else {
                    break;
                }
            }
            if q > 0 && subword_kind(doc.char(q - 1)) == SubwordKind::Upper {
                q -= 1;
            }
            q
        }
        SubwordKind::Upper => {
            while q > 0 && subword_kind(doc.char(q - 1)) == SubwordKind::Upper {
                q -= 1;
            }
            q
        }
        SubwordKind::Digit => {
            while q > 0 && subword_kind(doc.char(q - 1)) == SubwordKind::Digit {
                q -= 1;
            }
            q
        }
        _ => q,
    }
}

pub fn cursor_subword_right(state: &EditorState) -> Option<Transaction> {
    let new: Vec<Range> = state
        .selection
        .ranges
        .iter()
        .map(|r| Range::cursor(next_subword_boundary_forward(&state.doc, r.head)))
        .collect();
    let new_sel = Selection::new(new, state.selection.primary);
    if new_sel == state.selection {
        return None;
    }
    Some(Transaction::new().select(new_sel))
}

pub fn cursor_subword_left(state: &EditorState) -> Option<Transaction> {
    let new: Vec<Range> = state
        .selection
        .ranges
        .iter()
        .map(|r| Range::cursor(next_subword_boundary_backward(&state.doc, r.head)))
        .collect();
    let new_sel = Selection::new(new, state.selection.primary);
    if new_sel == state.selection {
        return None;
    }
    Some(Transaction::new().select(new_sel))
}

pub fn select_subword_right(state: &EditorState) -> Option<Transaction> {
    let new: Vec<Range> = state
        .selection
        .ranges
        .iter()
        .map(|r| Range::new(r.anchor, next_subword_boundary_forward(&state.doc, r.head)))
        .collect();
    let new_sel = Selection::new(new, state.selection.primary);
    if new_sel == state.selection {
        return None;
    }
    Some(Transaction::new().select(new_sel))
}

pub fn select_subword_left(state: &EditorState) -> Option<Transaction> {
    let new: Vec<Range> = state
        .selection
        .ranges
        .iter()
        .map(|r| Range::new(r.anchor, next_subword_boundary_backward(&state.doc, r.head)))
        .collect();
    let new_sel = Selection::new(new, state.selection.primary);
    if new_sel == state.selection {
        return None;
    }
    Some(Transaction::new().select(new_sel))
}

// ---- Word motion -------------------------------------------------------

/// Find the position of the next word boundary going forward from `p`.
/// Skips trailing whitespace then runs through one same-class run.
fn next_word_boundary_forward(doc: &Rope, p: usize) -> usize {
    let len = doc.len_chars();
    let mut q = p;
    // Skip whitespace (non-newline) and newlines.
    while q < len {
        let c = doc.char(q);
        if classify(c) == CharClass::Space || classify(c) == CharClass::Newline {
            q += 1;
        } else {
            break;
        }
    }
    if q >= len {
        return q;
    }
    let cls = classify(doc.char(q));
    while q < len && classify(doc.char(q)) == cls {
        q += 1;
    }
    q
}

fn next_word_boundary_backward(doc: &Rope, p: usize) -> usize {
    let mut q = p;
    while q > 0 {
        let c = doc.char(q - 1);
        if classify(c) == CharClass::Space || classify(c) == CharClass::Newline {
            q -= 1;
        } else {
            break;
        }
    }
    if q == 0 {
        return 0;
    }
    let cls = classify(doc.char(q - 1));
    while q > 0 && classify(doc.char(q - 1)) == cls {
        q -= 1;
    }
    q
}

// ---- Vertical cursor motion -------------------------------------------

/// Move/extend each cursor up/down by `delta_lines`. Without goal-column
/// tracking yet — column is just clamped to the target line's length. CM6's
/// real implementation tracks an "intended" column so going through a short
/// line and back to a long one restores the original column.
fn move_lines(state: &EditorState, delta_lines: isize, extend: bool) -> Option<Transaction> {
    let doc = &state.doc;
    let last = last_real_line(doc);
    let new_ranges: Vec<Range> = state
        .selection
        .ranges
        .iter()
        .map(|r| {
            let line = doc.char_to_line(r.head);
            let col = r.head - doc.line_to_char(line);
            let target_line_signed = line as isize + delta_lines;
            if target_line_signed < 0 {
                if extend { return Range::new(r.anchor, 0); }
                return Range::cursor(0);
            }
            let target_line = (target_line_signed as usize).min(last);
            let target_start = doc.line_to_char(target_line);
            let target_end = if target_line + 1 < doc.len_lines() {
                let next = doc.line_to_char(target_line + 1);
                if next > 0 && doc.char(next - 1) == '\n' { next - 1 } else { next }
            } else {
                doc.len_chars()
            };
            let pos = (target_start + col).min(target_end);
            if extend { Range::new(r.anchor, pos) } else { Range::cursor(pos) }
        })
        .collect();
    let new_sel = Selection::new(new_ranges, state.selection.primary);
    if new_sel == state.selection {
        return None;
    }
    Some(Transaction::new().select(new_sel))
}

pub fn cursor_line_up(state: &EditorState) -> Option<Transaction> {
    move_lines(state, -1, false)
}

pub fn cursor_line_down(state: &EditorState) -> Option<Transaction> {
    move_lines(state, 1, false)
}

pub fn select_line_up(state: &EditorState) -> Option<Transaction> {
    move_lines(state, -1, true)
}

pub fn select_line_down(state: &EditorState) -> Option<Transaction> {
    move_lines(state, 1, true)
}

// ---- Tab key handling ------------------------------------------------

/// Insert a literal `\t` at every cursor (replacing any selection). Mirrors
/// CM6's `insertTab` for the cursor-only case.
pub fn insert_tab(state: &EditorState) -> Option<Transaction> {
    let mut indexed: Vec<(usize, Range)> = state
        .selection
        .ranges
        .iter()
        .copied()
        .enumerate()
        .collect();
    indexed.sort_by_key(|(_, r)| r.from());
    let mut changes = Vec::new();
    let mut new_positions: Vec<(usize, usize)> = Vec::new();
    let mut shift: isize = 0;
    for (orig_idx, r) in indexed {
        let from = r.from();
        let to = r.to();
        let removed = (to - from) as isize;
        changes.push(Change::new(from, to, "\t".to_string()));
        let new_pos = (from as isize + shift + 1) as usize;
        new_positions.push((orig_idx, new_pos));
        shift += 1 - removed;
    }
    new_positions.sort_by_key(|(idx, _)| *idx);
    let new_ranges = new_positions
        .iter()
        .map(|(_, p)| Range::cursor(*p))
        .collect();
    Some(Transaction {
        changes,
        selection: Some(Selection::new(new_ranges, state.selection.primary)),
    })
}

/// Smart Tab handler: if any range covers more than one line, indent_more;
/// otherwise insert a tab/indent_unit at the cursor. CM6 calls this
/// `indentWithTab`.
pub fn indent_with_tab(state: &EditorState) -> Option<Transaction> {
    let doc = &state.doc;
    let multiline = state.selection.ranges.iter().any(|r| {
        if r.is_empty() {
            return false;
        }
        doc.char_to_line(r.from()) != doc.char_to_line(r.to())
    });
    if multiline {
        indent_more(state)
    } else {
        insert_tab(state)
    }
}

/// Same as `delete_char_backward` — CM6's "strict" variant only differs in
/// how it handles UTF-16 surrogate pairs, which our char-indexed model
/// doesn't expose.
pub fn delete_char_backward_strict(state: &EditorState) -> Option<Transaction> {
    delete_char_backward(state)
}

// ---- Aliases / direction-agnostic versions ----------------------------

/// Alias for `cursor_word_right` — CM6 also exposes this as `cursorGroupForward`.
pub fn cursor_group_forward(state: &EditorState) -> Option<Transaction> {
    cursor_word_right(state)
}

pub fn cursor_group_backward(state: &EditorState) -> Option<Transaction> {
    cursor_word_left(state)
}

pub fn select_group_forward(state: &EditorState) -> Option<Transaction> {
    select_word_right(state)
}

pub fn select_group_backward(state: &EditorState) -> Option<Transaction> {
    select_word_left(state)
}

// CM6 has bidi-aware `*Left`/`*Right` variants that pick the visual side.
// Without bidi support we treat them as direct LTR aliases of left/right.
pub fn cursor_char_backward(state: &EditorState) -> Option<Transaction> {
    cursor_char_left(state)
}

pub fn cursor_char_forward(state: &EditorState) -> Option<Transaction> {
    cursor_char_right(state)
}

pub fn cursor_char_backward_logical(state: &EditorState) -> Option<Transaction> {
    cursor_char_left(state)
}

pub fn cursor_char_forward_logical(state: &EditorState) -> Option<Transaction> {
    cursor_char_right(state)
}

pub fn select_char_backward(state: &EditorState) -> Option<Transaction> {
    select_char_left(state)
}

pub fn select_char_forward(state: &EditorState) -> Option<Transaction> {
    select_char_right(state)
}

pub fn select_char_backward_logical(state: &EditorState) -> Option<Transaction> {
    select_char_left(state)
}

pub fn select_char_forward_logical(state: &EditorState) -> Option<Transaction> {
    select_char_right(state)
}

pub fn cursor_group_left(state: &EditorState) -> Option<Transaction> {
    cursor_word_left(state)
}

pub fn cursor_group_right(state: &EditorState) -> Option<Transaction> {
    cursor_word_right(state)
}

pub fn select_group_left(state: &EditorState) -> Option<Transaction> {
    select_word_left(state)
}

pub fn select_group_right(state: &EditorState) -> Option<Transaction> {
    select_word_right(state)
}

pub fn cursor_line_boundary_left(state: &EditorState) -> Option<Transaction> {
    cursor_line_start(state)
}

pub fn cursor_line_boundary_right(state: &EditorState) -> Option<Transaction> {
    cursor_line_end(state)
}

pub fn select_line_boundary_left(state: &EditorState) -> Option<Transaction> {
    select_line_start(state)
}

pub fn select_line_boundary_right(state: &EditorState) -> Option<Transaction> {
    select_line_end(state)
}

/// Alias for `cursor_line_start` — CM6 calls these "line boundary backward".
pub fn cursor_line_boundary_backward(state: &EditorState) -> Option<Transaction> {
    cursor_line_start(state)
}

pub fn cursor_line_boundary_forward(state: &EditorState) -> Option<Transaction> {
    cursor_line_end(state)
}

pub fn select_line_boundary_backward(state: &EditorState) -> Option<Transaction> {
    select_line_start(state)
}

pub fn select_line_boundary_forward(state: &EditorState) -> Option<Transaction> {
    select_line_end(state)
}

/// Same delete as `delete_to_line_start` — CM6's `deleteLineBoundaryBackward`
/// has the same logical-line semantics in our (no soft-wrap) model.
pub fn delete_line_boundary_backward(state: &EditorState) -> Option<Transaction> {
    delete_to_line_start(state)
}

pub fn delete_line_boundary_forward(state: &EditorState) -> Option<Transaction> {
    delete_to_line_end(state)
}

// ---- Multi-cursor: select next occurrence -----------------------------

/// Add the next occurrence of the primary range's text as another selection
/// range. If no occurrence exists, returns `None`. The primary range must
/// be non-empty.
pub fn select_next_occurrence(state: &EditorState) -> Option<Transaction> {
    let primary = state.selection.primary_range();
    if primary.is_empty() {
        return None;
    }
    let needle: String = state.doc.slice(primary.from()..primary.to()).to_string();
    let doc_str = state.doc.to_string();
    // Map char index to byte index for &str search.
    let after_primary_byte = doc_str
        .char_indices()
        .nth(primary.to())
        .map(|(b, _)| b)
        .unwrap_or(doc_str.len());
    let mut byte_match = doc_str[after_primary_byte..]
        .find(&needle)
        .map(|off| after_primary_byte + off);
    if byte_match.is_none() {
        // Wrap around, but skip if we'd land on the primary again.
        byte_match = doc_str.find(&needle).filter(|&b| {
            let char_pos = doc_str[..b].chars().count();
            char_pos != primary.from()
        });
    }
    let pos_byte = byte_match?;
    let from_char = doc_str[..pos_byte].chars().count();
    let to_char = from_char + needle.chars().count();
    let new_range = Range::new(from_char, to_char);
    if state.selection.ranges.iter().any(|r| *r == new_range) {
        return None;
    }
    let mut new_ranges = state.selection.ranges.clone();
    new_ranges.push(new_range);
    let new_primary = new_ranges.len() - 1;
    Some(Transaction::new().select(Selection::new(new_ranges, new_primary)))
}

pub fn cursor_word_right(state: &EditorState) -> Option<Transaction> {
    let new = state
        .selection
        .ranges
        .iter()
        .map(|r| Range::cursor(next_word_boundary_forward(&state.doc, r.head)))
        .collect();
    let new_sel = Selection::new(new, state.selection.primary);
    if new_sel == state.selection {
        return None;
    }
    Some(Transaction::new().select(new_sel))
}

pub fn cursor_word_left(state: &EditorState) -> Option<Transaction> {
    let new = state
        .selection
        .ranges
        .iter()
        .map(|r| Range::cursor(next_word_boundary_backward(&state.doc, r.head)))
        .collect();
    let new_sel = Selection::new(new, state.selection.primary);
    if new_sel == state.selection {
        return None;
    }
    Some(Transaction::new().select(new_sel))
}

pub fn select_word_right(state: &EditorState) -> Option<Transaction> {
    let new = state
        .selection
        .ranges
        .iter()
        .map(|r| Range::new(r.anchor, next_word_boundary_forward(&state.doc, r.head)))
        .collect();
    let new_sel = Selection::new(new, state.selection.primary);
    if new_sel == state.selection {
        return None;
    }
    Some(Transaction::new().select(new_sel))
}

pub fn select_word_left(state: &EditorState) -> Option<Transaction> {
    let new = state
        .selection
        .ranges
        .iter()
        .map(|r| Range::new(r.anchor, next_word_boundary_backward(&state.doc, r.head)))
        .collect();
    let new_sel = Selection::new(new, state.selection.primary);
    if new_sel == state.selection {
        return None;
    }
    Some(Transaction::new().select(new_sel))
}

// ---- Selection / cursor motion -----------------------------------------

/// Select the entire document.
pub fn select_all(state: &EditorState) -> Option<Transaction> {
    let len = state.doc.len_chars();
    let new = Range::new(0, len);
    if state.selection.ranges.len() == 1 && state.selection.ranges[0] == new {
        return None;
    }
    Some(Transaction::new().select(Selection::single(new)))
}

/// Extend each range to cover the entire line(s) it touches. Adjacent
/// extended ranges merge in `Selection::new`.
pub fn select_line(state: &EditorState) -> Option<Transaction> {
    let doc = &state.doc;
    let new_ranges: Vec<Range> = state
        .selection
        .ranges
        .iter()
        .map(|r| {
            let from_line = doc.char_to_line(r.from());
            let to = r.to();
            let to_line_idx = doc.char_to_line(to);
            let to_line = if !r.is_empty()
                && to_line_idx > from_line
                && to == doc.line_to_char(to_line_idx)
            {
                to_line_idx - 1
            } else {
                to_line_idx
            };
            let line_start = doc.line_to_char(from_line);
            let line_end = if to_line + 1 < doc.len_lines() {
                doc.line_to_char(to_line + 1)
            } else {
                doc.len_chars()
            };
            Range::new(line_start, line_end)
        })
        .collect();
    let new_sel = Selection::new(new_ranges, state.selection.primary);
    if new_sel == state.selection {
        return None;
    }
    Some(Transaction::new().select(new_sel))
}

/// Reduce the selection to just the primary range.
pub fn simplify_selection(state: &EditorState) -> Option<Transaction> {
    if state.selection.ranges.len() < 2 {
        return None;
    }
    let new = Selection::single(state.selection.primary_range());
    Some(Transaction::new().select(new))
}

/// Split each multi-line range into one range per line.
pub fn split_selection_by_line(state: &EditorState) -> Option<Transaction> {
    let doc = &state.doc;
    let mut new_ranges: Vec<Range> = Vec::new();
    let mut new_primary = 0;
    for (i, r) in state.selection.ranges.iter().enumerate() {
        let from_line = doc.char_to_line(r.from());
        let to_line = doc.char_to_line(r.to());
        if from_line == to_line {
            if i == state.selection.primary {
                new_primary = new_ranges.len();
            }
            new_ranges.push(*r);
            continue;
        }
        if i == state.selection.primary {
            new_primary = new_ranges.len();
        }
        // First line: from r.from() to end-of-line.
        for line in from_line..=to_line {
            let line_start = doc.line_to_char(line);
            let line_end = if line + 1 < doc.len_lines() {
                let next = doc.line_to_char(line + 1);
                if next > 0 && doc.char(next - 1) == '\n' {
                    next - 1
                } else {
                    next
                }
            } else {
                doc.len_chars()
            };
            let seg_from = if line == from_line { r.from() } else { line_start };
            let seg_to = if line == to_line { r.to() } else { line_end };
            if seg_from <= seg_to {
                new_ranges.push(Range::new(seg_from, seg_to));
            }
        }
    }
    if new_ranges.len() == state.selection.ranges.len() {
        return None;
    }
    let new_sel = Selection::new(new_ranges, new_primary.min(0));
    if new_sel == state.selection {
        return None;
    }
    // primary may have been clobbered by Selection::new merge; just use 0.
    let new_sel = Selection::new(new_sel.ranges, 0);
    Some(Transaction::new().select(new_sel))
}

/// Add a cursor on the line above each existing range, at the same column.
/// If there is no line above, that range is dropped.
pub fn add_cursor_above(state: &EditorState) -> Option<Transaction> {
    add_cursor_vertical(state, /*above=*/ true)
}

pub fn add_cursor_below(state: &EditorState) -> Option<Transaction> {
    add_cursor_vertical(state, /*above=*/ false)
}

fn add_cursor_vertical(state: &EditorState, above: bool) -> Option<Transaction> {
    let doc = &state.doc;
    let mut new_ranges: Vec<Range> = state.selection.ranges.iter().copied().collect();
    let mut added = false;
    for r in &state.selection.ranges {
        let head = r.head;
        let line = doc.char_to_line(head);
        let target_line: Option<usize> = if above {
            if line == 0 { None } else { Some(line - 1) }
        } else if line + 1 < doc.len_lines() {
            // Skip the phantom trailing-empty-line if doc ends in '\n'.
            let next = line + 1;
            if next < doc.len_lines() {
                Some(next)
            } else {
                None
            }
        } else {
            None
        };
        let Some(target_line) = target_line else { continue };
        if !above && target_line >= last_real_line(doc) + 1 {
            continue;
        }
        let line_start = doc.line_to_char(line);
        let col = head - line_start;
        let target_start = doc.line_to_char(target_line);
        let target_end = if target_line + 1 < doc.len_lines() {
            let next = doc.line_to_char(target_line + 1);
            if next > 0 && doc.char(next - 1) == '\n' {
                next - 1
            } else {
                next
            }
        } else {
            doc.len_chars()
        };
        let target_pos = (target_start + col).min(target_end);
        new_ranges.push(Range::cursor(target_pos));
        added = true;
    }
    if !added {
        return None;
    }
    let new_sel = Selection::new(new_ranges, state.selection.primary);
    if new_sel == state.selection {
        return None;
    }
    Some(Transaction::new().select(new_sel))
}

/// Insert a newline at each cursor without moving the cursor (cursor stays
/// before the inserted `\n`). Mirrors CM6's `splitLine`.
pub fn split_line(state: &EditorState) -> Option<Transaction> {
    let mut indexed: Vec<(usize, Range)> = state
        .selection
        .ranges
        .iter()
        .copied()
        .enumerate()
        .collect();
    indexed.sort_by_key(|(_, r)| r.from());

    let mut changes = Vec::new();
    let mut new_positions: Vec<(usize, usize)> = Vec::new();
    let mut shift: isize = 0;

    for (orig_idx, range) in indexed {
        let from = range.from();
        let to = range.to();
        let removed = (to - from) as isize;
        changes.push(Change::new(from, to, "\n".to_string()));
        let new_pos = (from as isize + shift) as usize;
        new_positions.push((orig_idx, new_pos));
        shift += 1 - removed;
    }

    new_positions.sort_by_key(|(idx, _)| *idx);
    let new_ranges: Vec<Range> = new_positions
        .iter()
        .map(|(_, p)| Range::cursor(*p))
        .collect();

    Some(Transaction {
        changes,
        selection: Some(Selection::new(new_ranges, state.selection.primary)),
    })
}

/// Move/extend each range's head by `delta` chars (negative for left).
fn move_each(
    state: &EditorState,
    extend: bool,
    f: impl Fn(usize, &EditorState) -> usize,
) -> Option<Transaction> {
    let new_ranges: Vec<Range> = state
        .selection
        .ranges
        .iter()
        .map(|r| {
            let new_head = f(r.head, state);
            if extend {
                Range::new(r.anchor, new_head)
            } else {
                Range::cursor(new_head)
            }
        })
        .collect();
    let new_sel = Selection::new(new_ranges, state.selection.primary);
    if new_sel == state.selection {
        return None;
    }
    Some(Transaction::new().select(new_sel))
}

fn clamp_minus(p: usize) -> usize {
    p.saturating_sub(1)
}

pub fn cursor_char_left(state: &EditorState) -> Option<Transaction> {
    move_each(state, false, |p, _| clamp_minus(p))
}

pub fn cursor_char_right(state: &EditorState) -> Option<Transaction> {
    move_each(state, false, |p, s| (p + 1).min(s.doc.len_chars()))
}

pub fn select_char_left(state: &EditorState) -> Option<Transaction> {
    move_each(state, true, |p, _| clamp_minus(p))
}

pub fn select_char_right(state: &EditorState) -> Option<Transaction> {
    move_each(state, true, |p, s| (p + 1).min(s.doc.len_chars()))
}

pub fn cursor_line_start(state: &EditorState) -> Option<Transaction> {
    move_each(state, false, |p, s| s.doc.line_to_char(s.doc.char_to_line(p)))
}

pub fn cursor_line_end(state: &EditorState) -> Option<Transaction> {
    move_each(state, false, |p, s| {
        let line = s.doc.char_to_line(p);
        if line + 1 < s.doc.len_lines() {
            // line_to_char(line + 1) is start of next line; back off the \n.
            let next = s.doc.line_to_char(line + 1);
            if next > 0 && s.doc.char(next - 1) == '\n' {
                next - 1
            } else {
                next
            }
        } else {
            s.doc.len_chars()
        }
    })
}

pub fn select_line_start(state: &EditorState) -> Option<Transaction> {
    move_each(state, true, |p, s| s.doc.line_to_char(s.doc.char_to_line(p)))
}

pub fn select_line_end(state: &EditorState) -> Option<Transaction> {
    move_each(state, true, |p, s| {
        let line = s.doc.char_to_line(p);
        if line + 1 < s.doc.len_lines() {
            let next = s.doc.line_to_char(line + 1);
            if next > 0 && s.doc.char(next - 1) == '\n' {
                next - 1
            } else {
                next
            }
        } else {
            s.doc.len_chars()
        }
    })
}

pub fn cursor_doc_start(state: &EditorState) -> Option<Transaction> {
    move_each(state, false, |_, _| 0)
}

pub fn cursor_doc_end(state: &EditorState) -> Option<Transaction> {
    move_each(state, false, |_, s| s.doc.len_chars())
}

pub fn select_doc_start(state: &EditorState) -> Option<Transaction> {
    move_each(state, true, |_, _| 0)
}

pub fn select_doc_end(state: &EditorState) -> Option<Transaction> {
    move_each(state, true, |_, s| s.doc.len_chars())
}

// ---- Bracket-based indentation ----------------------------------------

/// Read the leading whitespace string from a line.
fn leading_ws(line: &str) -> String {
    line.chars().take_while(|c| *c == ' ' || *c == '\t').collect()
}

/// Last non-whitespace char in `s`, or None.
fn last_non_ws(s: &str) -> Option<char> {
    s.chars().rev().find(|c| !c.is_whitespace())
}

/// First non-whitespace char in `s`, or None.
fn first_non_ws(s: &str) -> Option<char> {
    s.chars().find(|c| !c.is_whitespace())
}

/// Compute the indent string for a NEW line that follows `prev_text` (the
/// effective text of the previous line, used to detect trailing brackets and
/// read leading whitespace). Adds one `indent_unit` if the prev line ends
/// with an "open" bracket.
fn indent_after(state: &EditorState, prev_text: &str) -> String {
    let mut indent = leading_ws(prev_text);
    if last_non_ws(prev_text)
        .map_or(false, |c| state.indent_rules.open.contains(&c))
    {
        indent.push_str(&state.indent_unit);
    }
    indent
}

pub fn insert_newline_and_indent(state: &EditorState) -> Option<Transaction> {
    let doc = &state.doc;
    let n = state.selection.ranges.len();
    let mut indexed: Vec<(usize, Range)> = state
        .selection
        .ranges
        .iter()
        .copied()
        .enumerate()
        .collect();
    indexed.sort_by_key(|(_, r)| r.from());

    let mut changes = Vec::with_capacity(n);
    let mut new_positions: Vec<(usize, usize)> = Vec::with_capacity(n);
    let mut shift: isize = 0;

    for (orig_idx, range) in indexed {
        let from = range.from();
        let to = range.to();

        // Try bracket explosion first. Only when range is empty.
        if range.is_empty() {
            if let Some(action) = try_explode(state, from) {
                let removed = (action.delete_to - action.delete_from) as isize;
                let insert_chars = action.insert.chars().count() as isize;
                let cursor_offset = action.cursor_offset as isize;
                changes.push(Change::new(
                    action.delete_from,
                    action.delete_to,
                    action.insert,
                ));
                let new_pos =
                    (action.delete_from as isize + shift + cursor_offset) as usize;
                new_positions.push((orig_idx, new_pos));
                shift += insert_chars - removed;
                continue;
            }
        }

        let line_idx = doc.char_to_line(from);
        let line_start = doc.line_to_char(line_idx);
        let prefix: String = doc.slice(line_start..from).to_string();
        // The "clear leading whitespace" branch only fires for actual
        // whitespace — an *empty* prefix (cursor at column 0) falls through
        // to the normal newline-and-indent path, otherwise we'd insert
        // `"\n    "` at column 0 and accidentally push the next line's
        // content to the right.
        let prefix_is_ws = !prefix.is_empty() && prefix.chars().all(|c| c == ' ' || c == '\t');

        if prefix_is_ws {
            // Clear the leading whitespace on this line, then add a blank
            // line with the proper indent (based on previous non-cleared
            // line's content).
            let indent = if line_idx == 0 {
                String::new()
            } else {
                let prev_start = doc.line_to_char(line_idx - 1);
                let prev_end = doc.line_to_char(line_idx);
                let prev_text: String = doc.slice(prev_start..prev_end).to_string();
                let prev_no_nl = prev_text.strip_suffix('\n').unwrap_or(&prev_text).to_string();
                indent_after(state, &prev_no_nl)
            };
            let insert = format!("\n{}", indent);
            let removed = (to - line_start) as isize;
            let insert_chars = insert.chars().count() as isize;
            changes.push(Change::new(line_start, to, insert));
            let new_pos = (line_start as isize + shift + insert_chars) as usize;
            new_positions.push((orig_idx, new_pos));
            shift += insert_chars - removed;
        } else {
            let indent = indent_after(state, &prefix);
            let insert = format!("\n{}", indent);
            let removed = (to - from) as isize;
            let insert_chars = insert.chars().count() as isize;
            changes.push(Change::new(from, to, insert));
            let new_pos = (from as isize + shift + insert_chars) as usize;
            new_positions.push((orig_idx, new_pos));
            shift += insert_chars - removed;
        }
    }

    if changes.is_empty() {
        return None;
    }
    new_positions.sort_by_key(|(idx, _)| *idx);
    let new_ranges = new_positions
        .iter()
        .map(|(_, p)| Range::cursor(*p))
        .collect();
    Some(Transaction {
        changes,
        selection: Some(Selection::new(new_ranges, state.selection.primary)),
    })
}

struct ExplodeAction {
    delete_from: usize,
    delete_to: usize,
    insert: String,
    /// chars from delete_from to where the cursor should land in the insert.
    cursor_offset: usize,
}

fn try_explode(state: &EditorState, p: usize) -> Option<ExplodeAction> {
    let doc = &state.doc;
    let len = doc.len_chars();
    if p == 0 || p >= len {
        return None;
    }
    // Walk back over space/tab (NOT newlines) to find the open bracket.
    let mut left = p - 1;
    while left > 0 {
        let c = doc.char(left);
        if c == ' ' || c == '\t' {
            left -= 1;
        } else {
            break;
        }
    }
    let open = doc.char(left);
    let expected_close = state.indent_rules.matching_close(open)?;
    // Walk forward over space/tab to find the close.
    let mut right = p;
    while right < len {
        let c = doc.char(right);
        if c == ' ' || c == '\t' {
            right += 1;
        } else {
            break;
        }
    }
    if right >= len {
        return None;
    }
    let close = doc.char(right);
    if close != expected_close {
        return None;
    }
    // Don't re-explode if a newline already sits between the brackets.
    let between: String = doc.slice(left + 1..right).to_string();
    if between.contains('\n') {
        return None;
    }
    // Build the explosion. Use the line's leading-whitespace as the close
    // line's indent; inner indent = that + one unit.
    let line_idx = doc.char_to_line(p);
    let line_start = doc.line_to_char(line_idx);
    let line_end = if line_idx + 1 < doc.len_lines() {
        doc.line_to_char(line_idx + 1)
    } else {
        doc.len_chars()
    };
    let line_text: String = doc.slice(line_start..line_end).to_string();
    let line_no_nl = line_text.strip_suffix('\n').unwrap_or(&line_text);
    let outer_indent = leading_ws(line_no_nl);
    let inner_indent = format!("{}{}", outer_indent, state.indent_unit);
    let insert = format!("\n{}\n{}", inner_indent, outer_indent);
    let cursor_offset = 1 + inner_indent.chars().count();
    Some(ExplodeAction {
        delete_from: left + 1,
        delete_to: right,
        insert,
        cursor_offset,
    })
}

pub fn indent_selection(state: &EditorState) -> Option<Transaction> {
    let doc = &state.doc;
    let line_set = unique_line_starts(state);
    if line_set.is_empty() {
        return None;
    }
    // Collect line indices.
    let mut line_indices: Vec<usize> = line_set
        .iter()
        .map(|&start| doc.char_to_line(start))
        .collect();
    line_indices.sort_unstable();
    line_indices.dedup();

    // For each line in order, compute target indent. Track "logical previous
    // line indent" using the most recent updated line (or original if not
    // touched).
    let mut new_indents: std::collections::HashMap<usize, String> =
        std::collections::HashMap::new();
    let mut changes = Vec::new();

    for &line_idx in &line_indices {
        let line_start = doc.line_to_char(line_idx);
        let line_end = if line_idx + 1 < doc.len_lines() {
            doc.line_to_char(line_idx + 1)
        } else {
            doc.len_chars()
        };
        let line_text: String = doc.slice(line_start..line_end).to_string();
        let line_no_nl = line_text.strip_suffix('\n').unwrap_or(&line_text);
        let cur_ws_chars: usize = line_no_nl
            .chars()
            .take_while(|c| *c == ' ' || *c == '\t')
            .count();
        let cur_ws_str: String = line_no_nl.chars().take(cur_ws_chars).collect();
        let cur_content_first = first_non_ws(line_no_nl);

        // Compute target indent: walk back to find the first prior non-empty
        // line (using updated indent if applicable).
        let mut prev_indent_str = String::new();
        let mut prev_ends_with_open = false;
        for prev_l in (0..line_idx).rev() {
            let prev_start = doc.line_to_char(prev_l);
            let prev_end = doc.line_to_char(prev_l + 1);
            let mut prev_text: String = doc.slice(prev_start..prev_end).to_string();
            // If we updated this previous line, splice in the new leading ws.
            if let Some(new_ws) = new_indents.get(&prev_l) {
                let prev_no_nl = prev_text.strip_suffix('\n').unwrap_or(&prev_text);
                let prev_old_ws_chars: usize = prev_no_nl
                    .chars()
                    .take_while(|c| *c == ' ' || *c == '\t')
                    .count();
                let rest: String = prev_no_nl.chars().skip(prev_old_ws_chars).collect();
                prev_text = format!("{}{}{}", new_ws, rest, if prev_text.ends_with('\n') { "\n" } else { "" });
            }
            let prev_no_nl = prev_text.strip_suffix('\n').unwrap_or(&prev_text);
            if prev_no_nl.chars().all(char::is_whitespace) {
                continue;
            }
            prev_indent_str = leading_ws(prev_no_nl);
            prev_ends_with_open = last_non_ws(prev_no_nl)
                .map_or(false, |c| state.indent_rules.open.contains(&c));
            break;
        }

        let mut target = prev_indent_str.clone();
        if prev_ends_with_open {
            target.push_str(&state.indent_unit);
        }
        // Dedent if current line starts with a close bracket.
        if let Some(c) = cur_content_first {
            if state.indent_rules.close.contains(&c) {
                let unit_chars = state.indent_unit.chars().count();
                let target_chars = target.chars().count();
                if target_chars >= unit_chars {
                    target = target.chars().take(target_chars - unit_chars).collect();
                }
            }
        }

        new_indents.insert(line_idx, target.clone());
        if target == cur_ws_str {
            continue;
        }
        changes.push(Change::new(line_start, line_start + cur_ws_chars, target));
    }

    if changes.is_empty() {
        return None;
    }
    // Use the default position mapping. For cursors that fell inside the
    // *original* leading whitespace, this snaps them to right after the new
    // leading whitespace — which is what CM6's "moves the cursor ahead of
    // the indentation" test expects.
    Some(Transaction { changes, selection: None })
}

pub fn insert_newline_keep_indent(state: &EditorState) -> Option<Transaction> {
    let doc = &state.doc;
    let n = state.selection.ranges.len();

    // Process ranges left-to-right so we can track cumulative position shift.
    let mut indexed: Vec<(usize, Range)> = state
        .selection
        .ranges
        .iter()
        .copied()
        .enumerate()
        .collect();
    indexed.sort_by_key(|(_, r)| r.from());

    let mut changes = Vec::with_capacity(n);
    let mut new_positions: Vec<(usize, usize)> = Vec::with_capacity(n);
    let mut shift: isize = 0;

    for (orig_idx, range) in indexed {
        let from = range.from();
        let to = range.to();
        let line_idx = doc.char_to_line(from);
        let line = doc.line(line_idx);
        let mut indent = String::new();
        for c in line.chars() {
            if c == ' ' || c == '\t' {
                indent.push(c);
            } else {
                break;
            }
        }
        let mut insert = String::with_capacity(1 + indent.len());
        insert.push('\n');
        insert.push_str(&indent);
        let insert_len = insert.chars().count() as isize;
        let removed = (to - from) as isize;
        changes.push(Change::new(from, to, insert));
        let new_pos = (from as isize + shift + insert_len) as usize;
        new_positions.push((orig_idx, new_pos));
        shift += insert_len - removed;
    }

    new_positions.sort_by_key(|(idx, _)| *idx);
    let new_ranges: Vec<Range> = new_positions
        .iter()
        .map(|(_, p)| Range::cursor(*p))
        .collect();

    Some(Transaction {
        changes,
        selection: Some(Selection::new(new_ranges, state.selection.primary)),
    })
}

pub fn delete_trailing_whitespace(state: &EditorState) -> Option<Transaction> {
    let doc = &state.doc;
    let mut changes = Vec::new();
    for line_idx in 0..doc.len_lines() {
        let line_start = doc.line_to_char(line_idx);
        let line = doc.line(line_idx);
        // Strip the line ending — we only delete trailing whitespace before it.
        let mut content_len = line.len_chars();
        if line_idx + 1 < doc.len_lines() {
            // Non-final line ends in '\n' (or '\r\n', but ropey gives us '\n');
            // exclude that one char from "trailing ws" candidates.
            let last_char = line.char(content_len - 1);
            if last_char == '\n' {
                content_len -= 1;
            }
        }
        let mut trailing = 0usize;
        while trailing < content_len {
            let c = line.char(content_len - 1 - trailing);
            if c == ' ' || c == '\t' {
                trailing += 1;
            } else {
                break;
            }
        }
        if trailing > 0 {
            let from = line_start + content_len - trailing;
            let to = line_start + content_len;
            changes.push(Change::delete(from, to));
        }
    }
    if changes.is_empty() {
        None
    } else {
        Some(Transaction { changes, selection: None })
    }
}

/// Visual column at the end of `leading` (counting tabs to the next 4-stop).
fn visual_cols(leading: &str) -> usize {
    const TAB_SIZE: usize = 4;
    let mut col = 0;
    for c in leading.chars() {
        if c == '\t' {
            col = (col / TAB_SIZE + 1) * TAB_SIZE;
        } else {
            col += 1;
        }
    }
    col
}

pub fn indent_less(state: &EditorState) -> Option<Transaction> {
    // CM6's `indent_less`: dedent by `indent_unit` columns. Tabs participate
    // in the visual-column measurement and may be split into spaces if a
    // partial tab gets dedented.
    let unit_cols = state.indent_unit.chars().count();
    let mut changes = Vec::new();
    let mut shifts: Vec<(usize, isize)> = Vec::new();
    for &start in &unique_line_starts(state) {
        let line = state.doc.line(state.doc.char_to_line(start));
        let ws_count: usize = line
            .chars()
            .take_while(|c| *c == ' ' || *c == '\t')
            .count();
        if ws_count == 0 {
            continue;
        }
        let leading: String = line.chars().take(ws_count).collect();
        let cur_cols = visual_cols(&leading);
        if cur_cols == 0 {
            continue;
        }
        let target_cols = cur_cols.saturating_sub(unit_cols);
        let new_leading = " ".repeat(target_cols);
        if new_leading == leading {
            continue;
        }
        let delta = new_leading.chars().count() as isize - ws_count as isize;
        changes.push(Change::new(start, start + ws_count, new_leading));
        shifts.push((start, delta));
    }
    if changes.is_empty() {
        None
    } else {
        let selection = rebuild_selection(state, &shifts);
        Some(Transaction { changes, selection: Some(selection) })
    }
}
