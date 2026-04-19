//! Comment-toggle commands. Mirrors CM6's `toggleLineComment`,
//! `toggleBlockComment`, `toggleBlockCommentByLine`.

use std::collections::BTreeSet;

use ropey::Rope;

use crate::selection::{Range, Selection};
use crate::state::EditorState;
use crate::transaction::{Change, Transaction};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommentTokens {
    pub line: Option<String>,
    pub block: Option<(String, String)>,
}

impl Default for CommentTokens {
    fn default() -> Self {
        Self {
            line: Some("//".into()),
            block: Some(("/*".into(), "*/".into())),
        }
    }
}

fn line_end_char(doc: &Rope, line_idx: usize) -> usize {
    if line_idx + 1 < doc.len_lines() {
        doc.line_to_char(line_idx + 1)
    } else {
        doc.len_chars()
    }
}

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

struct LineInfo {
    text_start: usize,
    commented_at: Option<usize>,
    commented_len: usize,
    is_empty: bool,
}

/// Find the line-start char index for `text_start`, given that we know
/// `text_start` is on one of the lines in `line_set`.
fn line_start_for(doc: &Rope, _line_set: &BTreeSet<usize>, text_start: usize) -> usize {
    doc.line_to_char(doc.char_to_line(text_start))
}

fn classify_line(doc: &Rope, line_idx: usize, token: &str) -> LineInfo {
    let line_start = doc.line_to_char(line_idx);
    let line_end = line_end_char(doc, line_idx);
    let line_str: String = doc.slice(line_start..line_end).to_string();
    let line_no_nl = line_str.strip_suffix('\n').unwrap_or(&line_str);

    let ws_count: usize = line_no_nl
        .chars()
        .take_while(|c| *c == ' ' || *c == '\t')
        .count();
    // CM6 treats a line with no non-whitespace chars as "empty" for the
    // purpose of comment-toggle: it gets skipped when other lines have
    // content, but participates if the whole range is whitespace-only.
    let is_empty = line_no_nl.chars().all(|c| c == ' ' || c == '\t');
    let text_start = line_start + ws_count;
    let after_ws: String = line_no_nl.chars().skip(ws_count).collect();
    let token_chars = token.chars().count();

    let commented = if after_ws.starts_with(token) {
        let after_token: String = after_ws.chars().skip(token_chars).collect();
        let space_len = if after_token.starts_with(' ') { 1 } else { 0 };
        Some(token_chars + space_len)
    } else {
        None
    };

    LineInfo {
        text_start,
        commented_at: commented.map(|_| text_start),
        commented_len: commented.unwrap_or(0),
        is_empty,
    }
}

pub fn toggle_block_comment(state: &EditorState) -> Option<Transaction> {
    let (open, close) = state.comment_tokens.block.as_ref()?.clone();
    block_comment_changes(state, &open, &close, false)
}

pub fn toggle_block_comment_by_line(state: &EditorState) -> Option<Transaction> {
    let (open, close) = state.comment_tokens.block.as_ref()?.clone();
    block_comment_changes(state, &open, &close, true)
}

/// A wrap detected for a single range. Each range either has the open/close
/// tokens *inside* its bounds (the user selected the comments along with the
/// content) or *outside* (the user selected text inside an existing comment).
struct DetectedWrap {
    open_from: usize,
    open_to: usize,
    close_from: usize,
    close_to: usize,
}

fn slice_eq(doc: &Rope, from: usize, to: usize, expected: &str) -> bool {
    if to > doc.len_chars() {
        return false;
    }
    let s: String = doc.slice(from..to).to_string();
    s == expected
}

fn detect_wrap(
    doc: &Rope,
    from: usize,
    to: usize,
    open: &str,
    close: &str,
) -> Option<DetectedWrap> {
    let open_chars = open.chars().count();
    let close_chars = close.chars().count();
    let with_space_open = format!("{} ", open);
    let with_space_close = format!(" {}", close);

    // Try INSIDE first: `<open>[<space>]...[<space>]<close>` within from..to.
    if to >= from + open_chars + close_chars {
        let starts_open_space =
            slice_eq(doc, from, from + open_chars + 1, &with_space_open);
        let starts_open = !starts_open_space && slice_eq(doc, from, from + open_chars, open);
        if starts_open_space || starts_open {
            let open_to = from + if starts_open_space { open_chars + 1 } else { open_chars };
            let ends_space_close = to >= close_chars + 1
                && slice_eq(doc, to - close_chars - 1, to, &with_space_close);
            let ends_close = !ends_space_close
                && slice_eq(doc, to - close_chars, to, close);
            if ends_space_close || ends_close {
                let close_from = to - if ends_space_close { close_chars + 1 } else { close_chars };
                if open_to <= close_from {
                    return Some(DetectedWrap {
                        open_from: from,
                        open_to,
                        close_from,
                        close_to: to,
                    });
                }
            }
        }
    }

    // Try OUTSIDE: `<open>[<space>]` immediately before from, and
    // `[<space>]<close>` immediately after to.
    let outside_open = if from >= open_chars + 1
        && slice_eq(doc, from - open_chars - 1, from, &with_space_open)
    {
        Some((from - open_chars - 1, from))
    } else if from >= open_chars && slice_eq(doc, from - open_chars, from, open) {
        Some((from - open_chars, from))
    } else {
        None
    };
    let outside_close = if to + close_chars + 1 <= doc.len_chars()
        && slice_eq(doc, to, to + close_chars + 1, &with_space_close)
    {
        Some((to, to + close_chars + 1))
    } else if to + close_chars <= doc.len_chars()
        && slice_eq(doc, to, to + close_chars, close)
    {
        Some((to, to + close_chars))
    } else {
        None
    };
    if let (Some(o), Some(c)) = (outside_open, outside_close) {
        return Some(DetectedWrap {
            open_from: o.0,
            open_to: o.1,
            close_from: c.0,
            close_to: c.1,
        });
    }

    None
}

fn block_comment_changes(
    state: &EditorState,
    open: &str,
    close: &str,
    by_line: bool,
) -> Option<Transaction> {
    block_comment_changes_with_mode(state, open, close, by_line, BlockCommentMode::Toggle)
}

fn block_comment_changes_with_mode(
    state: &EditorState,
    open: &str,
    close: &str,
    by_line: bool,
    mode: BlockCommentMode,
) -> Option<Transaction> {
    let doc = &state.doc;

    // Compute per-range (from, to) — either the original range, or the
    // line-extended range for `by_line` mode (with adjacent ranges merged).
    let targets: Vec<(usize, usize)> = if by_line {
        // Union all line blocks across all ranges, merge adjacent.
        let mut blocks: Vec<(usize, usize)> = state
            .selection
            .ranges
            .iter()
            .map(|r| range_lines(state, *r))
            .collect();
        blocks.sort_by_key(|b| b.0);
        let mut merged: Vec<(usize, usize)> = Vec::new();
        for b in blocks {
            match merged.last_mut() {
                Some(m) if b.0 <= m.1 + 1 => m.1 = m.1.max(b.1),
                _ => merged.push(b),
            }
        }
        merged
            .into_iter()
            .map(|(a, b)| {
                let from = doc.line_to_char(a);
                let to = if b + 1 < doc.len_lines() {
                    let next_start = doc.line_to_char(b + 1);
                    // Don't include the trailing newline in the wrapped span.
                    if next_start > from && doc.char(next_start - 1) == '\n' {
                        next_start - 1
                    } else {
                        next_start
                    }
                } else {
                    doc.len_chars()
                };
                (from, to)
            })
            .collect()
    } else {
        state
            .selection
            .ranges
            .iter()
            .map(|r| (r.from(), r.to()))
            .collect()
    };

    // Build changes per target plus a tagged action for selection mapping.
    enum Action {
        Wrap { from: usize, to: usize, open_len: usize, close_len: usize },
        Unwrap {
            open_from: usize,
            open_to: usize,
            close_from: usize,
            close_to: usize,
        },
    }

    let mut changes: Vec<Change> = Vec::new();
    let mut actions: Vec<Action> = Vec::new();
    for &(from, to) in &targets {
        if from == to && !by_line {
            continue;
        }
        let detected = detect_wrap(doc, from, to, open, close);
        let do_unwrap = match (mode, &detected) {
            (BlockCommentMode::Toggle, Some(_)) => true,
            (BlockCommentMode::Toggle, None) => false,
            (BlockCommentMode::CommentOnly, _) => false,
            (BlockCommentMode::UncommentOnly, Some(_)) => true,
            (BlockCommentMode::UncommentOnly, None) => continue,
        };
        if do_unwrap {
            let w = detected.unwrap();
            changes.push(Change::delete(w.open_from, w.open_to));
            changes.push(Change::delete(w.close_from, w.close_to));
            actions.push(Action::Unwrap {
                open_from: w.open_from,
                open_to: w.open_to,
                close_from: w.close_from,
                close_to: w.close_to,
            });
        } else {
            let open_insert = format!("{} ", open);
            let close_insert = format!(" {}", close);
            let open_len = open_insert.chars().count();
            let close_len = close_insert.chars().count();
            changes.push(Change::insert(from, open_insert));
            changes.push(Change::insert(to, close_insert));
            actions.push(Action::Wrap { from, to, open_len, close_len });
        }
    }

    if changes.is_empty() {
        return None;
    }

    fn shift_for(p: usize, is_empty_cursor: bool, a: &Action) -> isize {
        match a {
            Action::Wrap { from, to, open_len, close_len } => {
                if p < *from {
                    0
                } else if p == *from {
                    // Empty cursor at the wrap start sticks *before* the open
                    // insert (CM6: `|/* ... */`). A non-empty range whose left
                    // edge is at the wrap start expands to include the open
                    // insert (the selection contains the wrapped content).
                    if is_empty_cursor { 0 } else { *open_len as isize }
                } else if p <= *to {
                    *open_len as isize
                } else {
                    (*open_len + *close_len) as isize
                }
            }
            Action::Unwrap {
                open_from,
                open_to,
                close_from,
                close_to,
            } => {
                let open_len = (open_to - open_from) as isize;
                let close_len = (close_to - close_from) as isize;
                if p <= *open_from {
                    0
                } else if p < *open_to {
                    -((p - open_from) as isize)
                } else if p <= *close_from {
                    -open_len
                } else if p < *close_to {
                    -open_len - ((p - close_from) as isize)
                } else {
                    -open_len - close_len
                }
            }
        }
    }

    fn map_pos_through(p: usize, is_empty_cursor: bool, actions: &[Action]) -> usize {
        let mut shift: isize = 0;
        for a in actions {
            shift += shift_for(p, is_empty_cursor, a);
        }
        (p as isize + shift).max(0) as usize
    }

    let new_ranges: Vec<Range> = state
        .selection
        .ranges
        .iter()
        .map(|r| {
            let empty = r.is_empty();
            let new_anchor = map_pos_through(r.anchor, empty, &actions);
            let new_head = map_pos_through(r.head, empty, &actions);
            Range::new(new_anchor, new_head)
        })
        .collect();

    Some(Transaction {
        changes,
        selection: Some(Selection::new(new_ranges, state.selection.primary)),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LineCommentMode {
    Toggle,
    CommentOnly,
    UncommentOnly,
}

fn line_comment_inner(state: &EditorState, mode: LineCommentMode) -> Option<Transaction> {
    let token = state.comment_tokens.line.as_deref()?;
    let doc = &state.doc;

    let mut line_set: BTreeSet<usize> = BTreeSet::new();
    for r in &state.selection.ranges {
        let (a, b) = range_lines(state, *r);
        for l in a..=b {
            line_set.insert(l);
        }
    }

    let line_indices: Vec<usize> = line_set.iter().copied().collect();
    let infos: Vec<LineInfo> = line_indices
        .iter()
        .map(|&l| classify_line(doc, l, token))
        .collect();

    let cursor_lines: BTreeSet<usize> = state
        .selection
        .ranges
        .iter()
        .filter(|r| r.is_empty())
        .map(|r| doc.char_to_line(r.from()))
        .collect();

    let non_empty: Vec<usize> = infos
        .iter()
        .enumerate()
        .filter(|(_, i)| !i.is_empty)
        .map(|(j, _)| j)
        .collect();
    let candidate_idxs: Vec<usize> = if !non_empty.is_empty() {
        non_empty
    } else {
        infos
            .iter()
            .enumerate()
            .filter(|(j, _)| cursor_lines.contains(&line_indices[*j]))
            .map(|(j, _)| j)
            .collect()
    };

    if candidate_idxs.is_empty() {
        return None;
    }

    let all_commented = candidate_idxs
        .iter()
        .all(|&j| infos[j].commented_at.is_some());

    let do_uncomment = match mode {
        LineCommentMode::Toggle => all_commented,
        LineCommentMode::CommentOnly => false,
        LineCommentMode::UncommentOnly => true,
    };

    let changes: Vec<Change> = if do_uncomment {
        candidate_idxs
            .iter()
            .filter(|&&j| infos[j].commented_at.is_some())
            .map(|&j| {
                let info = &infos[j];
                let from = info.commented_at.unwrap();
                Change::delete(from, from + info.commented_len)
            })
            .collect()
    } else {
        // Commenting: insert at the shallowest indent column across candidate
        // lines so the tokens align in a column.
        let candidate_starts: Vec<usize> = candidate_idxs
            .iter()
            .filter(|&&j| {
                if mode == LineCommentMode::CommentOnly {
                    // Don't double-comment a line that's already commented.
                    infos[j].commented_at.is_none()
                } else {
                    true
                }
            })
            .map(|&j| {
                let info = &infos[j];
                info.text_start - line_start_for(doc, &line_set, info.text_start)
            })
            .collect();
        if candidate_starts.is_empty() {
            return None;
        }
        let min_col = *candidate_starts.iter().min().unwrap();
        candidate_idxs
            .iter()
            .filter(|&&j| {
                if mode == LineCommentMode::CommentOnly {
                    infos[j].commented_at.is_none()
                } else {
                    true
                }
            })
            .map(|&j| {
                let info = &infos[j];
                let line_start = line_start_for(doc, &line_set, info.text_start);
                let insert_pos = line_start + min_col;
                let insert = format!("{} ", token);
                Change::insert(insert_pos, insert)
            })
            .collect()
    };

    if changes.is_empty() {
        return None;
    }

    Some(Transaction { changes, selection: None })
}

pub fn line_comment(state: &EditorState) -> Option<Transaction> {
    line_comment_inner(state, LineCommentMode::CommentOnly)
}

pub fn line_uncomment(state: &EditorState) -> Option<Transaction> {
    line_comment_inner(state, LineCommentMode::UncommentOnly)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockCommentMode {
    Toggle,
    CommentOnly,
    UncommentOnly,
}

fn block_comment_inner(
    state: &EditorState,
    mode: BlockCommentMode,
    by_line: bool,
) -> Option<Transaction> {
    let (open, close) = state.comment_tokens.block.as_ref()?.clone();
    block_comment_changes_with_mode(state, &open, &close, by_line, mode)
}

pub fn block_comment(state: &EditorState) -> Option<Transaction> {
    block_comment_inner(state, BlockCommentMode::CommentOnly, false)
}

pub fn block_uncomment(state: &EditorState) -> Option<Transaction> {
    block_comment_inner(state, BlockCommentMode::UncommentOnly, false)
}

pub fn toggle_line_comment(state: &EditorState) -> Option<Transaction> {
    line_comment_inner(state, LineCommentMode::Toggle)
}
