//! Pure-function tests for the layout math: the editor is strict
//! monospace, so positioning is column × cell_width arithmetic — no
//! per-glyph geometry. Caret Y is `line * LINE_HEIGHT`, which is our
//! grid, so it needs no tests.

use editor_bevy::{
    caret_x_in_line, char_from_line_col, char_to_line_col, line_selection_span,
    mouse_col_at_x,
};
use editor_core::selection::Selection;
use editor_core::state::EditorState;
use ropey::Rope;

const W: f32 = 10.0;

// ---------- caret_x_in_line ----------

#[test]
fn caret_at_col_zero_is_zero() {
    assert_eq!(caret_x_in_line(0, W), 0.0);
}

#[test]
fn caret_past_last_col_uses_column_count() {
    assert_eq!(caret_x_in_line(10, W), 100.0);
}

#[test]
fn caret_mid_line_multiplies_column_by_cell_width() {
    assert_eq!(caret_x_in_line(5, W), 50.0);
}

// ---------- char_to_line_col ----------

#[test]
fn char_to_line_col_ascii_single_line() {
    let rope = Rope::from_str("hello");
    assert_eq!(char_to_line_col(&rope, 3), (0, 3));
}

#[test]
fn char_to_line_col_multi_line() {
    let rope = Rope::from_str("ab\ncd\nef");
    assert_eq!(char_to_line_col(&rope, 0), (0, 0));
    assert_eq!(char_to_line_col(&rope, 3), (1, 0));
    assert_eq!(char_to_line_col(&rope, 4), (1, 1));
    assert_eq!(char_to_line_col(&rope, 6), (2, 0));
}

#[test]
fn char_to_line_col_multi_byte_utf8_counts_chars_not_bytes() {
    // "café" — 'é' is 2 bytes but 1 char. col is char-based, so past
    // 'é' is col 4, not 5.
    let rope = Rope::from_str("café");
    assert_eq!(char_to_line_col(&rope, 0), (0, 0));
    assert_eq!(char_to_line_col(&rope, 3), (0, 3));
    assert_eq!(char_to_line_col(&rope, 4), (0, 4));
}

#[test]
fn char_to_line_col_at_end_of_line_is_before_newline() {
    let rope = Rope::from_str("ab\ncd");
    assert_eq!(char_to_line_col(&rope, 2), (0, 2));
}

// ---------- line_selection_span ----------

#[test]
fn selection_span_covers_inner_cols() {
    assert_eq!(line_selection_span(2, 5, W), (20.0, 50.0));
}

#[test]
fn selection_span_full_line() {
    assert_eq!(line_selection_span(0, 10, W), (0.0, 100.0));
}

#[test]
fn selection_span_empty_is_zero_width() {
    assert_eq!(line_selection_span(3, 3, W), (30.0, 30.0));
}

// ---------- mouse_col_at_x ----------

#[test]
fn mouse_click_before_first_col_is_zero() {
    assert_eq!(mouse_col_at_x(-5.0, W), 0);
}

#[test]
fn mouse_click_past_cells_returns_that_column() {
    // x=1000 with 10px cells → col 100 (caller is responsible for
    // clamping to the actual line length).
    assert_eq!(mouse_col_at_x(1000.0, W), 100);
}

#[test]
fn mouse_hit_test_rounds_to_nearest_cell_boundary() {
    // x=4 is left half of cell 0 → col 0.
    assert_eq!(mouse_col_at_x(4.0, W), 0);
    // x=6 is right half of cell 0 → col 1 (past that cell's midpoint).
    assert_eq!(mouse_col_at_x(6.0, W), 1);
    // x=15 is exactly on cell 1's midpoint — rounds up to col 2.
    assert_eq!(mouse_col_at_x(15.0, W), 2);
}

#[test]
fn mouse_hit_test_with_zero_cell_width_is_zero() {
    // Defensive: if metrics haven't been measured yet, don't panic.
    assert_eq!(mouse_col_at_x(50.0, 0.0), 0);
}

// ---------- char_from_line_col ----------

#[test]
fn char_from_line_col_clamps_past_end_of_mid_doc_line() {
    // "ab\ncd" — clicking past end of line 0 should clamp to char 2
    // (end of line 0, before the newline), not char 3.
    let rope = Rope::from_str("ab\ncd");
    let state = EditorState::new(rope, Selection::cursor(0));
    assert_eq!(char_from_line_col(&state, 0, 1000), 2);
}

#[test]
fn char_from_line_col_on_last_line_allows_doc_end() {
    let rope = Rope::from_str("ab\ncd");
    let state = EditorState::new(rope, Selection::cursor(0));
    assert_eq!(char_from_line_col(&state, 1, 2), 5);
}

#[test]
fn char_from_line_col_clamps_line_past_last() {
    let rope = Rope::from_str("ab\ncd");
    let state = EditorState::new(rope, Selection::cursor(0));
    // Passing line=99 clamps to the last line.
    assert_eq!(char_from_line_col(&state, 99, 0), 3);
}
