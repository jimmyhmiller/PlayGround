//! Pure-function tests for the layout math that's caused every
//! rendering bug so far: caret X within a line, char↔byte conversion,
//! line-selection spans, and mouse hit-testing per line. Caret Y is
//! `line * LINE_HEIGHT` — our grid — and needs no tests.

use bevy::math::{IVec2, Vec2};
use bevy::prelude::*;
use bevy::text::{GlyphAtlasInfo, GlyphAtlasLocation, PositionedGlyph};
use editor_bevy::{
    caret_x_in_line, char_from_line_byte, char_to_line_byte, line_selection_span,
    mouse_byte_in_line,
};
use editor_core::selection::Selection;
use editor_core::state::EditorState;
use ropey::Rope;

/// Build a PositionedGlyph fixture for a single line. `x` is the glyph
/// *center*; `width` is the full horizontal advance. All glyphs belong
/// to `line_index = 0` since each line has its own Text2d now.
fn glyph(byte_index: usize, byte_length: usize, x: f32, width: f32) -> PositionedGlyph {
    PositionedGlyph {
        position: Vec2::new(x, 10.0),
        size: Vec2::new(width, 20.0),
        atlas_info: GlyphAtlasInfo {
            texture: AssetId::default(),
            texture_atlas: AssetId::default(),
            location: GlyphAtlasLocation {
                glyph_index: 0,
                offset: IVec2::ZERO,
            },
        },
        span_index: 0,
        line_index: 0,
        byte_index,
        byte_length,
    }
}

/// 10 identical 10-pixel monospace glyphs on one line, ASCII.
fn mono_line() -> Vec<PositionedGlyph> {
    (0..10)
        .map(|i| glyph(i, 1, i as f32 * 10.0 + 5.0, 10.0))
        .collect()
}

// ---------- caret_x_in_line ----------

#[test]
fn caret_at_start_is_left_edge_of_first_glyph() {
    let gs = mono_line();
    assert_eq!(caret_x_in_line(&gs, 1.0, 0), 0.0);
}

#[test]
fn caret_past_last_glyph_is_right_edge_of_last_glyph() {
    let gs = mono_line();
    assert_eq!(caret_x_in_line(&gs, 1.0, 10), 100.0);
}

#[test]
fn caret_on_empty_line_is_zero() {
    assert_eq!(caret_x_in_line(&[], 1.0, 0), 0.0);
}

#[test]
fn caret_between_glyphs_is_next_glyphs_left_edge() {
    let gs = mono_line();
    // byte 5 should land at left edge of glyph[5] (starts at x=50).
    assert_eq!(caret_x_in_line(&gs, 1.0, 5), 50.0);
}

#[test]
fn caret_respects_scale_factor() {
    // 2x display: positions are physical pixels; divide to get logical.
    let gs = vec![glyph(0, 1, 20.0, 20.0)];
    assert_eq!(caret_x_in_line(&gs, 2.0, 0), 5.0);
}

#[test]
fn caret_with_multi_byte_glyph_matches_byte_index() {
    // Emoji (4 bytes), then 'a' (1 byte). Caret at byte 4 = left edge of 'a'.
    let gs = vec![
        glyph(0, 4, 10.0, 20.0),
        glyph(4, 1, 25.0, 10.0),
    ];
    assert_eq!(caret_x_in_line(&gs, 1.0, 4), 20.0);
}

// ---------- char_to_line_byte ----------

#[test]
fn char_to_line_byte_ascii_single_line() {
    let rope = Rope::from_str("hello");
    assert_eq!(char_to_line_byte(&rope, 3), (0, 3));
}

#[test]
fn char_to_line_byte_multi_line() {
    let rope = Rope::from_str("ab\ncd\nef");
    assert_eq!(char_to_line_byte(&rope, 0), (0, 0));
    assert_eq!(char_to_line_byte(&rope, 3), (1, 0));
    assert_eq!(char_to_line_byte(&rope, 4), (1, 1));
    assert_eq!(char_to_line_byte(&rope, 6), (2, 0));
}

#[test]
fn char_to_line_byte_multi_byte_utf8() {
    let rope = Rope::from_str("café");
    assert_eq!(char_to_line_byte(&rope, 0), (0, 0));
    assert_eq!(char_to_line_byte(&rope, 3), (0, 3));
    assert_eq!(char_to_line_byte(&rope, 4), (0, 5));
}

#[test]
fn char_to_line_byte_at_end_of_line_is_before_newline() {
    let rope = Rope::from_str("ab\ncd");
    assert_eq!(char_to_line_byte(&rope, 2), (0, 2));
}

// ---------- line_selection_span ----------

#[test]
fn selection_span_covers_inner_range() {
    let gs = mono_line();
    // Bytes 2..5 → glyphs 2,3,4 → x range [20, 50].
    assert_eq!(line_selection_span(&gs, 1.0, 2, 5), Some((20.0, 50.0)));
}

#[test]
fn selection_span_full_line() {
    let gs = mono_line();
    assert_eq!(line_selection_span(&gs, 1.0, 0, usize::MAX), Some((0.0, 100.0)));
}

#[test]
fn selection_span_outside_line_returns_none() {
    let gs = mono_line();
    // Range starts past the last glyph.
    assert_eq!(line_selection_span(&gs, 1.0, 100, 200), None);
}

#[test]
fn selection_span_empty_glyphs_returns_none() {
    assert_eq!(line_selection_span(&[], 1.0, 0, 10), None);
}

// ---------- mouse_byte_in_line ----------

#[test]
fn mouse_click_before_first_glyph_is_byte_zero() {
    let gs = mono_line();
    assert_eq!(mouse_byte_in_line(-5.0, &gs, 1.0), 0);
}

#[test]
fn mouse_click_past_last_glyph_is_line_end_byte() {
    let gs = mono_line();
    // Past all midpoints → end of last glyph's byte range.
    assert_eq!(mouse_byte_in_line(1000.0, &gs, 1.0), 10);
}

#[test]
fn mouse_hit_test_picks_glyph_whose_midpoint_is_right_of_pointer() {
    let gs = mono_line();
    // x=12 < midpoint of glyph 1 (15); caret to left of that glyph = byte 1.
    assert_eq!(mouse_byte_in_line(12.0, &gs, 1.0), 1);
}

#[test]
fn mouse_hit_test_on_empty_line_is_byte_zero() {
    assert_eq!(mouse_byte_in_line(50.0, &[], 1.0), 0);
}

// ---------- char_from_line_byte ----------

#[test]
fn char_from_line_byte_clamps_past_end_of_mid_doc_line() {
    // "ab\ncd" — clicking past end of line 0 should clamp to char 2
    // (end of line 0, before the newline), not char 3.
    let rope = Rope::from_str("ab\ncd");
    let state = EditorState::new(rope, Selection::cursor(0));
    assert_eq!(char_from_line_byte(&state, 0, 1000), 2);
}

#[test]
fn char_from_line_byte_on_last_line_allows_doc_end() {
    let rope = Rope::from_str("ab\ncd");
    let state = EditorState::new(rope, Selection::cursor(0));
    assert_eq!(char_from_line_byte(&state, 1, 2), 5);
}

#[test]
fn char_from_line_byte_clamps_line_past_last() {
    let rope = Rope::from_str("ab\ncd");
    let state = EditorState::new(rope, Selection::cursor(0));
    // Passing line=99 clamps to the last line.
    assert_eq!(char_from_line_byte(&state, 99, 0), 3);
}
