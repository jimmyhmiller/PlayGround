//! Port of CodeMirror 6's test DSL (`commands/test/state.ts`):
//!   `|`     → empty cursor
//!   `<...>` → selection range, anchor at start, head at end
//!
//! Positions are *char* indices into the cleaned doc.

use regex::Regex;

use crate::selection::{Range, Selection};
use crate::state::EditorState;

pub fn parse(spec: &str) -> EditorState {
    let re = Regex::new(r"\||<([^>]*)>").unwrap();
    let mut doc = String::new();
    let mut ranges: Vec<Range> = Vec::new();
    let mut last_end = 0; // byte offset in `spec`
    let mut clean_chars: usize = 0; // char count in `doc`

    for m in re.find_iter(spec) {
        // Copy any literal text before this marker.
        let between = &spec[last_end..m.start()];
        doc.push_str(between);
        clean_chars += between.chars().count();

        let matched = m.as_str();
        if matched == "|" {
            ranges.push(Range::cursor(clean_chars));
        } else {
            // <...> — strip the angle brackets, keep the inside.
            let inside = &matched[1..matched.len() - 1];
            let start = clean_chars;
            doc.push_str(inside);
            clean_chars += inside.chars().count();
            ranges.push(Range::new(start, clean_chars));
        }

        last_end = m.end();
    }
    doc.push_str(&spec[last_end..]);

    let selection = if ranges.is_empty() {
        Selection::cursor(0)
    } else {
        Selection::new(ranges, 0)
    };
    EditorState::new(doc, selection)
}

pub fn render(state: &EditorState) -> String {
    let mut text = state.doc.to_string();
    // Insert markers from highest to lowest so earlier indices stay valid.
    let mut sorted: Vec<&Range> = state.selection.ranges.iter().collect();
    sorted.sort_by_key(|r| std::cmp::Reverse(r.from()));
    for r in sorted {
        if r.is_empty() {
            insert_at_char(&mut text, r.from(), "|");
        } else {
            insert_at_char(&mut text, r.to(), ">");
            insert_at_char(&mut text, r.from(), "<");
        }
    }
    text
}

fn insert_at_char(s: &mut String, char_idx: usize, marker: &str) {
    let byte_idx = s
        .char_indices()
        .nth(char_idx)
        .map(|(b, _)| b)
        .unwrap_or(s.len());
    s.insert_str(byte_idx, marker);
}

/// Port of CodeMirror 6 `commands/test/test-comment.ts`'s `s()` helper.
/// Markers `|` mean: 1 marker → cursor, even count → selection range pairs.
/// Required to be 1 or even.
pub fn parse_comment_dsl(spec: &str) -> EditorState {
    let mut markers = Vec::new();
    let mut doc = String::new();
    for c in spec.chars() {
        if c == '|' {
            markers.push(doc.chars().count());
        } else {
            doc.push(c);
        }
    }
    let ranges = if markers.len() == 1 {
        vec![Range::cursor(markers[0])]
    } else {
        assert!(
            markers.len() % 2 == 0,
            "comment DSL needs 1 marker or an even count, got {}",
            markers.len()
        );
        markers
            .chunks(2)
            .map(|w| Range::new(w[0], w[1]))
            .collect()
    };
    let selection = if ranges.is_empty() {
        Selection::cursor(0)
    } else {
        Selection::new(ranges, 0)
    };
    EditorState::new(doc, selection)
}

/// Inverse: render with `|` markers (1 per cursor; pair per selection).
pub fn render_comment_dsl(state: &EditorState) -> String {
    let mut text = state.doc.to_string();
    let mut sorted: Vec<&Range> = state.selection.ranges.iter().collect();
    sorted.sort_by_key(|r| std::cmp::Reverse(r.from()));
    for r in sorted {
        if r.is_empty() {
            insert_at_char(&mut text, r.from(), "|");
        } else {
            insert_at_char(&mut text, r.to(), "|");
            insert_at_char(&mut text, r.from(), "|");
        }
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_cursor() {
        let st = parse("ab|cd");
        assert_eq!(st.doc.to_string(), "abcd");
        assert_eq!(st.selection.ranges, vec![Range::cursor(2)]);
    }

    #[test]
    fn parse_selection() {
        let st = parse("a<bc>d");
        assert_eq!(st.doc.to_string(), "abcd");
        assert_eq!(st.selection.ranges, vec![Range::new(1, 3)]);
    }

    #[test]
    fn round_trip_cursor() {
        let s = "one\n  two|\nthree";
        assert_eq!(render(&parse(s)), s);
    }

    #[test]
    fn round_trip_selection() {
        let s = "one\n<two\nthree>";
        assert_eq!(render(&parse(s)), s);
    }

    #[test]
    fn parse_multiple_cursors() {
        let st = parse("a|b|c");
        assert_eq!(st.doc.to_string(), "abc");
        assert_eq!(st.selection.ranges, vec![Range::cursor(1), Range::cursor(2)]);
    }
}
