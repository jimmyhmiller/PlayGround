//! Cursor motion / selection commands. Not ports of CM6 — those tests live
//! in `view/test/webtest-motion.ts` which is DOM-driven. Hand-written
//! fixtures using the same `|`/`<...>` DSL as `cm_commands.rs`.

use editor_core::commands::{
    add_cursor_above, add_cursor_below, copy_line_down, copy_line_up, cursor_char_backward,
    cursor_char_backward_logical, cursor_char_forward, cursor_char_forward_logical,
    cursor_char_left, cursor_char_right, cursor_doc_end, cursor_doc_start, cursor_group_backward,
    cursor_group_forward, cursor_group_left, cursor_group_right, cursor_line_boundary_backward,
    cursor_line_boundary_forward, cursor_line_boundary_left, cursor_line_boundary_right,
    cursor_line_down, cursor_line_end, cursor_line_start, cursor_line_up, cursor_matching_bracket,
    cursor_subword_left, cursor_subword_right, cursor_word_left, cursor_word_right,
    delete_char_backward, delete_char_backward_strict, delete_char_forward, delete_line,
    delete_line_boundary_backward, delete_line_boundary_forward, delete_to_line_end,
    delete_to_line_start, indent_with_tab, insert_blank_line, insert_tab, select_all,
    select_char_backward, select_char_forward, select_char_left, select_char_right,
    select_doc_end, select_doc_start, select_group_backward, select_group_forward,
    select_group_left, select_group_right, select_line, select_line_boundary_backward,
    select_line_boundary_forward, select_line_boundary_left, select_line_boundary_right,
    select_line_down, select_line_end, select_line_start, select_line_up,
    select_matching_bracket, select_next_occurrence, select_subword_right, select_word_left,
    select_word_right, simplify_selection, split_line, split_selection_by_line, toggle_comment,
    transpose_chars, Command,
};
use editor_core::test_dsl::{parse, render};

fn run(before: &str, cmd: Command, after: &str) {
    let state = parse(before);
    let tr = cmd(&state).unwrap_or_else(|| panic!("no-op for {before:?}"));
    let new_state = state.apply(&tr);
    assert_eq!(render(&new_state), after);
}

fn no_apply(before: &str, cmd: Command) {
    let state = parse(before);
    assert!(cmd(&state).is_none(), "expected no-op for {before:?}");
}

mod select_all_tests {
    use super::*;

    #[test]
    fn selects_whole_doc_from_cursor() {
        run("hel|lo", select_all, "<hello>");
    }

    #[test]
    fn no_op_when_already_selecting_all() {
        no_apply("<hello>", select_all);
    }
}

mod split_line_tests {
    use super::*;

    #[test]
    fn splits_at_cursor() {
        run("foo|bar", split_line, "foo|\nbar");
    }

    #[test]
    fn replaces_selection_with_newline() {
        run("foo<bar>baz", split_line, "foo|\nbaz");
    }

    #[test]
    fn handles_multiple_cursors() {
        run("a|b|c", split_line, "a|\nb|\nc");
    }
}

mod cursor_char_tests {
    use super::*;

    #[test]
    fn left_moves_one_char() {
        run("abc|def", cursor_char_left, "ab|cdef");
    }

    #[test]
    fn right_moves_one_char() {
        run("abc|def", cursor_char_right, "abcd|ef");
    }

    #[test]
    fn left_at_start_is_no_op() {
        no_apply("|abc", cursor_char_left);
    }

    #[test]
    fn right_at_end_is_no_op() {
        no_apply("abc|", cursor_char_right);
    }

    #[test]
    fn left_collapses_selection_to_left_edge() {
        // CM6 actually uses different semantics — collapses to left for ←,
        // here we just call cursor_char_left which moves the head one back
        // from any range. So "<abc>def" head=3 -> head=2.
        run("<abc>def", cursor_char_left, "ab|cdef");
    }
}

mod select_char_tests {
    use super::*;

    #[test]
    fn extends_left() {
        run("abc|def", select_char_left, "ab<c>def");
    }

    #[test]
    fn extends_right() {
        run("abc|def", select_char_right, "abc<d>ef");
    }

    #[test]
    fn extends_existing_selection() {
        run("a<bc>def", select_char_right, "a<bcd>ef");
    }
}

mod cursor_line_tests {
    use super::*;

    #[test]
    fn line_start_moves_to_column_zero() {
        run("foo\nb|ar\nbaz", cursor_line_start, "foo\n|bar\nbaz");
    }

    #[test]
    fn line_end_moves_to_eol() {
        run("foo\nb|ar\nbaz", cursor_line_end, "foo\nbar|\nbaz");
    }

    #[test]
    fn line_end_on_last_line_goes_to_doc_end() {
        run("foo\nbar\nb|az", cursor_line_end, "foo\nbar\nbaz|");
    }

    #[test]
    fn line_start_already_there_is_no_op() {
        no_apply("foo\n|bar", cursor_line_start);
    }
}

mod select_line_tests {
    use super::*;

    #[test]
    fn extends_to_line_start() {
        // Backward selection (anchor=6, head=4); render emits `<from..to>`
        // since the DSL has no syntax for direction.
        run("foo\nba|r", select_line_start, "foo\n<ba>r");
    }

    #[test]
    fn extends_to_line_end() {
        run("foo\nb|ar", select_line_end, "foo\nb<ar>");
    }
}

mod cursor_doc_tests {
    use super::*;

    #[test]
    fn doc_start_jumps_to_zero() {
        run("foo\nb|ar", cursor_doc_start, "|foo\nbar");
    }

    #[test]
    fn doc_end_jumps_to_end() {
        run("foo\nb|ar", cursor_doc_end, "foo\nbar|");
    }

    #[test]
    fn doc_start_already_there_is_no_op() {
        no_apply("|foo", cursor_doc_start);
    }
}

mod select_doc_tests {
    use super::*;

    #[test]
    fn extends_to_doc_start() {
        // Backward; see note in select_line_tests.
        run("foo\nb|ar", select_doc_start, "<foo\nb>ar");
    }

    #[test]
    fn extends_to_doc_end() {
        run("fo|o\nbar", select_doc_end, "fo<o\nbar>");
    }
}

mod delete_char_tests {
    use super::*;

    #[test]
    fn backward_deletes_one_char() {
        run("foo|bar", delete_char_backward, "fo|bar");
    }

    #[test]
    fn forward_deletes_one_char() {
        run("foo|bar", delete_char_forward, "foo|ar");
    }

    #[test]
    fn backward_at_start_is_no_op() {
        no_apply("|abc", delete_char_backward);
    }

    #[test]
    fn forward_at_end_is_no_op() {
        no_apply("abc|", delete_char_forward);
    }

    #[test]
    fn backward_with_selection_deletes_selection() {
        run("foo<bar>baz", delete_char_backward, "foo|baz");
    }

    #[test]
    fn forward_with_selection_deletes_selection() {
        run("foo<bar>baz", delete_char_forward, "foo|baz");
    }
}

mod delete_line_tests {
    use super::*;

    #[test]
    fn deletes_current_line() {
        run("foo\nb|ar\nbaz", delete_line, "foo\n|baz");
    }

    #[test]
    fn deletes_first_line() {
        run("foo|\nbar", delete_line, "|bar");
    }

    #[test]
    fn deletes_last_line_including_preceding_newline() {
        // Cursor on the final line which lacks a trailing newline. Deleting
        // it removes all chars in the line; the previous line keeps its \n.
        let state = parse("foo\nb|ar");
        let tr = delete_line(&state).expect("apply");
        let new_state = state.apply(&tr);
        assert_eq!(new_state.doc.to_string(), "foo\n");
    }

    #[test]
    fn deletes_multiple_lines_in_range() {
        let state = parse("a\n<b\nc>\nd");
        let tr = delete_line(&state).expect("apply");
        assert_eq!(state.apply(&tr).doc.to_string(), "a\nd");
    }
}

mod delete_to_line_tests {
    use super::*;

    #[test]
    fn end_deletes_to_eol() {
        run("foo|bar\nbaz", delete_to_line_end, "foo|\nbaz");
    }

    #[test]
    fn end_at_eol_is_no_op() {
        no_apply("foo|\nbar", delete_to_line_end);
    }

    #[test]
    fn start_deletes_to_bol() {
        run("foo\nba|r", delete_to_line_start, "foo\n|r");
    }

    #[test]
    fn start_at_bol_is_no_op() {
        no_apply("foo\n|bar", delete_to_line_start);
    }
}

mod copy_line_tests {
    use super::*;

    #[test]
    fn copy_line_up_duplicates_above() {
        run("a\nb|\nc", copy_line_up, "a\nb\nb|\nc");
    }

    #[test]
    fn copy_line_down_duplicates_below() {
        run("a\nb|\nc", copy_line_down, "a\nb\nb|\nc");
    }

    #[test]
    fn copy_line_down_on_last_line_no_trailing_newline() {
        let state = parse("a\nb|");
        let tr = copy_line_down(&state).expect("apply");
        let new_state = state.apply(&tr);
        assert_eq!(new_state.doc.to_string(), "a\nb\nb");
    }

    #[test]
    fn copy_line_handles_multiline_block() {
        let state = parse("a\n<b\nc>\nd");
        let tr = copy_line_up(&state).expect("apply");
        let new_state = state.apply(&tr);
        assert_eq!(new_state.doc.to_string(), "a\nb\nc\nb\nc\nd");
    }
}

mod select_line_tests_full {
    use super::*;

    #[test]
    fn extends_single_cursor_to_full_line() {
        // line 0 is "foo\n" (4 chars).
        run("foo|\nbar", select_line, "<foo\n>bar");
    }

    #[test]
    fn extends_to_doc_end_when_no_trailing_newline() {
        run("foo\nb|ar", select_line, "foo\n<bar>");
    }

    #[test]
    fn keeps_existing_full_line_selection() {
        no_apply("<foo\n>bar", select_line);
    }

    #[test]
    fn covers_multiple_lines_when_selection_spans_them() {
        // Input is a single selection that spans into line 2; select_line
        // expands it to cover all touched lines as one range.
        run("a\nb<b\nc>c\nd", select_line, "a\n<bb\ncc\n>d");
    }
}

mod simplify_selection_tests {
    use super::*;

    #[test]
    fn collapses_multiple_cursors_to_primary() {
        // Multi-cursor: a|b|c → primary is index 0 (cursor at 1).
        run("a|b|c", simplify_selection, "a|bc");
    }

    #[test]
    fn no_op_with_single_range() {
        no_apply("a|bc", simplify_selection);
    }
}

mod split_selection_tests {
    use super::*;

    #[test]
    fn splits_multi_line_range_into_one_per_line() {
        // <ab\ncd\nef> spans 3 lines.
        run(
            "<ab\ncd\nef>",
            split_selection_by_line,
            "<ab>\n<cd>\n<ef>",
        );
    }

    #[test]
    fn no_op_on_single_line_selection() {
        no_apply("a<bc>d", split_selection_by_line);
    }
}

mod add_cursor_tests {
    use super::*;

    #[test]
    fn adds_cursor_on_line_above() {
        // Cursor `b|ar` is at column 1 (after `b`); above goes to col 1 of
        // "foo" which is after `f`.
        run("foo\nb|ar", add_cursor_above, "f|oo\nb|ar");
    }

    #[test]
    fn adds_cursor_on_line_below() {
        run("fo|o\nbar", add_cursor_below, "fo|o\nba|r");
    }

    #[test]
    fn add_above_at_top_line_is_no_op() {
        no_apply("f|oo", add_cursor_above);
    }

    #[test]
    fn add_below_at_bottom_line_is_no_op() {
        no_apply("foo\nb|ar", add_cursor_below);
    }

    #[test]
    fn add_below_clamps_to_shorter_line() {
        // Cursor at column 3 of "abcd"; line below is "xy" (only 2 chars).
        // The new cursor clamps to end of "xy".
        run("abc|d\nxy", add_cursor_below, "abc|d\nxy|");
    }
}

mod bracket_matching_tests {
    use super::*;

    #[test]
    fn jumps_from_open_to_close() {
        // CM6 convention: cursor lands AT the matching bracket (not past).
        run("|{abc}", cursor_matching_bracket, "{abc|}");
    }

    #[test]
    fn jumps_from_close_to_open() {
        run("{abc}|", cursor_matching_bracket, "|{abc}");
    }

    #[test]
    fn jumps_from_inside_open_to_close() {
        run("{|abc}", cursor_matching_bracket, "{abc|}");
    }

    #[test]
    fn handles_nested_brackets() {
        run("|{a{b}c}", cursor_matching_bracket, "{a{b}c|}");
    }

    #[test]
    fn no_op_when_not_on_a_bracket() {
        no_apply("ab|c", cursor_matching_bracket);
    }

    #[test]
    fn select_extends_to_matching() {
        run("|{abc}", select_matching_bracket, "<{abc>}");
    }
}

mod insert_blank_line_tests {
    use super::*;

    #[test]
    fn opens_a_blank_line_after_cursor() {
        // Cursor stays at original position; new \n opens a blank line below.
        run("foo|bar", insert_blank_line, "foo|\nbar");
    }

    #[test]
    fn handles_multiple_cursors() {
        run("a|b|c", insert_blank_line, "a|\nb|\nc");
    }
}

mod toggle_comment_tests {
    use super::*;

    #[test]
    fn dispatches_to_line_comment_when_line_token_exists() {
        run("foo|", toggle_comment, "// foo|");
    }
}

mod subword_motion_tests {
    use super::*;

    #[test]
    fn forward_stops_at_camel_case_boundary() {
        run("|fooBar", cursor_subword_right, "foo|Bar");
        run("foo|Bar", cursor_subword_right, "fooBar|");
    }

    #[test]
    fn forward_stops_at_underscore() {
        run("|foo_bar", cursor_subword_right, "foo|_bar");
    }

    #[test]
    fn backward_stops_at_camel_case_boundary() {
        run("fooBar|", cursor_subword_left, "foo|Bar");
        run("foo|Bar", cursor_subword_left, "|fooBar");
    }

    #[test]
    fn backward_stops_at_underscore() {
        run("foo_bar|", cursor_subword_left, "foo_|bar");
    }

    #[test]
    fn forward_handles_allcaps_then_camel() {
        // "XMLHttp" — the boundary is between "XML" and "Http", landing on
        // the L (so the next subword starts at the H).
        run("|XMLHttp", cursor_subword_right, "XML|Http");
    }

    #[test]
    fn select_subword_right_extends() {
        run("|fooBar", select_subword_right, "<foo>Bar");
    }
}

mod cursor_line_vertical_tests {
    use super::*;

    #[test]
    fn down_moves_to_same_column() {
        run("foo\nb|ar\nbaz", cursor_line_down, "foo\nbar\nb|az");
    }

    #[test]
    fn up_moves_to_same_column() {
        run("foo\nbar\nb|az", cursor_line_up, "foo\nb|ar\nbaz");
    }

    #[test]
    fn down_clamps_to_shorter_line() {
        run("abcd|\nxy", cursor_line_down, "abcd\nxy|");
    }

    #[test]
    fn up_at_top_line_goes_to_col_zero() {
        run("ab|c\ndef", cursor_line_up, "|abc\ndef");
    }

    #[test]
    fn down_at_bottom_line_no_op() {
        no_apply("foo\nb|ar", cursor_line_down);
    }

    #[test]
    fn select_line_down_extends() {
        run("foo\nb|ar\nbaz", select_line_down, "foo\nb<ar\nb>az");
    }

    #[test]
    fn select_line_up_extends() {
        // Backward selection.
        run("foo\nbar\nb|az", select_line_up, "foo\nb<ar\nb>az");
    }
}

mod tab_tests {
    use super::*;

    #[test]
    fn insert_tab_inserts_tab_at_cursor() {
        run("foo|bar", insert_tab, "foo\t|bar");
    }

    #[test]
    fn insert_tab_replaces_selection() {
        run("foo<bar>baz", insert_tab, "foo\t|baz");
    }

    #[test]
    fn indent_with_tab_indents_multiline_selection() {
        run(
            "<a\nb\nc>",
            indent_with_tab,
            "  <a\n  b\n  c>",
        );
    }

    #[test]
    fn indent_with_tab_inserts_tab_for_cursor() {
        run("foo|bar", indent_with_tab, "foo\t|bar");
    }
}

mod strict_delete_tests {
    use super::*;

    #[test]
    fn delete_char_backward_strict_aliases_backward() {
        run("foo|bar", delete_char_backward_strict, "fo|bar");
    }
}

mod bidi_alias_tests {
    use super::*;

    #[test]
    fn cursor_char_backward_aliases_left() {
        run("ab|c", cursor_char_backward, "a|bc");
    }

    #[test]
    fn cursor_char_forward_aliases_right() {
        run("a|bc", cursor_char_forward, "ab|c");
    }

    #[test]
    fn logical_variants_match_left_right() {
        run("ab|c", cursor_char_backward_logical, "a|bc");
        run("a|bc", cursor_char_forward_logical, "ab|c");
    }

    #[test]
    fn select_char_backward_extends_left() {
        run("ab|c", select_char_backward, "a<b>c");
    }

    #[test]
    fn select_char_forward_extends_right() {
        run("a|bc", select_char_forward, "a<b>c");
    }

    #[test]
    fn cursor_group_left_right_aliases() {
        run("foo bar|", cursor_group_left, "foo |bar");
        run("|foo bar", cursor_group_right, "foo| bar");
    }

    #[test]
    fn select_group_left_right_aliases() {
        run("foo bar|", select_group_left, "foo <bar>");
        run("|foo bar", select_group_right, "<foo> bar");
    }

    #[test]
    fn cursor_line_boundary_left_right_aliases() {
        run("foo\nb|ar", cursor_line_boundary_left, "foo\n|bar");
        run("foo\nb|ar", cursor_line_boundary_right, "foo\nbar|");
    }

    #[test]
    fn select_line_boundary_left_right_aliases() {
        run("foo\nba|r", select_line_boundary_left, "foo\n<ba>r");
        run("foo\nb|ar", select_line_boundary_right, "foo\nb<ar>");
    }
}

mod alias_tests {
    use super::*;

    #[test]
    fn cursor_group_forward_aliases_word_right() {
        run("|foo bar", cursor_group_forward, "foo| bar");
    }

    #[test]
    fn cursor_group_backward_aliases_word_left() {
        run("foo bar|", cursor_group_backward, "foo |bar");
    }

    #[test]
    fn select_group_forward_aliases_word_select() {
        run("|foo bar", select_group_forward, "<foo> bar");
    }

    #[test]
    fn select_group_backward_aliases_word_select() {
        run("foo bar|", select_group_backward, "foo <bar>");
    }

    #[test]
    fn line_boundary_aliases() {
        run("foo\nb|ar\nbaz", cursor_line_boundary_backward, "foo\n|bar\nbaz");
        run("foo\nb|ar\nbaz", cursor_line_boundary_forward, "foo\nbar|\nbaz");
        run("foo\nba|r", select_line_boundary_backward, "foo\n<ba>r");
        run("foo\nb|ar", select_line_boundary_forward, "foo\nb<ar>");
    }

    #[test]
    fn delete_line_boundary_aliases() {
        run("foo|bar", delete_line_boundary_forward, "foo|");
        run("foo\nba|r", delete_line_boundary_backward, "foo\n|r");
    }
}

mod select_next_occurrence_tests {
    use super::*;

    #[test]
    fn adds_next_occurrence() {
        run("<foo> bar foo", select_next_occurrence, "<foo> bar <foo>");
    }

    #[test]
    fn wraps_around() {
        // No occurrence after primary; should wrap.
        run("foo bar <foo>", select_next_occurrence, "<foo> bar <foo>");
    }

    #[test]
    fn returns_none_when_only_one_occurrence() {
        no_apply("<foo> bar baz", select_next_occurrence);
    }

    #[test]
    fn returns_none_when_primary_is_empty() {
        no_apply("foo|bar", select_next_occurrence);
    }
}

mod transpose_tests {
    use super::*;

    #[test]
    fn swaps_adjacent_chars_and_moves_cursor_right() {
        run("ab|cd", transpose_chars, "acb|d");
    }

    #[test]
    fn at_doc_start_is_no_op() {
        no_apply("|abc", transpose_chars);
    }

    #[test]
    fn at_doc_end_is_no_op() {
        no_apply("abc|", transpose_chars);
    }

    #[test]
    fn at_line_start_is_no_op() {
        no_apply("foo\n|bar", transpose_chars);
    }

    #[test]
    fn at_line_end_is_no_op() {
        no_apply("foo|\nbar", transpose_chars);
    }
}

mod word_motion_tests {
    use super::*;

    #[test]
    fn cursor_word_right_moves_past_next_word() {
        run("|foo bar baz", cursor_word_right, "foo| bar baz");
    }

    #[test]
    fn cursor_word_right_skips_leading_spaces() {
        run("foo|   bar", cursor_word_right, "foo   bar|");
    }

    #[test]
    fn cursor_word_left_moves_past_prev_word() {
        run("foo bar baz|", cursor_word_left, "foo bar |baz");
    }

    #[test]
    fn cursor_word_left_skips_trailing_spaces() {
        run("foo   |bar", cursor_word_left, "|foo   bar");
    }

    #[test]
    fn select_word_right_extends() {
        run("|foo bar", select_word_right, "<foo> bar");
    }

    #[test]
    fn select_word_left_extends() {
        // Backward selection.
        run("foo bar|", select_word_left, "foo <bar>");
    }
}
