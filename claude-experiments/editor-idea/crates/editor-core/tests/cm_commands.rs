//! Ports of CodeMirror 6's `commands/test/test-commands.ts`.
//! Each test mirrors an `it(...)` from upstream, same `(before, after)` shape.

use editor_core::commands::{
    delete_group_backward, delete_group_forward, delete_trailing_whitespace, indent_less,
    indent_more, indent_selection, insert_newline_and_indent, insert_newline_keep_indent,
    move_line_down, move_line_up, Command,
};
use editor_core::state::EditorState;
use editor_core::test_dsl::{parse, render};

fn run(before: &str, cmd: Command, after: &str) {
    let state = parse(before);
    let tr = cmd(&state).unwrap_or_else(|| panic!("command did not apply on: {before:?}"));
    let new_state = state.apply(&tr);
    let got = render(&new_state);
    assert_eq!(
        got, after,
        "\n  before: {before:?}\n  expect: {after:?}\n  got:    {got:?}"
    );
}

/// Like `run`, but checks doc text only (mirrors CM6 tests that assert
/// `cmd(...).doc.toString()`).
fn run_doc(before: &str, cmd: Command, expected_doc: &str) {
    let state = parse(before);
    let tr = cmd(&state).unwrap_or_else(|| panic!("command did not apply on: {before:?}"));
    let new_state = state.apply(&tr);
    let got = new_state.doc.to_string();
    assert_eq!(got, expected_doc);
}

fn no_apply(before: &str, cmd: Command) {
    let state = parse(before);
    assert!(cmd(&state).is_none(), "expected no-op for: {before:?}");
}

mod indent_more_tests {
    use super::*;

    #[test]
    fn adds_indentation() {
        run("one\ntwo|\nthree", indent_more, "one\n  two|\nthree");
    }

    #[test]
    fn indents_all_lines_in_range() {
        run("one\n<two\nthree>", indent_more, "one\n  <two\n  three>");
    }

    #[test]
    fn doesnt_double_indent_a_given_line() {
        run(
            "on|e|\n<two\nth><ree\nfour>",
            indent_more,
            "  on|e|\n  <two\n  th><ree\n  four>",
        );
    }

    #[test]
    fn ignores_trailing_line_when_range_ends_at_line_start() {
        run("on<e\ntwo\n>three", indent_more, "  on<e\n  two\n>three");
    }
}

mod indent_less_tests {
    use super::*;

    #[test]
    fn removes_indentation() {
        run("one\n  two|\nthree", indent_less, "one\ntwo|\nthree");
    }

    #[test]
    fn no_op_when_already_unindented() {
        no_apply("one\ntwo|\nthree", indent_less);
    }

    #[test]
    fn removes_one_unit_of_indentation() {
        run(
            "one\n    two|\n     three|",
            indent_less,
            "one\n  two|\n   three|",
        );
    }

    #[test]
    fn dedents_all_lines_in_range() {
        run("one\n  <two\n  three>", indent_less, "one\n<two\nthree>");
    }

    #[test]
    fn takes_tabs_into_account() {
        run(
            "   \tone|\n  \ttwo|",
            indent_less,
            "  one|\n  two|",
        );
    }

    #[test]
    fn can_split_tabs() {
        run("\tone|", indent_less, "  one|");
    }
}

mod indent_selection_tests {
    use super::*;

    #[test]
    fn auto_indents_the_current_line() {
        // CM6 with `javascriptLanguage` indents this to `"if (0)\n  foo()|"`
        // because the JS grammar knows `if (...)` introduces a block. Our
        // bracket-only rule sees no open bracket on the previous line, so
        // no indent is added — the command is a no-op.
        no_apply("if (0)\nfoo()|", indent_selection);
    }

    #[test]
    fn moves_cursor_ahead_of_indentation() {
        // CM6: "if (0)\n | foo()" → "if (0)\n  |foo()". Without a language
        // our rule won't add indent for `if (0)`; the cursor still ends up
        // after our (zero) indent.
        run("if (0)\n | foo()", indent_selection, "if (0)\n|foo()");
    }

    #[test]
    fn indents_blocks_of_lines() {
        run(
            "if (0) {\n<one\ntwo\nthree>\n}",
            indent_selection,
            "if (0) {\n  <one\n  two\n  three>\n}",
        );
    }

    #[test]
    fn includes_previous_indentation_changes_in_relative_indentation() {
        run(
            "<{\n{\n{\n{}\n}\n}\n}>",
            indent_selection,
            "<{\n  {\n    {\n      {}\n    }\n  }\n}>",
        );
    }
}

mod insert_newline_and_indent_tests {
    use super::*;

    #[test]
    fn indents_the_new_line() {
        run("{|", insert_newline_and_indent, "{\n  |");
    }

    #[test]
    fn handles_multiple_selections() {
        run(
            "{|\n  foo()|",
            insert_newline_and_indent,
            "{\n  |\n  foo()\n  |",
        );
    }

    #[test]
    fn isnt_confused_by_text_after_cursor() {
        run("{|two", insert_newline_and_indent, "{\n  |two");
    }

    #[test]
    fn clears_empty_lines_before_cursor() {
        run("    |", insert_newline_and_indent, "\n|");
    }

    #[test]
    fn deletes_selected_text() {
        run("{<one>two", insert_newline_and_indent, "{\n  |two");
    }

    #[test]
    fn can_explode_brackets() {
        run("let x = [|]", insert_newline_and_indent, "let x = [\n  |\n]");
    }

    #[test]
    fn can_explode_in_indented_positions() {
        run(
            "{\n  foo(|)",
            insert_newline_and_indent,
            "{\n  foo(\n    |\n  )",
        );
    }

    #[test]
    fn can_explode_brackets_with_whitespace() {
        run("foo( | )", insert_newline_and_indent, "foo(\n  |\n)");
    }

    #[test]
    fn doesnt_re_explode_already_exploded_brackets() {
        run("foo(\n  |\n)", insert_newline_and_indent, "foo(\n\n  |\n)");
    }

    #[test]
    fn at_column_zero_does_not_indent_following_content() {
        // Regression: previously the "all-whitespace prefix" branch fired
        // for an empty prefix too, so pressing Enter at column 0 of a `}`
        // line would insert "\n    " and push the `}` to column 4.
        run("foo\n|}", insert_newline_and_indent, "foo\n\n|}");
    }

    #[test]
    fn no_indent_on_plain_line() {
        run("foo|", insert_newline_and_indent, "foo\n|");
    }

    #[test]
    fn preserves_indent_from_prev_line() {
        run("  foo|", insert_newline_and_indent, "  foo\n  |");
    }

    #[test]
    fn preserves_four_space_indent_from_prev_line() {
        run("    foo|", insert_newline_and_indent, "    foo\n    |");
    }

    #[test]
    fn preserves_tab_indent_from_prev_line() {
        let state = parse("\tfoo|").with_indent_unit("\t");
        let tr = insert_newline_and_indent(&state).expect("apply");
        let new_state = state.apply(&tr);
        assert_eq!(render(&new_state), "\tfoo\n\t|");
    }

    #[test]
    fn preserves_tab_plus_spaces_from_prev_line() {
        let state = parse("\t  foo|").with_indent_unit("\t");
        let tr = insert_newline_and_indent(&state).expect("apply");
        let new_state = state.apply(&tr);
        assert_eq!(render(&new_state), "\t  foo\n\t  |");
    }
}

mod insert_newline_keep_indent_tests {
    use super::*;

    #[test]
    fn keeps_indentation() {
        run("  one|", insert_newline_keep_indent, "  one\n  |");
    }

    #[test]
    fn keeps_zero_indentation() {
        run("one|two", insert_newline_keep_indent, "one\n|two");
    }

    #[test]
    fn deletes_the_selection() {
        run(
            "if x:\n  one<two\n  three>four",
            insert_newline_keep_indent,
            "if x:\n  one\n  |four",
        );
    }
}

mod delete_trailing_whitespace_tests {
    use super::*;

    #[test]
    fn deletes_trailing_whitespace() {
        run_doc("foo   ", delete_trailing_whitespace, "foo");
    }

    #[test]
    fn checks_multiple_lines() {
        run_doc(
            "one\ntwo \nthree   \n   ",
            delete_trailing_whitespace,
            "one\ntwo\nthree\n",
        );
    }

    #[test]
    fn handles_empty_lines() {
        run_doc("one  \n\ntwo ", delete_trailing_whitespace, "one\n\ntwo");
    }
}

mod delete_group_forward_tests {
    use super::*;

    #[test]
    fn deletes_a_word() {
        run("one |two three", delete_group_forward, "one | three");
    }

    #[test]
    fn deletes_a_word_with_leading_space() {
        run("one| two three", delete_group_forward, "one| three");
    }

    #[test]
    fn deletes_a_group_of_punctuation() {
        run("one|...two", delete_group_forward, "one|two");
    }

    #[test]
    fn deletes_a_group_of_space() {
        run("one|  \ttwo", delete_group_forward, "one|two");
    }

    #[test]
    fn deletes_a_newline() {
        run("one|\ntwo", delete_group_forward, "one|two");
    }

    #[test]
    fn stops_deleting_at_a_newline() {
        run("one| \n two", delete_group_forward, "one|\n two");
    }

    #[test]
    fn stops_deleting_after_a_newline() {
        run("one|\n two", delete_group_forward, "one| two");
    }

    #[test]
    fn deletes_up_to_end_of_doc() {
        run("one|two", delete_group_forward, "one|");
    }

    #[test]
    fn does_nothing_at_end_of_doc() {
        no_apply("one|", delete_group_forward);
    }
}

mod delete_group_backward_tests {
    use super::*;

    #[test]
    fn deletes_a_word() {
        run("one two| three", delete_group_backward, "one | three");
    }

    #[test]
    fn deletes_a_word_with_trailing_space() {
        run("one two |three", delete_group_backward, "one |three");
    }

    #[test]
    fn deletes_a_group_of_punctuation() {
        run("one...|two", delete_group_backward, "one|two");
    }

    #[test]
    fn deletes_a_group_of_space() {
        run("one \t |two", delete_group_backward, "one|two");
    }

    #[test]
    fn deletes_a_newline() {
        run("one\n|two", delete_group_backward, "one|two");
    }

    #[test]
    fn stops_deleting_at_a_newline() {
        run("one \n |two", delete_group_backward, "one \n|two");
    }

    #[test]
    fn stops_deleting_after_a_newline() {
        run("one \n|two", delete_group_backward, "one |two");
    }

    #[test]
    fn deletes_up_to_start_of_doc() {
        run("one|two", delete_group_backward, "|two");
    }
}

mod move_line_up_tests {
    use super::*;

    #[test]
    fn moves_a_line_up() {
        run("one\ntwo|\nthree", move_line_up, "two|\none\nthree");
    }

    #[test]
    fn preserves_multiple_cursors_on_a_single_line() {
        run("one\nt|w|o|\n", move_line_up, "t|w|o|\none\n");
    }

    #[test]
    fn moves_selected_blocks_as_one() {
        run(
            "one\ntwo\nthr<ee\nfour\nfive>\n",
            move_line_up,
            "one\nthr<ee\nfour\nfive>\ntwo\n",
        );
    }

    #[test]
    fn moves_blocks_made_of_multiple_ranges_as_one() {
        run(
            "one\n<two\nth>ree\nfo|u<r\nfive>\n",
            move_line_up,
            "<two\nth>ree\nfo|u<r\nfive>\none\n",
        );
    }

    #[test]
    fn does_not_include_trailing_line_after_a_range() {
        run(
            "one\n<two\nthree\n>four",
            move_line_up,
            "<two\nthree\n>one\nfour",
        );
    }
}

mod move_line_down_tests {
    use super::*;

    #[test]
    fn moves_a_line_down() {
        run("one\ntwo|\nthree", move_line_down, "one\nthree\ntwo|");
    }

    #[test]
    fn preserves_multiple_cursors_on_a_single_line() {
        run(
            "one\nt|w|o|\nthree",
            move_line_down,
            "one\nthree\nt|w|o|",
        );
    }

    #[test]
    fn moves_selected_blocks_as_one() {
        run(
            "one\ntwo\nthr<ee\nfour\nfive>\nsix",
            move_line_down,
            "one\ntwo\nsix\nthr<ee\nfour\nfive>",
        );
    }

    #[test]
    fn moves_blocks_made_of_multiple_ranges_as_one() {
        run(
            "one\n<two\nth>ree\nfo|u<r\nfive>\nsix\n",
            move_line_down,
            "one\nsix\n<two\nth>ree\nfo|u<r\nfive>\n",
        );
    }

    #[test]
    fn does_not_include_trailing_line_after_a_range() {
        run(
            "one\n<two\nthree\n>four\n",
            move_line_down,
            "one\nfour\n<two\nthree\n>",
        );
    }

    #[test]
    fn clips_selection_when_moving_to_end_of_doc() {
        run(
            "one\n<two\nthree\n>four",
            move_line_down,
            "one\nfour\n<two\nthree>",
        );
    }
}

#[test]
fn dsl_round_trip_baseline() {
    // Sanity: parse then render any spec we plan to use as a fixture.
    for s in [
        "one\ntwo|\nthree",
        "one\n  two|\nthree",
        "one\n<two\nthree>",
        "a|b|c",
    ] {
        let parsed = parse(s);
        // Note: render doesn't preserve <...> direction, so this round-trips
        // because we always emit anchor<head selections in the parser.
        assert_eq!(render(&parsed), s, "round-trip failed for {s:?}");
        let _ = EditorState::clone(&parsed); // touch the type to keep the import live
    }
}
