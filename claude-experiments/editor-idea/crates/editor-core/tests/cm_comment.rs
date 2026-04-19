//! Ports of CodeMirror 6 `commands/test/test-comment.ts` (line-comment subset).
//! Block-comment + multi-language tests deferred (block needs language config).

use editor_core::comment::{
    block_comment, block_uncomment, line_comment, line_uncomment, toggle_block_comment,
    toggle_block_comment_by_line, toggle_line_comment, CommentTokens,
};
use editor_core::state::EditorState;
use editor_core::test_dsl::{parse_comment_dsl, render_comment_dsl};
use editor_core::transaction::Transaction;

fn s(spec: &str, line_token: &str) -> EditorState {
    parse_comment_dsl(spec).with_comment_tokens(CommentTokens {
        line: Some(line_token.into()),
        block: None,
    })
}

/// Like CM6's `checkToggleChain` — applying `toggle` repeatedly walks the
/// chain of provided docs. The last entry is the steady-state: applying
/// `toggle` to it returns to the second-to-last.
fn check_chain(line_token: &str, docs: &[&str]) {
    let mut state = s(docs[0], line_token);
    for i in 1..=docs.len() {
        // CM6 returns false from the command on no-op — equivalent to
        // dispatch nothing, leaving state unchanged. We mirror by treating
        // `None` as "state stays the same".
        if let Some(tr) = toggle_line_comment(&state) {
            state = state.apply(&tr);
        }
        let expected_idx = if i == docs.len() { docs.len() - 2 } else { i };
        let expected = s(docs[expected_idx], line_token);
        assert_eq!(
            state.doc.to_string(),
            expected.doc.to_string(),
            "step {i}: doc differs"
        );
        assert_eq!(
            render_comment_dsl(&state),
            render_comment_dsl(&expected),
            "step {i}: selection differs"
        );
    }
}

mod line_comments_slash {
    use super::*;
    const K: &str = "//";

    #[test]
    fn toggles_in_an_empty_single_selection() {
        check_chain(
            K,
            &[
                "\nline 1\n  // // // //line| 2\nline 3\n",
                "\nline 1\n  // // //line| 2\nline 3\n",
                "\nline 1\n  // //line| 2\nline 3\n",
                "\nline 1\n  //line| 2\nline 3\n",
                "\nline 1\n  line| 2\nline 3\n",
                "\nline 1\n  // line| 2\nline 3\n",
            ],
        );

        check_chain(
            K,
            &[
                "\nline 1\n  //line 2|\nline 3\n",
                "\nline 1\n  line 2|\nline 3\n",
                "\nline 1\n  // line 2|\nline 3\n",
            ],
        );

        check_chain(
            K,
            &[
                "\nline 1\n|  //line 2\nline 3\n",
                "\nline 1\n|  line 2\nline 3\n",
                "\nline 1\n|  // line 2\nline 3\n",
            ],
        );

        check_chain(
            K,
            &[
                "\nline 1\n|//\nline 3\n",
                "\nline 1\n|\nline 3\n",
                "\nline 1\n// |\nline 3\n",
            ],
        );

        check_chain(
            K,
            &[
                "\nline 1\n line 2\nline 3\n|//",
                "\nline 1\n line 2\nline 3\n|",
                "\nline 1\n line 2\nline 3\n// |",
            ],
        );
    }

    #[test]
    fn toggles_when_cursor_at_beginning_of_line() {
        check_chain(K, &["line 1\n  |line 2\nline 3\n", "line 1\n  // |line 2\nline 3\n"]);
    }

    #[test]
    fn toggles_in_a_single_line_selection() {
        check_chain(
            K,
            &[
                "line 1\n  //li|ne |2\nline 3\n",
                "line 1\n  li|ne |2\nline 3\n",
                "line 1\n  // li|ne |2\nline 3\n",
            ],
        );
    }

    #[test]
    fn toggles_in_a_multi_line_selection() {
        check_chain(
            K,
            &[
                "\n  //lin|e 1\n  //  line 2\n  // line |3\n",
                "\n  lin|e 1\n   line 2\n  line |3\n",
                "\n  // lin|e 1\n  //  line 2\n  // line |3\n",
            ],
        );

        check_chain(
            K,
            &[
                "\n  //lin|e 1\n  //  line 2\n   line 3\n  // li|ne 4\n",
                "\n  // //lin|e 1\n  // //  line 2\n  //  line 3\n  // // li|ne 4\n",
            ],
        );

        check_chain(
            K,
            &[
                "\n  // lin|e 1\n\n  // line |3\n",
                "\n  lin|e 1\n\n  line |3\n",
            ],
        );

        check_chain(
            K,
            &[
                "\n  // lin|e 1\n     \n  // line |3\n",
                "\n  lin|e 1\n     \n  line |3\n",
            ],
        );

        check_chain(
            K,
            &[
                "\n|\n  // line 2\n    | \n",
                "\n|\n  line 2\n    | \n",
            ],
        );

        check_chain(K, &["\n|\n\n    | \n", "\n|\n\n    | \n"]);
    }

    #[test]
    fn toggles_in_multi_line_multi_range_selection() {
        check_chain(
            K,
            &[
                "\n  lin|e 1\n  line |2\n  line 3\n  l|ine 4\n  line| 5\n",
                "\n  // lin|e 1\n  // line |2\n  line 3\n  // l|ine 4\n  // line| 5\n",
            ],
        );
    }

    #[test]
    fn handles_multiple_selections_on_one_line() {
        check_chain(K, &["|line| |with| |ranges|", "// |line| |with| |ranges|"]);
    }

    #[test]
    fn doesnt_include_lines_in_which_a_selection_range_ends() {
        check_chain(
            K,
            &[
                "line| 1\nline 2\n|line 3",
                "// line| 1\n// line 2\n|line 3",
            ],
        );
    }

    #[test]
    fn leaves_empty_lines_alone() {
        check_chain(K, &["line| 1\n\nline 3|", "// line| 1\n\n// line 3|"]);
    }

    #[test]
    fn comments_empty_lines_with_a_cursor() {
        check_chain(K, &["|\nline 2", "// |\nline 2"]);
    }
}

fn s_block(spec: &str, open: &str, close: &str) -> EditorState {
    parse_comment_dsl(spec).with_comment_tokens(CommentTokens {
        line: None,
        block: Some((open.into(), close.into())),
    })
}

fn check_block_chain<F>(open: &str, close: &str, docs: &[&str], cmd: F)
where
    F: Fn(&EditorState) -> Option<Transaction>,
{
    let mut state = s_block(docs[0], open, close);
    for i in 1..=docs.len() {
        if let Some(tr) = cmd(&state) {
            state = state.apply(&tr);
        }
        let expected_idx = if i == docs.len() { docs.len() - 2 } else { i };
        let expected = s_block(docs[expected_idx], open, close);
        assert_eq!(state.doc.to_string(), expected.doc.to_string(), "step {i} doc");
        assert_eq!(
            render_comment_dsl(&state),
            render_comment_dsl(&expected),
            "step {i} sel"
        );
    }
}

mod block_comments_c_style {
    use super::*;
    const O: &str = "/*";
    const C: &str = "*/";

    #[test]
    fn toggles_in_multi_line_selection() {
        check_block_chain(
            O,
            C,
            &[
                "\n  lin|e 1\n  line 2\n  line 3\n  line |4\n  line 5\n",
                "\n  lin/* |e 1\n  line 2\n  line 3\n  line | */4\n  line 5\n",
            ],
            toggle_block_comment,
        );
    }

    #[test]
    fn toggles_in_multi_line_multi_range_selection() {
        check_block_chain(
            O,
            C,
            &[
                "\n  lin|e 1\n  line |2\n  l|ine 3\n  line 4\n  line |5\n",
                "\n  lin/* |e 1\n  line | */2\n  l/* |ine 3\n  line 4\n  line | */5\n",
            ],
            toggle_block_comment,
        );
    }

    #[test]
    fn toggles_inside_the_selection() {
        check_block_chain(
            O,
            C,
            &[
                "|/* one\ntwo */| three",
                "|one\ntwo| three",
                "/* |one\ntwo| */ three",
            ],
            toggle_block_comment,
        );
    }

    #[test]
    fn comments_the_entire_line() {
        check_block_chain(
            O,
            C,
            &["one|\ntwo", "/* one| */\ntwo"],
            toggle_block_comment_by_line,
        );
    }

    #[test]
    fn comments_multiple_lines() {
        check_block_chain(
            O,
            C,
            &["on|e\nt|wo", "/* on|e\nt|wo */"],
            toggle_block_comment_by_line,
        );
    }

    #[test]
    fn joins_selected_blocks_of_lines() {
        check_block_chain(
            O,
            C,
            &[
                "on|e\nt|w|o\nth|ree",
                "/* on|e\nt|w|o\nth|ree */",
            ],
            toggle_block_comment_by_line,
        );
    }

    #[test]
    fn doesnt_include_lines_that_selection_stops_at_start_of() {
        check_block_chain(
            O,
            C,
            &["|one\n|two", "/* |one */\n|two"],
            toggle_block_comment_by_line,
        );
    }

    #[test]
    fn includes_lines_with_cursor_selection_at_start() {
        check_block_chain(
            O,
            C,
            &["|one\ntwo", "|/* one */\ntwo"],
            toggle_block_comment_by_line,
        );
    }
}

mod one_way_line_comment_tests {
    use super::*;

    #[test]
    fn line_comment_adds_token() {
        let state = s("foo|", "//");
        let tr = line_comment(&state).expect("apply");
        let new_state = state.apply(&tr);
        assert_eq!(new_state.doc.to_string(), "// foo");
    }

    #[test]
    fn line_comment_skips_already_commented_lines() {
        // Cursor on line 0 only. Line 0 is already commented → no candidate
        // remains for `line_comment`, so it returns None (no-op).
        let state = s("// foo|\nbar", "//");
        assert!(line_comment(&state).is_none());
    }

    #[test]
    fn line_comment_adds_to_uncommented_lines_in_mixed_range() {
        // Selection covers two lines; one is already commented, the other
        // isn't. `line_comment` should comment only the uncommented line.
        let state = s("// fo|o\nba|r", "//");
        let tr = line_comment(&state).expect("apply");
        let new_state = state.apply(&tr);
        assert_eq!(new_state.doc.to_string(), "// foo\n// bar");
    }

    #[test]
    fn line_uncomment_removes_token() {
        let state = s("// foo|", "//");
        let tr = line_uncomment(&state).expect("apply");
        let new_state = state.apply(&tr);
        assert_eq!(new_state.doc.to_string(), "foo");
    }

    #[test]
    fn line_uncomment_no_op_on_uncommented() {
        let state = s("foo|", "//");
        assert!(line_uncomment(&state).is_none());
    }
}

mod one_way_block_comment_tests {
    use super::*;

    #[test]
    fn block_comment_adds_wrap() {
        let state = s_block("|foo|", "/*", "*/");
        let tr = block_comment(&state).expect("apply");
        let new_state = state.apply(&tr);
        assert_eq!(new_state.doc.to_string(), "/* foo */");
    }

    #[test]
    fn block_comment_no_op_when_already_wrapped() {
        // CommentOnly mode: even if already wrapped, it adds another wrap.
        // Verify it actually adds (doesn't toggle off).
        let state = s_block("|/* foo */|", "/*", "*/");
        let tr = block_comment(&state).expect("apply");
        let new_state = state.apply(&tr);
        assert_eq!(new_state.doc.to_string(), "/* /* foo */ */");
    }

    #[test]
    fn block_uncomment_removes_wrap() {
        let state = s_block("|/* foo */|", "/*", "*/");
        let tr = block_uncomment(&state).expect("apply");
        let new_state = state.apply(&tr);
        assert_eq!(new_state.doc.to_string(), "foo");
    }

    #[test]
    fn block_uncomment_no_op_when_unwrapped() {
        let state = s_block("|foo|", "/*", "*/");
        assert!(block_uncomment(&state).is_none());
    }
}

mod block_comments_html_style {
    use super::*;
    const O: &str = "<!--";
    const C: &str = "-->";

    #[test]
    fn toggles_in_multi_line_selection() {
        check_block_chain(
            O,
            C,
            &[
                "\n  lin|e 1\n  line 2\n  line 3\n  line |4\n  line 5\n",
                "\n  lin<!-- |e 1\n  line 2\n  line 3\n  line | -->4\n  line 5\n",
            ],
            toggle_block_comment,
        );
    }

    #[test]
    fn comments_the_entire_line() {
        check_block_chain(
            O,
            C,
            &["one|\ntwo", "<!-- one| -->\ntwo"],
            toggle_block_comment_by_line,
        );
    }
}

mod line_comments_hash {
    use super::*;
    const K: &str = "#";

    #[test]
    fn toggles_when_cursor_at_beginning_of_line() {
        check_chain(K, &["line 1\n  |line 2\nline 3\n", "line 1\n  # |line 2\nline 3\n"]);
    }

    #[test]
    fn handles_multiple_selections_on_one_line() {
        check_chain(K, &["|line| |with| |ranges|", "# |line| |with| |ranges|"]);
    }
}
