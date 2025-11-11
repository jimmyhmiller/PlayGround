use paredit_like::parinfer_simple::Parinfer;
use proptest::prelude::*;

// ============================================================================
// Generators
// ============================================================================

/// Generate a simple atom (symbol, number, keyword, etc.)
fn atom() -> impl Strategy<Value = String> {
    prop_oneof![
        "[a-z][a-z0-9-]*",           // symbols
        "[0-9]+",                     // numbers
        ":[a-z][a-z0-9-]*",          // keywords
        "true|false|nil",             // literals
        "@[a-z]+",                    // at-symbols (MLIR style)
        "%[a-z0-9]+",                 // percent-symbols (MLIR style)
        "![a-z][a-z0-9]*",           // bang-types (MLIR style)
        "\\^bb[0-9]+"                 // block labels (MLIR style)
    ]
}

/// Generate a string, possibly multiline
fn string_literal() -> impl Strategy<Value = String> {
    prop_oneof![
        // Single-line strings
        "[a-zA-Z0-9 ]+".prop_map(|s| format!("\"{}\"", s)),
        // Multiline strings
        "[a-zA-Z0-9 ]+\n[a-zA-Z0-9 ]+".prop_map(|s| format!("\"{}\"", s)),
        // Strings with escaped quotes
        "[a-z]+".prop_map(|s| format!("\"{}\\\"{}\"", s, s)),
        // Strings with delimiters inside
        Just("\"has (paren) inside\"".to_string()),
        Just("\"has [bracket] inside\"".to_string()),
        Just("\"has {brace} inside\"".to_string()),
    ]
}

/// Generate a line comment
fn comment() -> impl Strategy<Value = String> {
    "[a-zA-Z0-9 ()\\[\\]{}]+".prop_map(|s| format!("; {}", s))
}

/// Generate properly nested s-expressions
fn well_formed_sexpr(depth: u32) -> impl Strategy<Value = String> {
    let leaf = prop_oneof![
        atom(),
        string_literal(),
    ];

    leaf.prop_recursive(
        depth,  // depth
        256,    // max size
        10,     // items per collection
        move |inner| {
            prop_oneof![
                // Lists
                prop::collection::vec(inner.clone(), 0..5)
                    .prop_map(|items| format!("({})", items.join(" "))),
                // Vectors
                prop::collection::vec(inner.clone(), 0..5)
                    .prop_map(|items| format!("[{}]", items.join(" "))),
                // Maps/braces
                prop::collection::vec(inner.clone(), 0..5)
                    .prop_map(|items| format!("{{{}}}", items.join(" "))),
            ]
        },
    )
}

/// Generate s-expressions with proper indentation
fn indented_sexpr(depth: u32) -> impl Strategy<Value = String> {
    (well_formed_sexpr(depth), prop::sample::select(vec![0, 2, 4]))
        .prop_map(|(s, indent)| {
            let lines: Vec<&str> = s.lines().collect();
            lines
                .iter()
                .enumerate()
                .map(|(i, line)| {
                    if i == 0 {
                        line.to_string()
                    } else {
                        format!("{}{}", " ".repeat(indent), line)
                    }
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
}

/// Generate broken s-expressions (missing closers)
fn broken_sexpr_missing_closers() -> impl Strategy<Value = String> {
    well_formed_sexpr(3).prop_map(|s| {
        // Remove some closing delimiters
        let mut chars: Vec<char> = s.chars().collect();
        let mut to_remove = vec![];

        for (i, ch) in chars.iter().enumerate().rev() {
            if matches!(ch, ')' | ']' | '}') && to_remove.len() < 3 {
                to_remove.push(i);
            }
        }

        for idx in to_remove {
            chars.remove(idx);
        }

        chars.into_iter().collect()
    })
}

/// Generate broken s-expressions (extra closers)
fn broken_sexpr_extra_closers() -> impl Strategy<Value = String> {
    well_formed_sexpr(3).prop_map(|s| {
        // Add some extra closing delimiters
        let extras = vec![')', ']', '}'];
        format!("{}{}", s, extras.iter().cycle().take(3).collect::<String>())
    })
}

/// Generate broken s-expressions (mismatched closers)
fn broken_sexpr_mismatched() -> impl Strategy<Value = String> {
    "(a [b {c".prop_map(|s| s.to_string())
        .prop_flat_map(|s| {
            // Add wrong closing delimiters
            prop::sample::select(vec![
                format!("{})}}}}",s),
                format!("{}])}}}}",s),
                format!("{}}})]",s),
            ])
        })
}

// ============================================================================
// Property Tests
// ============================================================================

// Helper function to count delimiters outside strings and comments
fn count_structural_delimiters(s: &str) -> (usize, usize, usize, usize, usize, usize) {
    let mut in_string = false;
    let mut in_comment = false;
    let mut escape_next = false;

    let mut open_parens = 0;
    let mut close_parens = 0;
    let mut open_brackets = 0;
    let mut close_brackets = 0;
    let mut open_braces = 0;
    let mut close_braces = 0;

    for ch in s.chars() {
        if escape_next {
            escape_next = false;
            continue;
        }

        if ch == '\\' && in_string {
            escape_next = true;
            continue;
        }

        if ch == '\n' {
            in_comment = false;
            continue;
        }

        if ch == '"' && !in_comment {
            in_string = !in_string;
            continue;
        }

        if ch == ';' && !in_string {
            in_comment = true;
            continue;
        }

        if in_string || in_comment {
            continue;
        }

        // Count structural delimiters
        match ch {
            '(' => open_parens += 1,
            ')' => close_parens += 1,
            '[' => open_brackets += 1,
            ']' => close_brackets += 1,
            '{' => open_braces += 1,
            '}' => close_braces += 1,
            _ => {}
        }
    }

    (open_parens, close_parens, open_brackets, close_brackets, open_braces, close_braces)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// Property: Output always has balanced structural delimiters
    /// (except when there are unclosed strings/comments which prevent balancing)
    #[test]
    fn prop_output_is_balanced(input in ".*") {
        let parinfer = Parinfer::new(&input);
        if let Ok(output) = parinfer.balance() {
            // Check if output has unclosed strings or ends in comment
            let mut in_comment = false;
            let mut escape_next = false;
            let mut quote_count = 0;
            let mut ends_in_comment = false;

            for ch in output.chars() {
                if escape_next {
                    escape_next = false;
                    continue;
                }
                if ch == '\\' {
                    escape_next = true;
                    continue;
                }
                if ch == '\n' {
                    in_comment = false;
                }
                if ch == ';' && !in_comment {
                    in_comment = true;
                }
                if ch == '"' && !in_comment {
                    quote_count += 1;
                }
                ends_in_comment = in_comment;
            }

            // If there's an unclosed string or ends in a comment, skip balance check
            if quote_count % 2 != 0 || ends_in_comment {
                return Ok(());
            }

            let (open_parens, close_parens, open_brackets, close_brackets, open_braces, close_braces)
                = count_structural_delimiters(&output);

            prop_assert_eq!(open_parens, close_parens, "Parens must balance");
            prop_assert_eq!(open_brackets, close_brackets, "Brackets must balance");
            prop_assert_eq!(open_braces, close_braces, "Braces must balance");
        }
    }

    /// Property: Balancing is idempotent
    #[test]
    fn prop_idempotent(input in ".*") {
        let parinfer = Parinfer::new(&input);
        if let Ok(output1) = parinfer.balance() {
            let parinfer2 = Parinfer::new(&output1);
            if let Ok(output2) = parinfer2.balance() {
                prop_assert_eq!(output1, output2, "balance(balance(x)) should equal balance(x)");
            }
        }
    }

    /// Property: Well-formed input is preserved
    #[test]
    fn prop_well_formed_preserved(sexpr in well_formed_sexpr(4)) {
        let parinfer = Parinfer::new(&sexpr);
        if let Ok(output) = parinfer.balance() {
            // Remove whitespace differences for comparison
            let input_normalized = sexpr.chars().filter(|c| !c.is_whitespace()).collect::<String>();
            let output_normalized = output.chars().filter(|c| !c.is_whitespace()).collect::<String>();

            prop_assert_eq!(input_normalized, output_normalized,
                "Well-formed input should be preserved");
        }
    }

    /// Property: Strings are never modified
    #[test]
    fn prop_strings_preserved(
        prefix in well_formed_sexpr(2),
        string in string_literal(),
        suffix in well_formed_sexpr(2)
    ) {
        let input = format!("({} {} {})", prefix, string, suffix);
        let parinfer = Parinfer::new(&input);

        if let Ok(output) = parinfer.balance() {
            prop_assert!(output.contains(&string),
                "String literal must be preserved exactly: {}", string);
        }
    }

    /// Property: Comments are preserved
    #[test]
    fn prop_comments_preserved(
        sexpr in well_formed_sexpr(2),
        comment_text in comment()
    ) {
        let input = format!("{} {}", sexpr, comment_text);
        let parinfer = Parinfer::new(&input);

        if let Ok(output) = parinfer.balance() {
            prop_assert!(output.contains(&comment_text),
                "Comment must be preserved: {}", comment_text);
        }
    }

    /// Property: Leading indentation is preserved
    #[test]
    fn prop_indentation_preserved(input in indented_sexpr(3)) {
        let parinfer = Parinfer::new(&input);

        if let Ok(output) = parinfer.balance() {
            let input_lines: Vec<&str> = input.lines().collect();
            let output_lines: Vec<&str> = output.lines().collect();

            for (in_line, out_line) in input_lines.iter().zip(output_lines.iter()) {
                let in_indent = in_line.chars().take_while(|c| c.is_whitespace()).count();
                let out_indent = out_line.chars().take_while(|c| c.is_whitespace()).count();

                prop_assert_eq!(in_indent, out_indent,
                    "Indentation must be preserved on line: {}", in_line);
            }
        }
    }

    /// Property: Opening delimiters are never removed
    #[test]
    fn prop_openers_never_removed(input in well_formed_sexpr(4)) {
        let parinfer = Parinfer::new(&input);

        if let Ok(output) = parinfer.balance() {
            let input_opens = input.chars().filter(|&c| c == '(' || c == '[' || c == '{').count();
            let output_opens = output.chars().filter(|&c| c == '(' || c == '[' || c == '{').count();

            prop_assert!(output_opens >= input_opens,
                "Opening delimiters should never be removed");
        }
    }

    /// Property: Broken input gets fixed
    #[test]
    fn prop_broken_gets_balanced(input in broken_sexpr_missing_closers()) {
        let parinfer = Parinfer::new(&input);

        if let Ok(output) = parinfer.balance() {
            let (open_parens, close_parens, _, _, _, _) = count_structural_delimiters(&output);

            prop_assert_eq!(open_parens, close_parens,
                "Broken input should be balanced after processing");
        }
    }

    /// Property: Extra closers are removed
    #[test]
    fn prop_extra_closers_removed(input in broken_sexpr_extra_closers()) {
        let parinfer = Parinfer::new(&input);

        if let Ok(output) = parinfer.balance() {
            let (open_parens, close_parens, open_brackets, close_brackets, open_braces, close_braces)
                = count_structural_delimiters(&output);

            prop_assert_eq!(open_parens, close_parens, "Parens must balance");
            prop_assert_eq!(open_brackets, close_brackets, "Brackets must balance");
            prop_assert_eq!(open_braces, close_braces, "Braces must balance");
        }
    }

    /// Property: Multiline strings work correctly
    #[test]
    fn prop_multiline_strings(
        prefix in atom(),
        string_content in "[a-zA-Z0-9 ]+\n[a-zA-Z0-9 ]+"
    ) {
        let input = format!("({} \"{}\")", prefix, string_content);
        let parinfer = Parinfer::new(&input);

        if let Ok(output) = parinfer.balance() {
            // Check the string content is preserved
            prop_assert!(output.contains(&string_content),
                "Multiline string content must be preserved");

            // Check delimiters balance
            let (open_parens, close_parens, _, _, _, _) = count_structural_delimiters(&output);
            prop_assert_eq!(open_parens, close_parens);
        }
    }

    /// Property: Never panics on any input
    #[test]
    fn prop_never_panics(input in ".*") {
        let parinfer = Parinfer::new(&input);
        // Just ensure it doesn't panic
        let _ = parinfer.balance();
    }

    /// Property: Atoms are never modified
    #[test]
    fn prop_atoms_preserved(
        atom1 in atom(),
        atom2 in atom(),
        atom3 in atom()
    ) {
        let input = format!("({} {} {})", atom1, atom2, atom3);
        let parinfer = Parinfer::new(&input);

        if let Ok(output) = parinfer.balance() {
            prop_assert!(output.contains(&atom1), "Atom must be preserved: {}", atom1);
            prop_assert!(output.contains(&atom2), "Atom must be preserved: {}", atom2);
            prop_assert!(output.contains(&atom3), "Atom must be preserved: {}", atom3);
        }
    }
}

// ============================================================================
// Regression Tests (specific cases found by property testing)
// ============================================================================

#[cfg(test)]
mod regression_tests {
    use super::*;

    #[test]
    fn test_multiline_string_with_parens() {
        let input = "(foo \"has (paren\ninside)\" bar)";
        let parinfer = Parinfer::new(input);
        let output = parinfer.balance().unwrap();

        // Should have exactly 1 opening and 1 closing paren (not counting those in the string)
        let open_parens = output.chars().filter(|&c| c == '(').count();
        let close_parens = output.chars().filter(|&c| c == ')').count();
        assert_eq!(open_parens, close_parens);

        // String content should be preserved
        assert!(output.contains("has (paren\ninside)"));
    }

    #[test]
    fn test_deeply_nested() {
        let input = "((((((((((foo))))))))))";
        let parinfer = Parinfer::new(input);
        let output = parinfer.balance().unwrap();

        assert_eq!(input, output, "Well-formed deeply nested should be preserved");
    }

    #[test]
    fn test_empty_structures() {
        let input = "() [] {}";
        let parinfer = Parinfer::new(input);
        let output = parinfer.balance().unwrap();

        assert_eq!(input, output, "Empty structures should be preserved");
    }

    #[test]
    fn test_mixed_delimiters_nested() {
        let input = "({[(){}]})";
        let parinfer = Parinfer::new(input);
        let output = parinfer.balance().unwrap();

        assert_eq!(input, output, "Mixed nested delimiters should be preserved");
    }
}
