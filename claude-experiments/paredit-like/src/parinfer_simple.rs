use anyhow::Result;

#[derive(Debug, Clone, Copy, PartialEq)]
enum DelimType {
    Paren,
    Bracket,
    Brace,
}

impl DelimType {
    fn open_char(&self) -> char {
        match self {
            DelimType::Paren => '(',
            DelimType::Bracket => '[',
            DelimType::Brace => '{',
        }
    }

    fn close_char(&self) -> char {
        match self {
            DelimType::Paren => ')',
            DelimType::Bracket => ']',
            DelimType::Brace => '}',
        }
    }

    fn from_open(ch: char) -> Option<Self> {
        match ch {
            '(' => Some(DelimType::Paren),
            '[' => Some(DelimType::Bracket),
            '{' => Some(DelimType::Brace),
            _ => None,
        }
    }

    fn from_close(ch: char) -> Option<Self> {
        match ch {
            ')' => Some(DelimType::Paren),
            ']' => Some(DelimType::Bracket),
            '}' => Some(DelimType::Brace),
            _ => None,
        }
    }

    fn matches(&self, close_type: &DelimType) -> bool {
        self == close_type
    }
}

#[derive(Debug)]
struct DelimInfo {
    delim_type: DelimType,
    line_idx: usize,
    line_indent: usize,
    col_pos: usize, // Column position where the opening delimiter appears
}

pub struct Parinfer {
    source: String,
}

impl Parinfer {
    pub fn new(source: &str) -> Self {
        Self {
            source: source.to_string(),
        }
    }

    /// True when every closer matches the most recent opener (so `([)]` is
    /// rejected, not merely count-balanced), skipping string literals (with `\`
    /// escapes) and `;` line comments. Well-formed input has no missing/extra
    /// parens to repair, so `balance` preserves it verbatim.
    fn is_well_formed(source: &str) -> bool {
        let mut stack: Vec<char> = Vec::new();
        let mut in_string = false;
        let mut escape_next = false;
        let mut in_comment = false;
        for ch in source.chars() {
            if escape_next {
                escape_next = false;
                continue;
            }
            // `\` escapes the next char: a string escape inside a string, a
            // character literal (`\(`, `\"`, `\;`, …) in code. The following char
            // is never structural.
            if ch == '\\' && !in_comment {
                escape_next = true;
                continue;
            }
            if ch == '"' && !in_comment {
                in_string = !in_string;
                continue;
            }
            if in_string {
                continue;
            }
            if ch == ';' {
                in_comment = true;
            }
            if ch == '\n' || ch == '\r' {
                in_comment = false;
            }
            if in_comment {
                continue;
            }
            match ch {
                '(' | '[' | '{' => stack.push(ch),
                ')' => {
                    if stack.pop() != Some('(') {
                        return false;
                    }
                }
                ']' => {
                    if stack.pop() != Some('[') {
                        return false;
                    }
                }
                '}' => {
                    if stack.pop() != Some('{') {
                        return false;
                    }
                }
                _ => {}
            }
        }
        !in_string && stack.is_empty()
    }

    pub fn balance(&self) -> Result<String> {
        // If the input is already well-formed there are no missing/extra parens to
        // fix, so preserve it verbatim. The indentation-driven balancing below
        // reflows structure from leading whitespace, which corrupts structurally
        // valid code whose indentation deliberately doesn't track nesting depth.
        // This is common in Coil: tail-`if` staircases, `let` bindings aligned with
        // the body at the same column, and multi-line string literals (e.g.
        // `llvm-ir` IR blocks) whose lines are dedented below the enclosing form —
        // the string's leading whitespace was being misread as a dedent that closed
        // the enclosing paren. Only genuinely broken paren structure falls through.
        if Self::is_well_formed(&self.source) {
            return Ok(self.source.clone());
        }

        let lines: Vec<&str> = self.source.lines().collect();
        let mut result_lines = Vec::new();
        let mut delim_stack: Vec<DelimInfo> = Vec::new();
        let mut in_string = false;
        let mut in_comment = false;
        // Set when the final line ends on a dangling `\` (an incomplete character
        // literal). Like an unclosed string, that is an unterminated token we can't
        // sensibly complete, so it blocks auto-closing at EOF and keeps balancing
        // idempotent (a synthesized closer would just be eaten as the literal's
        // char on the next pass).
        let mut trailing_escape = false;

        for (line_idx, line) in lines.iter().enumerate() {
            let line_indent = line.chars().take_while(|c| c.is_whitespace()).count();
            let mut new_line = String::new();
            let mut escape_next = false;

            // Reset comment state at start of each line (comments don't span lines)
            in_comment = false;

            // Check if line ends with multiple closing delimiters
            let trimmed = line.trim_end();
            let has_closing_tail = trimmed.len() > 0 &&
                trimmed.chars().rev().take_while(|&c| matches!(c, ')' | ']' | '}')).count() > 1;

            // Convert line to vec of chars for easier indexing
            let line_chars: Vec<char> = line.chars().collect();

            // Process characters in the line
            for (char_idx, &ch) in line_chars.iter().enumerate() {
                // Handle escape sequences
                if escape_next {
                    escape_next = false;
                    new_line.push(ch);
                    continue;
                }

                // A backslash escapes the next character: inside a string it is a
                // string escape (`\"`), and in code it introduces a Lisp/Coil
                // character literal (`\(`, `\)`, `\"`, `\;`, `\space`, …). Either
                // way the following char is literal and must not be treated as a
                // structural delimiter, string quote, or comment start.
                if ch == '\\' && !in_comment {
                    escape_next = true;
                    new_line.push(ch);
                    continue;
                }

                // Handle string delimiters
                if ch == '"' && !in_comment {
                    in_string = !in_string;
                    new_line.push(ch);
                    continue;
                }

                // Handle comments
                if ch == ';' && !in_string {
                    in_comment = true;
                }

                // If in string or comment, just copy the character
                if in_string || in_comment {
                    new_line.push(ch);
                    continue;
                }

                // Handle delimiters
                if let Some(delim_type) = DelimType::from_open(ch) {
                    delim_stack.push(DelimInfo {
                        delim_type,
                        line_idx,
                        line_indent,
                        col_pos: char_idx,
                    });
                    new_line.push(ch);
                } else if let Some(close_type) = DelimType::from_close(ch) {
                    // Check if this closing delimiter matches the top of stack
                    if let Some(open_info) = delim_stack.last() {
                        if open_info.delim_type.matches(&close_type) {
                            // Check if remaining chars on this line are all closing delimiters
                            let in_closing_tail = line_chars[char_idx..].iter()
                                .all(|&c| matches!(c, ')' | ']' | '}' | ' ' | '\t'));

                            // Only apply indentation-based filtering if:
                            // - We have multiple closing delimiters at end of line, AND
                            // - This closer is part of that tail
                            let should_filter = has_closing_tail && in_closing_tail;

                            if should_filter {
                                // Check indentation to decide if we should accept this closer
                                let next_indent = Self::find_next_indent(&lines, line_idx);
                                let at_eof = line_idx == lines.len() - 1 ||
                                            lines.iter().skip(line_idx + 1).all(|l| l.trim().is_empty() || l.trim().starts_with(';'));

                                let dedenting = next_indent < line_indent;
                                // Use the greater of line_indent and col_pos as the effective nesting level
                                let opener_indent = open_info.line_indent.max(open_info.col_pos);
                                let should_close = !dedenting || opener_indent >= next_indent;

                                if should_close || at_eof {
                                    delim_stack.pop();
                                    new_line.push(ch);
                                }
                                // Otherwise skip it - delimiter should stay open for next line
                            } else {
                                // Normal closing delimiter (not in a tail), always accept
                                delim_stack.pop();
                                new_line.push(ch);
                            }
                        }
                        // If doesn't match, skip it (mismatched delimiter)
                    }
                    // If stack empty, skip it (extra closing delimiter)
                } else {
                    new_line.push(ch);
                }
            }

            // A `\` at end of line leaves a pending char-literal escape; remember it
            // for the EOF close and don't auto-close past it on this line.
            trailing_escape = escape_next;

            // After processing the line, check if we need to auto-close delimiters
            // based on indentation of the next non-empty line
            if !in_string && !in_comment && !escape_next {
                let next_indent = Self::find_next_indent(&lines, line_idx);

                // Check if next line starts with closing delimiters
                let next_line_has_closers = lines.get(line_idx + 1)
                    .and_then(|l| l.trim_start().chars().next())
                    .map(|c| matches!(c, ')' | ']' | '}'))
                    .unwrap_or(false);

                // Close delimiters that were opened at effective indent > next_indent
                // But only if the next line doesn't already have closing delimiters
                // Add them to the end of the current line (Lisp convention)
                //
                // IMPORTANT: Use line_indent only, NOT col_pos.
                // Using col_pos breaks cases like scf.if with sibling regions:
                //   (scf.if cond      <- col_pos=12 for scf.if's (
                //     (region ...)    <- indent=4
                //     (region ...))   <- indent=4, same as sibling, NOT dedenting from scf.if
                // If we used col_pos, we'd wrongly close scf.if after first region
                // because 4 < 12. But both regions are children of scf.if!
                if !next_line_has_closers {
                    while let Some(open_info) = delim_stack.last() {
                        // Only use line_indent, not col_pos, for determining nesting level
                        let opener_indent = open_info.line_indent;
                        if next_indent < line_indent && opener_indent > next_indent {
                            let delim_type = open_info.delim_type;
                            delim_stack.pop();
                            new_line.push(delim_type.close_char());
                        } else {
                            break;
                        }
                    }
                }
            }

            result_lines.push(new_line);
        }

        // Close any remaining open delimiters at the end of file
        // Add them all to the last line (Lisp convention)
        // Only if we're not inside a string (unclosed strings leave delimiters
        // unclosed) and not on a dangling char-literal escape.
        if !in_string && !in_comment && !trailing_escape {
            if let Some(last_line) = result_lines.last_mut() {
                while let Some(open_info) = delim_stack.pop() {
                    last_line.push(open_info.delim_type.close_char());
                }
            }
        }

        Ok(result_lines.join("\n"))
    }

    fn find_next_indent(lines: &[&str], current_idx: usize) -> usize {
        for line in lines.iter().skip(current_idx + 1) {
            let trimmed = line.trim();
            if !trimmed.is_empty() && !trimmed.starts_with(';') {
                return line.chars().take_while(|c| c.is_whitespace()).count();
            }
        }
        0 // End of file
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_brace() {
        let source = "(foo {\n  bar\n  baz\n})";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        // Should preserve the structure since it's already correct
        assert_eq!(result, "(foo {\n  bar\n  baz\n})");
    }

    #[test]
    fn test_attributes_style() {
        let source = "         (attributes {\n           :sym_name @test\n           :function_type (!function (inputs) (results i32))\n         })";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        // Should preserve the structure
        assert!(result.contains("(attributes {"));
        assert!(result.contains(":sym_name @test"));
        assert!(result.contains("})"));
    }

    #[test]
    fn test_inline_brace() {
        let source = "(foo {bar baz})";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        assert_eq!(result, "(foo {bar baz})");
    }

    #[test]
    fn test_extra_indent() {
        let source = "(foo {\n       bar\n       baz\n     })";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        assert!(result.contains("})"));
    }

    #[test]
    fn test_preserve_leading_whitespace() {
        let source = "         (foo bar)";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        println!("Input:  {:?}", source);
        println!("Output: {:?}", result);
        assert!(result.starts_with("         "));
        assert_eq!(result.trim(), "(foo bar)");
    }

    #[test]
    fn test_nested_braces_with_consistent_indent() {
        let source = "(outer {\n  (inner {\n    content\n  })\n})";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        // Should preserve correct structure
        assert!(result.contains("(inner {"));
        assert!(result.contains("content"));
        assert!(result.contains("})"));
    }

    #[test]
    fn test_mixed_delimiters() {
        let source = "(defn foo [x y]\n  {:a x\n   :b y\n   :c (+ x y)})";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        let open_parens = result.chars().filter(|&c| c == '(').count();
        let close_parens = result.chars().filter(|&c| c == ')').count();
        let open_brackets = result.chars().filter(|&c| c == '[').count();
        let close_brackets = result.chars().filter(|&c| c == ']').count();
        let open_braces = result.chars().filter(|&c| c == '{').count();
        let close_braces = result.chars().filter(|&c| c == '}').count();

        assert_eq!(open_parens, close_parens);
        assert_eq!(open_brackets, close_brackets);
        assert_eq!(open_braces, close_braces);
    }

    #[test]
    fn test_missing_closing_delimiters() {
        let source = "(foo {\n  bar\n  baz";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        // Should auto-close the missing delimiters
        let open_parens = result.chars().filter(|&c| c == '(').count();
        let close_parens = result.chars().filter(|&c| c == ')').count();
        let open_braces = result.chars().filter(|&c| c == '{').count();
        let close_braces = result.chars().filter(|&c| c == '}').count();

        assert_eq!(open_parens, close_parens);
        assert_eq!(open_braces, close_braces);
    }

    #[test]
    fn test_extra_closing_delimiters() {
        let source = "(foo bar}})";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        // Should remove mismatched delimiters
        assert_eq!(result, "(foo bar)");
    }

    #[test]
    fn test_string_with_delimiters() {
        let source = r#"(foo "{bar}" baz)"#;
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        // Should not treat delimiters in strings as structural
        assert_eq!(result, source);
    }

    #[test]
    fn test_comment_with_delimiters() {
        let source = "(foo bar) ; {unclosed";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        // Should not treat delimiters in comments as structural
        assert_eq!(result, source);
    }

    // Clojure-specific patterns
    #[test]
    fn test_clojure_defn() {
        let source = "(defn add [a b]\n  (+ a b))";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        assert_eq!(result, source);
    }

    #[test]
    fn test_clojure_let_binding() {
        let source = "(let [x 1\n      y 2]\n  (+ x y))";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        assert_eq!(result, source);
    }

    #[test]
    fn test_clojure_map_literal() {
        let source = "{:name \"Alice\"\n :age 30\n :city \"NYC\"}";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        assert_eq!(result, source);
    }

    #[test]
    fn test_clojure_nested_maps() {
        let source = "{:person {:name \"Alice\"\n          :age 30}\n :location {:city \"NYC\"}}";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        let open_braces = result.chars().filter(|&c| c == '{').count();
        let close_braces = result.chars().filter(|&c| c == '}').count();
        assert_eq!(open_braces, close_braces);
    }

    #[test]
    fn test_clojure_threading_macro() {
        let source = "(-> data\n    (map inc)\n    (filter even?)\n    (reduce +))";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        let open_parens = result.chars().filter(|&c| c == '(').count();
        let close_parens = result.chars().filter(|&c| c == ')').count();
        assert_eq!(open_parens, close_parens);
    }

    #[test]
    fn test_clojure_cond() {
        let source = "(cond\n  (< x 0) :negative\n  (> x 0) :positive\n  :else :zero)";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        assert_eq!(result, source);
    }

    #[test]
    fn test_clojure_missing_closing_in_let() {
        let source = "(let [x 1\n      y 2\n  (+ x y";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        let open_parens = result.chars().filter(|&c| c == '(').count();
        let close_parens = result.chars().filter(|&c| c == ')').count();
        let open_brackets = result.chars().filter(|&c| c == '[').count();
        let close_brackets = result.chars().filter(|&c| c == ']').count();
        assert_eq!(open_parens, close_parens);
        assert_eq!(open_brackets, close_brackets);
    }

    #[test]
    fn test_clojure_vector_of_maps() {
        let source = "[{:a 1\n  :b 2}\n {:c 3\n  :d 4}]";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        let open_braces = result.chars().filter(|&c| c == '{').count();
        let close_braces = result.chars().filter(|&c| c == '}').count();
        let open_brackets = result.chars().filter(|&c| c == '[').count();
        let close_brackets = result.chars().filter(|&c| c == ']').count();
        assert_eq!(open_braces, close_braces);
        assert_eq!(open_brackets, close_brackets);
    }

    #[test]
    fn test_type_annotation_with_nested_calls() {
        // Test case for bug: type annotations should not be moved inside function calls
        let source = "(: (func.call {:callee @add_two}\n              (arith.constant {:value (: 5 i32)})\n              (arith.constant {:value (: 7 i32)}))\n   i32)";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        // The i32 should remain as a sibling to func.call, not moved inside it
        assert_eq!(result, source);
    }

    #[test]
    fn test_simple_type_annotation() {
        // Simpler test case for nested function calls with type annotations
        let source = "(: (func.call arg1\n              arg2)\n   i32)";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        // The i32 should remain outside func.call
        assert_eq!(result, source);
    }

    #[test]
    fn test_scf_if_sibling_regions() {
        // Bug fix: Two sibling regions inside scf.if should both stay inside scf.if
        // Previously, the second region would become a sibling of scf.if (child of def)
        // because its indent (4) was less than scf.if's col_pos (12).
        let source = r#"(def result (scf.if {:result i32} cond
    (region
      (block []
        (scf.yield true_val)))
    (region
      (block []
        (scf.yield false_val)))))"#;

        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        // The balanced input should be preserved exactly
        assert_eq!(result, source, "scf.if should keep both regions as children");
    }

    #[test]
    fn test_sibling_forms_same_indent_preserved() {
        // Multiple siblings at the same indent should all stay inside parent
        let source = "(parent\n  (child1)\n  (child2)\n  (child3))";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        assert_eq!(result, source);
    }

    // ---- Coil regression cases -------------------------------------------
    // These are valid, already-balanced Coil forms whose indentation does not
    // track nesting depth. Indentation-driven balancing used to corrupt them;
    // well-formed input must now round-trip verbatim.

    #[test]
    fn test_coil_llvm_ir_multiline_string_dedented() {
        // A multi-line string argument whose lines sit at column 0 — dedented
        // below the enclosing `(llvm-ir ...)`. The string's leading whitespace
        // must not be read as a dedent that closes the enclosing paren.
        let source = "(defn uart-byte [(c i64)] (-> i64)\n  (llvm-ir i64 [c]\n\"%p = inttoptr i64 150994944 to ptr\nret i64 0\"))";
        let parinfer = Parinfer::new(source);
        assert_eq!(parinfer.balance().unwrap(), source);
    }

    #[test]
    fn test_coil_let_bindings_aligned_with_body() {
        // `let` binding elements and body forms at the SAME column: the `]`
        // must stay where it is, not migrate past the body.
        let source = "(let [sig a\n     fv b]\n     (use sig)\n     (use fv))";
        let parinfer = Parinfer::new(source);
        assert_eq!(parinfer.balance().unwrap(), source);
    }

    #[test]
    fn test_coil_tail_if_staircase() {
        // Deeply nested tail `if`s drawn at shallow (back-dented) indentation,
        // closed by a long run of parens on the last line.
        let source = "(if a\n    x\n (if b\n     y\n  (if c\n      z\n   w)))";
        let parinfer = Parinfer::new(source);
        assert_eq!(parinfer.balance().unwrap(), source);
    }

    #[test]
    fn test_coil_trailing_newline_preserved() {
        // A well-formed file with a trailing newline must keep it (the old
        // line-join path silently dropped it).
        let source = "(defn f [] (-> i64) 0)\n";
        let parinfer = Parinfer::new(source);
        assert_eq!(parinfer.balance().unwrap(), source);
    }

    #[test]
    fn test_char_literal_delimiters_not_structural() {
        // Lisp/Coil character literals `\(` `\)` `\]` are the *characters*, not
        // structural delimiters. These forms are already balanced and must be
        // preserved verbatim.
        for source in [
            "(str \\()",           // \( is a char, ) closes (str
            "[\\\"]",              // \" is a char, brackets balance
            "(list \\) \\( \\])",  // several delimiter char-literals
        ] {
            let out = Parinfer::new(source).balance().unwrap();
            assert_eq!(out, source, "char literals must be preserved: {source:?}");
        }
    }

    #[test]
    fn test_char_literal_does_not_absorb_real_closer_on_repair() {
        // `(foo \(` is missing its real closer; the `\(` must not consume the
        // synthesized `)`. Expect `(foo \()`.
        assert_eq!(Parinfer::new("(foo \\(").balance().unwrap(), "(foo \\()");
        // `[\"` — missing `]`; the `\"` is a char literal, not a string opener.
        assert_eq!(Parinfer::new("[\\\"").balance().unwrap(), "[\\\"]");
    }

    #[test]
    fn test_dangling_backslash_is_left_alone() {
        // A trailing `\` is an incomplete character literal — like an unclosed
        // string, it can't be sensibly completed, so balancing is a no-op and
        // stays idempotent.
        let source = "[\\ \\";
        let once = Parinfer::new(source).balance().unwrap();
        assert_eq!(once, source, "dangling backslash left as-is");
        assert_eq!(Parinfer::new(&once).balance().unwrap(), once, "idempotent");
    }

    #[test]
    fn test_coil_broken_input_still_repaired() {
        // The fix must not disable repair: genuinely unbalanced input still
        // gets its missing closer synthesized.
        let source = "(iadd 1 2";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        let opens = result.chars().filter(|&c| c == '(').count();
        let closes = result.chars().filter(|&c| c == ')').count();
        assert_eq!(opens, closes, "unbalanced input should be repaired: {result:?}");
    }
}
