//! Indentation-driven balancer for Lisp-family source code.
//!
//! Given a string of (possibly malformed) Lisp/Clojure source, [`balance`]
//! returns a structurally-balanced version: missing closing delimiters are
//! inserted based on indentation, and stray/mismatched closing delimiters
//! are dropped. Strings and `;` line comments are respected.
//!
//! ```
//! use lisp_balance::balance;
//!
//! let fixed = balance("(defn foo\n  (+ 1 2");
//! assert_eq!(fixed, "(defn foo\n  (+ 1 2))");
//! ```

#[derive(Debug, Clone, Copy, PartialEq)]
enum DelimType {
    Paren,
    Bracket,
    Brace,
}

impl DelimType {
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
    line_indent: usize,
    col_pos: usize,
}

/// Balance brackets in a piece of Lisp source.
///
/// Returns a string in which `()`, `[]`, and `{}` are structurally balanced.
/// The algorithm:
///
/// - Closes any delimiters left open at the end of the file.
/// - Closes a delimiter when the next non-blank, non-comment line dedents
///   below the column where the delimiter was opened.
/// - Drops stray or mismatched closing delimiters.
/// - Ignores delimiters inside `"..."` strings (with `\` escape) and inside
///   `;` line comments.
///
/// Already-balanced input is returned unchanged.
pub fn balance(source: &str) -> String {
    let lines: Vec<&str> = source.lines().collect();
    let mut result_lines = Vec::new();
    let mut delim_stack: Vec<DelimInfo> = Vec::new();
    let mut in_string = false;
    let mut in_comment;

    for (line_idx, line) in lines.iter().enumerate() {
        let line_indent = line.chars().take_while(|c| c.is_whitespace()).count();
        let mut new_line = String::new();
        let mut escape_next = false;

        // Comments don't span lines.
        in_comment = false;

        let trimmed = line.trim_end();
        let has_closing_tail = !trimmed.is_empty()
            && trimmed
                .chars()
                .rev()
                .take_while(|&c| matches!(c, ')' | ']' | '}'))
                .count()
                > 1;

        let line_chars: Vec<char> = line.chars().collect();

        for (char_idx, &ch) in line_chars.iter().enumerate() {
            if escape_next {
                escape_next = false;
                new_line.push(ch);
                continue;
            }

            if ch == '\\' && in_string {
                escape_next = true;
                new_line.push(ch);
                continue;
            }

            if ch == '"' && !in_comment {
                in_string = !in_string;
                new_line.push(ch);
                continue;
            }

            if ch == ';' && !in_string {
                in_comment = true;
            }

            if in_string || in_comment {
                new_line.push(ch);
                continue;
            }

            if let Some(delim_type) = DelimType::from_open(ch) {
                delim_stack.push(DelimInfo {
                    delim_type,
                    line_indent,
                    col_pos: char_idx,
                });
                new_line.push(ch);
            } else if let Some(close_type) = DelimType::from_close(ch) {
                if let Some(open_info) = delim_stack.last() {
                    if open_info.delim_type.matches(&close_type) {
                        let in_closing_tail = line_chars[char_idx..]
                            .iter()
                            .all(|&c| matches!(c, ')' | ']' | '}' | ' ' | '\t'));

                        let should_filter = has_closing_tail && in_closing_tail;

                        if should_filter {
                            let next_indent = find_next_indent(&lines, line_idx);
                            let at_eof = line_idx == lines.len() - 1
                                || lines.iter().skip(line_idx + 1).all(|l| {
                                    l.trim().is_empty() || l.trim().starts_with(';')
                                });

                            let dedenting = next_indent < line_indent;
                            let opener_indent = open_info.line_indent.max(open_info.col_pos);
                            let should_close = !dedenting || opener_indent >= next_indent;

                            if should_close || at_eof {
                                delim_stack.pop();
                                new_line.push(ch);
                            }
                            // Otherwise drop this closer; the delimiter stays
                            // open for a subsequent line.
                        } else {
                            delim_stack.pop();
                            new_line.push(ch);
                        }
                    }
                    // Mismatched closer: drop it.
                }
                // No opener on stack: drop this stray closer.
            } else {
                new_line.push(ch);
            }
        }

        // After the line, auto-close delimiters whose indent is greater than
        // the next non-blank line's indent.
        //
        // Use line_indent (not col_pos) when comparing the opener's nesting
        // level here. Using col_pos would wrongly close forms whose siblings
        // line up under their first child rather than under the form head —
        // e.g. an `scf.if` with two `(region ...)` siblings.
        if !in_string && !in_comment {
            let next_indent = find_next_indent(&lines, line_idx);

            let next_line_has_closers = lines
                .get(line_idx + 1)
                .and_then(|l| l.trim_start().chars().next())
                .map(|c| matches!(c, ')' | ']' | '}'))
                .unwrap_or(false);

            if !next_line_has_closers {
                while let Some(open_info) = delim_stack.last() {
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

    // Close anything still open at EOF, appending to the last line. An open
    // string suppresses this (we don't want to inject closers inside a
    // string literal that the user simply hasn't terminated yet).
    if !in_string {
        if let Some(last_line) = result_lines.last_mut() {
            while let Some(open_info) = delim_stack.pop() {
                last_line.push(open_info.delim_type.close_char());
            }
        }
    }

    result_lines.join("\n")
}

fn find_next_indent(lines: &[&str], current_idx: usize) -> usize {
    for line in lines.iter().skip(current_idx + 1) {
        let trimmed = line.trim();
        if !trimmed.is_empty() && !trimmed.starts_with(';') {
            return line.chars().take_while(|c| c.is_whitespace()).count();
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_brace() {
        let source = "(foo {\n  bar\n  baz\n})";
        assert_eq!(balance(source), "(foo {\n  bar\n  baz\n})");
    }

    #[test]
    fn test_attributes_style() {
        let source = "         (attributes {\n           :sym_name @test\n           :function_type (!function (inputs) (results i32))\n         })";
        let result = balance(source);
        assert!(result.contains("(attributes {"));
        assert!(result.contains(":sym_name @test"));
        assert!(result.contains("})"));
    }

    #[test]
    fn test_inline_brace() {
        assert_eq!(balance("(foo {bar baz})"), "(foo {bar baz})");
    }

    #[test]
    fn test_extra_indent() {
        let source = "(foo {\n       bar\n       baz\n     })";
        assert!(balance(source).contains("})"));
    }

    #[test]
    fn test_preserve_leading_whitespace() {
        let source = "         (foo bar)";
        let result = balance(source);
        assert!(result.starts_with("         "));
        assert_eq!(result.trim(), "(foo bar)");
    }

    #[test]
    fn test_nested_braces_with_consistent_indent() {
        let source = "(outer {\n  (inner {\n    content\n  })\n})";
        let result = balance(source);
        assert!(result.contains("(inner {"));
        assert!(result.contains("content"));
        assert!(result.contains("})"));
    }

    #[test]
    fn test_mixed_delimiters() {
        let source = "(defn foo [x y]\n  {:a x\n   :b y\n   :c (+ x y)})";
        let result = balance(source);
        assert_eq!(
            result.chars().filter(|&c| c == '(').count(),
            result.chars().filter(|&c| c == ')').count()
        );
        assert_eq!(
            result.chars().filter(|&c| c == '[').count(),
            result.chars().filter(|&c| c == ']').count()
        );
        assert_eq!(
            result.chars().filter(|&c| c == '{').count(),
            result.chars().filter(|&c| c == '}').count()
        );
    }

    #[test]
    fn test_missing_closing_delimiters() {
        let source = "(foo {\n  bar\n  baz";
        let result = balance(source);
        assert_eq!(
            result.chars().filter(|&c| c == '(').count(),
            result.chars().filter(|&c| c == ')').count()
        );
        assert_eq!(
            result.chars().filter(|&c| c == '{').count(),
            result.chars().filter(|&c| c == '}').count()
        );
    }

    #[test]
    fn test_extra_closing_delimiters() {
        assert_eq!(balance("(foo bar}})"), "(foo bar)");
    }

    #[test]
    fn test_string_with_delimiters() {
        let source = r#"(foo "{bar}" baz)"#;
        assert_eq!(balance(source), source);
    }

    #[test]
    fn test_comment_with_delimiters() {
        let source = "(foo bar) ; {unclosed";
        assert_eq!(balance(source), source);
    }

    #[test]
    fn test_clojure_defn() {
        let source = "(defn add [a b]\n  (+ a b))";
        assert_eq!(balance(source), source);
    }

    #[test]
    fn test_clojure_let_binding() {
        let source = "(let [x 1\n      y 2]\n  (+ x y))";
        assert_eq!(balance(source), source);
    }

    #[test]
    fn test_clojure_map_literal() {
        let source = "{:name \"Alice\"\n :age 30\n :city \"NYC\"}";
        assert_eq!(balance(source), source);
    }

    #[test]
    fn test_clojure_nested_maps() {
        let source = "{:person {:name \"Alice\"\n          :age 30}\n :location {:city \"NYC\"}}";
        let result = balance(source);
        assert_eq!(
            result.chars().filter(|&c| c == '{').count(),
            result.chars().filter(|&c| c == '}').count()
        );
    }

    #[test]
    fn test_clojure_threading_macro() {
        let source = "(-> data\n    (map inc)\n    (filter even?)\n    (reduce +))";
        let result = balance(source);
        assert_eq!(
            result.chars().filter(|&c| c == '(').count(),
            result.chars().filter(|&c| c == ')').count()
        );
    }

    #[test]
    fn test_clojure_cond() {
        let source = "(cond\n  (< x 0) :negative\n  (> x 0) :positive\n  :else :zero)";
        assert_eq!(balance(source), source);
    }

    #[test]
    fn test_clojure_missing_closing_in_let() {
        let source = "(let [x 1\n      y 2\n  (+ x y";
        let result = balance(source);
        assert_eq!(
            result.chars().filter(|&c| c == '(').count(),
            result.chars().filter(|&c| c == ')').count()
        );
        assert_eq!(
            result.chars().filter(|&c| c == '[').count(),
            result.chars().filter(|&c| c == ']').count()
        );
    }

    #[test]
    fn test_clojure_vector_of_maps() {
        let source = "[{:a 1\n  :b 2}\n {:c 3\n  :d 4}]";
        let result = balance(source);
        assert_eq!(
            result.chars().filter(|&c| c == '{').count(),
            result.chars().filter(|&c| c == '}').count()
        );
        assert_eq!(
            result.chars().filter(|&c| c == '[').count(),
            result.chars().filter(|&c| c == ']').count()
        );
    }

    #[test]
    fn test_type_annotation_with_nested_calls() {
        let source = "(: (func.call {:callee @add_two}\n              (arith.constant {:value (: 5 i32)})\n              (arith.constant {:value (: 7 i32)}))\n   i32)";
        assert_eq!(balance(source), source);
    }

    #[test]
    fn test_simple_type_annotation() {
        let source = "(: (func.call arg1\n              arg2)\n   i32)";
        assert_eq!(balance(source), source);
    }

    #[test]
    fn test_scf_if_sibling_regions() {
        let source = r#"(def result (scf.if {:result i32} cond
    (region
      (block []
        (scf.yield true_val)))
    (region
      (block []
        (scf.yield false_val)))))"#;
        assert_eq!(balance(source), source);
    }

    #[test]
    fn test_sibling_forms_same_indent_preserved() {
        let source = "(parent\n  (child1)\n  (child2)\n  (child3))";
        assert_eq!(balance(source), source);
    }
}
