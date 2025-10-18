use anyhow::Result;

#[derive(Debug, Clone, Copy, PartialEq)]
enum ParenType {
    Open(char),
    Close(char),
}

impl ParenType {
    fn matches(&self, other: &ParenType) -> bool {
        match (self, other) {
            (ParenType::Open('('), ParenType::Close(')')) => true,
            (ParenType::Open('['), ParenType::Close(']')) => true,
            (ParenType::Open('{'), ParenType::Close('}')) => true,
            _ => false,
        }
    }

    fn closing_char(&self) -> Option<char> {
        match self {
            ParenType::Open('(') => Some(')'),
            ParenType::Open('[') => Some(']'),
            ParenType::Open('{') => Some('}'),
            _ => None,
        }
    }
}

#[derive(Debug)]
struct Line {
    text: String,
    indent: usize,
}

pub struct Parinfer {
    lines: Vec<Line>,
}

impl Parinfer {
    pub fn new(source: &str) -> Self {
        let lines = source
            .lines()
            .map(|line| {
                let indent = line.chars().take_while(|c| c.is_whitespace()).count();
                Line {
                    text: line.to_string(),
                    indent,
                }
            })
            .collect();

        Parinfer { lines }
    }

    pub fn balance(&self) -> Result<String> {
        let mut result = Vec::new();
        let mut paren_stack: Vec<ParenType> = Vec::new();
        let mut indent_stack: Vec<usize> = Vec::new(); // Column where each paren was opened

        for (line_idx, line) in self.lines.iter().enumerate() {
            let mut new_line = String::new();
            let mut in_string = false;
            let mut in_comment = false;
            let mut escape_next = false;
            let line_start_stack_len = paren_stack.len(); // Track how many opens were before this line

            // Process line: handle parens, keep track of what needs closing
            for ch in line.text.chars() {
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
                    new_line.push(ch);
                    continue;
                }

                if !in_string && !in_comment {
                    match ch {
                        '(' | '[' | '{' => {
                            let paren_type = ParenType::Open(ch);
                            paren_stack.push(paren_type);
                            indent_stack.push(new_line.len()); // Track column position
                            new_line.push(ch);
                        }
                        ')' | ']' | '}' => {
                            let close_type = ParenType::Close(ch);
                            // Only allow closing parens that close something opened on THIS line
                            if paren_stack.len() > line_start_stack_len {
                                if let Some(open_type) = paren_stack.last() {
                                    if open_type.matches(&close_type) {
                                        // This closes something from the current line, keep it
                                        paren_stack.pop();
                                        indent_stack.pop();
                                        new_line.push(ch);
                                    }
                                    // Otherwise skip it (wrong type)
                                }
                            }
                            // Otherwise skip it (closes something from previous line or nothing)
                        }
                        _ => new_line.push(ch),
                    }
                } else {
                    new_line.push(ch);
                }
            }

            // Now add closing parens based on indentation
            if !in_string && !in_comment {
                // Find the next non-empty line to determine indentation
                let next_indent = {
                    let mut idx = line_idx + 1;
                    loop {
                        match self.lines.get(idx) {
                            Some(next_line) if next_line.text.trim().is_empty() => {
                                idx += 1;
                                continue;
                            }
                            Some(next_line) => break next_line.indent,
                            None => break 0,
                        }
                    }
                };

                // Close all parens that were opened at or after the next line's indentation
                let mut to_close = Vec::new();
                while let Some(paren) = paren_stack.last() {
                    if let Some(indent) = indent_stack.last() {
                        if *indent >= next_indent {
                            to_close.push(*paren);
                            paren_stack.pop();
                            indent_stack.pop();
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }

                // Add the closing parens
                for paren in to_close {
                    if let Some(close_char) = paren.closing_char() {
                        new_line.push(close_char);
                    }
                }
            }

            result.push(new_line);
        }

        // Close any remaining open parens at the end of the file
        while let Some(paren) = paren_stack.pop() {
            if let Some(close_char) = paren.closing_char() {
                if let Some(last_line) = result.last_mut() {
                    last_line.push(close_char);
                }
            }
        }

        Ok(result.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_balance_simple() {
        let source = "(defn foo\n  (+ 1 2";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        assert!(result.contains("(+ 1 2))"));
    }

    #[test]
    fn test_balance_nested() {
        let source = "(let [x 1\n      y 2\n  (+ x y";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        // Should close the vector and let form
        assert!(result.ends_with("))"));
        assert!(result.contains("y 2]"));
    }

    #[test]
    fn test_remove_extra_parens() {
        let source = "(defn foo []))";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        assert_eq!(result, "(defn foo [])");
    }

    #[test]
    fn test_preserve_strings() {
        let source = r#"(str "hello (world)")"#;
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        assert_eq!(result, r#"(str "hello (world)")"#);
    }

    #[test]
    fn test_preserve_comments() {
        let source = "(foo) ; comment with (parens)";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        assert_eq!(result, "(foo) ; comment with (parens)");
    }

    #[test]
    fn test_balance_multiple_missing_parens() {
        let source = "(defn fibonacci [n]\n  (if (<= n 1)\n    n\n    (+ (fibonacci (- n 1))\n       (fibonacci (- n 2";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();

        // Should have the right number of closing parens
        let open_count = source.chars().filter(|&c| c == '(' || c == '[').count();
        let result_close_count = result.chars().filter(|&c| c == ')' || c == ']').count();
        assert_eq!(open_count, result_close_count);
    }

    #[test]
    fn test_balance_mixed_brackets() {
        let source = "[1 2 {3 4\n      5 6\n  7 8";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        assert!(result.contains("}"));
        assert!(result.ends_with("]"));
    }

    #[test]
    fn test_balance_no_changes_needed() {
        let source = "(defn foo [x y]\n  (+ x y))";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        assert_eq!(result, source);
    }

    #[test]
    fn test_balance_empty_input() {
        let source = "";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_balance_only_whitespace() {
        let source = "   \n  \t  ";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        assert_eq!(result, source);
    }

    #[test]
    fn test_balance_mismatched_parens() {
        let source = "(defn foo [x)\n  (+ x 1]";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();

        // Should ignore all closing parens and regenerate correctly
        assert!(result.contains("[x]"));
        assert!(result.contains("(+ x 1))"));
    }

    #[test]
    fn test_balance_extra_closing_parens() {
        let source = "(+ 1 2)))";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        assert_eq!(result, "(+ 1 2)");
    }

    #[test]
    fn test_balance_string_with_escapes() {
        let source = r#"(str "hello \"world\" with (parens)")"#;
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        assert_eq!(result, source);
    }

    #[test]
    fn test_balance_multiline_string() {
        let source = "\"this is a\nmultiline string\nwith (parens)\"";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        assert_eq!(result, source);
    }

    #[test]
    fn test_balance_comment_at_end_of_line() {
        let source = "(defn foo [x] ; comment\n  (+ x 1";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        assert!(result.contains("; comment"));
        assert!(result.ends_with("))"));
    }

    #[test]
    fn test_balance_comment_with_parens() {
        let source = "(foo) ; (this is a comment with (parens))";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        assert_eq!(result, source);
    }

    #[test]
    fn test_balance_indentation_based_closing() {
        let source = "(let [x 1\n      y 2]\n  (if true\n    x\n    y))";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        assert_eq!(result, source);
    }

    #[test]
    fn test_balance_deeply_nested() {
        let source = "((((\n    inner\n  (\n    content";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();

        let open_count = source.chars().filter(|&c| c == '(').count();
        let result_close_count = result.chars().filter(|&c| c == ')').count();
        assert_eq!(open_count, result_close_count);
    }

    #[test]
    fn test_balance_vector_in_function() {
        let source = "(defn foo [a b c\n  (+ a b c";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        assert!(result.contains("[a b c]"));
        assert!(result.ends_with("))"));
    }

    #[test]
    fn test_balance_map_structure() {
        let source = "{:a 1\n :b 2\n :c {:nested 3\n     :value 4";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();

        let open_count = source.chars().filter(|&c| c == '{').count();
        let result_close_count = result.chars().filter(|&c| c == '}').count();
        assert_eq!(open_count, result_close_count);
    }

    #[test]
    fn test_balance_complex_indentation() {
        let source = r#"(defn complex-fn [x y z]
  (let [sum (+ x y z)
        product (* x y z)]
    (if (> sum 10)
      {:sum sum
       :product product
       :status :large}
      {:sum sum
       :product product
       :status :small"#;

        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();

        // Check that all structures are properly closed
        let open_parens = source.chars().filter(|&c| c == '(').count();
        let open_brackets = source.chars().filter(|&c| c == '[').count();
        let open_braces = source.chars().filter(|&c| c == '{').count();

        let close_parens = result.chars().filter(|&c| c == ')').count();
        let close_brackets = result.chars().filter(|&c| c == ']').count();
        let close_braces = result.chars().filter(|&c| c == '}').count();

        assert_eq!(open_parens, close_parens);
        assert_eq!(open_brackets, close_brackets);
        assert_eq!(open_braces, close_braces);
    }

    #[test]
    fn test_balance_preserve_existing_structure() {
        let source = "(defn well-formed [x]\n  (* x 2))";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        assert_eq!(result, source);
    }

    #[test]
    fn test_balance_handle_character_literals() {
        let source = r"(str \( \) \[ \])";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();
        assert_eq!(result, source);
    }

    #[test]
    fn test_paren_type_matching() {
        let paren_open = ParenType::Open('(');
        let paren_close = ParenType::Close(')');
        let bracket_open = ParenType::Open('[');
        let bracket_close = ParenType::Close(']');
        let brace_open = ParenType::Open('{');
        let brace_close = ParenType::Close('}');

        assert!(paren_open.matches(&paren_close));
        assert!(bracket_open.matches(&bracket_close));
        assert!(brace_open.matches(&brace_close));

        assert!(!paren_open.matches(&bracket_close));
        assert!(!bracket_open.matches(&brace_close));
        assert!(!brace_open.matches(&paren_close));
    }

    #[test]
    fn test_paren_type_closing_char() {
        assert_eq!(ParenType::Open('(').closing_char(), Some(')'));
        assert_eq!(ParenType::Open('[').closing_char(), Some(']'));
        assert_eq!(ParenType::Open('{').closing_char(), Some('}'));
        assert_eq!(ParenType::Close(')').closing_char(), None);
    }

    #[test]
    fn test_balance_dangling_zero() {
        // This is the specific bug case from parser.lisp
        let source = "(defn main-fn []\n  (let [a 1]\n    (foo)))))))))\n\n    0))\n\n(main-fn)";
        let parinfer = Parinfer::new(source);
        let result = parinfer.balance().unwrap();

        // The 0 should be inside the function, not dangling
        assert!(!result.contains("0\n\n(main-fn)"), "0 should not be dangling before (main-fn)");
        // Should have the 0 followed by closing parens before the final (main-fn)
        assert!(result.contains("0))\n\n(main-fn)"));
    }
}
