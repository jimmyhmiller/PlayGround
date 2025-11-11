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

    pub fn balance(&self) -> Result<String> {
        let lines: Vec<&str> = self.source.lines().collect();
        let mut result_lines = Vec::new();
        let mut delim_stack: Vec<DelimInfo> = Vec::new();

        for (line_idx, line) in lines.iter().enumerate() {
            let line_indent = line.chars().take_while(|c| c.is_whitespace()).count();
            let mut new_line = String::new();
            let mut in_string = false;
            let mut in_comment = false;
            let mut escape_next = false;

            // Process characters in the line
            for ch in line.chars() {
                // Handle escape sequences
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
                    });
                    new_line.push(ch);
                } else if let Some(close_type) = DelimType::from_close(ch) {
                    // Check if this closing delimiter matches the top of stack
                    if let Some(open_info) = delim_stack.last() {
                        if open_info.delim_type.matches(&close_type) {
                            delim_stack.pop();
                            new_line.push(ch);
                        }
                        // If doesn't match, skip it (mismatched delimiter)
                    }
                    // If stack empty, skip it (extra closing delimiter)
                } else {
                    new_line.push(ch);
                }
            }

            // After processing the line, check if we need to auto-close delimiters
            // based on indentation of the next non-empty line
            if !in_string && !in_comment {
                let next_indent = Self::find_next_indent(&lines, line_idx);

                // Close delimiters that were opened on lines with indent > next_indent
                let mut to_close = Vec::new();
                while let Some(open_info) = delim_stack.last() {
                    if next_indent < line_indent && open_info.line_indent > next_indent {
                        to_close.push(open_info.delim_type);
                        delim_stack.pop();
                    } else {
                        break;
                    }
                }

                // Add closing delimiters to the line
                for delim_type in to_close {
                    new_line.push(delim_type.close_char());
                }
            }

            result_lines.push(new_line);
        }

        // Close any remaining open delimiters at the end of file
        while let Some(open_info) = delim_stack.pop() {
            if let Some(last_line) = result_lines.last_mut() {
                last_line.push(open_info.delim_type.close_char());
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
}
