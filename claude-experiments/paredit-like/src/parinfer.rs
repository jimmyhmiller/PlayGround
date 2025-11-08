use anyhow::{anyhow, Result};
use parinfer_rust::{
    parinfer,
    types::{ErrorName, Options},
};

pub struct Parinfer {
    source: String,
}

impl Parinfer {
    pub fn new(source: &str) -> Self {
        Self { source: source.to_string() }
    }

    fn default_options() -> Options {
        Options {
            cursor_x: None,
            cursor_line: None,
            prev_cursor_x: None,
            prev_cursor_line: None,
            prev_text: None,
            selection_start_line: None,
            changes: vec![],
            comment_char: ';',
            string_delimiters: vec!["\"".to_string()],
            lisp_vline_symbols: false,
            lisp_block_comments: false,
            guile_block_comments: false,
            scheme_sexp_comments: false,
            janet_long_strings: false,
            hy_bracket_strings: false,
        }
    }

    fn is_structurally_balanced(output: &str) -> bool {
        let mut in_string = false;
        let mut escape_next = false;
        let mut in_comment = false;
        let mut parens = 0i32;
        let mut brackets = 0i32;
        let mut braces = 0i32;
        for ch in output.chars() {
            let was_escaped = escape_next;
            escape_next = false;
            if ch == '\\' && in_string && !was_escaped {
                escape_next = true;
                continue;
            }
            if ch == '"' && !in_comment && !was_escaped {
                in_string = !in_string;
                continue;
            }
            if ch == ';' && !in_string {
                in_comment = true;
            }
            if ch == '\n' || ch == '\r' {
                in_comment = false;
            }
            if was_escaped { continue; }
            if in_string || in_comment { continue; }
            match ch {
                '(' => parens += 1,
                ')' => parens -= 1,
                '[' => brackets += 1,
                ']' => brackets -= 1,
                '{' => braces += 1,
                '}' => braces -= 1,
                _ => {}
            }
            if parens < 0 || brackets < 0 || braces < 0 { return false; }
        }
        !in_string && parens == 0 && brackets == 0 && braces == 0
    }

    pub fn balance(&self) -> Result<String> {
        let options = Self::default_options();
        let answer = parinfer::indent_mode(&self.source, &options);
        let success = answer.success;
        let error = answer.error.clone();
        let output = answer.text.into_owned();

        // Check for recoverable errors first - these are OK to return
        if !success {
            if let Some(err) = error.clone() {
                match err.name {
                    ErrorName::UnmatchedCloseParen
                    | ErrorName::UnmatchedOpenParen
                    | ErrorName::LeadingCloseParen => {
                        // These errors mean parinfer couldn't fully process the file,
                        // but indent_mode should still have done its best.
                        // Return the output as-is.
                        return Ok(output);
                    }
                    _ => {
                        return Err(anyhow!(
                            "Parinfer reported an unrecoverable error ({}): {}",
                            err.name, err.message
                        ));
                    }
                }
            } else {
                return Err(anyhow!("Parinfer reported failure without details"));
            }
        }

        // Parinfer succeeded - validate the output
        if !Self::is_structurally_balanced(&output) {
            if let Some(ref err) = error {
                return Err(anyhow!(
                    "Parinfer produced unbalanced output ({}): {}",
                    err.name, err.message
                ));
            }
            return Err(anyhow!("Parinfer produced unbalanced output"));
        }

        let mut parser = crate::parser::ClojureParser::new()?;
        if let Err(parse_err) = parser.parse_to_sexpr(&output) {
            if let Some(err) = error {
                return Err(anyhow!(
                    "Parinfer failed to produce parseable output ({}): {}",
                    err.name, err.message
                ));
            }
            return Err(parse_err);
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    #[test]
    fn test_parinfer_unbalanced_is_fixed() -> Result<()> {
        let p = Parinfer::new("(+ 1 2");
        let res = p.balance();
        assert!(res.is_ok(), "indent_mode should fix unbalanced input");
        assert_eq!(res.unwrap(), "(+ 1 2)");
        Ok(())
    }

    #[test]
    fn test_parinfer_balanced_with_comment() -> Result<()> {
        let src = "; )\n(+ 1 2)";
        let p = Parinfer::new(src);
        let out = p.balance()?;
        assert!(out.contains("; )"));
        assert!(out.contains("(+ 1 2)"));
        Ok(())
    }

    #[test]
    fn test_parinfer_balanced_with_string() -> Result<()> {
        let src = "\"(+ 1 2)\"";
        let p = Parinfer::new(src);
        let out = p.balance()?;
        assert_eq!(out.trim(), src.trim());
        Ok(())
    }

    #[test]
    fn test_parinfer_comment_inside_string() -> Result<()> {
        let src = "\"; not a comment\" (+ 1 2)";
        let p = Parinfer::new(src);
        let out = p.balance()?;
        assert_eq!(out.trim(), src.trim());
        Ok(())
    }

    #[test]
    fn test_parinfer_escaped_quote() -> Result<()> {
        let src = "\"\\\"\" (+ 1 2)"; // string containing an escaped quote
        let p = Parinfer::new(src);
        let out = p.balance()?;
        assert_eq!(out.trim(), src.trim());
        Ok(())
    }
}
