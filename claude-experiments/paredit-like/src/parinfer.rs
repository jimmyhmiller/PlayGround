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
        Self {
            source: source.to_string(),
        }
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

            if was_escaped {
                continue;
            }

            if in_string || in_comment {
                continue;
            }

            match ch {
                '(' => parens += 1,
                ')' => parens -= 1,
                '[' => brackets += 1,
                ']' => brackets -= 1,
                '{' => braces += 1,
                '}' => braces -= 1,
                _ => {}
            }

            if parens < 0 || brackets < 0 || braces < 0 {
                return false;
            }
        }

        !in_string && parens == 0 && brackets == 0 && braces == 0
    }

    pub fn balance(&self) -> Result<String> {
        let options = Self::default_options();
        let answer = parinfer::paren_mode(&self.source, &options);

        let success = answer.success;
        let error = answer.error.clone();
        let output = answer.text.into_owned();

        let mut parser = crate::parser::ClojureParser::new()?;
        if !Self::is_structurally_balanced(&output) {
            if let Some(err) = error.clone() {
                return Err(anyhow!(
                    "Parinfer produced unbalanced output ({}): {}",
                    err.name,
                    err.message
                ));
            }
            return Err(anyhow!("Parinfer produced unbalanced output"));
        }

        if let Err(parse_err) = parser.parse_to_sexpr(&output) {
            if let Some(err) = error {
                return Err(anyhow!(
                    "Parinfer failed to produce parseable output ({}): {}",
                    err.name,
                    err.message
                ));
            }
            return Err(parse_err);
        }

        if !success {
            if let Some(err) = error {
                match err.name {
                    ErrorName::UnmatchedCloseParen
                    | ErrorName::UnmatchedOpenParen
                    | ErrorName::LeadingCloseParen => {
                        return Ok(output);
                    }
                    _ => {
                        return Err(anyhow!(
                            "Parinfer reported an unrecoverable error ({}): {}",
                            err.name,
                            err.message
                        ));
                    }
                }
            } else {
                return Err(anyhow!("Parinfer reported failure without details"));
            }
        }

        Ok(output)
    }
}
