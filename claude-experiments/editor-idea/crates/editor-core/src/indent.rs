//! Plain-text indentation rules. No tree-sitter — rules are bracket-driven:
//!  * a line that ends with an "open" bracket increases the indent of the
//!    next line by one unit
//!  * a line that begins with a "close" bracket dedents one unit
//! Bracket-pair info also drives "bracket explosion" in
//! `insert_newline_and_indent` (see `commands::insert_newline_and_indent`).

#[derive(Debug, Clone)]
pub struct IndentRules {
    /// Characters whose presence at the end of the previous line signals
    /// "indent the next line".
    pub open: Vec<char>,
    /// Characters whose presence at the start of the current line signals
    /// "dedent this line".
    pub close: Vec<char>,
    /// Matched pairs for bracket explosion (open, close).
    pub pairs: Vec<(char, char)>,
}

impl Default for IndentRules {
    fn default() -> Self {
        Self {
            open: vec!['{', '[', '('],
            close: vec!['}', ']', ')'],
            pairs: vec![('{', '}'), ('[', ']'), ('(', ')')],
        }
    }
}

impl IndentRules {
    pub fn matching_close(&self, open: char) -> Option<char> {
        self.pairs.iter().find(|(o, _)| *o == open).map(|p| p.1)
    }
}
