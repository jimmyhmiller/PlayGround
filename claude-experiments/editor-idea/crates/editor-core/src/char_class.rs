//! Public character categorizer (mirrors CM6's `CharCategory`).
//! Used by external code that wants a coarse classification: Word, Space,
//! or Other. `commands.rs` maintains a finer-grained internal enum for
//! delete-by-group rules that distinguishes newlines and punctuation.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CharCategory {
    Word,
    Space,
    Other,
}

pub fn categorize(c: char) -> CharCategory {
    if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
        CharCategory::Space
    } else if c.is_alphanumeric() || c == '_' {
        CharCategory::Word
    } else {
        CharCategory::Other
    }
}
