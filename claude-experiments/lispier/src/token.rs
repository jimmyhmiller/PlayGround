use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    LeftParen,    // (
    RightParen,   // )
    LeftBracket,  // [
    RightBracket, // ]
    LeftBrace,    // {
    RightBrace,   // }
    Symbol,       // any identifier
    String,       // "..."
    Number,       // 123, 3.14
    Keyword,      // :keyword
    BlockLabel,   // ^label
    Colon,        // : (for type annotations)
    Backtick,     // ` (quasiquote)
    Tilde,        // ~ (unquote)
    TildeAt,      // ~@ (unquote-splice)
    Eof,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub token_type: TokenType,
    pub lexeme: String,
    pub line: usize,
    pub column: usize,
}

impl Token {
    pub fn new(token_type: TokenType, lexeme: impl Into<String>, line: usize, column: usize) -> Self {
        Self {
            token_type,
            lexeme: lexeme.into(),
            line,
            column,
        }
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Token({:?}, \"{}\", {}:{})",
            self.token_type, self.lexeme, self.line, self.column
        )
    }
}
