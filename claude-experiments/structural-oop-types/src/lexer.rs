//! Lexer for the expression language (JavaScript-like syntax)
//!
//! Tokens:
//! - Keywords: true, false, let, in, this
//! - Symbols: =>, ==, =, ., :, ?, {, }, (, ), ,, &&, ||
//! - Identifiers: [a-zA-Z_][a-zA-Z0-9_]*
//! - Integers: [0-9]+

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Literals
    True,
    False,
    Int(i64),

    // Keywords
    Let,
    Rec,
    And,
    In,
    This,

    // Identifiers
    Ident(String),

    // Symbols
    Arrow,      // =>
    EqEq,       // ==
    Equals,     // =
    Dot,        // .
    Colon,      // :
    Question,   // ?
    Comma,      // ,
    LBrace,     // {
    RBrace,     // }
    LParen,     // (
    RParen,     // )
    AndAnd,     // &&
    OrOr,       // ||
    Plus,       // +
    Minus,      // -
    Star,       // *
    Slash,      // /
    PlusPlus,   // ++ (string concat)

    // String literals
    String(String),

    // End of input
    Eof,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::True => write!(f, "true"),
            Token::False => write!(f, "false"),
            Token::Int(n) => write!(f, "{}", n),
            Token::Let => write!(f, "let"),
            Token::Rec => write!(f, "rec"),
            Token::And => write!(f, "and"),
            Token::In => write!(f, "in"),
            Token::This => write!(f, "this"),
            Token::Ident(s) => write!(f, "{}", s),
            Token::Arrow => write!(f, "=>"),
            Token::EqEq => write!(f, "=="),
            Token::Equals => write!(f, "="),
            Token::Dot => write!(f, "."),
            Token::Colon => write!(f, ":"),
            Token::Question => write!(f, "?"),
            Token::Comma => write!(f, ","),
            Token::LBrace => write!(f, "{{"),
            Token::RBrace => write!(f, "}}"),
            Token::LParen => write!(f, "("),
            Token::RParen => write!(f, ")"),
            Token::AndAnd => write!(f, "&&"),
            Token::OrOr => write!(f, "||"),
            Token::Plus => write!(f, "+"),
            Token::Minus => write!(f, "-"),
            Token::Star => write!(f, "*"),
            Token::Slash => write!(f, "/"),
            Token::PlusPlus => write!(f, "++"),
            Token::String(s) => write!(f, "\"{}\"", s),
            Token::Eof => write!(f, "EOF"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LexError {
    pub message: String,
    pub position: usize,
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Lexer error at position {}: {}", self.position, self.message)
    }
}

pub struct Lexer<'a> {
    input: &'a str,
    chars: std::iter::Peekable<std::str::CharIndices<'a>>,
    position: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Lexer {
            input,
            chars: input.char_indices().peekable(),
            position: 0,
        }
    }

    fn advance(&mut self) -> Option<char> {
        if let Some((pos, ch)) = self.chars.next() {
            self.position = pos + ch.len_utf8();
            Some(ch)
        } else {
            None
        }
    }

    fn peek(&mut self) -> Option<char> {
        self.chars.peek().map(|(_, ch)| *ch)
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn read_identifier(&mut self, first: char) -> String {
        let mut ident = String::new();
        ident.push(first);

        while let Some(ch) = self.peek() {
            if ch.is_alphanumeric() || ch == '_' {
                ident.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        ident
    }

    fn read_number(&mut self, first: char) -> i64 {
        let mut num_str = String::new();
        num_str.push(first);

        while let Some(ch) = self.peek() {
            if ch.is_ascii_digit() {
                num_str.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        num_str.parse().unwrap_or(0)
    }

    fn read_string(&mut self) -> Result<Token, LexError> {
        let mut s = String::new();
        loop {
            match self.advance() {
                Some('"') => return Ok(Token::String(s)),
                Some('\\') => {
                    // Escape sequences
                    match self.advance() {
                        Some('n') => s.push('\n'),
                        Some('t') => s.push('\t'),
                        Some('r') => s.push('\r'),
                        Some('\\') => s.push('\\'),
                        Some('"') => s.push('"'),
                        Some(c) => {
                            return Err(LexError {
                                message: format!("Unknown escape sequence: \\{}", c),
                                position: self.position,
                            })
                        }
                        None => {
                            return Err(LexError {
                                message: "Unterminated string".to_string(),
                                position: self.position,
                            })
                        }
                    }
                }
                Some(c) => s.push(c),
                None => {
                    return Err(LexError {
                        message: "Unterminated string".to_string(),
                        position: self.position,
                    })
                }
            }
        }
    }

    pub fn next_token(&mut self) -> Result<Token, LexError> {
        self.skip_whitespace();

        let ch = match self.advance() {
            Some(ch) => ch,
            None => return Ok(Token::Eof),
        };

        match ch {
            // Single-character tokens
            '.' => Ok(Token::Dot),
            ',' => Ok(Token::Comma),
            ':' => Ok(Token::Colon),
            '?' => Ok(Token::Question),
            '{' => Ok(Token::LBrace),
            '}' => Ok(Token::RBrace),
            '(' => Ok(Token::LParen),
            ')' => Ok(Token::RParen),

            // Two-character tokens
            '=' => {
                if self.peek() == Some('>') {
                    self.advance();
                    Ok(Token::Arrow)
                } else if self.peek() == Some('=') {
                    self.advance();
                    Ok(Token::EqEq)
                } else {
                    Ok(Token::Equals)
                }
            }
            '&' => {
                if self.peek() == Some('&') {
                    self.advance();
                    Ok(Token::AndAnd)
                } else {
                    Err(LexError {
                        message: format!("Unexpected character: {}", ch),
                        position: self.position,
                    })
                }
            }
            '|' => {
                if self.peek() == Some('|') {
                    self.advance();
                    Ok(Token::OrOr)
                } else {
                    Err(LexError {
                        message: format!("Unexpected character: {}", ch),
                        position: self.position,
                    })
                }
            }
            '+' => {
                if self.peek() == Some('+') {
                    self.advance();
                    Ok(Token::PlusPlus)
                } else {
                    Ok(Token::Plus)
                }
            }
            '*' => Ok(Token::Star),
            '/' => {
                if self.peek() == Some('/') {
                    // Comment - skip to end of line
                    self.advance(); // consume second /
                    while let Some(ch) = self.advance() {
                        if ch == '\n' {
                            break;
                        }
                    }
                    // Recurse to get next token
                    self.next_token()
                } else {
                    Ok(Token::Slash)
                }
            }
            '"' => self.read_string(),
            '-' => {
                // Could be negative number or minus operator
                // For simplicity, we always return Minus token
                // Parser will handle negative literals as (0 - n)
                Ok(Token::Minus)
            }

            // Numbers
            ch if ch.is_ascii_digit() => {
                let num = self.read_number(ch);
                Ok(Token::Int(num))
            }

            // Identifiers and keywords
            ch if ch.is_alphabetic() || ch == '_' => {
                let ident = self.read_identifier(ch);
                match ident.as_str() {
                    "true" => Ok(Token::True),
                    "false" => Ok(Token::False),
                    "let" => Ok(Token::Let),
                    "rec" => Ok(Token::Rec),
                    "and" => Ok(Token::And),
                    "in" => Ok(Token::In),
                    "this" => Ok(Token::This),
                    _ => Ok(Token::Ident(ident)),
                }
            }

            _ => Err(LexError {
                message: format!("Unexpected character: {}", ch),
                position: self.position,
            }),
        }
    }

    pub fn tokenize(&mut self) -> Result<Vec<Token>, LexError> {
        let mut tokens = Vec::new();
        loop {
            let token = self.next_token()?;
            if token == Token::Eof {
                tokens.push(token);
                break;
            }
            tokens.push(token);
        }
        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lex_literals() {
        let mut lexer = Lexer::new("true false 42");
        assert_eq!(lexer.next_token().unwrap(), Token::True);
        assert_eq!(lexer.next_token().unwrap(), Token::False);
        assert_eq!(lexer.next_token().unwrap(), Token::Int(42));
    }

    #[test]
    fn test_lex_minus() {
        let mut lexer = Lexer::new("-10");
        assert_eq!(lexer.next_token().unwrap(), Token::Minus);
        assert_eq!(lexer.next_token().unwrap(), Token::Int(10));
    }

    #[test]
    fn test_lex_arrow_function() {
        let mut lexer = Lexer::new("x => x");
        assert_eq!(lexer.next_token().unwrap(), Token::Ident("x".to_string()));
        assert_eq!(lexer.next_token().unwrap(), Token::Arrow);
        assert_eq!(lexer.next_token().unwrap(), Token::Ident("x".to_string()));
    }

    #[test]
    fn test_lex_object() {
        let mut lexer = Lexer::new("{ x: 42, y: true }");
        assert_eq!(lexer.next_token().unwrap(), Token::LBrace);
        assert_eq!(lexer.next_token().unwrap(), Token::Ident("x".to_string()));
        assert_eq!(lexer.next_token().unwrap(), Token::Colon);
        assert_eq!(lexer.next_token().unwrap(), Token::Int(42));
        assert_eq!(lexer.next_token().unwrap(), Token::Comma);
        assert_eq!(lexer.next_token().unwrap(), Token::Ident("y".to_string()));
        assert_eq!(lexer.next_token().unwrap(), Token::Colon);
        assert_eq!(lexer.next_token().unwrap(), Token::True);
        assert_eq!(lexer.next_token().unwrap(), Token::RBrace);
    }

    #[test]
    fn test_lex_ternary() {
        let mut lexer = Lexer::new("x ? 1 : 2");
        assert_eq!(lexer.next_token().unwrap(), Token::Ident("x".to_string()));
        assert_eq!(lexer.next_token().unwrap(), Token::Question);
        assert_eq!(lexer.next_token().unwrap(), Token::Int(1));
        assert_eq!(lexer.next_token().unwrap(), Token::Colon);
        assert_eq!(lexer.next_token().unwrap(), Token::Int(2));
    }

    #[test]
    fn test_lex_comment() {
        let mut lexer = Lexer::new("42 // this is a comment\n true");
        assert_eq!(lexer.next_token().unwrap(), Token::Int(42));
        assert_eq!(lexer.next_token().unwrap(), Token::True);
    }
}
