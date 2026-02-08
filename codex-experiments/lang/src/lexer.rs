use crate::token::{Span, Token, TokenKind};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LexError {
    pub message: String,
    pub span: Span,
}

pub struct Lexer<'a> {
    input: &'a str,
    bytes: &'a [u8],
    pos: usize,
    len: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            input,
            bytes: input.as_bytes(),
            pos: 0,
            len: input.len(),
        }
    }

    pub fn lex_all(mut self) -> Result<Vec<Token>, Vec<LexError>> {
        let mut tokens = Vec::new();
        let mut errors = Vec::new();
        while self.skip_ws_and_comments(&mut errors) {
            if self.pos >= self.len {
                break;
            }
            match self.next_token() {
                Ok(tok) => tokens.push(tok),
                Err(err) => {
                    errors.push(err);
                    // Advance at least one byte to avoid infinite loops.
                    self.pos = self.pos.saturating_add(1).min(self.len);
                }
            }
        }
        if errors.is_empty() {
            Ok(tokens)
        } else {
            Err(errors)
        }
    }

    fn skip_ws_and_comments(&mut self, errors: &mut Vec<LexError>) -> bool {
        let mut progressed = false;
        loop {
            let start = self.pos;
            while self.pos < self.len {
                let b = self.bytes[self.pos];
                if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
                    self.pos += 1;
                } else {
                    break;
                }
            }

            if self.pos + 1 < self.len && self.bytes[self.pos] == b'/' && self.bytes[self.pos + 1] == b'/' {
                // Line comment
                self.pos += 2;
                while self.pos < self.len && self.bytes[self.pos] != b'\n' {
                    self.pos += 1;
                }
                progressed = true;
                continue;
            }

            if self.pos + 1 < self.len && self.bytes[self.pos] == b'/' && self.bytes[self.pos + 1] == b'*' {
                // Block comment (non-nesting for v0)
                let comment_start = self.pos;
                self.pos += 2;
                let mut found_end = false;
                while self.pos + 1 < self.len {
                    if self.bytes[self.pos] == b'*' && self.bytes[self.pos + 1] == b'/' {
                        self.pos += 2;
                        found_end = true;
                        break;
                    }
                    self.pos += 1;
                }
                if !found_end {
                    errors.push(LexError {
                        message: "Unterminated block comment".to_string(),
                        span: Span::new(comment_start, self.len),
                    });
                    self.pos = self.len;
                }
                progressed = true;
                continue;
            }

            if self.pos == start {
                break;
            }
            progressed = true;
        }
        progressed || self.pos < self.len
    }

    fn next_token(&mut self) -> Result<Token, LexError> {
        if self.pos >= self.len {
            return Err(LexError {
                message: "Unexpected end of input".to_string(),
                span: Span::new(self.pos, self.pos),
            });
        }
        let b = self.bytes[self.pos];
        if is_ident_start(b) {
            return Ok(self.lex_ident_or_keyword());
        }
        if is_digit(b) {
            return self.lex_number();
        }
        match b {
            b'"' => self.lex_string(),
            b'\'' => self.lex_char(),
            b'(' => Ok(self.simple_token(TokenKind::LParen, 1)),
            b')' => Ok(self.simple_token(TokenKind::RParen, 1)),
            b'{' => Ok(self.simple_token(TokenKind::LBrace, 1)),
            b'}' => Ok(self.simple_token(TokenKind::RBrace, 1)),
            b',' => Ok(self.simple_token(TokenKind::Comma, 1)),
            b';' => Ok(self.simple_token(TokenKind::Semi, 1)),
            b'.' => {
                if self.match_bytes(b"...") {
                    Ok(self.simple_token(TokenKind::Ellipsis, 3))
                } else {
                    Ok(self.simple_token(TokenKind::Dot, 1))
                }
            }
            b':' => {
                if self.match_bytes(b"::") {
                    Ok(self.simple_token(TokenKind::ColonColon, 2))
                } else {
                    Ok(self.simple_token(TokenKind::Colon, 1))
                }
            }
            b'-' => {
                if self.match_bytes(b"->") {
                    Ok(self.simple_token(TokenKind::Arrow, 2))
                } else {
                    Ok(self.simple_token(TokenKind::Minus, 1))
                }
            }
            b'+' => Ok(self.simple_token(TokenKind::Plus, 1)),
            b'*' => Ok(self.simple_token(TokenKind::Star, 1)),
            b'%' => Ok(self.simple_token(TokenKind::Percent, 1)),
            b'!' => {
                if self.match_bytes(b"!=") {
                    Ok(self.simple_token(TokenKind::NotEq, 2))
                } else {
                    Ok(self.simple_token(TokenKind::Bang, 1))
                }
            }
            b'=' => {
                if self.match_bytes(b"==") {
                    Ok(self.simple_token(TokenKind::EqEq, 2))
                } else if self.match_bytes(b"=>") {
                    Ok(self.simple_token(TokenKind::FatArrow, 2))
                } else {
                    Ok(self.simple_token(TokenKind::Eq, 1))
                }
            }
            b'<' => {
                if self.match_bytes(b"<=") {
                    Ok(self.simple_token(TokenKind::LtEq, 2))
                } else {
                    Ok(self.simple_token(TokenKind::Lt, 1))
                }
            }
            b'>' => {
                if self.match_bytes(b">=") {
                    Ok(self.simple_token(TokenKind::GtEq, 2))
                } else {
                    Ok(self.simple_token(TokenKind::Gt, 1))
                }
            }
            b'/' => Ok(self.simple_token(TokenKind::Slash, 1)),
            b'&' => {
                if self.match_bytes(b"&&") {
                    Ok(self.simple_token(TokenKind::AndAnd, 2))
                } else {
                    Err(LexError {
                        message: "Unexpected '&'".to_string(),
                        span: Span::new(self.pos, self.pos + 1),
                    })
                }
            }
            b'|' => {
                if self.match_bytes(b"||") {
                    Ok(self.simple_token(TokenKind::OrOr, 2))
                } else {
                    Err(LexError {
                        message: "Unexpected '|'".to_string(),
                        span: Span::new(self.pos, self.pos + 1),
                    })
                }
            }
            _ => Err(LexError {
                message: format!("Unexpected character '{}'", b as char),
                span: Span::new(self.pos, self.pos + 1),
            }),
        }
    }

    fn simple_token(&mut self, kind: TokenKind, width: usize) -> Token {
        let start = self.pos;
        self.pos += width;
        Token {
            kind,
            span: Span::new(start, self.pos),
        }
    }

    fn match_bytes(&self, pat: &[u8]) -> bool {
        let end = self.pos + pat.len();
        end <= self.len && &self.bytes[self.pos..end] == pat
    }

    fn lex_ident_or_keyword(&mut self) -> Token {
        let start = self.pos;
        self.pos += 1;
        while self.pos < self.len && is_ident_continue(self.bytes[self.pos]) {
            self.pos += 1;
        }
        let text = &self.input[start..self.pos];
        let kind = match text {
            "module" => TokenKind::Module,
            "use" => TokenKind::Use,
            "pub" => TokenKind::Pub,
            "struct" => TokenKind::Struct,
            "enum" => TokenKind::Enum,
            "trait" => TokenKind::Trait,
            "impl" => TokenKind::Impl,
            "fn" => TokenKind::Fn,
            "let" => TokenKind::Let,
            "mut" => TokenKind::Mut,
            "if" => TokenKind::If,
            "else" => TokenKind::Else,
            "while" => TokenKind::While,
            "match" => TokenKind::Match,
            "return" => TokenKind::Return,
            "break" => TokenKind::Break,
            "continue" => TokenKind::Continue,
            "extern" => TokenKind::Extern,
            "repr" => TokenKind::Repr,
            "true" => TokenKind::True,
            "false" => TokenKind::False,
            _ => TokenKind::Ident(text.to_string()),
        };
        Token {
            kind,
            span: Span::new(start, self.pos),
        }
    }

    fn lex_number(&mut self) -> Result<Token, LexError> {
        let start = self.pos;
        self.pos += 1;
        while self.pos < self.len && (is_digit(self.bytes[self.pos]) || self.bytes[self.pos] == b'_') {
            self.pos += 1;
        }
        let mut is_float = false;
        if self.pos < self.len && self.bytes[self.pos] == b'.' {
            if self.pos + 1 < self.len && is_digit(self.bytes[self.pos + 1]) {
                is_float = true;
                self.pos += 1;
                while self.pos < self.len && (is_digit(self.bytes[self.pos]) || self.bytes[self.pos] == b'_') {
                    self.pos += 1;
                }
            }
        }
        let text = self.input[start..self.pos].to_string();
        let kind = if is_float {
            TokenKind::Float(text)
        } else {
            TokenKind::Int(text)
        };
        Ok(Token {
            kind,
            span: Span::new(start, self.pos),
        })
    }

    fn lex_string(&mut self) -> Result<Token, LexError> {
        let start = self.pos;
        self.pos += 1; // skip opening quote
        let mut result = String::new();
        while self.pos < self.len {
            let b = self.bytes[self.pos];
            if b == b'"' {
                self.pos += 1;
                return Ok(Token {
                    kind: TokenKind::Str(result),
                    span: Span::new(start, self.pos),
                });
            }
            if b == b'\\' {
                if self.pos + 1 >= self.len {
                    break;
                }
                let esc = self.bytes[self.pos + 1];
                match esc {
                    b'n' => result.push('\n'),
                    b't' => result.push('\t'),
                    b'"' => result.push('"'),
                    b'\\' => result.push('\\'),
                    _ => {
                        return Err(LexError {
                            message: "Invalid string escape".to_string(),
                            span: Span::new(self.pos, self.pos + 2),
                        });
                    }
                }
                self.pos += 2;
                continue;
            }
            result.push(b as char);
            self.pos += 1;
        }
        Err(LexError {
            message: "Unterminated string literal".to_string(),
            span: Span::new(start, self.pos),
        })
    }

    fn lex_char(&mut self) -> Result<Token, LexError> {
        let start = self.pos;
        self.pos += 1; // skip opening '
        if self.pos >= self.len {
            return Err(LexError {
                message: "Unterminated char literal".to_string(),
                span: Span::new(start, self.pos),
            });
        }
        let b = self.bytes[self.pos];
        let value = if b == b'\\' {
            if self.pos + 1 >= self.len {
                return Err(LexError {
                    message: "Unterminated char escape".to_string(),
                    span: Span::new(start, self.pos),
                });
            }
            let esc = self.bytes[self.pos + 1];
            match esc {
                b'n' => {
                    self.pos += 2;
                    b'\n'
                }
                b't' => {
                    self.pos += 2;
                    b'\t'
                }
                b'r' => {
                    self.pos += 2;
                    b'\r'
                }
                b'0' => {
                    self.pos += 2;
                    0
                }
                b'\'' => {
                    self.pos += 2;
                    b'\''
                }
                b'\\' => {
                    self.pos += 2;
                    b'\\'
                }
                b'x' => {
                    if self.pos + 3 >= self.len {
                        return Err(LexError {
                            message: "Invalid hex escape".to_string(),
                            span: Span::new(self.pos, self.pos + 2),
                        });
                    }
                    let h1 = self.bytes[self.pos + 2];
                    let h2 = self.bytes[self.pos + 3];
                    let v1 = hex_val(h1);
                    let v2 = hex_val(h2);
                    if v1.is_none() || v2.is_none() {
                        return Err(LexError {
                            message: "Invalid hex escape".to_string(),
                            span: Span::new(self.pos, self.pos + 4),
                        });
                    }
                    self.pos += 4;
                    (v1.unwrap() << 4) | v2.unwrap()
                }
                _ => {
                    return Err(LexError {
                        message: "Invalid char escape".to_string(),
                        span: Span::new(self.pos, self.pos + 2),
                    });
                }
            }
        } else {
            if b == b'\n' || b == b'\r' {
                return Err(LexError {
                    message: "Unterminated char literal".to_string(),
                    span: Span::new(start, self.pos),
                });
            }
            if b >= 0x80 {
                return Err(LexError {
                    message: "Non-ASCII char literal".to_string(),
                    span: Span::new(self.pos, self.pos + 1),
                });
            }
            self.pos += 1;
            b
        };
        if self.pos >= self.len || self.bytes[self.pos] != b'\'' {
            return Err(LexError {
                message: "Char literal must be a single byte".to_string(),
                span: Span::new(start, self.pos),
            });
        }
        self.pos += 1; // closing '
        Ok(Token {
            kind: TokenKind::Char(value),
            span: Span::new(start, self.pos),
        })
    }
}

fn is_ident_start(b: u8) -> bool {
    (b'A'..=b'Z').contains(&b) || (b'a'..=b'z').contains(&b) || b == b'_'
}

fn is_ident_continue(b: u8) -> bool {
    is_ident_start(b) || is_digit(b)
}

fn is_digit(b: u8) -> bool {
    (b'0'..=b'9').contains(&b)
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(10 + (b - b'a')),
        b'A'..=b'F' => Some(10 + (b - b'A')),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lex_basic_tokens() {
        let src = r#"module a::b; fn main() -> I64 { let x = 1 + 2; }"#;
        let tokens = Lexer::new(src).lex_all().unwrap();
        assert!(tokens.iter().any(|t| matches!(t.kind, TokenKind::Module)));
        assert!(tokens.iter().any(|t| matches!(t.kind, TokenKind::Fn)));
        assert!(tokens.iter().any(|t| matches!(t.kind, TokenKind::Arrow)));
        assert!(tokens.iter().any(|t| matches!(t.kind, TokenKind::Int(_))));
    }

    #[test]
    fn lex_string_with_escape() {
        let src = "\"hi\\n\"";
        let tokens = Lexer::new(src).lex_all().unwrap();
        assert_eq!(tokens.len(), 1);
        match &tokens[0].kind {
            TokenKind::Str(s) => assert_eq!(s, "hi\n"),
            _ => panic!("expected string"),
        }
    }

    #[test]
    fn lex_line_and_block_comments() {
        let src = "// line\nfn x() -> I64 { /* block */ 1 }";
        let tokens = Lexer::new(src).lex_all().unwrap();
        assert!(tokens.iter().any(|t| matches!(t.kind, TokenKind::Fn)));
        assert!(tokens.iter().any(|t| matches!(t.kind, TokenKind::Int(_))));
    }

    #[test]
    fn lex_char_literal() {
        let src = "'a' '\\n' '\\x41'";
        let tokens = Lexer::new(src).lex_all().unwrap();
        assert!(matches!(tokens[0].kind, TokenKind::Char(b'a')));
        assert!(matches!(tokens[1].kind, TokenKind::Char(b'\n')));
        assert!(matches!(tokens[2].kind, TokenKind::Char(0x41)));
    }
}
