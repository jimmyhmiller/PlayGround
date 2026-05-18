//! Lexer for the Rust-flavored surface syntax.
//!
//! Produces a flat list of `Token { kind, span }` from a `&str`. The lexer
//! handles whitespace, line comments (`// ...`), block comments (`/* ... */`,
//! nestable), integer literals, string literals (with escapes), identifiers,
//! keywords, and operators.
//!
//! Tokens carry byte spans into the original source. Line/column for error
//! messages is computed on demand from the source string — we don't track
//! it here.

use core::fmt;

// =============================================================================
// Tokens
// =============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenKind {
    // ---- Literals ----
    Int(i64),
    Str(String),
    Ident(String),

    // ---- Keywords ----
    Def,
    Local,
    Fn,
    Let,
    Struct,
    Enum,
    Match,
    If,
    Else,
    True,
    False,
    Extern,

    // ---- Punctuation ----
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Semi,
    Colon,
    ColonColon,
    Dot,
    Underscore,

    // ---- Operators ----
    Arrow,    // ->
    FatArrow, // =>
    Eq,       // =
    EqEq,     // ==
    NotEq,    // !=
    Lt,
    LtEq,
    Gt,
    GtEq,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Bang,     // !
    AmpAmp,   // &&
    PipePipe, // ||
    Pipe,     // | (closure delimiter)
    Question, // ?

    // ---- End ----
    Eof,
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenKind::Int(n) => write!(f, "{}", n),
            TokenKind::Str(s) => write!(f, "\"{}\"", s),
            TokenKind::Ident(s) => write!(f, "{}", s),
            TokenKind::Def => f.write_str("def"),
            TokenKind::Local => f.write_str("local"),
            TokenKind::Fn => f.write_str("fn"),
            TokenKind::Let => f.write_str("let"),
            TokenKind::Struct => f.write_str("struct"),
            TokenKind::Enum => f.write_str("enum"),
            TokenKind::Match => f.write_str("match"),
            TokenKind::If => f.write_str("if"),
            TokenKind::Else => f.write_str("else"),
            TokenKind::True => f.write_str("true"),
            TokenKind::False => f.write_str("false"),
            TokenKind::Extern => f.write_str("extern"),
            TokenKind::LParen => f.write_str("("),
            TokenKind::RParen => f.write_str(")"),
            TokenKind::LBrace => f.write_str("{"),
            TokenKind::RBrace => f.write_str("}"),
            TokenKind::LBracket => f.write_str("["),
            TokenKind::RBracket => f.write_str("]"),
            TokenKind::Comma => f.write_str(","),
            TokenKind::Semi => f.write_str(";"),
            TokenKind::Colon => f.write_str(":"),
            TokenKind::ColonColon => f.write_str("::"),
            TokenKind::Dot => f.write_str("."),
            TokenKind::Underscore => f.write_str("_"),
            TokenKind::Arrow => f.write_str("->"),
            TokenKind::FatArrow => f.write_str("=>"),
            TokenKind::Eq => f.write_str("="),
            TokenKind::EqEq => f.write_str("=="),
            TokenKind::NotEq => f.write_str("!="),
            TokenKind::Lt => f.write_str("<"),
            TokenKind::LtEq => f.write_str("<="),
            TokenKind::Gt => f.write_str(">"),
            TokenKind::GtEq => f.write_str(">="),
            TokenKind::Plus => f.write_str("+"),
            TokenKind::Minus => f.write_str("-"),
            TokenKind::Star => f.write_str("*"),
            TokenKind::Slash => f.write_str("/"),
            TokenKind::Percent => f.write_str("%"),
            TokenKind::Bang => f.write_str("!"),
            TokenKind::AmpAmp => f.write_str("&&"),
            TokenKind::PipePipe => f.write_str("||"),
            TokenKind::Pipe => f.write_str("|"),
            TokenKind::Question => f.write_str("?"),
            TokenKind::Eof => f.write_str("<eof>"),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

// =============================================================================
// Errors
// =============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LexError {
    UnknownChar { ch: char, span: Span },
    UnterminatedString { span: Span },
    UnterminatedBlockComment { span: Span },
    BadEscape { ch: char, span: Span },
    IntTooLarge { text: String, span: Span },
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LexError::UnknownChar { ch, .. } => write!(f, "unknown character {:?}", ch),
            LexError::UnterminatedString { .. } => f.write_str("unterminated string literal"),
            LexError::UnterminatedBlockComment { .. } => {
                f.write_str("unterminated /* ... */ block comment")
            }
            LexError::BadEscape { ch, .. } => {
                write!(f, "unknown escape sequence \\{}", ch)
            }
            LexError::IntTooLarge { text, .. } => {
                write!(f, "integer literal {} does not fit in i64", text)
            }
        }
    }
}

impl std::error::Error for LexError {}

impl LexError {
    pub fn span(&self) -> Span {
        match self {
            LexError::UnknownChar { span, .. }
            | LexError::UnterminatedString { span }
            | LexError::UnterminatedBlockComment { span }
            | LexError::BadEscape { span, .. }
            | LexError::IntTooLarge { span, .. } => *span,
        }
    }
}

// =============================================================================
// Lexer
// =============================================================================

/// Tokenise the entire input. Returns all tokens including a final `Eof`,
/// or the first lex error encountered.
pub fn lex(src: &str) -> Result<Vec<Token>, LexError> {
    let mut lx = Lexer::new(src);
    let mut out = Vec::new();
    loop {
        let tok = lx.next_token()?;
        let is_eof = matches!(tok.kind, TokenKind::Eof);
        out.push(tok);
        if is_eof {
            return Ok(out);
        }
    }
}

struct Lexer<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> Lexer<'a> {
    fn new(src: &'a str) -> Self {
        Lexer {
            src: src.as_bytes(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn peek_at(&self, offset: usize) -> Option<u8> {
        self.src.get(self.pos + offset).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let b = self.peek()?;
        self.pos += 1;
        Some(b)
    }

    fn span_from(&self, start: usize) -> Span {
        Span {
            start,
            end: self.pos,
        }
    }

    fn skip_trivia(&mut self) -> Result<(), LexError> {
        loop {
            match self.peek() {
                Some(b' ') | Some(b'\t') | Some(b'\n') | Some(b'\r') => {
                    self.pos += 1;
                }
                Some(b'/') if self.peek_at(1) == Some(b'/') => {
                    // Line comment.
                    self.pos += 2;
                    while let Some(b) = self.peek() {
                        self.pos += 1;
                        if b == b'\n' {
                            break;
                        }
                    }
                }
                Some(b'/') if self.peek_at(1) == Some(b'*') => {
                    // Nestable block comment.
                    let start = self.pos;
                    self.pos += 2;
                    let mut depth: u32 = 1;
                    while depth > 0 {
                        match (self.peek(), self.peek_at(1)) {
                            (Some(b'/'), Some(b'*')) => {
                                self.pos += 2;
                                depth += 1;
                            }
                            (Some(b'*'), Some(b'/')) => {
                                self.pos += 2;
                                depth -= 1;
                            }
                            (Some(_), _) => {
                                self.pos += 1;
                            }
                            (None, _) => {
                                return Err(LexError::UnterminatedBlockComment {
                                    span: Span {
                                        start,
                                        end: self.pos,
                                    },
                                });
                            }
                        }
                    }
                }
                _ => return Ok(()),
            }
        }
    }

    fn next_token(&mut self) -> Result<Token, LexError> {
        self.skip_trivia()?;
        let start = self.pos;
        let Some(b) = self.peek() else {
            return Ok(Token {
                kind: TokenKind::Eof,
                span: self.span_from(start),
            });
        };

        // ---- Identifiers and keywords ----
        if is_ident_start(b) {
            return Ok(self.lex_ident_or_keyword(start));
        }

        // ---- Integer literals ----
        if b.is_ascii_digit() {
            return self.lex_int(start);
        }

        // ---- String literals ----
        if b == b'"' {
            return self.lex_string(start);
        }

        // ---- Punctuation / operators ----
        self.bump();
        let kind = match b {
            b'(' => TokenKind::LParen,
            b')' => TokenKind::RParen,
            b'{' => TokenKind::LBrace,
            b'}' => TokenKind::RBrace,
            b'[' => TokenKind::LBracket,
            b']' => TokenKind::RBracket,
            b',' => TokenKind::Comma,
            b';' => TokenKind::Semi,
            b'.' => TokenKind::Dot,
            b'?' => TokenKind::Question,
            b'+' => TokenKind::Plus,
            b'*' => TokenKind::Star,
            b'/' => TokenKind::Slash,
            b'%' => TokenKind::Percent,
            b':' => {
                if self.peek() == Some(b':') {
                    self.bump();
                    TokenKind::ColonColon
                } else {
                    TokenKind::Colon
                }
            }
            b'-' => {
                if self.peek() == Some(b'>') {
                    self.bump();
                    TokenKind::Arrow
                } else {
                    TokenKind::Minus
                }
            }
            b'=' => match self.peek() {
                Some(b'=') => {
                    self.bump();
                    TokenKind::EqEq
                }
                Some(b'>') => {
                    self.bump();
                    TokenKind::FatArrow
                }
                _ => TokenKind::Eq,
            },
            b'!' => {
                if self.peek() == Some(b'=') {
                    self.bump();
                    TokenKind::NotEq
                } else {
                    TokenKind::Bang
                }
            }
            b'<' => {
                if self.peek() == Some(b'=') {
                    self.bump();
                    TokenKind::LtEq
                } else {
                    TokenKind::Lt
                }
            }
            b'>' => {
                if self.peek() == Some(b'=') {
                    self.bump();
                    TokenKind::GtEq
                } else {
                    TokenKind::Gt
                }
            }
            b'&' => {
                if self.peek() == Some(b'&') {
                    self.bump();
                    TokenKind::AmpAmp
                } else {
                    return Err(LexError::UnknownChar {
                        ch: '&',
                        span: self.span_from(start),
                    });
                }
            }
            b'|' => {
                if self.peek() == Some(b'|') {
                    self.bump();
                    TokenKind::PipePipe
                } else {
                    TokenKind::Pipe
                }
            }
            other => {
                // Recover the character for the error message. ASCII path is
                // common; for non-ASCII we re-decode at the start byte.
                let ch = if other < 0x80 {
                    other as char
                } else {
                    decode_utf8_at(self.src, start).unwrap_or('\u{FFFD}')
                };
                return Err(LexError::UnknownChar {
                    ch,
                    span: self.span_from(start),
                });
            }
        };
        Ok(Token {
            kind,
            span: self.span_from(start),
        })
    }

    fn lex_ident_or_keyword(&mut self, start: usize) -> Token {
        while let Some(b) = self.peek() {
            if is_ident_continue(b) {
                self.pos += 1;
            } else {
                break;
            }
        }
        let text = core::str::from_utf8(&self.src[start..self.pos])
            .expect("ident is ASCII-only by construction");
        let kind = match text {
            "_" => TokenKind::Underscore,
            "def" => TokenKind::Def,
            "local" => TokenKind::Local,
            "fn" => TokenKind::Fn,
            "let" => TokenKind::Let,
            "struct" => TokenKind::Struct,
            "enum" => TokenKind::Enum,
            "match" => TokenKind::Match,
            "if" => TokenKind::If,
            "else" => TokenKind::Else,
            "true" => TokenKind::True,
            "false" => TokenKind::False,
            "extern" => TokenKind::Extern,
            _ => TokenKind::Ident(text.to_owned()),
        };
        Token {
            kind,
            span: self.span_from(start),
        }
    }

    fn lex_int(&mut self, start: usize) -> Result<Token, LexError> {
        while let Some(b) = self.peek() {
            if b.is_ascii_digit() {
                self.pos += 1;
            } else {
                break;
            }
        }
        let text = core::str::from_utf8(&self.src[start..self.pos])
            .expect("digit run is ASCII");
        let n: i64 = text.parse().map_err(|_| LexError::IntTooLarge {
            text: text.to_owned(),
            span: self.span_from(start),
        })?;
        Ok(Token {
            kind: TokenKind::Int(n),
            span: self.span_from(start),
        })
    }

    fn lex_string(&mut self, start: usize) -> Result<Token, LexError> {
        // We've not yet consumed the opening quote.
        self.pos += 1;
        let mut out = String::new();
        loop {
            let Some(b) = self.peek() else {
                return Err(LexError::UnterminatedString {
                    span: Span {
                        start,
                        end: self.pos,
                    },
                });
            };
            match b {
                b'"' => {
                    self.pos += 1;
                    return Ok(Token {
                        kind: TokenKind::Str(out),
                        span: self.span_from(start),
                    });
                }
                b'\\' => {
                    let esc_start = self.pos;
                    self.pos += 1;
                    let Some(e) = self.peek() else {
                        return Err(LexError::UnterminatedString {
                            span: Span {
                                start,
                                end: self.pos,
                            },
                        });
                    };
                    self.pos += 1;
                    let c = match e {
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'\\' => '\\',
                        b'"' => '"',
                        b'0' => '\0',
                        b'e' => '\x1b', // ESC, for ANSI escape sequences
                        other => {
                            return Err(LexError::BadEscape {
                                ch: other as char,
                                span: Span {
                                    start: esc_start,
                                    end: self.pos,
                                },
                            });
                        }
                    };
                    out.push(c);
                }
                _ => {
                    // Decode one UTF-8 codepoint starting here.
                    let (ch, len) = decode_utf8_at_len(self.src, self.pos)
                        .ok_or(LexError::UnterminatedString {
                            span: Span {
                                start,
                                end: self.pos,
                            },
                        })?;
                    out.push(ch);
                    self.pos += len;
                }
            }
        }
    }
}

fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

fn is_ident_continue(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Decode a single UTF-8 codepoint starting at `pos`. Returns `(char, byte_len)`.
fn decode_utf8_at_len(bytes: &[u8], pos: usize) -> Option<(char, usize)> {
    let s = core::str::from_utf8(bytes.get(pos..)?).ok()?;
    let ch = s.chars().next()?;
    Some((ch, ch.len_utf8()))
}

fn decode_utf8_at(bytes: &[u8], pos: usize) -> Option<char> {
    decode_utf8_at_len(bytes, pos).map(|(c, _)| c)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(src: &str) -> Vec<TokenKind> {
        lex(src).unwrap().into_iter().map(|t| t.kind).collect()
    }

    #[test]
    fn empty_input_is_just_eof() {
        assert_eq!(kinds(""), vec![TokenKind::Eof]);
    }

    #[test]
    fn whitespace_and_comments_are_skipped() {
        let toks = kinds("   \n  // a line comment\n  /* block /* nested */ end */ 42 ");
        assert_eq!(toks, vec![TokenKind::Int(42), TokenKind::Eof]);
    }

    #[test]
    fn keywords_vs_identifiers() {
        let toks = kinds("def local fn xs let_ matches");
        assert_eq!(
            toks,
            vec![
                TokenKind::Def,
                TokenKind::Local,
                TokenKind::Fn,
                TokenKind::Ident("xs".to_owned()),
                TokenKind::Ident("let_".to_owned()),
                TokenKind::Ident("matches".to_owned()),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn underscore_is_its_own_token() {
        let toks = kinds("_ _x");
        assert_eq!(
            toks,
            vec![
                TokenKind::Underscore,
                TokenKind::Ident("_x".to_owned()),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn multi_char_operators() {
        let toks = kinds("-> => == != <= >= && || ::");
        assert_eq!(
            toks,
            vec![
                TokenKind::Arrow,
                TokenKind::FatArrow,
                TokenKind::EqEq,
                TokenKind::NotEq,
                TokenKind::LtEq,
                TokenKind::GtEq,
                TokenKind::AmpAmp,
                TokenKind::PipePipe,
                TokenKind::ColonColon,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn single_pipe_for_closures() {
        let toks = kinds("|x|");
        assert_eq!(
            toks,
            vec![
                TokenKind::Pipe,
                TokenKind::Ident("x".to_owned()),
                TokenKind::Pipe,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn arrow_does_not_eat_following_gt() {
        // `-> >` lexes as Arrow then Gt, not Arrow then... something weird.
        let toks = kinds("-> >");
        assert_eq!(toks, vec![TokenKind::Arrow, TokenKind::Gt, TokenKind::Eof]);
    }

    #[test]
    fn integer_literal_and_max_i64() {
        assert_eq!(kinds("0"), vec![TokenKind::Int(0), TokenKind::Eof]);
        assert_eq!(
            kinds("9223372036854775807"),
            vec![TokenKind::Int(i64::MAX), TokenKind::Eof]
        );
    }

    #[test]
    fn integer_literal_too_large_is_an_error() {
        let err = lex("99999999999999999999999").unwrap_err();
        match err {
            LexError::IntTooLarge { text, .. } => {
                assert_eq!(text, "99999999999999999999999");
            }
            other => panic!("expected IntTooLarge, got {:?}", other),
        }
    }

    #[test]
    fn string_literal_with_escapes() {
        let toks = kinds(r#" "hello\tworld\n" "#);
        assert_eq!(
            toks,
            vec![
                TokenKind::Str("hello\tworld\n".to_owned()),
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn unterminated_string_is_an_error() {
        assert!(matches!(
            lex(r#" "no closing quote "#).unwrap_err(),
            LexError::UnterminatedString { .. }
        ));
    }

    #[test]
    fn unknown_escape_is_an_error() {
        let err = lex(r#" "\q" "#).unwrap_err();
        assert!(matches!(err, LexError::BadEscape { ch: 'q', .. }));
    }

    #[test]
    fn unknown_char_is_an_error() {
        let err = lex("@").unwrap_err();
        assert!(matches!(err, LexError::UnknownChar { ch: '@', .. }));
    }

    #[test]
    fn unterminated_block_comment_is_an_error() {
        let err = lex("/* never closed").unwrap_err();
        assert!(matches!(err, LexError::UnterminatedBlockComment { .. }));
    }

    // ---- Spans ----

    #[test]
    fn spans_are_byte_offsets_into_source() {
        let toks = lex("def x").unwrap();
        assert_eq!(toks[0].kind, TokenKind::Def);
        assert_eq!(toks[0].span, Span { start: 0, end: 3 });
        assert_eq!(toks[1].kind, TokenKind::Ident("x".to_owned()));
        assert_eq!(toks[1].span, Span { start: 4, end: 5 });
        assert_eq!(toks[2].kind, TokenKind::Eof);
        assert_eq!(toks[2].span, Span { start: 5, end: 5 });
    }

    #[test]
    fn lexes_full_def_double() {
        let toks = kinds("def double(x: Int) -> Int = x * 2");
        assert_eq!(
            toks,
            vec![
                TokenKind::Def,
                TokenKind::Ident("double".to_owned()),
                TokenKind::LParen,
                TokenKind::Ident("x".to_owned()),
                TokenKind::Colon,
                TokenKind::Ident("Int".to_owned()),
                TokenKind::RParen,
                TokenKind::Arrow,
                TokenKind::Ident("Int".to_owned()),
                TokenKind::Eq,
                TokenKind::Ident("x".to_owned()),
                TokenKind::Star,
                TokenKind::Int(2),
                TokenKind::Eof,
            ]
        );
    }
}
