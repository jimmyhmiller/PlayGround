//! Lexer for gc-rust. Produces a flat token stream with source spans.
//!
//! See `docs/language.md` §1 for the lexical grammar. The lexer is
//! whitespace-insensitive (no significant indentation), supports line and
//! nested block comments, and recognizes typed numeric literals.

use std::fmt;

/// A byte range into the source, `[start, end)`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Span { start: start as u32, end: end as u32 }
    }
    pub fn to(self, other: Span) -> Span {
        Span { start: self.start, end: other.end }
    }
}

impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
}

/// Integer/float literal suffix recording the requested concrete type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumSuffix {
    None,
    I8, I16, I32, I64,
    U8, U16, U32, U64,
    F32, F64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TokKind {
    // literals
    Int(u64, NumSuffix),
    Float(f64, NumSuffix),
    Str(String),
    Char(char),
    // identifiers + keywords
    Ident(String),
    Keyword(Kw),
    // punctuation / operators
    LParen, RParen, LBrace, RBrace, LBracket, RBracket,
    Comma, Semi, Colon, ColonColon, Arrow, FatArrow, Dot, DotDot, DotDotEq,
    Question, At,
    Plus, Minus, Star, Slash, Percent,
    Eq, EqEq, Ne, Lt, Le, Gt, Ge,
    AndAnd, OrOr, Not,
    Amp, Pipe, Caret, Shl, Shr,
    PlusEq, MinusEq, StarEq, SlashEq, PercentEq,
    Eof,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kw {
    Fn, Let, Mut, Struct, Enum, Impl, Trait, For, In, If, Else, Match,
    While, Loop, Return, Break, Continue, True, False, As, Where, Pub, Mod,
    Use, Type, Const, Static, SelfValue, SelfType, Value,
}

impl Kw {
    fn from_ident(s: &str) -> Option<Kw> {
        Some(match s {
            "fn" => Kw::Fn, "let" => Kw::Let, "mut" => Kw::Mut,
            "struct" => Kw::Struct, "enum" => Kw::Enum, "impl" => Kw::Impl,
            "trait" => Kw::Trait, "for" => Kw::For, "in" => Kw::In,
            "if" => Kw::If, "else" => Kw::Else, "match" => Kw::Match,
            "while" => Kw::While, "loop" => Kw::Loop, "return" => Kw::Return,
            "break" => Kw::Break, "continue" => Kw::Continue,
            "true" => Kw::True, "false" => Kw::False, "as" => Kw::As,
            "where" => Kw::Where, "pub" => Kw::Pub, "mod" => Kw::Mod,
            "use" => Kw::Use, "type" => Kw::Type, "const" => Kw::Const,
            "static" => Kw::Static, "self" => Kw::SelfValue,
            "Self" => Kw::SelfType, "value" => Kw::Value,
            _ => return None,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Token {
    pub kind: TokKind,
    pub span: Span,
}

#[derive(Debug)]
pub struct LexError {
    pub msg: String,
    pub span: Span,
}

pub fn lex(src: &str) -> Result<Vec<Token>, LexError> {
    Lexer::new(src).run()
}

struct Lexer<'a> {
    src: &'a [u8],
    text: &'a str,
    pos: usize,
    toks: Vec<Token>,
}

impl<'a> Lexer<'a> {
    fn new(text: &'a str) -> Self {
        Lexer { src: text.as_bytes(), text, pos: 0, toks: Vec::new() }
    }

    fn peek(&self) -> u8 {
        if self.pos < self.src.len() { self.src[self.pos] } else { 0 }
    }
    fn peek2(&self) -> u8 {
        if self.pos + 1 < self.src.len() { self.src[self.pos + 1] } else { 0 }
    }
    fn bump(&mut self) -> u8 {
        let c = self.peek();
        self.pos += 1;
        c
    }

    fn err(&self, msg: impl Into<String>, start: usize) -> LexError {
        LexError { msg: msg.into(), span: Span::new(start, self.pos) }
    }

    fn run(mut self) -> Result<Vec<Token>, LexError> {
        loop {
            self.skip_trivia()?;
            let start = self.pos;
            if self.pos >= self.src.len() {
                self.push(TokKind::Eof, start);
                return Ok(self.toks);
            }
            let c = self.peek();
            match c {
                b'0'..=b'9' => self.lex_number(start)?,
                b'"' => self.lex_string(start)?,
                b'\'' => self.lex_char(start)?,
                c if is_ident_start(c) => self.lex_ident(start),
                _ => self.lex_punct(start)?,
            }
        }
    }

    fn push(&mut self, kind: TokKind, start: usize) {
        self.toks.push(Token { kind, span: Span::new(start, self.pos) });
    }

    fn skip_trivia(&mut self) -> Result<(), LexError> {
        loop {
            match self.peek() {
                b' ' | b'\t' | b'\r' | b'\n' => { self.pos += 1; }
                b'/' if self.peek2() == b'/' => {
                    while self.pos < self.src.len() && self.peek() != b'\n' {
                        self.pos += 1;
                    }
                }
                b'/' if self.peek2() == b'*' => {
                    let start = self.pos;
                    self.pos += 2;
                    let mut depth = 1;
                    while depth > 0 {
                        if self.pos >= self.src.len() {
                            return Err(self.err("unterminated block comment", start));
                        }
                        if self.peek() == b'/' && self.peek2() == b'*' {
                            depth += 1; self.pos += 2;
                        } else if self.peek() == b'*' && self.peek2() == b'/' {
                            depth -= 1; self.pos += 2;
                        } else {
                            self.pos += 1;
                        }
                    }
                }
                _ => return Ok(()),
            }
        }
    }

    fn lex_ident(&mut self, start: usize) {
        while is_ident_continue(self.peek()) {
            self.pos += 1;
        }
        let s = &self.text[start..self.pos];
        match Kw::from_ident(s) {
            Some(kw) => self.push(TokKind::Keyword(kw), start),
            None => self.push(TokKind::Ident(s.to_string()), start),
        }
    }

    fn lex_number(&mut self, start: usize) -> Result<(), LexError> {
        // Radix prefixes.
        if self.peek() == b'0' && matches!(self.peek2(), b'x' | b'b' | b'o') {
            let radix = match self.peek2() { b'x' => 16, b'b' => 2, _ => 8 };
            self.pos += 2;
            let digits_start = self.pos;
            while is_radix_digit(self.peek(), radix) || self.peek() == b'_' {
                self.pos += 1;
            }
            let raw: String =
                self.text[digits_start..self.pos].chars().filter(|c| *c != '_').collect();
            if raw.is_empty() {
                return Err(self.err("missing digits after radix prefix", start));
            }
            let v = u64::from_str_radix(&raw, radix)
                .map_err(|_| self.err("integer literal out of range", start))?;
            let suffix = self.lex_num_suffix();
            self.push(TokKind::Int(v, suffix), start);
            return Ok(());
        }

        // Decimal integer or float.
        while self.peek().is_ascii_digit() || self.peek() == b'_' {
            self.pos += 1;
        }
        let mut is_float = false;
        // Fractional part: a dot NOT followed by another dot (range) or an
        // ident-start (method/field access like `1.foo`).
        if self.peek() == b'.' && self.peek2().is_ascii_digit() {
            is_float = true;
            self.pos += 1;
            while self.peek().is_ascii_digit() || self.peek() == b'_' {
                self.pos += 1;
            }
        }
        // Exponent.
        if matches!(self.peek(), b'e' | b'E') {
            let save = self.pos;
            self.pos += 1;
            if matches!(self.peek(), b'+' | b'-') {
                self.pos += 1;
            }
            if self.peek().is_ascii_digit() {
                is_float = true;
                while self.peek().is_ascii_digit() || self.peek() == b'_' {
                    self.pos += 1;
                }
            } else {
                // Not an exponent (e.g. a suffix starting with e); back off.
                self.pos = save;
            }
        }

        let body: String =
            self.text[start..self.pos].chars().filter(|c| *c != '_').collect();
        let suffix = self.lex_num_suffix();
        let suffix_is_float = matches!(suffix, NumSuffix::F32 | NumSuffix::F64);
        if is_float || suffix_is_float {
            let v: f64 = body
                .parse()
                .map_err(|_| self.err("invalid float literal", start))?;
            self.push(TokKind::Float(v, suffix), start);
        } else {
            let v: u64 = body
                .parse()
                .map_err(|_| self.err("integer literal out of range", start))?;
            self.push(TokKind::Int(v, suffix), start);
        }
        Ok(())
    }

    fn lex_num_suffix(&mut self) -> NumSuffix {
        let start = self.pos;
        if !is_ident_start(self.peek()) {
            return NumSuffix::None;
        }
        while is_ident_continue(self.peek()) {
            self.pos += 1;
        }
        match &self.text[start..self.pos] {
            "i8" => NumSuffix::I8, "i16" => NumSuffix::I16,
            "i32" => NumSuffix::I32, "i64" => NumSuffix::I64,
            "u8" => NumSuffix::U8, "u16" => NumSuffix::U16,
            "u32" => NumSuffix::U32, "u64" => NumSuffix::U64,
            "f32" => NumSuffix::F32, "f64" => NumSuffix::F64,
            // Unknown suffix: rewind so it lexes as a separate token (will be
            // a parse error in context, which is the right place to report it).
            _ => { self.pos = start; NumSuffix::None }
        }
    }

    fn lex_string(&mut self, start: usize) -> Result<(), LexError> {
        self.pos += 1; // opening quote
        let mut s = String::new();
        loop {
            if self.pos >= self.src.len() {
                return Err(self.err("unterminated string literal", start));
            }
            let c = self.bump();
            match c {
                b'"' => break,
                b'\\' => s.push(self.lex_escape(start)?),
                _ => {
                    // Reconstruct the UTF-8 char from the source slice.
                    let cstart = self.pos - 1;
                    let ch = self.text[cstart..].chars().next().unwrap();
                    // bump() already advanced one byte; advance the rest.
                    self.pos = cstart + ch.len_utf8();
                    s.push(ch);
                }
            }
        }
        self.push(TokKind::Str(s), start);
        Ok(())
    }

    fn lex_char(&mut self, start: usize) -> Result<(), LexError> {
        self.pos += 1; // opening quote
        if self.pos >= self.src.len() {
            return Err(self.err("unterminated char literal", start));
        }
        let ch = if self.peek() == b'\\' {
            self.pos += 1;
            self.lex_escape(start)?
        } else {
            let cstart = self.pos;
            let ch = self.text[cstart..].chars().next().unwrap();
            self.pos = cstart + ch.len_utf8();
            ch
        };
        if self.peek() != b'\'' {
            return Err(self.err("unterminated char literal", start));
        }
        self.pos += 1; // closing quote
        self.push(TokKind::Char(ch), start);
        Ok(())
    }

    fn lex_escape(&mut self, start: usize) -> Result<char, LexError> {
        let e = self.bump();
        Ok(match e {
            b'n' => '\n', b't' => '\t', b'r' => '\r', b'0' => '\0',
            b'\\' => '\\', b'\'' => '\'', b'"' => '"',
            b'u' => {
                if self.bump() != b'{' {
                    return Err(self.err("expected `{` in unicode escape", start));
                }
                let hs = self.pos;
                while self.peek() != b'}' && self.pos < self.src.len() {
                    self.pos += 1;
                }
                let code = u32::from_str_radix(&self.text[hs..self.pos], 16)
                    .map_err(|_| self.err("invalid unicode escape", start))?;
                self.pos += 1; // closing brace
                char::from_u32(code)
                    .ok_or_else(|| self.err("invalid unicode scalar", start))?
            }
            _ => return Err(self.err("unknown escape", start)),
        })
    }

    fn lex_punct(&mut self, start: usize) -> Result<(), LexError> {
        let c = self.bump();
        let two = |l: &mut Self, k: TokKind| { l.pos += 1; k };
        let kind = match c {
            b'(' => TokKind::LParen, b')' => TokKind::RParen,
            b'{' => TokKind::LBrace, b'}' => TokKind::RBrace,
            b'[' => TokKind::LBracket, b']' => TokKind::RBracket,
            b',' => TokKind::Comma, b';' => TokKind::Semi,
            b'@' => TokKind::At,
            b':' => if self.peek() == b':' { two(self, TokKind::ColonColon) } else { TokKind::Colon },
            b'-' => match self.peek() {
                b'>' => two(self, TokKind::Arrow),
                b'=' => two(self, TokKind::MinusEq),
                _ => TokKind::Minus,
            },
            b'+' => if self.peek() == b'=' { two(self, TokKind::PlusEq) } else { TokKind::Plus },
            b'*' => if self.peek() == b'=' { two(self, TokKind::StarEq) } else { TokKind::Star },
            b'/' => if self.peek() == b'=' { two(self, TokKind::SlashEq) } else { TokKind::Slash },
            b'%' => if self.peek() == b'=' { two(self, TokKind::PercentEq) } else { TokKind::Percent },
            b'=' => match self.peek() {
                b'=' => two(self, TokKind::EqEq),
                b'>' => two(self, TokKind::FatArrow),
                _ => TokKind::Eq,
            },
            b'!' => if self.peek() == b'=' { two(self, TokKind::Ne) } else { TokKind::Not },
            b'<' => match self.peek() {
                b'=' => two(self, TokKind::Le),
                b'<' => two(self, TokKind::Shl),
                _ => TokKind::Lt,
            },
            b'>' => match self.peek() {
                b'=' => two(self, TokKind::Ge),
                b'>' => two(self, TokKind::Shr),
                _ => TokKind::Gt,
            },
            b'&' => if self.peek() == b'&' { two(self, TokKind::AndAnd) } else { TokKind::Amp },
            b'|' => if self.peek() == b'|' { two(self, TokKind::OrOr) } else { TokKind::Pipe },
            b'^' => TokKind::Caret,
            b'?' => TokKind::Question,
            b'.' => match self.peek() {
                b'.' => {
                    self.pos += 1;
                    if self.peek() == b'=' { self.pos += 1; TokKind::DotDotEq } else { TokKind::DotDot }
                }
                _ => TokKind::Dot,
            },
            _ => return Err(self.err(format!("unexpected character {:?}", c as char), start)),
        };
        self.push(kind, start);
        Ok(())
    }
}

fn is_ident_start(c: u8) -> bool {
    c == b'_' || c.is_ascii_alphabetic()
}
fn is_ident_continue(c: u8) -> bool {
    c == b'_' || c.is_ascii_alphanumeric()
}
fn is_radix_digit(c: u8, radix: u32) -> bool {
    (c as char).is_digit(radix)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(src: &str) -> Vec<TokKind> {
        lex(src).unwrap().into_iter().map(|t| t.kind).collect()
    }

    #[test]
    fn keywords_and_idents() {
        let k = kinds("fn foo let mut value struct");
        assert_eq!(k[0], TokKind::Keyword(Kw::Fn));
        assert_eq!(k[1], TokKind::Ident("foo".into()));
        assert_eq!(k[2], TokKind::Keyword(Kw::Let));
        assert_eq!(k[3], TokKind::Keyword(Kw::Mut));
        assert_eq!(k[4], TokKind::Keyword(Kw::Value));
        assert_eq!(k[5], TokKind::Keyword(Kw::Struct));
    }

    #[test]
    fn numbers() {
        assert_eq!(kinds("42")[0], TokKind::Int(42, NumSuffix::None));
        assert_eq!(kinds("255u8")[0], TokKind::Int(255, NumSuffix::U8));
        assert_eq!(kinds("0xFF")[0], TokKind::Int(255, NumSuffix::None));
        assert_eq!(kinds("0b1010")[0], TokKind::Int(10, NumSuffix::None));
        assert_eq!(kinds("0o17")[0], TokKind::Int(15, NumSuffix::None));
        assert_eq!(kinds("1_000")[0], TokKind::Int(1000, NumSuffix::None));
        assert_eq!(kinds("3.14")[0], TokKind::Float(3.14, NumSuffix::None));
        assert_eq!(kinds("2.5f32")[0], TokKind::Float(2.5, NumSuffix::F32));
        assert_eq!(kinds("1.0e-9")[0], TokKind::Float(1.0e-9, NumSuffix::None));
    }

    #[test]
    fn dot_vs_float_vs_range() {
        // `1.foo` is int, dot, ident — not a float.
        let k = kinds("1.foo");
        assert_eq!(k[0], TokKind::Int(1, NumSuffix::None));
        assert_eq!(k[1], TokKind::Dot);
        assert_eq!(k[2], TokKind::Ident("foo".into()));
        // `0..5` is int, dotdot, int.
        let k = kinds("0..5");
        assert_eq!(k[0], TokKind::Int(0, NumSuffix::None));
        assert_eq!(k[1], TokKind::DotDot);
        assert_eq!(k[2], TokKind::Int(5, NumSuffix::None));
    }

    #[test]
    fn operators() {
        let k = kinds("-> => :: == != <= >= && || << >> += ..=");
        use TokKind::*;
        assert_eq!(
            k[..12],
            [Arrow, FatArrow, ColonColon, EqEq, Ne, Le, Ge, AndAnd, OrOr, Shl, Shr, PlusEq]
        );
        assert_eq!(k[12], DotDotEq);
    }

    #[test]
    fn strings_and_chars() {
        assert_eq!(kinds(r#""hi\n""#)[0], TokKind::Str("hi\n".into()));
        assert_eq!(kinds("'a'")[0], TokKind::Char('a'));
        assert_eq!(kinds(r"'\n'")[0], TokKind::Char('\n'));
        assert_eq!(kinds(r"'\u{41}'")[0], TokKind::Char('A'));
    }

    #[test]
    fn comments() {
        let k = kinds("1 // line\n /* a /* nested */ b */ 2");
        assert_eq!(k[0], TokKind::Int(1, NumSuffix::None));
        assert_eq!(k[1], TokKind::Int(2, NumSuffix::None));
    }

    #[test]
    fn unterminated_block_comment_errors() {
        assert!(lex("/* oops").is_err());
    }
}
