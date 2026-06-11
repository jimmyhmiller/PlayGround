//! Lexer. Newlines are significant (statement terminators) per spec §3.1:
//! a Newline token is suppressed when the previous token can't end a statement
//! (operator, comma, open bracket, `|>`, `=>`, …) or when the next line starts
//! with a continuation token (`|>`, `.`, `else`, closing bracket, …).

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    // literals
    Int(i64),
    Float(f64),
    /// String literal split into chunks for interpolation.
    Str(Vec<StrPart>),
    Ident(String),
    TypeName(String),
    // keywords
    Let,
    Mut,
    Fn,
    Match,
    If,
    Else,
    While,
    For,
    In,
    Type,
    Import,
    Export,
    Extern,
    Return,
    Break,
    Continue,
    True,
    False,
    And,
    Or,
    Not,
    As,
    // operators / punctuation
    Pipe,      // |>
    FatArrow,  // =>
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    StarStar,
    EqEq,
    NotEq,
    Lt,
    Le,
    Gt,
    Ge,
    Assign,    // =
    PlusEq,
    MinusEq,
    StarEq,
    SlashEq,
    PercentEq,
    DotDot,
    DotDotEq,
    Dot,
    Question,
    At,
    Underscore,
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
    Colon,
    VBar, // |  (variant separator / or-pattern)
    Hash, // #  (attributes: #[test])
    Newline,
    Eof,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StrPart {
    Lit(String),
    /// Tokens of an embedded `{expr}` (no trailing Eof/Newline tokens).
    Interp(Vec<Token>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub tok: Tok,
    pub line: u32,
}

impl fmt::Display for Tok {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub struct Lexer<'a> {
    src: &'a [u8],
    pos: usize,
    line: u32,
}

pub fn lex(src: &str) -> Result<Vec<Token>, String> {
    let mut lx = Lexer { src: src.as_bytes(), pos: 0, line: 1 };
    let mut raw = Vec::new();
    loop {
        if let Some(t) = lx.next_token()? {
            let is_eof = t.tok == Tok::Eof;
            raw.push(t);
            if is_eof {
                break;
            }
        }
    }
    Ok(filter_newlines(raw))
}

/// Drop Newline tokens that are clearly continuations.
fn filter_newlines(raw: Vec<Token>) -> Vec<Token> {
    let mut out: Vec<Token> = Vec::with_capacity(raw.len());
    let mut i = 0;
    while i < raw.len() {
        let t = raw[i].clone();
        if t.tok == Tok::Newline {
            // collapse runs of newlines
            let mut j = i;
            while j + 1 < raw.len() && raw[j + 1].tok == Tok::Newline {
                j += 1;
            }
            let prev = out.last().map(|t| &t.tok);
            let next = raw.get(j + 1).map(|t| &t.tok);
            let suppress_by_prev = match prev {
                None => true,
                Some(p) => matches!(
                    p,
                    Tok::Pipe
                        | Tok::FatArrow
                        | Tok::Plus
                        | Tok::Minus
                        | Tok::Star
                        | Tok::Slash
                        | Tok::Percent
                        | Tok::StarStar
                        | Tok::EqEq
                        | Tok::NotEq
                        | Tok::Lt
                        | Tok::Le
                        | Tok::Gt
                        | Tok::Ge
                        | Tok::Assign
                        | Tok::PlusEq
                        | Tok::MinusEq
                        | Tok::StarEq
                        | Tok::SlashEq
                        | Tok::PercentEq
                        | Tok::DotDot
                        | Tok::DotDotEq
                        | Tok::Dot
                        | Tok::At
                        | Tok::LParen
                        | Tok::LBracket
                        | Tok::LBrace
                        | Tok::Comma
                        | Tok::Colon
                        | Tok::VBar
                        | Tok::And
                        | Tok::Or
                        | Tok::Not
                        | Tok::In
                        | Tok::If
                        | Tok::Else
                        | Tok::While
                        | Tok::For
                        | Tok::Match
                        | Tok::Return
                        | Tok::Let
                        | Tok::Mut
                        | Tok::Fn
                        | Tok::Type
                ),
            };
            let suppress_by_next = match next {
                None => true,
                Some(n) => matches!(
                    n,
                    // these can never start a statement, so a line beginning
                    // with one continues the previous line
                    Tok::Pipe | Tok::Dot | Tok::Else | Tok::VBar | Tok::And | Tok::Or | Tok::Eof
                ),
            };
            if !(suppress_by_prev || suppress_by_next) {
                out.push(t);
            }
            i = j + 1;
        } else {
            out.push(t);
            i += 1;
        }
    }
    out
}

impl<'a> Lexer<'a> {
    fn peek(&self) -> u8 {
        *self.src.get(self.pos).unwrap_or(&0)
    }

    fn peek2(&self) -> u8 {
        *self.src.get(self.pos + 1).unwrap_or(&0)
    }

    fn bump(&mut self) -> u8 {
        let c = self.peek();
        self.pos += 1;
        if c == b'\n' {
            self.line += 1;
        }
        c
    }

    fn next_token(&mut self) -> Result<Option<Token>, String> {
        // skip spaces/tabs/CR and comments
        loop {
            match self.peek() {
                b' ' | b'\t' | b'\r' => {
                    self.bump();
                }
                b'/' if self.peek2() == b'/' => {
                    while self.peek() != b'\n' && self.peek() != 0 {
                        self.bump();
                    }
                }
                _ => break,
            }
        }
        let line = self.line;
        let c = self.peek();
        if c == 0 {
            return Ok(Some(Token { tok: Tok::Eof, line }));
        }
        if c == b'\n' {
            self.bump();
            return Ok(Some(Token { tok: Tok::Newline, line }));
        }
        if c.is_ascii_digit() {
            return Ok(Some(self.number()?));
        }
        if c == b'"' {
            return Ok(Some(self.string()?));
        }
        if c.is_ascii_alphabetic() || c == b'_' {
            return Ok(Some(self.ident_or_kw()));
        }
        // operators
        self.bump();
        let t = match c {
            b'|' => match self.peek() {
                b'>' => {
                    self.bump();
                    Tok::Pipe
                }
                _ => Tok::VBar,
            },
            b'=' => match self.peek() {
                b'>' => {
                    self.bump();
                    Tok::FatArrow
                }
                b'=' => {
                    self.bump();
                    Tok::EqEq
                }
                _ => Tok::Assign,
            },
            b'+' => {
                if self.peek() == b'=' {
                    self.bump();
                    Tok::PlusEq
                } else {
                    Tok::Plus
                }
            }
            b'-' => {
                if self.peek() == b'=' {
                    self.bump();
                    Tok::MinusEq
                } else {
                    Tok::Minus
                }
            }
            b'*' => match self.peek() {
                b'*' => {
                    self.bump();
                    Tok::StarStar
                }
                b'=' => {
                    self.bump();
                    Tok::StarEq
                }
                _ => Tok::Star,
            },
            b'/' => {
                if self.peek() == b'=' {
                    self.bump();
                    Tok::SlashEq
                } else {
                    Tok::Slash
                }
            }
            b'%' => {
                if self.peek() == b'=' {
                    self.bump();
                    Tok::PercentEq
                } else {
                    Tok::Percent
                }
            }
            b'!' => {
                if self.peek() == b'=' {
                    self.bump();
                    Tok::NotEq
                } else {
                    return Err(format!("line {}: unexpected '!'", line));
                }
            }
            b'<' => {
                if self.peek() == b'=' {
                    self.bump();
                    Tok::Le
                } else {
                    Tok::Lt
                }
            }
            b'>' => {
                if self.peek() == b'=' {
                    self.bump();
                    Tok::Ge
                } else {
                    Tok::Gt
                }
            }
            b'.' => {
                if self.peek() == b'.' {
                    self.bump();
                    if self.peek() == b'=' {
                        self.bump();
                        Tok::DotDotEq
                    } else {
                        Tok::DotDot
                    }
                } else {
                    Tok::Dot
                }
            }
            b'?' => Tok::Question,
            b'@' => Tok::At,
            b'(' => Tok::LParen,
            b')' => Tok::RParen,
            b'[' => Tok::LBracket,
            b']' => Tok::RBracket,
            b'{' => Tok::LBrace,
            b'}' => Tok::RBrace,
            b',' => Tok::Comma,
            b';' => Tok::Newline, // ';' acts as a statement terminator
            b':' => Tok::Colon,
            b'#' => Tok::Hash,
            _ => return Err(format!("line {}: unexpected character '{}'", line, c as char)),
        };
        Ok(Some(Token { tok: t, line }))
    }

    fn number(&mut self) -> Result<Token, String> {
        let line = self.line;
        let mut s = String::new();
        while self.peek().is_ascii_digit() || self.peek() == b'_' {
            let c = self.bump();
            if c != b'_' {
                s.push(c as char);
            }
        }
        if self.peek() == b'.' && self.peek2().is_ascii_digit() {
            s.push(self.bump() as char); // '.'
            while self.peek().is_ascii_digit() || self.peek() == b'_' {
                let c = self.bump();
                if c != b'_' {
                    s.push(c as char);
                }
            }
            let f: f64 = s.parse().map_err(|e| format!("line {}: bad float: {}", line, e))?;
            Ok(Token { tok: Tok::Float(f), line })
        } else {
            let i: i64 = s.parse().map_err(|e| format!("line {}: bad int: {}", line, e))?;
            Ok(Token { tok: Tok::Int(i), line })
        }
    }

    fn string(&mut self) -> Result<Token, String> {
        let line = self.line;
        self.bump(); // opening quote
        let mut parts: Vec<StrPart> = Vec::new();
        let mut cur: Vec<u8> = Vec::new();
        loop {
            match self.peek() {
                0 => return Err(format!("line {}: unterminated string", line)),
                b'"' => {
                    self.bump();
                    break;
                }
                b'\\' => {
                    self.bump();
                    let e = self.bump();
                    cur.push(match e {
                        b'n' => b'\n',
                        b't' => b'\t',
                        b'\\' => b'\\',
                        b'"' => b'"',
                        b'$' => b'$',  // \$ → literal $ (e.g. before a { to avoid ${)
                        b'{' => b'{',  // braces need no escaping now; kept for compatibility
                        b'}' => b'}',
                        b'0' => 0u8,
                        _ => return Err(format!("line {}: bad escape '\\{}'", self.line, e as char)),
                    });
                }
                b'$' if self.peek2() == b'{' => {
                    self.bump(); // $
                    self.bump(); // {
                    if !cur.is_empty() {
                        let lit = String::from_utf8(std::mem::take(&mut cur))
                            .map_err(|_| format!("line {}: invalid utf8 in string", line))?;
                        parts.push(StrPart::Lit(lit));
                    }
                    // lex the embedded expression until the matching '}'
                    let start = self.pos;
                    let mut depth = 1usize;
                    while depth > 0 {
                        match self.peek() {
                            0 => return Err(format!("line {}: unterminated interpolation", line)),
                            b'{' => {
                                depth += 1;
                                self.bump();
                            }
                            b'}' => {
                                depth -= 1;
                                if depth > 0 {
                                    self.bump();
                                }
                            }
                            b'"' => {
                                // nested string inside interpolation: skip it
                                self.bump();
                                loop {
                                    match self.peek() {
                                        0 => {
                                            return Err(format!(
                                                "line {}: unterminated string",
                                                line
                                            ))
                                        }
                                        b'\\' => {
                                            self.bump();
                                            self.bump();
                                        }
                                        b'"' => {
                                            self.bump();
                                            break;
                                        }
                                        _ => {
                                            self.bump();
                                        }
                                    }
                                }
                            }
                            _ => {
                                self.bump();
                            }
                        }
                    }
                    let inner = std::str::from_utf8(&self.src[start..self.pos])
                        .map_err(|_| format!("line {}: invalid utf8", line))?
                        .to_string();
                    self.bump(); // closing '}'
                    let mut toks = lex(&inner)?;
                    // drop trailing Eof
                    toks.retain(|t| t.tok != Tok::Eof && t.tok != Tok::Newline);
                    parts.push(StrPart::Interp(toks));
                }
                _ => {
                    // copy raw byte (UTF-8 passthrough; validated at the end)
                    cur.push(self.bump());
                }
            }
        }
        if !cur.is_empty() || parts.is_empty() {
            let lit = String::from_utf8(cur)
                .map_err(|_| format!("line {}: invalid utf8 in string", line))?;
            parts.push(StrPart::Lit(lit));
        }
        Ok(Token { tok: Tok::Str(parts), line })
    }

    fn ident_or_kw(&mut self) -> Token {
        let line = self.line;
        let mut s = String::new();
        while self.peek().is_ascii_alphanumeric() || self.peek() == b'_' {
            s.push(self.bump() as char);
        }
        // trailing '!' for swap!/reset! convention
        if self.peek() == b'!' && self.peek2() != b'=' {
            s.push(self.bump() as char);
        }
        let tok = match s.as_str() {
            "let" => Tok::Let,
            "mut" => Tok::Mut,
            "fn" => Tok::Fn,
            "match" => Tok::Match,
            "if" => Tok::If,
            "else" => Tok::Else,
            "while" => Tok::While,
            "for" => Tok::For,
            "in" => Tok::In,
            "type" => Tok::Type,
            "import" => Tok::Import,
            "export" => Tok::Export,
            "extern" => Tok::Extern,
            "return" => Tok::Return,
            "break" => Tok::Break,
            "continue" => Tok::Continue,
            "true" => Tok::True,
            "false" => Tok::False,
            "and" => Tok::And,
            "or" => Tok::Or,
            "not" => Tok::Not,
            "as" => Tok::As,
            "_" => Tok::Underscore,
            _ => {
                if s.chars().next().map(|c| c.is_ascii_uppercase()).unwrap_or(false) {
                    Tok::TypeName(s)
                } else {
                    Tok::Ident(s)
                }
            }
        };
        Token { tok, line }
    }
}
