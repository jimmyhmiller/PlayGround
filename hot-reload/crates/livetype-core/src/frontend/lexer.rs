//! Hand-written lexer. Produces a flat token stream with line numbers for
//! diagnostics.

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    // keywords
    Struct,
    Enum,
    Match,
    Fn,
    Let,
    LetOnce,
    Foreign,
    If,
    Else,
    While,
    Return,
    Emit,
    Yield,
    True,
    False,
    // literals & names
    Ident(String),
    Int(i64),
    Float(f64),
    Str(String),
    // punctuation
    LBrace,
    RBrace,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Colon,
    Semi,
    Comma,
    Dot,
    Eq,
    EqEq,
    BangEq,
    Bang,
    Plus,
    Minus,
    Star,
    Slash,
    Lt,
    Le,
    Gt,
    Ge,
    Arrow,      // ->
    FatArrow,   // =>
    ColonColon, // ::
    Eof,
}

#[derive(Clone, Debug)]
pub struct Token {
    pub tok: Tok,
    pub line: usize,
}

pub fn lex(src: &str) -> Result<Vec<Token>, String> {
    let bytes = src.as_bytes();
    let mut i = 0;
    let mut line = 1;
    let mut out = Vec::new();
    while i < bytes.len() {
        let c = bytes[i] as char;
        match c {
            '\n' => {
                line += 1;
                i += 1;
            }
            c if c.is_whitespace() => i += 1,
            '/' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                    while i < bytes.len() && bytes[i] != b'\n' {
                        i += 1;
                    }
                } else {
                    push(&mut out, Tok::Slash, line, &mut i);
                }
            }
            '{' => push(&mut out, Tok::LBrace, line, &mut i),
            '}' => push(&mut out, Tok::RBrace, line, &mut i),
            '[' => push(&mut out, Tok::LBracket, line, &mut i),
            ']' => push(&mut out, Tok::RBracket, line, &mut i),
            '(' => push(&mut out, Tok::LParen, line, &mut i),
            ')' => push(&mut out, Tok::RParen, line, &mut i),
            ':' => two(&mut out, bytes, &mut i, line, b':', Tok::ColonColon, Tok::Colon),
            ';' => push(&mut out, Tok::Semi, line, &mut i),
            ',' => push(&mut out, Tok::Comma, line, &mut i),
            '.' => push(&mut out, Tok::Dot, line, &mut i),
            '+' => push(&mut out, Tok::Plus, line, &mut i),
            '*' => push(&mut out, Tok::Star, line, &mut i),
            '=' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'=' {
                    out.push(Token { tok: Tok::EqEq, line });
                    i += 2;
                } else if i + 1 < bytes.len() && bytes[i + 1] == b'>' {
                    out.push(Token { tok: Tok::FatArrow, line });
                    i += 2;
                } else {
                    push(&mut out, Tok::Eq, line, &mut i);
                }
            }
            '"' => {
                i += 1;
                let mut text = String::new();
                loop {
                    if i >= bytes.len() {
                        return Err(format!("line {line}: unterminated string literal"));
                    }
                    match bytes[i] {
                        b'"' => {
                            i += 1;
                            break;
                        }
                        b'\\' => {
                            i += 1;
                            let esc = *bytes
                                .get(i)
                                .ok_or_else(|| format!("line {line}: unterminated escape"))?;
                            text.push(match esc {
                                b'n' => '\n',
                                b't' => '\t',
                                b'"' => '"',
                                b'\\' => '\\',
                                other => {
                                    return Err(format!(
                                        "line {line}: unknown escape '\\{}'",
                                        other as char
                                    ));
                                }
                            });
                            i += 1;
                        }
                        b'\n' => return Err(format!("line {line}: unterminated string literal")),
                        _ => {
                            // Consume one UTF-8 character (the source is valid UTF-8).
                            let start = i;
                            i += 1;
                            while i < bytes.len() && (bytes[i] & 0xC0) == 0x80 {
                                i += 1;
                            }
                            text.push_str(&src[start..i]);
                        }
                    }
                }
                out.push(Token { tok: Tok::Str(text), line });
            }
            '!' => two(&mut out, bytes, &mut i, line, b'=', Tok::BangEq, Tok::Bang),
            '<' => two(&mut out, bytes, &mut i, line, b'=', Tok::Le, Tok::Lt),
            '>' => two(&mut out, bytes, &mut i, line, b'=', Tok::Ge, Tok::Gt),
            '-' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'>' {
                    out.push(Token { tok: Tok::Arrow, line });
                    i += 2;
                } else {
                    push(&mut out, Tok::Minus, line, &mut i);
                }
            }
            c if c.is_ascii_digit() => {
                let start = i;
                while i < bytes.len() && (bytes[i] as char).is_ascii_digit() {
                    i += 1;
                }
                // `1.5` is a float; `1.foo` stays Int + Dot + Ident.
                if i + 1 < bytes.len()
                    && bytes[i] == b'.'
                    && (bytes[i + 1] as char).is_ascii_digit()
                {
                    i += 1;
                    while i < bytes.len() && (bytes[i] as char).is_ascii_digit() {
                        i += 1;
                    }
                    let x: f64 = src[start..i]
                        .parse()
                        .map_err(|_| format!("line {line}: bad float literal"))?;
                    out.push(Token { tok: Tok::Float(x), line });
                } else {
                    let n: i64 = src[start..i]
                        .parse()
                        .map_err(|_| format!("line {line}: integer literal out of range"))?;
                    out.push(Token { tok: Tok::Int(n), line });
                }
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let start = i;
                while i < bytes.len()
                    && ((bytes[i] as char).is_ascii_alphanumeric() || bytes[i] == b'_')
                {
                    i += 1;
                }
                let word = &src[start..i];
                let tok = match word {
                    "struct" => Tok::Struct,
                    "enum" => Tok::Enum,
                    "match" => Tok::Match,
                    "fn" => Tok::Fn,
                    "let" => Tok::Let,
                    "letonce" => Tok::LetOnce,
                    "foreign" => Tok::Foreign,
                    "if" => Tok::If,
                    "else" => Tok::Else,
                    "while" => Tok::While,
                    "return" => Tok::Return,
                    "emit" => Tok::Emit,
                    "yield" => Tok::Yield,
                    "true" => Tok::True,
                    "false" => Tok::False,
                    _ => Tok::Ident(word.to_string()),
                };
                out.push(Token { tok, line });
            }
            other => return Err(format!("line {line}: unexpected character '{other}'")),
        }
    }
    out.push(Token { tok: Tok::Eof, line });
    Ok(out)
}

fn push(out: &mut Vec<Token>, tok: Tok, line: usize, i: &mut usize) {
    out.push(Token { tok, line });
    *i += 1;
}

/// Lex a one-or-two character token: if the next byte is `second`, emit `two`
/// (consuming both); otherwise emit `one`.
fn two(out: &mut Vec<Token>, bytes: &[u8], i: &mut usize, line: usize, second: u8, two: Tok, one: Tok) {
    if *i + 1 < bytes.len() && bytes[*i + 1] == second {
        out.push(Token { tok: two, line });
        *i += 2;
    } else {
        out.push(Token { tok: one, line });
        *i += 1;
    }
}
