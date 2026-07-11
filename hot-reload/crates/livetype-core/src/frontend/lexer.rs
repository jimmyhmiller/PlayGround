//! Hand-written lexer. Produces a flat token stream with line numbers for
//! diagnostics.

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    // keywords
    Struct,
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
    // punctuation
    LBrace,
    RBrace,
    LParen,
    RParen,
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
    Lt,
    Le,
    Gt,
    Ge,
    Arrow, // ->
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
            '/' if i + 1 < bytes.len() && bytes[i + 1] == b'/' => {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            '{' => push(&mut out, Tok::LBrace, line, &mut i),
            '}' => push(&mut out, Tok::RBrace, line, &mut i),
            '(' => push(&mut out, Tok::LParen, line, &mut i),
            ')' => push(&mut out, Tok::RParen, line, &mut i),
            ':' => push(&mut out, Tok::Colon, line, &mut i),
            ';' => push(&mut out, Tok::Semi, line, &mut i),
            ',' => push(&mut out, Tok::Comma, line, &mut i),
            '.' => push(&mut out, Tok::Dot, line, &mut i),
            '+' => push(&mut out, Tok::Plus, line, &mut i),
            '*' => push(&mut out, Tok::Star, line, &mut i),
            '=' => two(&mut out, bytes, &mut i, line, b'=', Tok::EqEq, Tok::Eq),
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
                let n: i64 = src[start..i]
                    .parse()
                    .map_err(|_| format!("line {line}: integer literal out of range"))?;
                out.push(Token { tok: Tok::Int(n), line });
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
