//! Lexer shared by program (.dl) and declaration (.decl) files. Keywords are
//! recognized contextually in the parsers, so the token set stays small.

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    Int(i64),
    Text(String),
    Ident(String),
    LParen,
    RParen,
    LBrace,
    RBrace,
    Comma,
    Semi,
    Colon,
    Assign,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Eof,
}

#[derive(Clone, Debug)]
pub struct Token {
    pub tok: Tok,
    pub line: usize,
}

pub fn lex(src: &str) -> Result<Vec<Token>, String> {
    let chars: Vec<char> = src.chars().collect();
    let mut i = 0;
    let mut line = 1;
    let mut out = Vec::new();
    let push = |out: &mut Vec<Token>, t: Tok, line: usize| out.push(Token { tok: t, line });

    while i < chars.len() {
        let c = chars[i];
        match c {
            '\n' => {
                line += 1;
                i += 1;
            }
            ' ' | '\t' | '\r' => i += 1,
            '#' => {
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
            }
            '(' => {
                push(&mut out, Tok::LParen, line);
                i += 1;
            }
            ')' => {
                push(&mut out, Tok::RParen, line);
                i += 1;
            }
            '{' => {
                push(&mut out, Tok::LBrace, line);
                i += 1;
            }
            '}' => {
                push(&mut out, Tok::RBrace, line);
                i += 1;
            }
            ',' => {
                push(&mut out, Tok::Comma, line);
                i += 1;
            }
            ';' => {
                push(&mut out, Tok::Semi, line);
                i += 1;
            }
            ':' => {
                push(&mut out, Tok::Colon, line);
                i += 1;
            }
            '+' => {
                push(&mut out, Tok::Plus, line);
                i += 1;
            }
            '-' => {
                push(&mut out, Tok::Minus, line);
                i += 1;
            }
            '*' => {
                push(&mut out, Tok::Star, line);
                i += 1;
            }
            '/' => {
                push(&mut out, Tok::Slash, line);
                i += 1;
            }
            '%' => {
                push(&mut out, Tok::Percent, line);
                i += 1;
            }
            '=' => {
                if next_is(&chars, i, '=') {
                    push(&mut out, Tok::Eq, line);
                    i += 2;
                } else {
                    push(&mut out, Tok::Assign, line);
                    i += 1;
                }
            }
            '!' => {
                if next_is(&chars, i, '=') {
                    push(&mut out, Tok::Ne, line);
                    i += 2;
                } else {
                    return Err(format!("line {}: unexpected '!'", line));
                }
            }
            '<' => {
                if next_is(&chars, i, '=') {
                    push(&mut out, Tok::Le, line);
                    i += 2;
                } else {
                    push(&mut out, Tok::Lt, line);
                    i += 1;
                }
            }
            '>' => {
                if next_is(&chars, i, '=') {
                    push(&mut out, Tok::Ge, line);
                    i += 2;
                } else {
                    push(&mut out, Tok::Gt, line);
                    i += 1;
                }
            }
            '"' => {
                i += 1;
                let mut s = String::new();
                while i < chars.len() && chars[i] != '"' {
                    if chars[i] == '\\' && i + 1 < chars.len() {
                        i += 1;
                        s.push(match chars[i] {
                            'n' => '\n',
                            't' => '\t',
                            '\\' => '\\',
                            '"' => '"',
                            other => other,
                        });
                    } else {
                        if chars[i] == '\n' {
                            line += 1;
                        }
                        s.push(chars[i]);
                    }
                    i += 1;
                }
                if i >= chars.len() {
                    return Err(format!("line {}: unterminated string", line));
                }
                i += 1;
                push(&mut out, Tok::Text(s), line);
            }
            c if c.is_ascii_digit() => {
                let start = i;
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
                let num: String = chars[start..i].iter().collect();
                let n = num
                    .parse::<i64>()
                    .map_err(|_| format!("line {}: bad integer '{}'", line, num))?;
                push(&mut out, Tok::Int(n), line);
            }
            c if c.is_alphabetic() || c == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                push(&mut out, Tok::Ident(word), line);
            }
            other => return Err(format!("line {}: unexpected character '{}'", line, other)),
        }
    }
    push(&mut out, Tok::Eof, line);
    Ok(out)
}

fn next_is(chars: &[char], i: usize, c: char) -> bool {
    chars.get(i + 1).copied() == Some(c)
}
