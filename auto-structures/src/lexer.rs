//! Hand-written lexer.

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    Int(i64),
    Str(String),
    Ident(String),
    // keywords
    Let,
    If,
    Else,
    While,
    For,
    In,
    True,
    False,
    Not,
    And,
    Or,
    Nil,
    Break,
    Continue,
    // punctuation / operators
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
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
                out.push(t(Tok::LParen, line));
                i += 1;
            }
            ')' => {
                out.push(t(Tok::RParen, line));
                i += 1;
            }
            '{' => {
                out.push(t(Tok::LBrace, line));
                i += 1;
            }
            '}' => {
                out.push(t(Tok::RBrace, line));
                i += 1;
            }
            '[' => {
                out.push(t(Tok::LBracket, line));
                i += 1;
            }
            ']' => {
                out.push(t(Tok::RBracket, line));
                i += 1;
            }
            ',' => {
                out.push(t(Tok::Comma, line));
                i += 1;
            }
            '+' => {
                out.push(t(Tok::Plus, line));
                i += 1;
            }
            '-' => {
                out.push(t(Tok::Minus, line));
                i += 1;
            }
            '*' => {
                out.push(t(Tok::Star, line));
                i += 1;
            }
            '/' => {
                out.push(t(Tok::Slash, line));
                i += 1;
            }
            '%' => {
                out.push(t(Tok::Percent, line));
                i += 1;
            }
            '=' => {
                if peek(&chars, i + 1) == Some('=') {
                    out.push(t(Tok::Eq, line));
                    i += 2;
                } else {
                    out.push(t(Tok::Assign, line));
                    i += 1;
                }
            }
            '!' => {
                if peek(&chars, i + 1) == Some('=') {
                    out.push(t(Tok::Ne, line));
                    i += 2;
                } else {
                    return Err(format!("line {}: unexpected '!'", line));
                }
            }
            '<' => {
                if peek(&chars, i + 1) == Some('=') {
                    out.push(t(Tok::Le, line));
                    i += 2;
                } else {
                    out.push(t(Tok::Lt, line));
                    i += 1;
                }
            }
            '>' => {
                if peek(&chars, i + 1) == Some('=') {
                    out.push(t(Tok::Ge, line));
                    i += 2;
                } else {
                    out.push(t(Tok::Gt, line));
                    i += 1;
                }
            }
            '"' => {
                i += 1;
                let mut s = String::new();
                while i < chars.len() && chars[i] != '"' {
                    if chars[i] == '\\' && i + 1 < chars.len() {
                        i += 1;
                        let e = chars[i];
                        s.push(match e {
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
                i += 1; // closing quote
                out.push(t(Tok::Str(s), line));
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
                out.push(t(Tok::Int(n), line));
            }
            c if c.is_alphabetic() || c == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                let tok = match word.as_str() {
                    "let" => Tok::Let,
                    "if" => Tok::If,
                    "else" => Tok::Else,
                    "while" => Tok::While,
                    "for" => Tok::For,
                    "in" => Tok::In,
                    "true" => Tok::True,
                    "false" => Tok::False,
                    "not" => Tok::Not,
                    "and" => Tok::And,
                    "or" => Tok::Or,
                    "nil" => Tok::Nil,
                    "break" => Tok::Break,
                    "continue" => Tok::Continue,
                    _ => Tok::Ident(word),
                };
                out.push(t(tok, line));
            }
            other => return Err(format!("line {}: unexpected character '{}'", line, other)),
        }
    }
    out.push(t(Tok::Eof, line));
    Ok(out)
}

fn t(tok: Tok, line: usize) -> Token {
    Token { tok, line }
}

fn peek(chars: &[char], i: usize) -> Option<char> {
    chars.get(i).copied()
}
