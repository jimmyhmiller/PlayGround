//! Lexer for Glaze. Newlines are significant: they terminate properties and
//! token defs (the brace syntax has no `;`). `//` and `/* */` are comments.

use crate::GlazeError;

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Ident(String),
    /// numeric literal + optional unit suffix (`px`, `%`, `em`, …); `8px` → (8.0, "px")
    Num(f64, Option<String>),
    /// `#c9a96a` (the leading `#` is stripped)
    Hex(String),
    LParen,
    RParen,
    LBrace,
    RBrace,
    Comma,
    Colon,
    Question,
    /// `=` (assignment in a token def). Distinct from the `==` operator.
    Assign,
    /// an operator: `+ - * / > < >= <= ==`
    Op(String),
    /// statement terminator
    Newline,
    Eof,
}

pub fn lex(src: &str) -> Result<Vec<Tok>, GlazeError> {
    let cs: Vec<char> = src.chars().collect();
    let n = cs.len();
    let mut i = 0;
    let mut out = Vec::new();
    while i < n {
        let c = cs[i];
        // line comment
        if c == '/' && i + 1 < n && cs[i + 1] == '/' {
            while i < n && cs[i] != '\n' {
                i += 1;
            }
            continue;
        }
        // block comment
        if c == '/' && i + 1 < n && cs[i + 1] == '*' {
            i += 2;
            while i + 1 < n && !(cs[i] == '*' && cs[i + 1] == '/') {
                i += 1;
            }
            i += 2;
            continue;
        }
        if c == '\n' {
            out.push(Tok::Newline);
            i += 1;
            continue;
        }
        if c == ' ' || c == '\t' || c == '\r' {
            i += 1;
            continue;
        }
        if c == '#' {
            let start = i + 1;
            i += 1;
            while i < n && cs[i].is_ascii_hexdigit() {
                i += 1;
            }
            out.push(Tok::Hex(cs[start..i].iter().collect()));
            continue;
        }
        if c.is_ascii_digit() || (c == '.' && i + 1 < n && cs[i + 1].is_ascii_digit()) {
            let start = i;
            while i < n && (cs[i].is_ascii_digit() || cs[i] == '.') {
                i += 1;
            }
            let num: f64 = cs[start..i]
                .iter()
                .collect::<String>()
                .parse()
                .map_err(|_| GlazeError::Lex(format!("bad number near `{}`", c)))?;
            // optional unit: `%` or an alphabetic run, attached with no space
            let mut unit = None;
            if i < n && cs[i] == '%' {
                unit = Some("%".to_string());
                i += 1;
            } else if i < n && cs[i].is_ascii_alphabetic() {
                let us = i;
                while i < n && cs[i].is_ascii_alphabetic() {
                    i += 1;
                }
                unit = Some(cs[us..i].iter().collect());
            }
            out.push(Tok::Num(num, unit));
            continue;
        }
        if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < n && (cs[i].is_ascii_alphanumeric() || cs[i] == '_' || cs[i] == '.') {
                i += 1;
            }
            out.push(Tok::Ident(cs[start..i].iter().collect()));
            continue;
        }
        // operators & punctuation
        match c {
            '(' => out.push(Tok::LParen),
            ')' => out.push(Tok::RParen),
            '{' => out.push(Tok::LBrace),
            '}' => out.push(Tok::RBrace),
            ',' => out.push(Tok::Comma),
            ':' => out.push(Tok::Colon),
            '?' => out.push(Tok::Question),
            '+' | '-' | '*' | '/' => out.push(Tok::Op(c.to_string())),
            '=' => {
                if i + 1 < n && cs[i + 1] == '=' {
                    out.push(Tok::Op("==".into()));
                    i += 1;
                } else {
                    out.push(Tok::Assign);
                }
            }
            '>' | '<' => {
                if i + 1 < n && cs[i + 1] == '=' {
                    out.push(Tok::Op(format!("{}=", c)));
                    i += 1;
                } else {
                    out.push(Tok::Op(c.to_string()));
                }
            }
            _ => return Err(GlazeError::Lex(format!("unexpected character `{}`", c))),
        }
        i += 1;
    }
    out.push(Tok::Eof);
    Ok(out)
}
