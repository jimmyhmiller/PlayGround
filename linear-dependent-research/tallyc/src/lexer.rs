//! Tokeniser for the lambda-Tally surface language (v0 core).

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    Int(i64),
    Ident(String),
    Punc(char), // one of { } ( ) ; , . = + :
    Eof,
}

pub fn lex(src: &str) -> Result<Vec<Tok>, String> {
    let b = src.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    while i < b.len() {
        let c = b[i] as char;
        if c.is_whitespace() {
            i += 1;
        } else if c == '/' && i + 1 < b.len() && b[i + 1] as char == '/' {
            while i < b.len() && b[i] as char != '\n' {
                i += 1;
            }
        } else if c.is_ascii_digit() {
            let s = i;
            while i < b.len() && (b[i] as char).is_ascii_digit() {
                i += 1;
            }
            out.push(Tok::Int(src[s..i].parse().map_err(|_| "bad int")?));
        } else if c.is_alphabetic() || c == '_' {
            let s = i;
            while i < b.len() && {
                let ch = b[i] as char;
                ch.is_alphanumeric() || ch == '_'
            } {
                i += 1;
            }
            out.push(Tok::Ident(src[s..i].to_string()));
        } else if "{}();,.=+:<>".contains(c) {
            out.push(Tok::Punc(c));
            i += 1;
        } else {
            return Err(format!("unexpected character {c:?}"));
        }
    }
    out.push(Tok::Eof);
    Ok(out)
}
