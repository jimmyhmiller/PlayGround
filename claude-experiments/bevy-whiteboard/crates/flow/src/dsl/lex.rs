//! Tokenizer for the Flow DSL.
//!
//! Produces a flat `Vec<Token>` with line/col info for error messages.

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    // Punctuation / operators
    LBrace, RBrace, LParen, RParen, LBracket, RBracket,
    Comma, Colon, Semi,
    Arrow,        // ->
    LArrow,       // <-
    Assign,       // :=
    Equals,       // =  (slot/param initializer)
    Eq, NEq,      // ==  !=
    Lt, Le, Gt, Ge,
    Plus, Minus, Star, Slash, Percent, Caret,
    AndAnd, OrOr, Bang,
    Dot,

    // Keywords
    Params, Node, Compound, In, Out, Edges, Scenario, Template,
    Slots, Rule, On, OnSpawn, When, Do, Probes,
    Push, Pop, Drop, Emit, EmitEach, Record, Spawn, Error,
    To, As, At, Inject, SetParam, SetSlot, Kill,
    True, False, Nil, Self_,
    // Packet metadata / return_path modifiers for Emit/EmitEach.
    // `Meta` doubles as the `meta(key)` expression function; the
    // parser distinguishes by the following `{` vs `(`.
    Meta, ForgetMeta, Pushing, Popping, ReturnPath,

    // Literals
    Int(i64),
    Float(f64),
    Str(String),
    TimeNs(u64),              // pre-parsed time literal (already in ns)

    // Identifiers
    Ident(String),

    Eof,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: Tok,
    pub line: usize,
    pub col: usize,
}

pub fn lex(src: &str) -> Result<Vec<Token>, String> {
    let mut out = Vec::new();
    let bytes = src.as_bytes();
    let mut i = 0;
    let mut line = 1usize;
    let mut col = 1usize;

    let update_pos = |c: u8, line: &mut usize, col: &mut usize| {
        if c == b'\n' { *line += 1; *col = 1; } else { *col += 1; }
    };

    while i < bytes.len() {
        let c = bytes[i];

        // Whitespace
        if c.is_ascii_whitespace() { update_pos(c, &mut line, &mut col); i += 1; continue; }

        // Line comments  (# …  or  // …)
        if c == b'#' || (c == b'/' && i + 1 < bytes.len() && bytes[i+1] == b'/') {
            while i < bytes.len() && bytes[i] != b'\n' { i += 1; col += 1; }
            continue;
        }

        let start_line = line; let start_col = col;
        let push = |out: &mut Vec<Token>, k: Tok, line: usize, col: usize| {
            out.push(Token { kind: k, line, col });
        };

        // Multi-char punctuation
        let peek = |ofs: usize| -> Option<u8> { bytes.get(i + ofs).copied() };
        match (c, peek(1)) {
            (b'-', Some(b'>')) => { push(&mut out, Tok::Arrow, start_line, start_col); i += 2; col += 2; continue; }
            (b'<', Some(b'-')) => { push(&mut out, Tok::LArrow, start_line, start_col); i += 2; col += 2; continue; }
            (b':', Some(b'=')) => { push(&mut out, Tok::Assign, start_line, start_col); i += 2; col += 2; continue; }
            (b'=', Some(b'=')) => { push(&mut out, Tok::Eq, start_line, start_col); i += 2; col += 2; continue; }
            (b'!', Some(b'=')) => { push(&mut out, Tok::NEq, start_line, start_col); i += 2; col += 2; continue; }
            (b'<', Some(b'=')) => { push(&mut out, Tok::Le, start_line, start_col); i += 2; col += 2; continue; }
            (b'>', Some(b'=')) => { push(&mut out, Tok::Ge, start_line, start_col); i += 2; col += 2; continue; }
            (b'&', Some(b'&')) => { push(&mut out, Tok::AndAnd, start_line, start_col); i += 2; col += 2; continue; }
            (b'|', Some(b'|')) => { push(&mut out, Tok::OrOr, start_line, start_col); i += 2; col += 2; continue; }
            _ => {}
        }

        // Single-char punctuation
        let single = match c {
            b'{' => Some(Tok::LBrace),
            b'}' => Some(Tok::RBrace),
            b'(' => Some(Tok::LParen),
            b')' => Some(Tok::RParen),
            b'[' => Some(Tok::LBracket),
            b']' => Some(Tok::RBracket),
            b',' => Some(Tok::Comma),
            b':' => Some(Tok::Colon),
            b';' => Some(Tok::Semi),
            b'<' => Some(Tok::Lt),
            b'>' => Some(Tok::Gt),
            b'=' => Some(Tok::Equals),
            b'+' => Some(Tok::Plus),
            b'-' => Some(Tok::Minus),
            b'*' => Some(Tok::Star),
            b'/' => Some(Tok::Slash),
            b'%' => Some(Tok::Percent),
            b'^' => Some(Tok::Caret),
            b'!' => Some(Tok::Bang),
            b'.' => Some(Tok::Dot),
            _ => None,
        };
        if let Some(k) = single {
            push(&mut out, k, start_line, start_col);
            i += 1; col += 1;
            continue;
        }

        // String literal
        if c == b'"' {
            i += 1; col += 1;
            let mut s = String::new();
            while i < bytes.len() && bytes[i] != b'"' {
                if bytes[i] == b'\\' && i + 1 < bytes.len() {
                    let esc = bytes[i+1];
                    s.push(match esc {
                        b'n' => '\n', b't' => '\t', b'r' => '\r',
                        b'"' => '"',  b'\\' => '\\', other => other as char,
                    });
                    i += 2; col += 2;
                } else {
                    s.push(bytes[i] as char);
                    update_pos(bytes[i], &mut line, &mut col);
                    i += 1;
                }
            }
            if i >= bytes.len() {
                return Err(format!("{}:{}: unterminated string", start_line, start_col));
            }
            i += 1; col += 1; // closing "
            push(&mut out, Tok::Str(s), start_line, start_col);
            continue;
        }

        // Number, possibly with time suffix
        if c.is_ascii_digit() {
            let start_i = i;
            while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.' || bytes[i] == b'_') {
                i += 1; col += 1;
            }
            let num_str: String = bytes[start_i..i].iter().filter(|&&b| b != b'_').map(|&b| b as char).collect();
            // Time suffix?
            let mut suffix_end = i;
            while suffix_end < bytes.len() && bytes[suffix_end].is_ascii_alphabetic() {
                suffix_end += 1;
            }
            let suffix: String = bytes[i..suffix_end].iter().map(|&b| b as char).collect();
            let mul: Option<u64> = match suffix.as_str() {
                "ns" => Some(1),
                "us" => Some(1_000),
                "ms" => Some(1_000_000),
                "s"  => Some(1_000_000_000),
                "" => None,
                _ => None,
            };
            if let Some(m) = mul {
                let v: f64 = num_str.parse().map_err(|_| format!("{}:{}: bad number `{}`", start_line, start_col, num_str))?;
                let ns = (v * m as f64) as u64;
                col += suffix_end - i; i = suffix_end;
                push(&mut out, Tok::TimeNs(ns), start_line, start_col);
            } else if num_str.contains('.') {
                let v: f64 = num_str.parse().map_err(|_| format!("{}:{}: bad float `{}`", start_line, start_col, num_str))?;
                push(&mut out, Tok::Float(v), start_line, start_col);
            } else {
                let v: i64 = num_str.parse().map_err(|_| format!("{}:{}: bad int `{}`", start_line, start_col, num_str))?;
                push(&mut out, Tok::Int(v), start_line, start_col);
            }
            continue;
        }

        // Identifier / keyword
        if c.is_ascii_alphabetic() || c == b'_' {
            let start_i = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1; col += 1;
            }
            let ident: String = bytes[start_i..i].iter().map(|&b| b as char).collect();
            let kw = match ident.as_str() {
                "params" => Some(Tok::Params),
                "node" => Some(Tok::Node),
                "compound" => Some(Tok::Compound),
                "in" => Some(Tok::In),
                "out" => Some(Tok::Out),
                "edges" => Some(Tok::Edges),
                "scenario" => Some(Tok::Scenario),
                "template" => Some(Tok::Template),
                "slots" => Some(Tok::Slots),
                "probes" => Some(Tok::Probes),
                "rule" => Some(Tok::Rule),
                "on" => Some(Tok::On),
                "on_spawn" => Some(Tok::OnSpawn),
                "when" => Some(Tok::When),
                "do" => Some(Tok::Do),
                "push" => Some(Tok::Push),
                "pop" => Some(Tok::Pop),
                "drop" => Some(Tok::Drop),
                "emit" => Some(Tok::Emit),
                "emit_each" => Some(Tok::EmitEach),
                "meta" => Some(Tok::Meta),
                "forget_meta" => Some(Tok::ForgetMeta),
                "pushing" => Some(Tok::Pushing),
                "popping" => Some(Tok::Popping),
                "return_path" => Some(Tok::ReturnPath),
                "record" => Some(Tok::Record),
                "error" => Some(Tok::Error),
                "spawn" => Some(Tok::Spawn),
                "to" => Some(Tok::To),
                "as" => Some(Tok::As),
                "at" => Some(Tok::At),
                "inject" => Some(Tok::Inject),
                "set_param" => Some(Tok::SetParam),
                "set_slot" => Some(Tok::SetSlot),
                "kill" => Some(Tok::Kill),
                "true" => Some(Tok::True),
                "false" => Some(Tok::False),
                "nil" => Some(Tok::Nil),
                "self" => Some(Tok::Self_),
                _ => None,
            };
            let k = kw.unwrap_or(Tok::Ident(ident));
            push(&mut out, k, start_line, start_col);
            continue;
        }

        return Err(format!("{}:{}: unexpected character `{}`", line, col, c as char));
    }
    out.push(Token { kind: Tok::Eof, line, col });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn basic_tokens() {
        let toks = lex("node Pool { slots { x: Int = 0 } }").unwrap();
        let kinds: Vec<Tok> = toks.into_iter().map(|t| t.kind).collect();
        assert!(matches!(kinds[0], Tok::Node));
        assert!(matches!(kinds[1], Tok::Ident(ref s) if s == "Pool"));
    }
    #[test]
    fn time_literals() {
        let toks = lex("1ms 25us 1s 500ns").unwrap();
        let times: Vec<u64> = toks.into_iter().filter_map(|t| if let Tok::TimeNs(n) = t.kind { Some(n) } else { None }).collect();
        assert_eq!(times, vec![1_000_000, 25_000, 1_000_000_000, 500]);
    }
    #[test]
    fn arrows() {
        let toks = lex("-> <- := == != <= >=").unwrap();
        let kinds: Vec<_> = toks.into_iter().map(|t| t.kind).collect();
        assert_eq!(kinds[..7], [Tok::Arrow, Tok::LArrow, Tok::Assign, Tok::Eq, Tok::NEq, Tok::Le, Tok::Ge]);
    }
}
