//! S-expression reader.
//!
//! Surface is intentionally tiny: lists `( )`, vectors `[ ]`, integers,
//! keywords `:kw`, and symbols. `;` starts a line comment. The prettier
//! surface sugar from the design doc (`@rdi`, `#{...}`) is deferred to the
//! macro layer; semantics first.
//!
//! Every node carries a `Span` (byte range into the source) so the parser and
//! checker can point diagnostics at the exact offending form. Equality
//! (`PartialEq`) deliberately ignores the span: the macro layer compares forms
//! structurally and must not see two otherwise-identical forms as different just
//! because they were read from different positions.

use crate::span::{Diag, Span};

/// A read form: its `kind` plus the source `span` it came from.
#[derive(Debug, Clone)]
pub struct Sexp {
    pub kind: SexpKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SexpKind {
    Int(i64),
    /// Floating-point literal, e.g. `3.14` or `1e9`.
    Float(f64),
    Sym(String),
    /// Keyword `:foo` stored without the leading colon.
    Keyword(String),
    /// String literal `"..."` — at the value level a `(slice u8)` view; also
    /// used by macros for name munging (compile-time string ops).
    Str(String),
    /// C-string literal `c"..."` — a NUL-terminated `(ptr i8)` for FFI. Kept
    /// distinct from `Str` so the slice string and the FFI cstring don't conflate.
    CStr(String),
    List(Vec<Sexp>),
    Vector(Vec<Sexp>),
}

/// Structural equality — span-insensitive (see the module note).
impl PartialEq for Sexp {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
    }
}

impl Sexp {
    pub fn new(kind: SexpKind, span: Span) -> Sexp {
        Sexp { kind, span }
    }

    // Ergonomic, span-less constructors for synthesized forms (macro output and
    // tests). The bytes don't exist in any source, so the span is `DUMMY`.
    pub fn sym(s: impl Into<String>) -> Sexp {
        Sexp::new(SexpKind::Sym(s.into()), Span::DUMMY)
    }
    pub fn keyword(s: impl Into<String>) -> Sexp {
        Sexp::new(SexpKind::Keyword(s.into()), Span::DUMMY)
    }
    pub fn cstring(s: impl Into<String>) -> Sexp {
        Sexp::new(SexpKind::CStr(s.into()), Span::DUMMY)
    }
    pub fn string(s: impl Into<String>) -> Sexp {
        Sexp::new(SexpKind::Str(s.into()), Span::DUMMY)
    }
    pub fn int(n: i64) -> Sexp {
        Sexp::new(SexpKind::Int(n), Span::DUMMY)
    }
    pub fn float(x: f64) -> Sexp {
        Sexp::new(SexpKind::Float(x), Span::DUMMY)
    }
    pub fn list(items: Vec<Sexp>) -> Sexp {
        Sexp::new(SexpKind::List(items), Span::DUMMY)
    }
    pub fn vector(items: Vec<Sexp>) -> Sexp {
        Sexp::new(SexpKind::Vector(items), Span::DUMMY)
    }
}

/// Read every top-level form in `src`, stamping each node's span with `source`
/// (the source's id in the [`SourceMap`](crate::span::SourceMap)). Every file the
/// compiler reads — the main source, each `import`, the prelude — is registered as
/// its own source and read with its own id, so a diagnostic resolves against the
/// right file.
pub fn read_all(src: &str, source: u32) -> Result<Vec<Sexp>, Diag> {
    let tokens = tokenize(src, source)?;
    let mut p = Parser { toks: tokens, pos: 0 };
    let mut out = Vec::new();
    while p.pos < p.toks.len() {
        out.push(p.parse()?);
    }
    Ok(out)
}

#[derive(Debug, Clone)]
struct Tok {
    kind: TokKind,
    span: Span,
}

#[derive(Debug, Clone, PartialEq)]
enum TokKind {
    Open(char),  // ( or [
    Close(char), // ) or ]
    /// A reader-macro prefix: `'`→quote, `` ` ``→quasiquote, `~`→unquote,
    /// `~@`→unquote-splicing. The next form is wrapped in `(<sym> form)`.
    Prefix(&'static str),
    Str(String),
    /// A `c"..."` C-string literal.
    CStr(String),
    Atom(String),
}

/// Read a string body after the opening quote has been consumed, returning the
/// unescaped contents and the byte offset just past the closing quote. `open` is
/// the byte offset of the literal's start (for error spans).
fn read_string_body(
    chars: &mut std::iter::Peekable<std::str::CharIndices<'_>>,
    open: usize,
    src_len: usize,
    source: u32,
) -> Result<(String, usize), Diag> {
    let mut s = String::new();
    loop {
        match chars.next() {
            None => return Err(Diag::at(Span::new(source, open, src_len), "unterminated string literal")),
            Some((j, '"')) => return Ok((s, j + 1)),
            Some((_, '\\')) => match chars.next() {
                Some((_, 'n')) => s.push('\n'),
                Some((_, 't')) => s.push('\t'),
                Some((_, '"')) => s.push('"'),
                Some((_, '\\')) => s.push('\\'),
                Some((_, c)) => s.push(c),
                None => {
                    return Err(Diag::at(Span::new(source, open, src_len), "unterminated string escape"))
                }
            },
            Some((_, c)) => s.push(c),
        }
    }
}

fn tokenize(src: &str, source: u32) -> Result<Vec<Tok>, Diag> {
    let mut toks = Vec::new();
    let mut chars = src.char_indices().peekable();
    while let Some(&(i, c)) = chars.peek() {
        match c {
            c if c.is_whitespace() => {
                chars.next();
            }
            ';' => {
                // comment to end of line
                while let Some(&(_, c)) = chars.peek() {
                    chars.next();
                    if c == '\n' {
                        break;
                    }
                }
            }
            '(' | '[' => {
                toks.push(Tok { kind: TokKind::Open(c), span: Span::new(source, i, i + 1) });
                chars.next();
            }
            ')' | ']' => {
                toks.push(Tok { kind: TokKind::Close(c), span: Span::new(source, i, i + 1) });
                chars.next();
            }
            '\'' => {
                toks.push(Tok { kind: TokKind::Prefix("quote"), span: Span::new(source, i, i + 1) });
                chars.next();
            }
            '`' => {
                toks.push(Tok { kind: TokKind::Prefix("quasiquote"), span: Span::new(source, i, i + 1) });
                chars.next();
            }
            '~' => {
                chars.next();
                if matches!(chars.peek(), Some(&(_, '@'))) {
                    let (j, _) = chars.next().unwrap();
                    toks.push(Tok {
                        kind: TokKind::Prefix("unquote-splicing"),
                        span: Span::new(source, i, j + 1),
                    });
                } else {
                    toks.push(Tok { kind: TokKind::Prefix("unquote"), span: Span::new(source, i, i + 1) });
                }
            }
            '"' => {
                chars.next();
                let (s, end) = read_string_body(&mut chars, i, src.len(), source)?;
                toks.push(Tok { kind: TokKind::Str(s), span: Span::new(source, i, end) });
            }
            _ => {
                let start = i;
                let mut end = i;
                let mut s = String::new();
                while let Some(&(j, c)) = chars.peek() {
                    if c.is_whitespace()
                        || matches!(c, '(' | ')' | '[' | ']' | ';' | '\'' | '`' | '~' | '"')
                    {
                        break;
                    }
                    s.push(c);
                    end = j + c.len_utf8();
                    chars.next();
                }
                // `c"..."` (a bare `c` immediately followed by an opening quote) is
                // a C-string literal, distinct from a `"..."` (slice) string.
                if s == "c" && matches!(chars.peek(), Some((_, '"'))) {
                    chars.next(); // consume the opening quote
                    let (cs, cend) = read_string_body(&mut chars, start, src.len(), source)?;
                    toks.push(Tok { kind: TokKind::CStr(cs), span: Span::new(source, start, cend) });
                } else {
                    toks.push(Tok { kind: TokKind::Atom(s), span: Span::new(source, start, end) });
                }
            }
        }
    }
    Ok(toks)
}

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn parse(&mut self) -> Result<Sexp, Diag> {
        let tok = self
            .toks
            .get(self.pos)
            .cloned()
            .ok_or_else(|| Diag::new("unexpected end of input"))?;
        self.pos += 1;
        match tok.kind {
            TokKind::Open(open) => {
                let close = if open == '(' { ')' } else { ']' };
                let mut items = Vec::new();
                loop {
                    match self.toks.get(self.pos) {
                        None => {
                            return Err(Diag::at(tok.span, format!("unclosed '{open}'")))
                        }
                        Some(t) if matches!(t.kind, TokKind::Close(_)) => {
                            let c = match t.kind {
                                TokKind::Close(c) => c,
                                _ => unreachable!(),
                            };
                            if c != close {
                                return Err(Diag::at(
                                    t.span,
                                    format!("mismatched delimiters: '{open}' .. '{c}'"),
                                ));
                            }
                            let span = tok.span.to(t.span);
                            self.pos += 1;
                            let kind = if open == '(' {
                                SexpKind::List(items)
                            } else {
                                SexpKind::Vector(items)
                            };
                            return Ok(Sexp::new(kind, span));
                        }
                        Some(_) => items.push(self.parse()?),
                    }
                }
            }
            TokKind::Prefix(sym) => {
                let inner = self.parse()?;
                let span = tok.span.to(inner.span);
                Ok(Sexp::new(
                    SexpKind::List(vec![Sexp::new(SexpKind::Sym(sym.to_string()), tok.span), inner]),
                    span,
                ))
            }
            TokKind::Str(s) => Ok(Sexp::new(SexpKind::Str(s), tok.span)),
            TokKind::CStr(s) => Ok(Sexp::new(SexpKind::CStr(s), tok.span)),
            TokKind::Close(c) => Err(Diag::at(tok.span, format!("unexpected '{c}'"))),
            TokKind::Atom(s) => Ok(Sexp::new(atom(&s), tok.span)),
        }
    }
}

impl std::fmt::Display for Sexp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            SexpKind::Int(n) => write!(f, "{n}"),
            SexpKind::Float(x) => write!(f, "{x:?}"),
            SexpKind::Sym(s) => write!(f, "{s}"),
            SexpKind::Keyword(k) => write!(f, ":{k}"),
            SexpKind::Str(s) => write!(f, "{s:?}"),
            SexpKind::CStr(s) => write!(f, "c{s:?}"),
            SexpKind::List(items) => write_seq(f, items, '(', ')'),
            SexpKind::Vector(items) => write_seq(f, items, '[', ']'),
        }
    }
}

fn write_seq(
    f: &mut std::fmt::Formatter<'_>,
    items: &[Sexp],
    open: char,
    close: char,
) -> std::fmt::Result {
    write!(f, "{open}")?;
    for (i, it) in items.iter().enumerate() {
        if i > 0 {
            write!(f, " ")?;
        }
        write!(f, "{it}")?;
    }
    write!(f, "{close}")
}

/// Canonical, lossless, span-bearing textual encoding of read forms — the
/// differential-oracle target for the self-hosted reader (`coil dump-read`).
/// Each node is `(<tag>@<lo>:<hi> …)`; floats are dumped as raw IEEE bits and
/// strings via a fixed per-byte escape, so *formatting* can never be the axis
/// along which two faithful readers diverge. A dummy span prints as `@D:D`.
pub fn dump_canonical(forms: &[Sexp]) -> String {
    let mut out = String::new();
    for (i, f) in forms.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        dump_node(f, &mut out);
    }
    out
}

fn dump_span(span: Span, out: &mut String) {
    crate::span::dump_span_into(span, out);
}

fn dump_node(s: &Sexp, out: &mut String) {
    use std::fmt::Write;
    match &s.kind {
        SexpKind::Int(n) => {
            out.push_str("(int");
            dump_span(s.span, out);
            write!(out, " {n})").unwrap();
        }
        SexpKind::Float(x) => {
            out.push_str("(float");
            dump_span(s.span, out);
            write!(out, " 0x{:016x})", x.to_bits()).unwrap();
        }
        SexpKind::Sym(t) => dump_atom("sym", t, s.span, out),
        SexpKind::Keyword(k) => dump_atom("kw", k, s.span, out),
        SexpKind::Str(t) => dump_atom("str", t, s.span, out),
        SexpKind::CStr(t) => dump_atom("cstr", t, s.span, out),
        SexpKind::List(items) => dump_seq("list", items, s.span, out),
        SexpKind::Vector(items) => dump_seq("vec", items, s.span, out),
    }
}

fn dump_atom(tag: &str, text: &str, span: Span, out: &mut String) {
    out.push('(');
    out.push_str(tag);
    dump_span(span, out);
    out.push_str(" \"");
    esc(text, out);
    out.push_str("\")");
}

fn dump_seq(tag: &str, items: &[Sexp], span: Span, out: &mut String) {
    out.push('(');
    out.push_str(tag);
    dump_span(span, out);
    for it in items {
        out.push(' ');
        dump_node(it, out);
    }
    out.push(')');
}

/// Per-byte canonical escape: `\` `"` `\n` `\t` `\r` get a backslash form, the
/// printable ASCII range passes through literally, everything else (control
/// bytes and UTF-8 continuation bytes alike) becomes `\xHH`. Byte-oriented so a
/// `(slice u8)` port reproduces it exactly without any Unicode handling.
fn esc(text: &str, out: &mut String) {
    use std::fmt::Write;
    for &b in text.as_bytes() {
        match b {
            b'\\' => out.push_str("\\\\"),
            b'"' => out.push_str("\\\""),
            b'\n' => out.push_str("\\n"),
            b'\t' => out.push_str("\\t"),
            b'\r' => out.push_str("\\r"),
            0x20..=0x7e => out.push(b as char),
            _ => write!(out, "\\x{b:02x}").unwrap(),
        }
    }
}

fn atom(s: &str) -> SexpKind {
    if let Some(kw) = s.strip_prefix(':') {
        return SexpKind::Keyword(kw.to_string());
    }
    if let Some(n) = parse_int(s) {
        return SexpKind::Int(n);
    }
    // A float literal looks like a number (digit/sign-digit start) with a `.`
    // or exponent — so type names like `f64` and symbols like `inf` stay symbols.
    if looks_numeric(s) && (s.contains('.') || s.contains('e') || s.contains('E')) {
        if let Ok(x) = s.parse::<f64>() {
            return SexpKind::Float(x);
        }
    }
    SexpKind::Sym(s.to_string())
}

/// Parse an integer literal: plain decimal (unchanged), or `0x`/`0b`/`0o`-prefixed
/// hex/binary/octal, with an optional leading sign and `_` digit separators
/// (`0xFF`, `0b1010`, `0o17`, `-0x10`, `1_000`, `0xFFFF_FFFF`). Non-numbers return
/// `None` and stay symbols. Magnitudes are read as `u64` then reinterpreted as
/// `i64`, so a full-width mask like `0xFFFFFFFFFFFFFFFF` is `-1`.
fn parse_int(s: &str) -> Option<i64> {
    if let Ok(n) = s.parse::<i64>() {
        return Some(n); // plain decimal — exact prior behavior
    }
    let (neg, body) = match s.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, s.strip_prefix('+').unwrap_or(s)),
    };
    let (radix, digits) = if let Some(h) = body.strip_prefix("0x").or(body.strip_prefix("0X")) {
        (16u32, h)
    } else if let Some(b) = body.strip_prefix("0b").or(body.strip_prefix("0B")) {
        (2, b)
    } else if let Some(o) = body.strip_prefix("0o").or(body.strip_prefix("0O")) {
        (8, o)
    } else if body.contains('_') && looks_numeric(body) {
        (10, body) // decimal with separators: `1_000`
    } else {
        return None;
    };
    let cleaned: String = digits.chars().filter(|&c| c != '_').collect();
    if cleaned.is_empty() {
        return None;
    }
    let mag = u64::from_str_radix(&cleaned, radix).ok()? as i64;
    Some(if neg { mag.wrapping_neg() } else { mag })
}

fn looks_numeric(s: &str) -> bool {
    let b = s.as_bytes();
    match b.first() {
        Some(c) if c.is_ascii_digit() => true,
        Some(b'-') | Some(b'+') | Some(b'.') => b.get(1).is_some_and(u8::is_ascii_digit),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_nested() {
        let forms = read_all("(defn f [(n :i64)] (-> :i64) (iadd n 1))", 0).unwrap();
        assert_eq!(forms.len(), 1);
        match &forms[0].kind {
            SexpKind::List(items) => {
                assert_eq!(items[0], Sexp::sym("defn"));
                assert!(matches!(items[2].kind, SexpKind::Vector(_)));
            }
            _ => panic!("expected list"),
        }
    }

    #[test]
    fn ignores_comments() {
        let forms = read_all("; hi\n42 ; trailing\n", 0).unwrap();
        assert_eq!(forms, vec![Sexp::int(42)]);
    }

    #[test]
    fn tracks_spans() {
        // `iadd` starts at byte 1 in "(iadd 1 2)".
        let src = "(iadd 1 2)";
        let forms = read_all(src, 0).unwrap();
        let list = &forms[0];
        assert_eq!(list.span, Span::new(0, 0, src.len())); // whole list
        if let SexpKind::List(items) = &list.kind {
            assert_eq!(items[0].span, Span::new(0, 1, 5)); // `iadd`
            assert_eq!(&src[items[0].span.lo as usize..items[0].span.hi as usize], "iadd");
        } else {
            panic!("expected list");
        }
    }

    #[test]
    fn unclosed_paren_has_span() {
        let err = read_all("(iadd 1 2", 0).unwrap_err();
        assert!(!err.span.is_dummy(), "unclosed paren should carry a span");
        assert!(err.msg.contains("unclosed"));
    }

    #[test]
    fn reads_hex_bin_oct_and_separators() {
        let cases = [("0xFF", 255), ("0X10", 16), ("0b1010", 10), ("0o17", 15),
                     ("-0x10", -16), ("1_000", 1000), ("0xFFFF_FFFF", 0xFFFF_FFFF),
                     ("0xFFFFFFFFFFFFFFFF", -1)];
        for (src, want) in cases {
            match super::atom(src) {
                super::SexpKind::Int(n) => assert_eq!(n, want, "{src}"),
                other => panic!("{src} -> {other:?}, wanted Int({want})"),
            }
        }
        // type names / symbols with hex-ish chars stay symbols
        for sym in ["f64", "0xfoo", "0x", "abc"] {
            assert!(matches!(super::atom(sym), super::SexpKind::Sym(_)), "{sym} should stay a symbol");
        }
    }

}
