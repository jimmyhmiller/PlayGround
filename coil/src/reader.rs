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
    /// String literal `"..."` — used by macros (name munging); the core
    /// language has no string type.
    Str(String),
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

    /// Drop source provenance from a whole tree. Used when a form's bytes belong
    /// to a *different* source than the one diagnostics will render against
    /// (included/imported files), so we never draw a caret into the wrong file.
    pub fn unspanned(&self) -> Sexp {
        let kind = match &self.kind {
            SexpKind::List(items) => SexpKind::List(items.iter().map(Sexp::unspanned).collect()),
            SexpKind::Vector(items) => {
                SexpKind::Vector(items.iter().map(Sexp::unspanned).collect())
            }
            other => other.clone(),
        };
        Sexp::new(kind, Span::DUMMY)
    }
}

pub fn read_all(src: &str) -> Result<Vec<Sexp>, Diag> {
    let tokens = tokenize(src)?;
    let mut p = Parser { toks: tokens, pos: 0 };
    let mut out = Vec::new();
    while p.pos < p.toks.len() {
        out.push(p.parse()?);
    }
    Ok(out)
}

/// `read_all` with every span dropped — for forms read from a *different* source
/// than the one diagnostics render against (`include`/`import`). The error is
/// likewise reduced to a bare message (its span would be into the other file).
pub fn read_all_unspanned(src: &str) -> Result<Vec<Sexp>, String> {
    let forms = read_all(src).map_err(|d| d.msg)?;
    Ok(forms.iter().map(Sexp::unspanned).collect())
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
    Atom(String),
}

fn tokenize(src: &str) -> Result<Vec<Tok>, Diag> {
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
                toks.push(Tok { kind: TokKind::Open(c), span: Span::new(i, i + 1) });
                chars.next();
            }
            ')' | ']' => {
                toks.push(Tok { kind: TokKind::Close(c), span: Span::new(i, i + 1) });
                chars.next();
            }
            '\'' => {
                toks.push(Tok { kind: TokKind::Prefix("quote"), span: Span::new(i, i + 1) });
                chars.next();
            }
            '`' => {
                toks.push(Tok { kind: TokKind::Prefix("quasiquote"), span: Span::new(i, i + 1) });
                chars.next();
            }
            '~' => {
                chars.next();
                if matches!(chars.peek(), Some(&(_, '@'))) {
                    let (j, _) = chars.next().unwrap();
                    toks.push(Tok {
                        kind: TokKind::Prefix("unquote-splicing"),
                        span: Span::new(i, j + 1),
                    });
                } else {
                    toks.push(Tok { kind: TokKind::Prefix("unquote"), span: Span::new(i, i + 1) });
                }
            }
            '"' => {
                chars.next();
                let mut s = String::new();
                let end;
                loop {
                    match chars.next() {
                        None => {
                            return Err(Diag::at(
                                Span::new(i, src.len()),
                                "unterminated string literal",
                            ))
                        }
                        Some((j, '"')) => {
                            end = j + 1;
                            break;
                        }
                        Some((_, '\\')) => match chars.next() {
                            Some((_, 'n')) => s.push('\n'),
                            Some((_, 't')) => s.push('\t'),
                            Some((_, '"')) => s.push('"'),
                            Some((_, '\\')) => s.push('\\'),
                            Some((_, c)) => s.push(c),
                            None => {
                                return Err(Diag::at(
                                    Span::new(i, src.len()),
                                    "unterminated string escape",
                                ))
                            }
                        },
                        Some((_, c)) => s.push(c),
                    }
                }
                toks.push(Tok { kind: TokKind::Str(s), span: Span::new(i, end) });
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
                toks.push(Tok { kind: TokKind::Atom(s), span: Span::new(start, end) });
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

fn atom(s: &str) -> SexpKind {
    if let Some(kw) = s.strip_prefix(':') {
        return SexpKind::Keyword(kw.to_string());
    }
    if let Ok(n) = s.parse::<i64>() {
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
        let forms = read_all("(defn f [(n :i64)] (-> :i64) (iadd n 1))").unwrap();
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
        let forms = read_all("; hi\n42 ; trailing\n").unwrap();
        assert_eq!(forms, vec![Sexp::int(42)]);
    }

    #[test]
    fn tracks_spans() {
        // `iadd` starts at byte 1 in "(iadd 1 2)".
        let src = "(iadd 1 2)";
        let forms = read_all(src).unwrap();
        let list = &forms[0];
        assert_eq!(list.span, Span::new(0, src.len())); // whole list
        if let SexpKind::List(items) = &list.kind {
            assert_eq!(items[0].span, Span::new(1, 5)); // `iadd`
            assert_eq!(&src[items[0].span.lo as usize..items[0].span.hi as usize], "iadd");
        } else {
            panic!("expected list");
        }
    }

    #[test]
    fn unclosed_paren_has_span() {
        let err = read_all("(iadd 1 2").unwrap_err();
        assert!(!err.span.is_dummy(), "unclosed paren should carry a span");
        assert!(err.msg.contains("unclosed"));
    }
}
