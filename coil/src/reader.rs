//! S-expression reader.
//!
//! Surface is intentionally tiny: lists `( )`, vectors `[ ]`, integers,
//! keywords `:kw`, and symbols. `;` starts a line comment. The prettier
//! surface sugar from the design doc (`@rdi`, `#{...}`) is deferred to the
//! macro layer; semantics first.

#[derive(Debug, Clone, PartialEq)]
pub enum Sexp {
    Int(i64),
    Sym(String),
    /// Keyword `:foo` stored without the leading colon.
    Keyword(String),
    /// String literal `"..."` — used by macros (name munging); the core
    /// language has no string type.
    Str(String),
    List(Vec<Sexp>),
    Vector(Vec<Sexp>),
}

pub fn read_all(src: &str) -> Result<Vec<Sexp>, String> {
    let tokens = tokenize(src)?;
    let mut p = Parser { toks: tokens, pos: 0 };
    let mut out = Vec::new();
    while p.pos < p.toks.len() {
        out.push(p.parse()?);
    }
    Ok(out)
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Open(char),  // ( or [
    Close(char), // ) or ]
    /// A reader-macro prefix: `'`→quote, `` ` ``→quasiquote, `~`→unquote,
    /// `~@`→unquote-splicing. The next form is wrapped in `(<sym> form)`.
    Prefix(&'static str),
    Str(String),
    Atom(String),
}

fn tokenize(src: &str) -> Result<Vec<Tok>, String> {
    let mut toks = Vec::new();
    let mut chars = src.chars().peekable();
    while let Some(&c) = chars.peek() {
        match c {
            c if c.is_whitespace() => {
                chars.next();
            }
            ';' => {
                // comment to end of line
                while let Some(&c) = chars.peek() {
                    chars.next();
                    if c == '\n' {
                        break;
                    }
                }
            }
            '(' | '[' => {
                toks.push(Tok::Open(c));
                chars.next();
            }
            ')' | ']' => {
                toks.push(Tok::Close(c));
                chars.next();
            }
            '\'' => {
                toks.push(Tok::Prefix("quote"));
                chars.next();
            }
            '`' => {
                toks.push(Tok::Prefix("quasiquote"));
                chars.next();
            }
            '~' => {
                chars.next();
                if chars.peek() == Some(&'@') {
                    chars.next();
                    toks.push(Tok::Prefix("unquote-splicing"));
                } else {
                    toks.push(Tok::Prefix("unquote"));
                }
            }
            '"' => {
                chars.next();
                let mut s = String::new();
                loop {
                    match chars.next() {
                        None => return Err("unterminated string literal".to_string()),
                        Some('"') => break,
                        Some('\\') => match chars.next() {
                            Some('n') => s.push('\n'),
                            Some('t') => s.push('\t'),
                            Some('"') => s.push('"'),
                            Some('\\') => s.push('\\'),
                            Some(c) => s.push(c),
                            None => return Err("unterminated string escape".to_string()),
                        },
                        Some(c) => s.push(c),
                    }
                }
                toks.push(Tok::Str(s));
            }
            _ => {
                let mut s = String::new();
                while let Some(&c) = chars.peek() {
                    if c.is_whitespace()
                        || matches!(c, '(' | ')' | '[' | ']' | ';' | '\'' | '`' | '~' | '"')
                    {
                        break;
                    }
                    s.push(c);
                    chars.next();
                }
                toks.push(Tok::Atom(s));
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
    fn parse(&mut self) -> Result<Sexp, String> {
        let tok = self
            .toks
            .get(self.pos)
            .cloned()
            .ok_or_else(|| "unexpected end of input".to_string())?;
        self.pos += 1;
        match tok {
            Tok::Open(open) => {
                let close = if open == '(' { ')' } else { ']' };
                let mut items = Vec::new();
                loop {
                    match self.toks.get(self.pos) {
                        None => return Err(format!("unclosed '{open}'")),
                        Some(Tok::Close(c)) => {
                            if *c != close {
                                return Err(format!("mismatched delimiters: '{open}' .. '{c}'"));
                            }
                            self.pos += 1;
                            break;
                        }
                        Some(_) => items.push(self.parse()?),
                    }
                }
                Ok(if open == '(' {
                    Sexp::List(items)
                } else {
                    Sexp::Vector(items)
                })
            }
            Tok::Prefix(sym) => {
                let inner = self.parse()?;
                Ok(Sexp::List(vec![Sexp::Sym(sym.to_string()), inner]))
            }
            Tok::Str(s) => Ok(Sexp::Str(s)),
            Tok::Close(c) => Err(format!("unexpected '{c}'")),
            Tok::Atom(s) => Ok(atom(&s)),
        }
    }
}

impl std::fmt::Display for Sexp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Sexp::Int(n) => write!(f, "{n}"),
            Sexp::Sym(s) => write!(f, "{s}"),
            Sexp::Keyword(k) => write!(f, ":{k}"),
            Sexp::Str(s) => write!(f, "{s:?}"),
            Sexp::List(items) => write_seq(f, items, '(', ')'),
            Sexp::Vector(items) => write_seq(f, items, '[', ']'),
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

fn atom(s: &str) -> Sexp {
    if let Some(kw) = s.strip_prefix(':') {
        return Sexp::Keyword(kw.to_string());
    }
    if let Ok(n) = s.parse::<i64>() {
        return Sexp::Int(n);
    }
    Sexp::Sym(s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_nested() {
        let forms = read_all("(defn f [(n :i64)] (-> :i64) (iadd n 1))").unwrap();
        assert_eq!(forms.len(), 1);
        match &forms[0] {
            Sexp::List(items) => {
                assert_eq!(items[0], Sexp::Sym("defn".into()));
                assert!(matches!(items[2], Sexp::Vector(_)));
            }
            _ => panic!("expected list"),
        }
    }

    #[test]
    fn ignores_comments() {
        let forms = read_all("; hi\n42 ; trailing\n").unwrap();
        assert_eq!(forms, vec![Sexp::Int(42)]);
    }
}
