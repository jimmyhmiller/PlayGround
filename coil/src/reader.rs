//! The reader: source text → `Val` s-expressions.
//!
//! Implements `mlir-lisp-design/SPEC.md §1`. Recursive-descent over chars; no
//! external dependencies. Reader macros (`' ` `` ` `` `~` `~@`) desugar to list
//! forms; `!…`/`#…` become `TypeLit`/`AttrLit`; sigil symbols (`@ ^ %`) are kept
//! verbatim as symbols for the expander to interpret.

use crate::value::Val;
use std::rc::Rc;

#[derive(Debug, Clone, PartialEq)]
pub struct ReadError {
    pub msg: String,
    pub pos: usize,
}

impl std::fmt::Display for ReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "read error at byte {}: {}", self.pos, self.msg)
    }
}
impl std::error::Error for ReadError {}

pub struct Reader {
    chars: Vec<char>,
    pos: usize,
}

/// Read every top-level form from `src`.
pub fn read_all(src: &str) -> Result<Vec<Val>, ReadError> {
    Reader::new(src).read_all()
}

/// Read exactly one form; error if there is trailing junk.
pub fn read_one(src: &str) -> Result<Val, ReadError> {
    let mut r = Reader::new(src);
    let form = r
        .read_form()?
        .ok_or_else(|| r.err("expected a form, found end of input"))?;
    r.skip_ws()?;
    if r.peek().is_some() {
        return Err(r.err("unexpected trailing input after form"));
    }
    Ok(form)
}

impl Reader {
    pub fn new(src: &str) -> Self {
        Reader {
            chars: src.chars().collect(),
            pos: 0,
        }
    }

    pub fn read_all(&mut self) -> Result<Vec<Val>, ReadError> {
        let mut forms = Vec::new();
        while let Some(form) = self.read_form()? {
            forms.push(form);
        }
        Ok(forms)
    }

    // --- cursor helpers ---------------------------------------------------

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }
    fn peek2(&self) -> Option<char> {
        self.chars.get(self.pos + 1).copied()
    }
    fn bump(&mut self) -> Option<char> {
        let c = self.peek();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }
    fn err(&self, msg: impl Into<String>) -> ReadError {
        ReadError {
            msg: msg.into(),
            pos: self.pos,
        }
    }

    fn is_delim(c: char) -> bool {
        c.is_whitespace()
            || matches!(c, '(' | ')' | '[' | ']' | '{' | '}' | '"' | ';' | ',' | '\'' | '`' | '~')
    }

    // --- whitespace & comments -------------------------------------------

    /// Skip whitespace, commas, `;` line comments, `#| |#` block comments, and
    /// `#_` datum comments (which discard the following form).
    fn skip_ws(&mut self) -> Result<(), ReadError> {
        loop {
            match self.peek() {
                Some(c) if c.is_whitespace() || c == ',' => {
                    self.bump();
                }
                Some(';') => {
                    while let Some(c) = self.peek() {
                        self.bump();
                        if c == '\n' {
                            break;
                        }
                    }
                }
                Some('#') if self.peek2() == Some('|') => {
                    self.skip_block_comment()?;
                }
                Some('#') if self.peek2() == Some('_') => {
                    self.bump();
                    self.bump();
                    // discard one full form
                    self.read_form()?
                        .ok_or_else(|| self.err("`#_` with no following form"))?;
                }
                _ => return Ok(()),
            }
        }
    }

    fn skip_block_comment(&mut self) -> Result<(), ReadError> {
        // consume opening "#|"
        self.bump();
        self.bump();
        let mut depth = 1usize;
        while depth > 0 {
            match (self.bump(), self.peek()) {
                (Some('#'), Some('|')) => {
                    self.bump();
                    depth += 1;
                }
                (Some('|'), Some('#')) => {
                    self.bump();
                    depth -= 1;
                }
                (Some(_), _) => {}
                (None, _) => return Err(self.err("unterminated block comment")),
            }
        }
        Ok(())
    }

    // --- forms ------------------------------------------------------------

    /// Read one form, or `None` at end of input.
    fn read_form(&mut self) -> Result<Option<Val>, ReadError> {
        self.skip_ws()?;
        let c = match self.peek() {
            None => return Ok(None),
            Some(c) => c,
        };
        let val = match c {
            '(' => self.read_seq('(', ')')?,
            '[' => self.read_seq('[', ']')?,
            '{' => self.read_map()?,
            ')' | ']' | '}' => return Err(self.err(format!("unexpected `{c}`"))),
            '"' => self.read_string()?,
            '\'' => self.read_quote("quote")?,
            '`' => self.read_quote("quasiquote")?,
            '~' => {
                self.bump();
                if self.peek() == Some('@') {
                    self.bump();
                    Val::list(vec![Val::sym("unquote-splicing"), self.read_required()?])
                } else {
                    Val::list(vec![Val::sym("unquote"), self.read_required()?])
                }
            }
            '!' => self.read_sigil_lit('!')?,
            // `#` here is not a comment (skip_ws handled those). It's an attr
            // literal unless it stands alone as the bare symbol `#`.
            '#' if self.peek2().map_or(true, |n| n.is_whitespace() || matches!(n, ')' | ']' | '}')) => {
                self.bump();
                Val::sym("#")
            }
            '#' => self.read_sigil_lit('#')?,
            _ => self.read_atom()?,
        };
        Ok(Some(val))
    }

    fn read_required(&mut self) -> Result<Val, ReadError> {
        self.read_form()?
            .ok_or_else(|| self.err("expected a form, found end of input"))
    }

    fn read_quote(&mut self, sym: &str) -> Result<Val, ReadError> {
        self.bump(); // consume the quote char
        Ok(Val::list(vec![Val::sym(sym), self.read_required()?]))
    }

    fn read_seq(&mut self, open: char, close: char) -> Result<Val, ReadError> {
        self.bump(); // consume open
        let mut items = Vec::new();
        loop {
            self.skip_ws()?;
            match self.peek() {
                None => return Err(self.err(format!("unterminated `{open}` — expected `{close}`"))),
                Some(c) if c == close => {
                    self.bump();
                    break;
                }
                Some(c) if matches!(c, ')' | ']' | '}') => {
                    return Err(self.err(format!("mismatched `{c}`, expected `{close}`")));
                }
                _ => items.push(self.read_required()?),
            }
        }
        Ok(if open == '(' {
            Val::list(items)
        } else {
            Val::vector(items)
        })
    }

    fn read_map(&mut self) -> Result<Val, ReadError> {
        self.bump(); // consume '{'
        let mut items = Vec::new();
        loop {
            self.skip_ws()?;
            match self.peek() {
                None => return Err(self.err("unterminated `{` — expected `}`")),
                Some('}') => {
                    self.bump();
                    break;
                }
                Some(c) if matches!(c, ')' | ']') => {
                    return Err(self.err(format!("mismatched `{c}`, expected `}}`")));
                }
                _ => items.push(self.read_required()?),
            }
        }
        if items.len() % 2 != 0 {
            return Err(self.err("map literal has an odd number of forms"));
        }
        let pairs = items
            .chunks_exact(2)
            .map(|p| (p[0].clone(), p[1].clone()))
            .collect();
        Ok(Val::map(pairs))
    }

    fn read_string(&mut self) -> Result<Val, ReadError> {
        self.bump(); // consume opening quote
        let mut s = String::new();
        loop {
            match self.bump() {
                None => return Err(self.err("unterminated string literal")),
                Some('"') => break,
                Some('\\') => {
                    let e = self.bump().ok_or_else(|| self.err("unterminated escape"))?;
                    s.push(match e {
                        'n' => '\n',
                        't' => '\t',
                        'r' => '\r',
                        '0' => '\0',
                        '\\' => '\\',
                        '"' => '"',
                        other => return Err(self.err(format!("unknown escape `\\{other}`"))),
                    });
                }
                Some(c) => s.push(c),
            }
        }
        Ok(Val::Str(Rc::from(s)))
    }

    /// Read a `!…` or `#…` literal: a balanced token where `< ( [` raise depth
    /// so spaces inside `!llvm.struct<(i64, i64)>` don't terminate it.
    fn read_sigil_lit(&mut self, sigil: char) -> Result<Val, ReadError> {
        self.bump(); // consume sigil
        let mut s = String::new();
        let mut depth: i32 = 0;
        while let Some(c) = self.peek() {
            if depth == 0 && (c.is_whitespace() || matches!(c, ')' | ']' | '}' | ';' | ',')) {
                break;
            }
            match c {
                '<' | '(' | '[' => depth += 1,
                '>' | ')' | ']' => depth -= 1,
                _ => {}
            }
            s.push(c);
            self.bump();
        }
        if s.is_empty() {
            return Err(self.err(format!("empty `{sigil}` literal")));
        }
        Ok(if sigil == '!' {
            Val::TypeLit(Rc::from(s))
        } else {
            Val::AttrLit(Rc::from(s))
        })
    }

    /// Read an atom: number, bool, nil, keyword, or symbol.
    fn read_atom(&mut self) -> Result<Val, ReadError> {
        let start = self.pos;
        let mut tok = String::new();
        while let Some(c) = self.peek() {
            if Self::is_delim(c) {
                break;
            }
            tok.push(c);
            self.bump();
        }
        if tok.is_empty() {
            return Err(ReadError {
                msg: format!("unexpected character `{}`", self.peek().unwrap_or(' ')),
                pos: start,
            });
        }
        Ok(classify_atom(&tok))
    }
}

fn classify_atom(tok: &str) -> Val {
    match tok {
        "true" => return Val::Bool(true),
        "false" => return Val::Bool(false),
        "nil" => return Val::Nil,
        _ => {}
    }
    if let Some(rest) = tok.strip_prefix(':') {
        if !rest.is_empty() {
            return Val::keyword(rest);
        }
    }
    if let Some(n) = parse_number(tok) {
        return n;
    }
    Val::sym(tok)
}

/// Parse a numeric literal, or `None` if the token is a symbol.
fn parse_number(tok: &str) -> Option<Val> {
    let bytes = tok.as_bytes();
    let mut i = 0;
    if matches!(bytes[0], b'+' | b'-') {
        i = 1;
    }
    let rest = &tok[i..];
    let rb = rest.as_bytes();
    // Must look like a number: digit, or `.digit`.
    let looks_numeric = match rb.first() {
        Some(c) if c.is_ascii_digit() => true,
        Some(b'.') => rb.get(1).map_or(false, |c| c.is_ascii_digit()),
        _ => false,
    };
    if !looks_numeric {
        return None;
    }
    // Hex integer.
    if let Some(hex) = rest.strip_prefix("0x").or_else(|| rest.strip_prefix("0X")) {
        let v = i64::from_str_radix(hex, 16).ok()?;
        return Some(Val::Int(if tok.starts_with('-') { -v } else { v }));
    }
    if let Ok(n) = tok.parse::<i64>() {
        return Some(Val::Int(n));
    }
    if let Ok(f) = tok.parse::<f64>() {
        return Some(Val::Float(f));
    }
    None
}
