//! S-expression reader: text → cons-cell tree.

use crate::symbols::SymbolTable;
use crate::value::*;

pub struct Reader<'a> {
    src: &'a [u8],
    pos: usize,
    sym: &'a mut SymbolTable,
}

#[derive(Debug)]
pub enum ReadError {
    UnexpectedEof,
    UnexpectedChar(char, usize),
    BadNumber(String, usize),
}

impl std::fmt::Display for ReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadError::UnexpectedEof => write!(f, "unexpected end of input"),
            ReadError::UnexpectedChar(c, p) => write!(f, "unexpected character {:?} at byte {}", c, p),
            ReadError::BadNumber(s, p) => write!(f, "bad number {:?} at byte {}", s, p),
        }
    }
}

impl<'a> Reader<'a> {
    pub fn new(src: &'a str, sym: &'a mut SymbolTable) -> Self {
        Reader { src: src.as_bytes(), pos: 0, sym }
    }

    pub fn pos(&self) -> usize { self.pos }
    pub fn at_eof(&mut self) -> bool {
        self.skip_ws_and_comments();
        self.pos >= self.src.len()
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let c = self.peek()?;
        self.pos += 1;
        Some(c)
    }

    fn skip_ws_and_comments(&mut self) {
        loop {
            while let Some(c) = self.peek() {
                if c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' || c == b',' && self.next_is_comma_only() {
                    self.pos += 1;
                } else {
                    break;
                }
            }
            if self.peek() == Some(b';') {
                while let Some(c) = self.peek() {
                    self.pos += 1;
                    if c == b'\n' { break; }
                }
                continue;
            }
            break;
        }
    }

    fn next_is_comma_only(&self) -> bool {
        // never — leave commas alone, they're unquote.
        false
    }

    pub fn read(&mut self) -> Result<u64, ReadError> {
        self.skip_ws_and_comments();
        let c = self.peek().ok_or(ReadError::UnexpectedEof)?;
        match c {
            b'(' | b'[' => { self.bump(); self.read_list(c) }
            b')' | b']' => Err(ReadError::UnexpectedChar(c as char, self.pos)),
            b'\'' => { self.bump(); self.read_quoted("quote") }
            b'`' => { self.bump(); self.read_quoted("quasiquote") }
            b',' => {
                self.bump();
                if self.peek() == Some(b'@') {
                    self.bump();
                    self.read_quoted("unquote-splicing")
                } else {
                    self.read_quoted("unquote")
                }
            }
            b'"' => { self.bump(); Err(ReadError::UnexpectedChar('"', self.pos - 1)) } // strings deferred
            b'#' => self.read_hash(),
            b'-' | b'+' => {
                // could be a sign-only atom (e.g., the operator) or a signed number
                let start = self.pos;
                self.bump();
                if let Some(d) = self.peek() {
                    if d.is_ascii_digit() {
                        return self.read_number(start);
                    }
                }
                self.pos = start;
                self.read_symbol()
            }
            d if d.is_ascii_digit() => self.read_number(self.pos),
            _ => self.read_symbol(),
        }
    }

    fn read_list(&mut self, open: u8) -> Result<u64, ReadError> {
        let close = if open == b'(' { b')' } else { b']' };
        let mut items: Vec<u64> = Vec::new();
        let mut tail = NIL;
        let mut have_dot = false;
        loop {
            self.skip_ws_and_comments();
            match self.peek() {
                None => return Err(ReadError::UnexpectedEof),
                Some(c) if c == close => { self.bump(); break; }
                Some(b'.') => {
                    // Dotted-pair: only at this level if surrounded by whitespace.
                    let pos0 = self.pos;
                    self.bump();
                    if matches!(self.peek(), Some(b' ') | Some(b'\t') | Some(b'\n') | Some(b'\r')) {
                        let after = self.read()?;
                        tail = after;
                        have_dot = true;
                        self.skip_ws_and_comments();
                        if self.peek() != Some(close) {
                            return Err(ReadError::UnexpectedChar('.', pos0));
                        }
                        self.bump();
                        break;
                    } else {
                        // not a dotted pair — treat as part of a symbol
                        self.pos = pos0;
                        items.push(self.read_symbol()?);
                    }
                }
                Some(_) => items.push(self.read()?),
            }
        }
        if !have_dot { tail = NIL; }
        let _ = have_dot;
        let mut result = tail;
        for x in items.into_iter().rev() {
            result = alloc_cons(x, result);
        }
        Ok(result)
    }

    fn read_quoted(&mut self, head: &str) -> Result<u64, ReadError> {
        let inner = self.read()?;
        let head_id = self.sym.intern(head);
        let head_val = encode_sym(head_id);
        Ok(alloc_cons(head_val, alloc_cons(inner, NIL)))
    }

    fn read_hash(&mut self) -> Result<u64, ReadError> {
        let p = self.pos;
        self.bump(); // '#'
        match self.peek() {
            Some(b't') => { self.bump(); Ok(TRUE) }
            Some(b'f') => { self.bump(); Ok(FALSE) }
            Some(c) => Err(ReadError::UnexpectedChar(c as char, p)),
            None => Err(ReadError::UnexpectedEof),
        }
    }

    fn read_number(&mut self, start: usize) -> Result<u64, ReadError> {
        // already at start, consume digits/sign/decimal/exponent
        self.pos = start;
        let begin = self.pos;
        if matches!(self.peek(), Some(b'+') | Some(b'-')) { self.bump(); }
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) { self.bump(); }
        if self.peek() == Some(b'.') {
            self.bump();
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) { self.bump(); }
        }
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            self.bump();
            if matches!(self.peek(), Some(b'+') | Some(b'-')) { self.bump(); }
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) { self.bump(); }
        }
        let txt = std::str::from_utf8(&self.src[begin..self.pos]).unwrap();
        let n: f64 = txt.parse().map_err(|_| ReadError::BadNumber(txt.to_string(), begin))?;
        Ok(encode_num(n))
    }

    fn read_symbol(&mut self) -> Result<u64, ReadError> {
        let begin = self.pos;
        while let Some(c) = self.peek() {
            if c.is_ascii_whitespace() || c == b'(' || c == b')' || c == b'[' || c == b']'
                || c == b'\'' || c == b'`' || c == b',' || c == b';' || c == b'"' {
                break;
            }
            self.bump();
        }
        if begin == self.pos {
            return Err(ReadError::UnexpectedChar(self.peek().unwrap_or(b' ') as char, self.pos));
        }
        let name = std::str::from_utf8(&self.src[begin..self.pos]).unwrap();
        if name == "nil" { return Ok(NIL); }
        let id = self.sym.intern(name);
        Ok(encode_sym(id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Engine;

    // Reader allocations go through the GC, which requires a `Host` to be
    // installed. We piggy-back on `Engine` to set that up.
    fn read_one(src: &str) -> (Engine, u64) {
        let mut e = Engine::new();
        let result = e.with_thread_state(|host| {
            let mut sym = host.sym.borrow_mut();
            let mut r = Reader::new(src, &mut sym);
            r.read().unwrap()
        });
        (e, result)
    }

    #[test]
    fn read_atoms() {
        let (_e, v) = read_one("42");
        assert!(is_number(v));
        assert_eq!(as_number(v), 42.0);
    }

    #[test]
    fn read_simple_list() {
        let (_e, v) = read_one("(+ 1 2)");
        assert!(is_cons(v));
        assert_eq!(list_len(v), 3);
    }

    #[test]
    fn read_quoted() {
        let mut e = Engine::new();
        let v = e.with_thread_state(|host| {
            let mut sym = host.sym.borrow_mut();
            let mut r = Reader::new("'(a b)", &mut sym);
            r.read().unwrap()
        });
        assert!(is_cons(v));
        let head = car(v);
        assert!(is_symbol(head));
        let sym = e.host.sym.borrow();
        assert_eq!(sym.name(as_symbol_id(head)), "quote");
    }

    #[test]
    fn read_quasiquote_unquote() {
        let (_e, v) = read_one("`(a ,b ,@c)");
        assert!(is_cons(v));
    }

    #[test]
    fn read_dotted_pair() {
        let (_e, v) = read_one("(a . b)");
        assert!(is_cons(v));
        assert!(is_symbol(car(v)));
        assert!(is_symbol(cdr(v)));
    }
}
