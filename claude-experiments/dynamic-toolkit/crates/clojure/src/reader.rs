//! Reader: text → values.
//!
//! Bootstrap design: the Rust side reads enough of Clojure syntax to
//! load `core.clj`, after which the in-language reader takes over.
//!
//! Supports: integer/float literals, `nil`/`true`/`false`, symbols,
//! strings (`"..."`), keywords (`:foo`), char literals (`\c`), lists
//! (`(...)`), vectors (`[...]`), maps (`{k v ...}`), sets (`#{...}`),
//! and the reader macros `'`, `` ` ``, `~`, `~@` which expand to
//! `(quote …)`, `(quasiquote …)`, `(unquote …)`, `(unquote-splicing …)`.

use dynobj::roots::with_scope;

use crate::collections::{alloc_keyword, alloc_set, alloc_string, alloc_vector};
use crate::namespace::alloc_map_pairs;
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
    BadEscape(char, usize),
    OddMapEntries(usize),
}

impl std::fmt::Display for ReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadError::UnexpectedEof => write!(f, "unexpected end of input"),
            ReadError::UnexpectedChar(c, p) => {
                write!(f, "unexpected character {:?} at byte {}", c, p)
            }
            ReadError::BadNumber(s, p) => write!(f, "bad number {:?} at byte {}", s, p),
            ReadError::BadEscape(c, p) => {
                write!(f, "bad escape \\{} at byte {}", c, p)
            }
            ReadError::OddMapEntries(p) => write!(f, "map literal at byte {} has an odd number of forms", p),
        }
    }
}

impl<'a> Reader<'a> {
    pub fn new(src: &'a str, sym: &'a mut SymbolTable) -> Self {
        Reader {
            src: src.as_bytes(),
            pos: 0,
            sym,
        }
    }

    pub fn at_eof(&mut self) -> bool {
        self.skip_ws_and_comments();
        self.pos >= self.src.len()
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn peek_at(&self, off: usize) -> Option<u8> {
        self.src.get(self.pos + off).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let c = self.peek()?;
        self.pos += 1;
        Some(c)
    }

    fn skip_ws_and_comments(&mut self) {
        loop {
            while let Some(c) = self.peek() {
                if matches!(c, b' ' | b'\t' | b'\n' | b'\r' | b',') {
                    self.pos += 1;
                } else {
                    break;
                }
            }
            if self.peek() == Some(b';') {
                while let Some(c) = self.peek() {
                    self.pos += 1;
                    if c == b'\n' {
                        break;
                    }
                }
                continue;
            }
            break;
        }
    }

    pub fn read(&mut self) -> Result<u64, ReadError> {
        self.skip_ws_and_comments();
        let c = self.peek().ok_or(ReadError::UnexpectedEof)?;
        match c {
            b'(' => {
                self.bump();
                self.read_list_to_terminator(b')')
            }
            b'[' => {
                self.bump();
                self.read_vector(b']')
            }
            b'{' => {
                self.bump();
                self.read_map(b'}')
            }
            b'#' => {
                // #{ … }  — set literal
                self.bump();
                match self.peek() {
                    Some(b'{') => {
                        self.bump();
                        self.read_set(b'}')
                    }
                    _ => Err(ReadError::UnexpectedChar('#', self.pos.saturating_sub(1))),
                }
            }
            b')' | b']' | b'}' => Err(ReadError::UnexpectedChar(c as char, self.pos)),
            b'"' => {
                self.bump();
                self.read_string()
            }
            b':' => {
                self.bump();
                self.read_keyword()
            }
            b'\\' => {
                self.bump();
                self.read_char()
            }
            b'\'' => {
                self.bump();
                let inner = self.read()?;
                Ok(self.wrap_with_head("quote", inner))
            }
            b'`' => {
                self.bump();
                let inner = self.read()?;
                Ok(self.wrap_with_head("quasiquote", inner))
            }
            b'~' => {
                self.bump();
                let head = if self.peek() == Some(b'@') {
                    self.bump();
                    "unquote-splicing"
                } else {
                    "unquote"
                };
                let inner = self.read()?;
                Ok(self.wrap_with_head(head, inner))
            }
            b'@' => {
                self.bump();
                let inner = self.read()?;
                Ok(self.wrap_with_head("deref", inner))
            }
            b'^' => {
                // Metadata: `^X form` — for now, read X and discard. We
                // do not yet support metadata; just attach nothing.
                self.bump();
                let _meta = self.read()?;
                self.read()
            }
            b'-' | b'+' => {
                let start = self.pos;
                self.bump();
                if let Some(d) = self.peek() {
                    if d.is_ascii_digit() {
                        return self.read_number(start);
                    }
                }
                self.pos = start;
                self.read_atom()
            }
            d if d.is_ascii_digit() => self.read_number(self.pos),
            _ => self.read_atom(),
        }
    }

    /// Read items until `close`, returning a Clojure list.
    fn read_list_to_terminator(&mut self, close: u8) -> Result<u64, ReadError> {
        let items = self.read_until(close)?;
        Ok(build_list(&items))
    }

    fn read_vector(&mut self, close: u8) -> Result<u64, ReadError> {
        let items = self.read_until(close)?;
        let v = with_scope(2 + items.len(), |scope| {
            // Root each item once before alloc; items came back as raw
            // bits and could move on intermediate allocs.
            let mut rooted = Vec::with_capacity(items.len());
            for x in &items {
                rooted.push(scope.root::<NanBoxTag>(*x));
            }
            let raw_items: Vec<u64> = rooted.iter().map(|r| r.get()).collect();
            alloc_vector(scope, &raw_items).get()
        });
        Ok(v)
    }

    fn read_map(&mut self, close: u8) -> Result<u64, ReadError> {
        let start = self.pos;
        let items = self.read_until(close)?;
        if items.len() % 2 != 0 {
            return Err(ReadError::OddMapEntries(start));
        }
        let pairs: Vec<(u64, u64)> = items.chunks(2).map(|c| (c[0], c[1])).collect();
        let v = with_scope(2 + items.len(), |scope| {
            // Root all items first.
            let mut rooted: Vec<_> = items.iter().map(|x| scope.root::<NanBoxTag>(*x)).collect();
            let pairs_r: Vec<(u64, u64)> = rooted
                .chunks_mut(2)
                .map(|c| (c[0].get(), c[1].get()))
                .collect();
            let _ = pairs;
            alloc_map_pairs(scope, &pairs_r).get()
        });
        Ok(v)
    }

    fn read_set(&mut self, close: u8) -> Result<u64, ReadError> {
        let items = self.read_until(close)?;
        let v = with_scope(2 + items.len(), |scope| {
            let rooted: Vec<_> = items.iter().map(|x| scope.root::<NanBoxTag>(*x)).collect();
            let raw_items: Vec<u64> = rooted.iter().map(|r| r.get()).collect();
            alloc_set(scope, &raw_items).get()
        });
        Ok(v)
    }

    /// Read items until the matching closer; consume the closer.
    fn read_until(&mut self, close: u8) -> Result<Vec<u64>, ReadError> {
        let mut items: Vec<u64> = Vec::new();
        loop {
            self.skip_ws_and_comments();
            match self.peek() {
                None => return Err(ReadError::UnexpectedEof),
                Some(c) if c == close => {
                    self.bump();
                    break;
                }
                Some(_) => items.push(self.read()?),
            }
        }
        Ok(items)
    }

    fn read_string(&mut self) -> Result<u64, ReadError> {
        let mut buf: Vec<u8> = Vec::new();
        loop {
            let c = self.bump().ok_or(ReadError::UnexpectedEof)?;
            match c {
                b'"' => break,
                b'\\' => {
                    let e = self.bump().ok_or(ReadError::UnexpectedEof)?;
                    match e {
                        b'"' => buf.push(b'"'),
                        b'\\' => buf.push(b'\\'),
                        b'n' => buf.push(b'\n'),
                        b't' => buf.push(b'\t'),
                        b'r' => buf.push(b'\r'),
                        b'0' => buf.push(0),
                        other => return Err(ReadError::BadEscape(other as char, self.pos - 1)),
                    }
                }
                other => buf.push(other),
            }
        }
        let v = with_scope(2, |scope| alloc_string(scope, &buf).get());
        Ok(v)
    }

    fn read_keyword(&mut self) -> Result<u64, ReadError> {
        let begin = self.pos;
        while let Some(c) = self.peek() {
            if is_token_break(c) {
                break;
            }
            self.bump();
        }
        if begin == self.pos {
            return Err(ReadError::UnexpectedChar(':', begin));
        }
        let name = std::str::from_utf8(&self.src[begin..self.pos]).unwrap();
        let id = self.sym.intern(name);
        let v = with_scope(2, |scope| alloc_keyword(scope, id).get());
        Ok(v)
    }

    fn read_char(&mut self) -> Result<u64, ReadError> {
        // `\space`, `\newline`, `\tab`, or single-character forms like `\a`.
        let begin = self.pos;
        while let Some(c) = self.peek() {
            if is_token_break(c) {
                break;
            }
            self.bump();
        }
        let token = std::str::from_utf8(&self.src[begin..self.pos]).unwrap();
        let ch: u8 = match token {
            "space" => b' ',
            "newline" => b'\n',
            "tab" => b'\t',
            "return" => b'\r',
            "" => return Err(ReadError::UnexpectedChar('\\', begin)),
            t if t.len() == 1 => t.as_bytes()[0],
            other => return Err(ReadError::BadEscape(other.chars().next().unwrap_or('?'), begin)),
        };
        // Encode as a 1-byte string for now (no dedicated char tag yet).
        let v = with_scope(2, |scope| alloc_string(scope, &[ch]).get());
        Ok(v)
    }

    fn read_number(&mut self, start: usize) -> Result<u64, ReadError> {
        self.pos = start;
        let begin = self.pos;
        let mut is_float = false;
        if matches!(self.peek(), Some(b'+') | Some(b'-')) {
            self.bump();
        }
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.bump();
        }
        if self.peek() == Some(b'.') {
            // Treat the dot as part of the number ONLY if it's followed
            // by a digit. `(.method obj)` reads `.method` as a symbol,
            // so a bare `.` starting a token shouldn't make us greedy.
            if matches!(self.peek_at(1), Some(c) if c.is_ascii_digit()) {
                is_float = true;
                self.bump();
                while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                    self.bump();
                }
            }
        }
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            is_float = true;
            self.bump();
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.bump();
            }
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.bump();
            }
        }
        let txt = std::str::from_utf8(&self.src[begin..self.pos]).unwrap();
        if is_float {
            let n: f64 = txt
                .parse()
                .map_err(|_| ReadError::BadNumber(txt.to_string(), begin))?;
            Ok(encode_num(n))
        } else {
            let n: i64 = txt
                .parse()
                .map_err(|_| ReadError::BadNumber(txt.to_string(), begin))?;
            Ok(encode_int(n))
        }
    }

    fn read_atom(&mut self) -> Result<u64, ReadError> {
        let begin = self.pos;
        while let Some(c) = self.peek() {
            if is_token_break(c) {
                break;
            }
            self.bump();
        }
        if begin == self.pos {
            return Err(ReadError::UnexpectedChar(
                self.peek().unwrap_or(b' ') as char,
                self.pos,
            ));
        }
        let name = std::str::from_utf8(&self.src[begin..self.pos]).unwrap();
        match name {
            "nil" => Ok(NIL),
            "true" => Ok(TRUE),
            "false" => Ok(FALSE),
            _ => {
                let id = self.sym.intern(name);
                Ok(encode_sym_id(id))
            }
        }
    }

    /// Build `(<head> inner)` as a Clojure list. Used by the quote /
    /// quasiquote / unquote / deref reader macros.
    fn wrap_with_head(&mut self, head: &str, inner: u64) -> u64 {
        let head_id = self.sym.intern(head);
        let head_val = encode_sym_id(head_id);
        with_scope(4, |scope| {
            let inner_r = scope.root::<NanBoxTag>(inner);
            let head_r = scope.root::<NanBoxTag>(head_val);
            let tail = alloc_list_cell_from_raw(scope, inner_r.get(), NIL);
            alloc_list_cell(scope, &head_r, &tail).get()
        })
    }
}

fn is_token_break(c: u8) -> bool {
    c.is_ascii_whitespace()
        || matches!(
            c,
            b'(' | b')' | b'[' | b']' | b'{' | b'}' | b'\'' | b'`' | b','
                | b';' | b'"' | b'^' | b'@' | b'~'
        )
}

/// Build a Clojure list from a slice of NanBox-encoded items.
fn build_list(items: &[u64]) -> u64 {
    with_scope(2 + items.len(), |scope| {
        let acc = scope.root::<NanBoxTag>(NIL);
        for x in items.iter().rev() {
            let new_bits = with_scope(3, |inner| alloc_list_cell_from_raw(inner, *x, acc.get()).get());
            acc.set(new_bits);
        }
        acc.get()
    })
}
