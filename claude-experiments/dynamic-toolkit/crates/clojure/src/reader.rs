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

use dynobj::roots::{RootSet, with_scope};

use crate::collections::{alloc_set, alloc_string, alloc_vector};
use crate::host::with_host;
use crate::namespace::alloc_map_pairs;
use crate::symbols::SymbolTable;
use crate::value::*;

pub struct Reader<'a> {
    src: &'a [u8],
    pos: usize,
    sym: &'a SymbolTable,
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
            ReadError::OddMapEntries(p) => {
                write!(f, "map literal at byte {} has an odd number of forms", p)
            }
        }
    }
}

impl<'a> Reader<'a> {
    pub fn new(src: &'a str, sym: &'a SymbolTable) -> Self {
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
                // Metadata: `^X form`. Silently dropping the meta would
                // produce subtly wrong programs (`^:dynamic`, type
                // hints, `^{...}` maps all become no-ops). Refuse until
                // IObj-meta attachment is wired up.
                unimplemented!(
                    "reader: `^...` metadata not yet supported \
                     (see TODO.md). Refusing to silently discard meta."
                );
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

    /// Read items until `close`, returning a Clojure list. Items are
    /// rooted across the read loop via `read_collection`, so a GC
    /// fired during a later read can't invalidate earlier ones.
    fn read_list_to_terminator(&mut self, close: u8) -> Result<u64, ReadError> {
        self.read_collection(close, |scope, items| {
            // Build the list right-to-left from the rooted items.
            let acc = scope.root::<NanBoxTag>(NIL);
            for i in (0..items.len()).rev() {
                let new_bits = with_scope(3, |inner| {
                    alloc_list_cell_from_raw(inner, items.get(i), acc.get()).get()
                });
                acc.set(new_bits);
            }
            acc.get()
        })
    }

    fn read_vector(&mut self, close: u8) -> Result<u64, ReadError> {
        self.read_collection(close, |scope, items| {
            let raw: Vec<u64> = (0..items.len()).map(|i| items.get(i)).collect();
            alloc_vector(scope, &raw).get()
        })
    }

    fn read_map(&mut self, close: u8) -> Result<u64, ReadError> {
        let start = self.pos;
        self.read_collection(close, move |scope, items| {
            if items.len() % 2 != 0 {
                panic!("map literal at byte {} has an odd number of forms", start);
            }
            let pairs: Vec<(u64, u64)> = (0..items.len() / 2)
                .map(|i| (items.get(2 * i), items.get(2 * i + 1)))
                .collect();
            alloc_map_pairs(scope, &pairs).get()
        })
    }

    fn read_set(&mut self, close: u8) -> Result<u64, ReadError> {
        self.read_collection(close, |scope, items| {
            let raw: Vec<u64> = (0..items.len()).map(|i| items.get(i)).collect();
            alloc_set(scope, &raw).get()
        })
    }

    /// Read items until the matching closer (consuming it), keeping
    /// every parsed value GC-rooted across subsequent reads, then
    /// invoke `f` with a fresh `RootScope` and the live items still
    /// rooted. This is the only safe shape for accumulating heap
    /// values across a series of allocating reads — a plain
    /// `Vec<u64>` would leave earlier items unrooted while later
    /// reads allocated, and a relocating GC would invalidate them.
    ///
    /// The `RootSet` is pinned to a stable address (we only take
    /// `&` of it) and registered as an `extra_root_source` for the
    /// duration of this call.
    fn read_collection<F>(&mut self, close: u8, f: F) -> Result<u64, ReadError>
    where
        F: FnOnce(&dynobj::roots::RootScope<'_>, &RootSet) -> u64,
    {
        let mut items = RootSet::new();
        let host_gc = with_host(|h| h.gc);
        let items_src: *const dyn dynobj::RootSource = &items;
        let _root_guard = unsafe { (*host_gc).push_extra_root_source(items_src) };

        loop {
            self.skip_ws_and_comments();
            match self.peek() {
                None => return Err(ReadError::UnexpectedEof),
                Some(c) if c == close => {
                    self.bump();
                    break;
                }
                Some(_) => {
                    let v = self.read()?;
                    items.add(v);
                }
            }
        }

        let result = with_scope(items.len() + 8, |scope| f(scope, &items));
        Ok(result)
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
        // Use the host's keyword intern table so identical literals
        // share the same heap object — `(= :foo :foo)` then works
        // under bitwise `clj_eq`.
        let v = with_host(|h| h.intern_keyword(id));
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
        // Don't fake characters as 1-byte strings: that makes `(= \a "a")`
        // silently true, breaks `char?`, and truncates multi-byte chars.
        // Wait until we have a real char tag (toolkit task #6).
        unimplemented!(
            "reader: character literal `\\{token}` not yet supported \
             (no Char tag yet — see TODO.md / toolkit task #6)."
        );
    }

    fn read_number(&mut self, start: usize) -> Result<u64, ReadError> {
        self.pos = start;
        let begin = self.pos;
        let mut is_float = false;
        if matches!(self.peek(), Some(b'+') | Some(b'-')) {
            self.bump();
        }
        // Hex literal: 0x... or 0X...
        if self.peek() == Some(b'0') && matches!(self.peek_at(1), Some(b'x') | Some(b'X')) {
            self.bump(); // '0'
            self.bump(); // 'x'
            let hex_start = self.pos;
            while let Some(c) = self.peek() {
                if c.is_ascii_hexdigit() {
                    self.bump();
                } else {
                    break;
                }
            }
            let hex_txt = std::str::from_utf8(&self.src[hex_start..self.pos]).unwrap();
            if hex_txt.is_empty() {
                return Err(ReadError::BadNumber(
                    String::from_utf8_lossy(&self.src[begin..self.pos]).to_string(),
                    begin,
                ));
            }
            let n = i64::from_str_radix(hex_txt, 16).map_err(|_| {
                ReadError::BadNumber(
                    String::from_utf8_lossy(&self.src[begin..self.pos]).to_string(),
                    begin,
                )
            })?;
            // Apply leading sign if present.
            let signed = if self.src[begin] == b'-' { -n } else { n };
            return Ok(encode_int(signed));
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
        with_scope(8, |scope| {
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
            b'(' | b')'
                | b'['
                | b']'
                | b'{'
                | b'}'
                | b'\''
                | b'`'
                | b','
                | b';'
                | b'"'
                | b'^'
                | b'@'
                | b'~'
        )
}
