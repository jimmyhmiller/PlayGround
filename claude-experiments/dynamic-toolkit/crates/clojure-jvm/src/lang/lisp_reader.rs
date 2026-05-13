//! Narrow port of `clojure.lang.LispReader`.
//!
//! Source: `~/Documents/Code/open-source/clojure/src/jvm/clojure/lang/LispReader.java`
//! (1702 lines).
//!
//! Java's reader is a dispatch table indexed by character on a `PushbackReader`.
//! We translate the surface behavior (the read paths each character triggers)
//! into a hand-written recursive-descent reader on a `&str` cursor. Same
//! observable behavior for the subset we ship; structurally simpler.
//!
//! What's ported:
//!   * `nil`, `true`, `false`
//!   * Integer + Double literals (decimal, optional sign)
//!   * String literals (`"…"` with `\n \t \r \" \\` escapes)
//!   * Symbols (Java's macro-character rules, narrowed)
//!   * Keywords (`:foo`, `:ns/name`)
//!   * Lists `(…)`
//!   * Vectors `[…]`
//!   * Quote `'x` → `(quote x)`
//!   * Line comments `;`
//!   * Whitespace (incl. commas — Clojure treats `,` as whitespace)
//!
//! Stubbed for later (panic with `unimplemented_port!`):
//!   * Maps `{…}` — needs IPersistentMap port
//!   * Sets `#{…}` and other dispatch-`#` macros (regex, var, anon-fn)
//!   * Syntax-quote ``` ` ``` and unquote
//!   * Numbers: BigInt / Ratio / hex / oct / radix prefixes
//!   * Reader metadata `^{…}` / `^:foo`
//!   * Character literals `\a` `\newline` etc.
//!   * Reader conditionals `#?` / `#?@`

use std::collections::HashMap;
use std::sync::Arc;

use super::keyword::Keyword;
use super::object::Object;
use super::persistent_hash_map::PersistentHashMap;
use super::persistent_hash_set::PersistentHashSet;
use super::persistent_list::PersistentList;
use super::persistent_vector::PersistentVector;
use super::symbol::Symbol;

/// A single read operation reads one form (or returns `None` at EOF).
pub struct Reader<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(src: &'a str) -> Self {
        Reader { src, pos: 0 }
    }

    /// Current byte cursor — used by progressive loaders that read+eval
    /// one form at a time and need to skip past the consumed bytes.
    pub fn byte_pos(&self) -> usize {
        self.pos
    }

    /// Read one form. Returns `Ok(None)` at EOF.
    pub fn read(&mut self) -> Result<Option<Object>, ReaderError> {
        self.skip_ws_and_comments();
        if self.pos >= self.src.len() {
            return Ok(None);
        }
        self.read_form().map(Some)
    }

    fn read_form(&mut self) -> Result<Object, ReaderError> {
        self.skip_ws_and_comments();
        let c = self.peek_byte().ok_or_else(|| self.err("EOF while reading"))?;
        match c {
            b'(' => self.read_list(),
            b'[' => self.read_vector(),
            b')' | b']' | b'}' => Err(self.err_with(format!("Unmatched delimiter `{}`", c as char))),
            b'"' => self.read_string(),
            b'\'' => self.read_quote(),
            b':' => self.read_keyword(),
            b';' => {
                self.skip_line_comment();
                self.read_form()
            }
            b'{' => self.read_map(),
            b'#' => self.read_dispatch_placeholder(),
            b'^' => self.read_meta(),
            b'`' => self.read_syntax_quote_placeholder(),
            b'~' => self.read_unquote_placeholder(),
            b'@' => {
                // `@x` → `(deref x)`. The `deref` symbol resolves to
                // clojure.core/deref at analyze time (defined as a defn
                // earlier in core.clj).
                self.pos += 1;
                let inner = self.read_form()?;
                Ok(Object::List(PersistentList::create(vec![
                    Object::Symbol(Symbol::intern("deref")),
                    inner,
                ])))
            }
            b'\\' => {
                // Character literal `\c` / `\space` / `\newline` / `\tab` /
                // `\return` / `\formfeed` / `\backspace` / `\uHHHH`. We
                // model chars as Long-valued NanBox (matching how the
                // analyzer treats integer-cast chars).
                self.pos += 1;
                let start = self.pos;
                // Always consume the FIRST byte even if it's a
                // terminating macro (e.g. `\,`, `\(`, `\)`, `\"`).
                if self.peek_byte().is_some() {
                    self.pos += 1;
                }
                while let Some(c) = self.peek_byte() {
                    if c.is_ascii_whitespace() || is_terminating_macro(c) {
                        break;
                    }
                    self.pos += 1;
                }
                let token = &self.src[start..self.pos];
                let code: i64 = match token {
                    "" => return Err(self.err("EOF after `\\` character literal start")),
                    "space" => 32,
                    "newline" => 10,
                    "tab" => 9,
                    "return" => 13,
                    "formfeed" => 12,
                    "backspace" => 8,
                    s if s.starts_with('u') && s.len() == 5 => {
                        i64::from_str_radix(&s[1..], 16).map_err(|_| {
                            self.err_with(format!("invalid unicode escape `\\{s}`"))
                        })?
                    }
                    s if s.chars().count() == 1 => s.chars().next().unwrap() as i64,
                    other => {
                        return Err(self.err_with(format!(
                            "unsupported character literal `\\{other}`"
                        )));
                    }
                };
                Ok(Object::Long(code))
            }
            _ => self.read_atom(),
        }
    }

    fn read_atom(&mut self) -> Result<Object, ReaderError> {
        let start = self.pos;
        // Consume a token: anything until whitespace or a *terminating*
        // macro character. Java's LispReader distinguishes terminating
        // macros (whitespace, parens, brackets, `"`, `;`) from
        // non-terminating ones (`'`, `~`, `@`, `^`, `` ` ``, `#`) — the
        // latter are only macros at the start of a token, otherwise they
        // can appear mid-symbol. This is how names like `inc'` and `set!`
        // read as single symbols.
        while let Some(c) = self.peek_byte() {
            if is_terminating_macro(c) {
                break;
            }
            self.pos += 1;
        }
        let tok = &self.src[start..self.pos];
        if tok.is_empty() {
            return Err(self.err("Empty token"));
        }
        // nil / true / false first.
        match tok {
            "nil" => return Ok(Object::Nil),
            "true" => return Ok(Object::Bool(true)),
            "false" => return Ok(Object::Bool(false)),
            _ => {}
        }
        // Numbers: leading digit or `-` / `+` followed by a digit.
        let bytes = tok.as_bytes();
        if bytes[0].is_ascii_digit()
            || ((bytes[0] == b'-' || bytes[0] == b'+')
                && bytes.len() >= 2
                && bytes[1].is_ascii_digit())
        {
            return self.parse_number(tok);
        }
        // Otherwise treat as a Symbol. Allow embedded `/` for ns/name.
        Ok(Object::Symbol(Symbol::intern(tok)))
    }

    fn parse_number(&self, tok: &str) -> Result<Object, ReaderError> {
        // Plain integer? Try i64 first, then fall back to double if it has `.`
        // or `e`/`E`.
        if tok.bytes().any(|b| matches!(b, b'.' | b'e' | b'E')) {
            tok.parse::<f64>()
                .map(Object::Double)
                .map_err(|e| ReaderError {
                    msg: format!("Invalid number `{tok}`: {e}"),
                    pos: self.pos,
                })
        } else {
            tok.parse::<i64>()
                .map(Object::Long)
                .map_err(|e| ReaderError {
                    msg: format!("Invalid number `{tok}`: {e}"),
                    pos: self.pos,
                })
        }
    }

    fn read_keyword(&mut self) -> Result<Object, ReaderError> {
        // We're sitting on the leading ':'.
        self.pos += 1;
        let start = self.pos;
        while let Some(c) = self.peek_byte() {
            if is_terminating_macro(c) {
                break;
            }
            self.pos += 1;
        }
        let body = &self.src[start..self.pos];
        if body.is_empty() {
            return Err(self.err("Invalid token: :"));
        }
        // Split on optional `/` into (ns, name).
        let (ns, name) = match body.find('/') {
            Some(i) if i > 0 && i < body.len() - 1 => (Some(&body[..i]), &body[i + 1..]),
            _ => (None, body),
        };
        Ok(Object::Keyword(Keyword::intern_ns_name(ns, name)))
    }

    fn read_string(&mut self) -> Result<Object, ReaderError> {
        // We're on the opening '"'.
        self.pos += 1;
        let mut out = String::new();
        loop {
            let c = self
                .peek_byte()
                .ok_or_else(|| self.err("EOF inside string literal"))?;
            match c {
                b'"' => {
                    self.pos += 1;
                    return Ok(Object::String(Arc::new(out)));
                }
                b'\\' => {
                    self.pos += 1;
                    let esc = self
                        .peek_byte()
                        .ok_or_else(|| self.err("EOF after backslash in string"))?;
                    let ch = match esc {
                        b'n' => '\n',
                        b't' => '\t',
                        b'r' => '\r',
                        b'"' => '"',
                        b'\\' => '\\',
                        b'0' => '\0',
                        other => {
                            return Err(self.err_with(format!(
                                "Unsupported string escape: \\{}",
                                other as char
                            )));
                        }
                    };
                    out.push(ch);
                    self.pos += 1;
                }
                _ => {
                    // Push the next full UTF-8 char.
                    let rest = &self.src[self.pos..];
                    let mut chars = rest.char_indices();
                    let (_, ch) = chars.next().expect("non-empty slice");
                    out.push(ch);
                    let consumed = chars.next().map(|(i, _)| i).unwrap_or(rest.len());
                    self.pos += consumed;
                }
            }
        }
    }

    fn read_list(&mut self) -> Result<Object, ReaderError> {
        self.pos += 1; // consume '('
        let items = self.read_delimited(b')')?;
        Ok(Object::List(PersistentList::create(items)))
    }

    fn read_vector(&mut self) -> Result<Object, ReaderError> {
        self.pos += 1; // consume '['
        let items = self.read_delimited(b']')?;
        Ok(Object::Vector(PersistentVector::create(items)))
    }

    /// Read forms until the matching closing delimiter.
    fn read_delimited(&mut self, close: u8) -> Result<Vec<Object>, ReaderError> {
        let mut out = Vec::new();
        loop {
            self.skip_ws_and_comments();
            match self.peek_byte() {
                None => {
                    return Err(self.err_with(format!(
                        "EOF while reading, expecting `{}`",
                        close as char
                    )))
                }
                Some(c) if c == close => {
                    self.pos += 1;
                    return Ok(out);
                }
                Some(_) => {
                    out.push(self.read_form()?);
                }
            }
        }
    }

    /// `^:keyword <form>` → `<form>` with `{:keyword true}` metadata.
    /// `^{:k v ...} <form>` → `<form>` with that map as metadata.
    /// `^Tag <form>`        → `<form>` with `{:tag Tag}` metadata.
    /// `^:kw <form>`        → `<form>` with `{:kw true}` metadata.
    /// `^"docstring" <form>`→ `<form>` with `{:tag <string>}` metadata.
    ///
    /// The metadata is attached as `Object::WithMeta(form, map)`. The
    /// `:macro` key on a def-name shorthand (`^:macro foo` /
    /// `^{:macro true} foo`) is no longer side-channeled through
    /// `Symbol::is_macro_meta` — `parse_def_form` reads it off the
    /// wrapper directly.
    fn read_meta(&mut self) -> Result<Object, ReaderError> {
        self.pos += 1; // consume '^'
        self.skip_ws_and_comments();
        let meta_map: Arc<PersistentHashMap> = match self.peek_byte() {
            Some(b':') => {
                // ^:kw  →  {:kw true}.
                let kw_obj = self.read_form()?;
                let key = match kw_obj {
                    Object::Keyword(k) => Object::Keyword(k),
                    other => return Err(self.err(&format!(
                        "Metadata `:` reader expects a keyword, got {other:?}"
                    ))),
                };
                PersistentHashMap::create_flat(vec![key, Object::Bool(true)])
            }
            Some(b'"') => {
                // ^"docstring" form — Clojure treats string metadata as
                // a `:tag` shorthand for the named class. We use it as
                // a `:tag` carrying the literal String, mirroring Java's
                // `LispReader.MetaReader.invoke`.
                let s = self.read_form()?;
                let tag_kw = Object::Keyword(Keyword::intern(Symbol::intern("tag")));
                PersistentHashMap::create_flat(vec![tag_kw, s])
            }
            Some(b'{') => {
                // ^{:k v :k2 v2 ...} — read as a map literal (delegating
                // to read_form so the existing `{...}` handler builds it).
                let map_obj = self.read_form()?;
                match map_obj {
                    Object::Map(m) => m,
                    other => return Err(self.err(&format!(
                        "Metadata `^{{...}}` did not read as a map, got {other:?}"
                    ))),
                }
            }
            Some(_) => {
                // ^Tag form — `Tag` is a Symbol or class name. Build
                // `{:tag <Tag-form>}`.
                let tag = self.read_form()?;
                let tag_kw = Object::Keyword(Keyword::intern(Symbol::intern("tag")));
                PersistentHashMap::create_flat(vec![tag_kw, tag])
            }
            None => return Err(self.err("EOF after `^`")),
        };

        self.skip_ws_and_comments();
        let form = self.read_form()?;
        Ok(Object::with_meta_map(form, meta_map))
    }

    fn read_quote(&mut self) -> Result<Object, ReaderError> {
        // 'x → (quote x)
        self.pos += 1;
        let inner = self.read_form()?;
        let quote_sym = Object::Symbol(Symbol::intern("quote"));
        Ok(Object::List(PersistentList::create(vec![quote_sym, inner])))
    }

    /// Placeholder for the `#` dispatch macro. Handles the most common
    /// dispatches we encounter while loading upstream `clojure/core.clj`:
    ///
    /// * `#{ … }` — set literal: read+discard contents, return `Nil`.
    ///   We don't have IPersistentSet yet.
    /// * `#_ form` — discard reader macro: skip the next form entirely
    ///   (and don't return anything for it). We implement this by reading
    ///   the discarded form and then recursing back into `read_form`.
    /// * `#'sym` — var reference shorthand: reads as `(var sym)`. Our
    ///   compiler doesn't have the `var` special form yet, so this is
    ///   substituted away by the loader's substitution pass.
    /// * Other dispatches (`##NaN`, `#?`, `#"regex"`, `#(...)`, ...) are
    ///   not yet handled and will panic with `unimplemented_port!`.
    fn read_dispatch_placeholder(&mut self) -> Result<Object, ReaderError> {
        self.pos += 1; // consume '#'
        let c = self.peek_byte().ok_or_else(|| self.err("EOF after `#` dispatch"))?;
        match c {
            b'{' => {
                self.pos += 1;
                let mut items: Vec<Object> = Vec::new();
                loop {
                    self.skip_ws_and_comments();
                    match self.peek_byte() {
                        Some(b'}') => {
                            self.pos += 1;
                            return Ok(Object::Set(PersistentHashSet::create(items)));
                        }
                        None => return Err(self.err("EOF inside set literal `#{...}`")),
                        Some(_) => {}
                    }
                    let e = self.read_form()?;
                    items.push(e);
                }
            }
            b'_' => {
                self.pos += 1;
                let _discarded = self.read_form()?;
                // Then read the next form for real.
                self.read_form()
            }
            b'\'' => {
                self.pos += 1;
                let sym = self.read_form()?;
                let var_sym = Object::Symbol(Symbol::intern("var"));
                Ok(Object::List(PersistentList::create(vec![var_sym, sym])))
            }
            b'(' => {
                // `#(body)` anonymous fn shorthand. Read body as a list,
                // walk it to discover `%`, `%1..%N`, `%&` references, then
                // emit `(fn* [p1 p2 ...] body)`.
                self.pos += 1;
                let mut items: Vec<Object> = Vec::new();
                loop {
                    self.skip_ws_and_comments();
                    match self.peek_byte() {
                        Some(b')') => { self.pos += 1; break; }
                        None => return Err(self.err("EOF inside #()")),
                        Some(_) => {}
                    }
                    items.push(self.read_form()?);
                }
                let body = Object::List(PersistentList::create(items));
                let mut max_n: usize = 0;
                let mut has_rest = false;
                scan_anon_fn_params(&body, &mut max_n, &mut has_rest);
                let mut params: Vec<Object> = Vec::with_capacity(max_n + 2);
                for i in 1..=max_n {
                    params.push(Object::Symbol(Symbol::intern(&format!("p{i}__#"))));
                }
                if has_rest {
                    params.push(Object::Symbol(Symbol::intern("&")));
                    params.push(Object::Symbol(Symbol::intern("rest__#")));
                }
                let renamed = rename_anon_fn_params(&body);
                Ok(Object::List(PersistentList::create(vec![
                    Object::Symbol(Symbol::intern("fn*")),
                    Object::Vector(crate::lang::persistent_vector::PersistentVector::create(params)),
                    renamed,
                ])))
            }
            b'"' => {
                // Regex literal `#"pattern"` — read until the matching
                // `"` WITHOUT interpreting escapes (so `\d`, `\s` etc.
                // pass through as literal chars). Returns a String;
                // we don't model Pattern objects.
                self.pos += 1; // consume opening "
                let start = self.pos;
                let mut prev_was_backslash = false;
                while let Some(c) = self.peek_byte() {
                    if c == b'"' && !prev_was_backslash {
                        break;
                    }
                    prev_was_backslash = c == b'\\' && !prev_was_backslash;
                    self.pos += 1;
                }
                if self.peek_byte() != Some(b'"') {
                    return Err(self.err("EOF inside regex literal"));
                }
                let body = self.src[start..self.pos].to_string();
                self.pos += 1; // consume closing "
                Ok(Object::String(Arc::new(body)))
            }
            b'?' => {
                // Reader-conditional `#?(:default form)` or `#?@(...)`.
                // Skip the `?@` if present, then read the next form
                // (a list of cond pairs) and return nil — we don't
                // model platform-specific code.
                self.pos += 1;
                if self.peek_byte() == Some(b'@') { self.pos += 1; }
                let _ = self.read_form()?;
                Ok(Object::Nil)
            }
            b'#' => {
                // `##NaN` / `##Inf` / `##-Inf` symbolic-value literal.
                self.pos += 1;
                let inner = self.read_form()?;
                Ok(match &inner {
                    Object::Symbol(s) => match s.get_name() {
                        "NaN" => Object::Double(f64::NAN),
                        "Inf" => Object::Double(f64::INFINITY),
                        "-Inf" => Object::Double(f64::NEG_INFINITY),
                        _ => Object::Nil,
                    },
                    _ => Object::Nil,
                })
            }
            b':' => {
                // Namespace-qualified map `#:ns{...}` or `#::{...}`.
                // Read the namespace token and the map body; return
                // the map unmodified (we don't auto-qualify keys).
                self.pos += 1;
                // Skip optional second `:` for auto-resolved.
                if self.peek_byte() == Some(b':') { self.pos += 1; }
                while let Some(c) = self.peek_byte() {
                    if c.is_ascii_whitespace() || c == b'{' { break; }
                    self.pos += 1;
                }
                self.skip_ws_and_comments();
                self.read_form()
            }
            b'^' => {
                // `#^Type form` — old-style metadata. Treat as `^Type form`.
                self.pos += 1;
                self.read_meta()
            }
            _ => {
                // Unknown dispatch — read one form and return nil so
                // the loader continues. Logs to stderr for visibility.
                let c_char = c as char;
                eprintln!(
                    "[cljvm-stub] unknown #-dispatch `{c_char}` — skipping one form, returning nil"
                );
                let _ = self.read_form()?;
                Ok(Object::Nil)
            }
        }
    }

    /// Syntax-quote: ``` `<form> ``` is read, then walked by
    /// `syntax_quote_form` into a runtime expression that constructs the
    /// form. `~x` and `~@x` get reified as `(unquote x)` /
    /// `(unquote-splicing x)` so the walker can recognize them when it
    /// descends through the form. Auto-gensyms (`sym#`) get unique-name
    /// resolution within one syntax-quote scope via the walker's env.
    fn read_syntax_quote_placeholder(&mut self) -> Result<Object, ReaderError> {
        self.pos += 1; // consume '`'
        let inner = self.read_form()?;
        let mut env: HashMap<String, Arc<Symbol>> = HashMap::new();
        Ok(syntax_quote_form(&inner, &mut env))
    }

    /// `~<form>` → `(unquote <form>)`; `~@<form>` →
    /// `(unquote-splicing <form>)`. The syntax-quote walker recognizes
    /// these wrappers and emits the unwrapped form (or splices it via
    /// concat) in the constructed result.
    fn read_unquote_placeholder(&mut self) -> Result<Object, ReaderError> {
        self.pos += 1; // consume '~'
        let splicing = self.peek_byte() == Some(b'@');
        if splicing {
            self.pos += 1;
        }
        let inner = self.read_form()?;
        let head = if splicing {
            Symbol::intern("unquote-splicing")
        } else {
            Symbol::intern("unquote")
        };
        Ok(Object::List(PersistentList::create(vec![
            Object::Symbol(head),
            inner,
        ])))
    }

    /// Read a `{k1 v1 k2 v2 ...}` map literal into an `Object::Map`.
    /// Matches `LispReader.MapReader` — reads `k v` pairs until `}`, then
    /// builds a `PersistentHashMap` via `create_pairs`. Duplicate keys
    /// keep the last value (Clojure's `PersistentArrayMap` reader does the
    /// same; the upstream check that errors on duplicates lives in a later
    /// macro-expansion pass, not the reader itself).
    fn read_map(&mut self) -> Result<Object, ReaderError> {
        self.pos += 1; // consume '{'
        let mut pairs: Vec<(Object, Object)> = Vec::new();
        loop {
            self.skip_ws_and_comments();
            match self.peek_byte() {
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Object::Map(PersistentHashMap::create_pairs(pairs)));
                }
                None => return Err(self.err("EOF inside map literal `{...}`")),
                Some(_) => {}
            }
            let k = self.read_form()?;
            self.skip_ws_and_comments();
            if matches!(self.peek_byte(), Some(b'}') | None) {
                return Err(self.err(
                    "Map literal must have an even number of forms",
                ));
            }
            let v = self.read_form()?;
            pairs.push((k, v));
        }
    }

    fn skip_ws_and_comments(&mut self) {
        loop {
            let start = self.pos;
            while let Some(c) = self.peek_byte() {
                // Clojure treats commas as whitespace.
                if c.is_ascii_whitespace() || c == b',' {
                    self.pos += 1;
                } else {
                    break;
                }
            }
            if self.peek_byte() == Some(b';') {
                self.skip_line_comment();
                continue;
            }
            if self.pos == start {
                return;
            }
        }
    }

    fn skip_line_comment(&mut self) {
        while let Some(c) = self.peek_byte() {
            self.pos += 1;
            if c == b'\n' {
                return;
            }
        }
    }

    fn peek_byte(&self) -> Option<u8> {
        self.src.as_bytes().get(self.pos).copied()
    }

    fn err(&self, msg: &str) -> ReaderError {
        ReaderError { msg: msg.to_string(), pos: self.pos }
    }

    fn err_with(&self, msg: String) -> ReaderError {
        ReaderError { msg, pos: self.pos }
    }
}

/// Terminating reader macros — these end a token wherever they appear.
/// Per Java's LispReader: whitespace, comma, open/close delimiters, `"`, `;`.
/// `'` `~` `@` `^` `` ` `` `#` are *non*-terminating — macros only at the
/// start of a token (handled in `read_form`'s top-level dispatch), but
/// allowed mid-symbol (so names like `inc'`, `set!`, `even?`, `<=` work).
fn is_terminating_macro(c: u8) -> bool {
    c.is_ascii_whitespace()
        || matches!(c, b',' | b'(' | b')' | b'[' | b']' | b'{' | b'}' | b'"' | b';')
}

#[derive(Debug)]
pub struct ReaderError {
    pub msg: String,
    pub pos: usize,
}

impl std::fmt::Display for ReaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ReaderError at byte {}: {}", self.pos, self.msg)
    }
}

impl std::error::Error for ReaderError {}

/// Convenience: read one form from a string. Returns an error if the string
/// is empty (no form) or has trailing forms.
pub fn read_str(src: &str) -> Result<Object, ReaderError> {
    let mut r = Reader::new(src);
    let form = r
        .read()?
        .ok_or_else(|| ReaderError { msg: "EOF: expected a form".to_string(), pos: 0 })?;
    // Don't error on trailing whitespace.
    r.skip_ws_and_comments();
    if r.pos < src.len() {
        return Err(ReaderError {
            msg: format!("Trailing input after form at byte {}", r.pos),
            pos: r.pos,
        });
    }
    Ok(form)
}

/// Read every form from a source string. Useful for loading a file.
pub fn read_all(src: &str) -> Result<Vec<Object>, ReaderError> {
    let mut r = Reader::new(src);
    let mut out = Vec::new();
    while let Some(form) = r.read()? {
        out.push(form);
    }
    Ok(out)
}

// ─── Syntax-quote walker ──────────────────────────────────────────────
//
// Transforms a syntax-quoted form into a runtime expression that
// constructs the form. Mirrors Clojure's `LispReader.SyntaxQuoteReader`
// at a smaller scope:
//   * `~x`            → x   (inserted into outer concat)
//   * `~@x`           → x   (spliced via concat)
//   * `sym#`          → unique gensym (per syntax-quote invocation)
//   * unqualified sym → (quote sym)   (no auto-namespacing yet)
//   * list `(a b c)`  → (seq (concat (list a') (list b') (list c')))
//   * vector [a b c]  → (apply vector (concat (list a') (list b') (list c')))
//   * map / set       → not yet
//   * other           → (quote form)   (self-evaluating types untouched)
//
// `gensym_env` keeps `sym#` consistent within one syntax-quote: every
// occurrence of the same `sym#` resolves to the same gensym.
fn syntax_quote_form(form: &Object, env: &mut HashMap<String, Arc<Symbol>>) -> Object {
    match form {
        // Self-evaluating literals — quote them so the surrounding
        // (concat …) treats them as data.
        Object::Nil
        | Object::Bool(_)
        | Object::Long(_)
        | Object::Double(_)
        | Object::String(_)
        | Object::Keyword(_) => quote(form.clone()),

        Object::Symbol(s) => {
            let name = s.get_name();
            if name.ends_with('#') && s.get_namespace().is_none() {
                let g = env.entry(name.to_string()).or_insert_with(|| {
                    use std::sync::atomic::{AtomicU64, Ordering};
                    static N: AtomicU64 = AtomicU64::new(0);
                    let id = N.fetch_add(1, Ordering::Relaxed);
                    Symbol::intern(&format!("{}__{id}__auto__", &name[..name.len() - 1]))
                });
                quote(Object::Symbol(g.clone()))
            } else {
                quote(Object::Symbol(s.clone()))
            }
        }

        Object::List(l) => {
            // Empty list → '()
            if l.count() == 0 {
                return quote(Object::List(PersistentList::empty()));
            }
            // `(. clojure.lang.RT (sqConcat (list <items>...)))` —
            // see `cljvm_rt_sq_concat` for why we route through the
            // RT host-method instead of `clojure.core/concat` (which
            // is multi-arity and binds its Var to nil currently).
            let items: Vec<Object> =
                l.iter().map(|item| syntax_quote_item(&item, env)).collect();
            sq_concat_call(items)
        }

        Object::Vector(v) => {
            // `(vec (. RT sqConcat …))` — same RT routing as the list case,
            // then convert the resulting seq to a Vector.
            let items: Vec<Object> =
                (0..v.count()).map(|i| syntax_quote_item(&v.nth(i), env)).collect();
            let inner = sq_concat_call(items);
            Object::List(PersistentList::create(vec![
                Object::Symbol(Symbol::intern("vec")),
                inner,
            ]))
        }

        // Anything else (Map, Set, …) — punt. Bootstrap macros don't use
        // these inside syntax-quote heavily, and a wrong here is loud
        // (compile error from a malformed result form) rather than silent.
        _ => quote(form.clone()),
    }
}

/// Per-element step inside a `(concat …)`. Plain forms wrap in `(list …)`;
/// unquote-spliced forms emit the spliced expression as-is.
fn syntax_quote_item(item: &Object, env: &mut HashMap<String, Arc<Symbol>>) -> Object {
    if let Object::List(l) = item {
        if l.count() == 2 {
            let head_obj = l.iter().next().unwrap();
            if let Object::Symbol(head) = &head_obj {
                let name = head.get_name();
                if (name == "clojure.core/unquote"
                    || (name == "unquote" && head.get_namespace().is_none()))
                    && head.get_namespace().map(|ns| ns == "clojure.core").unwrap_or(true)
                {
                    let inner = l.iter().nth(1).unwrap();
                    return Object::List(PersistentList::create(vec![
                        Object::Symbol(Symbol::intern("list")),
                        inner,
                    ]));
                }
                if name == "clojure.core/unquote-splicing"
                    || (name == "unquote-splicing" && head.get_namespace().is_none())
                {
                    let inner = l.iter().nth(1).unwrap();
                    return inner;
                }
            }
        }
    }
    Object::List(PersistentList::create(vec![
        Object::Symbol(Symbol::intern("list")),
        syntax_quote_form(item, env),
    ]))
}

/// Build `(. clojure.lang.RT (sqConcat (list <items>...)))`. The inner
/// `(list …)` builds the seq-of-seqs that `cljvm_rt_sq_concat` walks.
fn sq_concat_call(items: Vec<Object>) -> Object {
    let mut list_args = vec![Object::Symbol(Symbol::intern("list"))];
    list_args.extend(items);
    let list_call = Object::List(PersistentList::create(list_args));
    let method_call = Object::List(PersistentList::create(vec![
        Object::Symbol(Symbol::intern("sqConcat")),
        list_call,
    ]));
    Object::List(PersistentList::create(vec![
        Object::Symbol(Symbol::intern(".")),
        Object::Symbol(Symbol::intern("clojure.lang.RT")),
        method_call,
    ]))
}

/// Walk a `#(...)` body, recording the highest `%N` index seen
/// (`%` counts as `%1`) and whether `%&` (variadic-rest) appears.
fn scan_anon_fn_params(form: &Object, max_n: &mut usize, has_rest: &mut bool) {
    match form {
        Object::Symbol(s) if s.get_namespace().is_none() => {
            let name = s.get_name();
            if name == "%" {
                if *max_n < 1 { *max_n = 1; }
            } else if name == "%&" {
                *has_rest = true;
            } else if let Some(rest) = name.strip_prefix('%') {
                if let Ok(n) = rest.parse::<usize>() {
                    if n > *max_n { *max_n = n; }
                }
            }
        }
        Object::List(l) => {
            for item in l.iter() {
                scan_anon_fn_params(&item, max_n, has_rest);
            }
        }
        Object::Vector(v) => {
            for i in 0..v.count() {
                scan_anon_fn_params(&v.nth(i), max_n, has_rest);
            }
        }
        Object::WithMeta(inner, _) => {
            scan_anon_fn_params(inner, max_n, has_rest);
        }
        _ => {}
    }
}

/// Rename `%`/`%N`/`%&` symbols to the synthetic param names
/// (`p1__#`, `p2__#`, …, `rest__#`) chosen above.
fn rename_anon_fn_params(form: &Object) -> Object {
    match form {
        Object::Symbol(s) if s.get_namespace().is_none() => {
            let name = s.get_name();
            if name == "%" {
                return Object::Symbol(Symbol::intern("p1__#"));
            }
            if name == "%&" {
                return Object::Symbol(Symbol::intern("rest__#"));
            }
            if let Some(rest) = name.strip_prefix('%') {
                if let Ok(n) = rest.parse::<usize>() {
                    return Object::Symbol(Symbol::intern(&format!("p{n}__#")));
                }
            }
            form.clone()
        }
        Object::List(l) => {
            let new_items: Vec<Object> =
                l.iter().map(|i| rename_anon_fn_params(&i)).collect();
            Object::List(PersistentList::create(new_items))
        }
        Object::WithMeta(inner, _meta) => {
            // Drop metadata, recurse on inner. Metadata isn't useful
            // here and surviving WithMeta wrappers around `%N` would
            // shield them from the rename.
            rename_anon_fn_params(inner)
        }
        Object::Vector(v) => {
            let new_items: Vec<Object> = (0..v.count())
                .map(|i| rename_anon_fn_params(&v.nth(i)))
                .collect();
            Object::Vector(crate::lang::persistent_vector::PersistentVector::create(new_items))
        }
        _ => form.clone(),
    }
}

fn quote(form: Object) -> Object {
    Object::List(PersistentList::create(vec![
        Object::Symbol(Symbol::intern("quote")),
        form,
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rd(s: &str) -> Object {
        read_str(s).unwrap_or_else(|e| panic!("read_str({s:?}) → {e}"))
    }

    #[test]
    fn reads_nil_true_false() {
        assert!(matches!(rd("nil"), Object::Nil));
        assert!(matches!(rd("true"), Object::Bool(true)));
        assert!(matches!(rd("false"), Object::Bool(false)));
    }

    #[test]
    fn reads_integers() {
        assert!(matches!(rd("0"), Object::Long(0)));
        assert!(matches!(rd("42"), Object::Long(42)));
        assert!(matches!(rd("-7"), Object::Long(-7)));
        assert!(matches!(rd("+9"), Object::Long(9)));
    }

    #[test]
    fn reads_doubles() {
        assert!(matches!(rd("3.14"), Object::Double(x) if (x - 3.14).abs() < 1e-9));
        assert!(matches!(rd("-1.5e2"), Object::Double(x) if (x + 150.0).abs() < 1e-9));
    }

    #[test]
    fn reads_strings() {
        match rd(r#""hello""#) {
            Object::String(s) => assert_eq!(&*s, "hello"),
            other => panic!("expected String, got {other:?}"),
        }
        match rd(r#""line1\nline2""#) {
            Object::String(s) => assert_eq!(&*s, "line1\nline2"),
            other => panic!("expected String, got {other:?}"),
        }
        match rd(r#""\\path""#) {
            Object::String(s) => assert_eq!(&*s, "\\path"),
            other => panic!("expected String, got {other:?}"),
        }
    }

    #[test]
    fn reads_symbols() {
        match rd("foo") {
            Object::Symbol(s) => {
                assert!(s.get_namespace().is_none());
                assert_eq!(s.get_name(), "foo");
            }
            other => panic!("expected Symbol, got {other:?}"),
        }
        match rd("clojure.core/+") {
            Object::Symbol(s) => {
                assert_eq!(s.get_namespace(), Some("clojure.core"));
                assert_eq!(s.get_name(), "+");
            }
            other => panic!("expected Symbol, got {other:?}"),
        }
    }

    #[test]
    fn reads_keywords() {
        match rd(":foo") {
            Object::Keyword(k) => {
                assert!(k.get_namespace().is_none());
                assert_eq!(k.get_name(), "foo");
            }
            other => panic!("expected Keyword, got {other:?}"),
        }
        match rd(":user/name") {
            Object::Keyword(k) => {
                assert_eq!(k.get_namespace(), Some("user"));
                assert_eq!(k.get_name(), "name");
            }
            other => panic!("expected Keyword, got {other:?}"),
        }
    }

    #[test]
    fn reads_list() {
        match rd("(1 2 3)") {
            Object::List(l) => {
                assert_eq!(l.count(), 3);
            }
            other => panic!("expected List, got {other:?}"),
        }
    }

    #[test]
    fn reads_vector() {
        match rd("[1 2 3]") {
            Object::Vector(v) => assert_eq!(v.count(), 3),
            other => panic!("expected Vector, got {other:?}"),
        }
    }

    #[test]
    fn reads_nested_list_with_mixed_types() {
        // (def x 42) — first real form anyone writes.
        let form = rd("(def x 42)");
        match form {
            Object::List(l) => {
                assert_eq!(l.count(), 3);
                let v: Vec<Object> = l.iter().collect();
                assert!(matches!(&v[0], Object::Symbol(s) if s.get_name() == "def"));
                assert!(matches!(&v[1], Object::Symbol(s) if s.get_name() == "x"));
                assert!(matches!(&v[2], Object::Long(42)));
            }
            other => panic!("expected List, got {other:?}"),
        }
    }

    #[test]
    fn reads_quote_shorthand() {
        // 'x → (quote x)
        match rd("'x") {
            Object::List(l) => {
                assert_eq!(l.count(), 2);
                let v: Vec<Object> = l.iter().collect();
                assert!(matches!(&v[0], Object::Symbol(s) if s.get_name() == "quote"));
                assert!(matches!(&v[1], Object::Symbol(s) if s.get_name() == "x"));
            }
            other => panic!("expected (quote x), got {other:?}"),
        }
    }

    #[test]
    fn skips_line_comments() {
        let form = rd("; head comment\n(def x ; trailing\n  42)");
        match form {
            Object::List(l) => assert_eq!(l.count(), 3),
            other => panic!("expected List, got {other:?}"),
        }
    }

    #[test]
    fn treats_commas_as_whitespace() {
        match rd("[1, 2, 3]") {
            Object::Vector(v) => assert_eq!(v.count(), 3),
            other => panic!("expected Vector, got {other:?}"),
        }
    }

    #[test]
    fn reads_empty_list_and_vector() {
        match rd("()") {
            Object::List(l) => assert_eq!(l.count(), 0),
            other => panic!("expected List, got {other:?}"),
        }
        match rd("[]") {
            Object::Vector(v) => assert_eq!(v.count(), 0),
            other => panic!("expected Vector, got {other:?}"),
        }
    }

    #[test]
    fn read_all_returns_every_form() {
        let forms = read_all("(def x 1) (def y 2)").unwrap();
        assert_eq!(forms.len(), 2);
    }

    #[test]
    fn reads_realistic_defn_shape() {
        // Shape of a defn from clojure.core (without metadata).
        let form = rd("(def inc1 (fn* [n] (+ n 1)))");
        match form {
            Object::List(l) => assert_eq!(l.count(), 3),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn reader_carries_doc_meta_via_map_form() {
        // `^{:doc "..."} sym` — map metadata is real; the symbol is
        // wrapped in `Object::WithMeta(Symbol, {:doc "hello"})`.
        let form = rd(r#"^{:doc "hello"} foo"#);
        let meta = form.meta_of().expect("expected metadata wrapper");
        let doc_kw = Object::Keyword(Keyword::intern(Symbol::intern("doc")));
        match meta.val_at(&doc_kw) {
            Object::String(s) => assert_eq!(&*s, "hello"),
            other => panic!("expected :doc \"hello\", got {other:?}"),
        }
        match form.peel_meta_ref() {
            Object::Symbol(s) => assert_eq!(s.get_name(), "foo"),
            other => panic!("expected wrapped Symbol, got {other:?}"),
        }
    }

    #[test]
    fn reader_picks_up_macro_true_inside_map_meta() {
        let form = rd(r#"^{:macro true :doc "x"} bar"#);
        let meta = form.meta_of().expect("expected metadata wrapper");
        let macro_kw = Object::Keyword(Keyword::intern(Symbol::intern("macro")));
        assert!(matches!(meta.val_at(&macro_kw), Object::Bool(true)));
        match form.peel_meta_ref() {
            Object::Symbol(s) => assert_eq!(s.get_name(), "bar"),
            other => panic!("expected wrapped Symbol, got {other:?}"),
        }
    }

    #[test]
    fn reader_handles_macro_meta() {
        // `^:macro foo` → WithMeta(Symbol foo, {:macro true}).
        let form = rd("^:macro foo");
        let meta = form.meta_of().expect("expected metadata wrapper");
        let macro_kw = Object::Keyword(Keyword::intern(Symbol::intern("macro")));
        assert!(matches!(meta.val_at(&macro_kw), Object::Bool(true)));
        match form.peel_meta_ref() {
            Object::Symbol(s) => assert_eq!(s.get_name(), "foo"),
            other => panic!("expected wrapped Symbol, got {other:?}"),
        }
        // Bare `foo` — no wrapper.
        match rd("foo") {
            Object::Symbol(s) => assert_eq!(s.get_name(), "foo"),
            other => panic!("expected bare Symbol, got {other:?}"),
        }
    }

    #[test]
    fn error_on_unmatched_paren() {
        assert!(read_str("(").is_err());
        assert!(read_str(")").is_err());
        assert!(read_str("(a b").is_err());
    }
}
