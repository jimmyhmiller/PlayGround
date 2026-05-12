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

use std::sync::Arc;

use super::keyword::Keyword;
use super::object::Object;
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
            b'{' => crate::unimplemented_port!(
                "LispReader: map literal `{}`",
                "needs IPersistentMap port"
            ),
            b'#' => crate::unimplemented_port!(
                "LispReader: dispatch macro `#`",
                "needs set / regex / var / anon-fn / reader-conditional ports"
            ),
            b'^' => self.read_meta(),
            b'`' => crate::unimplemented_port!(
                "LispReader: syntax-quote ` `",
                "needs syntax-quote expansion"
            ),
            b'~' => crate::unimplemented_port!(
                "LispReader: unquote `~`",
                "only meaningful inside syntax-quote"
            ),
            b'@' => crate::unimplemented_port!(
                "LispReader: deref `@`",
                "needs `clojure.core/deref` Var"
            ),
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
    /// `^Tag <form>`  → `<form>` with `{:tag Tag}` metadata.
    ///
    /// Narrow port: until we have IPersistentMap to carry full metadata,
    /// the only meta we *act on* is `:macro` on a Symbol (toggles
    /// `Symbol::is_macro_meta`). Other metadata is parsed and discarded.
    /// The reader still walks past the meta correctly.
    fn read_meta(&mut self) -> Result<Object, ReaderError> {
        self.pos += 1; // consume '^'
        self.skip_ws_and_comments();
        // Read the metadata form. It's :keyword, {map}, Tag-symbol, or
        // "string" — but we only special-case :keyword (extract its name).
        let macro_meta = match self.peek_byte() {
            Some(b':') => {
                // :foo → meta is `{:foo true}`. We extract just the name to
                // detect `:macro`.
                self.pos += 1;
                let start = self.pos;
                while let Some(c) = self.peek_byte() {
                    if is_terminating_macro(c) {
                        break;
                    }
                    self.pos += 1;
                }
                let name = &self.src[start..self.pos];
                if name.is_empty() {
                    return Err(self.err("Invalid token: :"));
                }
                name == "macro"
            }
            Some(b'{') => {
                // For `^{:macro true}` we'd need a real map literal. Until
                // map support lands, refuse this shape rather than silently
                // ignoring meta the user clearly cares about.
                return Err(self.err_with(
                    "LispReader: ^{...} metadata not yet supported (needs map literal)"
                        .to_string(),
                ));
            }
            Some(_) => {
                // Probably `^Tag form` — read the tag form, ignore. (Not
                // implemented as actionable meta; preserves reader progress.)
                let _ignored = self.read_form()?;
                false
            }
            None => return Err(self.err("EOF after `^`")),
        };

        // Now read the target form and apply the meta (if recognized).
        self.skip_ws_and_comments();
        let form = self.read_form()?;
        if macro_meta {
            if let Object::Symbol(s) = &form {
                s.set_macro_meta();
            }
        }
        Ok(form)
    }

    fn read_quote(&mut self) -> Result<Object, ReaderError> {
        // 'x → (quote x)
        self.pos += 1;
        let inner = self.read_form()?;
        let quote_sym = Object::Symbol(Symbol::intern("quote"));
        Ok(Object::List(PersistentList::create(vec![quote_sym, inner])))
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
    fn reader_handles_macro_meta() {
        // `^:macro foo` → Symbol `foo` with is_macro_meta = true.
        match rd("^:macro foo") {
            Object::Symbol(s) => {
                assert_eq!(s.get_name(), "foo");
                assert!(s.has_macro_meta(), "expected is_macro_meta=true");
            }
            other => panic!("expected Symbol, got {other:?}"),
        }
        // Bare `foo` (no meta).
        match rd("foo") {
            Object::Symbol(s) => assert!(!s.has_macro_meta()),
            other => panic!("expected Symbol, got {other:?}"),
        }
    }

    #[test]
    fn error_on_unmatched_paren() {
        assert!(read_str("(").is_err());
        assert!(read_str(")").is_err());
        assert!(read_str("(a b").is_err());
    }
}
