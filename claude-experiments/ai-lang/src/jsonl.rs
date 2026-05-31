//! A tiny, hand-rolled JSON reader + writer for the JSONL edit-server.
//!
//! We deliberately take NO third-party JSON dependency (no serde). The protocol
//! values are simple — strings, integers, booleans, null, arrays, and objects —
//! so a small recursive-descent parser and a matching serializer are enough, and
//! they are kept minimal but *correct*: the parser handles the full JSON string
//! escape set (`\" \\ \/ \b \f \n \r \t \uXXXX`, including surrogate pairs) and
//! the writer emits valid JSON, escaping exactly the characters the spec
//! requires. Everything the server actually round-trips is covered by the unit
//! tests at the bottom of this file.
//!
//! This module lives strictly above the hashing line; it touches no canonical
//! AST machinery.

use std::collections::BTreeMap;
use std::fmt::Write as _;

// =============================================================================
// Value model
// =============================================================================

/// A parsed JSON value. Numbers are split into integer (`Int`) and floating
/// (`Float`) so the common protocol case (request ids, counts) round-trips as
/// exact integers rather than via lossy `f64`.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Array(Vec<Json>),
    /// Object. A `BTreeMap` keeps keys sorted so serialized output is
    /// deterministic (handy for tests and for diffing protocol traffic).
    Object(BTreeMap<String, Json>),
}

impl Json {
    /// Construct an object from an iterator of `(key, value)` pairs.
    pub fn obj<I, K>(pairs: I) -> Json
    where
        I: IntoIterator<Item = (K, Json)>,
        K: Into<String>,
    {
        Json::Object(pairs.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }

    /// Borrow a field of an object, if present.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(m) => m.get(key),
            _ => None,
        }
    }

    /// The string value, if this is a `Str`.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// The integer value, if this is an `Int` (or a `Float` with no fraction).
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Json::Int(i) => Some(*i),
            Json::Float(f) if f.fract() == 0.0 => Some(*f as i64),
            _ => None,
        }
    }

    /// The boolean value, if this is a `Bool`.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// The array elements, if this is an `Array`.
    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Array(v) => Some(v.as_slice()),
            _ => None,
        }
    }
}

// =============================================================================
// Serialization (Json -> String)
// =============================================================================

impl Json {
    /// Serialize to a compact (no whitespace) JSON string. Always valid JSON.
    pub fn to_string(&self) -> String {
        let mut out = String::new();
        self.write_into(&mut out);
        out
    }

    fn write_into(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(true) => out.push_str("true"),
            Json::Bool(false) => out.push_str("false"),
            Json::Int(i) => {
                let _ = write!(out, "{}", i);
            }
            Json::Float(f) => {
                // Non-finite floats are not representable in JSON; emit null
                // rather than invalid tokens like `NaN`/`Infinity`.
                if f.is_finite() {
                    let _ = write!(out, "{}", f);
                } else {
                    out.push_str("null");
                }
            }
            Json::Str(s) => write_json_string(s, out),
            Json::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    item.write_into(out);
                }
                out.push(']');
            }
            Json::Object(map) => {
                out.push('{');
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_json_string(k, out);
                    out.push(':');
                    v.write_into(out);
                }
                out.push('}');
            }
        }
    }
}

/// Write a JSON-escaped, double-quoted string. Escapes the characters the JSON
/// spec requires: quote, backslash, and the C0 control characters (the named
/// short escapes where they exist, `\u00XX` otherwise).
fn write_json_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

// =============================================================================
// Parsing (String -> Json)
// =============================================================================

/// A JSON parse error: a human-readable message plus the byte offset reached.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub message: String,
    pub at: usize,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JSON parse error at byte {}: {}", self.at, self.message)
    }
}

impl std::error::Error for ParseError {}

/// Parse exactly one JSON value from `input`. Trailing whitespace is allowed;
/// trailing non-whitespace content is an error (one value per line in JSONL).
pub fn parse(input: &str) -> Result<Json, ParseError> {
    let mut p = Parser {
        bytes: input.as_bytes(),
        pos: 0,
    };
    p.skip_ws();
    let v = p.parse_value()?;
    p.skip_ws();
    if p.pos != p.bytes.len() {
        return Err(p.err("trailing characters after JSON value"));
    }
    Ok(v)
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn err(&self, msg: &str) -> ParseError {
        ParseError {
            message: msg.to_owned(),
            at: self.pos,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while let Some(b) = self.peek() {
            if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn parse_value(&mut self) -> Result<Json, ParseError> {
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b'"') => Ok(Json::Str(self.parse_string()?)),
            Some(b't') | Some(b'f') => self.parse_bool(),
            Some(b'n') => self.parse_null(),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.parse_number(),
            Some(_) => Err(self.err("unexpected character at start of value")),
            None => Err(self.err("unexpected end of input")),
        }
    }

    fn expect(&mut self, b: u8) -> Result<(), ParseError> {
        if self.peek() == Some(b) {
            self.pos += 1;
            Ok(())
        } else {
            Err(self.err(&format!("expected '{}'", b as char)))
        }
    }

    fn parse_object(&mut self) -> Result<Json, ParseError> {
        self.expect(b'{')?;
        let mut map = BTreeMap::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Json::Object(map));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(self.err("expected string key in object"));
            }
            let key = self.parse_string()?;
            self.skip_ws();
            self.expect(b':')?;
            let value = self.parse_value()?;
            map.insert(key, value);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(self.err("expected ',' or '}' in object")),
            }
        }
        Ok(Json::Object(map))
    }

    fn parse_array(&mut self) -> Result<Json, ParseError> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Json::Array(items));
        }
        loop {
            let value = self.parse_value()?;
            items.push(value);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(self.err("expected ',' or ']' in array")),
            }
        }
        Ok(Json::Array(items))
    }

    fn parse_bool(&mut self) -> Result<Json, ParseError> {
        if self.bytes[self.pos..].starts_with(b"true") {
            self.pos += 4;
            Ok(Json::Bool(true))
        } else if self.bytes[self.pos..].starts_with(b"false") {
            self.pos += 5;
            Ok(Json::Bool(false))
        } else {
            Err(self.err("invalid literal (expected true/false)"))
        }
    }

    fn parse_null(&mut self) -> Result<Json, ParseError> {
        if self.bytes[self.pos..].starts_with(b"null") {
            self.pos += 4;
            Ok(Json::Null)
        } else {
            Err(self.err("invalid literal (expected null)"))
        }
    }

    fn parse_number(&mut self) -> Result<Json, ParseError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        let mut is_float = false;
        while let Some(b) = self.peek() {
            match b {
                b'0'..=b'9' => self.pos += 1,
                b'.' | b'e' | b'E' | b'+' | b'-' => {
                    is_float = true;
                    self.pos += 1;
                }
                _ => break,
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| self.err("number is not valid UTF-8"))?;
        if is_float {
            text.parse::<f64>()
                .map(Json::Float)
                .map_err(|_| ParseError {
                    message: format!("invalid float: {:?}", text),
                    at: start,
                })
        } else {
            text.parse::<i64>()
                .map(Json::Int)
                .map_err(|_| ParseError {
                    message: format!("invalid integer: {:?}", text),
                    at: start,
                })
        }
    }

    /// Parse a JSON string starting at the opening quote, decoding all escapes.
    fn parse_string(&mut self) -> Result<String, ParseError> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            let b = match self.peek() {
                Some(b) => b,
                None => return Err(self.err("unterminated string")),
            };
            match b {
                b'"' => {
                    self.pos += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.pos += 1;
                    self.parse_escape(&mut out)?;
                }
                // A raw control char inside a string is technically invalid
                // JSON; accept it leniently as the literal char (we never emit
                // such, but tolerate peers that do).
                _ => {
                    // Decode one UTF-8 scalar starting at self.pos.
                    let ch = self.decode_utf8_char()?;
                    out.push(ch);
                }
            }
        }
    }

    /// Decode one (possibly multibyte) UTF-8 scalar at `self.pos`, advancing.
    fn decode_utf8_char(&mut self) -> Result<char, ParseError> {
        let rest = &self.bytes[self.pos..];
        let s = std::str::from_utf8(rest).map_err(|e| {
            // Only the prefix up to the error is valid; if even the first byte
            // is bad, it's a hard error.
            let valid = e.valid_up_to();
            if valid == 0 {
                self.err("invalid UTF-8 in string")
            } else {
                self.err("invalid UTF-8 in string")
            }
        })?;
        let ch = s.chars().next().ok_or_else(|| self.err("empty string body"))?;
        self.pos += ch.len_utf8();
        Ok(ch)
    }

    /// Parse one escape sequence (the leading backslash already consumed).
    fn parse_escape(&mut self, out: &mut String) -> Result<(), ParseError> {
        let b = match self.peek() {
            Some(b) => b,
            None => return Err(self.err("unterminated escape")),
        };
        self.pos += 1;
        match b {
            b'"' => out.push('"'),
            b'\\' => out.push('\\'),
            b'/' => out.push('/'),
            b'b' => out.push('\u{08}'),
            b'f' => out.push('\u{0C}'),
            b'n' => out.push('\n'),
            b'r' => out.push('\r'),
            b't' => out.push('\t'),
            b'u' => {
                let cp = self.parse_hex4()?;
                if (0xD800..=0xDBFF).contains(&cp) {
                    // High surrogate: must be followed by \uDCxx low surrogate.
                    if self.peek() == Some(b'\\') {
                        self.pos += 1;
                        if self.peek() == Some(b'u') {
                            self.pos += 1;
                            let low = self.parse_hex4()?;
                            if (0xDC00..=0xDFFF).contains(&low) {
                                let c = 0x10000
                                    + ((cp - 0xD800) << 10)
                                    + (low - 0xDC00);
                                match char::from_u32(c) {
                                    Some(ch) => out.push(ch),
                                    None => {
                                        return Err(
                                            self.err("invalid surrogate pair scalar")
                                        )
                                    }
                                }
                                return Ok(());
                            }
                        }
                    }
                    return Err(self.err("unpaired high surrogate in \\u escape"));
                } else if (0xDC00..=0xDFFF).contains(&cp) {
                    return Err(self.err("unexpected low surrogate in \\u escape"));
                } else {
                    match char::from_u32(cp) {
                        Some(ch) => out.push(ch),
                        None => return Err(self.err("invalid \\u code point")),
                    }
                }
            }
            _ => return Err(self.err("invalid escape character")),
        }
        Ok(())
    }

    /// Read exactly four hex digits and return their value.
    fn parse_hex4(&mut self) -> Result<u32, ParseError> {
        let mut v: u32 = 0;
        for _ in 0..4 {
            let b = match self.peek() {
                Some(b) => b,
                None => return Err(self.err("unterminated \\u escape")),
            };
            let d = match b {
                b'0'..=b'9' => (b - b'0') as u32,
                b'a'..=b'f' => (b - b'a' + 10) as u32,
                b'A'..=b'F' => (b - b'A' + 10) as u32,
                _ => return Err(self.err("non-hex digit in \\u escape")),
            };
            v = (v << 4) | d;
            self.pos += 1;
        }
        Ok(v)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(v: Json) {
        let s = v.to_string();
        let back = parse(&s).expect("reparse");
        assert_eq!(v, back, "round-trip failed for {:?} (serialized {})", v, s);
    }

    #[test]
    fn primitives_roundtrip() {
        roundtrip(Json::Null);
        roundtrip(Json::Bool(true));
        roundtrip(Json::Bool(false));
        roundtrip(Json::Int(0));
        roundtrip(Json::Int(-42));
        roundtrip(Json::Int(9_000_000_000));
        roundtrip(Json::Str(String::new()));
        roundtrip(Json::Str("hello".to_owned()));
    }

    #[test]
    fn string_escapes_roundtrip() {
        // Quote, backslash, slash, newline, tab, CR, control char, unicode.
        let s = "a\"b\\c/d\ne\tf\rg\u{08}\u{0C}h\u{0001}\u{00e9}\u{2603}";
        roundtrip(Json::Str(s.to_owned()));
        // Source text with braces and quotes (an actual `view` payload shape).
        let src = "fn f(x: Int) -> Int {\n  \"x = \" ++ x\n}";
        roundtrip(Json::Str(src.to_owned()));
    }

    #[test]
    fn arrays_and_nested_objects_roundtrip() {
        let v = Json::Array(vec![
            Json::Int(1),
            Json::Bool(false),
            Json::Null,
            Json::Str("x".to_owned()),
            Json::obj([
                ("k".to_string(), Json::Int(7)),
                (
                    "nested".to_string(),
                    Json::Array(vec![Json::Str("a\nb".to_owned()), Json::Int(-1)]),
                ),
            ]),
        ]);
        roundtrip(v);
    }

    #[test]
    fn request_shape_roundtrips() {
        let req = Json::obj([
            ("id".to_string(), Json::Int(7)),
            ("op".to_string(), Json::Str("update".to_owned())),
            ("branch".to_string(), Json::Str("scratch/agent-1".to_owned())),
            (
                "params".to_string(),
                Json::obj([
                    ("name".to_string(), Json::Str("math.sqrt".to_owned())),
                    (
                        "source".to_string(),
                        Json::Str("fn sqrt(x: Float) -> Float { x }".to_owned()),
                    ),
                    ("dry_run".to_string(), Json::Bool(true)),
                ]),
            ),
        ]);
        roundtrip(req.clone());
        // Field access.
        assert_eq!(req.get("id").and_then(|j| j.as_i64()), Some(7));
        assert_eq!(req.get("op").and_then(|j| j.as_str()), Some("update"));
        let params = req.get("params").unwrap();
        assert_eq!(
            params.get("dry_run").and_then(|j| j.as_bool()),
            Some(true)
        );
    }

    #[test]
    fn parses_unicode_escape_and_surrogate_pair() {
        // é = é, surrogate pair for U+1F600 (😀).
        let v = parse(r#""café 😀""#).unwrap();
        assert_eq!(v, Json::Str("café \u{1F600}".to_owned()));
    }

    #[test]
    fn rejects_trailing_garbage_and_bad_tokens() {
        assert!(parse("123 456").is_err());
        assert!(parse("{").is_err());
        assert!(parse("nul").is_err());
        assert!(parse(r#"{"k": }"#).is_err());
        assert!(parse("").is_err());
    }

    #[test]
    fn float_roundtrip() {
        let v = parse("3.5").unwrap();
        assert_eq!(v, Json::Float(3.5));
        assert_eq!(v.to_string(), "3.5");
    }

    #[test]
    fn object_keys_are_sorted_deterministically() {
        let v = Json::obj([
            ("b".to_string(), Json::Int(2)),
            ("a".to_string(), Json::Int(1)),
        ]);
        assert_eq!(v.to_string(), r#"{"a":1,"b":2}"#);
    }
}
